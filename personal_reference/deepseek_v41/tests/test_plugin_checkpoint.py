"""Loading from a quantized HF checkpoint must give every rank exactly its slice.

The small reference model is written out in the checkpoint's storage formats -- FP8
e4m3 weights with 32x32 e8m0 scales, MXFP4 experts packed two per int8 byte, the Engram
table as FP8 rows with per-32 scales, BF16 where the checkpoint is BF16 -- and loaded
through ``weights.Checkpoint`` + ``CheckpointSource``, which reads only the 32x32 blocks
covering each slice. Every parameter on every rank must equal what ``DictSource`` cuts
from the fully dequantized tensors, bit for bit.
"""

import dataclasses
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from personal_reference.deepseek_v41 import oracle  # noqa: E402
from personal_reference.deepseek_v41.tests import harness  # noqa: E402
from personal_reference.deepseek_v41.tests import plugin_harness as ph  # noqa: E402

M = ph.import_plugin("vllm_neuron.model.deepseek_v41.model")
C = ph.import_plugin("vllm_neuron.model.deepseek_v41.config")
W = ph.import_plugin("vllm_neuron.model.deepseek_v41.weights")
ENGRAM = dict(engram_layer_ids=(1, 4), engram_num_embeddings=(400, 600), engram_max_ngram_size=4,
              engram_vocab_size=50, engram_n_heads=2, engram_head_dim=32, engram_pad_id=2,
              engram_compressed_vocab_size=300)


@pytest.fixture(scope="module")
def ckpt_dir(tmp_path_factory):
    oracle.load_reference("faithful")
    sys.modules["engram"].build_compressed_token_map = lambda tok: ([t % 300 for t in range(512)], 300)
    ref, args, _ = harness.build("faithful", 0, **ENGRAM)
    # the checkpoint stores these in bf16 and the reference upcasts them; round the
    # reference so both sides hold the stored values
    # wo_a is FP8 in the checkpoint but bf16 in the reference (convert.py dequantizes it):
    # quantize it and give the reference the dequantized values, which bf16 holds exactly
    wo_a = {}
    with torch.no_grad():
        for name, p in ref.named_parameters():
            if p.dtype == torch.float32 and (name == "head.weight" or ".compressor.w" in name):
                p.copy_(p.bfloat16().float())
            if name.endswith("attn.wo_a.weight"):
                n, k = p.shape
                blocks = p.float().view(n // 32, 32, k // 32, 32)
                s = torch.exp2(torch.ceil(torch.log2(blocks.abs().amax((1, 3)) / 448)))
                q = (blocks / s[:, None, :, None]).to(torch.float8_e4m3fn)
                p.copy_((q.float() * s[:, None, :, None]).view(n, k))
                wo_a[name] = q.view(n, k)
                wo_a[name.removesuffix("weight") + "scale"] = s.to(torch.float8_e8m0fnu)
    tensors = dict(wo_a)
    for name, t in ref.state_dict().items():
        if name in wo_a:
            continue
        if t.dtype == torch.float4_e2m1fn_x2:
            t = t.view(torch.int8)
        elif t.dtype == torch.float32 and (name == "head.weight" or ".compressor.w" in name):
            t = t.bfloat16()
        tensors[name] = t.contiguous()
    d = tmp_path_factory.mktemp("dsv41_tiny")
    names = sorted(tensors)
    half = len(names) // 2
    shards = {"model-00001-of-00002.safetensors": names[:half],
              "model-00002-of-00002.safetensors": names[half:]}
    weight_map = {}
    for fname, keys in shards.items():
        save_file({k: tensors[k] for k in keys}, str(d / fname))
        weight_map.update({k: fname for k in keys})
    (d / W.INDEX_FILE).write_text(json.dumps({"weight_map": weight_map}))
    return d, ref, args


def _text_args(args):
    f = {x.name for x in dataclasses.fields(C.DeepseekV41TextArgs)}
    extra = dict(dtype="fp8", expert_dtype="fp4", weight_block_size=(32, 32),
                 max_position_embeddings=args.max_seq_len)
    return C.DeepseekV41TextArgs(**{k: (tuple(v) if isinstance(v, list) else v)
                                    for k, v in {**{k: getattr(args, k) for k in f if k not in extra},
                                                 **extra}.items()})


@pytest.mark.parametrize("world,ep,ranks", [(1, 1, (0,)), (8, 4, (0, 3, 7)), (8, 8, (1, 6))])
def test_checkpoint_slices_equal_dense_slices(ckpt_dir, world, ep, ranks):
    d, ref, args = ckpt_dir
    dense = M.DictSource(ph.dequantized_state_dict(ref))
    etp = world // ep
    with W.Checkpoint(d, args=_text_args(args)) as ck:
        lazy = M.CheckpointSource(ck)
        for r in ranks:
            par = M.Parallel(tp=world, rank=r, ep=ep, ep_rank=r // etp, etp=etp, etp_rank=r % etp)
            with torch.device("meta"):
                model = M.DeepseekV41Model(args, block_size=8, par=par)
            for name, p in model.named_parameters():
                a, b = model.local_tensor(name, lazy), model.local_tensor(name, dense)
                assert tuple(a.shape) == tuple(p.shape), (name, r)
                if a.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
                    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), (name, r)
                else:
                    assert torch.equal(a.float(), b.float()), (name, r)


def test_checkpoint_loaded_model_matches_oracle(ckpt_dir):
    """End to end at TP=1: built on meta, loaded with ``load_weights``-style lazy reads."""
    d, ref, args = ckpt_dir
    with torch.device("meta"):
        model = M.DeepseekV41Model(args, block_size=8, cache_dtype=torch.float32).set_dtype(torch.float32)
    with W.Checkpoint(d, args=_text_args(args)) as ck:
        model.load_from(M.CheckpointSource(ck), device=torch.device("cpu"))
    want = ph.plugin_from_reference(ref, args, block_size=8)
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), want.named_parameters()):
        assert n1 == n2
        if p1.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
            assert torch.equal(p1.view(torch.uint8), p2.view(torch.uint8)), n1
        else:
            assert torch.equal(p1, p2), n1
    assert all(b.device.type == "cpu" for b in model.buffers())
