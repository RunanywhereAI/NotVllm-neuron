# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash weight plan, dequantization and TP/EP slicing (``weights.py``).

Three tiers, each skipped when its inputs are absent:

* **always** -- both dequantizations against the reference's own fp32 dequant
  (``kernel_torch``) and DeepSeek's ``convert.cast_e2m1fn_to_e4m3fn``; TP/EP slicing
  against ``ref/convert.py`` itself, run at ``--model-parallel 2`` on a small synthetic
  checkpoint under the real names.
* **``DSV41_HF_DIR`` with ``config.json`` and the index** (a laptop copy is enough) --
  every one of the 96,085 names is accounted for; the emitted names and dtypes equal the
  reference module tree's (``oracle.build`` on the meta device, text only, from DeepSeek's
  own inference ``config.json`` -- an independent source); shard shapes equal the
  reference's per-rank parameter shapes at world size 2 and 8, and the reference's
  ``wo_a`` ceiling at 16 is pinned.
* **``DSV41_HF_DIR`` with the shards** (the trn2 host) -- stored dtypes and shapes from all
  48 headers against the reference tree; exact dequantization of real tensors; the e2m1
  nibble order decided from the weights alone, not from anyone's code; and, with
  ``DSV41_REF_MP1`` (``ref/convert.py --model-parallel 1 --expert-dtype fp4`` output),
  the emitted tensors against convert.py's.

    DSV41_HF_DIR=/data/models/DeepSeek-V4.1-Flash \\
    DSV41_REF_MP1=/data/models/dsv41-ref-mp1/model0-mp1.safetensors \\
    python -m pytest test/unit/test_deepseek_v41_weights.py -v
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]


def _import_plugin():
    """``vllm_neuron/__init__`` imports vLLM, which has no macOS wheel. Where vLLM is
    absent, bind the parent packages by path so ``deepseek_v41`` (transformers only)
    imports without running their ``__init__``."""
    try:
        import vllm  # noqa: F401
    except ImportError:
        root = REPO / "vllm_neuron"
        for name, path in (("vllm_neuron", root), ("vllm_neuron.model", root / "model"),
                           ("vllm_neuron.model.deepseek_v41", root / "model" / "deepseek_v41")):
            if name not in sys.modules:
                mod = types.ModuleType(name)
                mod.__path__ = [str(path)]
                sys.modules[name] = mod
    from vllm_neuron.model.deepseek_v41 import weights
    from vllm_neuron.model.deepseek_v41.config import DeepseekV41TextArgs
    return weights, DeepseekV41TextArgs


W, DeepseekV41TextArgs = _import_plugin()
from personal_reference.deepseek_v41 import kernel_torch as K  # noqa: E402
from personal_reference.deepseek_v41 import oracle  # noqa: E402

FP8, E8M0, FP4 = torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.float4_e2m1fn_x2
HF_DIR = os.environ.get("DSV41_HF_DIR")
REF_MP1 = os.environ.get("DSV41_REF_MP1")
needs_index = pytest.mark.skipif(
    not HF_DIR or not (Path(HF_DIR) / W.INDEX_FILE).exists(),
    reason="set DSV41_HF_DIR to a directory with config.json and the safetensors index")
needs_shards = pytest.mark.skipif(
    not HF_DIR or not (Path(HF_DIR) / "model-00001-of-00048.safetensors").exists(),
    reason="needs the checkpoint shards (DSV41_HF_DIR on the trn2 host)")
needs_mp1 = pytest.mark.skipif(not REF_MP1 or not Path(REF_MP1).exists(),
                               reason="set DSV41_REF_MP1 to convert.py's mp=1 output")


def _load_convert():
    spec = importlib.util.spec_from_file_location("dsv41_ref_convert", oracle.REF_DIR / "convert.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fp8(shape, g):
    """Random finite e4m3fn codes (0x7F / 0xFF are NaN)."""
    b = torch.randint(0, 256, shape, generator=g, dtype=torch.uint8)
    return torch.where((b & 0x7F) == 0x7F, torch.zeros_like(b), b).view(FP8)


def _e8m0(shape, g, lo=110, hi=135):
    return torch.randint(lo, hi, shape, generator=g, dtype=torch.uint8).view(E8M0)


def _int8(shape, g):
    return torch.randint(0, 256, shape, generator=g, dtype=torch.uint8).view(torch.int8)


# ------------------------------------------------------------------ always: dequant
@pytest.mark.parametrize("shape", [(64, 96), (70, 100)])   # tiled and ragged
def test_fp8_block_dequant_is_exact_against_reference(shape):
    g = torch.Generator().manual_seed(0)
    w = _fp8(shape, g)
    s = _e8m0((-(-shape[0] // 32), -(-shape[1] // 32)), g)
    got = W.dequant_fp8_block(w, s)
    assert got.dtype == torch.bfloat16
    assert torch.equal(got.float(), K.dequant_fp8_weight(w, s, 32))   # bf16 loses nothing


def test_mxfp4_dequant_is_exact_against_reference_and_convert():
    g = torch.Generator().manual_seed(1)
    w = _int8((64, 48), g)                   # [N, K/2], K = 96
    s = _e8m0((64, 3), g, 124, 131)          # span <= 2^6 inside every 32-row block
    got = W.dequant_mxfp4(w, s)
    assert got.shape == (64, 96) and got.dtype == torch.bfloat16
    assert torch.equal(got.float(), K.dequant_fp4_weight(w.view(FP4), s))
    # DeepSeek's own fp4 -> fp8 recast, read back through the fp8 path, agrees too
    w8, s8 = _load_convert().cast_e2m1fn_to_e4m3fn(w, s)
    assert torch.equal(got, W.dequant_fp8_block(w8, s8))
    # and the packed dtype the reference uses is accepted unchanged
    assert torch.equal(W.dequant_mxfp4(w.view(FP4), s), got)


def test_unpack_e2m1_codes():
    # 0x71: low nibble 1 (0.5), high nibble 7 (6.0); 0x9F: low 15 (-6.0), high 9 (-0.5)
    packed = torch.tensor([[0x71, 0x9F]], dtype=torch.uint8).view(torch.int8)
    assert W.unpack_e2m1(packed).tolist() == [[0.5, 6.0, -6.0, -0.5]]


def test_dequant_refuses_bad_inputs():
    g = torch.Generator().manual_seed(2)
    w = _fp8((64, 64), g)
    with pytest.raises(ValueError, match="does not tile"):
        W.dequant_fp8_block(w, _e8m0((2, 3), g))
    with pytest.raises(TypeError):
        W.dequant_fp8_block(w.float(), _e8m0((2, 2), g))
    nan_scale = torch.full((2, 2), 0xFF, dtype=torch.uint8).view(E8M0)   # e8m0 NaN
    with pytest.raises(ValueError, match="non-finite"):
        W.dequant_fp8_block(w, nan_scale)
    with pytest.raises(ValueError, match="does not match"):
        W.dequant_mxfp4(_int8((64, 32), g), _e8m0((64, 1), g))


# --------------------------------------------- always: slicing against convert.py
def _synthetic_checkpoint(g):
    """Every kind ``shard`` treats differently, under real names, at toy sizes.
    Engram rows (13) deliberately do not divide by mp=2, to exercise the padding."""
    t = {"embed.weight": torch.randn(64, 32, generator=g).bfloat16(),
         "head.weight": torch.randn(64, 32, generator=g).bfloat16(),
         "norm.weight": torch.randn(32, generator=g).bfloat16(),
         "layers.0.attn.attn_sink": torch.randn(8, generator=g),
         "layers.0.attn.indexer.weights_proj.weight": torch.randn(4, 32, generator=g).bfloat16(),
         "layers.0.ffn.gate.bias": torch.randn(4, generator=g),
         "layers.0.engram.embed.weight": _fp8((13, 64), g),
         "layers.0.engram.embed.scale": _e8m0((13, 2), g)}
    for name, shape in {"attn.wq_a": (64, 32), "attn.wq_b": (128, 64), "attn.wkv": (64, 32),
                        "attn.wo_a": (128, 64), "attn.wo_b": (32, 128),
                        "attn.indexer.wq_b": (64, 64), "ffn.shared_experts.w1": (64, 32)}.items():
        t[f"layers.0.{name}.weight"] = _fp8(shape, g)
        t[f"layers.0.{name}.scale"] = _e8m0((shape[0] // 32, shape[1] // 32), g)
    for e in range(4):
        for w in ("w1", "w2", "w3"):
            t[f"layers.0.ffn.experts.{e}.{w}.weight"] = _int8((64, 32), g)
            t[f"layers.0.ffn.experts.{e}.{w}.scale"] = _e8m0((64, 2), g, 124, 131)
    return t


def _emit(name, tensors):
    """What the model loads for ``name``, from a {name: stored tensor} dict."""
    w, s = tensors[name], tensors.get(name.removesuffix(".weight") + ".scale")
    if ".experts." in name and ".shared_" not in name and w.dtype in (torch.int8, FP4):
        return W.dequant_mxfp4(w, s, name=name)
    if w.dtype == FP8 and ".engram.embed." not in name:
        return W.dequant_fp8_block(w, s, name=name)
    return w


def _same(a, b):
    if a.dtype in (FP8, E8M0):
        a, b = a.view(torch.uint8), b.view(torch.uint8)
    return a.dtype == b.dtype and torch.equal(a, b)


def test_shard_matches_convert_py_at_mp2(tmp_path):
    from safetensors.torch import load_file, save_file

    g = torch.Generator().manual_seed(3)
    full = _synthetic_checkpoint(g)
    hf, out = tmp_path / "hf", tmp_path / "out"
    hf.mkdir()
    save_file(full, str(hf / "model-00001-of-00001.safetensors"))
    threads = torch.get_num_threads()
    try:
        _load_convert().main(str(hf), str(out), 2, None)     # convert.py sets 8 threads
    finally:
        torch.set_num_threads(threads)
    targs = SimpleNamespace(n_routed_experts=4, o_groups=4, n_heads=8, index_n_heads=4)
    targets = [n for n in full if not (n.endswith(".scale") and ".engram." not in n)]
    for rank in range(2):
        theirs = load_file(str(out / f"model{rank}-mp2.safetensors"))
        ours = {n: W.shard(n, _emit(n, full), rank, 2, targs) for n in targets}
        ours = {n: t for n, t in ours.items() if t is not None}
        consumed = {n for n in theirs if n.endswith(".scale") and ".engram." not in n}
        assert set(ours) == set(theirs) - consumed, rank
        for n, t in ours.items():
            assert _same(t, _emit(n, theirs)), (rank, n)


# ------------------------------------------------------- the reference module tree
@contextlib.contextmanager
def _reference_world(m, world_size: int, rank: int):
    """Make the reference build as one rank of ``world_size`` (it reads torch.distributed
    in ``Transformer.__init__``) and skip Engram's hash state, which needs a tokenizer and
    holds only non-persistent buffers."""
    dist = m.dist
    saved = dist.is_initialized, dist.get_world_size, dist.get_rank, m.NgramHashState
    dist.is_initialized = lambda: True
    dist.get_world_size = lambda *a, **k: world_size
    dist.get_rank = lambda *a, **k: rank
    m.NgramHashState = lambda *a, **k: None
    try:
        yield
    finally:
        dist.is_initialized, dist.get_world_size, dist.get_rank, m.NgramHashState = saved
        m.precompute_freqs_cis.cache_clear()     # do not leave meta tensors in its cache


def _reference_params(world_size: int = 1, rank: int = 0):
    """name -> (dtype, shape) the text-only reference loads, after this module's
    dequantization: fp8 / fp4 parameters become bf16 (fp4 at twice the packed width) and
    their scales disappear -- except the Engram tables, which the reference keeps fp8."""
    m = oracle.load_reference("faithful")
    cfg = json.loads((oracle.REF_DIR / "config.json").read_text())
    cfg.update(vision_n_layers=0, dspark_block_size=0, max_batch_size=1, max_seq_len=256)
    with _reference_world(m, world_size, rank), torch.device("meta"):
        model = oracle.build(m, m.ModelArgs(**cfg))
    sd = model.state_dict()
    tables = {n for n, mod in model.named_modules() if isinstance(mod, m.ParallelEngramEmbedding)}
    out = {}
    for name, p in sd.items():
        owner = name.rsplit(".", 1)[0]
        quantized = owner not in tables and p.dtype in (FP8, FP4)
        if name.endswith(".scale") and owner not in tables \
                and sd.get(owner + ".weight", p).dtype in (FP8, FP4):
            continue
        shape = tuple(p.shape)
        if quantized and p.dtype == FP4:
            shape = (shape[0], 2 * shape[1])
        out[name] = (torch.bfloat16 if quantized else p.dtype, shape)
    return out, model


@pytest.fixture(scope="module")
def args():
    return DeepseekV41TextArgs.from_hf_config(json.loads((Path(HF_DIR) / "config.json").read_text()))


@pytest.fixture(scope="module")
def index_names():
    return list(json.loads((Path(HF_DIR) / W.INDEX_FILE).read_text())["weight_map"])


@pytest.fixture(scope="module")
def plan(args, index_names):
    return W.build_plan(index_names, args)


@pytest.fixture(scope="module")
def reference():
    return _reference_params()[0]


@needs_index
def test_plan_accounts_for_every_tensor(plan, index_names):
    assert len(index_names) == 96085
    assert len(plan.entries) == 96085
    counts = plan.counts()
    print(f"\n  {dict(counts)}")
    assert counts == {W.DEQUANT_MXFP4: 46080, W.DEQUANT_FP8: 330, W.SCALE: 46410,
                      W.TO_FP32: 7, W.KEEP: 551, W.DROP: 2707}
    assert sum(counts.values()) == 96085
    assert len(plan.targets()) == len(plan.emitted()) == 46968
    reasons = {r.split(":")[0]: n for r, n in plan.drop_reasons().items()}
    assert reasons == {"MTP / DSpark draft layers": 2401, "vision tower": 259,
                       "vision-to-text aligner": 4, "image span delimiters": 3,
                       "routing bias for image-span tokens": 40}
    # every scale is consumed by exactly one weight
    owners = [e.scale for e in plan.emitted() if e.scale]
    assert len(owners) == len(set(owners)) == counts[W.SCALE]


@needs_index
def test_plan_refuses_what_it_does_not_understand(args, index_names):
    names = list(index_names)
    with pytest.raises(ValueError, match="unrecognised"):
        W.build_plan(names + ["layers.0.attn.mystery.weight"], args)
    with pytest.raises(ValueError, match="is missing"):
        W.build_plan([n for n in names if n != "layers.0.attn.wq_a.scale"], args)
    with pytest.raises(ValueError, match="no dequantized weight"):
        W.build_plan([n for n in names if n != "layers.0.attn.wq_a.weight"], args)
    with pytest.raises(ValueError, match="no kv_source component"):
        W.build_plan(names + ["layers.0.attn.compressor.norm.weight"], args)
    with pytest.raises(ValueError, match="no pooling component"):   # ratio-1 layer 20
        W.build_plan(names + ["layers.20.attn.compressor.wgate.weight"], args)
    with pytest.raises(ValueError, match="n_routed_experts"):
        W.build_plan(names + ["layers.0.ffn.experts.384.w1.weight"], args)
    with pytest.raises(ValueError, match="n_layers"):
        W.build_plan(names + ["layers.40.attn_norm.weight"], args)
    dtypes = {n: "BF16" for n in names}
    with pytest.raises(ValueError, match="unexpected dtype"):
        W.build_plan(names, args, dtypes)


@needs_index
def test_emitted_names_and_dtypes_equal_the_reference_module_tree(plan, reference):
    assert plan.targets() == set(reference)
    wrong = [(e.target, e.target_dtype, reference[e.target][0]) for e in plan.emitted()
             if e.target_dtype != reference[e.target][0]]
    assert not wrong, wrong[:5]
    fp32 = sorted({e.target.split(".", 2)[-1] if e.target.startswith("layers.") else e.target
                   for e in plan.emitted() if e.target_dtype == torch.float32})
    print(f"\n  fp32: {fp32}")
    assert fp32 == ["attn.attn_sink", "attn.compressor.wgate.weight", "attn.compressor.wkv.weight",
                    "ffn.gate.bias", "hc_attn_base", "hc_attn_fn", "hc_attn_scale",
                    "hc_ffn_base", "hc_ffn_fn", "hc_ffn_scale", "head.weight"]


@needs_index
@pytest.mark.parametrize("world_size", [2, 8])
def test_shard_shapes_equal_the_reference_per_rank(plan, args, reference, world_size):
    for rank in (0, world_size - 1):
        local, _ = _reference_params(world_size, rank)
        got = {}
        for name in plan.targets():
            dtype, shape = reference[name]
            t = W.shard(name, torch.empty(shape, dtype=dtype, device="meta"), rank, world_size, args)
            if t is not None:
                got[name] = (t.dtype, tuple(t.shape))
        assert set(got) == set(local), (rank, sorted(set(got) ^ set(local))[:5])
        diff = [(n, got[n], local[n]) for n in got if got[n] != local[n]]
        assert not diff, (rank, diff[:5])


@needs_index
def test_wo_a_cannot_be_sharded_past_o_groups(args, reference):
    """The reference builds at world size 16 but its forward views wo_a as
    ``n_local_groups = o_groups // world_size = 0`` groups; ``shard`` refuses instead."""
    _, model = _reference_params(16, 0)
    assert model.layers[0].attn.n_local_groups == 0
    name = "layers.0.attn.wo_a.weight"
    t = torch.empty(reference[name][1], dtype=torch.bfloat16, device="meta")
    with pytest.raises(NotImplementedError, match="o_groups"):
        W.shard(name, t, 0, 16, args)
    assert W.shard(name, t, 0, 8, args).shape == (1024, 4096)


# --------------------------------------------------------------- real shards
SAMPLE = [
    "embed.weight", "head.weight", "norm.weight",
    "layers.0.attn.wq_a.weight", "layers.0.attn.wq_b.weight", "layers.0.attn.wkv.weight",
    "layers.39.attn.wo_a.weight", "layers.39.attn.wo_b.weight",
    "layers.0.attn.attn_sink", "layers.0.hc_attn_fn", "layers.0.ffn.gate.bias",
    "layers.2.attn.compressor.wkv.weight", "layers.2.attn.compressor.wgate.weight",
    "layers.20.attn.compressor.wkv.weight",
    "layers.2.attn.indexer.wq_b.weight", "layers.2.attn.indexer.wk.weight",
    "layers.1.engram.wkv.weight", "layers.1.engram.q_weight",
    "layers.0.ffn.shared_experts.w1.weight", "layers.0.ffn.shared_experts.w2.weight",
    "layers.0.ffn.experts.0.w1.weight", "layers.20.ffn.experts.200.w2.weight",
    "layers.39.ffn.experts.383.w3.weight",
]


@pytest.fixture(scope="module")
def ckpt():
    with W.Checkpoint(HF_DIR) as c:
        yield c


@needs_shards
def test_real_headers_match_the_plan_and_the_reference_shapes(ckpt, reference):
    headers = ckpt.headers()
    assert set(headers) == set(ckpt.weight_map)
    W.build_plan(ckpt.weight_map, ckpt.args, {n: d for n, (d, _) in headers.items()})
    wrong = [(e.target, e.target_shape(headers[e.source][1]), reference[e.target][1])
             for e in ckpt.plan.emitted()
             if e.target_shape(headers[e.source][1]) != reference[e.target][1]]
    assert not wrong, wrong[:5]


@needs_shards
def test_real_tensors_dequantize_exactly(ckpt):
    for name in SAMPLE:
        e = ckpt._by_target[name]
        got = ckpt.load(name)
        assert got.dtype == e.target_dtype, name
        raw = ckpt.raw(e.source)
        if e.action == W.DEQUANT_FP8:
            want = K.dequant_fp8_weight(raw, ckpt.raw(e.scale), 32)
        elif e.action == W.DEQUANT_MXFP4:
            want = K.dequant_fp4_weight(raw.view(FP4), ckpt.raw(e.scale))
        else:
            want = raw.float() if e.action == W.TO_FP32 else raw
        assert torch.equal(got.float() if want.dtype == torch.float32 else got, want), name


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


@needs_shards
def test_nibble_order_from_the_weights_alone(ckpt):
    """Every reference that unpacks e2m1 (DeepSeek's convert.py and kernels, our
    ``kernel_torch``, this module) says low nibble first: one shared assumption. The
    weights can decide it on their own. Swapping nibbles permutes K-columns pairwise, and
    an expert's w2 column for intermediate unit i should be large where its w1/w3 rows
    for unit i are large -- rows are not packed, so they are order-free. Measured on the
    real checkpoint: ~0.6-0.8 low-first, ~0.00 high-first."""
    for layer in (0, 39):
        low, high = [], []
        for e in (5, 101, 250, 377):
            p = f"layers.{layer}.ffn.experts.{e}."
            rows = ckpt.load(p + "w1.weight").float().norm(dim=1) \
                * ckpt.load(p + "w3.weight").float().norm(dim=1)
            w2 = ckpt.load(p + "w2.weight").float()
            low.append(_corr(w2.norm(dim=0), rows))
            swapped = w2.unflatten(1, (-1, 2)).flip(-1).flatten(1)   # high nibble first
            high.append(_corr(swapped.norm(dim=0), rows))
        print(f"\n  layer {layer}: low-first {[round(c, 3) for c in low]}  "
              f"high-first {[round(c, 3) for c in high]}")
        assert min(low) > 0.2 and max(abs(c) for c in high) < 0.1


@needs_shards
@needs_mp1
def test_emitted_tensors_equal_convert_py_output(ckpt):
    from safetensors import safe_open

    with safe_open(REF_MP1, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        emitted = ckpt.plan.targets()
        assert emitted <= keys
        extra = keys - emitted
        consumed = {e.scale for e in ckpt.plan.emitted() if e.scale}
        dropped = {n for n, e in ckpt.plan.entries.items() if e.action == W.DROP}
        # convert.py pops wo_a's scale itself and drops nothing; the rest is ours to drop
        assert extra <= consumed | dropped, sorted(extra - consumed - dropped)[:5]
        assert (consumed - keys) == {n for n in consumed if n.endswith("wo_a.scale")}
        for name in SAMPLE:
            theirs = {name: f.get_tensor(name)}
            scale = name.removesuffix(".weight") + ".scale"
            if scale in keys:
                theirs[scale] = f.get_tensor(scale)
            want = _emit(name, theirs)
            got = ckpt.load(name)
            if ckpt._by_target[name].action == W.TO_FP32:  # convert.py keeps the bf16;
                want = want.float()                        # the reference upcasts at load
            assert _same(got, want), name
        rows = f.get_slice("layers.14.engram.embed.weight")[1000:1064]
        assert _same(ckpt.raw_slice("layers.14.engram.embed.weight")[1000:1064], rows)
