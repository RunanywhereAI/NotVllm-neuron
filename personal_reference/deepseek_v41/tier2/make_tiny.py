# SPDX-License-Identifier: Apache-2.0
"""Write a tiny DeepSeek-V4.1-Flash checkpoint, stored and served the way the real one is.

    python personal_reference/deepseek_v41/tier2/make_tiny.py HF_DIR OUT_DIR [SEED]

``HF_DIR`` is the released snapshot. Only ``config.json``, the tokenizer files and
``encoding/`` are read from it. ``OUT_DIR`` gets:

* ``raw/``: the released ``config.json`` with ``text_config`` shrunk and
  ``quantization_config`` kept, the safetensors shards plus index, the tokenizer files
  and ``encoding/``. It is laid out like the HF snapshot.
* ``served/``: ``make_served_dir(raw)``, i.e. ``config.json`` without
  ``quantization_config`` (the original recorded) and everything else symlinked. This
  is what vLLM's ``ModelConfig`` and ``weights.Checkpoint`` both load, exactly as for
  the real model.
* ``reference_args.json``: the reference ``ModelArgs`` the weights were generated at.
  The oracle side (``oracle_side.py``) rebuilds from this, never from our HF-to-reference
  mapping. That mapping is checked here instead, by round-tripping the written config
  through ``DeepseekV41TextArgs`` and requiring every constant back.

Shapes are ``tests/harness.py``'s ``SMALL``. Everything SMALL does not set comes from
DeepSeek's inference ``ref/config.json`` (YaRN, mHC, routing, norm eps), so the tiny
model differs from the released one only in size. Fixed by the brief: ``vocab_size``
129280, because the tokenizer is the real one and Engram's compressed token map is
built from it; 32 index heads; Engram on; no MTP. The Engram tables are sized from
their own primes, which is how the released ``engram_num_embeddings`` arise.

Storage formats are the checkpoint's:

* FP8 e4m3 with 32x32 e8m0 scales for every FP8 Linear, including ``wo_a``. That one is
  quantized here, and the reference is given the dequantized values, which bf16 holds
  exactly.
* Routed experts as int8 holding two e2m1 values plus a per-32 e8m0 scale.
* Engram tables as FP8 rows plus a per-32 scale.
* ``head`` and the ratio>1 compressor weights as BF16. The reference keeps them in
  fp32 and is rounded to the stored values first.
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import math
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from personal_reference.deepseek_v41 import oracle  # noqa: E402
from personal_reference.deepseek_v41.tests import harness  # noqa: E402
from personal_reference.deepseek_v41.tests import plugin_harness as ph  # noqa: E402

SEED = 0
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
# what the brief fixes on top of SMALL; Engram table sizes are filled in from the primes
TINY = dict(
    vocab_size=129280, n_mtp_layers=0, index_n_heads=32,
    engram_layer_ids=(1, 4), engram_max_ngram_size=4, engram_vocab_size=1000,
    engram_n_heads=2, engram_head_dim=32, engram_pad_id=2, engram_compressed_vocab_size=99092,
    vision_n_layers=0, dspark_block_size=0, dspark_target_layer_ids=(),
    dspark_n_routed_experts=0, dspark_n_activated_experts=0,
    # runtime only (not in the HF config)
    max_batch_size=1, max_seq_len=2048, temperature=0.0,
)
_RUNTIME = ("max_batch_size", "max_seq_len", "temperature")
_FP32_STORED_BF16 = ("head.weight", ".compressor.w")      # reference fp32, checkpoint bf16


def plugin(dotted: str):
    """A plugin module; through the package where vLLM is installed, by path otherwise."""
    try:
        import vllm  # noqa: F401
    except ImportError:
        return ph.import_plugin(dotted)
    return importlib.import_module(dotted)


def reference_args(m) -> "m.ModelArgs":
    base = json.loads((oracle.REF_DIR / "config.json").read_text())
    base.update(harness.SMALL)
    base.update(TINY)
    base["compress_ratios"] = tuple(base["compress_ratios"])[: base["n_layers"] + base["n_mtp_layers"]]
    base["engram_num_embeddings"] = (1,) * len(base["engram_layer_ids"])
    layout = sys.modules["engram"].EngramLayout.from_args(m.ModelArgs(**base))
    base["engram_num_embeddings"] = tuple(sum(sum(p) for p in layer) for layer in layout.primes)
    return m.ModelArgs(**base)


def tiny_hf_config(released: dict, args, CFG) -> dict:
    """The released config with ``text_config`` set from ``args`` through the inverse of
    the plugin's mapping; then required to map back to ``args`` exactly."""
    cfg = json.loads(json.dumps(released))
    text = cfg["text_config"]
    for hf_key, ref_key in CFG._HF_TO_REF.items():
        v = getattr(args, ref_key)
        text[hf_key] = list(v) if isinstance(v, tuple) else v
    rope = dict(text["rope_scaling"])
    for rope_key, ref_key in CFG._ROPE_TO_REF.items():
        rope[rope_key] = getattr(args, ref_key)
    text["rope_scaling"] = rope
    back = CFG.DeepseekV41TextArgs.from_hf(text, cfg.get(CFG.QUANT_KEY)).to_reference_args()
    want = {k: (list(v) if isinstance(v, tuple) else v) for k, v in dataclasses.asdict(args).items()}
    wrong = {k: (v, want[k]) for k, v in back.items() if want.get(k) != v}
    assert not wrong, f"config does not map back to the reference args: {wrong}"
    return cfg


def _quantize_fp8_block(w: torch.Tensor):
    n, k = w.shape
    blocks = w.float().view(n // 32, 32, k // 32, 32)
    s = torch.exp2(torch.ceil(torch.log2(blocks.abs().amax((1, 3)).clamp_min(1e-30) / 448)))
    q = (blocks / s[:, None, :, None]).to(torch.float8_e4m3fn)
    return q.view(n, k), s.to(torch.float8_e8m0fnu), (q.float() * s[:, None, :, None]).view(n, k)


@torch.no_grad()
def storage_tensors(ref) -> dict[str, torch.Tensor]:
    """The reference's parameters in the checkpoint's storage formats. Mutates ``ref`` so
    that it holds exactly the stored values."""
    out = {}
    for name, p in ref.named_parameters():
        if p.dtype == torch.float32 and any(s in name for s in _FP32_STORED_BF16):
            p.copy_(p.bfloat16().float())
        if name.endswith("attn.wo_a.weight"):
            q, s, deq = _quantize_fp8_block(p)
            p.copy_(deq)
            out[name] = q
            out[name.removesuffix("weight") + "scale"] = s
    for name, t in ref.state_dict().items():
        if name in out:
            continue
        if t.dtype == torch.float4_e2m1fn_x2:
            t = t.view(torch.int8)
        elif t.dtype == torch.float32 and any(s in name for s in _FP32_STORED_BF16):
            t = t.bfloat16()
        out[name] = t.contiguous()
    return out


def write_shards(tensors: dict, d: Path, index_file: str, n_shards: int = 2) -> None:
    from safetensors.torch import save_file

    names = sorted(tensors)
    per = math.ceil(len(names) / n_shards)
    weight_map = {}
    for i in range(n_shards):
        fname = f"model-{i + 1:05d}-of-{n_shards:05d}.safetensors"
        keys = names[i * per:(i + 1) * per]
        save_file({k: tensors[k] for k in keys}, str(d / fname), metadata={"format": "pt"})
        weight_map.update({k: fname for k in keys})
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    (d / index_file).write_text(json.dumps({"metadata": {"total_size": total},
                                            "weight_map": weight_map}, indent=1))


def main(hf_dir: Path, out: Path, seed: int = SEED) -> Path:
    CFG = plugin("vllm_neuron.model.deepseek_v41.config")
    W = plugin("vllm_neuron.model.deepseek_v41.weights")
    from transformers import AutoTokenizer

    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    released = json.loads((hf_dir / "config.json").read_text())
    for f in TOKENIZER_FILES:
        if (hf_dir / f).exists():
            shutil.copy2(hf_dir / f, raw / f)
    if (hf_dir / "encoding").is_dir():
        shutil.copytree(hf_dir / "encoding", raw / "encoding", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "tests"))

    m = oracle.load_reference("faithful")
    args = reference_args(m)
    (raw / "config.json").write_text(json.dumps(tiny_hf_config(released, args, CFG), indent=2))
    tok = AutoTokenizer.from_pretrained(str(raw))
    prev = torch.get_default_dtype()
    try:
        ref = oracle.build(m, args, init=lambda mod: harness.init_weights(mod, seed), tokenizer=tok)
    finally:
        torch.set_default_dtype(prev)
    tensors = storage_tensors(ref)
    write_shards(tensors, raw, W.INDEX_FILE)
    (out / "reference_args.json").write_text(json.dumps(
        {**dataclasses.asdict(args), "seed": seed}, indent=1))

    served = CFG.make_served_dir(raw, out / "served")
    # the served directory must load through the plugin's own reader, every tensor planned
    with W.Checkpoint(served) as ck:
        counts = ck.plan.counts()
        assert counts[W.DROP] == 0 and set(ck.weight_map) == set(tensors), counts
        assert ck.args.to_reference_args() == {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in dataclasses.asdict(args).items()
            if k in ck.args.to_reference_args()}
    print(f"wrote {out}: {len(tensors)} tensors, {dict(counts)}; "
          f"engram_num_embeddings={args.engram_num_embeddings}")
    return served


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else SEED)
