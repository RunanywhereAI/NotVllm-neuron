# SPDX-License-Identifier: Apache-2.0
"""The exact-mode oracle on a tiny checkpoint from ``make_tiny.py``, teacher-forced.

    from personal_reference.deepseek_v41.tier2 import oracle_side
    ref = oracle_side.load(TINY_OUT_DIR)                 # exact mode, the on-disk weights
    lp = oracle_side.teacher_forced_logprobs(ref, tokens, n_prompt)

The model is rebuilt from ``reference_args.json`` and loaded from the shards on disk,
not from make_tiny's memory. The load mirrors what ``ref/convert.py`` does at mp=1:
FP8 and FP4 stay as stored, ``wo_a`` is dequantized, and bf16 is widened to the
parameter dtype. Every parameter must come from the checkpoint, and every stored tensor
must be used.

``teacher_forced_logprobs`` takes log-softmax at each generated position from ONE
forward over the full sequence, with the head switched to all positions. That is only
valid if the reference is causal, so it is checked against per-prefix forwards at a few
positions (``check_positions``) on every call.

    python personal_reference/deepseek_v41/tier2/oracle_side.py TINY_OUT_DIR
        # self-check: the reference's greedy on a shared-prefix pair, causality verified
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from personal_reference.deepseek_v41 import kernel_torch, oracle  # noqa: E402

FP8, E8M0, FP4 = torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.float4_e2m1fn_x2


def _stored(served: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    index = json.loads((served / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for fname in sorted(set(index.values())):
        out.update(load_file(str(served / fname)))
    return out


@torch.no_grad()
def load(tiny_dir: str | Path, mode: str = "exact"):
    """The reference at ``reference_args.json``, holding the checkpoint's values."""
    from transformers import AutoTokenizer

    tiny_dir = Path(tiny_dir)
    served = tiny_dir / "served"
    m = oracle.load_reference(mode)
    spec = json.loads((tiny_dir / "reference_args.json").read_text())
    spec.pop("seed", None)
    args = m.ModelArgs(**{k: tuple(v) if isinstance(v, list) else v for k, v in spec.items()})
    tok = AutoTokenizer.from_pretrained(str(served))
    prev = torch.get_default_dtype()
    try:
        model = oracle.build(m, args, tokenizer=tok)
    finally:
        torch.set_default_dtype(prev)
    stored = _stored(served)
    used = set()
    for name, p in model.named_parameters():
        if name.endswith("attn.wo_a.weight") and p.dtype not in (FP8,):
            w, s = stored[name], stored[name.removesuffix("weight") + "scale"]
            v = kernel_torch.dequant_fp8_weight(w, s, 32)
            used |= {name, name.removesuffix("weight") + "scale"}
        else:
            v = stored[name]
            used.add(name)
            if p.dtype == FP4:
                v = v.view(FP4)
        if p.dtype in (FP8, E8M0, FP4):
            assert v.dtype == p.dtype and v.shape == p.shape, (name, v.dtype, p.dtype)
            p.data = v.clone()
        else:
            assert v.shape == p.shape, (name, v.shape, p.shape)
            p.copy_(v.to(p.dtype))
    unused = set(stored) - used
    assert not unused, f"stored tensors the reference has no parameter for: {sorted(unused)[:5]}"
    return model.eval()


def _full_head(model):
    """Make the head return logits at every position for the next forward call."""
    head = model.head
    inner = head.forward

    def forward(x, full_logits=False):
        return inner(x, full_logits=True)

    head.forward = forward
    return lambda: setattr(head, "forward", inner)


@torch.no_grad()
def teacher_forced_logprobs(model, tokens: list[int], n_prompt: int,
                            check_positions: int = 3, atol: float = 1e-5) -> torch.Tensor:
    """``[len(tokens) - n_prompt + 1, vocab]`` fp32 log-probabilities: row ``j`` is the
    distribution of the token at position ``n_prompt + j`` given everything before it
    (the last row predicts one past the end)."""
    ids = torch.tensor([tokens])
    restore = _full_head(model)
    try:
        _, logits, _ = model(ids, 0)
    finally:
        restore()
    lp = torch.log_softmax(logits[0, n_prompt - 1:].float(), dim=-1)
    # causality: per-prefix forwards must agree with the one-shot rows
    rows = lp.shape[0]
    picks = sorted({0, rows // 2, rows - 1})[:check_positions]
    for j in picks:
        _, last, _ = model(ids[:, : n_prompt + j], 0)
        got = torch.log_softmax(last[0].float(), dim=-1)
        err = float((got - lp[j]).abs().max())
        assert err <= atol, f"one-shot row {j} differs from its prefix forward by {err:.3e}"
    return lp


@torch.no_grad()
def greedy(model, prompt: list[int], new_tokens: int) -> list[int]:
    seq = list(prompt)
    for _ in range(new_tokens):
        nxt, _, _ = model(torch.tensor([seq]), 0)
        seq.append(int(nxt[0]))
    return seq[len(prompt):]


def shared_prefix_prompts(vocab: int, prefix_len: int, suffix_lens=(23, 37), seed: int = 1234):
    """Two prompts sharing their first ``prefix_len`` tokens. Ids avoid the special-token
    tail of the vocab."""
    g = torch.Generator().manual_seed(seed)
    hi = min(vocab, 120000)
    prefix = torch.randint(1000, hi, (prefix_len,), generator=g).tolist()
    return [prefix + torch.randint(1000, hi, (n,), generator=g).tolist() for n in suffix_lens]


def main(tiny_dir: str) -> None:
    model = load(tiny_dir)
    args = json.loads((Path(tiny_dir) / "reference_args.json").read_text())
    prompts = shared_prefix_prompts(args["vocab_size"], 256)
    for p in prompts:
        out = greedy(model, p, 8)
        lp = teacher_forced_logprobs(model, p + out, len(p))
        top2 = lp[:-1].topk(2, dim=-1).values
        assert lp[:-1].argmax(-1).tolist() == out
        print(f"prompt {len(p)}: greedy {out}; min top1-top2 gap {float((top2[:, 0] - top2[:, 1]).min()):.3e}")


if __name__ == "__main__":
    main(sys.argv[1])
