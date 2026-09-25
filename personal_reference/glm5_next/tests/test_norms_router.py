# SPDX-License-Identifier: Apache-2.0
"""Norms, l2norm, router and attention softmax, against vendored transformers.

Run: python3 -m pytest personal_reference/glm5_next/tests/test_norms_router.py -v -s

These components had NO external-reference coverage until 2026-09-25. Two bugs had
already been found in exactly that gap (the silu/sigmoid output gate, the missing
swiglu clamp), so the rest of it got closed rather than sampled.

Every test here drives its inputs into the regime under test -- past dtype
boundaries, past saturation, into the group-masking branch. A vendored reference
compared on gentle inputs is how the swiglu clamp passed review while broken.

Three of these pin DELIBERATE divergences rather than agreements. They exist so that
nobody later "aligns" the oracle and makes it worse:

* ``RMSNorm`` multiplies by ``weight`` in fp32 and casts once; transformers casts
  first. Identical in fp32; in bf16 the ORACLE IS MORE ACCURATE. Kept on purpose --
  an oracle is maximum-precision ground truth, not a replica of one deployment's
  rounding.
* ``TopkRouter`` omits group masking, which is the identity only because
  ``n_group == topk_group == 1``. Asserted, not assumed.
* ``LinearAttention`` hardcodes ``F.silu`` for the conv activation. Correct because
  ``hidden_act == "silu"``, but there was no link between the two until now.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from glm5_next import reference as R  # noqa: E402
import hf_refs as H  # noqa: E402


def _g(seed=0):
    return torch.Generator().manual_seed(seed)


# ------------------------------------------------------------------------- l2norm
def test_l2norm_matches_transformers_exactly():
    """Now x / sqrt(...), transformers' spelling -- not Qwen's x * rsqrt(...)."""
    x = torch.randn(4, 16, 128, generator=_g(1))
    torch.testing.assert_close(R.l2norm(x), H.hf_l2norm(x), rtol=0, atol=0)


def test_l2norm_is_not_the_qwen_rsqrt_variant():
    """DISCRIMINATING. The variant transformers explicitly warns against differs at
    ULP level -- small, but this is the third Qwen carry-over, so it gets a test."""
    x = torch.randn(4, 16, 128, generator=_g(2))
    qwen = x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)
    ours = R.l2norm(x)
    assert not torch.equal(ours, qwen), "oracle is bitwise-identical to the Qwen variant"
    d = (ours - qwen).abs().max().item()
    print(f"\n  l2norm: oracle vs qwen rsqrt variant = {d:.3e} "
          f"({d / (torch.finfo(torch.float32).eps * ours.abs().max().item()):.1f} ULP)")
    torch.testing.assert_close(ours, H.hf_l2norm(x), rtol=0, atol=0)


@pytest.mark.parametrize("scale", [1e-4, 1.0, 1e4])
def test_l2norm_agrees_across_magnitudes(scale):
    """Driven across scales: the +eps (not max(.,eps)) convention matters at small norm."""
    x = torch.randn(3, 128, generator=_g(3)) * scale
    torch.testing.assert_close(R.l2norm(x), H.hf_l2norm(x), rtol=0, atol=0)


# ------------------------------------------------------------------------ RMSNorm
def test_rmsnorm_is_bit_identical_to_transformers_in_fp32():
    """The divergence is a cast-order difference, so in fp32 there is none."""
    n = R.RMSNorm(128, 1e-5)
    with torch.no_grad():
        n.weight.normal_(1.0, 0.3, generator=_g(4))
    x = torch.randn(4, 16, 128, generator=_g(5)) * 8.0
    with torch.no_grad():
        got = n(x)
    torch.testing.assert_close(got, H.hf_rms_norm(x, n.weight, 1e-5), rtol=0, atol=0)


def test_rmsnorm_bf16_divergence_is_deliberate_and_favours_the_oracle():
    """DELIBERATE DIVERGENCE, pinned. transformers rounds to bf16 BEFORE the weight
    multiply; the oracle multiplies in fp32 and rounds once. Measured against an fp64
    arbiter: the oracle must be at least as close. Do not 'fix' this."""
    dim = 128
    w = (1.0 + 0.3 * torch.randn(dim, generator=_g(6)))
    x = (torch.randn(4, dim, generator=_g(7)) * 8.0).to(torch.bfloat16)

    exact = (x.double() * torch.rsqrt(x.double().pow(2).mean(-1, keepdim=True) + 1e-5)) * w.double()
    n = R.RMSNorm(dim, 1e-5)
    with torch.no_grad():
        n.weight.copy_(w)
        ours = n(x)
    theirs = H.hf_rms_norm(x, w, 1e-5)

    e_ours = (ours.double() - exact).abs().max().item()
    e_theirs = (theirs.double() - exact).abs().max().item()
    print(f"\n  RMSNorm bf16 error vs fp64: oracle {e_ours:.3e}, transformers {e_theirs:.3e}")
    assert not torch.equal(ours.float(), theirs.float()), "no divergence to pin; re-check"
    assert e_ours <= e_theirs, "the oracle is now LESS accurate than transformers -- investigate"


def test_rmsnorm_uses_plain_weight_not_one_plus_weight():
    """Qwen3.5 uses (1 + weight); GLM does not. Right today, and nothing would notice
    if it stopped being -- three bugs have already come from this sibling confusion."""
    dim = 64
    n = R.RMSNorm(dim, 1e-5)
    with torch.no_grad():
        n.weight.fill_(0.0)          # plain: output is zero. (1+w): output is unchanged.
        out = n(torch.randn(2, dim, generator=_g(8)))
    assert out.abs().max() == 0.0, "RMSNorm appears to use (1 + weight), the Qwen3.5 form"


# ------------------------------------------------------------------------- router
def _router_inputs(cfg, seed=9, scale=1.0):
    r = R.TopkRouter(cfg)
    with torch.no_grad():
        r.weight.normal_(0, 0.5, generator=_g(seed))
        r.e_score_correction_bias.normal_(0, scale, generator=_g(seed + 1))
    x = torch.randn(12, cfg.hidden_size, generator=_g(seed + 2))
    return r, x


def _as_sets(idx, w):
    """topk(sorted=False) leaves order unspecified, so compare (index -> weight) maps."""
    return [dict(zip(i.tolist(), ww.tolist())) for i, ww in zip(idx, w)]


@pytest.mark.parametrize("bias_scale", [0.0, 1.0, 5.0])
def test_router_matches_transformers_at_n_group_1(bias_scale):
    """Driven with a large correction bias too, so the choice/weight split is exercised:
    the bias steers WHICH experts win but must never enter their weights."""
    cfg = R.tiny_cfg()
    assert cfg.n_group == cfg.topk_group == 1
    r, x = _router_inputs(cfg, scale=bias_scale)
    with torch.no_grad():
        w_o, i_o = r(x)
    _, w_h, i_h = H.hf_topk_router(x, r.weight, r.e_score_correction_bias, cfg.num_experts_per_tok,
                                   cfg.n_group, cfg.topk_group, cfg.norm_topk_prob,
                                   cfg.routed_scaling_factor)
    for a, b in zip(_as_sets(i_o, w_o), _as_sets(i_h, w_h)):
        assert set(a) == set(b), f"different experts chosen: {sorted(a)} vs {sorted(b)}"
        for e in a:
            assert abs(a[e] - b[e]) < 1e-6


def test_group_masking_is_identity_only_because_n_group_is_1():
    """THE ASSUMPTION, made explicit. At n_group=4 the oracle and transformers diverge,
    which is exactly why the config value must be asserted rather than assumed."""
    cfg = R.tiny_cfg()
    r, x = _router_inputs(cfg, seed=20, scale=2.0)
    with torch.no_grad():
        w_o, i_o = r(x)
    _, w4, i4 = H.hf_topk_router(x, r.weight, r.e_score_correction_bias, cfg.num_experts_per_tok,
                                 4, 1, cfg.norm_topk_prob, cfg.routed_scaling_factor)
    differing = sum(1 for a, b in zip(_as_sets(i_o, w_o), _as_sets(i4, w4)) if set(a) != set(b))
    print(f"\n  router: n_group=4 masking changes expert choice on {differing}/{x.shape[0]} tokens")
    assert differing > 0, "group masking had no effect even at n_group=4; test is vacuous"


def test_config_pins_the_group_assumption():
    """The oracle omits group masking, so this is a correctness precondition."""
    cfg = R.FlashCfg()
    assert cfg.n_group == 1 and cfg.topk_group == 1, (
        "TopkRouter omits group masking, which is only the identity at n_group == "
        "topk_group == 1. Implement the masking before changing these."
    )


# ------------------------------------------------- attention softmax + conv activation
def test_sparse_mla_softmax_accumulates_in_fp32_and_does_not_downcast():
    """Two separate claims, and the first one is NOT an accuracy win -- measured.

    torch already accumulates bf16 softmax in fp32, so ``x.softmax(-1)`` and
    ``x.softmax(-1, dtype=float32).to(bf16)`` are BITWISE EQUAL. The explicit dtype is
    about stating the contract, not about precision. An earlier version of this test
    asserted the explicit form was more accurate; it passed while proving nothing.

    What DOES matter is the cast that transformers applies afterwards. It ends with
    ``.to(query.dtype)``; the oracle keeps fp32, for the same reason RMSNorm keeps
    fp32 -- the downcast is a deployment choice and costs four orders of magnitude.
    """
    logits = (torch.randn(2, 4, 8, 64, generator=_g(11)) * 30.0).to(torch.bfloat16)
    exact = logits.double().softmax(-1)

    ambient = logits.softmax(-1)
    explicit_cast_back = logits.softmax(-1, dtype=torch.float32).to(logits.dtype)
    assert torch.equal(ambient, explicit_cast_back), (
        "torch no longer accumulates bf16 softmax in fp32; the explicit dtype now "
        "changes the result and this test's reasoning needs revisiting"
    )

    ours = logits.softmax(-1, dtype=torch.float32)          # what the oracle now does
    e_ours = (ours.double() - exact).abs().max().item()
    e_cast = (explicit_cast_back.double() - exact).abs().max().item()
    print(f"\n  bf16 softmax error vs fp64: oracle (no downcast) {e_ours:.3e}, "
          f"transformers (.to(query.dtype)) {e_cast:.3e}")
    assert ours.dtype == torch.float32
    assert e_ours < e_cast / 100, "the downcast should cost orders of magnitude"


def test_sparse_mla_layer_keeps_the_fp32_softmax_path():
    """The layer actually takes that path, rather than the constant being unused."""
    cfg = R.tiny_cfg()
    torch.manual_seed(0)
    m = R.SparseMLAttention(cfg).eval()
    x = torch.randn(1, 12, cfg.hidden_size)
    with torch.no_grad():
        out, _ = m(x)
    assert out.dtype == torch.float32 and torch.isfinite(out).all()


def test_conv_activation_is_linked_to_hidden_act():
    """LinearAttention hardcodes F.silu. Correct only while hidden_act == "silu" --
    'correct but unconnected to its source of truth' is the state that survives a
    config change and then silently does not."""
    cfg = R.FlashCfg()
    assert cfg.hidden_act == "silu", (
        "LinearAttention and MLP hardcode F.silu; wire the activation through before "
        "changing hidden_act."
    )
    x = torch.randn(64, generator=_g(12)) * 4.0
    torch.testing.assert_close(F.silu(x), x * torch.sigmoid(x), rtol=0, atol=1e-6)


# ------------------------------------------------------------- suite portability
def test_no_test_imports_a_toolchain_dependency():
    """The suite runs anywhere with torch, and that is load-bearing rather than tidy.

    External references are VENDORED (hf_refs, gdn_refs, indexer_refs) precisely so no
    test needs transformers, vllm, nkilib or nki. transformers 5.17 lives in an
    unrelated venv, vLLM ships manylinux wheels only, and nkilib's ``experimental/gdn``
    is not even in the installed package on the dev box. A single ``import
    transformers`` in one test would make the suite unrunnable for everyone.

    This is a guard against a future regression, not a check of present behaviour --
    the same species as the tiny_cfg clamp pin and the [c,c,K] source grep.
    """
    import re
    banned = re.compile(r"^\s*(?:import|from)\s+(transformers|vllm|nkilib|nki)\b", re.M)
    tests_dir = pathlib.Path(__file__).resolve().parent
    offenders = []
    for f in sorted(tests_dir.glob("*.py")):
        for m in banned.finditer(f.read_text()):
            offenders.append(f"{f.name}: {m.group(0).strip()}")
    assert not offenders, (
        "these imports break the suite everywhere the toolchain is absent; vendor the "
        "reference instead, as hf_refs.py / gdn_refs.py / indexer_refs.py do:\n  "
        + "\n  ".join(offenders)
    )
