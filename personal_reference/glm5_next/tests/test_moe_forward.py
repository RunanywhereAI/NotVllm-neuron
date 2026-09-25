# SPDX-License-Identifier: Apache-2.0
"""``MoE.forward`` against transformers -- the last row with no external reference.

Run: python3 -m pytest personal_reference/glm5_next/tests/test_moe_forward.py -v -s

``test_mlp_moe.py`` covers the swiglu limit that ``MLP`` and the routed experts share,
and ``test_norms_router.py`` covers ``TopkRouter`` against the vendored router. Neither
tests what ``MoE.forward`` **returns**, so the composition -- router, routed experts and
shared expert assembled in one particular order -- had never been compared against
anything external. Every real defect found in this oracle so far was in a component
nobody had pointed an external reference at, which is the whole argument for this file.

FOUR ORDERINGS ARE AT STAKE, all of them silent if wrong:

1. the shared expert sees the **layer input**, not the routed output;
2. it is added **after** the routed sum;
3. it is **not** multiplied by any routing weight;
4. ``routed_scaling_factor`` (2.5) is applied **once**, in the router, to the routed
   path only.

Each has a sabotage test below that confirms this file would catch it.

TWO CONSTRAINTS INHERITED FROM TODAY'S TWO BUGS, both load-bearing here:

* **The inputs must reach the behaviour.** The swiglu clamp is a no-op below +-10 and
  ``tiny_cfg`` initialises at ``normal_(0, 0.02)``, so a comparison at default scale
  passes with a missing clamp (``test_mlp_moe.py`` measures that trap). Every agreement
  test here runs at a scale where ``test_routed_clamp_is_actually_reached`` has measured
  that the routed experts' pre-activations exceed the limit.
* **A scaling factor of 2.5 must be distinguishable from 1.0.**
  ``test_scaling_factor_is_not_a_no_op_at_this_scale`` measures that too, because an
  agreement test cannot validate a factor whose effect is inside its own tolerance.
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

# n_group == topk_group == 1 in the live config, where the group masking is provably the
# identity (one group holds every expert and is always chosen). hf_refs documents that.
N_GROUP = TOPK_GROUP = 1


def _moe_at_scale(std, *, cfg=None, seed=0):
    """A ``MoE`` whose routed pre-activations have roughly the given std, plus an input.

    ``x ~ N(0,1)`` over ``D`` inputs against ``W ~ N(0,s^2)`` gives pre-activations of
    std ``s*sqrt(D)``, so ``s = std / sqrt(D)`` -- the same construction
    ``test_mlp_moe.py`` uses, applied to the 3-D expert tensors and the shared expert.
    """
    cfg = cfg or R.tiny_cfg()
    D, I, E = cfg.hidden_size, cfg.moe_intermediate_size, cfg.n_routed_experts
    g = torch.Generator().manual_seed(seed)
    m = R.MoE(cfg)
    s = std / D ** 0.5
    with torch.no_grad():
        m.gate_up_proj.copy_(torch.randn(E, 2 * I, D, generator=g) * s)
        m.down_proj.copy_(torch.randn(E, D, I, generator=g) * I ** -0.5)
        m.shared_experts.gate_proj.weight.copy_(
            torch.randn(I * cfg.n_shared_experts, D, generator=g) * s)
        m.shared_experts.up_proj.weight.copy_(
            torch.randn(I * cfg.n_shared_experts, D, generator=g) * s)
        m.shared_experts.down_proj.weight.copy_(
            torch.randn(D, I * cfg.n_shared_experts, generator=g)
            * (I * cfg.n_shared_experts) ** -0.5)
        # A non-zero correction bias so it cannot be silently ignored; small enough that
        # it perturbs the choice without reordering everything.
        m.gate.e_score_correction_bias.copy_(torch.randn(E, generator=g) * 0.1)
        m.gate.weight.copy_(torch.randn(E, D, generator=g) * 0.05)
    x = torch.randn(2, 5, D, generator=g)
    return m, x, cfg


def _hf(m, x, cfg, **over):
    """The vendored transformers MoE over the same parameters."""
    kw = dict(top_k=cfg.num_experts_per_tok, n_group=N_GROUP, topk_group=TOPK_GROUP,
              norm_topk_prob=cfg.norm_topk_prob,
              routed_scaling_factor=cfg.routed_scaling_factor,
              swiglu_limit=cfg.swiglu_limit)
    kw.update(over)
    return H.hf_moe(x, m.gate.weight, m.gate.e_score_correction_bias,
                    m.gate_up_proj, m.down_proj,
                    m.shared_experts.gate_proj.weight, m.shared_experts.up_proj.weight,
                    m.shared_experts.down_proj.weight, **kw)


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


# ------------------------------------------------------- non-vacuity, measured not assumed

def test_routed_clamp_is_actually_reached():
    """The regime check. Without this the agreement tests below prove nothing about the
    clamp, exactly as ``test_mlp_moe.py``'s trap test demonstrates for the dense MLP."""
    for std, want in ((0.5, False), (40.0, True)):
        m, x, cfg = _moe_at_scale(std)
        flat = x.view(-1, x.shape[-1])
        _, idx = m.gate(flat)
        peak = max(
            F.linear(flat[torch.where(idx == e)[0]], m.gate_up_proj[e]).abs().max().item()
            for e in idx.unique()
        )
        reached = peak > cfg.swiglu_limit
        print(f"  std={std:>5}: peak routed pre-activation {peak:8.2f} "
              f"vs limit {cfg.swiglu_limit} -> clamp {'ACTIVE' if reached else 'inert'}")
        assert reached is want, (
            f"std={std} was meant to leave the clamp {'active' if want else 'inert'}; "
            f"peak {peak:.2f} vs limit {cfg.swiglu_limit}. This file's premise has changed."
        )


def test_scaling_factor_is_not_a_no_op_at_this_scale():
    """2.5 must be distinguishable from 1.0, or the agreement tests cannot see it."""
    m, x, cfg = _moe_at_scale(40.0)
    ref = _hf(m, x, cfg)
    unscaled = _hf(m, x, cfg, routed_scaling_factor=1.0)
    rel = _rel(unscaled, ref)
    print(f"  routed_scaling_factor 2.5 vs 1.0: {rel:.3e} relative")
    assert rel > 0.05, (
        f"scaling factor moves the output by only {rel:.3e}; an agreement test at this "
        "scale could not distinguish it from 1.0"
    )


# --------------------------------------------------------------------------- agreement

@pytest.mark.parametrize("std", [0.5, 5.0, 40.0, 200.0])
def test_moe_matches_transformers(std):
    """``MoE.forward`` vs the vendored ``Glm5NextTextMoE.forward``, across scales."""
    m, x, cfg = _moe_at_scale(std)
    got, ref = m(x), _hf(m, x, cfg)
    rel = _rel(got, ref)
    print(f"  std={std:>6}: {rel:.3e} relative")
    # Bit-exact in practice -- see test_agreement_is_bit_exact_and_why for the reason
    # this is not merely a tight tolerance.
    assert torch.equal(got, ref), f"MoE disagrees with transformers at std={std}: {rel:.3e}"


@pytest.mark.parametrize("b,s", [(2, 5), (8, 64), (16, 256)])
def test_agreement_is_bit_exact_and_why(b, s):
    """Agreement is **exactly** zero, and the reason is worth recording.

    I expected an ``index_add_`` ordering floor: the two implementations recover an
    expert's token list differently -- transformers takes ``(top_k_pos, token_idx)`` from
    a ``[top_k, tokens]`` one-hot mask, the oracle takes ``(tok, pos)`` from a
    ``[tokens, top_k]`` index tensor -- and float addition is not associative. **That
    hypothesis was wrong**, and measuring it rather than assuming it is the only reason
    this docstring is right:

    * within a single expert each token appears **at most once**, because a token selects
      an expert at most once in top-k. So no two source rows of one ``index_add_`` share
      a destination and there is nothing to reorder.
    * accumulation *across* experts is where order would matter, and both iterate experts
      in **ascending id** -- ``nonzero()`` and ``unique()`` are both sorted -- so the
      per-token sums are formed in the same sequence.

    Checked up to 4,096 tokens with 1,993 of them on one expert, so this is not an
    artifact of a tiny population. A tolerance would have hidden the fact that the
    tolerance was unnecessary.
    """
    m, _, cfg = _moe_at_scale(40.0)
    g = torch.Generator().manual_seed(7)
    x = torch.randn(b, s, cfg.hidden_size, generator=g)
    got, ref = m(x), _hf(m, x, cfg)
    flat = x.view(-1, x.shape[-1])
    _, idx = m.gate(flat)
    busiest = max(int((idx == e).sum()) for e in idx.unique())
    print(f"  {b * s:>5} tokens, busiest expert holds {busiest:>5}: "
          f"exact={torch.equal(got, ref)}")
    assert torch.equal(got, ref), (
        f"{_rel(got, ref):.3e} relative at {b * s} tokens -- if this ever fails, the "
        "ordering hypothesis in this docstring has become true and a tolerance is needed"
    )


def test_top8_width_composes(std=40.0):
    """The real config routes top-8 of 288. tiny_cfg is top-2 of 8, so widen it: the
    concern is that top_k > 1 composes with the gather/normalise/scale chain, not the
    absolute expert count."""
    cfg = R.tiny_cfg(n_routed_experts=16, num_experts_per_tok=8)
    m, x, cfg = _moe_at_scale(std, cfg=cfg)
    got, ref = m(x), _hf(m, x, cfg)
    print(f"  top-8 of 16: {_rel(got, ref):.3e} relative")
    assert torch.equal(got, ref)


# ---------------------------------------------------------------------------- sabotage
#
# Each of these breaks the REAL implementation in a specific, plausible way and asserts
# the comparison notices. A test that cannot fail is not evidence.

def _sabotage_rel(fn, std=40.0):
    """Run ``fn(m, x)`` in place of ``m(x)`` and return its disagreement with the ref."""
    m, x, cfg = _moe_at_scale(std)
    return _rel(fn(m, x, cfg), _hf(m, x, cfg))


def test_catches_shared_expert_weighted_by_routing():
    """Sabotage 3: multiply the shared expert by a routing weight. In the reference it is
    never scaled by routing at all."""
    def bad(m, x, cfg):
        flat = x.view(-1, x.shape[-1])
        w, idx = m.gate(flat)
        routed = _routed_only(m, flat, w, idx).view(x.shape)
        # scale the shared branch by each token's mean routing weight
        return routed + m.shared_experts(x) * w.mean(-1).view(*x.shape[:2], 1)
    rel = _sabotage_rel(bad)
    print(f"  shared expert weighted by routing: {rel:.3e} relative")
    assert rel > 1e-3, "sabotage not detected: the comparison cannot see this error"


def test_catches_shared_expert_fed_routed_output():
    """Sabotage 1: feed the shared expert the routed output instead of the layer input.
    ``residuals`` is captured before anything runs precisely to prevent this."""
    def bad(m, x, cfg):
        flat = x.view(-1, x.shape[-1])
        w, idx = m.gate(flat)
        routed = _routed_only(m, flat, w, idx).view(x.shape)
        return routed + m.shared_experts(routed)
    rel = _sabotage_rel(bad)
    print(f"  shared expert fed routed output: {rel:.3e} relative")
    assert rel > 1e-3, "sabotage not detected"


def test_catches_scaling_applied_to_the_sum():
    """Sabotage 2+4: scale AFTER adding the shared expert, so the shared branch is
    scaled too. The router applies 2.5 to ``topk_weights``, so only the routed path
    should carry it."""
    def bad(m, x, cfg):
        flat = x.view(-1, x.shape[-1])
        w, idx = m.gate(flat)
        unscaled = w / cfg.routed_scaling_factor
        routed = _routed_only(m, flat, unscaled, idx).view(x.shape)
        return (routed + m.shared_experts(x)) * cfg.routed_scaling_factor
    rel = _sabotage_rel(bad)
    print(f"  scaling applied to routed+shared: {rel:.3e} relative")
    assert rel > 1e-3, "sabotage not detected"


def test_catches_scaling_applied_twice():
    """Sabotage 4: apply ``routed_scaling_factor`` in ``MoE`` as well as the router --
    the error someone makes when they cannot find where it is applied."""
    def bad(m, x, cfg):
        flat = x.view(-1, x.shape[-1])
        w, idx = m.gate(flat)
        routed = _routed_only(m, flat, w * cfg.routed_scaling_factor, idx).view(x.shape)
        return routed + m.shared_experts(x)
    rel = _sabotage_rel(bad)
    print(f"  scaling applied twice: {rel:.3e} relative")
    assert rel > 1e-3, "sabotage not detected"


def test_catches_missing_shared_expert():
    """The coarsest failure, included because the shared expert is one term in a sum and
    a dropped term is exactly the kind of thing an end-to-end check at loose tolerance
    misses."""
    def bad(m, x, cfg):
        flat = x.view(-1, x.shape[-1])
        w, idx = m.gate(flat)
        return _routed_only(m, flat, w, idx).view(x.shape)
    rel = _sabotage_rel(bad)
    print(f"  shared expert omitted: {rel:.3e} relative")
    assert rel > 1e-2, "sabotage not detected"


def _routed_only(m, flat, w, idx):
    """The oracle's routed path alone, lifted from ``MoE.forward`` so the sabotage
    variants differ from it only in the term under test."""
    out = torch.zeros_like(flat)
    for e in idx.unique():
        tok, pos = torch.where(idx == e)
        gu = F.linear(flat[tok], m.gate_up_proj[e])
        g, u = gu.chunk(2, -1)
        h = F.silu(g.clamp(max=m.limit)) * u.clamp(-m.limit, m.limit)
        out.index_add_(0, tok, (F.linear(h, m.down_proj[e]) * w[tok, pos, None]).to(out.dtype))
    return out


def test_routed_only_helper_reproduces_the_real_routed_path():
    """The sabotage tests are only meaningful if ``_routed_only`` is faithful: otherwise
    they would 'detect' the helper rather than the sabotage."""
    m, x, cfg = _moe_at_scale(40.0)
    flat = x.view(-1, x.shape[-1])
    w, idx = m.gate(flat)
    rebuilt = _routed_only(m, flat, w, idx).view(x.shape) + m.shared_experts(x)
    assert torch.equal(rebuilt, m(x)), "_routed_only has drifted from MoE.forward"


# ------------------------------------------------------------- a flagged difference

def test_expert_sentinel_guard_is_unreachable_here():
    """transformers skips ``expert_idx == num_experts``; the oracle has no counterpart.

    Recorded rather than fixed: top-k over ``num_experts`` cannot produce that index, so
    the branch is dead for this router. It would only matter under an expert-parallel
    layout that pads unrouted tokens with a sentinel, which this oracle does not model.
    """
    m, x, cfg = _moe_at_scale(1.0)
    _, idx = m.gate(x.view(-1, x.shape[-1]))
    assert int(idx.max()) < cfg.n_routed_experts
    print(f"  max routed index {int(idx.max())} < num_experts {cfg.n_routed_experts}: "
          "sentinel unreachable")
