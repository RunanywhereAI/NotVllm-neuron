# SPDX-License-Identifier: Apache-2.0
"""The anchored sub-block factorisation KDA prefill needs, validated in torch.

Run: python3 -m pytest personal_reference/glm5_next/tests/test_kda_prefill_anchor.py -v -s

THE PROBLEM. Adapting nkilib's ``gdn_cte`` to KDA is a rewrite, not a port, because of
one thing. GDN builds ``decay[i,j] = exp(cg[i] - cg[j])`` as a ``[CHUNK, CHUNK]``
matrix and forms the intra-chunk operator as ``A = -beta * (k @ k^T) * decay`` -- two
cheap ops. KDA's decay is per-channel, so what is actually needed is

    A[i,j] = -beta[i] * sum_d  k[i,d] * k[j,d] * exp(cg[i,d] - cg[j,d])

with the decay **inside the d-sum**, where it does not factor out. Done literally it
needs a ``[CHUNK, CHUNK, K]`` intermediate: 64x64x128 fp32 = 2 MB per chunk per head.

THE FIX, which FLA/vLLM already use. Anchor the gate at each row-block's first token:

    exp(cg[i,d] - cg[j,d]) = exp(cg[i,d] - cg_n[d]) * exp(cg_n[d] - cg[j,d])

Pre-scale ``k`` on each side by its own factor and ``A = -(Kb' @ K''^T)`` is a **plain
matmul** again. Verified below to reproduce ``chunk_kda``'s operator to fp32 noise.

THE SUB-BLOCK SIZE IS DERIVED HERE, NOT INHERITED. nkilib's ``_SUBBLK = 16`` is sized
for its nilpotent recursive-doubling solve; FLA's ``BC = min(16, BT)`` is unexplained
in source. Both happen to be 16, which is a coincidence of two different arguments and
not a reason. The constraint that actually applies to *this* factorisation:

* Row factors ``exp(cg[i] - cg_n)`` for ``i >= n`` are **<= 1** (cg decreases). Safe;
  underflow is benign, since a decay too small to represent is one that does not matter.
* Column factors ``exp(cg_n - cg[j])`` on the **diagonal block** have ``j >= n``, so
  they are **>= 1** and grow with the distance to the anchor. This is the binding side.
* Worst case over ``BC`` tokens is ``exp(BC * |gate_lower_bound|)``, so safety requires

      BC * |gate_lower_bound|  <  ln(float32_max) = 88.7

  With GLM's ``gate_lower_bound = -5.0`` that is **BC <= 17**, and the largest power of
  two is **16**. Measured: BC 4/8/16 exact, BC 32 overflows to NaN even on real gates.

So 16 is correct for GLM for a reason that has nothing to do with either source's
reason -- and it is **config-derived**: a different ``gate_lower_bound`` moves it
(-6.0 would give 14, and 16 would overflow). fp32 and bf16 give the same bound, since
they share an exponent range.
"""
from __future__ import annotations

import math
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from glm5_next import reference as R  # noqa: E402

CHUNK = 64
K = 128
F32_MAX = torch.finfo(torch.float32).max
LN_F32_MAX = math.log(F32_MAX)


def max_safe_subblock(gate_lower_bound, dtype=torch.float32):
    """Largest sub-block whose anchored column factor cannot overflow ``dtype``."""
    return int(math.log(torch.finfo(dtype).max) / abs(gate_lower_bound))


def max_safe_pow2_subblock(gate_lower_bound, dtype=torch.float32):
    """The size a kernel can actually use: the largest POWER OF TWO under the bound.

    This is the number that matters, and it moves in jumps. At
    ``gate_lower_bound = -5.0`` the bound is 17 and the usable size is 16. At -6.0 the
    bound is 14 -- which reads like a small adjustment -- but the usable size **halves
    to 8**, which is a different tiling, not a tweak.
    """
    n = max_safe_subblock(gate_lower_bound, dtype)
    return 1 << (n.bit_length() - 1)


def _inputs(seed=0, decay="real", B=1, Hh=2, T=CHUNK):
    g_ = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g_)
    k = R.l2norm(r(B, Hh, T, K))
    beta = torch.sigmoid(r(B, Hh, T))
    if decay == "real":
        g = -5.0 * torch.sigmoid(r(B, Hh, T, K))
    else:                                   # adversarial: every gate pinned at the bound
        g = torch.full((B, Hh, T, K), -5.0)
    return k, beta, g


def exact_operator(k, beta, g):
    """``chunk_kda``'s intra-chunk operator, with the decay inside the d-sum."""
    cg = g.cumsum(-2)
    k_beta = k * beta[..., None]
    stri = torch.triu(torch.ones(CHUNK, CHUNK, dtype=torch.bool), 1)
    tri = torch.triu(torch.ones(CHUNK, CHUNK, dtype=torch.bool), 0)
    decay = (cg.unsqueeze(-2) - cg.unsqueeze(-3)).masked_fill(stri[..., None], float("-inf")).exp()
    return -(k_beta.unsqueeze(-2) * k.unsqueeze(-3) * decay).sum(-1).masked_fill(tri, 0)


def anchored_operator(k, beta, g, BC):
    """The same operator as PLAIN MATMULS of anchored, pre-scaled blocks.

    ``CHUNK % BC == 0`` is required and asserted: a non-dividing BC silently leaves
    trailing tokens uncovered, which cost me a spurious "BC=17 is numerically unstable"
    result before I noticed the loop simply skipped 13 tokens.
    """
    assert CHUNK % BC == 0, f"BC={BC} must divide CHUNK={CHUNK}; a remainder is dropped"
    B, Hh = k.shape[:2]
    cg = g.cumsum(-2)
    k_beta = k * beta[..., None]
    A = torch.zeros(B, Hh, CHUNK, CHUNK)
    for bi in range(CHUNK // BC):
        n = bi * BC
        anchor = cg[..., n : n + 1, :]
        rows = slice(n, n + BC)
        kb_s = k_beta[..., rows, :] * (cg[..., rows, :] - anchor).exp()     # <= 1
        for bj in range(bi + 1):
            cols = slice(bj * BC, bj * BC + BC)
            k_s = k[..., cols, :] * (anchor - cg[..., cols, :]).exp()       # >= 1 on diagonal
            A[..., rows, cols] = -(kb_s @ k_s.transpose(-1, -2))            # PLAIN MATMUL
    return A.masked_fill(torch.triu(torch.ones(CHUNK, CHUNK, dtype=torch.bool), 0), 0)


def _worst_growth(g, BC):
    cg = g.cumsum(-2)
    return max((cg[..., bi * BC : bi * BC + 1, :] - cg[..., bi * BC : (bi + 1) * BC, :]).max().item()
               for bi in range(CHUNK // BC))


# ------------------------------------------------------- the factorisation is exact
@pytest.mark.parametrize("decay", ["real", "adversarial"])
@pytest.mark.parametrize("BC", [4, 8, 16])
def test_anchored_factorisation_reproduces_the_exact_operator(decay, BC):
    """Plain matmuls of pre-scaled blocks == the decay-inside-the-sum form."""
    k, beta, g = _inputs(decay=decay)
    ref = exact_operator(k, beta, g)
    got = anchored_operator(k, beta, g, BC)
    assert ref.abs().max() > 1e-3, "operator is ~zero; comparison would be vacuous"
    rel = (got - ref).abs().max().item() / ref.abs().max().item()
    assert torch.isfinite(got).all()
    assert rel < 1e-5, f"BC={BC} {decay}: relative error {rel:.2e}"


def test_the_anchor_position_bounds_the_range_not_the_answer():
    """A result I did not expect, and it sharpens why BC is the bound.

    Moving the anchor back by one block is **algebraically the identity** -- the two
    factors ``exp(cg_i - cg_wrong)`` and ``exp(cg_wrong - cg_j)`` still multiply to
    ``exp(cg_i - cg_j)`` for any anchor. So a "wrong" anchor cannot change the answer.

    What it changes is the RANGE each factor has to span: an anchor one block early
    makes the column factor reach across ``2 * BC`` tokens, which overflows exactly as
    ``BC = 32`` does. So the anchor is not a correctness choice at all -- it is purely
    a numerical-range choice, and that is precisely why the bound is on the SPAN from
    anchor to token rather than on the block size as such.
    """
    k, beta, g = _inputs(seed=3)
    cg = g.cumsum(-2)
    B, Hh = k.shape[:2]
    BC = 16
    A = torch.zeros(B, Hh, CHUNK, CHUNK)
    k_beta = k * beta[..., None]
    for bi in range(CHUNK // BC):
        n = bi * BC
        wrong = cg[..., max(n - BC, 0) : max(n - BC, 0) + 1, :]      # anchor one block early
        rows = slice(n, n + BC)
        kb_s = k_beta[..., rows, :] * (cg[..., rows, :] - wrong).exp()
        for bj in range(bi + 1):
            cols = slice(bj * BC, bj * BC + BC)
            k_s = k[..., cols, :] * (wrong - cg[..., cols, :]).exp()
            A[..., rows, cols] = -(kb_s @ k_s.transpose(-1, -2))
    finite = bool(torch.isfinite(A).all())
    print(f"\n  anchor one block early at BC=16: finite={finite} "
          f"(same algebra, twice the span -- overflows like BC=32)")
    assert not finite, (
        "an early anchor no longer overflows; the span-based bound needs re-deriving"
    )


# --------------------------------------------------------------- the sub-block bound
def test_the_bound_is_derived_from_gate_lower_bound():
    """BC * |gate_lower_bound| < ln(float_max). Config-derived, not inherited."""
    cfg = R.FlashCfg()
    assert cfg.linear_lower_bound == -5.0
    assert max_safe_subblock(cfg.linear_lower_bound) == 17
    assert max_safe_subblock(cfg.linear_lower_bound, torch.bfloat16) == 17, (
        "fp32 and bf16 share an exponent range, so the bound is the same"
    )
    # The bound moves with the config, and the USABLE size moves in jumps.
    assert max_safe_subblock(-6.0) == 14 and max_safe_subblock(-4.0) == 22
    assert max_safe_pow2_subblock(-5.0) == 16
    assert max_safe_pow2_subblock(-6.0) == 8, (
        "gate_lower_bound -6.0 does not shave the sub-block from 16 to 14 -- it HALVES "
        "it to 8, because only powers of two are usable. A config change here is a "
        "re-tiling of the kernel, not a constant adjustment."
    )
    assert max_safe_pow2_subblock(-4.0) == 16
    chosen = 16
    assert chosen == max_safe_pow2_subblock(cfg.linear_lower_bound)


@pytest.mark.parametrize("decay", ["real", "adversarial"])
def test_an_oversized_subblock_overflows(decay):
    """BC=32 is not merely inadvisable, it produces NaN -- on REAL gates, not just
    adversarial ones. This is what makes 16 a bound rather than a preference."""
    k, beta, g = _inputs(decay=decay)
    got = anchored_operator(k, beta, g, 32)
    growth = math.exp(min(_worst_growth(g, 32), 700))
    print(f"\n  {decay:>11} BC=32: worst growth exp() = {growth:.2e} vs fp32 max {F32_MAX:.2e}")
    assert growth > F32_MAX
    assert not torch.isfinite(got).all(), "BC=32 did not overflow; re-derive the bound"


@pytest.mark.parametrize("decay", ["real", "adversarial"])
def test_bc_16_has_real_but_finite_headroom(decay):
    """Recorded because it is thinner than it looks: ~6 orders of magnitude on
    adversarial gates. Not comfortable, and it is why the bound is worth asserting."""
    k, beta, g = _inputs(decay=decay)
    growth = math.exp(min(_worst_growth(g, 16), 700))
    headroom = F32_MAX / growth
    print(f"  {decay:>11} BC=16: growth {growth:.2e}, headroom {headroom:.2e}x")
    assert headroom > 1.0
    if decay == "adversarial":
        assert headroom < 1e12, "headroom is larger than measured; re-check the gate bound"


def test_non_dividing_subblock_is_rejected():
    """The guard for the harness bug that nearly became a false finding."""
    k, beta, g = _inputs()
    with pytest.raises(AssertionError, match="must divide"):
        anchored_operator(k, beta, g, 17)


# ===================================================== the whole prefill, anchored
# chunk_kda uses the [B,H,C,c,c,K] `decay` tensor in exactly TWO places -- the A
# operator and the intra-chunk output -- and both are the same bilinear form:
#
#     out[i,j] = sum_d  X[i,d] * Y[j,d] * exp(cg[i,d] - cg[j,d])
#
# So one anchored helper serves both, and NOTHING ELSE in chunk_kda needs a
# [c,c,K] intermediate. Everything else is elementwise or a plain matmul, and the
# per-channel gate lands on the partition axis where it is free.
#
# The other exp() uses are all <= 1 and underflow benignly: exp(cg) in `inter` and
# `k_cumdecay` (a fully decayed state contributes nothing), exp(cg_last - cg) in the
# state update, exp(cg_last) in the state decay. Only the anchored column factor
# grows, which is what BC bounds.

def anchored_bilinear(X, Y, cg, BC):
    """``out[i,j] = sum_d X[i,d] Y[j,d] exp(cg[i,d]-cg[j,d])`` as plain block matmuls.

    X, Y, cg: ``[..., c, K]``. Returns ``[..., c, c]``, lower-triangle region valid.
    """
    c = X.shape[-2]
    assert c % BC == 0, f"BC={BC} must divide chunk={c}; a remainder is dropped"
    out = X.new_zeros(*X.shape[:-2], c, c)
    for bi in range(c // BC):
        n = bi * BC
        anchor = cg[..., n : n + 1, :]
        rows = slice(n, n + BC)
        xs = X[..., rows, :] * (cg[..., rows, :] - anchor).exp()        # <= 1
        for bj in range(bi + 1):
            cols = slice(bj * BC, bj * BC + BC)
            ys = Y[..., cols, :] * (anchor - cg[..., cols, :]).exp()    # >= 1 on diagonal
            out[..., rows, cols] = xs @ ys.transpose(-1, -2)
    return out


def chunk_kda_anchored(q, k, v, g, beta, state=None, chunk=CHUNK, BC=16):
    """``reference.chunk_kda`` with every ``[c,c,K]`` intermediate removed.

    Structurally identical otherwise, so the diff against ``chunk_kda`` is exactly the
    part a NKI prefill kernel has to do differently. Validated against it below.
    """
    import torch.nn.functional as F
    dt = q.dtype
    q, k, v, beta, g = [t.transpose(1, 2).contiguous().float() for t in (q, k, v, beta, g)]
    q, k = R.l2norm(q), R.l2norm(k)
    B, Hh, T, Kd = k.shape
    V = v.shape[-1]
    pad = (chunk - T % chunk) % chunk
    Tp = T + pad
    q = F.pad(q, (0, 0, 0, pad)) * (Kd ** -0.5)
    k, v = F.pad(k, (0, 0, 0, pad)), F.pad(v, (0, 0, 0, pad))
    g, beta = F.pad(g, (0, 0, 0, pad)), F.pad(beta, (0, pad))
    v_beta, k_beta = v * beta[..., None], k * beta[..., None]
    rs = lambda t: t.reshape(B, Hh, -1, chunk, t.shape[-1])
    q, k, v, g, k_beta, v_beta = map(rs, (q, k, v, g, k_beta, v_beta))
    g = g.cumsum(-2)
    tri = torch.triu(torch.ones(chunk, chunk, dtype=torch.bool), 0)
    stri = torch.triu(torch.ones(chunk, chunk, dtype=torch.bool), 1)

    # THE CHANGE: A via anchored blocks instead of the [c,c,K] decay tensor.
    attn = -anchored_bilinear(k_beta, k, g, BC).masked_fill(tri, 0)
    for i in range(1, chunk):
        row, sub = attn[..., i, :i].clone(), attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk)
    v = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp())

    S = torch.zeros(B, Hh, Kd, V) if state is None else state.float()
    out = torch.zeros_like(v)
    for i in range(Tp // chunk):
        q_i, k_i, v_i, g_i = q[:, :, i], k[:, :, i], v[:, :, i], g[:, :, i]
        inter = (q_i * g_i.exp()) @ S
        # THE CHANGE, second site: intra output via anchored blocks.
        intra = anchored_bilinear(q_i, k_i, g_i, BC).masked_fill(stri, 0)
        v_new = v_i - k_cumdecay[:, :, i] @ S
        out[:, :, i] = inter + intra @ v_new
        S = S * g_i[:, :, -1].exp().unsqueeze(-1) + (k_i * (g_i[:, :, -1:] - g_i).exp()).transpose(-1, -2) @ v_new
    out = out.reshape(B, Hh, -1, V)[:, :, :T].transpose(1, 2).contiguous().to(dt)
    return out, S


def _prefill_inputs(T, seed=0, B=1, Hh=2, decay="real"):
    g_ = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g_)
    q, k, v = r(B, T, Hh, K), r(B, T, Hh, K), r(B, T, Hh, K)
    g = -5.0 * torch.sigmoid(r(B, T, Hh, K)) if decay == "real" else \
        torch.empty(B, T, Hh, K).uniform_(0.95, 1.0, generator=g_).log()
    beta = torch.sigmoid(r(B, T, Hh))
    return q, k, v, g, beta, r(B, Hh, K, K) * 0.1


@pytest.mark.parametrize("T", [64, 128, 192])
@pytest.mark.parametrize("decay", ["real", "long"])
def test_anchored_prefill_matches_chunk_kda(T, decay):
    """The whole prefill path, with no [c,c,K] intermediate anywhere, must reproduce
    chunk_kda. Multiple chunks, so the inter-chunk state carry is exercised too."""
    q, k, v, g, beta, s0 = _prefill_inputs(T, decay=decay)
    ref_o, ref_s = R.chunk_kda(q, k, v, g, beta, s0.clone(), chunk=CHUNK)
    got_o, got_s = chunk_kda_anchored(q, k, v, g, beta, s0.clone(), chunk=CHUNK, BC=16)
    assert ref_o.abs().max() > 1e-3, "output ~zero; comparison would be vacuous"
    rel_o = (got_o - ref_o).abs().max().item() / ref_o.abs().max().item()
    rel_s = (got_s - ref_s).abs().max().item() / ref_s.abs().max().item()
    assert rel_o < 1e-5, f"T={T} {decay}: output relative {rel_o:.2e}"
    assert rel_s < 1e-5, f"T={T} {decay}: state relative {rel_s:.2e}"


@pytest.mark.parametrize("BC", [4, 8, 16])
def test_anchored_prefill_is_subblock_size_invariant(BC):
    """BC is a numerical-range choice, not a modelling one: every safe BC gives the
    same answer. If a future BC changed the result, the factorisation would be wrong."""
    q, k, v, g, beta, s0 = _prefill_inputs(128, seed=4)
    ref_o, _ = R.chunk_kda(q, k, v, g, beta, s0.clone(), chunk=CHUNK)
    got_o, _ = chunk_kda_anchored(q, k, v, g, beta, s0.clone(), chunk=CHUNK, BC=BC)
    assert (got_o - ref_o).abs().max().item() / ref_o.abs().max().item() < 1e-5


def test_anchored_prefill_never_builds_a_chunk_chunk_k_tensor():
    """The whole point. Guards against a 'simplification' that reintroduces the 2 MB
    intermediate -- which would still pass every numerical test above."""
    import inspect
    src = inspect.getsource(anchored_bilinear) + inspect.getsource(chunk_kda_anchored)
    for banned in ("unsqueeze(-3)", "unsqueeze(-2) *"):
        assert banned not in src, (
            f"{banned!r} is the [c,c,K] broadcast pattern this factorisation exists to "
            f"avoid; it reintroduces a 64x64x128 intermediate per chunk per head"
        )
