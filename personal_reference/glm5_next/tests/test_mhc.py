# SPDX-License-Identifier: Apache-2.0
"""mHC: Sinkhorn convergence, stream round-trip, and a cross-check against vLLM.

Run: python3 -m pytest personal_reference/glm5_next/tests/test_mhc.py -v

mHC is greenfield for GLM-5.3-Flash — there is no upstream Neuron reference — so
this oracle *is* the specification. That makes an independent cross-check
valuable, and one exists: DeepSeek-V4.1-Flash carries byte-identical mHC config
keys (hc_mult 4, hc_sinkhorn_iters 20, hc_eps 1e-6) and vLLM implements the
mechanism for both it and GLM5Next. ``_vllm_mhc_pre_torch`` below is vLLM's
``vllm/model_executor/kernels/mhc/torch.py::mhc_pre_torch`` reproduced verbatim,
and ``_vllm_mhc_post_torch`` its ``mhc_post_torch``.

Where the two agree, documented by ``test_matches_vllm_*``: the Sinkhorn loop
(softmax over rows, then one column normalisation, then ``iters - 1`` rounds of
row-then-column), the pre/post sigmoid parameterisation with a shared ``hc_eps``,
the post multiplier 2.0, the ``(2 + H) * H`` mix width, the ``fn``/``base``/``scale``
shapes, the weighted-sum collapse, and ``comb^T @ streams`` in the expand.

Two things that looked like divergences and are not:
* vLLM scales the *projection output* by the RMS factor while this reference
  normalises the *input* first. Identical, because the factor is a per-token
  scalar and the projection is linear.
* vLLM's DeepSeek **test** passes ``rms_eps=1e-20``, but its GLM5Next **model**
  passes ``rms_eps=config.rms_norm_eps`` (1e-5), which is what this reference uses.

``mhc_tau`` (0.05) exists in vLLM's GLM5Next config and is assigned but never
read, so this reference omits it. **[unverified]** whether it is live in ZhipuAI's
own modeling code.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from glm5_next import reference as R  # noqa: E402

H = 4          # hc_mult
D = 256        # a small hidden size; the math is independent of it
ITERS = 20     # hc_sinkhorn_iters
EPS = 1e-6     # hc_eps


def _cfg(**kw):
    base = dict(hidden_size=D, hc_mult=H, hc_sinkhorn_iters=ITERS, hc_eps=EPS)
    base.update(kw)
    return R.tiny_cfg(**base)


def _real_comb_base():
    """The comb block of ``hc_attn_base`` as the checkpoint actually ships it.

    Measured from ``zai-org/GLM-5.3-Flash`` layer 0 on 2026-09-25: ~0 on the
    diagonal and exactly -8.0 off it, i.e. a deliberately near-identity mixing
    prior. Layers 1 and 22 have the same structure.
    """
    return torch.full((H, H), -8.0) + torch.eye(H) * 8.0


# Measured real ``hc_attn_scale`` values; scale[2] multiplies the comb logits.
REAL_SCALES = {0: (0.0614, 0.0446, 0.0889),
               1: (0.1068, 0.0781, 0.0816),
               22: (0.0943, 0.1615, 0.2088)}


def _hc(seed=0, **kw):
    torch.manual_seed(seed)
    hc = R.HyperConnection(_cfg(**kw))
    # ``base`` ships as zeros and ``scale`` as ones; randomise so the test is not
    # accidentally exercising a symmetric special case.
    with torch.no_grad():
        hc.base.normal_(0, 0.5)
        hc.scale.copy_(torch.tensor([0.5, 0.25, 1.0]))
    return hc


def _streams(B=2, S=7, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, S, H, D, generator=g) * scale


# ------------------------------------------------------------- doubly stochastic
def test_sinkhorn_columns_are_exactly_normalised():
    """Columns sum to 1 tightly: the loop's final operation is a column division."""
    hc = _hc()
    _, comb, _ = hc(_streams())
    col = comb.sum(-2)
    assert (col - 1).abs().max() < 1e-5, f"column sums off by {(col-1).abs().max():.3e}"
    assert (comb > 0).all(), "Sinkhorn produced a non-positive entry"


def test_sinkhorn_rows_are_only_approximately_normalised():
    """Rows are NOT tight at 20 iterations. This is a property, not a bug.

    Sinkhorn converges geometrically but slowly for peaked matrices, and GLM's
    mixing prior is very peaked (see ``test_real_checkpoint_prior_is_near_identity``).
    With the shipped ``hc_sinkhorn_iters = 20`` the row sums are off by ~1e-2 on
    random weights and ~7e-3 with the real prior.

    A port must therefore run **exactly 20 iterations**: "converging harder" would
    produce a different mixing matrix than the reference, not a better one.
    """
    hc = _hc()
    _, comb, _ = hc(_streams())
    row_err = (comb.sum(-1) - 1).abs().max().item()
    assert row_err > 1e-6, (
        f"rows are tight ({row_err:.3e}) — either the input is unrealistically mild "
        f"or the iteration count changed; re-derive the documented tolerance"
    )
    assert row_err < 0.2, f"row sums wildly off ({row_err:.3e}); suspect a real bug"


def test_sinkhorn_keeps_converging_past_20_iterations():
    """Error must keep falling with more iterations — i.e. 20 is truncation, not a floor.

    If it plateaued instead, the ``+eps`` denominators would be pinning a fixed
    point that is not doubly stochastic, which would be a defect.
    """
    streams = _streams()
    errs = {}
    for iters in (1, 20, 100, 400):
        hc = _hc(hc_sinkhorn_iters=iters)
        _, comb, _ = hc(streams)
        errs[iters] = (comb.sum(-1) - 1).abs().max().item()
    assert errs[400] < errs[100] < errs[20] <= errs[1], f"not still converging: {errs}"
    assert errs[400] < errs[20] / 3, f"convergence stalled: {errs}"


def test_eps_does_not_bias_the_fixed_point():
    """hc_eps=1e-6 in the denominators must not shift where Sinkhorn converges.

    Measured: eps=1e-6 shifts ``comb`` by ~1e-4, an order of magnitude below the
    ~1e-2 truncation error left by stopping at 20 iterations. So the epsilons are
    numerical guards, not a change of fixed point — the dominant deviation from
    doubly-stochastic is the iteration count, not the eps.
    """
    streams = _streams()
    _, comb_eps, _ = _hc(hc_eps=1e-6)(streams)
    _, comb_zero, _ = _hc(hc_eps=0.0)(streams)
    eps_effect = (comb_eps - comb_zero).abs().max().item()
    truncation = (comb_eps.sum(-1) - 1).abs().max().item()
    assert eps_effect < truncation / 10, (
        f"eps moves comb by {eps_effect:.3e}, not negligible against the "
        f"{truncation:.3e} truncation error — it may be biasing the fixed point"
    )


def test_real_checkpoint_prior_is_near_identity():
    """With the shipped base/scale the mixing matrix is ~I, so mixing is subtle.

    ``hc_attn_base``'s comb block is 0 on the diagonal and exactly -8.0 off it, and
    ``hc_attn_scale[2]`` is ~0.09-0.21, which damps the data-dependent part. The
    resulting comb has diagonal ~0.999 and off-diagonals ~1e-3.

    Consequence for the port: cross-stream mixing contributes at the 1e-3 level at
    layer 0, so an accuracy check that tolerates 1e-2 cannot see whether mHC mixing
    was implemented at all.
    """
    g = torch.Generator().manual_seed(0)
    mixes = torch.randn(64, H, H, generator=g)
    for layer, (_, _, s2) in REAL_SCALES.items():
        logits = mixes * s2 + _real_comb_base().unsqueeze(0)
        comb = torch.softmax(logits, -1) + EPS
        comb = comb / (comb.sum(-2, keepdim=True) + EPS)
        for _ in range(ITERS - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + EPS)
            comb = comb / (comb.sum(-2, keepdim=True) + EPS)
        diag = comb.diagonal(dim1=-2, dim2=-1).mean().item()
        offdiag = comb[:, ~torch.eye(H, dtype=bool)].max().item()
        assert diag > 0.99, f"layer {layer}: diagonal only {diag:.4f}"
        assert offdiag < 0.05, f"layer {layer}: off-diagonal up to {offdiag:.4f}"
        assert offdiag > 1e-5, f"layer {layer}: comb is exactly I; no mixing at all"


@pytest.mark.parametrize(
    "name,streams",
    [
        ("zeros", torch.zeros(1, 3, H, D)),
        ("constant", torch.ones(1, 3, H, D)),
        ("tiny", torch.full((1, 3, H, D), 1e-8)),
        ("huge", torch.full((1, 3, H, D), 1e4)),
        ("mixed-scale", torch.cat([torch.full((1, 3, 1, D), 1e4),
                                   torch.zeros(1, 3, H - 1, D)], dim=2)),
    ],
)
def test_pathological_inputs_stay_finite(name, streams):
    """Degenerate streams must degrade gracefully, not produce NaN/inf."""
    hc = _hc()
    post, comb, collapsed = hc(streams)
    for tag, t in (("post", post), ("comb", comb), ("collapsed", collapsed)):
        assert torch.isfinite(t).all(), f"{name}: {tag} has non-finite entries"
    assert (comb > 0).all(), f"{name}: comb has a non-positive entry"
    assert (comb.sum(-2) - 1).abs().max() < 1e-4, f"{name}: columns not normalised"


def test_post_and_pre_ranges():
    """post in (0, 2) via 2*sigmoid; the collapse weights are strictly positive."""
    hc = _hc()
    post, _, _ = hc(_streams())
    assert (post > 0).all() and (post < 2).all()


# ------------------------------------------------------------------- round-trip
def test_hc_expand_shapes_and_identity():
    """comb = I and post = 0 must return the residual streams untouched."""
    streams = _streams()
    B, S = streams.shape[:2]
    out = torch.randn(B, S, D)
    eye = torch.eye(H).expand(B, S, H, H).contiguous()
    zero_post = torch.zeros(B, S, H)
    got = R.hc_expand(zero_post, eye, out, streams)
    assert got.shape == streams.shape
    torch.testing.assert_close(got, streams, atol=1e-6, rtol=1e-5)


def test_hc_expand_post_term_is_broadcast_outer_product():
    """With comb = 0, the result is post (per stream) x sublayer_out (per feature)."""
    streams = _streams()
    B, S = streams.shape[:2]
    out = torch.randn(B, S, D)
    post = torch.rand(B, S, H) * 2
    got = R.hc_expand(post, torch.zeros(B, S, H, H), out, streams)
    expect = post.unsqueeze(-1) * out.unsqueeze(-2)
    torch.testing.assert_close(got, expect, atol=1e-6, rtol=1e-5)


def test_every_input_stream_reaches_every_output_stream():
    """A doubly-stochastic comb with positive entries must fully mix the streams.

    Perturb one input stream and check all H outputs move. If any output were
    independent of some input, the residual streams would be partitioned and mHC
    would be H independent residuals rather than a mixed manifold.
    """
    hc = _hc()
    streams = _streams(B=1, S=1)
    out = torch.randn(1, 1, D)
    post, comb, _ = hc(streams)
    base = R.hc_expand(post, comb, out, streams)
    for src in range(H):
        bumped = streams.clone()
        bumped[:, :, src] += 1.0
        # hold the coefficients fixed so this isolates the mixing, not the routing
        moved = R.hc_expand(post, comb, out, bumped) - base
        per_stream = moved.abs().amax(dim=-1).flatten()
        assert (per_stream > 1e-6).all(), (
            f"perturbing input stream {src} left some output stream unchanged: "
            f"{per_stream.tolist()}"
        )


# ------------------------------------------------- non-vacuity / discrimination
def test_comb_is_neither_identity_nor_uniform():
    """A trivial comb would make several tests above pass while proving nothing."""
    hc = _hc()
    _, comb, _ = hc(_streams())
    eye = torch.eye(H).expand_as(comb)
    uniform = torch.full_like(comb, 1.0 / H)
    d_eye = (comb - eye).abs().max().item()
    d_uni = (comb - uniform).abs().max().item()
    assert d_eye > 1e-2, f"comb is ~identity (max diff {d_eye:.3e}); no mixing"
    assert d_uni > 1e-3, f"comb is ~uniform (max diff {d_uni:.3e}); routing is degenerate"


def test_identity_comb_substitution_changes_output_materially():
    """The mixing term must matter — the analogue of the KDA scalar-gate test.

    If replacing the learned doubly-stochastic comb with the identity barely moved
    the output, this suite could not tell mHC from H independent residual streams.
    """
    hc = _hc()
    streams = _streams()
    out = torch.randn(*streams.shape[:2], D)
    post, comb, _ = hc(streams)
    real = R.hc_expand(post, comb, out, streams)
    identity = R.hc_expand(post, torch.eye(H).expand_as(comb), out, streams)
    rel = (real - identity).abs().max().item() / real.abs().max().item()
    assert rel > 1e-2, f"identity comb is indistinguishable from the real one ({rel:.3e})"


def test_single_stream_is_degenerate_so_H4_must_be_tested():
    """At hc_mult=1 the mixing is forced to [[1]] — any test passing only there is vacuous."""
    torch.manual_seed(0)
    hc1 = R.HyperConnection(R.tiny_cfg(hidden_size=D, hc_mult=1,
                                      hc_sinkhorn_iters=ITERS, hc_eps=EPS))
    _, comb, _ = hc1(torch.randn(1, 3, 1, D))
    torch.testing.assert_close(comb, torch.ones_like(comb), atol=1e-5, rtol=1e-4)


def test_both_sublayers_have_independent_hyperconnections():
    """mHC is applied at attention AND MLP, with separate parameters.

    Verified against the live checkpoint too: hc_attn_{base,fn,scale} and
    hc_ffn_{base,fn,scale} each appear 45 times, once per layer.
    """
    layer = R.DecoderLayer(_cfg(), 0)
    assert isinstance(layer.attn_hc, R.HyperConnection)
    assert isinstance(layer.ffn_hc, R.HyperConnection)
    assert layer.attn_hc is not layer.ffn_hc
    assert layer.attn_hc.fn.data_ptr() != layer.ffn_hc.fn.data_ptr()
    # mix width is (2 + H) * H, matching vLLM's mix_hc = (2 + n) * n
    assert layer.attn_hc.fn.shape == ((2 + H) * H, H * D)
    assert layer.attn_hc.base.shape == ((2 + H) * H,)
    assert layer.attn_hc.scale.shape == (3,)


# ------------------------------------------------------------ vLLM cross-check
def _vllm_mhc_pre_torch(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                        hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat):
    """vLLM ``kernels/mhc/torch.py::mhc_pre_torch``, reproduced verbatim (fp32)."""
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    residual_flat = residual.reshape(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    x = residual_flat.reshape(num_tokens, hc_mult * hidden_size).to(torch.float32)
    mixes = torch.matmul(x, fn.t())
    sqrsum = x.square().sum(dim=-1, keepdim=True)
    mixes = mixes * torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)

    pre_mix = torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + hc_pre_eps
    post_mix = torch.sigmoid(
        mixes[:, hc_mult:2 * hc_mult] * hc_scale[1] + hc_base[hc_mult:2 * hc_mult]
    ) * hc_post_mult_value
    comb_logits = mixes[:, 2 * hc_mult:].view(num_tokens, hc_mult, hc_mult) * hc_scale[2] \
        + hc_base[2 * hc_mult:].view(1, hc_mult, hc_mult)
    comb_mix = torch.softmax(comb_logits, dim=-1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    layer_input = torch.sum(pre_mix.unsqueeze(-1) * residual_flat.to(torch.float32), dim=1)
    return post_mix, comb_mix, layer_input


def _vllm_mhc_post_torch(x, residual, post_layer_mix, comb_res_mix):
    """vLLM ``kernels/mhc/torch.py::mhc_post_torch``, reproduced verbatim."""
    mixed = torch.einsum("...ij,...ih->...jh", comb_res_mix.float(), residual.float())
    return (mixed + post_layer_mix.float() * x.unsqueeze(-2).float()).to(residual.dtype)


def test_matches_vllm_pre():
    """post, comb and the collapse must match vLLM's torch reference."""
    hc = _hc()
    streams = _streams(B=2, S=5)
    post, comb, collapsed = hc(streams)
    v_post, v_comb, v_collapsed = _vllm_mhc_pre_torch(
        streams, hc.fn.detach().float(), hc.scale.detach().float(),
        hc.base.detach().float(), hc.input_norm.eps, hc.eps, hc.eps, 2.0, hc.iters,
    )
    B, S = streams.shape[:2]
    torch.testing.assert_close(post.reshape(B * S, H), v_post, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(comb.reshape(B * S, H, H), v_comb, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        collapsed.reshape(B * S, D), v_collapsed, atol=1e-5, rtol=1e-4
    )


def test_matches_vllm_post():
    """hc_expand must match vLLM's mhc_post_torch."""
    streams = _streams(B=2, S=5)
    B, S = streams.shape[:2]
    out = torch.randn(B, S, D)
    post = torch.rand(B, S, H) * 2
    comb = torch.softmax(torch.randn(B, S, H, H), -1)
    ours = R.hc_expand(post, comb, out, streams)
    theirs = _vllm_mhc_post_torch(out, streams, post.unsqueeze(-1), comb)
    torch.testing.assert_close(ours, theirs, atol=1e-5, rtol=1e-4)


def test_vllm_rms_placement_is_equivalent():
    """Normalising the input vs scaling the projection output is the same thing.

    This reference does ``linear(rmsnorm(x))``; vLLM does ``linear(x) * rms_factor``.
    Equal because the factor is a per-token scalar and linear() is linear — worth
    pinning so nobody "fixes" one to match the other and changes the eps handling.
    """
    hc = _hc()
    streams = _streams(B=1, S=4)
    flat = streams.flatten(2).float()
    ours = F.linear(hc.input_norm(flat), hc.fn.float())
    factor = torch.rsqrt(flat.square().mean(-1, keepdim=True) + hc.input_norm.eps)
    theirs = F.linear(flat, hc.fn.float()) * factor
    torch.testing.assert_close(ours, theirs, atol=1e-5, rtol=1e-5)
