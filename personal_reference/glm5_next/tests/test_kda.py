# SPDX-License-Identifier: Apache-2.0
"""KDA: chunked prefill must equal the single-step recurrence.

Run: python3 -m pytest personal_reference/glm5_next/tests/test_kda.py -v

Three things this file is careful about, because a weak version of each passes
while proving nothing:

1. **Chunk count.** ``chunk_kda`` pads to a multiple of ``chunk`` and carries state
   between chunks in a Python loop. At T < chunk that loop runs once and the carry
   is never exercised. Every tolerance test below spans 1 -> 4+ chunks and asserts
   the error does not grow with chunk count.
2. **Decay saturation.** GLM's real gate is ``-5.0 * sigmoid(...)``, so ``exp(g)``
   sits near 0.08 and the state is forgotten within a few tokens — which makes the
   carry almost irrelevant and any carry bug invisible. A long-memory regime
   (``exp(g) -> 1``) is tested explicitly.
3. **Per-channel vs scalar gate.** KDA's decay is per K-channel; GDN's is per-head
   scalar. ``test_gate_is_per_channel_not_scalar`` substitutes the scalar form and
   asserts the comparison *notices*, so the suite cannot silently accept a GDN
   kernel dropped in where KDA was meant.

**A SINGLE TEST REGIME IS NEVER SUFFICIENT HERE, AND THE RIGHT ONE DIFFERS PER
PROPERTY.** This has now bitten three times on this model, always the same way: the
regime that exercises one property is the weak one for the next.

* chunk-vs-recurrent carry: the LONG-decay regime is strong (the carry matters);
  the real gate forgets state within a few tokens and hides carry bugs.
* per-channel-vs-scalar gate: the REAL gate is strong (exp(g) spreads wide, so the
  two gates diverge); in the long regime everything sits near 1 and they nearly
  coincide -- only 4.2x the bf16 floor on the real kernel, too thin to gate on.
* input-perturbation probes: weak under the real gate (3x) precisely because it
  forgets state, stronger under long decay (6.5x).

So do not inherit a regime, or a threshold, from a neighbouring test. Pick the regime
that makes the property under test visible, and say which one you picked and why.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from glm5_next import reference as R  # noqa: E402

# Real GLM-5.3-Flash KDA geometry: 64 heads x head_dim 128, K == V.
H, K = 64, 128
CHUNK = 64


def _inputs(T, B=2, heads=H, dim=K, decay="real", seed=0):
    """Random KDA inputs shaped as the layer produces them: [B, T, H, dim].

    decay="real"  -> exp(g) ~ 0.08, GLM's actual gate (state forgotten fast)
    decay="long"  -> exp(g) in [0.95, 1.0], state persists across chunks
    """
    g_ = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g_)
    q, k, v = r(B, T, heads, dim), r(B, T, heads, dim), r(B, T, heads, dim)
    if decay == "real":
        # g = lower * sigmoid(exp(A_log) * raw), lower = -5.0  (ForgetGate)
        g = -5.0 * torch.sigmoid(r(B, T, heads, dim))
    else:
        g = torch.empty(B, T, heads, dim).uniform_(0.95, 1.0, generator=g_).log()
    beta = torch.sigmoid(r(B, T, heads))
    state = r(B, heads, dim, dim) * 0.1
    return q, k, v, g, beta, state


def _run_recurrent(q, k, v, g, beta, state):
    """Drive recurrent_kda one token at a time -> ([B,T,H,V], final_state)."""
    outs = []
    s = state.clone()
    for t in range(q.shape[1]):
        o, s = R.recurrent_kda(
            q[:, t : t + 1], k[:, t : t + 1], v[:, t : t + 1],
            g[:, t : t + 1], beta[:, t : t + 1], s,
        )
        outs.append(o)
    return torch.cat(outs, 1), s


def _max_diff(T, decay, seed=0, heads=H, dim=K):
    q, k, v, g, beta, s0 = _inputs(T, heads=heads, dim=dim, decay=decay, seed=seed)
    out_c, s_c = R.chunk_kda(q, k, v, g, beta, s0.clone(), chunk=CHUNK)
    out_r, s_r = _run_recurrent(q, k, v, g, beta, s0)
    return (
        (out_c - out_r).abs().max().item(),
        (s_c - s_r).abs().max().item(),
        out_r.abs().max().item(),
    )


# --------------------------------------------------------------------- tolerance
@pytest.mark.parametrize("T", [64, 128, 192, 256])
@pytest.mark.parametrize("decay", ["real", "long"])
def test_chunk_matches_recurrent(T, decay):
    """The two paths agree to ~1e-7 regardless of chunk count or decay regime."""
    d_out, d_state, scale = _max_diff(T, decay)
    assert d_out < 1e-6, f"T={T} decay={decay}: output diff {d_out:.3e}"
    assert d_state < 1e-5, f"T={T} decay={decay}: state diff {d_state:.3e}"
    # guard against a vacuous pass on all-zero outputs
    assert scale > 1e-3, f"outputs are ~0 ({scale:.3e}); the test proves nothing"


@pytest.mark.parametrize("decay", ["real", "long"])
def test_error_does_not_grow_with_chunk_count(decay):
    """A broken inter-chunk carry shows up as error scaling with chunk count."""
    diffs = [_max_diff(T, decay)[0] for T in (64, 128, 192, 256)]
    assert max(diffs) < 4 * max(min(diffs), 1e-12), (
        f"decay={decay}: error grows with chunk count {diffs} — suspect the carry"
    )


def test_ragged_sequence_length():
    """T not a multiple of chunk: padding must not leak into output or state."""
    for T in (1, 7, 65, 130):
        d_out, d_state, _ = _max_diff(T, "long", seed=T)
        assert d_out < 1e-6, f"T={T}: output diff {d_out:.3e}"
        assert d_state < 1e-5, f"T={T}: state diff {d_state:.3e}"


# ------------------------------------------------------------------- non-vacuity
def test_carry_reaches_across_chunks():
    """Changing the initial state must move the LAST token, 3 chunks downstream."""
    q, k, v, g, beta, s0 = _inputs(192, decay="long", seed=1)
    out_a, _ = R.chunk_kda(q, k, v, g, beta, s0.clone(), chunk=CHUNK)
    out_b, _ = R.chunk_kda(q, k, v, g, beta, (s0 * 2).clone(), chunk=CHUNK)
    moved = (out_a[:, -1] - out_b[:, -1]).abs().max().item()
    agreement = _max_diff(192, "long", seed=1)[0]
    assert moved > 100 * agreement, (
        f"initial state barely reaches the final token ({moved:.3e} vs agreement "
        f"{agreement:.3e}); the carry is not being exercised"
    )


def test_comparison_is_sensitive():
    """A small perturbation must be visible far above the agreement floor."""
    q, k, v, g, beta, s0 = _inputs(192, decay="long", seed=2)
    out, _ = R.chunk_kda(q, k, v, g, beta, s0.clone(), chunk=CHUNK)
    out_p, _ = R.chunk_kda(q, k, v, g, beta, (s0 + 1e-2).clone(), chunk=CHUNK)
    moved = (out - out_p).abs().max().item()
    assert moved > 100 * _max_diff(192, "long", seed=2)[0]


# --------------------------------------------------- KDA is not GDN (per-channel)
def _scalar_gate_chunk(q, k, v, g, beta, state, chunk=CHUNK):
    """GDN's gate: collapse the per-channel decay to one scalar per head.

    Mean over K is the most favourable possible scalar reduction — it preserves
    the average decay rate, so any difference this test sees is genuinely the
    per-channel structure and not a change in overall magnitude.
    """
    g_scalar = g.mean(-1, keepdim=True).expand_as(g)
    return R.chunk_kda(q, k, v, g_scalar, beta, state, chunk=chunk)


def test_gate_is_per_channel_not_scalar():
    """Substituting a per-head scalar decay must change the result materially.

    If this ever fails, the suite can no longer tell KDA from GDN, and a
    gated-DeltaNet kernel could be dropped in where KDA is required.
    """
    q, k, v, g, beta, s0 = _inputs(192, decay="long", seed=3)
    out_kda, s_kda = R.chunk_kda(q, k, v, g, beta, s0.clone(), chunk=CHUNK)
    out_gdn, s_gdn = _scalar_gate_chunk(q, k, v, g, beta, s0.clone())
    rel = (out_kda - out_gdn).abs().max().item() / out_kda.abs().max().item()
    agreement = _max_diff(192, "long", seed=3)[0]
    assert (out_kda - out_gdn).abs().max().item() > 1000 * agreement, (
        "per-head scalar decay is indistinguishable from per-channel decay here; "
        "this test cannot detect a GDN kernel substituted for KDA"
    )
    assert rel > 1e-3, f"relative difference only {rel:.3e}"


def test_forget_gate_is_per_channel_shaped():
    """ForgetGate must emit [B,S,H,K], not [B,S,H] — the structural difference."""
    cfg = R.tiny_cfg()
    fg = R.ForgetGate(cfg)
    g = fg(torch.randn(2, 5, cfg.hidden_size))
    assert g.shape == (2, 5, cfg.linear_num_heads, cfg.linear_head_dim), g.shape
    # the real gate is bounded in (lower, 0)
    assert (g <= 0).all() and (g >= cfg.linear_lower_bound).all()
    # and it must not be constant across the channel axis (that would be GDN)
    assert g.std(-1).min() > 1e-6, "gate is constant per head — that is GDN, not KDA"


# ------------------------------------------------------- full layer, incl. conv
def test_layer_prefill_matches_step_decode():
    """LinearAttention prefill == its own token-by-token decode path.

    Covers the depthwise conv state handoff as well as the recurrence, which the
    function-level tests above do not reach.
    """
    torch.manual_seed(0)
    cfg = R.tiny_cfg()
    la = R.LinearAttention(cfg).eval()
    x = torch.randn(1, 12, cfg.hidden_size)
    with torch.no_grad():
        full, _ = la(x)
        stepped = R.step_decode(la, x)
    d = (full - stepped).abs().max().item()
    assert full.abs().max().item() > 1e-3
    assert d < 1e-5, f"prefill vs decode max|diff| = {d:.3e}"


def test_conv_state_round_trips():
    """forward must RETURN an advanced conv_state; torch.roll does not mutate.

    Regression guard: the upstream reference returned only rec_state, so a caller
    stepping one token at a time silently reused an all-zero conv window.
    """
    torch.manual_seed(0)
    cfg = R.tiny_cfg()
    la = R.LinearAttention(cfg).eval()
    kernel = la.conv1d.weight.shape[-1]
    cs = torch.zeros(1, 3 * la.HK, kernel)
    rs = torch.zeros(1, la.H, la.K, la.K)
    with torch.no_grad():
        _, (cs2, rs2) = la(torch.randn(1, 1, cfg.hidden_size), conv_state=cs, rec_state=rs)
    assert cs2 is not None and cs2.shape == cs.shape
    assert not torch.equal(cs2, cs), "conv_state did not advance"
    assert not torch.equal(rs2, rs), "rec_state did not advance"
    # prefill must also produce a usable handoff window
    with torch.no_grad():
        _, (cs3, _) = la(torch.randn(1, 9, cfg.hidden_size))
    assert cs3 is not None and cs3.shape[-1] == kernel


# ============================================================ the output gate (o_norm)
# The gate on RMSNormGated is SIGMOID, not silu/swish, and there is no config key for it.
# The oracle read silu until 2026-09-25 -- carried over from Qwen3.8-27B, where
# ``output_gate_type: "swish"`` genuinely is silu. Nothing caught it, because nothing
# compared o_norm against an external reference and the only test that ran through it
# (``test_layer_prefill_matches_step_decode``) compares the layer with ITSELF, which a
# wrong activation satisfies exactly.
#
# Both formulas below are vendored with provenance, as tests/indexer_refs.py does.

def _hf_rmsnorm_gated(x, gate, weight, eps):
    """transformers 5.17 ``Glm5NextTextRMSNormGated.forward``, verbatim apart from
    taking ``weight``/``eps`` as arguments and ACT2FN["sigmoid"] spelled out."""
    input_dtype = x.dtype
    hidden_states = x.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    hidden_states = weight.to(torch.float32) * hidden_states
    hidden_states = hidden_states * torch.sigmoid(gate.to(torch.float32))
    return hidden_states.to(input_dtype)


def _fla_gate(y, g, activation):
    """vLLM ``third_party/flash_linear_attention/ops/kda.py`` lines 233-236, verbatim:

        if ACTIVATION == "swish" or ACTIVATION == "silu":  b_y = b_y * b_g * sigmoid(b_g)
        elif ACTIVATION == "sigmoid":                      b_y = b_y * sigmoid(b_g)
    """
    if activation in ("swish", "silu"):
        return y * g * torch.sigmoid(g)
    if activation == "sigmoid":
        return y * torch.sigmoid(g)
    raise ValueError(activation)


def _gated_inputs(seed=0, H=8, dim=128):
    g_ = torch.Generator().manual_seed(seed)
    x = torch.randn(2, H, dim, generator=g_)
    gate = torch.randn(2, H, dim, generator=g_)
    return x, gate


def test_output_gate_matches_transformers_exactly():
    """Oracle o_norm == transformers Glm5NextTextRMSNormGated, to fp32 rounding."""
    cfg = R.FlashCfg()
    n = R.RMSNormGated(cfg.linear_head_dim, cfg.rms_norm_eps)
    with torch.no_grad():
        n.weight.normal_(1.0, 0.2)
    x, gate = _gated_inputs()
    with torch.no_grad():
        got = n(x, gate)
    want = _hf_rmsnorm_gated(x, gate, n.weight, cfg.rms_norm_eps)
    torch.testing.assert_close(got, want, rtol=0, atol=1e-6)


def test_output_gate_matches_vllm_sigmoid_branch():
    """Same, against FLA's explicit ``activation="sigmoid"`` branch (vLLM kda.py:291)."""
    cfg = R.FlashCfg()
    n = R.RMSNormGated(cfg.linear_head_dim, cfg.rms_norm_eps)
    with torch.no_grad():
        n.weight.normal_(1.0, 0.2)
    x, gate = _gated_inputs(seed=1)
    xf = x.float()
    base = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * n.weight.float()
    with torch.no_grad():
        got = n(x, gate)
    torch.testing.assert_close(got, _fla_gate(base, gate.float(), "sigmoid"), rtol=0, atol=1e-6)


def test_output_gate_is_sigmoid_not_swish():
    """THE DISCRIMINATING TEST. swish/silu is the wrong branch here, and it is the
    branch a port lands on by accident: it is FLA's DEFAULT (``activation: str =
    "swish"``), and it is correct for the sibling model Qwen3.8-27B. So a port that
    instantiates FusedRMSNormGated without passing activation= gets this wrong silently.

    The two differ by a factor of ``gate``, so this is not a tolerance question.
    """
    cfg = R.FlashCfg()
    n = R.RMSNormGated(cfg.linear_head_dim, cfg.rms_norm_eps)
    x, gate = _gated_inputs(seed=2)
    xf = x.float()
    base = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * n.weight.float()
    with torch.no_grad():
        got = n(x, gate)
    swish = _fla_gate(base, gate.float(), "swish")
    rel = (got - swish).abs().max() / got.abs().max()
    cos = torch.nn.functional.cosine_similarity(got.flatten(), swish.flatten(), 0)
    print(f"\n  o_norm sigmoid vs swish: relative {rel:.3f}, cosine {cos:.4f}")
    assert rel > 1.0, f"swish substitution only moves the output by {rel:.3f} relative"
    assert cos < 0.9, f"swish substitution is {cos:.4f}-aligned with sigmoid; test is weak"
    # and silu is spelled two ways in FLA -- both must be the same wrong branch
    torch.testing.assert_close(swish, _fla_gate(base, gate.float(), "silu"), rtol=0, atol=0)


def test_output_gate_eps_is_rms_norm_eps_not_1e_6():
    """transformers passes ``eps=self.layer_norm_epsilon`` (= rms_norm_eps = 1e-5).
    nkilib's gdn_tkg hardcodes 1e-6 in its RMSNormGated fold, so a KDA kernel adapted
    from it must carry 1e-5 across or it is wrong by a small but systematic amount.
    """
    cfg = R.FlashCfg()
    assert cfg.rms_norm_eps == 1e-5
    la = R.LinearAttention(cfg)
    assert la.o_norm.eps == cfg.rms_norm_eps
    # non-vacuous: at small activations the two epsilons genuinely differ
    x = torch.full((1, 1, cfg.linear_head_dim), 1e-3)
    gate = torch.zeros(1, 1, cfg.linear_head_dim)
    w = torch.ones(cfg.linear_head_dim)
    a = _hf_rmsnorm_gated(x, gate, w, 1e-5)
    b = _hf_rmsnorm_gated(x, gate, w, 1e-6)
    assert (a - b).abs().max() > 1e-3


# ======================================== GDN vs KDA: the kernel adaptation's premise
# The KDA NKI kernel is adapted from nkilib's GDN kernels on the premise that GDN is
# KDA with a per-head SCALAR gate. Everything in that port rests on it, so it is tested
# here rather than inherited from the prior art's header -- which has already proved too
# narrow once (it missed silu-vs-sigmoid and eps 1e-6-vs-1e-5 in the same kernel).
#
# These are CROSS-REFERENCE tests against nkilib's own shipped reference, not
# self-consistency tests. See tests/gdn_refs.py for provenance.

import os  # noqa: E402

from gdn_refs import gdn_cte_torch_nki_ref  # noqa: E402

GDN_D = 128          # head_dim; gdn_cte_torch takes one head per call, so H == 1 here


def _gdn_inputs(B=2, S=128, D=GDN_D, seed=0):
    g_ = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g_)
    q, k, v = r(B, S, D), r(B, S, D), r(B, S, D)
    beta = torch.sigmoid(r(B, S))
    gate = -5.0 * torch.sigmoid(r(B, S))        # per-token SCALAR log-decay, GDN-shaped
    return q, k, v, beta, gate


def _oracle_recurrent_1head(q, k, v, g_full, beta):
    """Drive the oracle's recurrent_kda over S tokens with H == 1, from a zero state."""
    B, S, D = q.shape
    st = torch.zeros(B, 1, D, D)
    outs = []
    for t in range(S):
        o, st = R.recurrent_kda(q[:, t:t + 1, None], k[:, t:t + 1, None], v[:, t:t + 1, None],
                                g_full[:, t:t + 1], beta[:, t:t + 1, None], st)
        outs.append(o)
    return torch.cat(outs, 1)[:, :, 0], st[:, 0]


def test_gdn_reference_is_bit_identical_to_kda_with_a_scalar_gate():
    """THE PREMISE. With the gate constant along the channel axis, nkilib's GDN
    reference and the oracle's recurrent_kda are the same fp32 computation.

    Bit-exact, not merely close: the two perform the same operations in the same order.
    That is what licenses adapting the GDN kernels at all, and it localises the port's
    risk to the gate plus the surrounding folds.
    """
    q, k, v, beta, gate = _gdn_inputs()
    # gdn_cte takes PRE-l2normed q,k and applies only `scale`; recurrent_kda l2norms itself.
    o_gdn, s_gdn = gdn_cte_torch_nki_ref(R.l2norm(q), R.l2norm(k), v, beta, gate,
                                         scale=GDN_D ** -0.5)
    g_scalar = gate[:, :, None, None].expand(-1, -1, 1, GDN_D).contiguous()
    o_orc, s_orc = _oracle_recurrent_1head(q, k, v, g_scalar, beta)
    assert o_gdn.abs().max() > 1e-3, "output is ~zero; the comparison would be vacuous"
    assert torch.equal(o_gdn, o_orc), f"out differs by {(o_gdn - o_orc).abs().max():.3e}"
    assert torch.equal(s_gdn, s_orc), f"state differs by {(s_gdn - s_orc).abs().max():.3e}"


def test_gdn_reference_matches_the_chunked_path_too():
    """Same premise against chunk_kda, which is what the prefill kernel must reproduce.
    Chunking is not bit-exact -- it reassociates -- so this is a tolerance test.
    """
    q, k, v, beta, gate = _gdn_inputs(seed=1)
    o_gdn, s_gdn = gdn_cte_torch_nki_ref(R.l2norm(q), R.l2norm(k), v, beta, gate,
                                         scale=GDN_D ** -0.5)
    g_scalar = gate[:, :, None, None].expand(-1, -1, 1, GDN_D).contiguous()
    o_chk, s_chk = R.chunk_kda(q[:, :, None], k[:, :, None], v[:, :, None],
                               g_scalar, beta[:, :, None], torch.zeros(q.shape[0], 1, GDN_D, GDN_D),
                               chunk=64)
    assert (o_gdn - o_chk[:, :, 0]).abs().max() < 1e-6
    assert (s_gdn - s_chk[:, 0]).abs().max() < 1e-5


def test_a_per_channel_gate_does_not_match_gdn():
    """THE DISCRIMINATING DIRECTION. A real KDA gate must NOT reproduce the GDN
    reference -- otherwise the whole per-channel adaptation would be unnecessary and
    this suite could not tell the two apart.
    """
    q, k, v, beta, gate = _gdn_inputs(seed=2)
    o_gdn, _ = gdn_cte_torch_nki_ref(R.l2norm(q), R.l2norm(k), v, beta, gate,
                                     scale=GDN_D ** -0.5)
    g_chan = -5.0 * torch.sigmoid(torch.randn(q.shape[0], q.shape[1], 1, GDN_D,
                                              generator=torch.Generator().manual_seed(7)))
    o_pc, _ = _oracle_recurrent_1head(q, k, v, g_chan, beta)
    rel = (o_gdn - o_pc).abs().max() / o_gdn.abs().max()
    print(f"\n  per-channel gate vs GDN reference: {rel:.3f} relative")
    assert rel > 0.05, f"per-channel gate only moves the output {rel:.4f} relative"


@pytest.mark.skipif(not os.environ.get("NKILIB_SRC"),
                    reason="set NKILIB_SRC to <nki-library>/src/nkilib_src to diff the vendored copy")
def test_vendored_gdn_reference_still_matches_the_live_checkout():
    """Drift guard: the vendored copy must still agree with nkilib's live file."""
    import importlib.util
    import pathlib as _pl
    src = _pl.Path(os.environ["NKILIB_SRC"]) / "nkilib/experimental/gdn/gdn_cte_torch.py"
    spec = importlib.util.spec_from_file_location("_live_gdn_cte_torch", src)
    live = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(live)
    q, k, v, beta, gate = _gdn_inputs(seed=3)
    a = gdn_cte_torch_nki_ref(q, k, v, beta, gate, scale=GDN_D ** -0.5)
    b = live.gdn_cte_torch_nki_ref(q, k, v, beta, gate, scale=GDN_D ** -0.5)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
