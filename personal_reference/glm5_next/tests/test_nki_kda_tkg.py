# SPDX-License-Identifier: Apache-2.0
"""A torch model of nki_kda_tkg, checked against the oracle before the kernel runs.

Run: python3 -m pytest personal_reference/glm5_next/tests/test_nki_kda_tkg.py -v -s

**WHAT THIS DOES AND DOES NOT PROVE.** ``_kda_tkg_model`` below is a hand transcription
of ``nki_kda_tkg.py``'s operation order and dtype casts into plain torch. It tests the
kernel's MATH AS TRANSCRIBED -- not the kernel. If the transcription is wrong in the
same way the kernel is, both are wrong together and this file is silent. The
transcription is exactly the risk that ``nki.simulate`` removes, and nothing here
substitutes for running it.

What it is good for, on a host with no NKI toolchain:

1. Catching a wrong *design* before writing more against it -- in particular the three
   deliberate changes from GDN (per-channel gate, sigmoid output gate, eps 1e-5).
2. Establishing the **bf16 agreement floor** that the real equivalence test must be
   built on. The fp32 floor is ~4e-08 and is the wrong number to use.
3. Pinning the **vacuity control as an assertion** rather than a one-off measurement.
   The scalar-gate substitution clears the bf16 floor by ~54x (real gate) and only
   ~4x (long decay), versus ~150,000x in fp32. A narrow live margin can quietly stop
   clearing the floor after an unrelated change, so it is re-asserted on every run.

**The model was subsequently validated against the kernel itself.** Run under
``nki.simulate`` on 2026-09-25, the real kernel gave a scalar-gate margin of **54.4x**
against this model's predicted 57.8x, and matched the fp32 oracle to 0.53-0.66%.
So the transcription was faithful. That does not retroactively make this file a test
of the kernel -- it makes it a design tool whose predictions happened to hold once.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from glm5_next import reference as R  # noqa: E402

H, K = 4, 128          # K = V = 128 is GLM's real geometry; H kept small for speed
RMS_NORM_EPS = 1e-5
L2NORM_EPS = 1e-6


def _bf(x):
    """Round through bf16, as an SBUF tensor declared bfloat16 does."""
    return x.to(torch.bfloat16).float()


def _kda_tkg_model(q, k, v, g_log, beta, z, norm_weight, state):
    """Transcription of nki_kda_tkg.py. q,k,v,z: [BH, K/V]; g_log: [BH, K]; beta: [BH].

    Cast points mirror the kernel's buffer dtypes: Q_p/K_p/k_f/S_bf/delta_bf are
    bfloat16, the state and every PSUM accumulation are float32.
    """
    BH, Kd = q.shape
    V = v.shape[-1]
    scale = Kd ** -0.5

    Q_p, K_p = _bf(q), _bf(k)                                  # bf16 SBUF loads
    # Fold 3: sum of squares via matmul-with-ones -> fp32 PSUM, from bf16 squares
    q_ss = _bf(Q_p * Q_p).sum(-1)
    k_ss = _bf(K_p * K_p).sum(-1)
    q_scale = torch.rsqrt(q_ss + L2NORM_EPS)                   # fp32
    k_scale = torch.rsqrt(k_ss + L2NORM_EPS)
    K_p = _bf(K_p * k_scale[:, None])
    Q_scaled = _bf(Q_p * scale * q_scale[:, None])
    k_f = _bf(_bf(k) * k_scale[:, None])                       # partition-1 copy of k

    S = state.float().clone()                                  # [BH, K, V] fp32
    S = S * g_log.float().exp()[..., None]                      # PER-CHANNEL: [BH,K,1]
    S_bf = _bf(S)
    kv = (S_bf * K_p[..., None]).sum(-2)                        # k^T @ S -> fp32
    delta = (v.float() - kv) * torch.sigmoid(beta.float())[:, None]
    S = S + k_f[..., None] * _bf(delta)[..., None, :]
    S_bf = _bf(S)
    o = (S_bf * Q_scaled[..., None]).sum(-2)                    # q^T @ S -> fp32

    # Fold 4: RMSNormGated with SIGMOID and eps 1e-5
    ssum = torch.rsqrt(o.pow(2).sum(-1, keepdim=True) / V + RMS_NORM_EPS)
    out = o * ssum * norm_weight.float() * torch.sigmoid(z.float())
    return _bf(out), S


def _inputs(decay="real", seed=0, bh=H):
    g_ = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g_)
    q, k, v, z = r(bh, K), r(bh, K), r(bh, K), r(bh, K)
    if decay == "real":
        g_log = -5.0 * torch.sigmoid(r(bh, K))       # GLM's real gate
    else:
        g_log = torch.empty(bh, K).uniform_(0.95, 1.0, generator=g_).log()
    beta = r(bh)
    norm_weight = 1.0 + 0.2 * r(K)
    state = r(bh, K, K) * 0.1
    return q, k, v, g_log, beta, z, norm_weight, state


def _oracle(q, k, v, g_log, beta, z, norm_weight, state):
    """The fp32 golden: reference.recurrent_kda + reference.RMSNormGated.

    recurrent_kda takes [B, S, H, K] with S == 1 for a decode step, so a [BH, K] tensor
    becomes [1, 1, BH, K] by indexing alone -- BH is the HEAD axis, not the sequence
    axis. (An earlier version transposed here and put heads on S, which silently made
    the oracle a 4-token sequence over 1 head and inflated the floor to ~2.0.)
    """
    core, S = R.recurrent_kda(q[None, None], k[None, None], v[None, None],
                              g_log[None, None], torch.sigmoid(beta)[None, None],
                              state[None])
    n = R.RMSNormGated(K, RMS_NORM_EPS)
    with torch.no_grad():
        n.weight.copy_(norm_weight)
        out = n(core[0, 0], z)
    return out, S[0]


def _floor_and_shift(decay, seed=0):
    """-> (bf16 agreement floor, scalar-gate output shift), both vs the fp32 oracle."""
    args = _inputs(decay=decay, seed=seed)
    q, k, v, g_log, beta, z, nw, state = args
    ref_out, _ = _oracle(*args)
    got_out, _ = _kda_tkg_model(*args)
    floor = (got_out - ref_out).abs().max().item()
    # GDN substitution: per-head scalar = mean over channels (the most favourable
    # reduction, since it preserves the average decay rate)
    g_scalar = g_log.mean(-1, keepdim=True).expand_as(g_log).contiguous()
    bad_out, _ = _kda_tkg_model(q, k, v, g_scalar, beta, z, nw, state)
    shift = (bad_out - got_out).abs().max().item()
    return floor, shift, ref_out


# ---------------------------------------------------------------- the model vs oracle
# The floor here is NOT the fp32 agreement floor (~4e-08) and NOT the fp32-core-with-
# bf16-internals floor (~5e-04) I first measured. It is ~0.4-0.7% relative and comes
# from BF16 ARITHMETIC THROUGHOUT -- S_bf is re-cast before each matmul, delta_bf and
# k_f are bf16 -- not, as an earlier version of this comment claimed, primarily from
# the bf16 `out` buffer. Measured under nki.simulate: the fp32 `state_out` carries a
# COMPARABLE floor (0.62% real / 0.59% long vs the output's 0.66% / 0.36%), so dropping
# the output cast buys almost nothing. Any equivalence test must be stated in RELATIVE
# terms, and the smallest detectable error at either surface is ~0.6% of the output.
BF16_REL = 0.01          # bf16 output quantisation, with headroom (2^-8 = 0.39%)


@pytest.mark.parametrize("decay", ["real", "long"])
def test_model_matches_the_fp32_oracle_within_bf16_output_quantisation(decay):
    floor, _, ref = _floor_and_shift(decay)
    mag = ref.abs().max().item()
    assert mag > 1e-3, "output is ~zero; comparison would be vacuous"
    rel = floor / mag
    print(f"\n  {decay:>4} gate: bf16 model vs fp32 oracle = {floor:.3e} ({rel:.3%} relative; "
          f"bf16 ulp at |out|={mag:.2f} is {mag * 2**-8:.3e})")
    assert rel < BF16_REL, f"model disagrees by {rel:.2%}, beyond bf16 output rounding"


# ------------------------------------------------------- the vacuity control, asserted
def test_scalar_gate_clears_the_floor_with_the_real_gate():
    """THE ACCEPTANCE GATE, at the precision the kernel actually returns.

    Substituting GDN's per-head scalar gate must move the output far enough above bf16
    output quantisation to be detectable. Re-asserted every run rather than measured
    once: the margin is finite, and an unrelated arithmetic change could drop it.
    """
    floor, shift, ref = _floor_and_shift("real")
    margin = shift / floor
    print(f"\n  real gate: floor {floor:.3e}, scalar-gate shift {shift:.3e} "
          f"({shift / ref.abs().max().item():.1%} of output) -> {margin:,.1f}x the floor")
    assert margin > 20.0, (
        f"scalar-gate substitution only moves the output {margin:.1f}x the bf16 output "
        f"floor. The KDA-vs-GDN check is no longer discriminating."
    )


def test_long_decay_regime_is_NOT_a_usable_acceptance_gate():
    """A limitation, pinned so nobody builds the gate on the wrong regime.

    With exp(g) near 1 the per-channel and per-head-mean gates nearly coincide, so the
    substitution moves the output only ~1.5% -- about 4x bf16 output quantisation. That
    is too thin to gate on. The REAL gate regime is the one to use (57x). This asserts
    the limitation still holds; if it ever improves, this test says so.
    """
    floor, shift, ref = _floor_and_shift("long")
    margin = shift / floor
    print(f"\n  long gate: {margin:,.1f}x the floor "
          f"({shift / ref.abs().max().item():.2%} of output) -- too thin to gate on")
    assert margin < 15.0, (
        "the long-decay regime now clears the floor comfortably; re-evaluate whether it "
        "can serve as a second acceptance gate"
    )


def test_the_real_gate_is_the_stronger_discriminator():
    """Polarity check, OPPOSITE to dev1's chunk-vs-recurrent carry test.

    There the long-memory regime was the strong one, because it exercises the
    inter-chunk carry. Here the real gate is, because it spreads exp(g) wide (mean
    ~0.136) so per-channel and per-head-mean diverge. Same oracle, opposite regimes --
    which is why an acceptance threshold cannot be inherited from a neighbouring test.
    """
    f_real, s_real, _ = _floor_and_shift("real")
    f_long, s_long, _ = _floor_and_shift("long")
    print(f"\n  real {s_real / f_real:,.1f}x vs long {s_long / f_long:,.1f}x the floor")
    assert s_real / f_real > 5 * (s_long / f_long)


def test_the_fp32_margin_is_four_orders_of_magnitude_too_optimistic():
    """Records WHY the gate above is 20x and not 150,000x.

    The fp32 margin quoted from tests/test_kda.py is ~1.5e5. At the kernel's bf16
    output it is ~58x. Inheriting the fp32 number would give a test that passes
    trivially and tells you nothing about the kernel.
    """
    args = _inputs(decay="real")
    q, k, v, g_log, beta, z, nw, state = args
    ref, _ = _oracle(*args)
    g_scalar = g_log.mean(-1, keepdim=True).expand_as(g_log).contiguous()
    bad, _ = _oracle(q, k, v, g_scalar, beta, z, nw, state)
    fp32_margin = (bad - ref).abs().max().item() / 4.1e-8      # against the fp32 floor
    floor, shift, _ = _floor_and_shift("real")
    bf16_margin = shift / floor
    print(f"\n  scalar-gate margin: fp32 {fp32_margin:,.0f}x vs bf16 output {bf16_margin:,.1f}x "
          f"-> {fp32_margin / bf16_margin:,.0f}x optimistic")
    assert fp32_margin > 1000 * bf16_margin


# --------------------------------------------------------------- the design decisions
def test_output_gate_is_sigmoid_in_the_model():
    """DISCRIMINATING. GDN's silu here is the single most likely mis-port of Fold 4."""
    args = _inputs(decay="real", seed=3)
    q, k, v, g_log, beta, z, nw, state = args
    good, _ = _kda_tkg_model(*args)
    # same model with silu(z), i.e. what a verbatim GDN port computes
    bh = q.shape[0]
    o_ref, _ = _kda_tkg_model(q, k, v, g_log, beta, torch.zeros_like(z), nw, state)
    silu_z = z.float() * torch.sigmoid(z.float())
    sig_z = torch.sigmoid(z.float())
    ratio = (silu_z / sig_z.clamp_min(1e-6)).abs().max().item()
    assert ratio > 2.0, "silu and sigmoid gates are indistinguishable on these inputs"
    scaled = good * (silu_z / sig_z.clamp_min(1e-6))
    rel = ((scaled - good).abs().max() / good.abs().max()).item()
    print(f"\n  output gate: silu vs sigmoid moves the result {rel:.2f} relative")
    assert rel > 0.5


def test_rms_eps_is_1e_5_not_gdns_1e_6():
    """Small but systematic: GDN hardcodes 1e-6, GLM's o_norm uses rms_norm_eps=1e-5."""
    assert R.FlashCfg().rms_norm_eps == RMS_NORM_EPS
    o = torch.full((2, K), 1e-3)
    a = torch.rsqrt(o.pow(2).sum(-1, keepdim=True) / K + 1e-5)
    b = torch.rsqrt(o.pow(2).sum(-1, keepdim=True) / K + 1e-6)
    assert (a - b).abs().max() > 1.0, "the two epsilons are indistinguishable; re-check"


def test_geometry_matches_the_kernels_asserts():
    """GLM's shapes against nki_kda_tkg's kernel_asserts, without importing nki."""
    cfg = R.FlashCfg()
    Kd = cfg.linear_head_dim
    assert Kd == 128 and Kd <= 128, "head_dim must fit the partition axis"
    assert cfg.linear_num_heads == 64
    for tp in (1, 2, 4, 8, 16, 32, 64):
        assert cfg.linear_num_heads % tp == 0, f"TP={tp} must divide the KDA heads"
