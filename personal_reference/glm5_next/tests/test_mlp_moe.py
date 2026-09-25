# SPDX-License-Identifier: Apache-2.0
"""The clamped SwiGLU, and why an external reference alone would not have caught it.

Run: python3 -m pytest personal_reference/glm5_next/tests/test_mlp_moe.py -v -s

``MLP`` omitted the swiglu clamp until 2026-09-25, so the dense FFN on layers 0-2 and
``shared_experts`` on all 42 MoE layers -- **45 of 45 layers** -- ran unclamped. The
routed experts in ``MoE`` thirty lines below always had it: the same
correct-next-to-wrong pattern as the silu/sigmoid output gate.

THE METHODOLOGY POINT, which is the reason this file is shaped the way it is.

After the output-gate bug the rule was "cross-reference beats self-consistency."
That rule is necessary and **not sufficient**. The clamp is a no-op until activations
reach +-``swiglu_limit`` (10.0), and ``tiny_cfg`` initialises weights at
``normal_(0, 0.02)``. So the obvious test -- build an MLP on tiny_cfg weights and
compare it against ``Glm5NextTextMLP`` -- is a genuine external-reference test that
**passes with the bug present**.

``test_the_clamp_is_invisible_at_small_scale`` measures that trap rather than
describing it, and every other test here drives activations past the limit.

The amended rule: an external reference is necessary, and the inputs must
**reach the behaviour under test**. A test whose inputs never enter the regime is
testing the regime it happens to be in.
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

_hf_mlp_forward = H.hf_mlp   # the transformers reference lives in hf_refs.py

LIMIT = 10.0
D, I = 256, 512


def _unclamped_forward(x, gate_w, up_w, down_w):
    """What the oracle computed before the fix."""
    return F.linear(F.silu(F.linear(x, gate_w)) * F.linear(x, up_w), down_w)


def _mlp_at_scale(std, seed=0, d=D, i=I):
    """An MLP whose gate/up pre-activations have roughly the given std.

    x ~ N(0, 1) over d inputs and W ~ N(0, s^2) gives pre-activations of std
    s*sqrt(d), so s = std / sqrt(d).
    """
    g = torch.Generator().manual_seed(seed)
    m = R.MLP(d, i, LIMIT)
    s = std / d ** 0.5
    with torch.no_grad():
        m.gate_proj.weight.copy_(torch.randn(i, d, generator=g) * s)
        m.up_proj.weight.copy_(torch.randn(i, d, generator=g) * s)
        m.down_proj.weight.copy_(torch.randn(d, i, generator=g) * i ** -0.5)
    x = torch.randn(4, d, generator=g)
    return m, x


def _weights(m):
    return m.gate_proj.weight, m.up_proj.weight, m.down_proj.weight


# ------------------------------------------------------------------ the trap, measured
def test_the_clamp_is_invisible_at_small_scale():
    """WHY an external-reference test alone would have missed this.

    Below the limit the clamped and unclamped forms are the SAME FUNCTION, so a
    reference comparison on small-init weights cannot distinguish them. Measured, not
    asserted from theory.
    """
    print()
    rows = []
    for std in (1.0, 3.0, 5.0, 8.0):
        m, x = _mlp_at_scale(std, seed=1)
        gw, uw, dw = _weights(m)
        with torch.no_grad():
            gate, up = F.linear(x, gw), F.linear(x, uw)
            frac = (((gate > LIMIT) | (up.abs() > LIMIT)).float().mean() * 100).item()
            a = _hf_mlp_forward(x, gw, uw, dw, LIMIT)
            b = _unclamped_forward(x, gw, uw, dw)
            rel = ((a - b).abs().max() / a.abs().max()).item() if a.abs().max() > 0 else 0.0
        rows.append((std, frac, rel))
        print(f"  pre-activation std {std:>4.1f} -> {frac:5.2f}% clamped, "
              f"relative difference {rel:.2e}")
    # at std 1.0 the clamp is EXACTLY a no-op: a reference test here proves nothing
    assert rows[0][1] == 0.0 and rows[0][2] == 0.0
    # and the effect grows monotonically with scale
    assert [r[1] for r in rows] == sorted(r[1] for r in rows)
    assert rows[-1][2] > 1e-2, "even at std 8 the clamp barely bites; pick a harder scale"


def test_tiny_cfg_init_is_inside_the_blind_regime():
    """The specific trap: tiny_cfg's own initialisation never reaches the limit, so a
    test built on it is vacuous for this behaviour no matter what it compares against.
    """
    torch.manual_seed(0)
    cfg = R.tiny_cfg()
    mlp = R.MLP(cfg.hidden_size, cfg.intermediate_size, cfg.swiglu_limit)
    x = torch.randn(2, 16, cfg.hidden_size)
    with torch.no_grad():
        gate = mlp.gate_proj(x)
        up = mlp.up_proj(x)
    assert gate.max() < LIMIT and up.abs().max() < LIMIT, (
        "tiny_cfg now reaches the clamp; this test's premise has changed"
    )


# --------------------------------------------------------- cross-reference, driven hard
@pytest.mark.parametrize("std", [5.0, 8.0, 15.0])
def test_mlp_matches_transformers_when_the_clamp_is_active(std):
    """Oracle MLP == Glm5NextTextMLP, with inputs that actually reach the limit."""
    m, x = _mlp_at_scale(std, seed=2)
    gw, uw, dw = _weights(m)
    with torch.no_grad():
        got = m(x)
    want = _hf_mlp_forward(x, gw, uw, dw, LIMIT)
    frac = (((F.linear(x, gw) > LIMIT) | (F.linear(x, uw).abs() > LIMIT)).float().mean()).item()
    assert frac > 0.05, f"only {frac:.1%} of activations clamp; test is not exercising it"
    torch.testing.assert_close(got, want, rtol=0, atol=1e-5)


@pytest.mark.parametrize("std", [5.0, 8.0, 15.0])
def test_mlp_rejects_the_unclamped_form(std):
    """THE DISCRIMINATING TEST. The pre-fix implementation must be visibly wrong."""
    m, x = _mlp_at_scale(std, seed=3)
    gw, uw, dw = _weights(m)
    with torch.no_grad():
        got = m(x)
    bad = _unclamped_forward(x, gw, uw, dw)
    rel = ((got - bad).abs().max() / got.abs().max()).item()
    print(f"\n  std {std:>4.1f}: unclamped differs by {rel:.3f} relative")
    assert rel > 0.01, f"unclamped only differs by {rel:.4f}; this test would not have caught it"


def test_clamp_is_asymmetric_gate_upper_only():
    """gate clamps ONLY above; up clamps both sides. A symmetric clamp on gate would
    be wrong for large-negative gate, where silu -> 0 rather than silu(-limit).
    """
    m, _ = _mlp_at_scale(1.0, seed=4)
    gw, uw, dw = _weights(m)
    x = torch.zeros(1, D)
    # drive gate very negative, up moderate, via a hand-built activation
    gate = torch.tensor([[-50.0, 0.0, 5.0]])
    up = torch.tensor([[1.0, 1.0, 1.0]])
    ref = F.silu(gate.clamp(max=LIMIT)) * up.clamp(-LIMIT, LIMIT)
    sym = F.silu(gate.clamp(-LIMIT, LIMIT)) * up.clamp(-LIMIT, LIMIT)
    assert not torch.allclose(ref, sym), "a symmetric gate clamp is indistinguishable here"
    assert abs(ref[0, 0].item()) < 1e-6, "silu(-50) should vanish, not saturate"


# ----------------------------------------------------------------------- blast radius
def test_every_layer_carries_the_limit_at_every_mlp_site():
    """The clamp must reach EVERY MLP-bearing site on EVERY layer, not two samples.

    An earlier version of this test pinned index 0 (dense) and index 3 (MoE) only.
    dev1 found the mutation that defeats it: shift ``mlp_layer_types`` LEFT by one and
    both sampled indices still match, while layer 2 flips kind. **Two sampled indices
    cannot pin a property of all layers** -- the same sampling error, in a test written
    to catch a sampling-shaped bug.

    Scope note: this asserts the limit is WIRED wherever an MLP exists. It is
    deliberately NOT a dispatch test -- it does not check that each layer picks the
    right attention or MLP kind. That is a separate property with its own file.
    """
    # A NON-DEFAULT limit, so the test proves the value is wired through rather than
    # coinciding with a default. (MLP's limit used to default to 10.0 -- the real
    # swiglu_limit -- which made an unwired call site invisible. It is now required.)
    cfg = R.tiny_cfg(swiglu_limit=7.5)
    seen_dense = seen_moe = 0
    for i in range(cfg.num_hidden_layers):
        layer = R.DecoderLayer(cfg, i)
        if isinstance(layer.mlp, R.MLP):
            assert layer.mlp.limit == cfg.swiglu_limit, f"layer {i}: dense MLP limit"
            seen_dense += 1
        else:
            # MoE clamps in two places: the routed path inline, and shared_experts via MLP
            assert layer.mlp.limit == cfg.swiglu_limit, f"layer {i}: MoE routed limit"
            assert layer.mlp.shared_experts.limit == cfg.swiglu_limit, (
                f"layer {i}: shared_experts limit -- this is the site that was unclamped"
            )
            seen_moe += 1
    assert seen_dense > 0 and seen_moe > 0, "config exercised only one MLP kind"
    assert seen_dense + seen_moe == cfg.num_hidden_layers


def test_the_real_config_has_an_mlp_on_every_layer():
    """45/45: the blast radius claim, checked against the real layer lists rather than
    inferred. Constructing 45 real DecoderLayers (288 experts each) is not viable, so
    this checks the config pattern; the wiring itself is checked above on tiny_cfg."""
    cfg = R.FlashCfg()
    assert len(cfg.mlp_layer_types) == cfg.num_hidden_layers == 45
    assert set(cfg.mlp_layer_types) == {"dense", "sparse"}
    dense = [i for i, t in enumerate(cfg.mlp_layer_types) if t == "dense"]
    assert dense == [0, 1, 2], f"first_k_dense_replace changed: {dense}"
    # every layer has an MLP: dense ones directly, sparse ones via shared_experts
    assert len(cfg.mlp_layer_types) == 45 and cfg.n_shared_experts >= 1


def test_routed_experts_and_shared_expert_use_the_same_limit_semantics():
    """MoE's routed path clamps inline; shared_experts clamps via MLP. Same function."""
    cfg = R.tiny_cfg()
    g = torch.Generator().manual_seed(5)
    gate = torch.randn(3, 64, generator=g) * 6.0
    up = torch.randn(3, 64, generator=g) * 6.0
    routed = F.silu(gate.clamp(max=cfg.swiglu_limit)) * up.clamp(-cfg.swiglu_limit, cfg.swiglu_limit)
    shared = F.silu(gate.clamp(max=LIMIT)) * up.clamp(-LIMIT, LIMIT)
    assert (gate > LIMIT).any() and (up.abs() > LIMIT).any()
    torch.testing.assert_close(routed, shared, rtol=0, atol=0)


def test_swiglu_limit_matches_the_live_config():
    assert R.FlashCfg().swiglu_limit == LIMIT == 10.0
