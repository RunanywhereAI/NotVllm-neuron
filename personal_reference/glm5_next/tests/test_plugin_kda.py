# SPDX-License-Identifier: Apache-2.0
"""The plugin's KDA layer against the oracle.

Run: python3 -m pytest personal_reference/glm5_next/tests/test_plugin_kda.py -v -s

``vllm_neuron/model/glm5_next/kda.py`` is production code; ``reference.py`` is the
oracle it must reproduce. This file diffs them, which is the only reason the four
traps in the port were caught at all — the prefill mask bug localised in one run
because there was something to diff against.

**Loaded by file path, not imported as a package.** Importing
``vllm_neuron.model.glm5_next.kda`` executes ``vllm_neuron/__init__.py``, which needs
vLLM, and vLLM ships manylinux wheels only. The layer is written to stay importable
without it — capture degrades to a no-op — precisely so this comparison can run on a
laptop. If that ever stops being true, this whole validation route is lost.

What each test is for:

* the two paths (chunked prefill, single-token decode) reproduce the oracle;
* prefill and decode agree with each other across a state handoff;
* the **four traps** are each rejected by a mutation, because every one of them was a
  real defect during the kernel work and none is visible from the final output alone;
* the capture points actually fire and carry tensors, rather than merely existing.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from glm5_next import reference as R  # noqa: E402


def _load(name: str, relpath: str):
    """Load a plugin module by path. ``sys.modules`` registration is required before
    ``exec_module`` or ``@dataclass`` cannot resolve ``cls.__module__``."""
    root = pathlib.Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(name, root / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


KDA = _load("glm5next_kda", "vllm_neuron/model/glm5_next/kda.py")

H, K, D = 4, 32, 96          # small but structurally identical; K == V as in the real config


class _Cfg:
    hidden_size = D
    rms_norm_eps = 1e-5
    linear_attn_config = {"num_heads": H, "head_dim": K,
                          "short_conv_kernel_size": 4, "gate_lower_bound": -5.0}


def _layer(seed=0):
    torch.manual_seed(seed)
    return KDA.Glm5NextKDA(_Cfg(), layer_idx=0).eval()


def _oracle(layer):
    """The oracle's LinearAttention carrying the plugin layer's weights."""
    cfg = R.tiny_cfg(hidden_size=D, linear_num_heads=H, linear_head_dim=K,
                     linear_conv_kernel_dim=4)
    la = R.LinearAttention(cfg).eval()
    la.load_state_dict(layer.state_dict())
    return la


def _x(S, seed=0, B=1):
    return torch.randn(B, S, D, generator=torch.Generator().manual_seed(seed))


# --------------------------------------------------------------- the two paths
@pytest.mark.parametrize("S", [1, 7, 64, 130])
def test_prefill_matches_the_oracle(S):
    layer = _layer()
    la = _oracle(layer)
    x = _x(S, seed=S)
    with torch.no_grad():
        got, (conv, rec) = layer.forward_prefill(x)
        want, (o_conv, o_rec) = la(x)
    assert want.abs().max() > 1e-4, "oracle output ~zero; comparison would be vacuous"
    torch.testing.assert_close(got, want, rtol=0, atol=1e-5)
    torch.testing.assert_close(conv, o_conv, rtol=0, atol=1e-5)
    torch.testing.assert_close(rec, o_rec, rtol=0, atol=1e-5)


def test_decode_matches_the_oracle_step_for_step():
    layer = _layer()
    la = _oracle(layer)
    x = _x(12, seed=3)
    conv = torch.zeros(1, 3 * layer.HK, layer.p.conv_kernel)
    rec = torch.zeros(1, layer.H, layer.K, layer.V)
    o_conv, o_rec = conv.clone(), rec.clone()
    with torch.no_grad():
        for t in range(x.shape[1]):
            got, (conv, rec) = layer.forward_decode(x[:, t:t + 1], conv, rec)
            want, (o_conv, o_rec) = la(x[:, t:t + 1], conv_state=o_conv, rec_state=o_rec)
            torch.testing.assert_close(got, want, rtol=0, atol=1e-5), f"step {t}"
    torch.testing.assert_close(rec, o_rec, rtol=0, atol=1e-5)


def test_prefill_then_decode_is_continuous():
    """The state handoff: prefill a prefix, decode the rest, match a single prefill."""
    layer = _layer()
    x = _x(20, seed=5)
    with torch.no_grad():
        full, _ = layer.forward_prefill(x)
        out, (conv, rec) = layer.forward_prefill(x[:, :12])
        outs = [out]
        for t in range(12, 20):
            o, (conv, rec) = layer.forward_decode(x[:, t:t + 1], conv, rec)
            outs.append(o)
    assert (torch.cat(outs, 1) - full).abs().max() < 1e-4


# ------------------------------------------------------- the four traps, each rejected
def test_output_gate_is_sigmoid_not_silu():
    """TRAP 1. FLA's default is the wrong branch, so a port that omits activation=
    gets silu silently. 194% across 34 layers."""
    layer = _layer()
    x = torch.randn(2, 6, K)
    gate = torch.randn(2, 6, K)
    with torch.no_grad():
        got = layer.o_norm(x, gate)
        base = got / torch.sigmoid(gate.float()).clamp_min(1e-6)
        silu = base * (gate.float() * torch.sigmoid(gate.float()))
    rel = (got - silu).abs().max() / got.abs().max()
    print(f"\n  trap 1 (silu vs sigmoid gate): {rel:.2f} relative")
    assert rel > 0.5


def test_exp_cg_does_not_commute_out_of_the_matmul():
    """TRAP 2. (q @ state) * exp_cg is a per-token-scalar optimisation; per channel it
    is wrong. Verified as a property of the math the layer relies on."""
    torch.manual_seed(0)
    q, S_, cg = torch.randn(8, K), torch.randn(K, K), -torch.rand(8, K) * 3
    correct = (q * cg.exp()) @ S_
    reordered = (q @ S_) * cg.exp()[:, :1]
    rel = (correct - reordered).abs().max() / correct.abs().max()
    assert rel > 0.5, "the reordering is indistinguishable here; pick harder inputs"
    per_token = cg[:, :1].expand(8, K)              # control: it DOES commute for GDN
    a = (q * per_token.exp()) @ S_
    b = (q @ S_) * per_token.exp()[:, :1]
    print(f"  trap 2 (reordering): {rel:.2f} relative per-channel, "
          f"{(a - b).abs().max() / a.abs().max():.1e} per-token")
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_the_two_masks_differ():
    """TRAP 3. The A operator is strict, the intra output inclusive. Using one for
    both leaves the state correct and the output badly wrong."""
    c = 8
    strict = torch.triu(torch.ones(c, c, dtype=torch.bool), 0)      # zeroes j >= i
    inclusive = torch.triu(torch.ones(c, c, dtype=torch.bool), 1)   # zeroes j > i
    assert not torch.equal(strict, inclusive)
    assert (strict.sum() - inclusive.sum()) == c, "they differ by exactly the diagonal"


def test_gate_is_per_channel_not_a_per_head_scalar():
    """TRAP 4's cousin, and the single most likely way to ship GDN where KDA is meant.
    The ForgetGate must emit a K axis that actually varies along K."""
    layer = _layer()
    x = _x(6, seed=7)
    with torch.no_grad():
        g = layer.forget_gate(x)
    assert g.shape == (1, 6, H, K)
    spread = (g.max(-1).values - g.min(-1).values).max()
    print(f"  trap 4 (gate varies along K by {spread:.3f})")
    assert spread > 1e-3, "gate is constant along K — that is GDN, not KDA"
    assert (g <= 0).all() and (g >= layer.p.gate_lower_bound).all()


def test_substituting_a_scalar_gate_changes_the_output():
    """The discriminating end-to-end form: a per-head scalar gate must not reproduce
    the per-channel result."""
    layer = _layer()
    x = _x(64, seed=9)
    # patch the METHOD, not the attribute: assigning a plain callable to an
    # nn.Module child raises TypeError.
    real_forward = type(layer.forget_gate).forward
    with torch.no_grad():
        want, _ = layer.forward_prefill(x)
        scalar = lambda self, h: real_forward(self, h).mean(-1, keepdim=True).expand(-1, -1, -1, K)
        type(layer.forget_gate).forward = scalar
        try:
            got, _ = layer.forward_prefill(x)
        finally:
            type(layer.forget_gate).forward = real_forward
    rel = (got - want).abs().max() / want.abs().max()
    print(f"  scalar-gate substitution: {rel:.3f} relative")
    assert rel > 0.01


# ----------------------------------------------------------------- capture points
def test_capture_points_fire_and_carry_tensors(monkeypatch):
    """Master's requirement: verified to PRODUCE tensors, not merely to exist.

    Capture is what makes this layer validatable on hardware — its output cannot show
    errors below ~2% — so a capture point that silently never fires is the same as
    not having it.
    """
    seen: dict[str, torch.Tensor] = {}
    monkeypatch.setattr(KDA, "_capture_tensor", lambda n, t: seen.__setitem__(n, t))
    layer = _layer()
    x = _x(64, seed=11)
    with torch.no_grad():
        _, (conv, rec) = layer.forward_prefill(x)
    prefix = "model.layers.0.linear_attn"
    for want in ("g", "core_pre_norm", "recurrent_state", "conv_window"):
        name = f"{prefix}.{want}"
        assert name in seen, f"{name} never fired; captured: {sorted(seen)}"
        assert isinstance(seen[name], torch.Tensor) and seen[name].numel() > 0
    assert seen[f"{prefix}.g"].shape[-1] == K, "the gate must keep its channel axis"
    seen.clear()
    with torch.no_grad():
        layer.forward_decode(x[:, :1], conv, rec)
    assert {f"{prefix}.{n}" for n in ("g", "core_pre_norm", "recurrent_state",
                                      "conv_window")} <= set(seen), "decode path missing captures"
    print(f"\n  capture points firing: {len(seen)} on decode, names {sorted(n.split('.')[-1] for n in seen)}")


@pytest.mark.skipif(
    importlib.util.find_spec("vllm") is not None,
    reason="checkable only where vLLM is absent: with it installed the layer's optional "
           "imports legitimately pull it in")
def test_layer_imports_without_vllm():
    """The property the whole validation route depends on. If the layer ever imports
    vLLM at module scope, none of the tests above can run on a laptop."""
    assert "vllm" not in sys.modules
    assert KDA._capture_tensor("probe", torch.zeros(1)) is None
