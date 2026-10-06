# SPDX-License-Identifier: Apache-2.0
"""The routed experts' kernel layout against ``NF.moe_cte``'s own CPU fallback.

Run: python3 -m pytest personal_reference/glm5_next/tests/test_plugin_moe_layout.py -v -s

``Glm5NextMoE`` stores experts as ``[E_l, D, 2, I]`` / ``[E_l, I, D]`` and hands them,
with the router's ``[T, E_l]`` affinities flattened to ``[T*E_l, 1]``, to ``NF.moe_cte``
(prefill) and ``NF.moe_tkg`` (decode). ``moe_cte.py`` imports nki at module level, so
it cannot be imported here; instead its ``_torch_moe_impl`` and ``_apply_activation``
are extracted FROM THE FILE with ``ast`` and executed against stub enums. That is the
plugin's real fallback code, not a copy that could drift, and it fixes the contract the
NKI kernel implements: gate at ``[..., 0, :]``, up at ``[..., 1, :]``, affinity
``[t * E + e]``, clamps before the activation, POST_SCALE weighting.

The dense reference must agree with it exactly, including with the clamp engaged, and
must disagree when the gate/up halves are swapped -- the layout error a shape check
cannot see.
"""
from __future__ import annotations

import ast
import enum
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from glm5_next.tests import plugin_harness as H  # noqa: E402

M = H.import_plugin("vllm_neuron.model.glm5_next.model")
_MOE_CTE = H.ROOT / "vllm_neuron" / "functional" / "moe" / "moe_cte.py"


class ActFnType(enum.Enum):
    SiLU = "silu"
    GELU = "gelu"
    Swish = "swish"


class ExpertAffinityScaleMode(enum.Enum):
    POST_SCALE = "post"
    PRE_SCALE = "pre"
    PRE_SCALE_DELAYED = "pre_delayed"


def _plugin_fallback():
    tree = ast.parse(_MOE_CTE.read_text())
    wanted = {"_torch_moe_impl", "_apply_activation"}
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {f.name for f in funcs} == wanted, "moe_cte.py no longer defines the fallback"
    ns = {"torch": torch, "Tensor": torch.Tensor, "Optional": __import__("typing").Optional,
          "ActFnType": ActFnType, "ExpertAffinityScaleMode": ExpertAffinityScaleMode}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(_MOE_CTE), "exec"), ns)
    return ns["_torch_moe_impl"]


FALLBACK = _plugin_fallback()


def _moe(seed=0, E=8, D=64, I=24, top_k=2, gain=6.0):
    torch.manual_seed(seed)
    moe = M.Glm5NextMoE(D, I, E, top_k, 1, 2.5, True, 10.0, M.ExpertLayout(), world=1)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        moe.gate.weight.copy_(torch.randn(E, D, generator=g) / D ** 0.5)
        moe.gate.e_score_correction_bias.copy_(0.1 * torch.randn(E, generator=g))
        moe.gate_up_proj.copy_(torch.randn(moe.gate_up_proj.shape, generator=g) * gain / D ** 0.5)
        moe.down_proj.copy_(torch.randn(moe.down_proj.shape, generator=g) / I ** 0.5)
    return moe


def _nf(moe, x, local, gate_up=None):
    return FALLBACK(
        hidden_states=x, expert_affinities=local.reshape(-1, 1),
        gate_up_proj_weight=moe.gate_up_proj if gate_up is None else gate_up,
        down_proj_weight=moe.down_proj, activation_function=ActFnType.SiLU,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        **moe._nf_clamps())


def test_dense_reference_equals_the_plugins_moe_cte_fallback():
    moe = _moe()
    x = torch.randn(37, 64, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        routing, idx = moe.gate(x)
        local = routing.narrow(1, moe.e0, moe.E_local)
        dense = moe._routed_dense(x, local)
        nf = _nf(moe, x, local)
        gu = torch.einsum("td,edgi->tegi", x, moe.gate_up_proj)
    clamped = ((gu[:, :, 0] > 10) | (gu[:, :, 1].abs() > 10)).float().mean().item()
    print(f"\n  clamp engaged on {100 * clamped:.1f}% of (token, expert, channel)")
    assert clamped > 0.02, "the clamp never engages; a clamp mismatch would pass"
    assert (routing > 0).sum(-1).eq(2).all()
    torch.testing.assert_close(dense, nf, rtol=1e-5, atol=1e-5)


def test_swapping_gate_and_up_is_caught():
    """Gate and up have the same shape, so only the values can tell them apart."""
    moe = _moe()
    x = torch.randn(37, 64, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        local = moe.gate(x)[0]
        swapped = moe.gate_up_proj.flip(2)
        err = (_nf(moe, x, local, gate_up=swapped) - moe._routed_dense(x, local)).abs().max()
    assert err > 1e-2


def test_the_router_bias_selects_but_does_not_weight():
    """noaux_tc: ``e_score_correction_bias`` changes WHICH experts run, never their weight.
    The weights are the unbiased sigmoid scores, normalised over the top-k, x 2.5."""
    moe = _moe()
    x = torch.randn(64, 64, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        routing, idx = moe.gate(x)
        scores = torch.sigmoid(x @ moe.gate.weight.T)
        w = scores.gather(1, idx)
        torch.testing.assert_close(routing.gather(1, idx), 2.5 * w / w.sum(-1, keepdim=True))
        moe.gate.e_score_correction_bias.mul_(50)               # dominate the choice
        routing2, idx2 = moe.gate(x)
    assert not torch.equal(idx.sort(-1).values, idx2.sort(-1).values), "bias did not select"
    w2 = scores.gather(1, idx2)
    torch.testing.assert_close(routing2.gather(1, idx2), 2.5 * w2 / w2.sum(-1, keepdim=True))
