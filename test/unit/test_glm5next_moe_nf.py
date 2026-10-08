# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash routed experts through the plugin's NF MoE ops, against the dense path.

Run where nki and vLLM are installed (the trn2 host; CPU mode is enough):

    VLLM_NEURON_CPU_MODE=1 python -m pytest test/unit/test_glm5next_moe_nf.py -v -s

``VLLM_NEURON_GLM5NEXT_MOE=nf`` routes ``Glm5NextMoE`` through
``NF.build_blockwise_mapping`` + ``NF.moe_cte``; off device both run their torch
fallbacks. A pass could also mean the NF branch was never taken -- the dispatch falls
back to dense silently where the plugin runtime is absent -- so the ops are wrapped and
their calls counted: the test requires that they ran AND that the result equals the
dense reference, with the SwiGLU clamp engaged.
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("nki", reason="NF MoE ops import nki at module level")

from vllm_neuron import functional as NF  # noqa: E402
from vllm_neuron.model.glm5_next.model import ExpertLayout, Glm5NextMoE  # noqa: E402


def _moe(E=8, D=128, I=32, top_k=2, gain=6.0, seed=0):
    moe = Glm5NextMoE(D, I, E, top_k, 1, 2.5, True, 10.0, ExpertLayout(), world=1)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in moe.parameters():
            p.copy_(torch.randn(p.shape, generator=g) / p.shape[-1] ** 0.5)
        moe.gate_up_proj.mul_(gain * (moe.gate_up_proj.shape[-1] / D) ** 0.5)
    return moe


@pytest.mark.parametrize("T", [7, 64, 300])
def test_nf_prefill_path_runs_and_matches_dense(monkeypatch, T):
    calls = {"map": 0, "cte": 0}
    real_map, real_cte = NF.build_blockwise_mapping, NF.moe_cte

    def count_map(*a, **k):
        calls["map"] += 1
        return real_map(*a, **k)

    def count_cte(*a, **k):
        calls["cte"] += 1
        return real_cte(*a, **k)

    monkeypatch.setattr(NF, "build_blockwise_mapping", count_map)
    monkeypatch.setattr(NF, "moe_cte", count_cte)
    moe = _moe()
    x = torch.randn(T, 128, generator=torch.Generator().manual_seed(T))
    with torch.no_grad():
        monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_MOE", "dense")
        dense = moe(x)
        monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_MOE", "nf")
        nf = moe(x, is_decode=False, real=torch.ones(T, dtype=torch.bool))
        gu = torch.einsum("td,edgi->tegi", x, moe.gate_up_proj)
    assert calls == {"map": 1, "cte": 1}, f"NF ops not taken: {calls}"
    clamped = ((gu[:, :, 0] > 10) | (gu[:, :, 1].abs() > 10)).float().mean().item()
    assert clamped > 0.02
    err = ((nf - dense).abs().max() / dense.abs().max()).item()
    print(f"\n  T={T}: NF moe_cte (torch fallback) vs dense, max rel {err:.2e}; "
          f"clamp on {100 * clamped:.1f}%")
    torch.testing.assert_close(nf, dense, rtol=1e-4, atol=1e-4)
