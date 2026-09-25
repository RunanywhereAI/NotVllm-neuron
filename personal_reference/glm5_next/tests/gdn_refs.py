# SPDX-License-Identifier: Apache-2.0
"""nkilib's own GDN torch reference, vendored for tests/test_kda.py.

Provenance: ``nkilib/experimental/gdn/gdn_cte_torch.py`` from the nki-library
checkout at ``/Users/aman/code/aws_infer/third_party/nki-library`` (synced
2026-09-07, commit ``92d11f6``). ``gdn_cte_torch_nki_ref`` is **verbatim**; only
the module docstring and the license header differ.

Vendored rather than imported for two reasons. The checkout lives at a
machine-specific path outside this repo, so a committed test that imported it
would not run anywhere else; and freezing it pins exactly what was verified.
Set ``NKILIB_SRC`` to the checkout's ``src/nkilib_src`` to additionally diff this
copy against the live file (that test is otherwise skipped).

Why this file exists at all: the KDA kernel is adapted from nkilib's GDN kernels
on the premise that **GDN is KDA with a per-head scalar gate**. That premise is
load-bearing and was inherited from a prior-art header whose framing has already
proved too narrow once (it missed silu-vs-sigmoid and eps 1e-6-vs-1e-5 in the
same kernel). So it gets tested rather than trusted.

Note the precision regime: this reference computes in **fp32 throughout** and does
not emulate the kernel's bf16 tile arithmetic -- unlike nkilib's
``attention_tkg_torch_ref``, which does. Comparing an fp32 ref against an fp32
oracle is therefore a structurally valid comparison; comparing either against the
real kernel is not, and needs the bf16 threshold instead.
"""
from __future__ import annotations

from typing import Optional

import torch


def gdn_cte_torch_nki_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    gate: Optional[torch.Tensor] = None,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token recurrence reference matching gdn_cte kernel semantics.

    Args:
        q, k, v: [B, S, D] float32
        beta: [B, S] float32, sigmoid-gated update strength
        gate: [B, S] float32 or None, log-decay per token (negative).
              If None, no decay (gate=0).
        scale: float, query scaling factor (typically 1/sqrt(D))

    Returns:
        out: [B, S, D] float32
        state: [B, D, D] float32 (final recurrent state)
    """
    B, S, D = q.shape
    device = q.device
    q = q.float() * scale
    k = k.float()
    v = v.float()
    beta = beta.float()

    state = torch.zeros(B, D, D, device=device, dtype=torch.float32)
    outs = []

    for t in range(S):
        q_t = q[:, t]
        k_t = k[:, t]
        v_t = v[:, t]
        beta_t = beta[:, t]

        if gate is not None:
            g_t = gate[:, t].float().exp()
            state = state * g_t[:, None, None]

        v_old = (state * k_t[:, :, None]).sum(1)
        delta = (v_t - v_old) * beta_t[:, None]
        state = state + k_t[:, :, None] * delta[:, None, :]

        o_t = (state * q_t[:, :, None]).sum(1)
        outs.append(o_t)

    out = torch.stack(outs, dim=1)
    return out, state
