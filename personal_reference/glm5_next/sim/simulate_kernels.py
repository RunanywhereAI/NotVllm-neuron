#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Acceptance harness for both KDA NKI kernels under ``nki.simulate``.

    python3 simulate_kernels.py [tkg|cte|all]

**Needs the NKI toolchain, so it does NOT run on a laptop** -- see ``../README.md``
for the box invocation. Exits non-zero on failure, so it is usable as a gate rather
than only a printout. This replaces five ad-hoc scripts that lived only in ``~/kda/``
on one machine: the two kernels that took the most work to get right were validated
by code nobody else could reach.

What it checks, and why each one is here rather than being a nice-to-have:

* **vs the fp32 oracle.** ``recurrent_kda`` + ``RMSNormGated`` for decode,
  ``chunk_kda`` for prefill. The tolerance is **relative** and ~1%, because the floor
  is bf16 arithmetic throughout the kernel -- not, as I first concluded, the bf16
  output cast. (``state_out`` is fp32 and carries a comparable floor, which is what
  disproved that.)
* **The scalar-gate trap.** Substituting GDN's per-head scalar gate must move the
  output far above that floor. This is the single most likely way to get KDA subtly
  wrong, and at fp32 it looks 260,000x easier to catch than it actually is.

  The ratio is computed from **mean-abs**, not max-abs, and that is not cosmetic. A
  max-abs ratio divides one extreme-value statistic by another: measured across input
  draws it spans **2.2-4.4x** and *degrades with head count* (floor is a max over more
  heads), reaching a minimum of **19.8 at BH=32** -- under the 20x threshold this
  harness first used. The mean-abs ratio spans only **1.2-1.6x** with no head-count
  trend and a minimum of 75.5 over the same draws. Quoting a single max-abs number as
  though it were a property of the kernel is what produced two different "measurements"
  (54.4x and 88.6x) of the same thing.
* **The converse control.** With a genuinely scalar gate the kernel must still match
  the reference. A kernel that mangled the channel axis some other way would pass the
  trap alone.
* **LNC1 == LNC2, byte-identical**, including an odd head count so the
  ``BH_local = BH - h_start`` remainder branch is exercised.

Simulation is not silicon. It does not exercise the real DMA engine, PSUM bank
allocation or SBUF capacity, so the batch-vs-TP envelope in ``nki_kda_tkg.py`` remains
arithmetic rather than measurement.
"""
from __future__ import annotations

import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))          # glm5_next/
sys.path.insert(0, str(_HERE.parents[2]))          # personal_reference/

import numpy as np  # noqa: E402
import torch  # noqa: E402

import nki  # noqa: E402
import nki.language as nl  # noqa: E402
import neuron_dtypes as dt  # noqa: E402

from glm5_next import reference as R  # noqa: E402
from glm5_next.nki_kda_tkg import kda_tkg, RMS_NORM_EPS  # noqa: E402
from glm5_next.nki_kda_cte import kda_cte  # noqa: E402

K = V = 128
FAILURES: list[str] = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def _rand(*shape, seed):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


# ----------------------------------------------------------------------- decode (tkg)
def _tkg_inputs(BH, seed=0, decay="real"):
    r = lambda *s, o=0: _rand(*s, seed=seed + o)
    g_log = (-5.0 * torch.sigmoid(r(BH, K, o=4)) if decay == "real"
             else torch.empty(BH, K).uniform_(0.95, 1.0,
                  generator=torch.Generator().manual_seed(seed + 4)).log())
    return (r(BH, K, o=0), r(BH, K, o=1), r(BH, V, o=2), r(BH, o=3), g_log,
            r(BH, V, o=5), 1.0 + 0.2 * r(V, o=6), r(BH, K, V, o=7) * 0.1)


def _tkg_run(q, k, v, b, g_log, z, nw, state, lnc=1):
    kern = kda_tkg[lnc] if lnc > 1 else kda_tkg
    o, s = nki.simulate(kern)(
        q=dt.static_cast(q.numpy(), nl.bfloat16), k=dt.static_cast(k.numpy(), nl.bfloat16),
        v=dt.static_cast(v.numpy(), nl.bfloat16), b=b.numpy().astype(np.float32),
        g_log=g_log.numpy().astype(np.float32), z=dt.static_cast(z.numpy(), nl.bfloat16),
        norm_weight=nw.numpy().astype(np.float32), state_in=state.numpy().astype(np.float32))
    return (torch.from_numpy(np.asarray(o, dtype=np.float32)),
            torch.from_numpy(np.asarray(s, dtype=np.float32)))


def _tkg_oracle(q, k, v, b, g_log, z, nw, state):
    core, S = R.recurrent_kda(q[None, None], k[None, None], v[None, None],
                              g_log[None, None], torch.sigmoid(b)[None, None], state[None])
    n = R.RMSNormGated(V, RMS_NORM_EPS)
    with torch.no_grad():
        n.weight.copy_(nw)
        return n(core[0, 0], z), S[0]


def run_tkg():
    print("\nkda_tkg (decode)")
    a = _tkg_inputs(4)
    got, _ = _tkg_run(*a)
    ref, _ = _tkg_oracle(*a)
    mag = ref.abs().max().item()
    floor = (got - ref).abs().max().item()
    check("vs fp32 oracle", floor / mag < 0.01, f"{floor / mag:.3%} relative")

    a_bad = list(a)
    a_bad[4] = a[4].mean(-1, keepdim=True).expand_as(a[4]).contiguous()
    ratio = ((_tkg_run(*a_bad)[0] - got).abs().mean().item()
             / (got - ref).abs().mean().item())          # mean-abs: see the docstring
    check("scalar-gate trap (real gate)", ratio > 20, f"{ratio:,.1f}x the floor (mean-abs)")

    for BH in (8, 6):                      # 6 exercises the LNC remainder branch
        a = _tkg_inputs(BH, seed=BH)
        o1, s1 = _tkg_run(*a, lnc=1)
        o2, s2 = _tkg_run(*a, lnc=2)
        check(f"LNC1 == LNC2 byte-identical (BH={BH})",
              torch.equal(o1, o2) and torch.equal(s1, s2))


# ---------------------------------------------------------------------- prefill (cte)
def _cte_run(qn, kn, v, beta, gate):
    o, s = nki.simulate(kda_cte)(
        dt.static_cast(qn.numpy(), nl.bfloat16), dt.static_cast(kn.numpy(), nl.bfloat16),
        dt.static_cast(v.numpy(), nl.bfloat16), beta.numpy().astype(np.float32),
        gate.numpy().astype(np.float32), K ** -0.5)
    return (torch.from_numpy(np.asarray(o, dtype=np.float32)),
            torch.from_numpy(np.asarray(s, dtype=np.float32)))


def _cte_case(BH, S, seed=0, gate=None):
    q, k, v = _rand(BH, S, K, seed=seed), _rand(BH, S, K, seed=seed + 1), _rand(BH, S, K, seed=seed + 2)
    beta = torch.sigmoid(_rand(BH, S, seed=seed + 3))
    if gate is None:
        gate = -5.0 * torch.sigmoid(_rand(BH, S, K, seed=seed + 4))
    got, _ = _cte_run(R.l2norm(q), R.l2norm(k), v, beta, gate)
    ref, _ = R.chunk_kda(q[:, :, None], k[:, :, None], v[:, :, None], gate[:, :, None],
                         beta[:, :, None], torch.zeros(BH, 1, K, K), chunk=64)
    return got, ref[:, :, 0], gate


def run_cte():
    print("\nkda_cte (prefill)")
    for BH, S in ((1, 64), (1, 128), (1, 192), (2, 128)):
        got, ref, _ = _cte_case(BH, S, seed=S)
        rel = (got - ref).abs().max().item() / ref.abs().max().item()
        check(f"vs chunk_kda (BH={BH}, S={S})", rel < 0.015, f"{rel:.3%} relative")

    # the trap, and its converse
    got_pc, ref_pc, gate_pc = _cte_case(2, 128, seed=11)
    floor = (got_pc - ref_pc).abs().mean().item()
    gate_sc = gate_pc.mean(-1, keepdim=True).expand_as(gate_pc).contiguous()
    got_sc, ref_sc, _ = _cte_case(2, 128, seed=11, gate=gate_sc)
    ratio = (got_sc - got_pc).abs().mean().item() / floor
    check("scalar-gate trap", ratio > 10, f"{ratio:,.1f}x the floor (mean-abs)")
    rel_sc = (got_sc - ref_sc).abs().max().item() / ref_sc.abs().max().item()
    check("still correct under a genuinely scalar gate", rel_sc < 0.015, f"{rel_sc:.3%}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("tkg", "all"):
        run_tkg()
    if which in ("cte", "all"):
        run_cte()
    print(f"\n{'FAILED: ' + ', '.join(FAILURES) if FAILURES else 'all checks passed'}")
    sys.exit(1 if FAILURES else 0)
