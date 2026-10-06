#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Acceptance harness for both KDA NKI kernels under ``nki.simulate``.

    python3 simulate_kernels.py [tkg|cte|chain|all]

**Needs the NKI toolchain, so it does NOT run on a laptop** -- see ``../README.md``
for the box invocation. Exits non-zero on failure, so it is usable as a gate rather
than only a printout. This replaces five ad-hoc scripts that lived only in ``~/kda/``
on one machine: the two kernels that took the most work to get right were validated
by code nobody else could reach.

The kernels themselves live in ``vllm_neuron/model/glm5_next/`` (they are production
code); this harness loads them **by path** so that neither it nor they require vLLM.

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


def _load_kernels():
    """Load the kernels BY PATH, not as ``vllm_neuron.model.glm5_next.*``.

    The kernels live in the plugin (production code); the oracle lives here
    (reference material). Importing them as a package would execute
    ``vllm_neuron/__init__.py``, which pulls in vLLM — and this harness must run on a
    box that has the Neuron toolchain but not necessarily vLLM. Loading by path keeps
    that property; ``vllm_neuron/model/glm5_next/__init__.py`` is deliberately empty
    for the same reason.

    Override with ``GLM5NEXT_KERNELS=<dir>`` if the tree is laid out differently, e.g.
    when only part of the repo has been copied to a machine.
    """
    import importlib.util
    import os

    env = os.environ.get("GLM5NEXT_KERNELS")
    cand = (pathlib.Path(env) if env
            else _HERE.parents[3] / "vllm_neuron" / "model" / "glm5_next")
    mods = {}
    for name in ("nki_kda_tkg", "nki_kda_cte"):
        f = cand / f"{name}.py"
        if not f.is_file():
            raise SystemExit(
                f"kernel not found: {f}\n"
                f"Copy the repo subtree, or set GLM5NEXT_KERNELS to the directory "
                f"holding nki_kda_tkg.py and nki_kda_cte.py."
            )
        spec = importlib.util.spec_from_file_location(name, f)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        mods[name] = m
    return mods


_K = _load_kernels()
kda_tkg, RMS_NORM_EPS = _K["nki_kda_tkg"].kda_tkg, _K["nki_kda_tkg"].RMS_NORM_EPS
kda_cte = _K["nki_kda_cte"].kda_cte

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


# ------------------------------------------------------- the handoff between them
# Each kernel is checked against the oracle INDEPENDENTLY above. That leaves the
# composition untested, and a real serving path is prefill-then-decode: cte writes a
# final state, tkg consumes it as its initial state. If those layouts or semantics
# disagree, every check above still passes and the model produces garbage from the
# second token on.
#
# THE TWO KERNELS DO NOT SHARE AN INPUT CONTRACT, which is the trap a caller falls
# into by treating them uniformly:
#
#   | input | kda_cte        | kda_tkg            |
#   |-------|----------------|--------------------|
#   | input | kda_cte           | kda_tkg                | uniform caller?      |
#   | q, k  | ALREADY l2-normed | RAW (normed in-kernel) | harmless — idempotent |
#   | beta  | POST-sigmoid      | RAW, pre-sigmoid       | **WRONG** — 20% shift |
#   | gate  | per-channel log   | same                   | fine                  |
#
# Only beta bites. l2norm(l2norm(x)) == l2norm(x) to 5e-7, but sigmoid(sigmoid(b))
# compresses 0.48-0.70 into 0.62-0.67, so beta quietly loses its dynamic range.
#
# The state itself is [BH, D, D] fp32 from both, so it hands over directly.

def run_chain():
    print("\nhandoff: kda_cte state -> kda_tkg")
    BH, S = 2, 64
    r = lambda *sh, o=0: _rand(*sh, seed=100 + o)
    q, k, v = r(BH, S + 1, K, o=0), r(BH, S + 1, K, o=1), r(BH, S + 1, V, o=2)
    b_raw = r(BH, S + 1, o=3)                       # RAW; cte wants sigmoid, tkg does not
    gate = -5.0 * torch.sigmoid(r(BH, S + 1, K, o=4))
    z, nw = r(BH, V, o=5), 1.0 + 0.2 * r(V, o=6)

    # prefill the first S tokens, each kernel fed ITS OWN contract
    cte_out, cte_state = _cte_run(R.l2norm(q[:, :S]), R.l2norm(k[:, :S]), v[:, :S],
                                  torch.sigmoid(b_raw[:, :S]), gate[:, :S])
    # decode token S using cte's state, with RAW q/k and RAW b
    tkg_out, tkg_state = _tkg_run(q[:, S], k[:, S], v[:, S], b_raw[:, S], gate[:, S],
                                  z, nw, cte_state)

    # oracle: chunk over S, then one recurrent step, then the gated norm
    o_chunk, o_state = R.chunk_kda(q[:, :S, None], k[:, :S, None], v[:, :S, None],
                                   gate[:, :S, None], torch.sigmoid(b_raw[:, :S, None]),
                                   torch.zeros(BH, 1, K, V), chunk=64)
    core, o_state2 = R.recurrent_kda(q[:, S:S + 1, None], k[:, S:S + 1, None],
                                     v[:, S:S + 1, None], gate[:, S:S + 1, None],
                                     torch.sigmoid(b_raw[:, S:S + 1, None]), o_state)
    n = R.RMSNormGated(V, RMS_NORM_EPS)
    with torch.no_grad():
        n.weight.copy_(nw)
        o_out = n(core[:, 0, 0], z)

    ds = (cte_state - o_state[:, 0]).abs().max().item() / o_state.abs().max().item()
    check("cte state matches the oracle's chunk state", ds < 0.02, f"{ds:.3%}")
    d2 = (tkg_state - o_state2[:, 0]).abs().max().item() / o_state2.abs().max().item()
    check("tkg state after consuming cte's", d2 < 0.02, f"{d2:.3%}")
    do = (tkg_out - o_out).abs().max().item() / o_out.abs().max().item()
    check("chained output matches the oracle", do < 0.02, f"{do:.3%}")

    # non-vacuity: a corrupted handoff must be visible. Transposing [K,V] is the
    # layout error this check exists to catch, and it is symmetric-shaped so a shape
    # check alone would not see it.
    bad_out, _ = _tkg_run(q[:, S], k[:, S], v[:, S], b_raw[:, S], gate[:, S], z, nw,
                          cte_state.transpose(-1, -2).contiguous())
    shift = (bad_out - tkg_out).abs().max().item() / tkg_out.abs().max().item()
    check("a transposed state handoff is caught", shift > 0.05, f"{shift:.1%} shift")

    # The two contract asymmetries are NOT equally dangerous, which is the useful
    # half of this check and the opposite of what I first asserted.
    #
    #   q/k: l2norm is IDEMPOTENT (double-norming differs by 5e-7), so a caller that
    #        uniformly l2-norms both kernels' inputs is fine.
    #   beta: sigmoid is NOT. Double-sigmoid shifts 19.6% and compresses the range
    #        from 0.48-0.70 to 0.62-0.67, so beta silently loses its dynamic range.
    #
    # So the trap is beta alone. Both are asserted, in the directions they actually go.
    pre_out, _ = _tkg_run(R.l2norm(q[:, S]), R.l2norm(k[:, S]), v[:, S], b_raw[:, S],
                          gate[:, S], z, nw, cte_state)
    d3 = (pre_out - tkg_out).abs().max().item() / tkg_out.abs().max().item()
    check("pre-normed q/k is harmless (l2norm is idempotent)", d3 < 0.02, f"{d3:.1%} shift")
    sig_out, _ = _tkg_run(q[:, S], k[:, S], v[:, S], torch.sigmoid(b_raw[:, S]),
                          gate[:, S], z, nw, cte_state)
    d4 = (sig_out - tkg_out).abs().max().item() / tkg_out.abs().max().item()
    check("but pre-sigmoided beta is NOT (sigmoid is not idempotent)", d4 > 0.02,
          f"{d4:.1%} shift")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("tkg", "all"):
        run_tkg()
    if which in ("cte", "all"):
        run_cte()
    if which in ("chain", "all"):
        run_chain()
    print(f"\n{'FAILED: ' + ', '.join(FAILURES) if FAILURES else 'all checks passed'}")
    sys.exit(1 if FAILURES else 0)
