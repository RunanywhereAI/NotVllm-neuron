#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""``cache_row_scatter_kernel`` under ``nki.simulate``, against ``index_copy_``.

    python3 simulate_row_scatter.py          # needs the NKI toolchain (the box)

The kernel is the folded MLA page's only write path on device (latent rows, pool keys
and tail rows). Its torch fallback is what every CPU test exercises, so the kernel is
checked here against exactly that fallback, at both row widths the page is written at,
across more than one 128-row tile, with the dead-row duplicates to the sink page the
model really sends, and at LNC1 and LNC2 -- where tiles alternate between the two
programs, so a wrong split would drop or double-write rows.

Loaded by path (``kernel.py`` imports only ``nki``/``nkilib``), so vLLM is not needed.
Exits non-zero on failure.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np

import nki

_HERE = pathlib.Path(__file__).resolve()
_KERNEL = (_HERE.parents[3] / "vllm_neuron" / "functional" / "vendored_kernels"
           / "latent_cache_write" / "kernel.py")
spec = importlib.util.spec_from_file_location("cache_row_scatter", _KERNEL)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
kernel = mod.cache_row_scatter_kernel

FAILURES: list[str] = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def case(num_pages, page_elems, d, n_live, n_dead, lnc, seed):
    rng = np.random.default_rng(seed)
    cache = rng.standard_normal((num_pages, page_elems)).astype(np.float32)
    rows_per = num_pages * page_elems // d
    sink_row = rows_per - page_elems // d          # first row of the last (sink) page
    live = rng.choice(sink_row, size=n_live, replace=False)
    idx = np.concatenate([live, np.full(n_dead, sink_row)]).astype(np.int32)
    perm = rng.permutation(len(idx))
    idx = idx[perm]
    new = rng.standard_normal((len(idx), d)).astype(np.float32)

    want = cache.copy().reshape(-1, d)
    want[idx[idx != sink_row]] = new[idx != sink_row]
    k = kernel[lnc] if lnc > 1 else kernel
    got = np.asarray(nki.simulate(k)(cache=cache.copy(), new_rows=new,
                                     row_idx=idx.reshape(-1, 1)), dtype=np.float32)
    got = got.reshape(-1, d)
    mask = np.ones(rows_per, dtype=bool)
    mask[sink_row: sink_row + page_elems // d] = False        # sink content unspecified
    ok = np.array_equal(got[mask], want[mask])
    sink_ok = any(np.array_equal(got[sink_row], new[j]) for j in np.where(idx == sink_row)[0]) \
        if n_dead else True
    check(f"d={d} rows={len(idx)} (live {n_live}, dead {n_dead}) LNC{lnc}", ok and sink_ok,
          "" if ok else f"{int((got[mask] != want[mask]).any(-1).sum())} rows differ")


def main() -> int:
    print("cache_row_scatter_kernel vs index_copy_")
    # tiny-test page: 1280 fp32 elements = 40 latent rows of 32 = 80 index rows of 16
    for lnc in (1, 2):
        case(num_pages=12, page_elems=1280, d=32, n_live=200, n_dead=56, lnc=lnc, seed=1)
        case(num_pages=12, page_elems=1280, d=16, n_live=300, n_dead=13, lnc=lnc, seed=2)
        case(num_pages=12, page_elems=1280, d=32, n_live=5, n_dead=0, lnc=lnc, seed=3)
    print("FAILED: " + ", ".join(FAILURES) if FAILURES else "all passed")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
