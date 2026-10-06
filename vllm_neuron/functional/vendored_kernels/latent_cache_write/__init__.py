# SPDX-License-Identifier: Apache-2.0
"""In-place row scatter into a single paged cache buffer -- the MLA latent write.

Why it exists
-------------
``Tensor.index_put_`` (or ``index_copy_``) on a cache parameter is not an in-place
write under neuronx-cc: XLA has no in-place scatter, so each write lowers to a
full-pool ``scatter`` that materialises a copy of the whole buffer. vLLM shares one
raw buffer between one layer of every KV-cache group, and the FX aliasing pass only
rewrites the *last* write per placeholder into an in-place update, so the
intermediate copies survive. On MiMo-V2.5 that was ~124 GB of pool traffic per decode
step (PR #40, TPOT 199 -> 128.5 ms once fixed). GLM-5.3-Flash writes three regions of
each of 11 shared MLA pages per step, so the same pathology applies.

At ``d_head > 128`` nkilib's TKG attention block forbids in-kernel cache update
(``attention_block_tkg.py:519-526``), and the latent is 512 wide, so the write must be
external regardless.

Provenance
----------
Adapted from upstream vllm-neuron PR #40 (``whn09``), file
``vllm_neuron/functional/attention/kv_cache_write.py`` at ``refs/pull/40/head`` =
``254b0ee`` (last touched by ``7b8a0e7``, "Cut MiMo-V2.5 decode from 199 to 128 ms with
an in-place KV scatter"). PR #40 is **not** merged into this fork -- it is ~300 files --
so only this kernel is lifted, and changed:

* **One buffer, not a K/V pair.** PR #40's guard requires
  ``k_cache.shape == v_cache.shape`` and scatters both. That is structurally wrong for a
  latent cache, where one tensor is both K and V: passing the latent page as both
  arguments double-scatters every row and hands the aliasing pass two outputs on one
  buffer -- the hazard the kernel exists to prevent. Reading may alias; writing must not.
* **Row width is the caller's**, not ``head_dim``. The folded MLA page holds three
  regions with two row widths (``kv_lora_rank`` for the latent, ``index_head_dim`` for
  pool keys and the tail); ``model/glm5_next/cache_layout.py`` guarantees the page is a
  whole number of rows at either width, and owns every row index.
* **No skip sentinel in the protocol.** Dead rows are sent to the runner's private sink
  page by the caller, so the torch and kernel paths write the same bytes.
* Tiles alternate between the two logical cores instead of K on one and V on the other.

SYNC: if PR #40 or a successor lands upstream with a single-buffer variant, delete this
package and call that instead.
"""

from __future__ import annotations

import os

import torch
from torch import Tensor

_KERNEL_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _can_use_kernel(cache: Tensor, rows: Tensor) -> bool:
    """Whether the NKI scatter applies; otherwise the torch fallback runs."""
    try:
        from vllm_neuron.utils.neuron_utils import can_run_kernel
    except ImportError:  # a host without the plugin's runtime (the oracle laptop)
        return False
    if not can_run_kernel(rows):
        return False
    # Diagnostic escape hatch, as PR #40's VLLM_NEURON_KV_WRITE_FORCE_TORCH: isolates
    # "is the wrong answer coming from the cache write" without disabling every kernel.
    if os.environ.get("VLLM_NEURON_KV_WRITE_FORCE_TORCH") == "1":
        return False
    if cache.dtype not in _KERNEL_DTYPES or rows.dtype != cache.dtype:
        return False
    return True


def write_cache_rows(cache: Tensor, rows: Tensor, row_idx: Tensor) -> Tensor:
    """Write ``rows`` into ``cache`` viewed as ``[-1, rows.shape[1]]`` at ``row_idx``.

    Args:
        cache: ``[num_pages, page_elems]``, the bound page-major view. Written in place.
        rows: ``[N, d]``; cast to the cache dtype here.
        row_idx: ``[N]`` destination rows, all inside the buffer. Two entries may name
            the same row only if both are dead rows sent to the sink page: which one
            lands is unspecified on both paths.

    Returns:
        ``cache``. On the kernel path this is the kernel's aliased output. Callers do
        not rebind module state to it -- under ``torch.compile`` that would mutate the
        module mid-forward -- and the aliasing pass rewires downstream readers itself
        (PR #40's MiMo model discards it the same way). Callers must also not depend on
        whether a read later in the same step sees this write: substitute the value
        being written instead.
    """
    if cache.dim() != 2:
        raise ValueError(f"cache must be the 2-D page view, got {tuple(cache.shape)}")
    if rows.dim() != 2 or row_idx.dim() != 1 or row_idx.shape[0] != rows.shape[0]:
        raise ValueError(
            f"rows {tuple(rows.shape)} and row_idx {tuple(row_idx.shape)} must be "
            f"[N, d] and [N]"
        )
    d = rows.shape[1]
    if cache.shape[1] % d:
        raise ValueError(
            f"page of {cache.shape[1]} elements is not a whole number of {d}-wide rows; "
            f"cache_layout.LatentPageLayout pads the page so that it always is"
        )
    rows = rows.to(cache.dtype)

    if not _can_use_kernel(cache, rows):
        cache.view(-1, d).index_copy_(0, row_idx.to(torch.long), rows)
        return cache

    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .kernel import cache_row_scatter_kernel

    wrapped = wrap_nki(cache_row_scatter_kernel)
    return wrapped[2](
        cache=cache,
        new_rows=rows,
        row_idx=row_idx.to(torch.int32).view(-1, 1),
    )


__all__ = ["write_cache_rows"]
