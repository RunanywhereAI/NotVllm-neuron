# SPDX-License-Identifier: Apache-2.0
"""NKI row scatter into ONE paged cache buffer. See ``__init__.py`` for provenance.

Imported only when the kernel path is taken: it needs ``nki`` and ``nkilib`` at import
time, and the dispatcher in ``__init__`` must stay importable on a host that has
neither (the CPU oracle comparison runs on a laptop).
"""

from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import oob_mode
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info


@nki.jit
def cache_row_scatter_kernel(cache: Tensor, new_rows: Tensor, row_idx: Tensor):
    """Scatter ``new_rows`` into ``cache`` at rows ``row_idx`` and return ``cache``.

    PR #40's ``_kv_cache_scatter_kernel`` with the K/V pair collapsed to one buffer.
    Three properties are kept from it deliberately:

    * the cache arrives in its **bound** shape ``[num_pages, page_elems]`` and is
      flattened to ``[-1, d]`` HERE -- the aliasing pass only rewires a write whose FX
      shape matches the placeholder exactly, so a caller-side reshape would silently
      disable the write chaining this kernel exists for;
    * the cache is **returned**, which is what makes the compiler report
      ``operand_output_aliases``; a kernel that writes an input and returns nothing has
      its write dropped from the graph;
    * ``oob_mode.skip``: a destination row outside the buffer is discarded rather than
      wrapped. The dispatcher never sends one (dead rows go to the runner's sink page),
      so this is a backstop, not a protocol.

    What changed: there is no V. PR #40 split K and V across the two logical cores;
    here tiles alternate between them instead, so with LNC=2 each row is written
    exactly once rather than twice.

    Args:
        cache: ``[num_pages, page_elems]``; ``page_elems`` must be a multiple of ``d``.
        new_rows: ``[N, d]``, already at the cache dtype.
        row_idx: ``[N, 1]`` int32 destination rows in the ``[-1, d]`` view.
    """
    _, n_prgs, prg_id = get_verified_program_sharding_info("cache_row_scatter", (0, 1), 2)

    num_pages = cache.shape[0]
    page_elems = cache.shape[1]
    n_rows = new_rows.shape[0]
    d = new_rows.shape[1]
    flat = cache.reshape((num_pages * page_elems // d, d))

    tile_sz = nl.tile_size.pmax
    # A plain ``for start in range`` target: NKI's parser rejects tuple unpacking (and
    # so ``enumerate``) in a ``for`` target -- PR #40 hit the same restriction.
    for start in range(0, n_rows, tile_sz):
        if n_prgs == 1 or (start // tile_sz) % n_prgs == prg_id:
            tile_n = min(tile_sz, n_rows - start)
            idx_tile = nl.ndarray((tile_n, 1), dtype=row_idx.dtype, buffer=nl.sbuf)
            nisa.dma_copy(idx_tile, row_idx[nl.ds(start, tile_n)])
            tile = nl.ndarray((tile_n, d), dtype=new_rows.dtype, buffer=nl.sbuf)
            nisa.dma_copy(tile, new_rows[nl.ds(start, tile_n), :])
            nisa.dma_copy(
                dst=flat.ap(
                    pattern=[[d, tile_n], [1, d]],
                    offset=0,
                    vector_offset=idx_tile,
                    indirect_dim=0,
                ),
                src=tile,
                oob_mode=oob_mode.skip,
            )

    return cache
