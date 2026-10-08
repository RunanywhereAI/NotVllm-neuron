# SPDX-License-Identifier: Apache-2.0
"""Page arithmetic for DeepSeek-V4.1's two KV cache groups. The single source of offsets.

V4.1 keeps three kinds of per-sequence state, and they want two different lifetimes:

* **Window rows**, needed only for the last ``window_size`` tokens: every layer's
  sliding-window K (= V), plus each ratio-2 compressor's ``(kv, score)`` for a group
  that has not closed yet. A sliding-window group, so vLLM frees blocks that fall
  out of the window and caches the rest for prefix hits.
* **Compressed entries**, kept for the whole sequence: per KV-source layer, one
  latent (``head_dim``) and one index key (``index_head_dim``) for every
  ``compress_ratio`` tokens. A full-attention group.

Both groups are presented to vLLM as single pseudo-layers whose pages hold
everything of their kind, so vLLM sees exactly two layers and two groups. vLLM
0.24.0 requires every group to have the same page size in bytes; it allows different
block sizes, and hashes at their GCD. So the window block size is the cache block
size, the compressed block is ``k`` times larger, and the window page is padded to
match. Window rows cost little, because a sequence holds only about
``window_size / window_block + 2`` of them.

Window page, field-major: ``[n_fields, window_block, field]`` in float32, where
``field == head_dim`` and each field is one 512-wide row per token. Float32 because
the compressor's open-group state is float32 in the reference, and storing it
narrower would make prefill and decode disagree. The window K is stored in float32
too, losslessly. That is cheap at this size.

Compressed page, region-major per source: ``latent[E_s, head_dim]`` then
``index[E_s, index_head_dim]`` for each source ``s`` in order, with
``E_s = comp_block // ratio_s``. Element type is the model dtype.

Every region starts at a multiple of its row width, so ``write_cache_rows`` can view a
page as rows of that width and address each row directly. A group never straddles
a compressed block, because ``comp_block`` is a multiple of every ratio, and prefix
hits land on block boundaries, so a cache hit never leaves a group half-open.
"""

from __future__ import annotations

import dataclasses
import math

import torch

WINDOW_LAYER = "deepseek_v41.window"
COMPRESSED_LAYER = "deepseek_v41.compressed"


def _elem(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


@dataclasses.dataclass(frozen=True)
class CacheLayout:
    head_dim: int
    index_head_dim: int
    window_size: int
    # window group
    window_block: int
    n_fields: int
    kv_field: tuple[int, ...]               # per layer
    state_fields: dict[int, tuple[int, int]]  # ratio-2 source layer -> (kv, score) fields
    # compressed group
    comp_block: int
    comp_dtype: torch.dtype
    sources: tuple[int, ...]                # kv source layers, in order
    ratio: dict[int, int]
    latent_off: dict[int, int]              # elements from page start
    index_off: dict[int, int]
    comp_page_elems: int

    # -- construction ------------------------------------------------------------------
    @classmethod
    def build(cls, args, window_block: int, comp_dtype: torch.dtype,
              max_k: int = 4096) -> "CacheLayout":
        """``args`` uses the reference ``ModelArgs`` field names. Backbone layers only."""
        n_layers = args.n_layers
        ratios = tuple(args.compress_ratios[:n_layers])
        sources = tuple(s for s in args.kv_source_layers if s < n_layers)
        for s in sources:
            if ratios[s] < 1:
                raise ValueError(f"kv source layer {s} has compress ratio {ratios[s]}")
        dh, di = args.head_dim, args.index_head_dim
        state_src = tuple(s for s in sources if ratios[s] > 1)
        need = n_layers + 2 * len(state_src)

        # bytes of compressed state per sequence token; per-entry row widths must be
        # whole numbers of elements per token, which holds for every ratio dividing
        # (head_dim + index_head_dim)
        esz = _elem(comp_dtype)
        per_tok = 0
        for s in sources:
            if (dh + di) % ratios[s]:
                raise ValueError(f"ratio {ratios[s]} does not divide {dh}+{di}")
            per_tok += (dh + di) // ratios[s] * esz
        field_bytes = dh * 4
        lcm_r = math.lcm(*(ratios[s] for s in sources)) if sources else 1

        # smallest k (comp_block = k * window_block) giving an integral, sufficient
        # window field count and an integral entry count per compressed block
        for k in range(1, max_k + 1):
            comp_block = k * window_block
            if comp_block % lcm_r:
                continue
            page = per_tok * comp_block
            if page % (field_bytes * window_block):
                continue
            n_fields = page // (field_bytes * window_block)
            if n_fields >= need:
                break
        else:
            raise ValueError("no compressed block size equalises the two page sizes")

        state_fields, f = {}, n_layers
        for s in state_src:
            state_fields[s] = (f, f + 1)
            f += 2

        latent_off, index_off, off = {}, {}, 0
        for s in sources:
            e = comp_block // ratios[s]
            latent_off[s] = off
            off += e * dh
            index_off[s] = off
            off += e * di
        lay = cls(head_dim=dh, index_head_dim=di, window_size=args.window_size,
                  window_block=window_block, n_fields=n_fields,
                  kv_field=tuple(range(n_layers)), state_fields=state_fields,
                  comp_block=comp_block, comp_dtype=comp_dtype, sources=sources,
                  ratio={s: ratios[s] for s in sources}, latent_off=latent_off,
                  index_off=index_off, comp_page_elems=off)
        lay._check()
        return lay

    def _check(self) -> None:
        if self.window_page_bytes != self.comp_page_bytes:
            raise AssertionError((self.window_page_bytes, self.comp_page_bytes))
        for s in self.sources:
            if self.latent_off[s] % self.head_dim or self.index_off[s] % self.index_head_dim:
                raise AssertionError(f"region of source {s} is not row-aligned")
        if self.comp_page_elems % self.head_dim or self.comp_page_elems % self.index_head_dim:
            raise AssertionError("compressed page is not a whole number of rows")

    # -- sizes -------------------------------------------------------------------------
    @property
    def window_page_elems(self) -> int:
        return self.n_fields * self.window_block * self.head_dim

    @property
    def window_page_bytes(self) -> int:
        return self.window_page_elems * 4

    @property
    def comp_page_bytes(self) -> int:
        return self.comp_page_elems * _elem(self.comp_dtype)

    def entries_per_block(self, source: int) -> int:
        return self.comp_block // self.ratio[source]

    # -- row addressing ------------------------------------------------------------------
    # Every function returns row ids into a flat ``[-1, width]`` view of the bound pages.
    def window_row(self, page: torch.Tensor, tok: torch.Tensor, field: int) -> torch.Tensor:
        """Row of ``field`` for in-block token ``tok`` of window page ``page``."""
        return (page * self.n_fields + field) * self.window_block + tok

    def latent_row(self, source: int, page: torch.Tensor, entry: torch.Tensor) -> torch.Tensor:
        # truncating division: floor division on int64 lowers through float64 on Neuron
        return torch.div(page * self.comp_page_elems + self.latent_off[source], self.head_dim,
                         rounding_mode="trunc") + entry

    def index_row(self, source: int, page: torch.Tensor, entry: torch.Tensor) -> torch.Tensor:
        return torch.div(page * self.comp_page_elems + self.index_off[source], self.index_head_dim,
                         rounding_mode="trunc") + entry


def source_of(layer: int, sources) -> int | None:
    """The most recent source at or below ``layer``: the reference's shared-runtime rule."""
    best = None
    for s in sources:
        if s <= layer:
            best = s
    return best
