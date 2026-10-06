# SPDX-License-Identifier: Apache-2.0
"""The one place the folded MLA page's offsets are derived.

GLM-5.3-Flash needs four kinds of cache: recurrent state for its 34 KDA layers, a
latent KV cache for its 11 sparse-MLA layers, the DSA indexer's kpool-compressed
scoring cache, and the indexer's raw tail cache. vLLM 0.24.0 requires **every** KV
cache group to share one page size, and in vLLM's own (FP8) format the indexer's page
cannot be unified with the latent page **for any block size**:

    latent_page / indexer_page  =  1024*B / (0.25 * H * B)  =  4096 / H

``B`` cancels, so unification needs ``H | 4096``. vLLM's ``H`` is 132 bytes --
``index_head_dim`` 128 plus 4 bytes of inline FP8 block scale per pool entry -- and 132
does not divide 4096. It is structural, not a tuning problem. See
``personal_docs/GLM53-FLASH-FRAMEWORK-GAP.md`` §4 for the derivation and the three
options; this module implements option (c): **fold the indexer and tail caches inside
the MLA layer's own page**, so vLLM's planner still sees exactly two cache kinds
(attention and recurrent) -- the configuration PR #54 left working.

**Pool-key format.** The plugin has no FP8 indexer: no Hadamard rotation, no e4m3
quantisation, and neither does the oracle it is validated against. So the pool keys are
stored **at the page's element type** (``indexer_fp8=False``, the default), which makes
the torch path reproduce the oracle's selection exactly. vLLM's FP8 format stays here as
arithmetic (``indexer_fp8=True``) because it is the reason folding exists, but nothing
writes it and ``row index`` helpers refuse it. FP8 scoring changes the selected pool set
on 88-100% of rows above the dense-exact ceiling, so turning it on is a validation
project, not a flag.

**Why this is a module and not three expressions at the call sites.** The runner sizes
the page from these offsets and the model slices it with them. PR #54's own comment on
the identical problem is the reason:

    Reusing vLLM's implementation rather than reimplementing it matters because the
    same helper also sizes the state *pages* the planner allocates; the two have to
    agree exactly, and a second copy of the arithmetic is a silent memory-aliasing
    bug waiting to happen.

Silent, not loud -- which is why ``test/unit/test_glm5next_cache_layout.py`` fails if
this arithmetic is re-derived anywhere else rather than trusting the convention. The
same goes for addressing *inside* a page: the model turns vLLM slots and block ids into
row indices only through the methods below.

This module is also the entire blast radius of a future vLLM bump. A vLLM carrying
``cache_role=INDEXER`` and ``KpoolTailSpec`` makes folding unnecessary, and replacing
this file plus the runner's latent branch is then the whole change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# Bytes of FP8 block scale stored inline per pool entry, from vLLM's own indexer cache:
# ``head_dim=self.head_dim + self.head_dim // self.quant_block_size * 4``
# (``vllm/models/glm5next/common/attention.py``). These 4 bytes are the entire reason
# the indexer page cannot be unified with the latent page -- 128 divides 4096, 132 does
# not -- so the constant is named rather than inlined.
_FP8_SCALE_BYTES = 4

# vLLM hardcodes this and asserts it (``attention.py``: ``assert self.head_dim == 128
# and self.quant_block_size == 128``). It is not a config field, so it is a default
# here rather than something read from the checkpoint.
_DEFAULT_QUANT_BLOCK = 128


def _align_up(n: int, to: int) -> int:
    return -(-n // to) * to


@dataclass(frozen=True)
class LatentPageLayout:
    """Byte and element offsets of the three regions inside one MLA page.

    A page covers ``block_size`` tokens of one MLA layer and holds, in order:

    ===========  ==========================================  ==========================
    region       contents                                    size (default format)
    ===========  ==========================================  ==========================
    ``latent``   full-fidelity latent, one row per token      ``block_size * kvr`` elems
    ``indexer``  compressed pool key, one row per pool        ``pools * D`` elems
    ``tail``     raw K and gate score for the open pool       ``kpool * 2 * D`` elems
    ===========  ==========================================  ==========================

    Every region starts on a multiple of its own row width, and the page ends on a
    multiple of every row width, so the whole buffer can be viewed as ``[-1, width]``
    rows for any of the three widths. That is what lets one row-scatter kernel write
    every region of a shared page without a strided view (Neuron rejects strided
    in-place writes on bound tensors, and PR #40's scatter flattens the bound tensor
    into rows *inside* the kernel). At the real config this costs no padding at all:
    49152 + 3072 + 1024 = 53248 = 104 x 512.

    The tail is stored in **every** page and the model uses the one in the page
    holding the open pool. That is only correct while an incomplete pool cannot
    straddle a block boundary, i.e. while ``block_size % index_kpool == 0`` --
    asserted in ``__post_init__`` rather than assumed, and the same constraint vLLM's
    own indexer cache asserts so that "chunked-prefill boundaries stay pool-aligned".

    All three regions share one element type, because a page-major view is only
    contiguous at full width.
    """

    block_size: int
    kv_lora_rank: int
    index_head_dim: int
    index_kpool: int
    element_size: int
    quant_block_size: int = _DEFAULT_QUANT_BLOCK
    indexer_fp8: bool = False

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_config(cls, config, block_size: int, dtype: torch.dtype) -> "LatentPageLayout":
        """Derive the layout from a ``Glm5NextTextConfig`` and the resolved block size.

        Duck-typed on the attribute names rather than importing the config class, so
        this module has no dependency on the model package's import order.
        """
        return cls(
            block_size=block_size,
            kv_lora_rank=config.kv_lora_rank,
            index_head_dim=config.index_head_dim,
            index_kpool=config.index_kpool,
            element_size=dtype.itemsize,   # no tensor: this runs inside a traced forward
        )

    def __post_init__(self) -> None:
        if self.block_size % self.index_kpool:
            raise ValueError(
                f"block_size ({self.block_size}) must be a multiple of index_kpool "
                f"({self.index_kpool}): the tail region holds the open pool for the "
                f"page containing the current token, which is only well defined when "
                f"an incomplete pool cannot straddle a block boundary. vLLM's own "
                f"indexer cache asserts the same thing to keep prefill boundaries "
                f"pool-aligned."
            )
        if self.indexer_fp8 and self.index_head_dim % self.quant_block_size:
            raise ValueError(
                f"index_head_dim ({self.index_head_dim}) must be a multiple of "
                f"quant_block_size ({self.quant_block_size}); the inline scale count "
                f"is index_head_dim // quant_block_size."
            )
        for name in ("latent_bytes", "indexer_bytes", "tail_bytes"):
            if getattr(self, name) % self.element_size:
                raise ValueError(
                    f"{name} ({getattr(self, name)}) is not a multiple of the page "
                    f"element size ({self.element_size}); the regions must be "
                    f"addressable in a single typed view."
                )

    # ---------------------------------------------------------------------- per-token

    @property
    def latent_bytes_per_token(self) -> int:
        """``kv_lora_rank`` wide. ``qk_rope_head_dim`` is 0 -- GLM-5.3-Flash's MLA is
        NoPE -- so there is no separate rope half and the cache is one uniform width.
        That is also what lets a single tensor serve as both K and V: measured
        bit-identical, see MLA-DECODE-GAP.md §2.5."""
        return self.kv_lora_rank * self.element_size

    @property
    def indexer_bytes_per_pool(self) -> int:
        """One pool key. Default: ``index_head_dim`` elements at the page dtype (256 B
        in bf16). FP8: entry plus inline block scale, 128 + 4 = 132 B."""
        if self.indexer_fp8:
            return self.index_head_dim + (
                self.index_head_dim // self.quant_block_size
            ) * _FP8_SCALE_BYTES
        return self.index_head_dim * self.element_size

    # ------------------------------------------------------------------ region sizes

    @property
    def pools_per_page(self) -> int:
        return self.block_size // self.index_kpool

    @property
    def latent_bytes(self) -> int:
        return self.block_size * self.latent_bytes_per_token

    @property
    def indexer_bytes(self) -> int:
        return self.pools_per_page * self.indexer_bytes_per_pool

    @property
    def tail_bytes(self) -> int:
        """``index_kpool`` slots, each holding raw K **and** a gate score.

        vLLM stores these as the "K" and "V" halves of a two-head block
        (``Glm5NextTailCache``: ``num_kv_heads=2, head_size=head_dim``), so two
        ``index_head_dim``-wide rows per slot at the page's element size.
        """
        return self.index_kpool * 2 * self.index_head_dim * self.element_size

    @property
    def total_bytes(self) -> int:
        return self.total_elems * self.element_size

    # -------------------------------------------------------------- element offsets
    #
    # The model slices a typed view of the page, so it needs offsets in elements. These
    # exist so the model never divides a byte offset by an element size itself -- that
    # division is the second copy of the arithmetic this module is here to prevent.

    @property
    def latent_elems(self) -> int:
        return self.latent_bytes // self.element_size

    @property
    def indexer_elems(self) -> int:
        return self.indexer_bytes // self.element_size

    @property
    def tail_elems(self) -> int:
        return self.tail_bytes // self.element_size

    @property
    def latent_elem_offset(self) -> int:
        return 0

    @property
    def indexer_elem_offset(self) -> int:
        if self.indexer_fp8:
            return self.latent_elems
        return _align_up(self.latent_elems, self.index_head_dim)

    @property
    def tail_elem_offset(self) -> int:
        if self.indexer_fp8:
            return self.indexer_elem_offset + self.indexer_elems
        return _align_up(self.indexer_elem_offset + self.indexer_elems, self.index_head_dim)

    @property
    def total_elems(self) -> int:
        end = self.tail_elem_offset + self.tail_elems
        if self.indexer_fp8:
            return end
        return _align_up(end, math.lcm(self.kv_lora_rank, self.index_head_dim))

    # ----------------------------------------------------------------- byte offsets

    @property
    def latent_offset(self) -> int:
        return self.latent_elem_offset * self.element_size

    @property
    def indexer_offset(self) -> int:
        return self.indexer_elem_offset * self.element_size

    @property
    def tail_offset(self) -> int:
        return self.tail_elem_offset * self.element_size

    # ------------------------------------------------------------------- addressing
    #
    # Everything below maps vLLM's addressing (a slot is ``block * block_size +
    # offset``; a block id is a page index) to row indices inside the folded page.
    # Rows are counted in the flat ``[-1, width]`` view of the whole page buffer, which
    # is the frame both the torch fallback and the NKI row scatter write in.

    def _rows_only(self) -> None:
        if self.indexer_fp8:
            raise NotImplementedError(
                "row addressing is defined for the page-dtype pool format only; the "
                "FP8 indexer format has no scoring path in this plugin"
            )

    @property
    def latent_row_width(self) -> int:
        return self.kv_lora_rank

    @property
    def index_row_width(self) -> int:
        """Width of a pool row and of a tail row: ``index_head_dim`` elements."""
        return self.index_head_dim

    def slot_to_page(self, slot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """vLLM slot -> ``(page, token offset within the page)``."""
        return slot // self.block_size, slot % self.block_size

    def latent_row(self, page, token_in_page):
        """Row of one token's latent in the ``[-1, kv_lora_rank]`` view."""
        self._rows_only()
        per_page = self.total_elems // self.kv_lora_rank
        return page * per_page + self.latent_elem_offset // self.kv_lora_rank + token_in_page

    def pool_row(self, page, token_in_page):
        """Row of the pool holding ``token_in_page``, in the ``[-1, index_head_dim]``
        view. Pools are ``index_kpool``-aligned inside a page, so the token's pool is
        ``token_in_page // index_kpool``."""
        self._rows_only()
        per_page = self.total_elems // self.index_head_dim
        return (page * per_page + self.indexer_elem_offset // self.index_head_dim
                + token_in_page // self.index_kpool)

    def tail_rows(self, page, token_in_page):
        """``(k_row, gate_row)`` of the tail slot ``token_in_page % index_kpool`` in the
        ``[-1, index_head_dim]`` view. A slot is two rows: raw K, then gate score."""
        self._rows_only()
        per_page = self.total_elems // self.index_head_dim
        base = (page * per_page + self.tail_elem_offset // self.index_head_dim
                + 2 * (token_in_page % self.index_kpool))
        return base, base + 1

    def split(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gathered page rows ``[..., total_elems]`` -> typed region views.

        Returns ``latent [..., block_size, kvr]``, ``pools [..., pools_per_page, D]``
        and ``tail [..., index_kpool, 2, D]`` (``[..., 0, :]`` raw K, ``[..., 1, :]``
        gate score). For reading only: the slices are strided.
        """
        self._rows_only()
        lead = rows.shape[:-1]
        D = self.index_head_dim
        latent = rows[..., self.latent_elem_offset: self.latent_elem_offset + self.latent_elems]
        pools = rows[..., self.indexer_elem_offset: self.indexer_elem_offset + self.indexer_elems]
        tail = rows[..., self.tail_elem_offset: self.tail_elem_offset + self.tail_elems]
        return (latent.reshape(*lead, self.block_size, self.kv_lora_rank),
                pools.reshape(*lead, self.pools_per_page, D),
                tail.reshape(*lead, self.index_kpool, 2, D))

    # ------------------------------------------------------------------- reporting

    def describe(self) -> str:
        """One line per region, for the startup log.

        The capacity wall in GLM53-FLASH-FRAMEWORK-GAP.md §5 is worth printing rather
        than leaving in a document: the latent cache is replicated per rank, so it is
        the term that decides whether a context length fits at all.
        """
        return (
            f"MLA page {self.total_bytes} B over {self.block_size} tokens = "
            f"latent {self.latent_bytes} + indexer {self.indexer_bytes} "
            f"({self.pools_per_page} pools x {self.indexer_bytes_per_pool}) + "
            f"tail {self.tail_bytes} + pad "
            f"{self.total_bytes - self.latent_bytes - self.indexer_bytes - self.tail_bytes}"
        )

    def per_rank_latent_bytes(self, max_model_len: int, num_mla_layers: int) -> int:
        """Bytes of latent cache one rank holds for a single ``max_model_len`` sequence.

        MLA decodes as MQA with one KV head, and nothing shards a single head, so every
        rank holds the whole latent cache. At 1M tokens over 11 layers that is 11
        GiB/rank against roughly 24 GiB, which is the wall behind the recommendation to
        bring up at 128K first.
        """
        return max_model_len * self.latent_bytes_per_token * num_mla_layers


def latent_page_bytes(config, dtype: torch.dtype, block_size: int) -> int:
    """Bytes of one folded MLA page -- the number both the runner's spec and the
    platform's page alignment use, so neither can compute it differently."""
    return LatentPageLayout.from_config(config, block_size, dtype).total_bytes
