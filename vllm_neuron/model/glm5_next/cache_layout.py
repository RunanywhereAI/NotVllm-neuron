# SPDX-License-Identifier: Apache-2.0
"""The one place the folded MLA page's offsets are derived.

GLM-5.3-Flash needs four kinds of cache: recurrent state for its 34 KDA layers, a
latent KV cache for its 11 sparse-MLA layers, the DSA indexer's kpool-compressed
scoring cache, and the indexer's raw tail cache. vLLM 0.24.0 requires **every** KV
cache group to share one page size, and the indexer's page cannot be unified with the
latent page **for any block size**:

    latent_page / indexer_page  =  1024*B / (0.25 * H * B)  =  4096 / H

``B`` cancels, so unification needs ``H | 4096``. The indexer's ``H`` is 132 bytes --
``index_head_dim`` 128 plus 4 bytes of inline FP8 block scale per pool entry -- and 132
does not divide 4096. It is structural, not a tuning problem. See
``personal_docs/GLM53-FLASH-FRAMEWORK-GAP.md`` §4 for the derivation and the three
options; this module implements option (c): **fold the indexer and tail caches inside
the MLA layer's own page**, so vLLM's planner still sees exactly two cache kinds
(attention and recurrent) -- the configuration PR #54 left working.

**Why this is a module and not three expressions at the call sites.** The runner sizes
the page from these offsets and the model slices it with them. PR #54's own comment on
the identical problem is the reason:

    Reusing vLLM's implementation rather than reimplementing it matters because the
    same helper also sizes the state *pages* the planner allocates; the two have to
    agree exactly, and a second copy of the arithmetic is a silent memory-aliasing
    bug waiting to happen.

Silent, not loud -- which is why ``tests/test_cache_layout.py`` fails if this
arithmetic is re-derived anywhere else rather than trusting the convention.

This module is also the entire blast radius of a future vLLM bump. A vLLM carrying
``cache_role=INDEXER`` and ``KpoolTailSpec`` makes folding unnecessary, and replacing
this file plus the runner's latent branch is then the whole change.
"""

from __future__ import annotations

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


@dataclass(frozen=True)
class LatentPageLayout:
    """Byte and element offsets of the three regions inside one MLA page.

    A page covers ``block_size`` tokens of one MLA layer and holds, contiguously:

    ===========  ==========================================  ==========================
    region       contents                                    size
    ===========  ==========================================  ==========================
    ``latent``   full-fidelity latent, one per token          ``block_size * 1024`` B
    ``indexer``  fp8 kpool entry + inline scale, per pool     ``pools * 132`` B
    ``tail``     raw K and gate score for the open pool       ``index_kpool * 2 * 128``
    ===========  ==========================================  ==========================

    The tail is per *request*, not per block, and is stored in **every** page; the
    model uses the region in the page holding the current token. That is only correct
    while an incomplete pool cannot straddle a block boundary, i.e. while
    ``block_size % index_kpool == 0`` -- asserted in ``__post_init__`` rather than
    assumed, and the same constraint vLLM's own indexer cache asserts so that
    "chunked-prefill boundaries stay pool-aligned".

    All three regions share one element type, because a page-major view is only
    contiguous at full width and Neuron rejects strided in-place writes on bound
    tensors (PR #54 hit the same wall for recurrent state). The indexer's fp8 bytes are
    therefore addressed as raw bytes within a wider view by the model, not as a typed
    sub-tensor.
    """

    block_size: int
    kv_lora_rank: int
    index_head_dim: int
    index_kpool: int
    element_size: int
    quant_block_size: int = _DEFAULT_QUANT_BLOCK

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
            element_size=torch.tensor([], dtype=dtype).element_size(),
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
        if self.index_head_dim % self.quant_block_size:
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
        """fp8 entry plus its inline block scale: 128 + 4 = 132 at the real config."""
        return self.index_head_dim + (self.index_head_dim // self.quant_block_size) * _FP8_SCALE_BYTES

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
        return self.latent_bytes + self.indexer_bytes + self.tail_bytes

    # ----------------------------------------------------------------- byte offsets

    @property
    def latent_offset(self) -> int:
        return 0

    @property
    def indexer_offset(self) -> int:
        return self.latent_bytes

    @property
    def tail_offset(self) -> int:
        return self.latent_bytes + self.indexer_bytes

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
    def total_elems(self) -> int:
        return self.total_bytes // self.element_size

    @property
    def latent_elem_offset(self) -> int:
        return 0

    @property
    def indexer_elem_offset(self) -> int:
        return self.latent_elems

    @property
    def tail_elem_offset(self) -> int:
        return self.latent_elems + self.indexer_elems

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
            f"tail {self.tail_bytes}"
        )

    def per_rank_latent_bytes(self, max_model_len: int, num_mla_layers: int) -> int:
        """Bytes of latent cache one rank holds for a single ``max_model_len`` sequence.

        MLA decodes as MQA with one KV head, and nothing shards a single head, so every
        rank holds the whole latent cache. At 1M tokens over 11 layers that is ~11.5
        GiB/rank against roughly 24 GiB, which is the wall behind the recommendation to
        bring up at 128K first.
        """
        return max_model_len * self.latent_bytes_per_token * num_mla_layers
