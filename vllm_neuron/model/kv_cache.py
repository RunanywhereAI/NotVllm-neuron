# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from dataclasses import dataclass, field

import torch


@dataclass
class LayerSpec:
    """
    Defines the KV cache specification for a single transformer layer.

    Used to specify the memory requirements and configuration for storing
    key-value pairs in the attention mechanism of a transformer layer.
    """

    name: str
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    sliding_window_size: int | None = None
    chunk_size: int | None = None


@dataclass
class RecurrentLayerSpec:
    """Cache specification for a layer that keeps recurrent state, not a KV cache.

    Linear-attention layers (gated DeltaNet, Mamba) hold a **fixed-size** state
    per sequence however long that sequence is, so they have no block table, no
    per-token growth and no context length. ``LayerSpec`` cannot describe them:
    ``num_kv_heads``/``head_size`` are meaningless here, and what matters instead
    is the concrete state tensor shapes.

    Shapes are per-rank and given in vLLM's own order -- for gated DeltaNet that
    is ``(conv_state, recurrent_state)``. Take them from
    ``MambaStateShapeCalculator`` rather than deriving them: vLLM sizes the state
    pages from the same helper, so a divergent layout aliases memory instead of
    raising.

    Attributes:
        name: Layer name, matching the key vLLM uses for its cache tensor.
        shapes: One shape per state tensor, per rank.
        dtypes: One dtype per state tensor, same order as ``shapes``.
    """

    name: str
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[torch.dtype, ...]


@dataclass
class LatentLayerSpec:
    """Cache specification for an MLA layer whose page also holds its indexer caches.

    An MLA layer keeps a single **latent** KV entry per token -- ``kv_lora_rank`` wide,
    with no separate V, and for GLM-5.3-Flash no rope half either since
    ``qk_rope_head_dim`` is 0. ``LayerSpec`` cannot describe it: its page arithmetic
    assumes a K **and** a V (``2 * block_size * num_kv_heads * head_size``), which is
    twice the bytes and, worse, the shape the runner would then try to view.

    Why the indexer's caches ride inside this page rather than getting groups of their
    own: vLLM 0.24.0 requires every KV cache group to share one page size, and the
    DSA indexer's page cannot be unified with the latent page for **any** block size --
    the ratio is ``4096 / H`` with the block size cancelling, and ``H`` is 132 because
    of 4 bytes of inline FP8 scale per pool entry. Folding keeps the planner at two
    cache kinds, which is the configuration PR #54 left working. The offsets live in
    ``model/glm5_next/cache_layout.py`` and are derived exactly once; do not recompute
    them here or in the model.

    Attributes:
        name: Layer name, matching the key vLLM uses for its cache tensor.
        kv_lora_rank: Latent width per token.
        dtype: Element type of the page.
        page_bytes_for: ``block_size -> bytes`` for one page, i.e.
            ``LatentPageLayout.total_bytes`` -- latent plus indexer plus tail. A
            function rather than a number because the block size is resolved by the
            platform's hybrid alignment, which the model never sees; the runner calls
            this with the block size it is about to allocate with, and the platform
            calls the same layout code through the registered class, so the two cannot
            disagree.
    """

    name: str
    kv_lora_rank: int
    dtype: torch.dtype
    page_bytes_for: Callable[[int], int]


@dataclass
class PagedLayerSpec:
    """A pseudo-layer whose page content the model lays out itself.

    For models whose per-token state fits neither a K/V pair nor one latent, such as
    DeepSeek-V4.1, which keeps a sliding window of raw K and compressor state plus a
    full-length store of compressed entries. The runner allocates whole pages and
    binds one ``[num_pages, page_elems]`` view; the model owns every offset inside.

    ``sliding_window`` set makes it a sliding-window group, so vLLM frees blocks that
    leave the window. ``None`` makes it a full-attention group. Every paged layer of a
    model must report the same ``block_size * page_elems * itemsize``: vLLM 0.24.0
    requires one page size across groups, though block sizes may differ.

    Attributes:
        name: Layer name, the key of this group's ``attn_metadata`` and cache tensor.
        block_size: Tokens per block for this group.
        page_elems: Elements per page, at ``dtype``.
        dtype: Element type of the page view.
        sliding_window: Window in tokens, or None for a full-length group.
    """

    name: str
    block_size: int
    page_elems: int
    dtype: torch.dtype
    sliding_window: int | None = None


@dataclass
class KVSpec:
    """
    Defines the KV cache needs of a model by specifying all layer configurations.

    Contains a list of LayerSpec objects that collectively define the complete
    KV cache requirements for an entire transformer model.
    """

    layers: list[LayerSpec]
    recurrent_layers: list[RecurrentLayerSpec] = field(default_factory=list)
    latent_layers: list[LatentLayerSpec] = field(default_factory=list)
    paged_layers: list[PagedLayerSpec] = field(default_factory=list)


def reserved_pages(num_pages: int) -> tuple[int, int]:
    """``(zero_page, sink)``: the two private pages the runner appends past ``num_blocks``
    under ``kv_cache_page_major``.

    Dead rows READ the zero page -- nothing ever writes it, so it still holds the zeros
    the allocation put there -- and WRITE the sink. vLLM knows about neither.
    """
    sink = num_pages - 1
    return sink - 1, sink


def paged_block_ids(
    block_table: torch.Tensor, live_rows: torch.Tensor, num_pages: int
) -> torch.Tensor:
    """A decode batch's block table with every unusable entry sent to the zero page.

    Padded batch rows, and the unused tail of a live row's table, carry stale ids or
    the null block (0); on device an out-of-range id is an out-of-bound indirect DMA,
    and a recycled id hands back another group's bytes. Gathering the zero page instead
    keeps every gathered byte finite, which the caller still has to *select* away (not
    multiply away -- ``NaN * 0`` is NaN) for positions past the sequence.

    Args:
        block_table: ``[num_reqs, max_blocks]``.
        live_rows: ``[num_reqs]`` bool.
        num_pages: First dimension of the bound pages, reserved pages included.
    """
    zero_page, _ = reserved_pages(num_pages)
    bt = block_table.to(torch.long)
    ok = live_rows.view(-1, 1) & (bt > 0) & (bt < zero_page)
    return torch.where(ok, bt, torch.full_like(bt, zero_page))


def state_page_indices(
    metadata: dict, num_reqs: int, num_pages: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-request read and write page for a recurrent-state batch, as ``(read, write)``.

    Canonical implementation, shared by every model with recurrent state. It was lifted
    out of ``qwen3_5/deltanet.py`` (which now delegates here) rather than copied,
    because it carries two version-sensitive details and one non-obvious hazard, and a
    second copy is how two models drift apart silently.

    The recurrent group's ``mamba_block_size`` equals ``max_model_len``, so every
    sequence owns exactly one block and its state slot is that block's id.

    **Padded batch rows need two different redirects**, which is why this returns a
    pair. They must *read* zeros: their output is discarded but their logits are not,
    and the sampler's argmax reduces across the whole tile, so a dead row that reads
    another group's bytes as float32 state hands its NaN logits to every live row in the
    batch. And they must *write* somewhere else, since writing the zero page is what
    would stop it being zeros. The runner reserves one page for each, past
    ``num_blocks``.

    Args:
        metadata: This layer's ``attn_metadata`` entry.
        num_reqs: Padded batch width.
        num_pages: First dimension of the bound state pages, i.e. ``num_blocks`` plus
            the two reserved pages.
    """
    block_table = metadata["block_table_tensor"]
    indices = block_table[:num_reqs, 0].to(torch.long)
    slot_mapping = metadata["slot_mapping"].view(num_reqs, -1)[:, 0]
    zero_page, sink = reserved_pages(num_pages)
    # ``> 0``, not ``>= 0``: the runner's padding sentinel changed from PAD_SLOT_ID (-1)
    # on 0.21 to NULL_BLOCK_ID (0) on 0.24, and 0 is vLLM's reserved null block, so no
    # live token ever maps there. Bound-check the id too -- a padded batch row's
    # block_table entry is simply stale from an earlier step, and on device an
    # out-of-range index is an out-of-bound indirect DMA rather than a wrapped one.
    live = (slot_mapping > 0) & (indices > 0) & (indices < zero_page)
    return torch.where(
        live, indices, torch.full_like(indices, zero_page)
    ), torch.where(
        live, indices, torch.full_like(indices, sink)
    )
