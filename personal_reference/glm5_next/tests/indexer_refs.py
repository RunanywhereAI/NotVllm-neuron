# SPDX-License-Identifier: Apache-2.0
"""Two independent DSA-indexer references, vendored for tests/test_indexer.py.

Neither can be imported: vLLM ships manylinux wheels only and needs CUDA/Triton,
and the oracle is deliberately transformers-free. So the relevant functions are
reproduced here with provenance, as tests/test_mhc.py does for vLLM's mHC.
Anything that is not a verbatim copy says so.

* ``HFIndexer``: transformers 5.17.0,
  ``models/glm5_next/modeling_glm5_next.py:739-1027`` (``Glm5NextTextIndexer``).
  The code of ``forward``, ``get_visible_tokens``, ``get_pooled_states`` and
  ``append_visible_tail`` is verbatim. Docstrings and illustrative comments are
  trimmed, the ``past_key_values`` branch is removed (prefill only), and
  ``__init__`` takes plain values rather than a config. It has no short-sequence shortcut, so it always runs the sparse path.
* ``vllm_select``: vLLM @ 36fa72d2d0. The torch helpers ``expand_pools_to_tokens``,
  ``append_tail_to_topk`` (``nvidia/ops/kpool_compress.py:727-829``) and
  ``_fill_causal_indices`` (``common/sparse_indexer.py:109``) are verbatim.
  ``_ref_fp8_mqa_logits`` is verbatim from
  ``tests/kernels/attention/test_deepgemm_attention.py:67`` with
  ``device="cuda"`` removed. The pool compression is a **transcription** of the
  Triton kernel ``_kpool_softmax_rotate_write_cache_kernel`` (per-channel max,
  exp, weighted sum, divide), with its Hadamard + FP8 tail omitted because the
  oracle is exact-math. The top-k follows the reference in
  ``tests/kernels/test_top_k_per_row.py:229-233``. The glue follows
  ``nvidia/sparse_indexer.py`` (prefill path, ``index_kpool > 1``) and
  ``common/attention.py:316-401``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================== transformers 5.17
class HFIndexer(nn.Module):
    def __init__(self, hidden_size, n_heads, head_dim, q_lora_rank, index_topk, index_kpool,
                 index_kpool_always_select_tail=True):
        super().__init__()
        self.hidden_size: int = hidden_size
        self.n_heads: int = n_heads
        self.head_dim: int = head_dim
        self.index_topk: int = index_topk
        self.q_lora_rank: int = q_lora_rank

        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(self.hidden_size, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim**-0.5

        self.index_kpool = index_kpool
        self.index_kpool_always_select_tail = index_kpool_always_select_tail

        self.index_kpool_compress_ape = nn.Parameter(torch.zeros(self.index_kpool, self.head_dim))
        self.index_kpool_compress_gate = nn.Parameter(torch.zeros(self.head_dim, self.hidden_size))

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        attention_mask: torch.BoolTensor,
    ) -> torch.LongTensor:
        batch_size, seq_len = hidden_states.shape[:2]
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        q = self.wq_b(q_resid).view(hidden_shape)
        k = self.k_norm(self.wk(hidden_states)).view(hidden_shape).squeeze(2)

        gate_scores = F.linear(hidden_states, self.index_kpool_compress_gate)
        valid_channel = attention_mask.to(k.dtype)[..., None]

        packed_states = torch.cat([k, gate_scores, valid_channel], dim=-1)

        kv_len = seq_len
        current_length = seq_len
        # [vendoring] past_key_values branch removed: prefill only

        # Get pools based on the valid key entries (based on padding / causality)
        valid_keys = packed_states[..., -1].bool()
        visible_tokens = self.get_visible_tokens(
            valid_keys=valid_keys,
            q_length=seq_len,
            current_length=current_length,
        )

        # Key difference: Score across pools, not on a per token basis
        pool_keys, pool_indices, pool_valid = self.get_pooled_states(packed_states=packed_states)
        scores = torch.matmul(q.float(), pool_keys.transpose(-1, -2).float().unsqueeze(1))
        scores = F.relu(scores * self.softmax_scale)

        # Weight per head and sum across heads: [B, S, 1, H] @ [B, S, H, P] -> [B, S, P]
        weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float() * (self.n_heads**-0.5)
        index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

        # Clamp invalid / static pool ends
        pool_end = pool_indices[..., -1].clamp(0, kv_len - 1)
        pool_visible = visible_tokens.gather(
            dim=-1,
            index=pool_end[:, None, :].expand(batch_size, seq_len, -1),
        )
        # A pool is selectable only if its final token is visible to the query
        valid_candidates = pool_visible & pool_valid[:, None]

        index_scores = index_scores.masked_fill(
            ~valid_candidates,
            torch.finfo(index_scores.dtype).min,
        )

        # Similar budgeting as in original but compressed by its pool size
        select_k = min(self.index_topk // self.index_kpool, index_scores.shape[-1])

        selected = index_scores.topk(select_k, dim=-1).indices
        batch_idx = torch.arange(batch_size, device=hidden_states.device)[:, None, None]

        selected_valid = valid_candidates.gather(-1, selected)
        selected_indices = pool_indices[batch_idx, selected]

        # Convert selected pools back into the raw tokens
        # [B, S, K, P] -> [B, S, K * P]
        topk_indices = selected_indices.flatten(-2)
        topk_indices = topk_indices.masked_fill(
            ~selected_valid[..., None].expand_as(selected_indices).flatten(-2),
            -1,
        )

        output_width = self.index_topk
        if self.index_kpool_always_select_tail:
            topk_indices = self.append_visible_tail(topk_indices, visible_tokens, valid_keys)
            output_width += self.index_kpool - 1  # expanded tail size maximum

        # Pad so we fill up with invalid entries instead of gathered selections
        topk_indices = F.pad(topk_indices, (0, output_width - topk_indices.shape[-1]), value=-1)

        topk_indices = topk_indices[..., :output_width]
        topk_indices = topk_indices.masked_fill(~attention_mask[..., None], -1)

        return topk_indices.to(torch.int32)

    def get_visible_tokens(
        self,
        valid_keys: torch.BoolTensor,
        q_length: int,
        current_length: int,
    ) -> torch.BoolTensor:
        device = valid_keys.device

        kv_positions = torch.arange(valid_keys.shape[-1], device=device)
        q_positions = current_length - q_length + torch.arange(q_length, device=device)
        causal = kv_positions[None, None, :] <= q_positions[None, :, None]

        return causal & valid_keys[:, None, :]

    def get_pooled_states(
        self,
        packed_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.LongTensor, torch.BoolTensor]:
        keys, gate_scores, valid_keys = torch.split(
            packed_states,
            [self.head_dim, self.head_dim, 1],
            dim=-1,
        )
        valid_keys = valid_keys.bool().squeeze(-1)

        # Metadata
        batch_size, seq_len = keys.shape[:2]
        number_of_pools = (seq_len + self.index_kpool - 1) // self.index_kpool
        device = keys.device

        first_key = torch.where(
            valid_keys.any(-1),
            valid_keys.long().argmax(-1),
            torch.full((batch_size,), seq_len, dtype=torch.long, device=device),
        )
        pool_offsets = torch.arange(number_of_pools * self.index_kpool, device=device)
        pool_offsets = pool_offsets.view(1, number_of_pools, self.index_kpool)
        pool_indices = first_key[:, None, None] + pool_offsets

        batch_idx = torch.arange(batch_size, device=device)[:, None, None]
        safe_indices = pool_indices.clamp(0, seq_len - 1)

        grouped_keys = keys[batch_idx, safe_indices]
        grouped_gate_scores = gate_scores[batch_idx, safe_indices]
        grouped_valid_keys = valid_keys[batch_idx, safe_indices]

        # Only allow those within range (clamp)
        grouped_valid_keys = grouped_valid_keys & (pool_indices < seq_len)
        pool_valid = grouped_valid_keys.all(-1)
        pool_indices = pool_indices.masked_fill(~grouped_valid_keys, -1)

        # Learn a weighted average over the tokens inside each complete pool
        logits = grouped_gate_scores.float() + self.index_kpool_compress_ape.float()[None, None]
        logits = logits.masked_fill(~grouped_valid_keys[..., None], float("-inf"))
        probabilities = torch.nan_to_num(logits.softmax(dim=2)).to(
            grouped_keys.dtype
        )  # nan to num for full invalid pools
        pool_keys = (probabilities * grouped_keys).sum(dim=2)

        # Avoids static cache allocated positions
        keep = pool_valid.any(0)

        return pool_keys[:, keep], pool_indices[:, keep], pool_valid[:, keep]

    def append_visible_tail(
        self,
        topk_indices: torch.Tensor,
        token_visible: torch.BoolTensor,
        key_valid: torch.BoolTensor,
    ) -> torch.Tensor:
        if (max_tail_width := self.index_kpool - 1) == 0:
            return topk_indices

        batch_size, _, kv_length = token_visible.shape
        device = token_visible.device

        first_key = torch.where(
            key_valid.any(-1),
            key_valid.long().argmax(-1),
            torch.full((batch_size,), kv_length, dtype=torch.long, device=device),
        )
        visible_count = token_visible.long().sum(-1)
        tail_count = visible_count.remainder(self.index_kpool)
        tail_offsets = torch.arange(max_tail_width, device=device)

        tail_start = first_key[:, None] + visible_count - tail_count
        tail_indices = tail_start[..., None] + tail_offsets

        # We exclude tails that are just use to fill in positions + those that go beyond the max length
        tail_valid = (tail_offsets[None, None, :] < tail_count[..., None]) & tail_indices.lt(kv_length)

        # Also check for padding based tokens
        kv_idx = tail_indices.clamp(0, kv_length - 1)
        tail_visible = token_visible.gather(dim=-1, index=kv_idx)

        # Get the valid conclusion
        tail_indices = tail_indices.masked_fill(~(tail_valid & tail_visible), -1)

        return torch.cat([topk_indices, tail_indices], dim=-1)


# ============================================================================ vLLM
def history_group_budget_for_topk(topk: int, pool_size: int) -> int:
    """Number of pools to select so that expanding yields ``topk`` tokens."""
    assert topk % pool_size == 0
    return topk // pool_size


def expand_pools_to_tokens(
    group_ids: torch.Tensor,
    group_valid: torch.Tensor,
    topk: int,
    pool_size: int,
    page_table: torch.Tensor | None = None,
    topk_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand selected full-pool ids to a strict-width token topk tensor."""
    assert group_ids.ndim == 2
    assert group_valid.shape == group_ids.shape
    assert topk % pool_size == 0
    assert group_ids.shape[1] == history_group_budget_for_topk(topk, pool_size)
    assert page_table is None or topk_offsets is None

    device = group_ids.device
    offsets = torch.arange(pool_size, device=device, dtype=torch.int64)
    token_ids = group_ids.to(torch.int64).unsqueeze(-1) * pool_size + offsets
    token_ids = token_ids.reshape(group_ids.shape[0], topk)
    valid = (
        group_valid.unsqueeze(-1)
        .expand(-1, -1, pool_size)
        .reshape(group_ids.shape[0], topk)
    )

    if page_table is not None:
        assert page_table.ndim == 2
        safe_ids = token_ids.clamp(min=0, max=page_table.shape[1] - 1)
        output = torch.gather(page_table, dim=1, index=safe_ids).to(torch.int32)
    elif topk_offsets is not None:
        if topk_offsets.ndim == 2:
            assert topk_offsets.shape[1] == 1
            topk_offsets = topk_offsets.squeeze(1)
        output = (token_ids + topk_offsets.to(torch.int64).unsqueeze(1)).to(torch.int32)
    else:
        output = token_ids.to(torch.int32)

    return torch.where(valid, output, torch.full_like(output, -1))


def append_tail_to_topk(
    topk_result: torch.Tensor,
    seq_lens: torch.Tensor,
    pool_lens: torch.Tensor,
    pool_size: int,
    page_table: torch.Tensor | None = None,
    topk_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Append non-pooled tail tokens after expanded history tokens.

    ``index_kpool_always_select_tail`` keeps the (incomplete) trailing pool so
    the most recent tokens are always attended to.
    """
    assert topk_result.dtype == torch.int32
    assert seq_lens.ndim == 1
    assert pool_lens.ndim == 1

    tail_pool = pool_size - 1
    if tail_pool == 0:
        return topk_result

    rows, n_cols = topk_result.shape
    out_cols = n_cols + tail_pool
    out = torch.empty(
        (rows, out_cols), dtype=topk_result.dtype, device=topk_result.device
    )

    # tail tokens: [pool_len*pool_size, seq_len) for each row.
    pool_len = pool_lens.to(torch.int32)
    tail_start = pool_len * pool_size
    seq_len = seq_lens.to(torch.int32)
    tail_count = seq_len - tail_start  # in [0, pool_size)

    cols = torch.arange(out_cols, device=topk_result.device)[None, :]
    history_len = n_cols
    is_history = cols < history_len
    tail_off = cols - history_len
    is_tail = (tail_off >= 0) & (tail_off < tail_count[:, None])

    safe_hist = torch.minimum(cols, torch.full_like(cols, n_cols - 1)).expand(
        rows, out_cols
    )
    history_val = torch.gather(topk_result, 1, safe_hist)

    tail_raw = tail_start[:, None] + tail_off
    tail_val = tail_raw.to(torch.int32)
    if page_table is not None:
        safe_tail = tail_raw.clamp(min=0, max=page_table.shape[1] - 1)
        tail_val = torch.gather(page_table, 1, safe_tail).to(torch.int32)
    elif topk_offsets is not None:
        tail_val = (tail_raw + topk_offsets.to(torch.int64).unsqueeze(1)).to(
            torch.int32
        )

    out = torch.where(is_history, history_val, -1)
    out = torch.where(is_tail, tail_val, out)
    return out


def _fill_causal_indices(rows: torch.Tensor, positions: torch.Tensor) -> None:
    causal_range = torch.arange(rows.shape[1], device=rows.device, dtype=torch.int32)
    positions = positions.to(torch.int32)
    rows[:] = causal_range[None, :]
    rows[causal_range[None, :] > positions[:, None]] = -1


def _ref_fp8_mqa_logits(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
):
    seq_len_kv = kv.shape[0]

    k = kv
    q = q.float()
    k = k.float()

    mask_lo = (
        torch.arange(0, seq_len_kv)[None, :] >= cu_seqlen_ks[:, None]
    )
    mask_hi = (
        torch.arange(0, seq_len_kv)[None, :] < cu_seqlen_ke[:, None]
    )
    mask = mask_lo & mask_hi
    score = torch.einsum("mhd,nd->hmn", q, k)
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float("-inf"))

    return logits


def _kpool_compress_transcribed(slot_k, slot_score, ape):
    """Transcription of _kpool_softmax_rotate_write_cache_kernel, passes 1 and 2 only.
    slot_k, slot_score: [n_pools, pool_size, head_dim]; ape: [pool_size, head_dim]."""
    max_score = torch.full(slot_k[:, 0].shape, -float("inf"))
    for slot in range(slot_k.shape[1]):
        score = slot_score[:, slot].float() + ape[slot].float()
        max_score = torch.maximum(max_score, score)
    acc = torch.zeros(slot_k[:, 0].shape)
    denom = torch.zeros(slot_k[:, 0].shape)
    for slot in range(slot_k.shape[1]):
        score = slot_score[:, slot].float() + ape[slot].float()
        prob = torch.exp(score - max_score)
        denom += prob
        acc += slot_k[:, slot].float() * prob
    return acc / denom          # [vendoring] Hadamard-128 + FP8 quant omitted: exact-math oracle


@torch.no_grad()
def vllm_select(wq_b, wk, k_norm_w, k_norm_b, weights_proj, gate_w, ape, x, q_c, topk, kpool):
    """One request's prefill, x: [S, hidden], q_c: [S, q_lora_rank] -> [S, topk + kpool - 1] int32.

    Always the sparse path (vLLM would take its short-prefill shortcut for S <= topk)."""
    S = x.shape[0]
    n_head, head_dim = weights_proj.shape[0], wk.shape[0]
    # common/attention.py Indexer.forward (qk_rope_head_dim == 0: no rope branch)
    q = F.linear(q_c, wq_b).view(-1, n_head, head_dim)
    k = F.linear(x, wk)
    weights = torch.mm(x.float(), weights_proj.t().contiguous().float())
    k = F.layer_norm(k.float(), (head_dim,), k_norm_w, k_norm_b, 1e-6).type_as(k)
    q_scale = torch.ones(S, n_head, 1)                      # [vendoring] no FP8: dequant scale is 1
    weights = (weights.unsqueeze(-1) * q_scale * (head_dim**-0.5 * n_head**-0.5)).squeeze(-1)
    gate_score = F.linear(x, gate_w)
    # _kpool_compress_insert: one entry per complete pool, written at its last token
    n_pools = S // kpool
    pool_k = _kpool_compress_transcribed(
        k[: n_pools * kpool].view(n_pools, kpool, head_dim),
        gate_score[: n_pools * kpool].view(n_pools, kpool, head_dim), ape)
    # indexer metadata (compressed): ks = row start, ke = row start + (start_pos + 1 + offset) // ratio
    positions = torch.arange(S)
    ks = torch.zeros(S, dtype=torch.int64)
    ke = (positions + 1) // kpool
    logits = _ref_fp8_mqa_logits(q, pool_k, weights, ks, ke)
    # top_k_per_row_prefill, per its torch reference
    select_k = topk // kpool
    pool_topk = torch.full((S, select_k), -1, dtype=torch.int32)
    for i in range(S):
        row_end = int(ke[i])
        k_i = min(select_k, row_end)
        pool_topk[i, :k_i] = logits[i, :row_end].topk(k_i, dim=-1)[1].to(torch.int32)
    pool_ids = pool_topk.to(torch.int64)
    expanded = expand_pools_to_tokens(pool_ids, pool_ids >= 0, topk, kpool)
    seq_lens = positions + 1
    return append_tail_to_topk(expanded, seq_lens, seq_lens // kpool, kpool)


def vllm_short_prefill(S, width):
    """What vLLM writes when it takes its short-prefill shortcut (S <= topk)."""
    rows = torch.empty(S, width, dtype=torch.int32)
    _fill_causal_indices(rows, torch.arange(S))
    return rows
