# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash text decoder for the Neuron plugin.

The module tree and parameter names mirror DeepSeek's reference ``model.py``
(``personal_reference/deepseek_v41/ref``) one to one, so the dequantized checkpoint
and the CPU oracle load the same state dict. The one exception is the routed
experts: they are stacked at load time into the layout the NKI MoE kernels take
(``ffn.gate_up_proj [E, D, 2, I]``, ``ffn.down_proj [E, I, D]``).

**One step, two shapes.** Prefill is one sequence of ``T`` bucket-padded tokens,
possibly continuing a cached prefix. Decode is ``n`` sequences of one token each.
Both run the same code as ``[n, T]``. Every read of earlier tokens goes through the
paged caches in ``cache_layout.py``:

* the last ``window_size`` positions before the step come from the window group;
* compressed entries come from the compressed group, with this step's fresh
  entries substituted for whatever the page holds. A read never depends on whether
  this step's writes are visible yet.

Everything is written for static shapes: no Python branch reads a tensor value,
and dead rows are removed with ``where`` rather than multiplication, since an
unwritten page can hold NaN and ``NaN * 0`` is NaN.

What the reference does that this does not:

* **Fake quantization.** The reference rounds activations through fp8 and fp4 to
  imitate its kernels. This model computes in its own dtype. It is checked against
  the oracle's *exact* mode, which skips the fake quantization too.
* **The stale-index-key bug** (see ``personal_reference/deepseek_v41/oracle.py``).
  Index keys are read from the owner's cache region by construction here, so the
  bug cannot occur.
"""

from __future__ import annotations

import dataclasses
import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_neuron.functional.vendored_kernels.latent_cache_write import write_cache_rows
from vllm_neuron.model.kv_cache import KVSpec, PagedLayerSpec, paged_block_ids, reserved_pages

from .cache_layout import COMPRESSED_LAYER, WINDOW_LAYER, CacheLayout, source_of

try:
    from vllm_neuron.accuracy.tensor_capture import capture_tensor as _capture_tensor
except ImportError:  # pragma: no cover - hosts without vLLM
    def _capture_tensor(name, tensor):  # type: ignore[misc]
        return None


# ------------------------------------------------------------------------- basic ops
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


def _linear(in_f: int, out_f: int) -> nn.Linear:
    return nn.Linear(in_f, out_f, bias=False)


@functools.lru_cache(maxsize=8)
def rope_freqs(dim: int, original_seq_len: int, base: float, factor: float,
               beta_fast: int, beta_slow: int) -> torch.Tensor:
    """The reference's ``precompute_freqs_cis`` without the position table: fp32 [dim/2]."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return freqs


def rope_cos_sin(freqs: torch.Tensor, pos: torch.Tensor):
    """``pos`` any integer shape -> cos, sin of shape ``pos.shape + [dim/2]``, fp32.

    The angle is ``float32(pos) * freq``, as the reference's ``torch.outer`` computes it.
    """
    ang = pos.to(torch.float32).unsqueeze(-1) * freqs.to(pos.device)
    return torch.cos(ang), torch.sin(ang)


def rope_tail(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rd: int,
              inverse: bool = False) -> torch.Tensor:
    """Rotate the last ``rd`` dims of ``x`` as adjacent (re, im) pairs. ``cos``/``sin``
    broadcast against ``x[..., :rd/2]``."""
    head, tail = x[..., :-rd], x[..., -rd:].float().unflatten(-1, (-1, 2))
    a, b = tail[..., 0], tail[..., 1]
    if inverse:
        sin = -sin
    rot = torch.stack([a * cos - b * sin, a * sin + b * cos], dim=-1).flatten(-2)
    return torch.cat([head, rot.to(x.dtype)], dim=-1)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc: int, iters: int, eps: float):
    """pre / post / comb from one mHC projection; ``comb`` gets exactly ``iters`` Sinkhorn
    passes ending on a column division (truncation, not convergence -- reproduce it)."""
    mixes, hc_scale, hc_base = mixes.float(), hc_scale.float(), hc_base.float()
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def attend(q: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor, sink: torch.Tensor,
           scale: float) -> torch.Tensor:
    """Sparse attention as a masked dense one. K = V.

    q ``[n, T, H, D]``; keys ``[n, M, D]`` shared by the request's T queries; mask
    ``[n, T, M]``. The per-head ``sink`` logit enters the denominator only. A query with
    no valid key yields zeros (the reference kernel's -1e30 running max).
    """
    used = mask.any(dim=1).unsqueeze(-1)                       # [n, M, 1]
    keys = torch.where(used, keys, torch.zeros_like(keys)).float()
    s = torch.einsum("nthd,nmd->nthm", q.float(), keys) * scale
    s = s.masked_fill(~mask.unsqueeze(2), float("-inf"))
    smax = s.amax(-1).clamp_min(-1e30)
    p = torch.exp(s - smax.unsqueeze(-1))
    denom = p.sum(-1) + torch.exp(sink.float() - smax)
    o = torch.einsum("nthm,nmd->nthd", p, keys) / denom.unsqueeze(-1)
    return o.to(q.dtype)


def select_candidate_blocks(logits, compress_lens, topk_blocks: int, block_size: int):
    """Level one of the two-level top-k, verbatim from the reference."""
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, torch.inf)
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


# ----------------------------------------------------------------------------- step
@dataclasses.dataclass
class Step:
    """Per-forward addressing shared by every layer. ``n`` requests x ``T`` tokens."""

    decode: bool
    pos: torch.Tensor          # [n, T] long
    live: torch.Tensor         # [n, T] real token with a cache slot
    s: torch.Tensor            # [n] position of the step's first token
    # window group
    w_page: torch.Tensor       # [n, T] write page (sink when dead)
    w_tok: torch.Tensor        # [n, T]
    h_pos: torch.Tensor        # [n, W] history positions s-W .. s-1
    h_page: torch.Tensor       # [n, W] (zero page when unusable)
    h_tok: torch.Tensor        # [n, W]
    h_ok: torch.Tensor         # [n, W]
    # compressed group
    c_page: torch.Tensor       # [n, T] write page of each token's compressed block
    c_off: torch.Tensor        # [n, T] token offset within that block
    c_blocks: torch.Tensor     # [n, nbc] sanitised block table

    @property
    def n(self) -> int:
        return self.pos.shape[0]

    @property
    def T(self) -> int:
        return self.pos.shape[1]

    def entries(self, lay: CacheLayout, ratio: int):
        """Every compressed entry the block table can address: ``(page, entry)`` of shape
        ``[n, N_c]`` with ``N_c = nbc * comp_block / ratio``."""
        per = lay.comp_block // ratio
        j = torch.arange(self.c_blocks.shape[1] * per, device=self.pos.device)
        page = self.c_blocks.index_select(1, j // per)
        return page, (j % per).expand_as(page)


def build_step(lay: CacheLayout, num_pages: int, positions: torch.Tensor,
               attn_metadata: dict) -> Step:
    mw, mc = attn_metadata[WINDOW_LAYER], attn_metadata[COMPRESSED_LAYER]
    decode = mw["max_query_len"] <= mw["decode_token_threshold"]
    zero_page, sink = reserved_pages(num_pages)
    bt_w = mw["block_table_tensor"]
    n = bt_w.shape[0]
    pos = positions.to(torch.long).view(n, -1)
    T = pos.shape[1]
    slot_w = mw["slot_mapping"].to(torch.long).view(n, -1)[:, :T]
    slot_c = mc["slot_mapping"].to(torch.long).view(n, -1)[:, :T]
    if decode:
        live = slot_w > 0
    else:
        # pads are appended and repeat the last position, so they break the run
        offs = torch.arange(T, device=pos.device)
        live = ((pos - pos[:, :1]) == offs) & (slot_w > 0)
    live_rows = live.any(dim=1)
    s = pos[:, 0]

    Bw = lay.window_block
    w_page = torch.where(live, slot_w // Bw, torch.full_like(slot_w, sink))
    w_tok = torch.where(live, slot_w % Bw, torch.zeros_like(slot_w))

    W = lay.window_size
    h_pos = s.unsqueeze(1) - W + torch.arange(W, device=pos.device)
    blocks_w = paged_block_ids(bt_w, live_rows, num_pages)
    off = mw.get("swa_kv_pos_offset") if decode else None
    base = (h_pos - off.to(torch.long).view(n, 1)) if off is not None else h_pos
    bidx = base.div(Bw, rounding_mode="floor")
    h_ok = (h_pos >= 0) & (bidx >= 0) & (bidx < blocks_w.shape[1]) & live_rows.view(n, 1)
    h_page = blocks_w.gather(1, bidx.clamp(0, blocks_w.shape[1] - 1))
    h_page = torch.where(h_ok, h_page, torch.full_like(h_page, zero_page))
    h_tok = h_pos.remainder(Bw)

    Bc = lay.comp_block
    c_page = torch.where(live, slot_c // Bc, torch.full_like(slot_c, sink))
    c_off = slot_c % Bc
    c_blocks = paged_block_ids(mc["block_table_tensor"], live_rows, num_pages)
    return Step(decode=decode, pos=pos, live=live, s=s, w_page=w_page, w_tok=w_tok,
                h_pos=h_pos, h_page=h_page, h_tok=h_tok, h_ok=h_ok, c_page=c_page,
                c_off=c_off, c_blocks=c_blocks)


@dataclasses.dataclass
class Caches:
    """The bound page views, plus flat row views of each width the model addresses."""

    lay: CacheLayout
    window: torch.Tensor       # [num_pages, window_page_elems] fp32
    comp: torch.Tensor         # [num_pages, comp_page_elems] model dtype

    @property
    def num_pages(self) -> int:
        return self.window.shape[0]

    def read_window(self, field: int, page, tok) -> torch.Tensor:
        rows = self.lay.window_row(page, tok, field)
        flat = self.window.view(-1, self.lay.head_dim)
        return flat.index_select(0, rows.reshape(-1)).view(*rows.shape, -1)

    def write_window(self, field: int, page, tok, rows) -> None:
        write_cache_rows(self.window, rows.reshape(-1, self.lay.head_dim).float(),
                         self.lay.window_row(page, tok, field).reshape(-1))

    def read_comp(self, source: int, page, entry, index: bool) -> torch.Tensor:
        lay = self.lay
        if index:
            rows, width = lay.index_row(source, page, entry), lay.index_head_dim
        else:
            rows, width = lay.latent_row(source, page, entry), lay.head_dim
        flat = self.comp.view(-1, width)
        return flat.index_select(0, rows.reshape(-1)).view(*rows.shape, width)

    def write_comp(self, source: int, page, entry, rows, index: bool) -> None:
        lay = self.lay
        if index:
            ids, width = lay.index_row(source, page, entry), lay.index_head_dim
        else:
            ids, width = lay.latent_row(source, page, entry), lay.head_dim
        write_cache_rows(self.comp, rows.reshape(-1, width), ids.reshape(-1))


def _substitute(gathered: torch.Tensor, fresh: torch.Tensor, j0: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
    """``gathered [n, N, d]`` with entries ``j0 .. j0+G-1`` replaced by ``fresh [n, G, d]``
    wherever ``valid [n, G]``."""
    n, N, _ = gathered.shape
    G = fresh.shape[1]
    rel = torch.arange(N, device=gathered.device).unsqueeze(0) - j0.view(n, 1)
    inside = (rel >= 0) & (rel < G)
    g = rel.clamp(0, G - 1)
    take = fresh.gather(1, g.unsqueeze(-1).expand(n, N, fresh.shape[-1]))
    ok = inside & valid.gather(1, g)
    return torch.where(ok.unsqueeze(-1), take.to(gathered.dtype), gathered)


# --------------------------------------------------------------------- attention
@dataclasses.dataclass
class Fresh:
    """What a KV source produced this step, before any consumer reads its pages."""

    latent: torch.Tensor   # [n, G, head_dim], RoPE'd
    index_k: torch.Tensor  # [n, G, index_head_dim], RoPE'd
    j0: torch.Tensor       # [n] entry id of the first
    valid: torch.Tensor    # [n, G]


class Compressor(nn.Module):
    """Pools ``ratio`` consecutive tokens into one latent with a learned softmax gate.

    The reference carries the open group in ``kv_state``/``score_state``. Here it is
    the window group's state fields: every token writes its ``(kv, score)``, and a
    decode step that closes a group reads the previous token's back. Prefill starts
    on a group boundary because prefix hits are block-aligned.
    """

    def __init__(self, args, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.ratio = args.compress_ratios[layer_id]
        self.head_dim = args.head_dim
        self.norm = RMSNorm(args.head_dim, args.norm_eps)
        self.wkv = _linear(args.dim, args.head_dim)
        if self.ratio > 1:
            self.wgate = _linear(args.dim, args.head_dim)

    def forward(self, x, step: Step, caches: Caches):
        """-> ``(latent [n, G, head_dim] before RoPE, j0 [n], valid [n, G])``."""
        n, T, _ = x.shape
        r = self.ratio
        if r == 1:
            return self.norm(self.wkv(x)), step.s, step.live
        dtype = x.dtype
        xf = x.float()
        kv, score = F.linear(xf, self.wkv.weight.float()), F.linear(xf, self.wgate.weight.float())
        f_kv, f_sc = caches.lay.state_fields[self.layer_id]
        if step.decode:
            p = step.pos[:, 0]
            prev = step.pos - 1
            # the previous token is the newest history row
            prev_kv = caches.read_window(f_kv, step.h_page[:, -1:], step.h_tok[:, -1:]).float()
            prev_sc = caches.read_window(f_sc, step.h_page[:, -1:], step.h_tok[:, -1:]).float()
            closes = step.live[:, 0] & (p.remainder(r) == r - 1) & (prev[:, 0] >= 0)
            # only ratio 2 is released; a larger ratio would need r - 1 history rows
            if r != 2:
                raise NotImplementedError(f"compress ratio {r} in decode")
            kv2 = torch.cat([torch.where(closes.view(n, 1, 1), prev_kv, torch.zeros_like(prev_kv)), kv], 1)
            sc2 = torch.cat([torch.where(closes.view(n, 1, 1), prev_sc, torch.zeros_like(prev_sc)), score], 1)
            latent = (kv2 * sc2.softmax(dim=1)).sum(dim=1, keepdim=True)
            j0, valid = p.div(r, rounding_mode="floor"), closes.view(n, 1)
        else:
            G = T // r
            kvg = kv[:, :G * r].unflatten(1, (G, r))
            scg = score[:, :G * r].unflatten(1, (G, r))
            latent = (kvg * scg.softmax(dim=2)).sum(dim=2)
            valid = step.live[:, r - 1::r][:, :G] & step.live[:, 0::r][:, :G]
            j0 = step.s.div(r, rounding_mode="floor")
        caches.write_window(f_kv, step.w_page, step.w_tok, kv)
        caches.write_window(f_sc, step.w_page, step.w_tok, score)
        return self.norm(latent.to(dtype)), j0, valid


class Indexer(nn.Module):
    """Scores compressed positions and keeps the best ``index_topk`` per query."""

    def __init__(self, args, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.owns_k = layer_id in args.kv_source_layers
        self.ratio = args.compress_ratios[layer_id]
        self.is_candidate_source = layer_id == args.candidate_source_layer
        self.uses_candidates = 0 <= args.candidate_source_layer < layer_id
        self.candidate_topk_blocks = args.candidate_topk_blocks
        self.candidate_block_size = args.candidate_block_size
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rd = args.rope_head_dim
        self.index_topk = args.index_topk
        self.wq_b = _linear(args.q_lora_rank, self.n_heads * self.head_dim)
        self.weights_proj = _linear(args.dim, self.n_heads)
        if self.owns_k:
            self.wk = _linear(args.head_dim, self.head_dim)
            self.k_norm = RMSNorm(self.head_dim, args.norm_eps)

    def keys(self, latent, cos, sin):
        return rope_tail(self.k_norm(self.wk(latent)), cos, sin, self.rd)

    def forward(self, x, qr, cos, sin, step: Step, caches: Caches, shared: dict):
        n, T, _ = x.shape
        r = self.ratio
        owner = shared["index_owner"][self.layer_id]
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        q = rope_tail(q, cos.unsqueeze(2), sin.unsqueeze(2), self.rd)
        w = self.weights_proj(x).float() * (self.head_dim ** -0.5 * self.n_heads ** -0.5)

        page, entry = step.entries(caches.lay, r)
        k = caches.read_comp(owner, page, entry, index=True)
        fresh: Fresh = shared["fresh"][owner]
        k = _substitute(k, fresh.index_k, fresh.j0, fresh.valid)
        N = k.shape[1]
        lens = (step.pos + 1).div(r, rounding_mode="floor")                  # [n, T]
        reach = torch.arange(N, device=x.device) < lens.unsqueeze(-1)       # [n, T, N]
        k = torch.where(reach.any(dim=1).unsqueeze(-1), k, torch.zeros_like(k)).float()
        score = torch.einsum("nthd,nmd->nthm", q.float(), k).relu()
        score = (score * w.unsqueeze(-1)).sum(dim=2)
        score = score.masked_fill(~reach, float("-inf"))
        if self.is_candidate_source:
            shared["candidates"] = select_candidate_blocks(
                score, lens.unsqueeze(-1), self.candidate_topk_blocks, self.candidate_block_size)
        elif self.uses_candidates:
            score = score.masked_fill(~shared["candidates"], float("-inf"))
        idx = score.topk(min(self.index_topk, N), dim=-1).indices
        _capture_tensor(f"layers.{self.layer_id}.indexer.topk", idx)
        return idx, idx < lens.unsqueeze(-1)


class Attention(nn.Module):
    """Sliding window of raw K plus, at ratio > 0, ``index_topk`` compressed entries, in
    one softmax with a per-head sink."""

    def __init__(self, args, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.rd = args.rope_head_dim
        self.n_groups = args.o_groups
        self.o_lora_rank = args.o_lora_rank
        self.window_size = args.window_size
        self.ratio = args.compress_ratios[layer_id]
        self.softmax_scale = self.head_dim ** -0.5
        eps = args.norm_eps
        self.attn_sink = nn.Parameter(torch.zeros(self.n_heads))
        self.wq_a = _linear(args.dim, args.q_lora_rank)
        self.q_norm = RMSNorm(args.q_lora_rank, eps)
        self.wq_b = _linear(args.q_lora_rank, self.n_heads * self.head_dim)
        self.wkv = _linear(args.dim, self.head_dim)
        self.kv_norm = RMSNorm(self.head_dim, eps)
        self.wo_a = _linear(self.n_heads * self.head_dim // self.n_groups,
                            self.n_groups * self.o_lora_rank)
        self.wo_b = _linear(self.n_groups * self.o_lora_rank, args.dim)
        self.is_kv_source = layer_id in args.kv_source_layers
        self.is_index_source = layer_id in args.index_source_layers
        self.compressor = Compressor(args, layer_id) if self.is_kv_source else None
        self.indexer = Indexer(args, layer_id) if self.is_index_source else None
        if self.ratio:
            freqs = rope_freqs(self.rd, args.original_seq_len, args.compress_rope_theta,
                               args.rope_factor, args.beta_fast, args.beta_slow)
        else:
            freqs = rope_freqs(self.rd, 0, args.rope_theta, args.rope_factor,
                               args.beta_fast, args.beta_slow)
        self.register_buffer("freqs", freqs, persistent=False)

    def _compress(self, x, step: Step, caches: Caches) -> Fresh:
        latent, j0, valid = self.compressor(x, step, caches)
        r = self.ratio
        G = latent.shape[1]
        jpos = (j0.view(-1, 1) + torch.arange(G, device=x.device)) * r     # group's first token
        cos, sin = rope_cos_sin(self.freqs, jpos)
        index_k = self.indexer.keys(latent, cos, sin)
        latent = rope_tail(latent, cos, sin, self.rd)
        # address: the page and in-block entry of each group's first token
        if step.decode:
            tok_page, tok_off = step.c_page, step.c_off
        else:
            tok_page, tok_off = step.c_page[:, 0::r][:, :G], step.c_off[:, 0::r][:, :G]
        _, sink = reserved_pages(caches.num_pages)
        page = torch.where(valid, tok_page, torch.full_like(tok_page, sink))
        entry = torch.where(valid, tok_off.div(r, rounding_mode="floor"), torch.zeros_like(tok_off))
        caches.write_comp(self.layer_id, page, entry, latent, index=False)
        caches.write_comp(self.layer_id, page, entry, index_k, index=True)
        return Fresh(latent=latent, index_k=index_k, j0=j0, valid=valid)

    def forward(self, x, step: Step, caches: Caches, shared: dict):
        n, T, _ = x.shape
        cos, sin = rope_cos_sin(self.freqs, step.pos)                       # [n, T, rd/2]
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        q = rope_tail(q, cos.unsqueeze(2), sin.unsqueeze(2), self.rd)
        kv = rope_tail(self.kv_norm(self.wkv(x)), cos, sin, self.rd)       # [n, T, Dh]

        # window: history rows from the cache, then this step's
        field = caches.lay.kv_field[self.layer_id]
        hist = caches.read_window(field, step.h_page, step.h_tok).to(kv.dtype)
        caches.write_window(field, step.w_page, step.w_tok, kv)
        keys = torch.cat([hist, kv], dim=1)                                 # [n, W+T, Dh]
        kpos = torch.cat([step.h_pos, step.pos], dim=1)
        kok = torch.cat([step.h_ok, step.live], dim=1)
        qp = step.pos.unsqueeze(-1)
        mask = kok.unsqueeze(1) & (kpos.unsqueeze(1) <= qp) & (kpos.unsqueeze(1) > qp - self.window_size)

        if self.ratio:
            if self.compressor is not None:
                shared["fresh"][self.layer_id] = self._compress(x, step, caches)
            if self.indexer is not None:
                shared["topk"] = self.indexer(x, qr, cos, sin, step, caches, shared)
            idx, valid = shared["topk"]                                     # [n, T, K]
            src = shared["kv_source"][self.layer_id]
            fresh: Fresh = shared["fresh"][src]
            page, entry = step.entries(caches.lay, self.ratio)
            if step.decode:
                # one query per request: gather only the selected entries
                ii = idx[:, 0]
                lat = caches.read_comp(src, page.gather(1, ii), entry.gather(1, ii), index=False)
                hit = (ii == fresh.j0.view(n, 1)) & fresh.valid[:, :1]
                lat = torch.where(hit.unsqueeze(-1), fresh.latent[:, :1].to(lat.dtype), lat)
                cmask = valid                                               # [n, 1, K]
            else:
                lat = caches.read_comp(src, page, entry, index=False)
                lat = _substitute(lat, fresh.latent, fresh.j0, fresh.valid)
                N = lat.shape[1]
                # invalid picks go to a spill column: scattering their False in place
                # could clear a True that a valid pick of the same entry set
                cmask = torch.zeros(n, T, N + 1, dtype=torch.bool, device=x.device).scatter(
                    -1, torch.where(valid, idx, torch.full_like(idx, N)), True)[..., :N]
            keys = torch.cat([keys, lat.to(keys.dtype)], dim=1)
            mask = torch.cat([mask, cmask], dim=-1)

        o = attend(q, keys, mask, self.attn_sink, self.softmax_scale)       # [n, T, H, Dh]
        o = rope_tail(o, cos.unsqueeze(2), sin.unsqueeze(2), self.rd, inverse=True)
        G = self.n_groups
        o = o.reshape(n, T, G, -1)
        wo_a = self.wo_a.weight.view(G, self.o_lora_rank, -1)
        o = torch.einsum("ntgd,grd->ntgr", o, wo_a)
        return self.wo_b(o.flatten(2))


# --------------------------------------------------------------------------- MoE
def _clamped_swiglu(gate, up, limit: float):
    """up clamped to ``[-limit, limit]``, gate to ``(-inf, limit]``, then silu(gate) * up."""
    if limit > 0:
        up = up.clamp(-limit, limit)
        gate = gate.clamp(max=limit)
    return F.silu(gate) * up


class Expert(nn.Module):
    def __init__(self, dim: int, inter: int, limit: float):
        super().__init__()
        self.w1, self.w2, self.w3 = _linear(dim, inter), _linear(inter, dim), _linear(dim, inter)
        self.limit = limit

    def forward(self, x):
        h = _clamped_swiglu(self.w1(x).float(), self.w3(x).float(), self.limit)
        return self.w2(h.to(x.dtype))


class Gate(nn.Module):
    """The bias picks experts; the weights come from the unbiased scores."""

    def __init__(self, args):
        super().__init__()
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.gate_temp = args.gate_temp
        self.norm_topk_prob = args.norm_topk_prob
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.zeros(args.n_routed_experts, args.dim))
        self.bias = nn.Parameter(torch.zeros(args.n_routed_experts))

    def forward(self, x):
        scores = F.linear(x.float(), self.weight.float()) / self.gate_temp
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        idx = (scores + self.bias.float()).topk(self.topk, dim=-1).indices
        w = scores.gather(1, idx)
        if self.norm_topk_prob and self.topk > 1:
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
        return w * self.route_scale, idx


class MoE(nn.Module):
    """Routed experts in the NKI MoE layout plus one shared expert."""

    def __init__(self, args):
        super().__init__()
        if args.n_shared_experts != 1:
            raise NotImplementedError("exactly one shared expert")
        E, D, I = args.n_routed_experts, args.dim, args.moe_inter_dim
        self.E, self.limit = E, args.swiglu_limit
        self.gate = Gate(args)
        self.gate_up_proj = nn.Parameter(torch.zeros(E, D, 2, I))
        self.down_proj = nn.Parameter(torch.zeros(E, I, D))
        self.shared_experts = Expert(D, I, args.swiglu_limit)

    def _routed_dense(self, x, w, idx):
        """Every expert on every token, zero weight off the top-k: exact, static, CPU only."""
        dense = torch.zeros(x.shape[0], self.E, dtype=torch.float32, device=x.device)
        dense = dense.scatter(1, idx, w)
        gu = torch.einsum("td,edgi->tegi", x, self.gate_up_proj)
        h = _clamped_swiglu(gu[:, :, 0].float(), gu[:, :, 1].float(), self.limit)
        h = (h * dense.unsqueeze(-1)).to(x.dtype)
        return torch.einsum("tei,eid->td", h, self.down_proj).float()

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        w, idx = self.gate(x)
        y = self._routed_dense(x, w, idx)
        y = y + self.shared_experts(x).float()
        return y.to(x.dtype).view(shape)


# ------------------------------------------------------------------------ engram
class EngramEmbedding(nn.Module):
    """The n-gram table, FP8 with a per-32 e8m0 scale, dequantized to bf16 on lookup."""

    def __init__(self, rows: int, dim: int, block: int = 32):
        super().__init__()
        self.block = block
        self.weight = nn.Parameter(torch.zeros(rows, dim, dtype=torch.float8_e4m3fn), requires_grad=False)
        self.scale = nn.Parameter(torch.zeros(rows, dim // block, dtype=torch.float8_e8m0fnu),
                                  requires_grad=False)

    def forward(self, ids):
        v = F.embedding(ids, self.weight.view(torch.uint8)).view(torch.float8_e4m3fn)
        s = F.embedding(ids, self.scale.view(torch.uint8)).view(torch.float8_e8m0fnu)
        v = v.float().unflatten(-1, (-1, self.block)) * s.float().unsqueeze(-1)
        return v.flatten(-2).to(torch.bfloat16)


class Engram(nn.Module):
    """Gated n-gram lookup added to every hc copy of the residual stream."""

    def __init__(self, args, layer_id: int, rows: int):
        super().__init__()
        self.layer_id = layer_id
        self.dim, self.hc = args.dim, args.hc_mult
        self.eps = args.norm_eps
        cols = (args.engram_max_ngram_size - 1) * args.engram_n_heads
        self.embed = EngramEmbedding(rows, args.engram_head_dim)
        self.wkv = _linear(cols * args.engram_head_dim, args.dim * (args.hc_mult + 1))
        self.q_weight = nn.Parameter(torch.ones(args.hc_mult, args.dim))
        self.k_weight = nn.Parameter(torch.ones(args.hc_mult, args.dim))

    def forward(self, h, hash_ids):
        """h ``[..., hc, dim]``; hash_ids ``[..., cols]``."""
        emb = self.embed(hash_ids).flatten(-2).to(self.wkv.weight.dtype)
        kv = self.wkv(emb)
        key, value = kv.split([self.hc * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc, self.dim))
        weight = self.q_weight.float() * self.k_weight.float()
        hf = h.float()
        rstd = torch.rsqrt(hf.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (hf * weight * key).sum(-1) * rstd * self.dim ** -0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        return (hf + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(h.dtype)


# ------------------------------------------------------------------------- block
class Block(nn.Module):
    """mHC block. The pre-mix a sublayer computes is used by the NEXT sublayer."""

    def __init__(self, args, layer_id: int, engram_rows: int | None):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.hc, self.iters, self.hc_eps = args.hc_mult, args.hc_sinkhorn_iters, args.hc_eps
        self.attn = Attention(args, layer_id)
        self.ffn = MoE(args)
        self.engram = Engram(args, layer_id, engram_rows) if engram_rows is not None else None
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
        mix, hd = (2 + self.hc) * self.hc, self.hc * args.dim
        self.hc_attn_fn = nn.Parameter(torch.zeros(mix, hd))
        self.hc_ffn_fn = nn.Parameter(torch.zeros(mix, hd))
        self.hc_attn_base = nn.Parameter(torch.zeros(mix))
        self.hc_ffn_base = nn.Parameter(torch.zeros(mix))
        self.hc_attn_scale = nn.Parameter(torch.zeros(3))
        self.hc_ffn_scale = nn.Parameter(torch.zeros(3))

    def hc_mixes(self, h, fn, scale, base):
        x = h.flatten(-2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, fn.float()) * rsqrt
        return hc_split_sinkhorn(mixes, scale, base, self.hc, self.iters, self.hc_eps)

    @staticmethod
    def hc_pre(h, pre_mix):
        return torch.sum(pre_mix.unsqueeze(-1) * h.float(), dim=-2).to(h.dtype)

    @staticmethod
    def hc_post(x, residual, post, comb):
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
            comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=-3)
        return y.to(x.dtype)

    def forward(self, h, pre_mix, step, caches, shared, engram_ids):
        if self.engram is not None:
            h = self.engram(h, engram_ids)
        residual = h
        a_pre, a_post, a_comb = self.hc_mixes(h, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm(self.hc_pre(h, pre_mix))
        x = self.attn(x, step, caches, shared)
        h = self.hc_post(x, residual, a_post, a_comb)
        residual = h
        f_pre, f_post, f_comb = self.hc_mixes(h, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm(self.hc_pre(h, a_pre))
        x = self.ffn(x)
        h = self.hc_post(x, residual, f_post, f_comb)
        return h, f_pre


# ------------------------------------------------------------------------ model
# Parameters the reference stores in float32 whatever the model dtype.
_FP32_SUFFIXES = ("attn_sink", "gate.bias", "hc_attn_fn", "hc_ffn_fn", "hc_attn_base",
                  "hc_ffn_base", "hc_attn_scale", "hc_ffn_scale", "head.weight")


class DeepseekV41Model(nn.Module):
    """The decoder. Parameter names equal the reference's (``embed``, ``layers``, ``norm``,
    ``head``), experts excepted."""

    kv_cache_page_major = True

    def __init__(self, args, block_size: int = 32, cache_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.args = args
        n = args.n_layers
        ratios = args.compress_ratios[:n]
        self.kv_sources = tuple(s for s in args.kv_source_layers if s < n)
        self.index_sources = tuple(s for s in args.index_source_layers if s < n)
        if not set(self.kv_sources) <= set(self.index_sources):
            raise ValueError("every kv source must be an index source (it owns the keys)")
        self.kv_source = {i: source_of(i, self.kv_sources) for i in range(n) if ratios[i]}
        self.index_owner = {i: source_of(i, self.kv_sources) for i in self.index_sources}
        for i, s in self.kv_source.items():
            x = source_of(i, self.index_sources)
            if s is None or x is None or ratios[s] != ratios[i] or ratios[x] != ratios[i]:
                raise ValueError(f"layer {i}: no kv/index source of ratio {ratios[i]}")

        engram_rows = dict(zip(args.engram_layer_ids, args.engram_num_embeddings))
        self.embed = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList(Block(args, i, engram_rows.get(i)) for i in range(n))
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.head = nn.Module()
        self.head.weight = nn.Parameter(torch.zeros(args.vocab_size, args.dim))
        self.hc = args.hc_mult
        self.layout = CacheLayout.build(args, block_size, cache_dtype)
        self.engram_index = {lid: k for k, lid in enumerate(args.engram_layer_ids)}

    # -- dtypes ------------------------------------------------------------------------
    def set_dtype(self, dtype: torch.dtype) -> "DeepseekV41Model":
        for name, p in self.named_parameters():
            if not p.is_floating_point() or p.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
                continue
            keep32 = name.endswith(_FP32_SUFFIXES) or (
                ".compressor.w" in name and self.args.compress_ratios[int(name.split(".")[1])] > 1)
            p.data = p.data.to(torch.float32 if keep32 else dtype)
        return self

    # -- caches ------------------------------------------------------------------------
    def get_kv_spec(self) -> KVSpec:
        lay = self.layout
        return KVSpec(layers=[], paged_layers=[
            PagedLayerSpec(name=WINDOW_LAYER, block_size=lay.window_block,
                           page_elems=lay.window_page_elems, dtype=torch.float32,
                           sliding_window=lay.window_size),
            PagedLayerSpec(name=COMPRESSED_LAYER, block_size=lay.comp_block,
                           page_elems=lay.comp_page_elems, dtype=lay.comp_dtype),
        ])

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]) -> None:
        lay = self.layout
        (w,), (c,) = kv_caches[WINDOW_LAYER], kv_caches[COMPRESSED_LAYER]
        if w.shape[1:] != (lay.window_page_elems,) or c.shape[1:] != (lay.comp_page_elems,):
            raise ValueError(f"bound pages {tuple(w.shape)} / {tuple(c.shape)} do not match the "
                             f"layout ({lay.window_page_elems}, {lay.comp_page_elems})")
        if w.dtype != torch.float32 or c.dtype != lay.comp_dtype or w.shape[0] != c.shape[0]:
            raise ValueError("window pages must be float32 and both views must cover one buffer")
        self.caches = Caches(lay, w, c)

    # -- forward -------------------------------------------------------------------------
    def forward(self, input_ids, positions, attn_metadata, engram_ids=None):
        """``[tokens] -> [tokens, dim]`` hidden states after the final hc collapse and norm.

        ``engram_ids [tokens, n_engram_layers, cols]`` are the n-gram hash rows,
        computed on the host (``engram.EngramHasher``): the hash needs exact 64-bit
        integer arithmetic.
        """
        if not hasattr(self, "caches"):
            raise RuntimeError("bind_kv_cache() has not been called")
        step = build_step(self.layout, self.caches.num_pages, positions, attn_metadata)
        n, T = step.n, step.T
        h = self.embed(input_ids.view(n, T))
        h = h.unsqueeze(2).repeat(1, 1, self.hc, 1)
        pre_mix = torch.zeros(n, T, self.hc, dtype=torch.float32, device=h.device)
        pre_mix[..., 0] = 1.0
        if self.engram_index and engram_ids is None:
            raise ValueError("this config has engram layers: pass engram_ids")
        if engram_ids is not None:
            engram_ids = engram_ids.view(n, T, len(self.engram_index), -1)
        shared = {"fresh": {}, "kv_source": self.kv_source, "index_owner": self.index_owner}
        for layer in self.layers:
            ids = None
            if layer.engram is not None:
                ids = engram_ids[:, :, self.engram_index[layer.layer_id]]
            h, pre_mix = layer(h, pre_mix, step, self.caches, shared, ids)
        h = Block.hc_pre(h, pre_mix)
        return self.norm(h).view(n * T, -1)

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden.float(), self.head.weight.float())

    # -- weights -------------------------------------------------------------------------
    def load_reference_state_dict(self, sd: dict, strict: bool = True):
        """Load reference-named, dequantized tensors; stacks the routed experts."""
        sd = dict(sd)
        E = self.args.n_routed_experts
        for i, layer in enumerate(self.layers):
            p = f"layers.{i}.ffn.experts"
            w1 = [sd.pop(f"{p}.{e}.w1.weight") for e in range(E)]
            w3 = [sd.pop(f"{p}.{e}.w3.weight") for e in range(E)]
            w2 = [sd.pop(f"{p}.{e}.w2.weight") for e in range(E)]
            sd[f"layers.{i}.ffn.gate_up_proj"] = torch.stack(
                [torch.stack([a.T, b.T], dim=1) for a, b in zip(w1, w3)])
            sd[f"layers.{i}.ffn.down_proj"] = torch.stack([w.T for w in w2])
        own = self.state_dict()
        for k, v in sd.items():
            if k in own:
                sd[k] = v.to(own[k].dtype)
        return self.load_state_dict(sd, strict=strict)
