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

try:
    from vllm_neuron.functional.topk import topk as _nf_topk
except ImportError:  # pragma: no cover - hosts without nki (the oracle laptop)
    _nf_topk = None


def _idiv(x: torch.Tensor, d: int) -> torch.Tensor:
    """Integer division of a NON-NEGATIVE tensor. Truncating division lowers to an integer
    divide; floor division and ``remainder`` on int64 lower through float64, which
    neuronx-cc rejects (NCC_ESPP004). For non-negative operands the two agree."""
    return torch.div(x, d, rounding_mode="trunc")


def _imod(x: torch.Tensor, d: int) -> torch.Tensor:
    """``x mod d`` for a NON-NEGATIVE tensor; see ``_idiv``."""
    return torch.fmod(x, d)


def topk_indices(x: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the ``k`` largest along the last dim.

    ``torch.topk`` lowers to an HLO sort, which trn2 rejects (NCC_EVRF029); the plugin's
    rotational NKI top-k compiles at every V4.1 shape (6 of 384, 512 of up to 131k, 2048
    of 16k blocks) and falls back to ``torch.topk`` off device.
    """
    if _nf_topk is None:
        return x.topk(k, dim=-1).indices
    return _nf_topk(x, k, dim=-1, gather_dim=-1)[1].to(torch.long)


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


@dataclasses.dataclass(frozen=True)
class Parallel:
    """Where this rank sits.

    ``tp`` ranks share the attention: ``n_heads / tp`` query heads each, with the MQA
    caches, the compressor and the indexer replicated (one KV head cannot be split).
    The routed experts are ``ep`` groups x ``etp`` intermediate shards, ``ep * etp ==
    tp``. Every sublayer ends in one all-reduce over all ``tp`` ranks.

    ``group``: anything with vLLM ``GroupCoordinator``'s ``all_reduce(t) -> t`` and
    ``all_gather(t, dim)``.
    """

    tp: int = 1
    rank: int = 0
    ep: int = 1
    ep_rank: int = 0
    etp: int = 1
    etp_rank: int = 0
    group: object = None

    def __post_init__(self):
        if self.ep * self.etp != self.tp:
            raise ValueError(f"EP {self.ep} x expert TP {self.etp} != TP {self.tp}")

    def all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        return t if self.tp == 1 else self.group.all_reduce(t)

    def all_gather(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        return t if self.tp == 1 else self.group.all_gather(t, dim=dim)


def _part(n: int, parts: int, what: str) -> int:
    if n % parts:
        raise ValueError(f"{what}={n} does not divide over {parts} ranks")
    return n // parts


@functools.lru_cache(maxsize=8)
def rope_freqs(dim: int, original_seq_len: int, base: float, factor: float,
               beta_fast: int, beta_slow: int) -> torch.Tensor:
    """The reference's ``precompute_freqs_cis`` without the position table: fp32 [dim/2].
    Always on CPU, so a model built on the meta device still gets real frequencies."""
    cpu = torch.device("cpu")
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=cpu) / dim))
    if original_seq_len > 0:
        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32, device=cpu) - low)
                / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return freqs


def rope_cos_sin(freqs: torch.Tensor, pos: torch.Tensor):
    """``pos`` any integer shape -> cos, sin of shape ``pos.shape + [dim/2]``, fp32.

    The angle is ``float32(pos) * freq``, as the reference's ``torch.outer`` computes it.
    ``freqs`` must already be on ``pos``'s device: a ``.to(device)`` inside a compiled
    Neuron graph is an unimplemented cross-device copy.
    """
    ang = pos.to(torch.float32).unsqueeze(-1) * freqs
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
    lens = torch.as_tensor(compress_lens)
    last = torch.where(lens > 0, _idiv((lens - 1).clamp_min(0), block_size), torch.full_like(lens, -1))
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, torch.inf)
    top = topk_indices(scores, min(topk_blocks, num_blocks))
    # isneginf, not ``> -inf``: torch_xla promotes a tensor compared with a Python float
    # scalar to f64, which neuronx-cc rejects (NCC_ESPP004)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, top, ~torch.isneginf(scores.gather(-1, top)))
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
    # this rank as a 0-d tensor: rank-dependent offsets read it, so all ranks trace one
    # graph instead of baking 64 different constants
    rank: torch.Tensor = None

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
        page = self.c_blocks.index_select(1, _idiv(j, per))
        return page, _imod(j, per).expand_as(page)


def build_step(lay: CacheLayout, num_pages: int, positions: torch.Tensor,
               attn_metadata: dict, rank: torch.Tensor) -> Step:
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
    slot_w = slot_w.clamp_min(0)
    w_page = torch.where(live, _idiv(slot_w, Bw), torch.full_like(slot_w, sink))
    w_tok = torch.where(live, _imod(slot_w, Bw), torch.zeros_like(slot_w))

    W = lay.window_size
    h_pos = s.unsqueeze(1) - W + torch.arange(W, device=pos.device)
    blocks_w = paged_block_ids(bt_w, live_rows, num_pages)
    off = mw.get("swa_kv_pos_offset") if decode else None
    base = (h_pos - off.to(torch.long).view(n, 1)) if off is not None else h_pos
    # negative only where h_pos < 0, which h_ok rejects; clamped so every address is in range
    bidx = _idiv(base.clamp_min(0), Bw)
    h_ok = (h_pos >= 0) & (bidx >= 0) & (bidx < blocks_w.shape[1]) & live_rows.view(n, 1)
    h_page = blocks_w.gather(1, bidx.clamp(0, blocks_w.shape[1] - 1))
    h_page = torch.where(h_ok, h_page, torch.full_like(h_page, zero_page))
    h_tok = _imod(h_pos.clamp_min(0), Bw)

    Bc = lay.comp_block
    slot_c = slot_c.clamp_min(0)
    c_page = torch.where(live, _idiv(slot_c, Bc), torch.full_like(slot_c, sink))
    c_off = _imod(slot_c, Bc)
    c_blocks = paged_block_ids(mc["block_table_tensor"], live_rows, num_pages)
    return Step(decode=decode, pos=pos, live=live, s=s, w_page=w_page, w_tok=w_tok,
                h_pos=h_pos, h_page=h_page, h_tok=h_tok, h_ok=h_ok, c_page=c_page,
                c_off=c_off, c_blocks=c_blocks, rank=rank.to(torch.long).reshape(()))


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
            closes = step.live[:, 0] & (_imod(p, r) == r - 1) & (prev[:, 0] >= 0)
            # only ratio 2 is released; a larger ratio would need r - 1 history rows
            if r != 2:
                raise NotImplementedError(f"compress ratio {r} in decode")
            kv2 = torch.cat([torch.where(closes.view(n, 1, 1), prev_kv, torch.zeros_like(prev_kv)), kv], 1)
            sc2 = torch.cat([torch.where(closes.view(n, 1, 1), prev_sc, torch.zeros_like(prev_sc)), score], 1)
            latent = (kv2 * sc2.softmax(dim=1)).sum(dim=1, keepdim=True)
            j0, valid = _idiv(p, r), closes.view(n, 1)
        else:
            if r != 2:
                raise NotImplementedError(f"compress ratio {r} in prefill")
            # pair tokens along the LAST dim and take the two-way softmax explicitly: the
            # unflatten/sum form fed the indexer's wk matmul a pattern neuronx-cc's
            # NeuronInstComb could not delinearize (NCC_INIC901) at the real shapes
            G = T // r
            kv2 = kv[:, :G * r].reshape(n, G, r * self.head_dim)
            sc2 = score[:, :G * r].reshape(n, G, r * self.head_dim)
            a, b = kv2[..., :self.head_dim], kv2[..., self.head_dim:]
            sa, sb = sc2[..., :self.head_dim], sc2[..., self.head_dim:]
            m = torch.maximum(sa, sb)
            ea, eb = torch.exp(sa - m), torch.exp(sb - m)
            latent = (a * ea + b * eb) / (ea + eb)
            valid = step.live[:, r - 1::r][:, :G] & step.live[:, 0::r][:, :G]
            j0 = _idiv(step.s, r)
        caches.write_window(f_kv, step.w_page, step.w_tok, kv)
        caches.write_window(f_sc, step.w_page, step.w_tok, score)
        return self.norm(latent.to(dtype)), j0, valid


class Indexer(nn.Module):
    """Scores compressed positions and keeps the best ``index_topk`` per query."""

    query_chunk = 128

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
        # fp32 projection (see Compressor.forward on NCC_INIC901); cast back after
        k = F.linear(latent.float(), self.wk.weight.float()).to(latent.dtype)
        return rope_tail(self.k_norm(k), cos, sin, self.rd)

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
        lens = _idiv(step.pos + 1, r)                                       # [n, T]
        reach = torch.arange(N, device=x.device) < lens.unsqueeze(-1)       # [n, T, N]
        k = torch.where(reach.any(dim=1).unsqueeze(-1), k, torch.zeros_like(k)).float()
        # per-head scores are [T, heads, N] before the head reduction: bound that by
        # taking the queries a chunk at a time (a static, unrolled loop)
        qf, chunks = q.float(), []
        for t0 in range(0, T, self.query_chunk):
            s = torch.einsum("nthd,nmd->nthm", qf[:, t0:t0 + self.query_chunk], k).relu()
            chunks.append((s * w[:, t0:t0 + self.query_chunk].unsqueeze(-1)).sum(dim=2))
        score = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
        score = score.masked_fill(~reach, float("-inf"))
        if self.is_candidate_source:
            shared["candidates"] = select_candidate_blocks(
                score, lens.unsqueeze(-1), self.candidate_topk_blocks, self.candidate_block_size)
        elif self.uses_candidates:
            score = score.masked_fill(~shared["candidates"], float("-inf"))
        idx = topk_indices(score, min(self.index_topk, N))
        _capture_tensor(f"layers.{self.layer_id}.indexer.topk", idx)
        return idx, idx < lens.unsqueeze(-1)


class Attention(nn.Module):
    """Sliding window of raw K plus, at ratio > 0, ``index_topk`` compressed entries, in
    one softmax with a per-head sink."""

    def __init__(self, args, layer_id: int, par: Parallel = Parallel()):
        super().__init__()
        self.layer_id = layer_id
        self.head_dim = args.head_dim
        self.rd = args.rope_head_dim
        self.o_lora_rank = args.o_lora_rank
        self.window_size = args.window_size
        self.ratio = args.compress_ratios[layer_id]
        self.softmax_scale = self.head_dim ** -0.5
        eps = args.norm_eps
        # Heads split over the TP ranks. wo_a is block-diagonal over o_groups: up to
        # o_groups ranks each own whole groups; past that each owns part of one group
        # and holds that group's wo_b columns, and the sublayer all-reduce sums the
        # partial group projections -- wo_b is linear, so the order does not matter.
        self.n_heads = _part(args.n_heads, par.tp, "n_heads")
        self.h0 = par.rank * self.n_heads
        per_group = _part(args.n_heads, args.o_groups, "n_heads per o_group")
        if self.n_heads >= per_group:
            self.n_groups = _part(self.n_heads, per_group, "local heads per o_group")
            self.group_heads = per_group
        else:
            _part(per_group, self.n_heads, "o_group heads per rank")
            self.n_groups, self.group_heads = 1, self.n_heads
        self.g0 = self.h0 // per_group
        self.attn_sink = nn.Parameter(torch.zeros(self.n_heads))
        self.wq_a = _linear(args.dim, args.q_lora_rank)
        self.q_norm = RMSNorm(args.q_lora_rank, eps)
        self.wq_b = _linear(args.q_lora_rank, self.n_heads * self.head_dim)
        self.wkv = _linear(args.dim, self.head_dim)
        self.kv_norm = RMSNorm(self.head_dim, eps)
        self.wo_a = _linear(self.group_heads * self.head_dim, self.n_groups * self.o_lora_rank)
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
        entry = torch.where(valid, _idiv(tok_off, r), torch.zeros_like(tok_off))
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
        return self.wo_b(o.flatten(2))                                     # partial under TP


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
        self.gate_temp = getattr(args, "gate_temp", 1.0)   # not in the released config
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
        idx = topk_indices(scores + self.bias.float(), self.topk)
        w = scores.gather(1, idx)
        if self.norm_topk_prob and self.topk > 1:
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
        return w * self.route_scale, idx


class MoE(nn.Module):
    """Routed experts in the NKI MoE layout plus one shared expert. Returns this rank's
    partial sum in fp32: its local experts' share and its slice of the shared expert."""

    def __init__(self, args, par: Parallel = Parallel()):
        super().__init__()
        if args.n_shared_experts != 1:
            raise NotImplementedError("exactly one shared expert")
        E, D, I = args.n_routed_experts, args.dim, args.moe_inter_dim
        self.E, self.limit = E, args.swiglu_limit
        self.E_local = _part(E, par.ep, "n_routed_experts")
        self.e0 = par.ep_rank * self.E_local          # load time only; forward reads the rank tensor
        self.etp = par.etp
        self.I_local = _part(I, par.etp, "moe_inter_dim")
        self.gate = Gate(args)
        self.gate_up_proj = nn.Parameter(torch.zeros(self.E_local, D, 2, self.I_local))
        self.down_proj = nn.Parameter(torch.zeros(self.E_local, self.I_local, D))
        self.shared_experts = Expert(D, _part(I, par.tp, "moe_inter_dim"), args.swiglu_limit)

    def _routed_dense(self, x, w, idx, rank):
        """Every local expert on every token, zero weight off the top-k: exact and static.
        Clamping per intermediate shard is exact: the clamps are elementwise."""
        dense = torch.zeros(x.shape[0], self.E, dtype=torch.float32, device=x.device)
        local = _idiv(rank, self.etp) * self.E_local + torch.arange(self.E_local, device=x.device)
        dense = dense.scatter(1, idx, w).index_select(1, local)
        gu = torch.einsum("td,edgi->tegi", x, self.gate_up_proj)
        h = _clamped_swiglu(gu[:, :, 0].float(), gu[:, :, 1].float(), self.limit)
        h = (h * dense.unsqueeze(-1)).to(x.dtype)
        return torch.einsum("tei,eid->td", h, self.down_proj).float()

    def forward(self, x, rank):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        w, idx = self.gate(x)
        y = self._routed_dense(x, w, idx, rank)
        y = y + self.shared_experts(x).float()
        return y.view(*shape[:-1], -1)


# ------------------------------------------------------------------------ engram
def _byte_tables():
    """Exact fp32 values of every e4m3fn and e8m0 byte, built on the CPU."""
    b = torch.arange(256, dtype=torch.int32, device="cpu").to(torch.uint8)
    return b.view(torch.float8_e4m3fn).float(), b.view(torch.float8_e8m0fnu).float()


class EngramEmbedding(nn.Module):
    """The n-gram table, FP8 e4m3 rows with a per-32 e8m0 scale, dequantized to bf16 on
    lookup. Rows split ``ceil(rows / tp)`` per rank; a row off this rank reads as zero and
    the all-reduce sums in the one real copy, exactly.

    Stored as raw bytes and decoded through 256-entry tables: neuronx-cc has no e8m0
    type, and trn2's e4m3 is not the OCP e4m3fn the checkpoint uses, so no fp8 dtype may
    appear in the graph.
    """

    def __init__(self, rows: int, dim: int, par: Parallel = Parallel(), block: int = 32):
        super().__init__()
        self.block, self.par = block, par
        self.rows_local = -(-rows // par.tp)
        self.r0 = par.rank * self.rows_local          # load time only; forward reads the rank tensor
        self.weight = nn.Parameter(torch.zeros(self.rows_local, dim, dtype=torch.uint8),
                                   requires_grad=False)
        self.scale = nn.Parameter(torch.zeros(self.rows_local, dim // block, dtype=torch.uint8),
                                  requires_grad=False)
        e4m3, e8m0 = _byte_tables()
        self.register_buffer("e4m3", e4m3, persistent=False)
        self.register_buffer("e8m0", e8m0, persistent=False)

    def forward(self, ids, rank):
        local = ids - rank * self.rows_local
        off = (local < 0) | (local >= self.rows_local)
        local = torch.where(off, torch.zeros_like(local), local)
        v = self.e4m3[F.embedding(local, self.weight).long()]
        s = self.e8m0[F.embedding(local, self.scale).long()]
        v = v.unflatten(-1, (-1, self.block)) * s.unsqueeze(-1)
        v = v.flatten(-2).to(torch.bfloat16)
        v = torch.where(off.unsqueeze(-1), torch.zeros_like(v), v)
        return self.par.all_reduce(v)


class Engram(nn.Module):
    """Gated n-gram lookup added to every hc copy of the residual stream."""

    def __init__(self, args, layer_id: int, rows: int, par: Parallel = Parallel()):
        super().__init__()
        self.layer_id = layer_id
        self.dim, self.hc = args.dim, args.hc_mult
        self.eps = args.norm_eps
        self.par = par
        cols = (args.engram_max_ngram_size - 1) * args.engram_n_heads
        self.embed = EngramEmbedding(rows, args.engram_head_dim, par)
        # output rows split over ranks, gathered back: 25600 x 6144 is too big to replicate
        out = _part(args.dim * (args.hc_mult + 1), par.tp, "engram wkv rows")
        self.wkv = _linear(cols * args.engram_head_dim, out)
        self.q_weight = nn.Parameter(torch.ones(args.hc_mult, args.dim))
        self.k_weight = nn.Parameter(torch.ones(args.hc_mult, args.dim))

    def forward(self, h, hash_ids, rank):
        """h ``[..., hc, dim]``; hash_ids ``[..., cols]``."""
        emb = self.embed(hash_ids, rank).flatten(-2).to(self.wkv.weight.dtype)
        kv = self.par.all_gather(self.wkv(emb), dim=-1)
        key, value = kv.split([self.hc * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc, self.dim))
        weight = self.q_weight.float() * self.k_weight.float()
        hf = h.float()
        rstd = torch.rsqrt(hf.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (hf * weight * key).sum(-1) * rstd * self.dim ** -0.5
        # signed sqrt; a select rather than torch.copysign, which lowers to a custom call
        # neuronx-cc rejects (differs only at dot == -0.0)
        root = dot.abs().clamp_min(1e-6).sqrt()
        gate = torch.sigmoid(torch.where(dot < 0, -root, root))
        return (hf + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(h.dtype)


# ------------------------------------------------------------------------- block
class Block(nn.Module):
    """mHC block. The pre-mix a sublayer computes is used by the NEXT sublayer."""

    def __init__(self, args, layer_id: int, engram_rows: int | None, par: Parallel = Parallel()):
        super().__init__()
        self.layer_id = layer_id
        self.par = par
        self.norm_eps = args.norm_eps
        self.hc, self.iters, self.hc_eps = args.hc_mult, args.hc_sinkhorn_iters, args.hc_eps
        self.attn = Attention(args, layer_id, par)
        self.ffn = MoE(args, par)
        self.engram = Engram(args, layer_id, engram_rows, par) if engram_rows is not None else None
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
            h = self.engram(h, engram_ids, step.rank)
        residual = h
        a_pre, a_post, a_comb = self.hc_mixes(h, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm(self.hc_pre(h, pre_mix))
        # partial sums reduced in fp32, as the reference's RowParallelLinear and MoE do
        x = self.par.all_reduce(self.attn(x, step, caches, shared).float()).to(h.dtype)
        h = self.hc_post(x, residual, a_post, a_comb)
        residual = h
        f_pre, f_post, f_comb = self.hc_mixes(h, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm(self.hc_pre(h, a_pre))
        x = self.par.all_reduce(self.ffn(x, step.rank)).to(h.dtype)
        h = self.hc_post(x, residual, f_post, f_comb)
        return h, f_pre


# ------------------------------------------------------------------------ model
# Parameters the reference stores in float32 whatever the model dtype.
_FP32_SUFFIXES = ("attn_sink", "gate.bias", "hc_attn_fn", "hc_ffn_fn", "hc_attn_base",
                  "hc_ffn_base", "hc_attn_scale", "hc_ffn_scale", "head.weight")


class DeepseekV41Model(nn.Module):
    """The decoder. Parameter names equal the reference's (``embed``, ``layers``, ``norm``,
    ``head``), experts excepted; shapes are this rank's (see ``Parallel``)."""

    kv_cache_page_major = True

    def __init__(self, args, block_size: int = 32, cache_dtype: torch.dtype = torch.bfloat16,
                 par: Parallel = Parallel()):
        super().__init__()
        self.args, self.par = args, par
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

        self.vocab_local = _part(args.vocab_size, par.tp, "vocab_size")
        self.v0 = par.rank * self.vocab_local
        engram_rows = dict(zip(args.engram_layer_ids, args.engram_num_embeddings))
        self.embed = nn.Module()
        self.embed.weight = nn.Parameter(torch.zeros(self.vocab_local, args.dim))
        self.layers = nn.ModuleList(Block(args, i, engram_rows.get(i), par) for i in range(n))
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.head = nn.Module()
        self.head.weight = nn.Parameter(torch.zeros(self.vocab_local, args.dim))
        self.hc = args.hc_mult
        self.layout = CacheLayout.build(args, block_size, cache_dtype)
        self.engram_index = {lid: k for k, lid in enumerate(args.engram_layer_ids)}

    # -- dtypes ------------------------------------------------------------------------
    def keeps_fp32(self, name: str) -> bool:
        return name.endswith(_FP32_SUFFIXES) or (
            ".compressor.w" in name and self.args.compress_ratios[int(name.split(".")[1])] > 1)

    def set_dtype(self, dtype: torch.dtype) -> "DeepseekV41Model":
        for name, p in self.named_parameters():
            if not p.is_floating_point() or p.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
                continue
            p.data = p.data.to(torch.float32 if self.keeps_fp32(name) else dtype)
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
        # plain tensor attributes, as llama binds k_cache: the runner's graph-capture
        # trace swaps those for meta tensors, and would miss tensors inside a dataclass
        self.window_pages, self.comp_pages = w, c

    @property
    def caches(self) -> Caches:
        return Caches(self.layout, self.window_pages, self.comp_pages)

    # -- forward -------------------------------------------------------------------------
    def embed_tokens(self, ids: torch.Tensor, rank: torch.Tensor | None = None) -> torch.Tensor:
        """Vocab-sharded lookup: an id off this rank reads row 0 and is zeroed."""
        rank = self._rank(rank, ids.device)
        local = ids - rank * self.vocab_local
        off = (local < 0) | (local >= self.vocab_local)
        h = F.embedding(torch.where(off, torch.zeros_like(local), local), self.embed.weight)
        return self.par.all_reduce(torch.where(off.unsqueeze(-1), torch.zeros_like(h), h))

    def _rank(self, rank, device) -> torch.Tensor:
        if rank is None:
            rank = torch.tensor(self.par.rank, device=device)
        return rank.to(torch.long).reshape(())

    def hidden_states(self, input_ids, positions, attn_metadata, engram_ids=None, rank=None):
        """``[tokens] -> [tokens, dim]`` hidden states after the final hc collapse and norm.

        ``engram_ids [tokens, n_engram_layers, cols]`` are the n-gram hash rows,
        computed on the host (``engram.EngramHasher``): the hash needs exact 64-bit
        integer arithmetic.
        """
        if getattr(self, "window_pages", None) is None:
            raise RuntimeError("bind_kv_cache() has not been called")
        rank = self._rank(rank, input_ids.device)
        caches = self.caches
        step = build_step(self.layout, caches.num_pages, positions, attn_metadata, rank)
        n, T = step.n, step.T
        h = self.embed_tokens(input_ids.view(n, T), rank)
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
            h, pre_mix = layer(h, pre_mix, step, caches, shared, ids)
        h = Block.hc_pre(h, pre_mix)
        return self.norm(h).view(n * T, -1)

    def compute_logits(self, hidden: torch.Tensor, gather: bool = True) -> torch.Tensor:
        """fp32 logits; this rank's vocab slice unless ``gather``."""
        local = F.linear(hidden.float(), self.head.weight.float())
        return self.par.all_gather(local, dim=-1) if gather else local

    # -- the runner's surface --------------------------------------------------------------
    on_device_sampling_config = None
    _gather_logits = False

    def attach_sampler(self, neuron_config) -> None:
        """On-device sampling over the vocab-sharded logits, as GLM and Qwen3.5 do."""
        self.on_device_sampling_config = getattr(neuron_config, "on_device_sampling_config", None)
        self._gather_logits = neuron_config is not None and (
            getattr(neuron_config, "max_logprobs", 0) != 0
            or getattr(neuron_config, "debug_logits_dir", None) is not None)
        if self.on_device_sampling_config is not None:
            from vllm_neuron.nn.sampler import Sampler

            group = getattr(self.par.group, "device_group", None)
            self.sampler = Sampler(self.on_device_sampling_config, process_group=group)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    @torch.no_grad()
    def forward(self, input_ids, positions, rotary_position_ids=None, attn_metadata=None,
                sampling_positions=None, sampling_params=None, spec_decode_metadata=None,
                logit_mask=None, rank=None, engram_ids=None, **_unused):
        if spec_decode_metadata is not None:
            raise NotImplementedError("speculative decoding (MTP / DSpark) is out of scope")
        hidden = self.hidden_states(input_ids, positions, attn_metadata, engram_ids, rank)
        if sampling_positions is not None:
            hidden = torch.index_select(hidden, 0, sampling_positions)
        if self.on_device_sampling_config is None:
            return self.compute_logits(hidden)
        logits = self.compute_logits(hidden, gather=False)
        gathered = self.par.all_gather(logits, dim=-1) if self._gather_logits else None
        sampled = self.sampler(logits, sampling_params, logit_mask=logit_mask, tp_rank=rank)
        return sampled, gathered

    # -- Engram ids, computed on the host by the runner -------------------------------------
    @property
    def has_engram(self) -> bool:
        return bool(self.engram_index)

    def engram_cols(self) -> int:
        a = self.args
        return (a.engram_max_ngram_size - 1) * a.engram_n_heads

    def engram_dummy_ids(self, num_tokens: int) -> torch.Tensor:
        return torch.zeros(num_tokens, len(self.engram_index), self.engram_cols(), dtype=torch.int32)

    def set_engram_hasher(self, hasher) -> None:
        self.engram_hasher = hasher

    def engram_host_ids(self, token_ids: torch.Tensor, rows: torch.Tensor,
                        positions: torch.Tensor) -> torch.Tensor:
        """``token_ids [max_reqs, max_len]`` (the runner's ``token_ids_cpu``), and per
        scheduled token its request row (-1 for a padded row) and position ->
        ``[tokens, n_engram_layers, cols]`` int32. Padded rows hash position 0 of row 0;
        their output is discarded."""
        hasher = self.engram_hasher
        n = hasher.n
        dead = rows < 0
        rows = rows.clamp_min(0).to(torch.long)
        pos = torch.where(dead, torch.zeros_like(positions), positions).to(torch.long)
        shifts = torch.arange(n)
        src = (pos.unsqueeze(1) - shifts).clamp_min(0)
        window = token_ids[rows.unsqueeze(1), src]                         # [T, n] raw ids
        return hasher.hash_windows(window, pos).to(torch.int32)

    # -- weights -------------------------------------------------------------------------
    def local_tensor(self, name: str, src) -> torch.Tensor:
        """This rank's value of parameter ``name``, read from ``src`` (``DictSource`` /
        ``CheckpointSource``), which serves reference-named, dequantized slices."""
        a, par = self.args, self.par
        parts = name.split(".")
        if name in ("embed.weight", "head.weight"):
            return src.get(name, rows=(self.v0, self.v0 + self.vocab_local))
        if parts[0] != "layers":
            return src.get(name)
        i, rest = int(parts[1]), ".".join(parts[2:])
        layer = self.layers[i]
        pre = f"layers.{i}."
        at, Dh, R = layer.attn, a.head_dim, a.o_lora_rank
        if rest == "attn.attn_sink":
            return src.get(name, rows=(at.h0, at.h0 + at.n_heads))
        if rest == "attn.wq_b.weight":
            return src.get(name, rows=(at.h0 * Dh, (at.h0 + at.n_heads) * Dh))
        if rest == "attn.wo_a.weight":
            off = at.h0 % (a.n_heads // a.o_groups)
            return src.get(name, rows=(at.g0 * R, (at.g0 + at.n_groups) * R),
                           cols=(off * Dh, (off + at.group_heads) * Dh))
        if rest == "attn.wo_b.weight":
            return src.get(name, cols=(at.g0 * R, (at.g0 + at.n_groups) * R))
        if rest.startswith("ffn.shared_experts."):
            I_s = layer.ffn.shared_experts.w1.weight.shape[0]
            span = (par.rank * I_s, (par.rank + 1) * I_s)
            return src.get(name, cols=span) if ".w2." in rest else src.get(name, rows=span)
        if rest in ("ffn.gate_up_proj", "ffn.down_proj"):
            moe = layer.ffn
            span = (par.etp_rank * moe.I_local, (par.etp_rank + 1) * moe.I_local)
            experts = range(moe.e0, moe.e0 + moe.E_local)
            e = f"{pre}ffn.experts"
            if rest == "ffn.down_proj":
                return torch.stack([src.get(f"{e}.{x}.w2.weight", cols=span).T for x in experts])
            return torch.stack([torch.stack([src.get(f"{e}.{x}.w1.weight", rows=span).T,
                                             src.get(f"{e}.{x}.w3.weight", rows=span).T], dim=1)
                                for x in experts])
        if rest in ("engram.embed.weight", "engram.embed.scale"):
            emb = layer.engram.embed
            t = src.get(name, rows=(emb.r0, emb.r0 + emb.rows_local), pad=True)
            return t.contiguous().view(torch.uint8)               # bit-exact bytes
        if rest == "engram.wkv.weight":
            out = layer.engram.wkv.weight.shape[0]
            return src.get(name, rows=(par.rank * out, (par.rank + 1) * out))
        return src.get(name)

    def load_from(self, src, device=None) -> None:
        own = dict(self.named_parameters())
        sd = {}
        for name, p in own.items():
            t = self.local_tensor(name, src)
            if tuple(t.shape) != tuple(p.shape):
                raise ValueError(f"{name}: source gives {tuple(t.shape)}, rank expects {tuple(p.shape)}")
            sd[name] = t.to(dtype=p.dtype, device=device or p.device).contiguous()
        self.load_state_dict(sd, strict=True, assign=True)
        if device is not None:
            for m in self.modules():
                for k, b in list(m._buffers.items()):
                    if b is not None:
                        m._buffers[k] = b.to(device)

    def load_reference_state_dict(self, sd: dict) -> None:
        """Tests: a full reference-named, dequantized state dict."""
        self.load_from(DictSource(sd))

    def load_weights(self, checkpoint_path: str, device: torch.device, cache_dir: str | None = None):
        """The runner's entry point: this rank's slices of the HF checkpoint, dequantized."""
        from .weights import Checkpoint

        with Checkpoint(checkpoint_path) as ckpt:
            self.load_from(CheckpointSource(ckpt), device)
        if self.has_engram:
            from .engram import hasher_for

            self.set_engram_hasher(hasher_for(self.args, checkpoint_path))


def _span(n: int, span):
    return (0, n) if span is None else span


class DictSource:
    """Slices of full, already-dequantized tensors."""

    def __init__(self, sd: dict):
        self.sd = sd

    def get(self, name, rows=None, cols=None, pad=False):
        t = self.sd[name]
        r0, r1 = _span(t.shape[0], rows)
        part = t[r0:min(r1, t.shape[0])]
        if cols is not None:
            part = part[:, cols[0]:cols[1]]
        if pad and part.shape[0] < r1 - r0:
            part = torch.cat([part, part.new_zeros(r1 - r0 - part.shape[0], *part.shape[1:])])
        return part


class CheckpointSource:
    """Slices of the HF checkpoint, read lazily: an FP8 weight reads only the 32x32
    blocks covering the slice, so no rank materialises a tensor it does not need."""

    def __init__(self, ckpt):
        from . import weights as W

        self.ckpt, self.W = ckpt, W

    def get(self, name, rows=None, cols=None, pad=False):
        W, ck = self.W, self.ckpt
        e = ck.plan.entries[name]
        if e.action == W.DEQUANT_FP8:
            ws, ss = ck.raw_slice(e.source), ck.raw_slice(e.scale)
            n, k = ws.get_shape()
            (r0, r1), (c0, c1) = _span(n, rows), _span(k, cols)
            bn, bk = W.FP8_BLOCK
            R0, C0 = r0 // bn * bn, c0 // bk * bk
            R1, C1 = min(-(-r1 // bn) * bn, n), min(-(-c1 // bk) * bk, k)
            w = W.dequant_fp8_block(ws[R0:R1, C0:C1], ss[R0 // bn:-(-R1 // bn), C0 // bk:-(-C1 // bk)],
                                    name=name)
            return w[r0 - R0:r1 - R0, c0 - C0:c1 - C0]
        if e.action == W.DEQUANT_MXFP4:
            full = W.dequant_mxfp4(ck.raw(e.source), ck.raw(e.scale), name=name)
            return DictSource({name: full}).get(name, rows, cols)
        sl = ck.raw_slice(e.source)
        shape = sl.get_shape()
        r0, r1 = _span(shape[0], rows)
        if cols is None:
            part = sl[r0:min(r1, shape[0])]
        else:
            part = sl[r0:min(r1, shape[0]), cols[0]:cols[1]]
        if pad and part.shape[0] < r1 - r0:
            part = torch.cat([part, torch.zeros(r1 - r0 - part.shape[0], *part.shape[1:],
                                                dtype=part.dtype)])
        return part.float() if e.action == W.TO_FP32 else part


def from_configs(hf_config, text_neuron_config=None, **_):
    """The runner's constructor (reached through ``factory.DeepseekV41ForCausalLM``).

    Called under ``torch.device("meta")``; ``load_weights`` materialises. The window block
    is vLLM's cache block size; the compressed block follows from it (``cache_layout``).
    """
    from vllm.config import get_current_vllm_config
    from vllm.distributed.parallel_state import get_tp_group

    from .config import DeepseekV41TextArgs

    args = DeepseekV41TextArgs.from_hf_config(hf_config)
    tp = get_tp_group()
    world, rank = tp.world_size, tp.rank_in_group
    ep = getattr(text_neuron_config, "ep_degree", 1) or 1
    if ep > 1:
        # [unverified on device] the plugin's EP coordinates, as gpt_oss and GLM read them
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_ep_degree, get_neuron_ep_rank, get_neuron_ep_tp_group)
        ep_tp = get_neuron_ep_tp_group()
        par = Parallel(tp=world, rank=rank, ep=get_neuron_ep_degree(), ep_rank=get_neuron_ep_rank(),
                       etp=ep_tp.world_size, etp_rank=ep_tp.rank_in_group, group=tp)
    else:
        par = Parallel(tp=world, rank=rank, ep=1, ep_rank=0, etp=world, etp_rank=rank, group=tp)
    vc = get_current_vllm_config()
    dtype = vc.model_config.dtype
    model = DeepseekV41Model(args, block_size=vc.cache_config.block_size, cache_dtype=dtype, par=par)
    model.set_dtype(dtype)
    model.attach_sampler(text_neuron_config)
    if model.has_engram and vc.scheduler_config.async_scheduling:
        raise NotImplementedError(
            "DeepSeek-V4.1's Engram hashes the request's newest tokens on the host, which "
            "asynchronous scheduling has not written yet; serve with --no-async-scheduling")
    return model
