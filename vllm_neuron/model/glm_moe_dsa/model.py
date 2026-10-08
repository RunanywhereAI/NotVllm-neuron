# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 (``glm_moe_dsa``) text decoder for the Neuron plugin.

DeepSeek-V3.2's shape -- MLA with a 64-wide RoPE half, a DSA lightning indexer, 256
routed experts -- plus GLM's cross-layer top-k sharing: only the ``"full"`` layers of
``indexer_types`` own an indexer, and each ``"shared"`` layer attends to the previous
full layer's selection. Reference: transformers ``models/glm_moe_dsa`` (5.15+).

**One step, two shapes** (as ``deepseek_v41``). Prefill is one sequence of ``T``
bucket-padded tokens, possibly continuing a cached prefix; decode is ``n`` sequences of
one token. Both run as ``[n, T]`` against the whole context the block table addresses
(``C = blocks * block_size``), read from the paged cache with this step's fresh rows
substituted, so no read depends on whether this step's writes are visible yet.

**Attention is absorbed** and computed inline in fp32: ``q_nope @ W_uk`` scores
against the 512-wide latent, the RoPE halves add, and ``W_uv`` lifts the mixed latent
after the softmax. The cached row is 576 wide (latent + RoPE'd key), which fails the
fused TKG kernel's ``d_head`` checks -- irrelevant here, no fused kernel is called.

**Sparse selection.** A full layer's indexer scores every context position and keeps
the top ``index_topk`` (2048) among the causal ones. When the context the block table
can address is no wider than ``index_topk`` the selection is every causal position, so
no top-k is run at all (static: decided from shapes at trace time).

**FP8.** The checkpoint is e4m3 with one fp32 ``scale_inv`` per 128x128 block. Routed
experts and the one replicated FP8 projection (the indexer's ``wk``) stay FP8 on device
and are dequantized blockwise in the graph; everything TP-sharded is dequantized to the
model dtype at load. ``q_a_proj``, ``kv_a_proj_with_mqa`` and the indexer's ``wq_b`` are
output-sharded and all-gathered rather than replicated (1.2 GB per rank at full depth). trn2's e4m3
tops out at 240 while the checkpoint uses e4m3fn's 448, so every block whose values
exceed 240 is stored halved with its scale doubled (``fp8_le240``): exact except the
odd-LSB subnormals of such a block, which lose one bit (measured and logged at load).

**Norm eps.** ``q_a_layernorm`` / ``kv_a_layernorm`` use ``rms_norm_eps`` (1e-5) as vLLM's
``deepseek_v2`` does; transformers builds them at RMSNorm's default 1e-6. The references
disagree (~1e-5 relative on logits); this follows vLLM, GLM's serving reference.

Device rules followed (see ``deepseek_v41``): no ``.to(device)`` in the graph,
``torch.topk`` only through ``topk_indices`` (NKI), truncating integer division, no
tensor-vs-Python-float comparison, caches as plain module attributes, rank as a tensor.
"""

from __future__ import annotations

import dataclasses
import functools
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_neuron.functional.vendored_kernels.latent_cache_write import write_cache_rows
from vllm_neuron.model.kv_cache import KVSpec, PagedLayerSpec, paged_block_ids, reserved_pages

from .config import GlmMoeDsaArgs

logger = logging.getLogger(__name__)

try:
    from vllm_neuron.accuracy.tensor_capture import capture_tensor as _capture_tensor
except ImportError:  # pragma: no cover - hosts without vLLM
    def _capture_tensor(name, tensor):  # type: ignore[misc]
        return None

try:
    from vllm_neuron.functional.topk import topk as _nf_topk
except ImportError:  # pragma: no cover - hosts without nki
    _nf_topk = None

CACHE_LAYER = "glm_moe_dsa.cache"
FP8_MAX = 240.0   # trn2 e4m3


def _idiv(x: torch.Tensor, d: int) -> torch.Tensor:
    """``x // d`` for a NON-NEGATIVE tensor (int64 floor div lowers through f64)."""
    return torch.div(x, d, rounding_mode="trunc")


def _imod(x: torch.Tensor, d: int) -> torch.Tensor:
    return torch.fmod(x, d)


def topk_indices(x: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the ``k`` largest along the last dim, any order. ``torch.topk`` lowers to
    an HLO sort, which trn2 rejects; the plugin's NKI top-k falls back off device."""
    if _nf_topk is None:
        return x.topk(k, dim=-1).indices
    return _nf_topk(x, k, dim=-1, gather_dim=-1)[1].to(torch.long)


def fusion_barrier(x: torch.Tensor, dim: int, anchor: torch.Tensor) -> torch.Tensor:
    """``x`` unchanged, through a gather neuronx-cc cannot fold (NCC_INIC901 guard)."""
    zero = anchor.reshape(-1)[:1].to(torch.long).clamp(max=0)
    idx = torch.arange(x.shape[dim], device=x.device) + zero
    return x.index_select(dim, idx)


def _part(n: int, parts: int, what: str) -> int:
    if n % parts:
        raise ValueError(f"{what}={n} does not divide over {parts} ranks")
    return n // parts


# ------------------------------------------------------------------------- basic ops
class RMSNorm(nn.Module):
    """transformers' ``GlmMoeDsaRMSNorm``: normalise in fp32, cast, then scale."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        return self.weight * xf.to(x.dtype)


def fp8_le240(q: torch.Tensor, s: torch.Tensor, block: tuple[int, int]):
    """Block FP8 (``q`` e4m3fn ``[..., N, K]``, ``s`` fp32 ``[..., ceil(N/bn), K/bk]``) -> the
    same values with every block's ``|q| <= 240``: blocks above are halved, scale doubled.
    ``N`` may be ragged (``kv_a_proj_with_mqa`` is 576 = 4.5 x 128 rows). Load time only."""
    bn, bk = block
    *lead, N, K = q.shape
    nb = -(-N // bn)
    qf = F.pad(q.float(), (0, 0, 0, nb * bn - N)).view(*lead, nb, bn, K // bk, bk)
    big = qf.abs().amax(dim=(-3, -1)) > FP8_MAX                       # [..., nb, K/bk]
    half = torch.where(big[..., :, None, :, None], qf * 0.5, qf)
    q2 = half.reshape(*lead, nb * bn, K)[..., :N, :].to(torch.float8_e4m3fn)
    s2 = torch.where(big, s.float() * 2.0, s.float())
    return q2, s2


def dequant_block(q: torch.Tensor, s: torch.Tensor, block: tuple[int, int], dtype) -> torch.Tensor:
    """``q [..., N, K]`` e4m3fn x ``s [..., ceil(N/bn), K/bk]`` -> ``dtype``. In the graph: the
    (small) scale is broadcast to rows with expand/reshape/slice, then one multiply over a
    ``[N, K/bk, bk]`` view of the weight; no gather, no repeat_interleave."""
    bn, bk = block
    *lead, N, K = q.shape
    nb, kb = s.shape[-2:]
    if K != kb * bk or nb != -(-N // bn):
        raise ValueError(f"scale {tuple(s.shape)} does not tile weight {(N, K)} in {block} blocks")
    rows = s.to(dtype).unsqueeze(-2).expand(*lead, nb, bn, kb).reshape(*lead, nb * bn, kb)[..., :N, :]
    w = q.to(dtype).view(*lead, N, kb, bk) * rows.unsqueeze(-1)
    return w.view(*lead, N, K)


class Linear(nn.Module):
    """``y = x W^T``; ``W`` either in the model dtype or block FP8 (``weight`` +
    ``weight_scale_inv``, the checkpoint's own names)."""

    def __init__(self, in_f: int, out_f: int, fp8_block=None):
        super().__init__()
        self.block = fp8_block
        if fp8_block is None:
            self.weight = nn.Parameter(torch.zeros(out_f, in_f))
        else:
            bn, bk = fp8_block
            self.weight = nn.Parameter(torch.zeros(out_f, in_f, dtype=torch.float8_e4m3fn),
                                       requires_grad=False)
            self.weight_scale_inv = nn.Parameter(torch.ones(-(-out_f // bn), -(-in_f // bk)))

    def dense_weight(self, dtype) -> torch.Tensor:
        if self.block is None:
            return self.weight.to(dtype)
        return dequant_block(self.weight, self.weight_scale_inv, self.block, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.dense_weight(x.dtype))


@dataclasses.dataclass(frozen=True)
class Parallel:
    """TP over all ranks for attention (heads), dense MLPs and the shared expert; routed
    experts split ``ep`` ways (``ep * etp == tp``). See ``deepseek_v41.model.Parallel``."""

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


@functools.lru_cache(maxsize=4)
def rope_inv_freq(dim: int, theta: float) -> torch.Tensor:
    """transformers' default RoPE ``inv_freq`` (fp32). Always on the CPU, explicitly: it is
    first called while the runner builds the model under ``torch.device("meta")``, and
    the cache would otherwise keep a meta tensor."""
    cpu = torch.device("cpu")
    return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.int64, device=cpu).to(torch.float32) / dim))


def rope_interleave(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """transformers' ``apply_rotary_pos_emb_interleave`` for one tensor: pairs
    ``(x[2i], x[2i+1])`` rotate by angle ``i``; the output is laid out half-split
    (``cat(re, im)``), as transformers emits it. q and k share the layout, so the dot
    product is that of the interleaved rotation."""
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)


# ----------------------------------------------------------------------------- cache
@dataclasses.dataclass(frozen=True)
class CacheLayout:
    """One page per ``block_size`` tokens holding every layer's rows for them.

    ``[layer 0 latent rows (B x 576) | ... | layer L-1 | index layer 0 keys (B x 128) |
    ... | pad]``. The page is padded to a multiple of ``lcm(576, 128)`` so the flat buffer
    viewed as 576-wide or 128-wide rows has every region on a row boundary.
    """

    block_size: int
    n_layers: int
    latent_dim: int
    index_dim: int
    index_layers: tuple
    dtype: torch.dtype

    @property
    def _unit(self) -> int:
        import math
        return math.lcm(self.latent_dim, self.index_dim)

    @property
    def latent_elems(self) -> int:
        return self.n_layers * self.block_size * self.latent_dim

    @property
    def page_elems(self) -> int:
        raw = self.latent_elems + len(self.index_layers) * self.block_size * self.index_dim
        u = self._unit
        # also a multiple of block_size: the runner sizes one vector per token
        import math
        u = math.lcm(u, self.block_size)
        return -(-raw // u) * u

    def latent_row(self, layer: int, page: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
        return page * (self.page_elems // self.latent_dim) + layer * self.block_size + tok

    def index_row(self, layer: int, page: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
        j = self.index_layers.index(layer)
        base = self.latent_elems // self.index_dim + j * self.block_size
        return page * (self.page_elems // self.index_dim) + base + tok


@dataclasses.dataclass
class Step:
    """Per-forward addressing shared by every layer. ``n`` requests x ``T`` tokens over a
    context of ``C`` positions."""

    decode: bool
    pos: torch.Tensor      # [n, T] long
    live: torch.Tensor     # [n, T] real token with a cache slot
    s: torch.Tensor        # [n] position of the step's first token
    w_page: torch.Tensor   # [n, T] write page (sink when dead)
    w_tok: torch.Tensor    # [n, T]
    c_page: torch.Tensor   # [n, C] page of each context position (zero page when unusable)
    c_tok: torch.Tensor    # [n, C]
    causal: torch.Tensor   # [n, T, C] key position <= query position, live rows only
    used: torch.Tensor     # [n, C, 1] any query of the row may read the position
    cos: torch.Tensor      # [n, T, rope/2] fp32
    sin: torch.Tensor

    @property
    def C(self) -> int:
        return self.c_page.shape[1]


def build_step(lay: CacheLayout, num_pages: int, positions: torch.Tensor, md: dict,
               inv_freq: torch.Tensor) -> Step:
    decode = md["max_query_len"] <= md["decode_token_threshold"]
    zero_page, sink = reserved_pages(num_pages)
    bt = md["block_table_tensor"]
    n = bt.shape[0]
    pos = positions.to(torch.long).view(n, -1)
    T = pos.shape[1]
    slot = md["slot_mapping"].to(torch.long).view(n, -1)[:, :T]
    if decode:
        live = slot > 0
    else:
        # pads are appended and repeat the last position, so they break the run
        live = ((pos - pos[:, :1]) == torch.arange(T, device=pos.device)) & (slot > 0)
    live_rows = live.any(dim=1)
    B = lay.block_size
    slot = slot.clamp_min(0)
    w_page = torch.where(live, _idiv(slot, B), torch.full_like(slot, sink))
    w_tok = torch.where(live, _imod(slot, B), torch.zeros_like(slot))
    blocks = paged_block_ids(bt, live_rows, num_pages)                    # [n, nb]
    C = blocks.shape[1] * B
    cpos = torch.arange(C, device=pos.device)
    c_page = blocks.index_select(1, _idiv(cpos, B))
    c_tok = _imod(cpos, B).unsqueeze(0).expand(n, C)
    causal = (cpos.view(1, 1, C) <= pos.unsqueeze(-1)) & live_rows.view(n, 1, 1)
    used = causal.any(dim=1).unsqueeze(-1)
    ang = pos.to(torch.float32).unsqueeze(-1) * inv_freq
    return Step(decode=decode, pos=pos, live=live, s=pos[:, 0], w_page=w_page, w_tok=w_tok,
                c_page=c_page, c_tok=c_tok, causal=causal, used=used,
                cos=torch.cos(ang), sin=torch.sin(ang))


@dataclasses.dataclass
class Caches:
    lay: CacheLayout
    pages: torch.Tensor     # [num_pages, page_elems]

    @property
    def num_pages(self) -> int:
        return self.pages.shape[0]

    def _read(self, rows: torch.Tensor, width: int) -> torch.Tensor:
        flat = self.pages.view(-1, width)
        return flat.index_select(0, rows.reshape(-1)).view(*rows.shape, width)

    def read_latent(self, layer, page, tok):
        return self._read(self.lay.latent_row(layer, page, tok), self.lay.latent_dim)

    def read_index(self, layer, page, tok):
        return self._read(self.lay.index_row(layer, page, tok), self.lay.index_dim)

    def write_latent(self, layer, page, tok, rows):
        write_cache_rows(self.pages, rows.reshape(-1, self.lay.latent_dim),
                         self.lay.latent_row(layer, page, tok).reshape(-1))

    def write_index(self, layer, page, tok, rows):
        write_cache_rows(self.pages, rows.reshape(-1, self.lay.index_dim),
                         self.lay.index_row(layer, page, tok).reshape(-1))


def _substitute(gathered: torch.Tensor, fresh: torch.Tensor, j0: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
    """``gathered [n, C, d]`` with positions ``j0 .. j0+T-1`` replaced by ``fresh [n, T, d]``
    wherever ``valid [n, T]``."""
    n, C, _ = gathered.shape
    T = fresh.shape[1]
    rel = torch.arange(C, device=gathered.device).unsqueeze(0) - j0.view(n, 1)
    inside = (rel >= 0) & (rel < T)
    g = rel.clamp(0, T - 1)
    take = fresh.gather(1, g.unsqueeze(-1).expand(n, C, fresh.shape[-1]))
    ok = inside & valid.gather(1, g)
    return torch.where(ok.unsqueeze(-1), take.to(gathered.dtype), gathered)


# ------------------------------------------------------------------------- attention
class Indexer(nn.Module):
    """DSA lightning indexer (transformers ``GlmMoeDsaIndexer``): returns the ``[n, T, C]``
    selection mask, a subset of ``step.causal``."""

    query_chunk = 128

    def __init__(self, args: GlmMoeDsaArgs, layer_id: int, par: Parallel = Parallel()):
        super().__init__()
        self.layer_id, self.par = layer_id, par
        self.n_heads, self.head_dim = args.index_n_heads, args.index_head_dim
        self.rd = args.qk_rope_head_dim
        self.index_topk = args.index_topk
        # output rows split over TP and all-gathered: replicated it is 8.4M params x 21
        # layers per rank; the per-rank slice is small enough to keep in the model dtype
        self.wq_b = Linear(args.q_lora_rank, _part(self.n_heads * self.head_dim, par.tp, "index q width"))
        self.wk = Linear(args.dim, self.head_dim, args.fp8_block)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = Linear(args.dim, self.n_heads)

    def forward(self, x, qr, step: Step, caches: Caches):
        n, T, _ = x.shape
        rd = self.rd
        cos, sin = step.cos, step.sin
        q = self.par.all_gather(self.wq_b(qr), dim=qr.dim() - 1).unflatten(-1, (self.n_heads, self.head_dim))
        q = torch.cat([rope_interleave(q[..., :rd], cos.unsqueeze(2), sin.unsqueeze(2)),
                       q[..., rd:]], dim=-1)
        k = self.k_norm(self.wk(x))
        k = torch.cat([rope_interleave(k[..., :rd], cos, sin), k[..., rd:]], dim=-1)
        caches.write_index(self.layer_id, step.w_page, step.w_tok, k)
        ctx = caches.read_index(self.layer_id, step.c_page, step.c_tok)
        ctx = _substitute(ctx, k, step.s, step.live)
        C = ctx.shape[1]
        if self.index_topk >= C:
            # every causal position is selected: no scores, no top-k
            return step.causal
        ctx = torch.where(step.used, ctx, torch.zeros_like(ctx)).float()
        w = F.linear(x.float(), self.weights_proj.weight.float()) * (self.n_heads ** -0.5)
        qf, chunks = q.float(), []
        for t0 in range(0, T, self.query_chunk):
            s = torch.einsum("nthd,ncd->nthc", qf[:, t0:t0 + self.query_chunk], ctx)
            s = F.relu(s * (self.head_dim ** -0.5))
            chunks.append((s * w[:, t0:t0 + self.query_chunk].unsqueeze(-1)).sum(dim=2))
        score = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]   # [n, T, C]
        _capture_tensor(f"model.layers.{self.layer_id}.self_attn.indexer.scores", score)
        score = score.masked_fill(~step.causal, float("-inf"))
        idx = topk_indices(score, self.index_topk)
        sel = torch.zeros_like(step.causal).scatter(-1, idx, True)
        return sel & step.causal


class Attention(nn.Module):
    """MLA, absorbed, over the whole addressable context under a sparse mask."""

    def __init__(self, args: GlmMoeDsaArgs, layer_id: int, par: Parallel = Parallel()):
        super().__init__()
        self.layer_id = layer_id
        self.n_heads = _part(args.n_heads, par.tp, "num_attention_heads")
        self.h0 = par.rank * self.n_heads
        self.nope, self.rd, self.vd = args.qk_nope_head_dim, args.qk_rope_head_dim, args.v_head_dim
        self.kvr = args.kv_lora_rank
        self.qk = self.nope + self.rd
        self.scale = self.qk ** -0.5
        # q_a_proj / kv_a_proj_with_mqa: output rows split over TP and all-gathered. Every
        # rank needs the whole q latent (its head's q_b_proj, the indexer) and the whole
        # 576-wide KV row (the replicated cache), but replicating the weights cost 1.2 GB
        # per rank at full depth; the per-rank slices are kept in the model dtype.
        self.par = par
        self.q_a_proj = Linear(args.dim, _part(args.q_lora_rank, par.tp, "q_lora_rank"))
        self.q_a_layernorm = RMSNorm(args.q_lora_rank, args.norm_eps)
        self.q_b_proj = Linear(args.q_lora_rank, self.n_heads * self.qk)
        self.kv_a_proj_with_mqa = Linear(args.dim, _part(self.kvr + self.rd, par.tp, "kv_lora_rank + rope"))
        self.kv_a_layernorm = RMSNorm(self.kvr, args.norm_eps)
        self.kv_b_proj = Linear(self.kvr, self.n_heads * (self.nope + self.vd))
        self.o_proj = Linear(self.n_heads * self.vd, args.dim)
        self.indexer = Indexer(args, layer_id, par) if args.indexer_types[layer_id] == "full" else None

    def forward(self, x, step: Step, caches: Caches, shared: dict):
        n, T, _ = x.shape
        H = self.n_heads
        cos, sin = step.cos, step.sin
        last = x.dim() - 1
        qr = self.q_a_layernorm(self.par.all_gather(self.q_a_proj(x), dim=last))
        q = self.q_b_proj(qr).view(n, T, H, self.qk)
        q_nope = q[..., : self.nope]
        q_pe = rope_interleave(q[..., self.nope:], cos.unsqueeze(2), sin.unsqueeze(2))
        kv = self.par.all_gather(self.kv_a_proj_with_mqa(x), dim=last)
        lat = self.kv_a_layernorm(kv[..., : self.kvr])
        k_pe = rope_interleave(kv[..., self.kvr:], cos, sin)
        fresh = torch.cat([lat, k_pe], dim=-1)                             # [n, T, 576]
        caches.write_latent(self.layer_id, step.w_page, step.w_tok, fresh)
        ctx = caches.read_latent(self.layer_id, step.c_page, step.c_tok)
        ctx = _substitute(ctx, fresh, step.s, step.live)
        # select, never multiply: an unwritten page may hold anything
        ctx = torch.where(step.used, ctx, torch.zeros_like(ctx)).float()  # [n, C, 576]

        if self.indexer is not None:
            shared["mask"] = self.indexer(x, qr, step, caches)
        mask = shared["mask"]                                              # [n, T, C]

        w = self.kv_b_proj.weight.float().view(H, self.nope + self.vd, self.kvr)
        w_uk, w_uv = w[:, : self.nope], w[:, self.nope:]
        q_abs = torch.einsum("nthq,hqr->nthr", q_nope.float(), w_uk)
        qf = torch.cat([q_abs, q_pe.float()], dim=-1)                      # [n, T, H, 576]
        s = torch.einsum("nthr,ncr->nthc", qf, ctx) * self.scale
        s = s.masked_fill(~mask.unsqueeze(2), float("-inf"))
        p = torch.softmax(s, dim=-1)
        o = torch.einsum("nthc,ncr->nthr", p, ctx[..., : self.kvr])
        o = torch.einsum("nthr,hvr->nthv", o, w_uv)                        # [n, T, H, vd]
        return self.o_proj(o.reshape(n, T, H * self.vd).to(x.dtype))      # partial under TP


# ------------------------------------------------------------------------------- MLP
class MLP(nn.Module):
    """SwiGLU, intermediate split over TP (column gate/up, row down): a partial sum."""

    def __init__(self, dim: int, inter: int):
        super().__init__()
        self.gate_proj, self.up_proj = Linear(dim, inter), Linear(dim, inter)
        self.down_proj = Linear(inter, dim)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Gate(nn.Module):
    """sigmoid scores; the bias picks experts, the unbiased scores weight them."""

    def __init__(self, args: GlmMoeDsaArgs):
        super().__init__()
        self.topk = args.n_activated_experts
        self.norm_topk_prob = args.norm_topk_prob
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.zeros(args.n_routed_experts, args.dim))
        self.e_score_correction_bias = nn.Parameter(torch.zeros(args.n_routed_experts))

    def forward(self, x):
        scores = F.linear(x.float(), self.weight.float()).sigmoid()
        idx = topk_indices(scores + self.e_score_correction_bias.float(), self.topk)
        w = scores.gather(1, idx)
        if self.norm_topk_prob:
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
        return w * self.route_scale, idx


class Experts(nn.Module):
    """This rank's routed experts, stacked ``[E_local, ...]`` in the checkpoint's own
    orientation (gate/up ``[I, D]``, down ``[D, I]``), FP8 with block scales or plain."""

    def __init__(self, n_local: int, dim: int, inter: int, block=None):
        super().__init__()
        self.block = block
        shapes = {"gate_proj": (inter, dim), "up_proj": (inter, dim), "down_proj": (dim, inter)}
        for name, (a, b) in shapes.items():
            if block is None:
                setattr(self, name, nn.Parameter(torch.zeros(n_local, a, b)))
            else:
                setattr(self, name, nn.Parameter(
                    torch.zeros(n_local, a, b, dtype=torch.float8_e4m3fn), requires_grad=False))
                setattr(self, name + "_scale_inv", nn.Parameter(
                    torch.ones(n_local, a // block[0], b // block[1])))

    def weight(self, name: str, dtype) -> torch.Tensor:
        q = getattr(self, name)
        if self.block is None:
            return q.to(dtype)
        return dequant_block(q, getattr(self, name + "_scale_inv"), self.block, dtype)


class MoE(nn.Module):
    """Every local expert on every token, zero weight off the top-k: exact and static.
    Returns this rank's fp32 partial (local experts + its shared-expert slice)."""

    def __init__(self, args: GlmMoeDsaArgs, par: Parallel = Parallel()):
        super().__init__()
        if args.n_shared_experts != 1:
            raise NotImplementedError("exactly one shared expert")
        if par.etp != 1:
            raise NotImplementedError("routed experts are EP-only here (ep == tp)")
        E, D, I = args.n_routed_experts, args.dim, args.moe_inter_dim
        self.E = E
        self.E_local = _part(E, par.ep, "n_routed_experts")
        self.e0 = par.ep_rank * self.E_local          # load time only
        self.gate = Gate(args)
        self.experts = Experts(self.E_local, D, I, args.fp8_block)
        self.shared_experts = MLP(D, _part(I * args.n_shared_experts, par.tp, "moe_intermediate_size"))

    def forward(self, x, rank):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        w, idx = self.gate(x)
        dense = torch.zeros(x.shape[0], self.E, dtype=w.dtype, device=x.device)
        local = rank * self.E_local + torch.arange(self.E_local, device=x.device)
        dense = dense.scatter(1, idx, w).index_select(1, local)          # [t, E_local]
        ex = self.experts
        g = torch.einsum("td,eid->tei", x, ex.weight("gate_proj", x.dtype))
        u = torch.einsum("td,eid->tei", x, ex.weight("up_proj", x.dtype))
        h = (F.silu(g.float()) * u.float() * dense.unsqueeze(-1)).to(x.dtype)
        y = torch.einsum("tei,edi->td", h, ex.weight("down_proj", x.dtype)).float()
        y = y + self.shared_experts(x).float()
        return y.view(*shape[:-1], -1)


class Block(nn.Module):
    def __init__(self, args: GlmMoeDsaArgs, layer_id: int, par: Parallel = Parallel()):
        super().__init__()
        self.layer_id, self.par = layer_id, par
        self.self_attn = Attention(args, layer_id, par)
        self.is_moe = args.mlp_layer_types[layer_id] == "sparse"
        self.mlp = MoE(args, par) if self.is_moe else MLP(args.dim, _part(args.inter_dim, par.tp, "intermediate_size"))
        self.input_layernorm = RMSNorm(args.dim, args.norm_eps)
        self.post_attention_layernorm = RMSNorm(args.dim, args.norm_eps)

    def forward(self, h, step, caches, shared, ep_rank):
        a = self.self_attn(self.input_layernorm(h), step, caches, shared)
        h = h + self.par.all_reduce(a.float()).to(h.dtype)
        x = self.post_attention_layernorm(h)
        m = self.mlp(x, ep_rank) if self.is_moe else self.mlp(x).float()
        return h + self.par.all_reduce(m).to(h.dtype)


# ----------------------------------------------------------------------------- model
_FP32_SUFFIXES = ("e_score_correction_bias", "scale_inv")


class GlmMoeDsaModel(nn.Module):
    """The decoder. Parameter names are the HF checkpoint's (``model.*``, ``lm_head``),
    except the stacked routed experts; shapes are this rank's."""

    kv_cache_page_major = True
    # one KV buffer per layer (runner 152f21b): with a single paged group this is moot,
    # but two typed views of one shared buffer written in one graph are suspected of
    # clobbering each other on device, so never share.
    kv_cache_unshared = True

    def __init__(self, args: GlmMoeDsaArgs, block_size: int = 32,
                 cache_dtype: torch.dtype = torch.bfloat16, par: Parallel = Parallel()):
        super().__init__()
        self.args, self.par = args, par
        self.vocab_local = _part(args.vocab_size, par.tp, "vocab_size")
        self.v0 = par.rank * self.vocab_local
        self.model = nn.Module()
        self.model.embed_tokens = nn.Module()
        self.model.embed_tokens.weight = nn.Parameter(torch.zeros(self.vocab_local, args.dim))
        self.model.layers = nn.ModuleList(Block(args, i, par) for i in range(args.n_layers))
        self.model.norm = RMSNorm(args.dim, args.norm_eps)
        self.lm_head = nn.Module()
        self.lm_head.weight = nn.Parameter(torch.zeros(self.vocab_local, args.dim))
        self.layout = CacheLayout(block_size=block_size, n_layers=args.n_layers,
                                  latent_dim=args.latent_dim, index_dim=args.index_head_dim,
                                  index_layers=args.index_layers, dtype=cache_dtype)
        self.register_buffer("inv_freq", rope_inv_freq(args.qk_rope_head_dim, args.rope_theta),
                             persistent=False)

    def keeps_fp32(self, name: str) -> bool:
        return name.endswith(_FP32_SUFFIXES)

    def set_dtype(self, dtype: torch.dtype) -> "GlmMoeDsaModel":
        for name, p in self.named_parameters():
            if not p.is_floating_point() or p.dtype == torch.float8_e4m3fn:
                continue
            p.data = p.data.to(torch.float32 if self.keeps_fp32(name) else dtype)
        return self

    # -- caches ------------------------------------------------------------------------
    def get_kv_spec(self) -> KVSpec:
        lay = self.layout
        return KVSpec(layers=[], paged_layers=[
            PagedLayerSpec(name=CACHE_LAYER, block_size=lay.block_size,
                           page_elems=lay.page_elems, dtype=lay.dtype)])

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]) -> None:
        (c,) = kv_caches[CACHE_LAYER]
        lay = self.layout
        if c.shape[1:] != (lay.page_elems,) or c.dtype != lay.dtype:
            raise ValueError(f"bound pages {tuple(c.shape)} {c.dtype} do not match the layout "
                             f"({lay.page_elems}, {lay.dtype})")
        # a plain tensor attribute: the runner's graph-capture trace swaps those
        self.cache_pages = c

    # -- forward -------------------------------------------------------------------------
    def _rank(self, rank, device) -> torch.Tensor:
        if rank is None:
            rank = torch.tensor(self.par.rank, device=device)
        return rank.to(torch.long).reshape(())

    def embed_tokens(self, ids: torch.Tensor, rank: torch.Tensor | None = None) -> torch.Tensor:
        """Vocab-sharded lookup: an id off this rank reads row 0 and is zeroed."""
        rank = self._rank(rank, ids.device)
        local = ids - rank * self.vocab_local
        off = (local < 0) | (local >= self.vocab_local)
        h = F.embedding(torch.where(off, torch.zeros_like(local), local), self.model.embed_tokens.weight)
        h = torch.where(off.unsqueeze(-1), torch.zeros_like(h), h)
        return self.par.all_reduce(h)

    def hidden_states(self, input_ids, positions, attn_metadata, rank=None):
        if getattr(self, "cache_pages", None) is None:
            raise RuntimeError("bind_kv_cache() has not been called")
        rank = self._rank(rank, input_ids.device)
        caches = Caches(self.layout, self.cache_pages)
        step = build_step(self.layout, caches.num_pages, positions, attn_metadata[CACHE_LAYER],
                          self.inv_freq)
        n, T = step.pos.shape
        h = self.embed_tokens(input_ids.view(n, T), rank)
        # routed experts are EP-only (etp == 1): the EP rank is the TP rank
        shared: dict = {}
        for layer in self.model.layers:
            h = layer(h, step, caches, shared, rank)
        return self.model.norm(h).view(n * T, -1)

    def compute_logits(self, hidden: torch.Tensor, gather: bool = True) -> torch.Tensor:
        local = F.linear(hidden.float(), self.lm_head.weight.float())
        return self.par.all_gather(local, dim=-1) if gather else local

    on_device_sampling_config = None
    _gather_logits = False

    def attach_sampler(self, neuron_config) -> None:
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
                logit_mask=None, rank=None, **_unused):
        if spec_decode_metadata is not None:
            raise NotImplementedError("speculative decoding (MTP) is out of scope")
        hidden = self.hidden_states(input_ids, positions, attn_metadata, rank)
        if sampling_positions is not None:
            hidden = torch.index_select(hidden, 0, sampling_positions)
        if self.on_device_sampling_config is None:
            return self.compute_logits(hidden)
        logits = self.compute_logits(hidden, gather=False)
        gathered = self.par.all_gather(logits, dim=-1) if self._gather_logits else None
        sampled = self.sampler(logits, sampling_params, logit_mask=logit_mask, tp_rank=rank)
        return sampled, gathered

    # -- weights -------------------------------------------------------------------------
    def local_tensor(self, name: str, src) -> torch.Tensor:
        """This rank's value of parameter ``name`` from ``src`` (``CheckpointSource`` /
        ``DictSource``), which serve HF-named tensors: ``get`` dequantized to bf16/fp32
        slices, ``get_fp8`` raw e4m3 + block scales."""
        a, par = self.args, self.par
        if name in ("model.embed_tokens.weight", "lm_head.weight"):
            return src.get(name, rows=(self.v0, self.v0 + self.vocab_local))
        parts = name.split(".")
        if parts[1] != "layers":
            return src.get(name)
        i, rest = int(parts[2]), ".".join(parts[3:])
        pre = f"model.layers.{i}."
        layer = self.model.layers[i]
        at = layer.self_attn
        if rest in ("self_attn.q_a_proj.weight", "self_attn.kv_a_proj_with_mqa.weight",
                    "self_attn.indexer.wq_b.weight"):
            out = self.get_submodule(name.rsplit(".", 1)[0]).weight.shape[0]
            return src.get(name, rows=(par.rank * out, (par.rank + 1) * out))
        if rest == "self_attn.q_b_proj.weight":
            return src.get(name, rows=(at.h0 * at.qk, (at.h0 + at.n_heads) * at.qk))
        if rest == "self_attn.kv_b_proj.weight":
            w = (at.nope + at.vd)
            return src.get(name, rows=(at.h0 * w, (at.h0 + at.n_heads) * w))
        if rest == "self_attn.o_proj.weight":
            return src.get(name, cols=(at.h0 * at.vd, (at.h0 + at.n_heads) * at.vd))
        if rest.endswith((".weight", ".weight_scale_inv")) and (
                rest.startswith("mlp.shared_experts.") or (not layer.is_moe and rest.startswith("mlp."))):
            mlp = layer.mlp.shared_experts if layer.is_moe else layer.mlp
            I = mlp.gate_proj.weight.shape[0]
            span = (par.rank * I, (par.rank + 1) * I)
            return src.get(name, cols=span) if ".down_proj." in rest else src.get(name, rows=span)
        if rest.startswith("mlp.experts."):
            ex = layer.mlp.experts
            which = parts[-1]
            scale = which.endswith("_scale_inv")
            proj = which[: -len("_scale_inv")] if scale else which
            experts = range(layer.mlp.e0, layer.mlp.e0 + layer.mlp.E_local)
            names = [f"{pre}mlp.experts.{e}.{proj}.weight" for e in experts]
            if ex.block is None:
                return torch.stack([src.get(n) for n in names])
            cache = self.__dict__.setdefault("_fp8_cache", {})
            key = (i, proj)
            if key not in cache:
                qs = [src.get_fp8(n) for n in names]
                q, s = fp8_le240(torch.stack([x[0] for x in qs]), torch.stack([x[1] for x in qs]), ex.block)
                cache[key] = [q, s, 2]
            entry = cache[key]
            out = entry[1] if scale else entry[0]
            entry[2] -= 1
            if entry[2] == 0:
                del cache[key]
            return out
        mod_name = name.rsplit(".", 1)[0]
        mod = self.get_submodule(mod_name)
        if isinstance(mod, Linear) and mod.block is not None:
            # replicated FP8 projection: raw bytes + block scales, |q| <= 240
            cache = self.__dict__.setdefault("_fp8_cache", {})
            if mod_name not in cache:
                q, s = src.get_fp8(f"{mod_name}.weight")
                q, s = fp8_le240(q, s, mod.block)
                cache[mod_name] = [q, s, 2]
            entry = cache[mod_name]
            out = entry[1] if name.endswith("_scale_inv") else entry[0]
            entry[2] -= 1
            if entry[2] == 0:
                del cache[mod_name]
            return out
        return src.get(name)

    def load_from(self, src, device=None) -> None:
        own = dict(self.named_parameters())
        sd = {}
        for name, p in own.items():
            t = self.local_tensor(name, src)
            if tuple(t.shape) != tuple(p.shape):
                raise ValueError(f"{name}: source gives {tuple(t.shape)}, rank expects {tuple(p.shape)}")
            if p.dtype == torch.float8_e4m3fn and t.dtype != p.dtype:
                raise TypeError(f"{name}: expected float8_e4m3fn from the loader, got {t.dtype}")
            sd[name] = t.to(dtype=p.dtype, device=device or p.device).contiguous()
        self.load_state_dict(sd, strict=True, assign=True)
        dev = device
        if dev is not None:
            self.inv_freq = rope_inv_freq(self.args.qk_rope_head_dim, self.args.rope_theta).to(dev)
        elif self.inv_freq.is_meta:
            self.inv_freq = rope_inv_freq(self.args.qk_rope_head_dim, self.args.rope_theta)

    def load_weights(self, checkpoint_path: str, device: torch.device, cache_dir: str | None = None):
        with Checkpoint(checkpoint_path) as ckpt:
            if self.args.fp8_block is None and any(k.endswith("weight_scale_inv") for k in ckpt.weight_map):
                raise ValueError("the checkpoint is block FP8 but the config carries no "
                                 "(original_)quantization_config: serve from make_served_dir")
            self.load_from(CheckpointSource(ckpt, self.args.fp8_block), device)


# ------------------------------------------------------------------------ checkpoint
def _span(n: int, span):
    return (0, n) if span is None else span


class DictSource:
    """HF-named tensors from a dict: plain tensors, or ``(fp8, scale_inv)`` under
    ``name`` + ``weight_scale_inv`` as the checkpoint stores them."""

    def __init__(self, sd: dict, block=None):
        self.sd, self.block = sd, block

    def _full(self, name):
        t = self.sd[name]
        if t.dtype == torch.float8_e4m3fn:
            s = self.sd[name[: -len("weight")] + "weight_scale_inv"]
            return dequant_block(t, s, self.block, torch.float32)
        return t

    def get(self, name, rows=None, cols=None):
        t = self._full(name)
        r0, r1 = _span(t.shape[0], rows)
        part = t[r0:r1]
        if cols is not None:
            part = part[:, cols[0]:cols[1]]
        return part

    def get_fp8(self, name):
        return self.sd[name], self.sd[name[: -len("weight")] + "weight_scale_inv"].float()


class Checkpoint:
    """The HF checkpoint directory, opened lazily per shard."""

    def __init__(self, hf_dir):
        import json
        from pathlib import Path

        self.dir = Path(hf_dir)
        self.weight_map = json.loads((self.dir / "model.safetensors.index.json").read_text())["weight_map"]
        self._handles: dict = {}

    def raw_slice(self, name: str):
        from safetensors import safe_open

        f = self.weight_map[name]
        if f not in self._handles:
            self._handles[f] = safe_open(str(self.dir / f), framework="pt", device="cpu")
        return self._handles[f].get_slice(name)

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._handles.clear()


class CheckpointSource:
    """Slices of the HF checkpoint. An FP8 weight reads only the blocks covering the slice
    and is dequantized in fp32; ``load_from`` rounds it once, to the parameter's dtype
    (returning bf16 here cost an fp32 model 1e-2 in logprobs)."""

    def __init__(self, ckpt: Checkpoint, block):
        self.ckpt, self.block = ckpt, block

    def get(self, name, rows=None, cols=None):
        ck = self.ckpt
        scale_name = name[: -len("weight")] + "weight_scale_inv" if name.endswith("weight") else None
        sl = ck.raw_slice(name)
        shape = sl.get_shape()
        if len(shape) == 2 and scale_name and ck.has(scale_name):
            bn, bk = self.block
            n, k = shape
            (r0, r1), (c0, c1) = _span(n, rows), _span(k, cols)
            R0, C0 = r0 // bn * bn, c0 // bk * bk
            R1, C1 = min(-(-r1 // bn) * bn, n), min(-(-c1 // bk) * bk, k)
            q = sl[R0:R1, C0:C1]
            s = ck.raw_slice(scale_name)[R0 // bn:-(-R1 // bn), C0 // bk:-(-C1 // bk)].float()
            w = q.float() * s.repeat_interleave(bn, 0)[: R1 - R0].repeat_interleave(bk, 1)[:, : C1 - C0]
            if not torch.isfinite(w).all():
                raise ValueError(f"{name}: non-finite after dequantization")
            return w[r0 - R0:r1 - R0, c0 - C0:c1 - C0]
        r0, r1 = _span(shape[0], rows)
        if cols is None:
            return sl[r0:r1]
        return sl[r0:r1, cols[0]:cols[1]]

    def get_fp8(self, name):
        q = self.ckpt.raw_slice(name)[:]
        s = self.ckpt.raw_slice(name[: -len("weight")] + "weight_scale_inv")[:].float()
        if q.dtype != torch.float8_e4m3fn:
            raise TypeError(f"{name}: stored as {q.dtype}, expected float8_e4m3fn")
        return q, s


def from_configs(hf_config, text_neuron_config=None, **_):
    """The runner's constructor (through ``factory.GlmMoeDsaForCausalLM``), under the meta
    device; ``load_weights`` materialises.

    ``GLM53_N_LAYERS`` truncates the decoder (compile probes / smoke runs only)."""
    import os

    from vllm.config import get_current_vllm_config
    from vllm.distributed.parallel_state import get_tp_group

    nl = os.environ.get("GLM53_N_LAYERS")
    args = GlmMoeDsaArgs.from_hf(hf_config, int(nl) if nl else None)
    tp = get_tp_group()
    world, rank = tp.world_size, tp.rank_in_group
    ep = getattr(text_neuron_config, "ep_degree", 1) or 1
    if ep != world:
        raise NotImplementedError(f"routed experts need ep_degree == tp ({world}); got {ep}")
    par = Parallel(tp=world, rank=rank, ep=world, ep_rank=rank, etp=1, etp_rank=0, group=tp)
    vc = get_current_vllm_config()
    dtype = vc.model_config.dtype
    model = GlmMoeDsaModel(args, block_size=vc.cache_config.block_size, cache_dtype=dtype, par=par)
    model.set_dtype(dtype)
    model.attach_sampler(text_neuron_config)
    return model
