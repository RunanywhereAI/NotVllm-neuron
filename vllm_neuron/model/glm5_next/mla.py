# SPDX-License-Identifier: Apache-2.0
"""NoPE sparse-MLA + DSA indexer — GLM-5.3-Flash's 11 full-attention layers.

Same plugin contract as ``kda.py``: plain-tensor return, state in vLLM's cache,
``max_query_len`` dispatch.

**Absorbed latent.** ``q`` absorbs ``W_uk`` so attention runs directly against the
512-wide latent, which is then also the value. Two consequences, both load-bearing:

* the layer never materialises 64x512 K and V per token; and
* decode is **MQA with one KV head at ``d_head = 512``**, which is a *tested*
  configuration of the existing decode kernel. DeepSeek's 576 (``kv_lora_rank`` 512 +
  ``qk_rope_head_dim`` 64) fails both the ``_MAX_D_HEAD`` and the
  multiple-of-128 checks. GLM's ``qk_rope_head_dim`` is **0**, so it lands on 512 and
  passes both. NoPE is what makes this layer implementable at all.

**One latent tensor serves as both K and V** — dev3 measured this bit-identical. But
**reading may alias; writing must not**: handing one buffer to a scatter as both
``k_cache`` and ``v_cache`` gives the FX aliasing pass two outputs on one buffer.
The write path therefore goes through a single-buffer row scatter
(``functional/vendored_kernels/latent_cache_write``), never a K/V-shaped one.

**Two things the decode kernel forbids at ``d_head > 128``**, both of which suit MLA
rather than fighting it (``attention_block_tkg.py:514-526``):

* no in-kernel cache update — so the latent scatter is external anyway; and
* no in-kernel ``o_proj`` — which is required regardless, since V-up has to happen
  between attention and ``o_proj``.

**The indexer is where validation gets hard.** FP8 scoring changes which pools are
selected on 88-100% of rows above the dense-exact ceiling, so *exact index equality
fails a correct device*. Validation needs the **scores**, to tell a near-tie swap from
a real defect — hence ``indexer.scores`` is a capture point, and ``forward`` takes an
optional ``topk_indices`` override so attention can be checked given the device's own
selection. Without the override a 0.5% attention bug is invisible: measured
4.14e-3 vs 4.20e-3 without it, 1.79e-7 vs 7.19e-4 with it.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

try:
    from vllm_neuron.accuracy.tensor_capture import capture_tensor as _capture_tensor
except ImportError:  # pragma: no cover - hosts without vLLM; see kda.py
    def _capture_tensor(name, tensor):  # type: ignore[misc]
        return None

try:
    # The framework path (``forward``) needs the page layout, the reserved-page
    # convention and the row scatter; ``forward_core`` needs none of them, and that is
    # what the by-path oracle comparison exercises on a host without vLLM.
    from vllm_neuron.functional.vendored_kernels.latent_cache_write import write_cache_rows
    from vllm_neuron.model.glm5_next.cache_layout import LatentPageLayout
    from vllm_neuron.model.kv_cache import paged_block_ids, reserved_pages
except ImportError:  # pragma: no cover
    write_cache_rows = LatentPageLayout = paged_block_ids = reserved_pages = None


@dataclass(frozen=True)
class MLAParams:
    """The config surface this layer reads. dev1 owns the config object."""

    num_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    hidden_size: int
    rms_norm_eps: float
    index_topk: int
    index_kpool: int
    index_n_heads: int
    index_head_dim: int

    @classmethod
    def from_config(cls, config) -> "MLAParams":
        """Every field is REQUIRED -- no fallback to the released values. A default
        that equals the right answer makes an unwired call site indistinguishable from
        a wired one (``kda.py``'s params had exactly that defect)."""
        p = cls(
            num_heads=config.num_attention_heads,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            hidden_size=config.hidden_size,
            rms_norm_eps=config.rms_norm_eps,
            index_topk=config.index_topk,
            index_kpool=config.index_kpool,
            index_n_heads=config.index_n_heads,
            index_head_dim=config.index_head_dim,
        )
        if p.qk_rope_head_dim != 0:
            raise ValueError(
                f"qk_rope_head_dim={p.qk_rope_head_dim}; this layer is NoPE-only. A "
                f"non-zero value makes the latent {p.kv_lora_rank + p.qk_rope_head_dim} "
                f"wide, which fails the decode kernel's d_head checks (<= 512 and a "
                f"multiple of 128) — that is why DeepSeek's 576 cannot use this path."
            )
        if p.index_n_heads != 32:
            raise ValueError(
                f"index_n_heads={p.index_n_heads}; GLM-5.3-Flash uses 32. 64 is "
                f"DeepSeek's value, and vLLM's own source carries a stale '# 64' "
                f"comment three lines from the code that reads the config."
            )
        return p


class RMSNorm(nn.Module):
    """fp32 normalise, multiply by weight in fp32, cast once at the end."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        return (self.weight * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps))).to(x.dtype)


# ------------------------------------------------------------------------ indexer
def kpool_compress(k, gate, ape):
    """Pool key = softmax over the kpool slots, **per channel**, weighting raw K.

    k, gate: ``[..., kpool, D]``; ape: ``[kpool, D]``. The softmax runs over the slot
    axis separately for each of the 128 channels — not over channels.
    """
    p = torch.softmax(gate.float() + ape.float(), dim=-2)
    return (p * k.float()).sum(-2)


def select_tokens(scores, lens, topk, kpool):
    """Pool scores -> token indices ``[B, S, topk + kpool - 1]``, ``-1`` padded.

    Only **complete** pools (``j < lens // kpool``) are candidates — a pool holding
    future tokens is never scored — and the incomplete pool's ``lens % kpool`` tokens
    are appended raw. Selecting everything is therefore exact to
    ``topk + kpool - 1`` = 2051, not 2048; vLLM's own gate at 2048 is conservative.
    """
    B, S, P = scores.shape
    n_complete = lens // kpool
    cand = torch.arange(P, device=scores.device)[None, :] < n_complete[:, None]
    top = scores.masked_fill(~cand, float("-inf")).topk(min(topk // kpool, P), -1).indices
    ok = cand.expand(B, S, P).gather(-1, top)
    off = torch.arange(kpool, device=scores.device)
    tok = (top[..., None] * kpool + off).masked_fill(~ok[..., None], -1).flatten(-2)
    out = F.pad(tok, (0, topk - tok.shape[-1]), value=-1)
    toff = torch.arange(kpool - 1, device=scores.device)
    start = n_complete * kpool
    tail = (start[:, None] + toff).masked_fill(toff >= (lens - start)[:, None], -1)
    return torch.cat([out, tail.expand(B, S, -1)], -1)


@dataclass
class IndexerState:
    """The indexer's carried state — vLLM's two indexer caches, minus paging.

    ``pool_k``    ``[B, P, D]``      compressed COMPLETE pools; scoring reads only this
    ``tail_k``    ``[B, kpool, D]``  raw K ring, slot ``pos % kpool``
    ``tail_gate`` ``[B, kpool, D]``  raw gate ring, same slots
    ``length``    tokens seen so far

    Attention never reads any of it — it gathers latents by token index. On device
    these three regions live in the latent page (dev3's ``cache_layout.py``); here
    they are explicit so the decode path can be tested without the cache.
    """

    pool_k: torch.Tensor
    tail_k: torch.Tensor
    tail_gate: torch.Tensor
    length: int


def indexer_prefill(state, k, gate, ape, kpool):
    """Append S tokens, completing every pool possible and reseeding the tail ring."""
    B, S, D = k.shape
    if state is None:
        z = k.new_zeros(B, kpool, D)
        state = IndexerState(k.new_zeros(B, 0, D), z, z.clone(), 0)
    r = state.length % kpool                       # tokens of the incomplete pool
    raw_k = torch.cat([state.tail_k[:, :r], k], 1)
    raw_g = torch.cat([state.tail_gate[:, :r], gate], 1)
    n = raw_k.shape[1] // kpool
    pools = kpool_compress(raw_k[:, : n * kpool].unflatten(1, (n, kpool)),
                           raw_g[:, : n * kpool].unflatten(1, (n, kpool)), ape)
    tail_k, tail_g = state.tail_k.clone(), state.tail_gate.clone()
    for i in range(max(0, S - kpool), S):
        slot = (state.length + i) % kpool
        tail_k[:, slot], tail_g[:, slot] = k[:, i], gate[:, i]
    return IndexerState(torch.cat([state.pool_k, pools], 1), tail_k, tail_g,
                        state.length + S)


def indexer_decode(state, k, gate, ape, kpool):
    """One token: complete a pool if this token closes one, then stash it.

    EVERY token is stashed, not only pool-completing ones. vLLM once gated the stash
    on completion and thereafter compressed stale prompt-tail entries forever.
    """
    slot = state.length % kpool
    pool_k = state.pool_k
    if slot == kpool - 1:
        ks = torch.cat([state.tail_k[:, :slot], k], 1)
        gs = torch.cat([state.tail_gate[:, :slot], gate], 1)
        pool_k = torch.cat([pool_k, kpool_compress(ks, gs, ape)[:, None]], 1)
    tail_k, tail_g = state.tail_k.clone(), state.tail_gate.clone()
    tail_k[:, slot], tail_g[:, slot] = k[:, 0], gate[:, 0]
    return IndexerState(pool_k, tail_k, tail_g, state.length + 1)


class Glm5NextIndexer(nn.Module):
    """DSA lightning indexer with kpool compression.

    Its own module so a capture hook reaches its output. Emits **token** indices;
    kpool compression is scoring-only and never reaches attention.
    """

    def __init__(self, p: MLAParams, layer_name: str):
        super().__init__()
        self.p = p
        self.layer_name = layer_name
        Hi, D = p.index_n_heads, p.index_head_dim
        self.Hi, self.D = Hi, D
        self.wq_b = nn.Linear(p.q_lora_rank, Hi * D, bias=False)
        self.wk = nn.Linear(p.hidden_size, D, bias=False)
        self.k_norm = nn.LayerNorm(D, eps=1e-6)          # has a bias, unlike the RMSNorms
        self.weights_proj = nn.Linear(p.hidden_size, Hi, bias=False)
        self.index_kpool_compress_ape = nn.Parameter(torch.zeros(p.index_kpool, D))
        self.index_kpool_compress_gate = nn.Parameter(torch.randn(D, p.hidden_size) * 0.02)

    def project(self, x, q_c):
        """-> ``(q [B,S,Hi,D], k [B,S,D], gate [B,S,D], w [B,S,Hi] fp32)``.

        No RoPE: ``qk_rope_head_dim`` is 0, so ``indexer_rope_interleave`` is inert.
        """
        B, S, _ = x.shape
        q = self.wq_b(q_c).view(B, S, self.Hi, self.D)
        k = self.k_norm(self.wk(x))
        gate = F.linear(x, self.index_kpool_compress_gate)
        w = self.weights_proj(x).float() * self.Hi ** -0.5
        return q, k, gate, w

    def score(self, q, w, pool_k):
        """``sum_h w_h * relu(D**-0.5 * q_h . pool_k)`` -> ``[B, S, P]`` fp32."""
        scores = (w[:, :, None, :].float() @ F.relu(
            (q.float() @ pool_k.float().transpose(-1, -2).unsqueeze(1)) * self.D ** -0.5)
        ).squeeze(-2)
        # THE capture point for on-device validation. Exact index equality fails a
        # CORRECT fp8 device, so the scores are what distinguish a near-tie swap from
        # a real defect. Indices alone cannot.
        _capture_tensor(f"{self.layer_name}.indexer.scores", scores)
        return scores

    def select(self, scores, lens):
        """Scores ``[B, S, P]`` and ``lens`` ``[S]`` (tokens visible to each row) ->
        token indices ``[B, S, topk + kpool - 1]``."""
        idx = select_tokens(scores, lens, self.p.index_topk, self.p.index_kpool)
        _capture_tensor(f"{self.layer_name}.indexer.topk_indices", idx)
        return idx

    def forward(self, x, q_c, state=None):
        """-> (token indices ``[B, S, topk + kpool - 1]``, new ``IndexerState``).

        State is threaded rather than recomputed. Getting this wrong is not subtle but
        it IS invisible in a prefill-only test: on decode the layer re-pooled from the
        single new token, found no complete pool, and computed positions from the
        local length — so the token attended to itself alone, 1 of 21 positions.
        """
        S = x.shape[1]
        q, k, gate, w = self.project(x, q_c)
        ape = self.index_kpool_compress_ape
        if state is not None and S == 1:
            state = indexer_decode(state, k, gate, ape, self.p.index_kpool)
        else:
            state = indexer_prefill(state, k, gate, ape, self.p.index_kpool)
        scores = self.score(q, w, state.pool_k)
        # ABSOLUTE positions, not local: on decode the query sits at state.length - 1,
        # not at 0. Using local positions is what made the decode token select nothing.
        lens = torch.arange(state.length - S, state.length, device=x.device) + 1
        return self.select(scores, lens), state


def indices_to_mask(idx, L):
    """``[B,S,W]`` token indices (``-1`` = empty) -> bool ``[B,S,L]``."""
    safe = torch.where(idx < 0, L, idx)
    return torch.zeros(*idx.shape[:2], L + 1, dtype=torch.bool,
                       device=idx.device).scatter_(-1, safe, True)[..., :L]


# --------------------------------------------------------------------------- layer
class Glm5NextSparseMLA(nn.Module):
    """One of the 11 sparse-MLA layers."""

    def __init__(self, config, layer_idx: int, tp_size: int = 1):
        super().__init__()
        p = MLAParams.from_config(config)
        self.p = p
        self.layer_idx = layer_idx
        self.layer_name = f"model.layers.{layer_idx}.self_attn"   # dev3 owns this name
        if p.num_heads % tp_size:
            raise ValueError(f"num_attention_heads={p.num_heads} must divide tp={tp_size}")
        self.H = p.num_heads // tp_size
        self.qk = p.qk_nope_head_dim + p.qk_rope_head_dim
        self.vd = p.v_head_dim
        self.kvr = p.kv_lora_rank
        D = p.hidden_size

        self.q_a_proj = nn.Linear(D, p.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(p.q_lora_rank, p.rms_norm_eps)
        self.q_b_proj = nn.Linear(p.q_lora_rank, self.H * self.qk, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(D, self.kvr + p.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = RMSNorm(self.kvr, p.rms_norm_eps)
        self.kv_b_proj = nn.Linear(self.kvr, self.H * (p.qk_nope_head_dim + self.vd), bias=False)
        self.o_proj = nn.Linear(self.H * self.vd, D, bias=False)
        self.indexer = Glm5NextIndexer(p, self.layer_name)
        self.scaling = self.qk ** -0.5

    # -- absorbed projections ----------------------------------------------------
    def _uk_uv(self):
        """Split ``kv_b_proj`` into the per-head K-up and V-up matrices.

        ``kv_b_proj.weight`` is ``[H * (qk + vd), kvr]``; per head the first ``qk``
        rows lift the latent to K and the remaining ``vd`` rows lift it to V.
        """
        w = self.kv_b_proj.weight.view(self.H, self.qk + self.vd, self.kvr)
        return w[:, : self.qk, :], w[:, self.qk :, :]

    def _project(self, hidden_states):
        """hidden ``[B,S,D]`` -> ``(q_c, q [B,S,H,qk], latent [B,S,kvr])``.

        ``q_c`` is also the indexer's query input.
        """
        B, S, _ = hidden_states.shape
        q_c = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_c).view(B, S, self.H, self.qk)
        latent = self.kv_a_layernorm(self.kv_a_proj_with_mqa(hidden_states)[..., : self.kvr])
        return q_c, q, latent

    def _attend(self, q, latent, mask, out_dtype):
        """Absorbed attention: ``q [B,S,H,qk]`` against ``latent [B,L,kvr]`` under
        ``mask [B,S,L]`` -> ``[B,S,hidden]`` (this rank's partial ``o_proj``).

        ``q @ W_uk`` scores directly against the latent, and V-up is applied *after*
        attention, which is also why the decode kernel's in-kernel ``o_proj`` is
        unusable here (and forbidden at ``d_head = 512``).

        ``latent`` must be finite everywhere: a masked position gets probability
        exactly 0, but ``0 * NaN`` is NaN, so callers SELECT unwritten positions away
        before calling rather than relying on the mask.
        """
        B, S = q.shape[:2]
        W_uk, W_uv = self._uk_uv()
        q_abs = torch.einsum("bshq,hqr->bshr", q.float(), W_uk.float())
        att = torch.einsum("bshr,blr->bhsl", q_abs, latent.float()) * self.scaling
        att = att.masked_fill(~mask.unsqueeze(1), float("-inf")).softmax(-1, dtype=torch.float32)
        mixed = torch.einsum("bhsl,blr->bshr", att, latent.float())
        out = torch.einsum("bshr,hvr->bshv", mixed, W_uv.float())
        _capture_tensor(f"{self.layer_name}.attn_pre_oproj", out)
        return self.o_proj(out.reshape(B, S, -1).to(out_dtype))

    def forward_core(self, hidden_states, kv_cache=None, topk_indices=None,
                     indexer_state=None):
        """-> (output ``[B,S,D]``, (latent ``[B,L,kvr]``, ``IndexerState``)).

        Cache-free form, with state threaded explicitly: the oracle comparison drives
        this. ``forward`` is the same computation against the paged cache.

        ``topk_indices`` replaces the indexer's selection for this call **without**
        skipping the indexer, so state still advances. See the module docstring for
        why this override exists.
        """
        q_c, q, latent = self._project(hidden_states)
        if kv_cache is not None:
            latent = torch.cat([kv_cache, latent], 1)
        _capture_tensor(f"{self.layer_name}.latent", latent)
        L = latent.shape[1]
        own, indexer_state = self.indexer(hidden_states, q_c, indexer_state)
        idx = own if topk_indices is None else topk_indices
        mask = indices_to_mask(idx.long(), L)
        return (self._attend(q, latent, mask, hidden_states.dtype),
                (latent, indexer_state))

    # -- the framework path --------------------------------------------------------
    def bind_latent_pages(self, pages: torch.Tensor) -> None:
        """Bind this layer's page-major view ``[num_pages, page_elems]``.

        The width is checked against ``LatentPageLayout`` on every forward, where the
        block size is known; here only the rank can be.
        """
        if pages.dim() != 2:
            raise ValueError(f"{self.layer_name}: latent page view must be 2-D, got "
                             f"{tuple(pages.shape)}")
        self.pages = pages

    def _layout(self, block_size: int):
        lay = LatentPageLayout.from_config(self.p, block_size, self.pages.dtype)
        if self.pages.shape[1] != lay.total_elems:
            raise ValueError(
                f"{self.layer_name}: bound page is {self.pages.shape[1]} elements but "
                f"the layout at block_size={block_size} needs {lay.total_elems}; the "
                f"runner and the model disagree about the page"
            )
        return lay

    def forward(self, hidden_states, positions, attn_metadata: dict) -> torch.Tensor:
        """Framework entry point: ``[tokens, hidden] -> [tokens, hidden]`` (partial
        under TP; the caller reduces).

        The page holds the latent, the indexer's pool keys and its tail ring
        (``cache_layout.py`` owns every offset; nothing here computes one). Writes go
        through ``write_cache_rows`` -- never ``index_put_`` on the page, which is PR
        #40's full-pool-copy pathology on device -- and the latent is written as ONE
        buffer, never as a K/V pair.

        No read depends on whether this step's writes are visible yet: the value being
        written is substituted instead. On CPU the write is immediate; on device the
        aliasing pass decides, and neither may change the answer.

        ``topk_indices`` is deliberately not a parameter: it is a validation affordance
        on ``forward_core`` only, so it cannot be left on in a serving path.
        """
        if not hasattr(self, "pages"):
            raise RuntimeError(f"{self.layer_name}: bind_latent_pages() has not been called")
        metadata = attn_metadata[self.layer_name]
        lay = self._layout(metadata["block_size"])
        if metadata["max_query_len"] <= metadata["decode_token_threshold"]:
            return self._forward_decode(hidden_states, positions, metadata, lay)
        return self._forward_prefill(hidden_states, positions, metadata, lay)

    def _forward_prefill(self, hidden_states, positions, metadata, lay):
        """One sequence, from position 0, padded to a bucket by appending pads.

        Attention needs no cache read -- every key is in this call. The cache is only
        written: every real token's latent, every COMPLETE pool's key (a pool holding
        a pad is not complete), and the last ``index_kpool`` real tokens' raw K and
        gate into the tail ring of their own page, which is exactly the set decode will
        read back (the open pool's tokens are always among them).
        """
        T = hidden_states.shape[0]
        kp = self.p.index_kpool
        if T % kp:
            raise ValueError(f"prefill bucket {T} is not a multiple of index_kpool {kp}")
        x = hidden_states[None]
        q_c, q, latent = self._project(x)
        _capture_tensor(f"{self.layer_name}.latent", latent)
        iq, ik, igate, iw = self.indexer.project(x, q_c)
        P = T // kp
        # Score against the keys as STORED: decode reads them back from the page at the
        # page dtype, so rounding here too keeps a decode step and a one-shot prefill
        # selecting from identical keys. A no-op in fp32.
        pool_k = kpool_compress(ik.view(1, P, kp, -1), igate.view(1, P, kp, -1),
                                self.indexer.index_kpool_compress_ape).to(self.pages.dtype)
        scores = self.indexer.score(iq, iw, pool_k)
        lens = torch.arange(T, device=x.device) + 1
        mask = indices_to_mask(self.indexer.select(scores, lens).long(), T)
        out = self._attend(q, latent, mask, hidden_states.dtype)

        # -- writes
        offsets = torch.arange(T, device=positions.device, dtype=positions.dtype)
        real = (positions - positions[0]) == offsets
        slots = metadata["slot_mapping"].to(torch.long).view(-1)
        _, sink = reserved_pages(self.pages.shape[0])
        live = real & (slots > 0)
        page, tok = lay.slot_to_page(slots)
        page = torch.where(live, page, torch.full_like(page, sink))
        tok = torch.where(live, tok, torch.zeros_like(tok))
        # the last kp real tokens: real here, and not real kp positions later
        in_ring = live & ~F.pad(real[kp:], (0, kp), value=False)
        ring_page = torch.where(in_ring, page, torch.full_like(page, sink))
        # pool j is complete iff its last token is real; it lives with its first token
        first = torch.arange(0, T, kp, device=x.device)
        complete = real[first + kp - 1] & live[first]
        pool_page = torch.where(complete, page[first], torch.full_like(first, sink))
        write_cache_rows(self.pages, latent[0], lay.latent_row(page, tok))
        k_row, g_row = lay.tail_rows(ring_page, tok)
        write_cache_rows(self.pages, torch.cat([ik[0], igate[0]]), torch.cat([k_row, g_row]))
        write_cache_rows(self.pages, pool_k[0], lay.pool_row(pool_page, tok[first]))
        return out[0]

    def _forward_decode(self, hidden_states, positions, metadata, lay):
        """One token per request, against the paged latent, pools and tail ring."""
        block_table = metadata["block_table_tensor"]
        n = block_table.shape[0]
        if hidden_states.shape[0] != n:
            raise NotImplementedError(
                f"sparse-MLA decode expects one token per request, got "
                f"{hidden_states.shape[0]} for {n} requests"
            )
        kp, kvr = self.p.index_kpool, self.kvr
        num_pages = self.pages.shape[0]
        zero_page, sink = reserved_pages(num_pages)
        slots = metadata["slot_mapping"].to(torch.long).view(n, -1)[:, 0]
        live = slots > 0
        blocks = paged_block_ids(block_table, live, num_pages)           # [n, nb]
        nb = blocks.shape[1]
        ctx = nb * lay.block_size
        P = nb * lay.pools_per_page
        page, tok = lay.slot_to_page(slots)
        read_page = torch.where(live, page, torch.full_like(page, zero_page))
        write_page = torch.where(live, page, torch.full_like(page, sink))

        x = hidden_states[:, None]                                       # [n, 1, D]
        q_c, q, latent_new = self._project(x)
        iq, ik, igate, iw = self.indexer.project(x, q_c)
        p = positions.to(torch.long).view(n)
        L = p + 1

        # -- gather, then make every byte we keep one this request wrote
        rows = self.pages.index_select(0, blocks.reshape(-1)).view(n, nb, -1)
        lat, pools, _ = lay.split(rows)
        lat = lat.reshape(n, ctx, kvr)
        pools = pools.reshape(n, P, -1)
        _, _, ring = lay.split(self.pages.index_select(0, read_page))    # [n, kp, 2, D]
        pos = torch.arange(ctx, device=x.device)
        lat = torch.where((pos[None, :] == p[:, None])[..., None], latent_new, lat)
        lat = torch.where((pos[None, :] < L[:, None])[..., None], lat,
                          torch.zeros_like(lat))
        _capture_tensor(f"{self.layer_name}.latent", lat)

        # -- the indexer: close a pool if this token completes one
        slot = p % kp
        sidx = torch.arange(kp, device=x.device)
        before = (sidx[None, :] < slot[:, None])[..., None]
        now = (sidx[None, :] == slot[:, None])[..., None]
        win_k = torch.where(before, ring[:, :, 0], torch.where(now, ik, torch.zeros_like(ik)))
        win_g = torch.where(before, ring[:, :, 1], torch.where(now, igate, torch.zeros_like(igate)))
        new_pool = kpool_compress(win_k, win_g, self.indexer.index_kpool_compress_ape)  # [n, D]
        closes = slot == kp - 1
        jp = torch.arange(P, device=x.device)
        pools = torch.where(((jp[None, :] == (p // kp)[:, None]) & closes[:, None])[..., None],
                            new_pool[:, None].to(pools.dtype), pools)
        pools = torch.where((jp[None, :] < (L // kp)[:, None])[..., None], pools,
                            torch.zeros_like(pools))
        scores = self.indexer.score(iq, iw, pools)                       # [n, 1, P]
        idx = self.indexer.select(scores.view(1, n, P), L).view(n, 1, -1)
        mask = indices_to_mask(idx.long(), ctx)
        out = self._attend(q, lat, mask, hidden_states.dtype)

        # -- writes: this token's latent and ring slot; the pool only if it closed
        pool_page = torch.where(live & closes, page, torch.full_like(page, sink))
        write_cache_rows(self.pages, latent_new[:, 0], lay.latent_row(write_page, tok))
        k_row, g_row = lay.tail_rows(write_page, tok)
        write_cache_rows(self.pages, torch.cat([ik[:, 0], igate[:, 0]]),
                         torch.cat([k_row, g_row]))
        write_cache_rows(self.pages, new_pool, lay.pool_row(pool_page, tok))
        return out[:, 0]
