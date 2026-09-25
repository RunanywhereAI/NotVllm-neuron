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
The write path therefore goes through a latent-only scatter (dev3's
``NF.write_latent_cache``), never a K/V-shaped one.

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


@dataclass(frozen=True)
class MLAParams:
    """The config surface this layer reads. dev1 owns the config object."""

    num_heads: int = 64
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 256
    qk_rope_head_dim: int = 0
    v_head_dim: int = 256
    hidden_size: int = 4096
    rms_norm_eps: float = 1e-5
    index_topk: int = 2048
    index_kpool: int = 4
    index_n_heads: int = 32
    index_head_dim: int = 128

    @classmethod
    def from_config(cls, config) -> "MLAParams":
        g = lambda k, d: getattr(config, k, d)
        p = cls(
            num_heads=g("num_attention_heads", 64),
            q_lora_rank=g("q_lora_rank", 1536),
            kv_lora_rank=g("kv_lora_rank", 512),
            qk_nope_head_dim=g("qk_nope_head_dim", 256),
            qk_rope_head_dim=g("qk_rope_head_dim", 0),
            v_head_dim=g("v_head_dim", 256),
            hidden_size=g("hidden_size", 4096),
            rms_norm_eps=g("rms_norm_eps", 1e-5),
            index_topk=g("index_topk", 2048),
            index_kpool=g("index_kpool", 4),
            index_n_heads=g("index_n_heads", 32),
            index_head_dim=g("index_head_dim", 128),
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

    def forward(self, x, q_c, pool_k=None):
        """-> token indices ``[B, S, index_topk + index_kpool - 1]``, ``-1`` padded.

        No RoPE: ``qk_rope_head_dim`` is 0, so ``indexer_rope_interleave`` is inert.
        """
        B, S, _ = x.shape
        q = self.wq_b(q_c).view(B, S, self.Hi, self.D)
        k = self.k_norm(self.wk(x))
        gate = F.linear(x, self.index_kpool_compress_gate)
        w = self.weights_proj(x).float() * self.Hi ** -0.5
        if pool_k is None:                                # prefill: pool this sequence
            n = S // self.p.index_kpool
            kk = k[:, : n * self.p.index_kpool].unflatten(1, (n, self.p.index_kpool))
            gg = gate[:, : n * self.p.index_kpool].unflatten(1, (n, self.p.index_kpool))
            pool_k = kpool_compress(kk, gg, self.index_kpool_compress_ape)
        scores = (w[:, :, None, :].float() @ F.relu(
            (q.float() @ pool_k.float().transpose(-1, -2).unsqueeze(1)) * self.D ** -0.5)
        ).squeeze(-2)
        # THE capture point for on-device validation. Exact index equality fails a
        # CORRECT fp8 device, so the scores are what distinguish a near-tie swap from
        # a real defect. Indices alone cannot.
        _capture_tensor(f"{self.layer_name}.indexer.scores", scores)
        lens = torch.arange(S, device=x.device) + 1
        idx = select_tokens(scores, lens, self.p.index_topk, self.p.index_kpool)
        _capture_tensor(f"{self.layer_name}.indexer.topk_indices", idx)
        return idx


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

    def forward_core(self, hidden_states, kv_cache=None, topk_indices=None):
        """-> (output ``[B,S,D]``, latent ``[B,L,kvr]``).

        Absorbed form: ``q @ W_uk`` scores directly against the latent, and V-up is
        applied *after* attention. Returning the latent is for tests and for callers
        managing the cache directly; the framework path scatters it instead.

        ``topk_indices`` replaces the indexer's selection for this call **without**
        skipping the indexer, so state still advances. See the module docstring for
        why this override exists.
        """
        B, S, _ = hidden_states.shape
        q_c = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_c).view(B, S, self.H, self.qk)
        latent = self.kv_a_layernorm(self.kv_a_proj_with_mqa(hidden_states)[..., : self.kvr])
        if kv_cache is not None:
            latent = torch.cat([kv_cache, latent], 1)
        _capture_tensor(f"{self.layer_name}.latent", latent)
        L = latent.shape[1]

        W_uk, W_uv = self._uk_uv()
        # q_absorbed[b,s,h,:] = q[b,s,h,:] @ W_uk[h]  -> scores against the raw latent
        q_abs = torch.einsum("bshq,hqr->bshr", q.float(), W_uk.float())
        att = torch.einsum("bshr,blr->bhsl", q_abs, latent.float()) * self.scaling

        own = self.indexer(hidden_states, q_c)
        idx = own if topk_indices is None else topk_indices
        mask = indices_to_mask(idx.long(), L).unsqueeze(1)
        att = att.masked_fill(~mask, float("-inf")).softmax(-1, dtype=torch.float32)
        # attention mixes the LATENT; V-up comes after, which is also why the decode
        # kernel's in-kernel o_proj is unusable here (and forbidden at d_head=512).
        mixed = torch.einsum("bhsl,blr->bshr", att, latent.float())
        out = torch.einsum("bshr,hvr->bshv", mixed, W_uv.float())
        _capture_tensor(f"{self.layer_name}.attn_pre_oproj", out)
        return self.o_proj(out.reshape(B, S, -1).to(hidden_states.dtype)), latent

    def forward(self, hidden_states, positions, attn_metadata: dict) -> torch.Tensor:
        """Framework entry point.

        NOT WIRED: the latent page is sliced by ``glm5_next/cache_layout.py`` (dev3's,
        and not yet present) and written by ``NF.write_latent_cache``. Deliberately
        left as the one seam rather than computing page offsets here — a second copy
        of that arithmetic aliases memory silently instead of failing loudly.

        Note also that ``topk_indices`` is **not** reachable from this path. It is a
        validation affordance on ``forward_core`` only, so it cannot be left on
        accidentally in a serving path.
        """
        raise NotImplementedError(
            "needs cache_layout.py offsets and NF.write_latent_cache (dev3's surface)"
        )
