"""Pure-PyTorch CPU reference for GLM-5.3-Flash (model_type=glm5_next) text decoder.

This is the logit-matching ORACLE for the Neuron port: every Neuron layer is validated
against the corresponding function here before optimization. It is deliberately
framework-agnostic — torch only, no transformers, no vllm, no neuronx_distributed —
so it runs anywhere, including a laptop with no Neuron toolchain.

Provenance
----------
Carried over from the NxDI fork's ``contrib/models/GLM-5.3-Flash/reference/
glm53_flash_reference.py`` (branch ``work/qwen38-27b-and-glm53-flash``), which
transcribed it from ``transformers/models/glm5_next/modeling_glm5_next.py`` (v5.16).
The math is unchanged from that CPU-verified version; only the module docstring and
the ``__main__`` block differ. NxDI is in maintenance mode (see
``personal_docs/STRATEGY.md`` §1), but this file never depended on it.

Config
------
Every constant in ``FlashCfg`` was re-verified against the live
``zai-org/GLM-5.3-Flash`` ``config.json`` on 2026-09-25, including that
``layer_types`` matches the ``i % 4 != 3`` rule exactly (sparse-MLA at
[3, 7, ..., 43] -> 34 KDA + 11 sparse-MLA) and that ``mlp_layer_types`` matches
``first_k_dense_replace = 3``.

KDA is not GDN
--------------
The one mathematical difference, and the thing most likely to be got wrong when
adapting a gated-DeltaNet kernel: nkilib's GDN scales the [K, V] state by a
per-head **scalar** ``exp(g)``; KDA scales each K row by its **own** ``exp(g[k])``.
``ForgetGate`` therefore emits ``[B, S, H, K]``, not ``[B, S, H]``, and both
``recurrent_kda`` and ``chunk_kda`` broadcast the decay over the K axis.
``tests/test_kda.py`` includes a discriminating test that fails if the scalar form
is substituted — the two agree closely enough on short, fast-decaying sequences
that a weak test cannot tell them apart.

Scope, and the DSA indexer
--------------------------
Text decoder only. The 11 sparse-MLA layers run the **real DSA indexer**
(``Indexer``), so the oracle is ground truth above 2048 tokens. Its practical
limit is CPU time, not correctness.

The algorithm, cross-checked against both vLLM
(``vllm/models/glm5next/{common,nvidia}/``) and transformers 5.17's
``Glm5NextTextIndexer`` — ``tests/test_indexer.py`` compares against both:

* ``k = LayerNorm(wk(x))`` (eps 1e-6, with bias), ``q = wq_b(q_c)`` where ``q_c``
  is the MLA's own ``q_a_layernorm(q_a_proj(x))``; per-head weights
  ``weights_proj(x) * n_heads**-0.5``; per-token gate
  ``x @ index_kpool_compress_gate.T``.
* Pool ``j`` covers tokens ``[4j, 4j + 4)``. Its key is
  ``sum_i softmax_i(gate_i + ape_i) * k_i``, the softmax taken over the 4 slots
  **per channel**.
* Pool score ``sum_h w_h * relu(head_dim**-0.5 * q_h . pool_k)``.
* Candidates are **complete pools only**: ``j < L // 4`` for a query seeing ``L``
  tokens. A pool holding future tokens is never scored.
* Top ``index_topk // index_kpool`` (512) pools are expanded back to tokens, and
  ``index_kpool_always_select_tail`` appends the ``L % 4`` raw tokens of the
  incomplete pool. Attention runs over exactly those tokens.
* ``qk_rope_head_dim`` is 0, so neither reference applies RoPE in the indexer, and
  ``indexer_rope_interleave`` has no effect on this model.

Two consequences that look like bugs and are not:

* Selecting everything is exact to seq_len **2051** (``index_topk + index_kpool
  - 1``), not 2048. vLLM's short-sequence gate at 2048 is conservative. See
  ``tests/test_dense_mla_exactness.py``.
* Above that, when ``L % 4 == 0`` the tail is empty and the query's own token
  must win top-k on merit. **A token is not guaranteed to attend to itself.** Do
  not force-include it: both references would disagree.

Scoring here is exact fp32. vLLM scores in FP8 — Hadamard-128 then e4m3 with
power-of-two scales on both q and pool keys — which reorders near-ties, so a
device and this oracle can legitimately disagree about *which* tokens are
selected. Measured 2026-09-25 (dev/progress) with the REAL indexer weights of
layers 3/23/43 and synthetic activations: FP8 changes at least one selected pool
on 94-96% of lossy rows at S=3003 and 100% at S=16387, where mean pool overlap
falls to ~0.97 (worst row 0.92); bf16 alone changes 39-42% -> 87-89%. The top-k
cut gap shrinks as context grows. Validate on device by score closeness, the
pool-overlap distribution, and attention with the device's own selection
injected (``topk_indices``), never by exact index equality. [unverified] with
real activations.

``FlashCfg.sparse_mla_dense = True`` restores plain causal attention, which is
exact for seq_len <= 2051 and is the baseline the indexer's continuity test
compares against.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- config
class FlashCfg:
    """Subset of text_config needed by the reference. Defaults = zai-org/GLM-5.3-Flash."""
    hidden_size = 4096
    num_hidden_layers = 45
    rms_norm_eps = 1e-5
    # mHC
    hc_mult = 4
    hc_sinkhorn_iters = 20
    hc_eps = 1e-6
    # KDA (linear_attn_config)
    linear_num_heads = 64
    linear_head_dim = 128
    linear_conv_kernel_dim = 4
    linear_lower_bound = -5.0
    # sparse-MLA (NoPE)
    num_attention_heads = 64
    q_lora_rank = 1536
    kv_lora_rank = 512
    qk_nope_head_dim = 256
    qk_rope_head_dim = 0
    v_head_dim = 256
    # DSA indexer (one per sparse-MLA layer; indexer_types is "full" on all 45)
    index_n_heads = 32
    index_head_dim = 128
    index_topk = 2048
    index_kpool = 4
    index_kpool_compress = True
    index_kpool_always_select_tail = True
    sparse_mla_dense = False        # oracle-only: skip the indexer (exact to seq 2051)
    # MoE
    n_routed_experts = 288
    num_experts_per_tok = 8
    moe_intermediate_size = 2048
    n_shared_experts = 1
    routed_scaling_factor = 2.5
    norm_topk_prob = True
    swiglu_limit = 10.0
    hidden_act = "silu"          # conv1d + MLP activation; the oracle hardcodes F.silu
    n_group = 1                  # TopkRouter omits group masking: identity only at 1
    topk_group = 1
    first_k_dense_replace = 3
    intermediate_size = 12288
    layer_types = ["linear_attention" if i % 4 != 3 else "deepseek_sparse_attention" for i in range(45)]
    mlp_layer_types = ["dense" if i < 3 else "sparse" for i in range(45)]

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        if "layer_types" not in kw:
            self.layer_types = ["linear_attention" if i % 4 != 3 else "deepseek_sparse_attention"
                                for i in range(self.num_hidden_layers)]
        if "mlp_layer_types" not in kw:
            self.mlp_layer_types = ["dense" if i < self.first_k_dense_replace else "sparse"
                                    for i in range(self.num_hidden_layers)]


# ------------------------------------------------------------------------------ norms
class RMSNorm(nn.Module):
    """rmsnorm in fp32, multiply by ``weight`` in fp32, cast once at the end.

    **A deliberate divergence from transformers, kept on purpose.** transformers casts
    the normalised value back to the input dtype BEFORE multiplying by ``weight``
    (``Glm5NextTextRMSNorm.forward``), i.e. one extra rounding. In fp32 the two are
    bit-identical, so this only shows up in bf16 -- where the oracle is *more* accurate.
    An oracle's job is maximum-precision ground truth, not reproducing one deployment's
    rounding; emulating it would make this a worse reference. Pinned by
    ``tests/test_norms_router.py``, which asserts fp32 equivalence and records the bf16
    gap so nobody later "fixes" the oracle into being less accurate.

    ``weight``, NOT ``(1 + weight)``. Qwen3.5 uses ``(1 + weight)``; GLM does not, and
    transformers agrees. Right today and nothing would notice if it stopped being, which
    is why there is a test.
    """
    def __init__(self, dim, eps):
        super().__init__(); self.weight = nn.Parameter(torch.ones(dim)); self.eps = eps
    def forward(self, x):
        xf = x.float()
        return (self.weight * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps))).to(x.dtype)

class UnweightedRMSNorm(nn.Module):
    def __init__(self, eps): super().__init__(); self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)

class RMSNormGated(nn.Module):
    """rmsnorm(x) * weight * **sigmoid**(gate). Per-head (dim = head_dim).

    The gate is ``sigmoid``, NOT ``silu``/``swish``. This is an architecture constant
    with no config key: ``linear_attn_config`` carries no ``output_gate_type``, and the
    top-level ``hidden_act: silu`` is the MLP's activation, not this one. Both references
    hardcode sigmoid -- transformers 5.17 ``Glm5NextTextRMSNormGated.activation =
    "sigmoid"``, and vLLM ``kda.py:291`` ``FusedRMSNormGated(head_dim,
    activation="sigmoid")``. FLA offers both branches and its *default is swish*
    (``y*g*sigmoid(g)``), so vLLM's ``activation="sigmoid"`` is a deliberate override,
    not an omission.

    This read silu until 2026-09-25, inherited from the Qwen3.8-27B lineage where
    ``output_gate_type: "swish"`` genuinely is silu. GLM-5.3-Flash is not that model.
    The error was worth 194% relative / cosine 0.64 on every one of the 34 KDA layers,
    and no test saw it: nothing compared this against an external reference, and
    ``test_layer_prefill_matches_step_decode`` compares the layer with itself, which a
    wrong activation satisfies exactly. ``tests/test_kda.py`` now pins it both ways.
    """
    def __init__(self, dim, eps):
        super().__init__(); self.weight = nn.Parameter(torch.ones(dim)); self.eps = eps
    def forward(self, x, gate):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()
        return (y * torch.sigmoid(gate.float())).to(x.dtype)

def l2norm(x, dim=-1, eps=1e-6):
    """``x / sqrt(sum(x^2) + eps)`` -- transformers/FLA's exact spelling.

    NOT ``x * rsqrt(...)``, which is **Qwen's GDN variant**. transformers carries the
    warning verbatim: "main difference to qwen's gdn variation: intentionally use sqrt
    and / to match original triton". The oracle used the Qwen form until 2026-09-25 --
    the third carry-over from that sibling model after the output gate and the 2051
    ceiling. Measured 2.98e-08 apart, 0.7-0.9 ULP, not bitwise equal: tiny, but the
    oracle's whole value is being exactly comparable to the reference family, so an
    unexplained ULP divergence is a cost with no benefit.
    """
    return x / torch.sqrt((x * x).sum(dim, keepdim=True) + eps)


# ------------------------------------------------------------------------------- mHC
class HyperConnection(nn.Module):
    """Manifold-Constrained Hyper-Connections. streams: [B,S,H,D] -> (post[B,S,H], comb[B,S,H,H], collapsed[B,S,D])"""
    def __init__(self, cfg: FlashCfg):
        super().__init__()
        H, D = cfg.hc_mult, cfg.hidden_size
        self.H, self.iters, self.eps = H, cfg.hc_sinkhorn_iters, cfg.hc_eps
        self.input_norm = UnweightedRMSNorm(cfg.rms_norm_eps)
        mix = (2 + H) * H                                   # 24 for H=4
        self.fn = nn.Parameter(torch.randn(mix, H * D) * 0.02)   # checkpoint: hc_{attn,ffn}_fn
        self.base = nn.Parameter(torch.zeros(mix))              # checkpoint: hc_{attn,ffn}_base
        self.scale = nn.Parameter(torch.ones(3))                # checkpoint: hc_{attn,ffn}_scale

    def forward(self, streams):
        H = self.H
        flat = self.input_norm(streams.flatten(2).float())            # [B,S,H*D]
        pre_w, post_w, comb_w = F.linear(flat, self.fn.float()).split([H, H, H * H], -1)
        pre_b, post_b, comb_b = self.base.split([H, H, H * H])
        s0, s1, s2 = self.scale.unbind(0)
        pre = torch.sigmoid(pre_w * s0 + pre_b) + self.eps                       # [B,S,H]
        post = 2 * torch.sigmoid(post_w * s1 + post_b)                           # [B,S,H]
        comb = torch.softmax(comb_w.view(*comb_w.shape[:-1], H, H) * s2 + comb_b.view(H, H), -1) + self.eps
        comb = comb / (comb.sum(-2, keepdim=True) + self.eps)                    # Sinkhorn-Knopp
        for _ in range(self.iters - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + self.eps)
            comb = comb / (comb.sum(-2, keepdim=True) + self.eps)
        collapsed = (pre.unsqueeze(-1) * streams).sum(2).to(streams.dtype)      # [B,S,D]
        return post, comb, collapsed

def hc_expand(post, comb, sublayer_out, residual_streams):
    """new_streams = post ⊗ out  +  comb^T @ residual_streams   -> [B,S,H,D]"""
    dt = residual_streams.dtype
    return post.to(dt).unsqueeze(-1) * sublayer_out.unsqueeze(-2) + torch.matmul(comb.to(dt).transpose(-1, -2), residual_streams)


# ------------------------------------------------------------------------------- KDA
class ForgetGate(nn.Module):
    """g = lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))  -> [B,S,H,K]  (per-CHANNEL, unlike GDN)"""
    def __init__(self, cfg: FlashCfg):
        super().__init__()
        H, K, D = cfg.linear_num_heads, cfg.linear_head_dim, cfg.hidden_size
        self.H, self.K = H, K
        self.f_a_proj = nn.Linear(D, K, bias=False)          # checkpoint: self_attn.f_a_proj
        self.f_b_proj = nn.Linear(K, H * K, bias=False)      # checkpoint: self_attn.f_b_proj
        self.dt_bias = nn.Parameter(torch.zeros(H * K))      # checkpoint: self_attn.dt_bias
        self.A_log = nn.Parameter(torch.zeros(H))            # checkpoint: self_attn.A_log
        self.lower = cfg.linear_lower_bound
    def forward(self, x):
        g = (self.f_b_proj(self.f_a_proj(x)).float() + self.dt_bias.float()).view(*x.shape[:2], self.H, self.K)
        decay = torch.exp(self.A_log.float()).view(1, 1, self.H, 1)
        if self.lower is not None:
            return self.lower * torch.sigmoid(decay * g)
        return -decay * torch.where(g > 20.0, g, torch.log1p(torch.exp(g)))

def recurrent_kda(q, k, v, g, beta, state):
    """Single-token decode. q,k,v: [B,1,H,K/V]; g: [B,1,H,K]; beta: [B,1,H]; state: [B,H,K,V] fp32.
    Differs from GDN ONLY in `state * exp(g)` being a per-K-row (per-channel) scale, not a scalar."""
    q, k, v, g, beta = [t.float() for t in (q, k, v, g, beta)]
    q, k = l2norm(q), l2norm(k)
    q = q * (q.shape[-1] ** -0.5)
    q_i, k_i, v_i = q[:, 0], k[:, 0], v[:, 0]                     # [B,H,K] / [B,H,V]
    state = state * g[:, 0].exp()[..., None]                        # [B,H,K,V] * [B,H,K,1]  <-- KDA
    kv = (state * k_i[..., None]).sum(-2)                           # k^T S   -> [B,H,V]
    delta = (v_i - kv) * beta[:, 0][..., None]
    state = state + k_i[..., None] * delta[..., None, :]            # + k ⊗ delta
    out = (state * q_i[..., None]).sum(-2)                          # q^T S   -> [B,H,V]
    return out.unsqueeze(1), state

def chunk_kda(q, k, v, g, beta, state=None, chunk=64):
    """Prefill. Transcribed from chunk_kimi_delta_attention; per-channel cumulative decay."""
    dt = q.dtype
    q, k, v, beta, g = [t.transpose(1, 2).contiguous().float() for t in (q, k, v, beta, g)]  # [B,H,T,*]
    q, k = l2norm(q), l2norm(k)
    B, H, T, K = k.shape; V = v.shape[-1]
    pad = (chunk - T % chunk) % chunk; Tp = T + pad
    q = F.pad(q, (0, 0, 0, pad)) * (K ** -0.5); k = F.pad(k, (0, 0, 0, pad)); v = F.pad(v, (0, 0, 0, pad))
    g = F.pad(g, (0, 0, 0, pad)); beta = F.pad(beta, (0, pad))
    v_beta, k_beta = v * beta[..., None], k * beta[..., None]
    rs = lambda t: t.reshape(B, H, -1, chunk, t.shape[-1])
    q, k, v, g, k_beta, v_beta = map(rs, (q, k, v, g, k_beta, v_beta))
    g = g.cumsum(-2)                                                              # [B,H,C,c,K]
    tri = torch.triu(torch.ones(chunk, chunk, dtype=torch.bool), 0); stri = torch.triu(torch.ones(chunk, chunk, dtype=torch.bool), 1)
    decay = (g.unsqueeze(-2) - g.unsqueeze(-3)).masked_fill(stri[..., None], float("-inf")).exp()  # [B,H,C,c,c,K]
    attn = -(k_beta.unsqueeze(-2) * k.unsqueeze(-3) * decay).sum(-1).masked_fill(tri, 0)
    for i in range(1, chunk):
        row, sub = attn[..., i, :i].clone(), attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk)
    v = attn @ v_beta; k_cumdecay = attn @ (k_beta * g.exp())
    S = torch.zeros(B, H, K, V) if state is None else state.float()
    out = torch.zeros_like(v)
    for i in range(Tp // chunk):
        q_i, k_i, v_i, g_i = q[:, :, i], k[:, :, i], v[:, :, i], g[:, :, i]
        inter = (q_i * g_i.exp()) @ S
        intra = (q_i.unsqueeze(-2) * k_i.unsqueeze(-3) * decay[:, :, i]).sum(-1).masked_fill(stri, 0)
        v_new = v_i - k_cumdecay[:, :, i] @ S
        out[:, :, i] = inter + intra @ v_new
        S = S * g_i[:, :, -1].exp().unsqueeze(-1) + (k_i * (g_i[:, :, -1:] - g_i).exp()).transpose(-1, -2) @ v_new
    out = out.reshape(B, H, -1, V)[:, :, :T].transpose(1, 2).contiguous().to(dt)
    return out, S

class LinearAttention(nn.Module):
    def __init__(self, cfg: FlashCfg):
        super().__init__()
        H, K, D = cfg.linear_num_heads, cfg.linear_head_dim, cfg.hidden_size
        self.H, self.K, self.HK = H, K, H * K
        self.q_proj = nn.Linear(D, H * K, bias=False); self.k_proj = nn.Linear(D, H * K, bias=False); self.v_proj = nn.Linear(D, H * K, bias=False)
        # checkpoint stores THREE depthwise conv1ds (q_conv1d, k_conv1d, v_conv1d); modeling concatenates them channel-wise
        self.conv1d = nn.Conv1d(3 * H * K, 3 * H * K, cfg.linear_conv_kernel_dim, groups=3 * H * K, bias=False, padding=cfg.linear_conv_kernel_dim - 1)
        self.forget_gate = ForgetGate(cfg)
        self.b_proj = nn.Linear(D, H, bias=False)
        self.g_a_proj = nn.Linear(D, K, bias=False); self.g_b_proj = nn.Linear(K, H * K, bias=False)
        self.o_norm = RMSNormGated(K, cfg.rms_norm_eps)
        self.o_proj = nn.Linear(H * K, D, bias=False)
    def forward(self, x, conv_state=None, rec_state=None):
        """-> (out [B,S,D], (conv_state [B,3HK,kernel], rec_state [B,H,K,V]))

        Both states are *returned* rather than mutated in place, so a caller
        stepping one token at a time must reassign them. ``torch.roll`` below
        allocates, so an in-place-only contract would silently strand the conv
        window at its initial value.
        """
        B, S, _ = x.shape
        kernel = self.conv1d.weight.shape[-1]
        qkv = torch.cat([self.q_proj(x), self.k_proj(x), self.v_proj(x)], -1).transpose(1, 2)   # [B,3HK,S]
        if conv_state is not None and S == 1:
            conv_state = torch.roll(conv_state, -1, -1); conv_state[..., -1] = qkv[..., 0]
            qkv = F.silu((conv_state * self.conv1d.weight.squeeze(1)).sum(-1)).unsqueeze(-1)
        else:
            if conv_state is not None: qkv = torch.cat([conv_state, qkv], -1)
            pre = qkv                                                       # pre-activation window
            qkv = F.silu(self.conv1d(qkv)[..., :qkv.shape[-1]])[..., -S:]   # causal: first T outputs, then last S (drops prepended conv_state)
            # Hand off to decode: the newest `kernel` pre-conv columns, newest last.
            conv_state = F.pad(pre, (max(0, kernel - pre.shape[-1]), 0))[..., -kernel:]
        q, k, v = qkv.transpose(1, 2).split([self.HK] * 3, -1)
        q, k, v = [t.reshape(B, S, self.H, self.K) for t in (q, k, v)]
        g = self.forget_gate(x); beta = torch.sigmoid(self.b_proj(x))
        if rec_state is not None and S == 1: core, rec_state = recurrent_kda(q, k, v, g, beta, rec_state)
        else: core, rec_state = chunk_kda(q, k, v, g, beta, rec_state)
        gate = self.g_b_proj(self.g_a_proj(x)).view(B, S, self.H, self.K)
        return self.o_proj(self.o_norm(core, gate).reshape(B, S, -1)), (conv_state, rec_state)


# ------------------------------------------------------------------------ DSA indexer
def kpool_compress(k, gate, ape):
    """Pool key = softmax over the kpool slots, per channel, of (gate + ape), weighting raw K.
    k, gate: [..., kpool, D]; ape: [kpool, D] -> [..., D] fp32.
    vLLM's _kpool_softmax_rotate_write_cache_kernel without its Hadamard + FP8 quant."""
    p = torch.softmax(gate.float() + ape.float(), dim=-2)
    return (p * k.float()).sum(-2)

def index_scores(q, w, pool_k, scale, rows=256):
    """sum_h w[h] * relu(scale * q_h . pool_k[j])  -> [B,S,P] fp32.
    q: [B,S,Hi,D]; w: [B,S,Hi]; pool_k: [B,P,D]. Row-chunked: [B,S,Hi,P] is large at long S."""
    B, S = q.shape[:2]
    out = torch.zeros(B, S, pool_k.shape[1])
    kT = pool_k.float().transpose(-1, -2).unsqueeze(1)                       # [B,1,D,P]
    for s0 in range(0, S, rows):
        sc = F.relu((q[:, s0:s0 + rows].float() @ kT) * scale)              # [B,s,Hi,P]
        out[:, s0:s0 + rows] = (w[:, s0:s0 + rows, None, :].float() @ sc).squeeze(-2)
    return out

def select_tokens(scores, lens, topk, kpool, tail=True):
    """Pool scores -> token indices [B,S,topk (+ kpool-1 if tail)], -1 padded.

    lens: [S] tokens visible to each row. Only complete pools (j < lens // kpool) are
    candidates; the top min(topk // kpool, candidates) are expanded to tokens, then the
    incomplete pool's lens % kpool tokens are appended raw. Same layout as vLLM's
    expand_pools_and_append_tail.
    """
    B, S, P = scores.shape
    n_complete = lens // kpool                                               # [S]
    cand = torch.arange(P)[None, :] < n_complete[:, None]                   # [S,P]
    top = scores.masked_fill(~cand, float("-inf")).topk(min(topk // kpool, P), -1).indices
    ok = cand.expand(B, S, P).gather(-1, top)
    tok = (top[..., None] * kpool + torch.arange(kpool)).masked_fill(~ok[..., None], -1).flatten(-2)
    out = F.pad(tok, (0, topk - tok.shape[-1]), value=-1)
    if not tail:
        return out
    off = torch.arange(kpool - 1)
    start = n_complete * kpool
    t = (start[:, None] + off).masked_fill(off >= (lens - start)[:, None], -1)
    return torch.cat([out, t.expand(B, S, -1)], -1)

def indices_to_mask(idx, L):
    """[B,S,W] token indices (-1 = empty) -> bool [B,S,L]."""
    safe = torch.where(idx < 0, L, idx)
    return torch.zeros(*idx.shape[:2], L + 1, dtype=torch.bool).scatter_(-1, safe, True)[..., :L]

def injected_mask(idx, L):
    """Mask from externally supplied token indices [B,S,W] (-1 = empty), e.g. a device's
    captured topk_indices_buffer, for the query rows L-S .. L-1. Order and width are free.

    Refuses what no correct indexer emits -- an index < -1 or >= L, a future token, a
    duplicate -- because indices_to_mask would silently absorb it, and a device selection
    bug would then show up as an unexplained attention mismatch. (vLLM never duplicates:
    selected pools are distinct and lie below the tail.)"""
    S = idx.shape[1]
    if ((idx < -1) | (idx >= L)).any():
        raise ValueError(f"injected index out of range [-1, {L})")
    if ((idx >= 0) & (idx > torch.arange(L - S, L)[:, None])).any():
        raise ValueError("injected indices select a future token")
    srt = idx.sort(-1).values
    if ((srt[..., 1:] == srt[..., :-1]) & (srt[..., 1:] >= 0)).any():
        raise ValueError("injected indices contain a duplicate")
    return indices_to_mask(idx, L)

@dataclass
class IndexerState:
    """Per-layer indexer cache: vLLM's two indexer caches, minus paging and FP8.

    pool_k     [B,P,D]      compressed COMPLETE pools only   (Glm5NextIndexerCache; scoring only)
    tail_k     [B,kpool,D]  raw K ring, slot = pos % kpool   (Glm5NextTailCache, "K" half)
    tail_gate  [B,kpool,D]  raw gate-score ring, same slots  (Glm5NextTailCache, "V" half)
    length     tokens seen
    Attention never reads any of this; it reads the MLA latents by token index.
    """
    pool_k: torch.Tensor
    tail_k: torch.Tensor
    tail_gate: torch.Tensor
    length: int

def indexer_prefill(state, k, gate, ape, kpool):
    """Append S tokens (continuing ``state`` if given) -> new state; inputs are not mutated.
    Completes every pool it can, then seeds the ring with the newest min(kpool, S) tokens at
    pos % kpool, as vLLM's kpool_seed_tail_cache does."""
    B, S, D = k.shape
    if state is None:
        z = k.new_zeros(B, kpool, D)
        state = IndexerState(torch.zeros(B, 0, D), z, z.clone(), 0)
    r = state.length % kpool                          # incomplete pool's tokens sit in ring slots [0, r)
    raw_k, raw_g = torch.cat([state.tail_k[:, :r], k], 1), torch.cat([state.tail_gate[:, :r], gate], 1)
    n = raw_k.shape[1] // kpool
    pools = kpool_compress(raw_k[:, :n * kpool].unflatten(1, (n, kpool)), raw_g[:, :n * kpool].unflatten(1, (n, kpool)), ape)
    tail_k, tail_g = state.tail_k.clone(), state.tail_gate.clone()
    for i in range(max(0, S - kpool), S):
        slot = (state.length + i) % kpool
        tail_k[:, slot], tail_g[:, slot] = k[:, i], gate[:, i]
    return IndexerState(torch.cat([state.pool_k, pools], 1), tail_k, tail_g, state.length + S)

def indexer_decode(state, k, gate, ape, kpool):
    """One token, as vLLM's _kpool_decode_update_batched_kernel: if it completes a pool,
    compress ring slots [0, kpool-1) plus itself; then stash it at pos % kpool.
    Every token is stashed, not only pool-completing ones -- vLLM once gated the stash on
    completion and compressed stale prompt-tail entries forever after."""
    slot = state.length % kpool
    pool_k = state.pool_k
    if slot == kpool - 1:
        ks, gs = torch.cat([state.tail_k[:, :slot], k], 1), torch.cat([state.tail_gate[:, :slot], gate], 1)
        pool_k = torch.cat([pool_k, kpool_compress(ks, gs, ape)[:, None]], 1)
    tail_k, tail_g = state.tail_k.clone(), state.tail_gate.clone()
    tail_k[:, slot], tail_g[:, slot] = k[:, 0], gate[:, 0]
    return IndexerState(pool_k, tail_k, tail_g, state.length + 1)

class Indexer(nn.Module):
    """DSA lightning indexer with kpool compression. -> (token indices [B,S,topk+kpool-1], state).
    Parameter names are the checkpoint's (self_attn.indexer.*)."""
    def __init__(self, cfg: FlashCfg):
        super().__init__()
        assert cfg.index_kpool_compress and cfg.index_kpool > 1, "only the kpool-compressed indexer is implemented"
        assert cfg.index_topk % cfg.index_kpool == 0, "vLLM asserts this (history_group_budget_for_topk)"
        Hi, D = cfg.index_n_heads, cfg.index_head_dim
        self.Hi, self.D, self.topk, self.kpool = Hi, D, cfg.index_topk, cfg.index_kpool
        self.tail = cfg.index_kpool_always_select_tail
        self.wq_b = nn.Linear(cfg.q_lora_rank, Hi * D, bias=False)
        self.wk = nn.Linear(cfg.hidden_size, D, bias=False)
        self.k_norm = nn.LayerNorm(D, eps=1e-6)                                         # has a bias
        self.weights_proj = nn.Linear(cfg.hidden_size, Hi, bias=False)
        self.index_kpool_compress_ape = nn.Parameter(torch.zeros(self.kpool, D))        # [kpool, D]
        self.index_kpool_compress_gate = nn.Parameter(torch.randn(D, cfg.hidden_size) * 0.02)  # [D, hidden]
    def forward(self, x, q_c, state=None):
        B, S, _ = x.shape
        q = self.wq_b(q_c).view(B, S, self.Hi, self.D)
        k = self.k_norm(self.wk(x))
        gate = F.linear(x, self.index_kpool_compress_gate)
        w = self.weights_proj(x).float() * self.Hi ** -0.5
        ape = self.index_kpool_compress_ape
        if state is not None and S == 1: state = indexer_decode(state, k, gate, ape, self.kpool)
        else: state = indexer_prefill(state, k, gate, ape, self.kpool)
        lens = torch.arange(state.length - S, state.length) + 1
        scores = index_scores(q, w, state.pool_k, self.D ** -0.5)
        return select_tokens(scores, lens, self.topk, self.kpool, self.tail), state


# --------------------------------------------------------------------------- NoPE MLA
class SparseMLAttention(nn.Module):
    """DeepSeek-V3 MLA with qk_rope_head_dim=0 (NoPE), attending only to the tokens the indexer selects."""
    def __init__(self, cfg: FlashCfg):
        super().__init__()
        D, Hh = cfg.hidden_size, cfg.num_attention_heads
        self.Hh, self.qk, self.vd, self.kvr = Hh, cfg.qk_nope_head_dim + cfg.qk_rope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank
        self.q_a_proj = nn.Linear(D, cfg.q_lora_rank, bias=False); self.q_a_layernorm = RMSNorm(cfg.q_lora_rank, cfg.rms_norm_eps)
        self.q_b_proj = nn.Linear(cfg.q_lora_rank, Hh * self.qk, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(D, cfg.kv_lora_rank + cfg.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = RMSNorm(cfg.kv_lora_rank, cfg.rms_norm_eps)
        self.kv_b_proj = nn.Linear(cfg.kv_lora_rank, Hh * (cfg.qk_nope_head_dim + cfg.v_head_dim), bias=False)
        self.o_proj = nn.Linear(Hh * cfg.v_head_dim, D, bias=False)
        self.indexer = Indexer(cfg)
        self.dense = cfg.sparse_mla_dense
        self.scaling = self.qk ** -0.5
    def forward(self, x, state=None, topk_indices=None):
        """-> (out [B,S,D], (latent [B,L,kv_lora_rank], IndexerState | None)).
        ``dense`` skips the indexer and attends causally to everything: exact only to seq 2051.

        ``topk_indices`` [B,S,W] (-1 = empty) replaces the indexer's selection for this call,
        so attention can be checked against a device *given the device's own selection*.
        FP8 scoring legitimately selects different pools (see module docstring), and without
        this a real attention bug hides behind that divergence. The indexer still runs, so
        the state advances exactly as it would have; only the mask changes."""
        B, S, _ = x.shape
        kv_cache, idx_state = state if state is not None else (None, None)
        q_c = self.q_a_layernorm(self.q_a_proj(x))                                           # also the indexer's q input
        q = self.q_b_proj(q_c).view(B, S, self.Hh, self.qk).transpose(1, 2)
        latent = self.kv_a_layernorm(self.kv_a_proj_with_mqa(x)[..., :self.kvr])          # [B,S,512] -- THE KV cache entry
        if kv_cache is not None: latent = torch.cat([kv_cache, latent], 1)
        kv = self.kv_b_proj(latent).view(B, -1, self.Hh, self.qk + self.vd).transpose(1, 2)
        k, v = kv.split([self.qk, self.vd], -1)
        L = k.shape[2]
        if self.dense:
            mask = torch.ones(S, L, dtype=torch.bool).tril(L - S)
        else:
            assert (idx_state.length if idx_state is not None else 0) == L - S, "indexer state out of step with the KV cache"
            idx, idx_state = self.indexer(x, q_c, idx_state)
            mask = indices_to_mask(idx, L).unsqueeze(1)                                   # [B,1,S,L]
        if topk_indices is not None:
            assert topk_indices.shape[:2] == (B, S), "topk_indices must be [B, S, W]"
            mask = injected_mask(topk_indices.long(), L).unsqueeze(1)
        att = (q @ k.transpose(-1, -2)) * self.scaling
        # fp32 softmax, made explicit. transformers' eager_attention_forward spells this
        # as softmax(dtype=float32).to(query.dtype); the oracle keeps the fp32 result and
        # does NOT cast back down -- same reasoning as RMSNorm above. The downcast is a
        # deployment choice, and an oracle should not adopt one. torch already accumulates
        # bf16 softmax in fp32, so the explicit dtype is about the contract, not accuracy.
        att = att.masked_fill(~mask, float("-inf")).softmax(-1, dtype=torch.float32)
        return self.o_proj((att @ v).transpose(1, 2).reshape(B, S, -1)), (latent, idx_state)


# --------------------------------------------------------------------------------- MoE
class MLP(nn.Module):
    """Clamped SwiGLU: gate to (-inf, limit], up to [-limit, limit], THEN silu.

    The clamp is not optional and not a numerical guard -- transformers calls it
    "the key difference" (``Glm5NextTextMLP.forward``) and ``swiglu_limit`` is 10.0
    in the live config. ``MoE``'s routed experts below have always had it; this class
    did not, which meant the dense FFN on layers 0-2 AND ``shared_experts`` on all 42
    MoE layers -- **45 of 45 layers** -- ran unclamped until 2026-09-25.

    ``limit`` is REQUIRED, deliberately. It used to default to 10.0 -- which is the
    real ``swiglu_limit``, so a call site that forgot to wire it was indistinguishable
    from one that did. A required argument makes that failure impossible to construct
    rather than merely detectable.

    It is invisible until activations reach the limit (see tests/test_mlp_moe.py):
    at gate/up std 1.0 nothing clamps at all, so a reference comparison on
    small-init weights passes with the bug present. Found by dev1's provenance audit.
    """
    def __init__(self, D, I, limit):
        super().__init__(); self.gate_proj = nn.Linear(D, I, bias=False); self.up_proj = nn.Linear(D, I, bias=False); self.down_proj = nn.Linear(I, D, bias=False)
        self.limit = limit
    def forward(self, x):
        g = self.gate_proj(x).clamp(max=self.limit)
        u = self.up_proj(x).clamp(-self.limit, self.limit)
        return self.down_proj(F.silu(g) * u)

class TopkRouter(nn.Module):
    """sigmoid + e_score_correction_bias for CHOICE, raw sigmoid scores for WEIGHTS, norm to 1, x2.5. n_group=1 -> group logic is identity."""
    def __init__(self, cfg: FlashCfg):
        super().__init__()
        self.top_k, self.E = cfg.num_experts_per_tok, cfg.n_routed_experts
        self.weight = nn.Parameter(torch.randn(self.E, cfg.hidden_size) * 0.02)      # checkpoint: mlp.gate.weight
        self.e_score_correction_bias = nn.Parameter(torch.zeros(self.E))               # checkpoint: mlp.gate.e_score_correction_bias (fp32)
        self.scale, self.norm = cfg.routed_scaling_factor, cfg.norm_topk_prob
    def forward(self, x):
        scores = F.linear(x.float(), self.weight.float()).sigmoid()
        idx = torch.topk(scores + self.e_score_correction_bias, self.top_k, -1, sorted=False).indices
        w = scores.gather(1, idx)
        if self.norm: w = w / (w.sum(-1, keepdim=True) + 1e-20)
        return w * self.scale, idx

class MoE(nn.Module):
    def __init__(self, cfg: FlashCfg):
        super().__init__()
        D, I, E = cfg.hidden_size, cfg.moe_intermediate_size, cfg.n_routed_experts
        self.gate = TopkRouter(cfg); self.limit = cfg.swiglu_limit
        self.gate_up_proj = nn.Parameter(torch.randn(E, 2 * I, D) * 0.02)   # checkpoint: experts.N.{gate,up}_proj (separate)
        self.down_proj = nn.Parameter(torch.randn(E, D, I) * 0.02)          # checkpoint: experts.N.down_proj
        self.shared_experts = MLP(D, I * cfg.n_shared_experts, cfg.swiglu_limit)
    def forward(self, x):
        B, S, D = x.shape; flat = x.view(-1, D)
        w, idx = self.gate(flat)
        out = torch.zeros_like(flat)
        for e in idx.unique():
            tok, pos = torch.where(idx == e)
            gu = F.linear(flat[tok], self.gate_up_proj[e]); g, u = gu.chunk(2, -1)
            h = F.silu(g.clamp(max=self.limit)) * u.clamp(-self.limit, self.limit)   # swiglu_limit=10
            out.index_add_(0, tok, (F.linear(h, self.down_proj[e]) * w[tok, pos, None]).to(out.dtype))
        return out.view(B, S, D) + self.shared_experts(x)


# ------------------------------------------------------------------------- decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, cfg: FlashCfg, i: int):
        super().__init__()
        self.linear = cfg.layer_types[i] == "linear_attention"
        self.self_attn = LinearAttention(cfg) if self.linear else SparseMLAttention(cfg)
        self.mlp = MoE(cfg) if cfg.mlp_layer_types[i] == "sparse" else MLP(cfg.hidden_size, cfg.intermediate_size, cfg.swiglu_limit)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps); self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn_hc = HyperConnection(cfg); self.ffn_hc = HyperConnection(cfg)   # checkpoint: hc_attn_*, hc_ffn_*
    def forward(self, streams, state=None):
        res = streams
        post, comb, h = self.attn_hc(streams)
        h = self.input_layernorm(h)
        if self.linear: h, state = self.self_attn(h, *(state or (None, None)))
        else: h, state = self.self_attn(h, state)
        streams = hc_expand(post, comb, h, res)
        res = streams
        post, comb, h = self.ffn_hc(streams)
        h = self.mlp(self.post_attention_layernorm(h))
        return hc_expand(post, comb, h, res), state

class FlashTextModel(nn.Module):
    def __init__(self, cfg: FlashCfg, vocab=154880):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(vocab, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, vocab, bias=False)
    def forward(self, ids):
        h = self.embed_tokens(ids)
        streams = h.unsqueeze(2).expand(-1, -1, self.cfg.hc_mult, -1).contiguous()   # [B,S,H,D]
        for layer in self.layers: streams, _ = layer(streams)
        return self.lm_head(self.norm(streams.mean(2)))                                  # HyperHead = unweighted mean


def tiny_cfg(**kw) -> FlashCfg:
    """A small config with every structural feature of the real one still present.

    8 layers keeps the [linear x3, sparse-MLA] x2 pattern and both MLP kinds
    (``first_k_dense_replace=3`` -> layers 0-2 dense, 3+ MoE). Shapes shrink;
    nothing is removed. ``index_topk=16`` (kpool stays 4) puts the indexer's lossy
    regime at seq_len >= 20, so it is cheap to reach.
    """
    base = dict(hidden_size=256, num_hidden_layers=8, linear_num_heads=4, linear_head_dim=64,
                num_attention_heads=4, q_lora_rank=96, kv_lora_rank=64, qk_nope_head_dim=64,
                v_head_dim=64, n_routed_experts=8, num_experts_per_tok=2,
                moe_intermediate_size=128, intermediate_size=512,
                index_n_heads=4, index_head_dim=32, index_topk=16)
    base.update(kw)
    return FlashCfg(**base)


def step_decode(la: "LinearAttention", x: torch.Tensor):
    """Drive ``la``'s own single-token decode path over ``x`` -> [B, S, D].

    Calls ``la.forward`` once per token with a carried (conv_state, rec_state),
    i.e. it exercises the shipped decode branch rather than reimplementing it.
    ``conv_state`` is ``[B, 3HK, kernel]``, which is what that branch's
    roll-and-overwrite expects.
    """
    B, S, _ = x.shape
    kernel = la.conv1d.weight.shape[-1]
    cs = torch.zeros(B, 3 * la.HK, kernel, dtype=x.dtype)
    rs = torch.zeros(B, la.H, la.K, la.K, dtype=torch.float32)
    outs = []
    for t in range(S):
        out, (cs, rs) = la(x[:, t:t + 1], conv_state=cs, rec_state=rs)
        outs.append(out)
    return torch.cat(outs, 1)


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = tiny_cfg()
    m = FlashTextModel(cfg, vocab=1000).eval()
    with torch.no_grad():
        logits = m(torch.randint(0, 1000, (2, 37)))
    print("logits", tuple(logits.shape), "finite:", bool(torch.isfinite(logits).all()))
    print("Real validation lives in tests/ — this block is only a shape smoke.")
