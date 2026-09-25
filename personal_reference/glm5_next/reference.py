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

Scope and validity ceiling
--------------------------
Text decoder only. The 11 sparse-MLA layers run DENSE causal attention with no
indexer, which is **exact for seq_len <= 2051 and wrong above it** — not
approximate above it: the oracle attends to tokens the model's indexer excludes.

Verified 2026-09-25 against vLLM's implementation, not by re-deriving the pool
arithmetic. The question that mattered was whether ``index_kpool_compress`` means
attention runs over compressed pool representatives (which would make selecting
every pool still lossy). It does not: vLLM keeps the compressed entries in a
separate ``Glm5NextIndexerCache`` used only for scoring, keeps the in-progress
pool's **raw** K in ``Glm5NextTailCache``, and has attention gather full-fidelity
tokens by the token indices the indexer emits. vLLM then takes this very shortcut
itself — "Short sequences select every pool, so skip sparse scoring and fill the
top-k buffer with all causal token indices" for prefill, and
``_fill_short_decode_causal_indices`` for decode — both gated on
``seq_len <= topk_tokens`` where ``topk_tokens = config.index_topk = 2048`` —
three tokens conservative against the true 2051 boundary, which is
``floor(S / kpool) <= index_topk // kpool`` (corrected by dev2; only *complete*
pools are selection candidates, the incomplete tail is appended as raw tokens).

``tests/test_dense_mla_exactness.py`` pins the five things this depends on, so a
config change trips a test rather than silently invalidating the oracle.

**Milestone 2 targets 1M context; this oracle reaches 2051.** Validating anything
longer needs a real indexer here.
"""
from __future__ import annotations
import math
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
    # MoE
    n_routed_experts = 288
    num_experts_per_tok = 8
    moe_intermediate_size = 2048
    n_shared_experts = 1
    routed_scaling_factor = 2.5
    norm_topk_prob = True
    swiglu_limit = 10.0
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
    """FLA-style: rmsnorm(x) * weight * silu(gate). Per-head (dim = head_dim)."""
    def __init__(self, dim, eps):
        super().__init__(); self.weight = nn.Parameter(torch.ones(dim)); self.eps = eps
    def forward(self, x, gate):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()
        return (y * F.silu(gate.float())).to(x.dtype)

def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim, keepdim=True) + eps)


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


# --------------------------------------------------------------------------- NoPE MLA
class SparseMLAttention(nn.Module):
    """DeepSeek-V3 MLA with qk_rope_head_dim=0 (NoPE). Dense causal attention == DSA for seq<=2051 (see module docstring)."""
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
        self.scaling = self.qk ** -0.5
    def forward(self, x, kv_cache=None):
        B, S, _ = x.shape
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x))).view(B, S, self.Hh, self.qk).transpose(1, 2)
        latent = self.kv_a_layernorm(self.kv_a_proj_with_mqa(x)[..., :self.kvr])          # [B,S,512] -- THE KV cache entry
        if kv_cache is not None: latent = torch.cat([kv_cache, latent], 1)
        kv = self.kv_b_proj(latent).view(B, -1, self.Hh, self.qk + self.vd).transpose(1, 2)
        k, v = kv.split([self.qk, self.vd], -1)
        L = k.shape[2]
        mask = torch.ones(S, L, dtype=torch.bool).tril(L - S)
        att = (q @ k.transpose(-1, -2)) * self.scaling
        att = att.masked_fill(~mask, float("-inf")).softmax(-1)
        return self.o_proj((att @ v).transpose(1, 2).reshape(B, S, -1)), latent


# --------------------------------------------------------------------------------- MoE
class MLP(nn.Module):
    def __init__(self, D, I):
        super().__init__(); self.gate_proj = nn.Linear(D, I, bias=False); self.up_proj = nn.Linear(D, I, bias=False); self.down_proj = nn.Linear(I, D, bias=False)
    def forward(self, x): return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

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
        self.shared_experts = MLP(D, I * cfg.n_shared_experts)
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
        self.mlp = MoE(cfg) if cfg.mlp_layer_types[i] == "sparse" else MLP(cfg.hidden_size, cfg.intermediate_size)
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
    nothing is removed.
    """
    base = dict(hidden_size=256, num_hidden_layers=8, linear_num_heads=4, linear_head_dim=64,
                num_attention_heads=4, q_lora_rank=96, kv_lora_rank=64, qk_nope_head_dim=64,
                v_head_dim=64, n_routed_experts=8, num_experts_per_tok=2,
                moe_intermediate_size=128, intermediate_size=512)
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
