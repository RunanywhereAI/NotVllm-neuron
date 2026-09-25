# SPDX-License-Identifier: Apache-2.0
"""KDA (Kimi Delta Attention) — GLM-5.3-Flash's 34 linear-attention layers.

Shaped after ``qwen3_5/deltanet.py``, which is the plugin's established form for a
linear-attention layer: a plain-tensor return, state bound from vLLM's KV cache rather
than threaded through the signature, and a ``max_query_len`` dispatch between a
chunked prefill and a single-token recurrence.

**KDA is not GDN.** The gate is per **channel**, not a per-head scalar. That one fact
has cost four distinct errors during this port, and each is guarded here:

1. **The output gate is ``sigmoid``, not ``silu``/``swish``.** No config key selects
   it; transformers and vLLM both hardcode sigmoid, and FLA's *default* is the wrong
   branch, so a port that omits ``activation=`` gets it wrong silently. Worth 194%
   relative error across all 34 layers.
2. **``exp(cg)`` does not commute out of the matmul.** Per channel it must be
   ``(q * exp_cg) @ state``; ``(q @ state) * exp_cg`` is a per-token-scalar
   optimisation and gives 124% error here.
3. **Two different masks.** The intra-chunk output keeps ``j <= i`` *inclusive*; the
   ``A`` operator keeps ``j < i`` *strict*.
4. **The sub-block size is derived from ``gate_lower_bound``**, not a constant — see
   ``nki_kda_cte.max_safe_subblock``.

Validation: ``personal_reference/glm5_next/reference.py`` is the oracle (261 tests,
three defects removed), and ``tests/`` here diffs this layer against it. The torch
path below is deliberately written to be checkable that way rather than to be fast.

**Capture.** ``accuracy/tensor_capture.py`` hooks capture a *module's output* (and
only ``output[0]`` of a tuple), so intermediates that are not module outputs are
emitted explicitly with ``capture_tensor``. This is not instrumentation: KDA decode
cannot see errors below ~2% of its output — the bf16 floor is ~0.6% and a 5% input
perturbation moves the result only 3-6x that — so a layer exposing only its final
output gives on-device validation no instrument at all.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

try:
    from vllm_neuron.accuracy.tensor_capture import capture_tensor as _capture_tensor
except ImportError:  # pragma: no cover - exercised on hosts without vLLM
    # ``tensor_capture`` itself needs only the stdlib, but importing it through the
    # package executes ``vllm_neuron/__init__.py``, which needs vLLM. Capture is
    # optional instrumentation, and this layer must stay importable on a laptop so it
    # can be diffed against the oracle — which is how every defect here was found.
    # Tests substitute a recorder for this symbol to assert the capture points fire.
    def _capture_tensor(name, tensor):  # type: ignore[misc]
        return None


@dataclass(frozen=True)
class KDAParams:
    """The config surface this layer reads, in one place.

    dev1 owns the config object. Everything config-shaped is extracted here so that
    when its final form lands, only ``from_config`` changes and none of the numerics
    do. Values are the live ``zai-org/GLM-5.3-Flash`` ``linear_attn_config``.
    """

    num_heads: int = 64
    head_dim: int = 128
    conv_kernel: int = 4
    gate_lower_bound: float = -5.0
    hidden_size: int = 4096
    rms_norm_eps: float = 1e-5

    @classmethod
    def from_config(cls, config) -> "KDAParams":
        la = getattr(config, "linear_attn_config", None)
        get = (lambda k, d: la.get(k, d)) if isinstance(la, dict) else (
            lambda k, d: getattr(config, f"linear_{k}", getattr(config, k, d)))
        return cls(
            num_heads=get("num_heads", 64),
            head_dim=get("head_dim", 128),
            conv_kernel=get("short_conv_kernel_size", 4),
            gate_lower_bound=get("gate_lower_bound", -5.0),
            hidden_size=getattr(config, "hidden_size", 4096),
            rms_norm_eps=getattr(config, "rms_norm_eps", 1e-5),
        )


# --------------------------------------------------------------------------- math
def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """``x / sqrt(sum(x^2) + eps)`` — transformers/FLA's spelling.

    NOT ``x * rsqrt(...)``, which is Qwen's GDN variant; transformers comments that it
    "intentionally use[s] sqrt and / to match original triton". ~0.8 ULP apart, and
    the third Qwen-to-GLM carry-over found on this port.
    """
    return x / torch.sqrt((x * x).sum(dim, keepdim=True) + eps)


class RMSNormGated(nn.Module):
    """``rmsnorm(x) * weight * sigmoid(gate)`` — **sigmoid**, see module docstring."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()
        return (y * torch.sigmoid(gate.float())).to(x.dtype)


class ForgetGate(nn.Module):
    """``g = lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))`` -> [B,S,H,K].

    Per **channel**: the output has a K axis, unlike GDN's per-head scalar. Every
    consumer below broadcasts the decay over K rather than over heads.
    """

    def __init__(self, p: KDAParams):
        super().__init__()
        H, K, D = p.num_heads, p.head_dim, p.hidden_size
        self.H, self.K = H, K
        self.f_a_proj = nn.Linear(D, K, bias=False)
        self.f_b_proj = nn.Linear(K, H * K, bias=False)
        self.dt_bias = nn.Parameter(torch.zeros(H * K))
        self.A_log = nn.Parameter(torch.zeros(H))
        self.lower = p.gate_lower_bound

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = (self.f_b_proj(self.f_a_proj(x)).float() + self.dt_bias.float())
        g = g.view(*x.shape[:2], self.H, self.K)
        decay = torch.exp(self.A_log.float()).view(1, 1, self.H, 1)
        return self.lower * torch.sigmoid(decay * g)


def recurrent_step(q, k, v, g, beta, state):
    """One token. q,k,v: [B,1,H,K]; g: [B,1,H,K]; beta: [B,1,H]; state: [B,H,K,V] fp32.

    Differs from GDN only in ``state * exp(g)`` being a per-K-row scale.
    """
    q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))
    q, k = l2norm(q), l2norm(k)
    q = q * (q.shape[-1] ** -0.5)
    q_i, k_i, v_i = q[:, 0], k[:, 0], v[:, 0]
    state = state * g[:, 0].exp()[..., None]                 # [B,H,K,1] — PER CHANNEL
    kv = (state * k_i[..., None]).sum(-2)
    delta = (v_i - kv) * beta[:, 0][..., None]
    state = state + k_i[..., None] * delta[..., None, :]
    out = (state * q_i[..., None]).sum(-2)
    return out.unsqueeze(1), state


def chunk_prefill(q, k, v, g, beta, state=None, chunk: int = 64):
    """Chunked prefill. Transcribed from ``chunk_kimi_delta_attention``.

    Two things here are the port's recurring traps, marked at their sites: the
    inclusive-vs-strict masks, and ``exp(cg)`` applied *before* each matmul rather
    than after.
    """
    dt = q.dtype
    q, k, v, beta, g = (t.transpose(1, 2).contiguous().float() for t in (q, k, v, beta, g))
    q, k = l2norm(q), l2norm(k)
    B, H, T, K = k.shape
    V = v.shape[-1]
    pad = (chunk - T % chunk) % chunk
    Tp = T + pad
    q = F.pad(q, (0, 0, 0, pad)) * (K ** -0.5)
    k, v = F.pad(k, (0, 0, 0, pad)), F.pad(v, (0, 0, 0, pad))
    g, beta = F.pad(g, (0, 0, 0, pad)), F.pad(beta, (0, pad))
    v_beta, k_beta = v * beta[..., None], k * beta[..., None]
    rs = lambda t: t.reshape(B, H, -1, chunk, t.shape[-1])
    q, k, v, g, k_beta, v_beta = map(rs, (q, k, v, g, k_beta, v_beta))
    g = g.cumsum(-2)

    tri = torch.triu(torch.ones(chunk, chunk, dtype=torch.bool, device=q.device), 0)
    stri = torch.triu(torch.ones(chunk, chunk, dtype=torch.bool, device=q.device), 1)
    decay = (g.unsqueeze(-2) - g.unsqueeze(-3)).masked_fill(stri[..., None], float("-inf")).exp()
    # TRAP 3a: the A operator masks j >= i, keeping j < i STRICTLY.
    attn = -(k_beta.unsqueeze(-2) * k.unsqueeze(-3) * decay).sum(-1).masked_fill(tri, 0)
    for i in range(1, chunk):
        row, sub = attn[..., i, :i].clone(), attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk, device=q.device)
    v = attn @ v_beta
    # TRAP 2: exp(cg) is per channel, so it goes INSIDE the contraction.
    k_cumdecay = attn @ (k_beta * g.exp())

    S = torch.zeros(B, H, K, V, device=q.device) if state is None else state.float()
    out = torch.zeros_like(v)
    for i in range(Tp // chunk):
        q_i, k_i, v_i, g_i = q[:, :, i], k[:, :, i], v[:, :, i], g[:, :, i]
        inter = (q_i * g_i.exp()) @ S                        # TRAP 2 again
        # TRAP 3b: the intra output masks j > i, keeping j <= i INCLUSIVE.
        intra = (q_i.unsqueeze(-2) * k_i.unsqueeze(-3) * decay[:, :, i]).sum(-1).masked_fill(stri, 0)
        v_new = v_i - k_cumdecay[:, :, i] @ S
        out[:, :, i] = inter + intra @ v_new
        S = S * g_i[:, :, -1].exp().unsqueeze(-1) + (
            k_i * (g_i[:, :, -1:] - g_i).exp()).transpose(-1, -2) @ v_new
    out = out.reshape(B, H, -1, V)[:, :, :T].transpose(1, 2).contiguous().to(dt)
    return out, S


# -------------------------------------------------------------------------- module
class Glm5NextKDA(nn.Module):
    """One of the 34 linear-attention layers.

    ``forward`` returns a plain tensor. State is **not** threaded through the
    signature: it lives in vLLM's KV cache, bound by ``bind_kv_cache``, exactly as
    ``qwen3_5/deltanet.py`` does. Two state tensors per request:

    * ``conv_state``      ``[conv_kernel - 1, 3 * H * K]`` = ``[3, 24576]``, **bf16**
    * ``recurrent_state`` ``[H, K, V]`` = ``[64, 128, 128]`` **fp32**, sharded to
      ``H // tp_size`` heads per rank.

    Both shapes and dtypes come from ``MambaStateShapeCalculator.kda_state_shape`` /
    ``kda_state_dtype`` — **do not re-derive them here.** vLLM sizes the state pages
    from the same helper, so a divergent layout aliases memory rather than raising.
    The cache binding itself is dev3's surface; ``forward`` is left as the seam.
    """

    def __init__(self, config, layer_idx: int, tp_size: int = 1):
        super().__init__()
        p = KDAParams.from_config(config)
        self.p = p
        self.layer_idx = layer_idx
        self.layer_name = f"model.layers.{layer_idx}.linear_attn"  # dev3 owns this name
        if p.num_heads % tp_size:
            raise ValueError(f"linear num_heads={p.num_heads} must divide tp_size={tp_size}")
        self.H = p.num_heads // tp_size
        self.K = self.V = p.head_dim
        self.HK = self.H * self.K
        D = p.hidden_size

        self.q_proj = nn.Linear(D, self.HK, bias=False)
        self.k_proj = nn.Linear(D, self.HK, bias=False)
        self.v_proj = nn.Linear(D, self.HK, bias=False)
        # The checkpoint stores three depthwise conv1ds; they concatenate channel-wise.
        self.conv1d = nn.Conv1d(3 * self.HK, 3 * self.HK, p.conv_kernel,
                                groups=3 * self.HK, bias=False, padding=p.conv_kernel - 1)
        self.forget_gate = ForgetGate(p)
        self.b_proj = nn.Linear(D, self.H, bias=False)
        self.g_a_proj = nn.Linear(D, self.K, bias=False)
        self.g_b_proj = nn.Linear(self.K, self.HK, bias=False)
        self.o_norm = RMSNormGated(self.K, p.rms_norm_eps)
        self.o_proj = nn.Linear(self.HK, D, bias=False)

    # -- capture -----------------------------------------------------------------
    def _capture(self, name: str, t: torch.Tensor) -> None:
        """Emit an intermediate for on-device comparison.

        A no-op unless capture is active. These are not module outputs, so a forward
        hook cannot reach them — and without them the layer is unvalidatable on
        hardware below ~2% of its output. See the module docstring.
        """
        _capture_tensor(f"{self.layer_name}.{name}", t)

    # -- the shared body ---------------------------------------------------------
    def _project(self, hidden_states: torch.Tensor):
        """hidden -> (qkv [B, 3HK, S], g [B,S,H,K], beta [B,S,H], gate [B,S,H,K])."""
        qkv = torch.cat([self.q_proj(hidden_states), self.k_proj(hidden_states),
                         self.v_proj(hidden_states)], dim=-1).transpose(1, 2)
        g = self.forget_gate(hidden_states)
        beta = torch.sigmoid(self.b_proj(hidden_states))
        gate = self.g_b_proj(self.g_a_proj(hidden_states)).view(
            *hidden_states.shape[:2], self.H, self.K)
        self._capture("g", g)                       # per-channel gate: trap 1's witness
        return qkv, g, beta, gate

    def _finish(self, core: torch.Tensor, gate: torch.Tensor, B: int, S: int):
        """core [B,S,H,V] -> output [B,S,D], capturing before the gated norm.

        The norm's sigmoid gate attenuates, so an upstream error is smaller after it
        than before. ``core_pre_norm`` is where it is still visible.
        """
        self._capture("core_pre_norm", core)
        return self.o_proj(self.o_norm(core, gate).reshape(B, S, -1))

    def forward_prefill(self, hidden_states, conv_state=None, rec_state=None):
        """Chunked prefill -> (output, (conv_state, recurrent_state)).

        The state return is for testing and for callers that manage state directly;
        the framework path in ``forward`` writes it back to the KV cache instead.
        """
        B, S, _ = hidden_states.shape
        qkv, g, beta, gate = self._project(hidden_states)
        kernel = self.conv1d.weight.shape[-1]
        if conv_state is not None:
            qkv = torch.cat([conv_state, qkv], -1)
        pre = qkv
        qkv = F.silu(self.conv1d(qkv)[..., :qkv.shape[-1]])[..., -S:]
        conv_state = F.pad(pre, (max(0, kernel - pre.shape[-1]), 0))[..., -kernel:]
        self._capture("conv_window", conv_state)
        q, k, v = (t.reshape(B, S, self.H, self.K)
                   for t in qkv.transpose(1, 2).split([self.HK] * 3, -1))
        core, rec_state = chunk_prefill(q, k, v, g, beta, rec_state)
        self._capture("recurrent_state", rec_state)
        return self._finish(core, gate, B, S), (conv_state, rec_state)

    def forward_decode(self, hidden_states, conv_state, rec_state):
        """One recurrent step -> (output, (conv_state, recurrent_state))."""
        B, S, _ = hidden_states.shape
        if S != 1:
            raise ValueError(f"decode expects one token, got S={S}")
        qkv, g, beta, gate = self._project(hidden_states)
        # torch.roll allocates, so the advanced window must be returned, never assumed
        # to have been mutated in place.
        conv_state = torch.roll(conv_state, -1, -1)
        conv_state[..., -1] = qkv[..., 0]
        qkv = F.silu((conv_state * self.conv1d.weight.squeeze(1)).sum(-1)).unsqueeze(-1)
        self._capture("conv_window", conv_state)
        q, k, v = (t.reshape(B, S, self.H, self.K)
                   for t in qkv.transpose(1, 2).split([self.HK] * 3, -1))
        core, rec_state = recurrent_step(q, k, v, g, beta, rec_state)
        self._capture("recurrent_state", rec_state)
        return self._finish(core, gate, B, S), (conv_state, rec_state)

    def forward(self, hidden_states, positions, attn_metadata: dict) -> torch.Tensor:
        """Framework entry point. Returns a plain tensor; state goes to the KV cache.

        NOT YET WIRED: dev3 owns the cache plumbing, so the state read/write is left
        as the one seam rather than guessed at. Everything above this line is
        complete and tested against the oracle.
        """
        raise NotImplementedError(
            "state binding is dev3's surface; call forward_prefill/forward_decode "
            "directly until the cache plumbing lands"
        )
