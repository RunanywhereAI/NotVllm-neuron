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
    from vllm_neuron.model.kv_cache import state_page_indices
except ImportError:  # pragma: no cover - loaded by path on a host without vLLM
    # ``forward`` (the framework path) needs it; ``forward_prefill``/``forward_decode``
    # do not, and those are what the by-path oracle comparison exercises.
    state_page_indices = None

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

    num_heads: int
    head_dim: int
    conv_kernel: int
    gate_lower_bound: float
    hidden_size: int
    rms_norm_eps: float

    # checkpoint ``linear_attn_config`` key -> ``Glm5NextTextConfig`` field
    _FLAT = {
        "num_heads": "linear_num_heads",
        "head_dim": "linear_head_dim",
        "short_conv_kernel_size": "linear_conv_kernel_dim",
        "gate_lower_bound": "linear_lower_bound",
    }

    @classmethod
    def from_config(cls, config) -> "KDAParams":
        """Every field is REQUIRED.

        This used to fall back to the released values (kernel 4, bound -5.0) and read
        ``linear_short_conv_kernel_size`` / ``linear_gate_lower_bound`` -- names the
        front-end config does not have (it says ``linear_conv_kernel_dim`` /
        ``linear_lower_bound``). So against the real config object both silently took
        the default, which happens to be the right answer: a wired call site and an
        unwired one were indistinguishable, and any config with a different kernel
        width or gate bound would have been wrong without a sound. A missing field is
        now an ``AttributeError``/``KeyError``, not a guess.
        """
        la = getattr(config, "linear_attn_config", None)
        if isinstance(la, dict):
            get = la.__getitem__
        else:
            get = lambda k: getattr(config, cls._FLAT[k])  # noqa: E731
        return cls(
            num_heads=get("num_heads"),
            head_dim=get("head_dim"),
            conv_kernel=get("short_conv_kernel_size"),
            gate_lower_bound=get("gate_lower_bound"),
            hidden_size=config.hidden_size,
            rms_norm_eps=config.rms_norm_eps,
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

    def __init__(self, p: KDAParams, num_heads: int):
        """``num_heads`` is THIS RANK's head count, and required: ``p.num_heads`` is the
        global count, and under TP the gate must match the rank's q/k/v heads. It used to
        read ``p.num_heads`` -- right at TP=1, silently wrong (and a shape error at best)
        at any other degree."""
        super().__init__()
        H, K, D = num_heads, p.head_dim, p.hidden_size
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


def _last_real_columns(x: torch.Tensor, real: torch.Tensor, count: int) -> torch.Tensor:
    """The last ``count`` columns of ``x`` ``[B, C, L]`` at which ``real`` ``[B, L]`` is
    True, oldest first, zero-filled where the sequence is shorter than ``count``.

    Selected by arithmetic, not by an index computed from the data. Counting the real
    tokens and slicing at that count is a data-dependent index, and that form **silently
    miscompiles on Neuron** (``qwen3_5/deltanet.py::_tail_rows`` measured relative error
    1.5 on device, compiling without complaint). With ``r[t] = 1`` for real ``t``,
    ``r[t] - r[t+1]`` is one-hot at the last real column; shifting it left by ``j``
    moves it ``j`` columns back. Every index is a compile-time constant, and the
    contraction against a one-hot is exact. Requires the pads to follow the real
    tokens, which is how the plugin pads.
    """
    r = real.to(x.dtype)
    last = r - F.pad(r[:, 1:], (0, 1))                                  # [B, L]
    sel = torch.stack([F.pad(last[:, count - 1 - j:], (0, count - 1 - j))
                       for j in range(count)], 1)                       # [B, count, L]
    return torch.einsum("bcl,bjl->bcj", x, sel)


def strictly_lower_inverse(a: torch.Tensor, size: int) -> torch.Tensor:
    """``(I - A)^-1`` for strictly lower-triangular ``A``, in ``log2(size)`` steps.

    Taken from ``qwen3_5/deltanet.py::_strictly_lower_inverse`` (PR #54), which cannot
    be imported here: that module imports vLLM at module scope and this one must stay
    loadable on a laptop. Same sign convention -- ``A`` is the already-negated, masked
    operator -- so it replaces FLA's scalar forward substitution
    (``attn[..., i, :i] = ...`` for i in 1..size-1) exactly: that loop solves
    ``X = A + A X`` row by row, i.e. ``X + I = (I - A)^-1``.

    Why not the loop: each in-place slice assignment lowers to an XLA ``unselect``,
    which libtorch-neuronx-lite fails to lower (``Error while lowering: xla::unselect``,
    hit by GLM Tier 3 CPU compilation, 2026-10-08), and 63 dependent steps is a poor
    shape for the compiler anyway. Blocked elimination keeps every intermediate a true
    inverse of a sub-problem, so nothing grows (the Neumann product does not: deltanet
    measured 0.57 absolute error with it).
    """
    lead = (1,) * (a.dim() - 2)
    rows = torch.arange(size, device=a.device).reshape(size, 1)
    cols = torch.arange(size, device=a.device).reshape(1, size)
    inv = torch.eye(size, dtype=a.dtype, device=a.device).repeat(*a.shape[:-2], 1, 1)
    width = 1
    while width < size:
        quadrant = ((rows // (2 * width) == cols // (2 * width))
                    & (rows // width % 2 == 1) & (cols // width % 2 == 0))
        mask = quadrant.to(a.dtype).reshape(*lead, size, size)
        inv = inv + mask * (inv @ a @ inv)
        width *= 2
    return inv


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
    # (I - A)^-1 by blocked elimination, not FLA's in-place forward substitution:
    # the slice assignments do not lower on Neuron (see strictly_lower_inverse).
    attn = strictly_lower_inverse(attn, chunk)
    v = attn @ v_beta
    # TRAP 2: exp(cg) is per channel, so it goes INSIDE the contraction.
    k_cumdecay = attn @ (k_beta * g.exp())

    S = torch.zeros(B, H, K, V, device=q.device) if state is None else state.float()
    outs = []          # stacked after the loop: no in-place slice writes (see above)
    for i in range(Tp // chunk):
        q_i, k_i, v_i, g_i = q[:, :, i], k[:, :, i], v[:, :, i], g[:, :, i]
        inter = (q_i * g_i.exp()) @ S                        # TRAP 2 again
        # TRAP 3b: the intra output masks j > i, keeping j <= i INCLUSIVE.
        intra = (q_i.unsqueeze(-2) * k_i.unsqueeze(-3) * decay[:, :, i]).sum(-1).masked_fill(stri, 0)
        v_new = v_i - k_cumdecay[:, :, i] @ S
        outs.append(inter + intra @ v_new)
        S = S * g_i[:, :, -1].exp().unsqueeze(-1) + (
            k_i * (g_i[:, :, -1:] - g_i).exp()).transpose(-1, -2) @ v_new
    out = torch.stack(outs, 2).reshape(B, H, -1, V)[:, :, :T].transpose(1, 2).contiguous().to(dt)
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
    ``forward`` binds both to the cache through ``kv_cache.state_page_indices``.
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
        self.forget_gate = ForgetGate(p, self.H)
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

    def forward_prefill(self, hidden_states, conv_state=None, rec_state=None, real=None):
        """Chunked prefill -> (output, (conv_state, recurrent_state)).

        The state return is for testing and for callers that manage state directly;
        the framework path in ``forward`` writes it back to the KV cache instead.

        ``real`` (``[B, S]`` bool) marks real tokens when the sequence was padded to a
        compiled bucket. The plugin pads by *appending*, so every pad sits after the
        last real token: pads cannot affect a real token's output (both the conv and
        the recurrence are causal), but they would advance the state. So at a pad the
        projected q/k/v are zeroed before the conv, and ``g``/``beta`` are zeroed --
        decay ``exp(0) == 1``, delta update ``0`` -- which makes the recurrence a
        no-op there, and the conv window handed to decode is the last ``kernel``
        *real* columns rather than the last ``kernel`` columns.
        """
        B, S, _ = hidden_states.shape
        qkv, g, beta, gate = self._project(hidden_states)
        kernel = self.conv1d.weight.shape[-1]
        if real is not None:
            qkv = qkv * real[:, None, :].to(qkv.dtype)
            g = g * real[:, :, None, None].to(g.dtype)
            beta = beta * real[:, :, None].to(beta.dtype)
        if conv_state is not None:
            qkv = torch.cat([conv_state, qkv], -1)
            if real is not None:
                real = torch.cat([real.new_ones(B, conv_state.shape[-1]), real], -1)
        pre = qkv
        qkv = F.silu(self.conv1d(qkv)[..., :qkv.shape[-1]])[..., -S:]
        if real is None:
            conv_state = F.pad(pre, (max(0, kernel - pre.shape[-1]), 0))[..., -kernel:]
        else:
            conv_state = _last_real_columns(pre, real, kernel)
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
        # Shift in the new column out of place: a slice assignment lowers to an XLA
        # unselect that does not lower on Neuron (see strictly_lower_inverse).
        conv_state = torch.cat([conv_state[..., 1:], qkv.to(conv_state.dtype)], -1)
        qkv = F.silu((conv_state * self.conv1d.weight.squeeze(1)).sum(-1)).unsqueeze(-1)
        self._capture("conv_window", conv_state)
        q, k, v = (t.reshape(B, S, self.H, self.K)
                   for t in qkv.transpose(1, 2).split([self.HK] * 3, -1))
        core, rec_state = recurrent_step(q, k, v, g, beta, rec_state)
        # ``recurrent_step`` computes in fp32 and returns fp32; ``chunk_prefill`` casts
        # back to the input dtype. Do the same here, or a bf16 ``o_proj`` refuses it.
        core = core.to(qkv.dtype)
        self._capture("recurrent_state", rec_state)
        return self._finish(core, gate, B, S), (conv_state, rec_state)

    # -- framework surface (the pattern is deltanet.py's) --
    @property
    def conv_numel(self) -> int:
        return (self.p.conv_kernel - 1) * 3 * self.HK

    @property
    def rec_numel(self) -> int:
        return self.H * self.K * self.V

    def bind_state_pages(self, pages: torch.Tensor) -> None:
        """Bind the page-major state view, as ``deltanet.bind_state_pages`` does.

        One row per request, holding ``conv ‖ recurrent`` contiguously. The page is
        wider than the two states; the remainder is vLLM's padding.
        """
        if pages.dim() != 2 or pages.shape[1] < self.conv_numel + self.rec_numel:
            raise ValueError(
                f"{self.layer_name}: state page view {tuple(pages.shape)} cannot hold "
                f"{self.conv_numel} conv + {self.rec_numel} recurrent elements"
            )
        self.state_pages = pages

    def _read_states(self, indices: torch.Tensor):
        rows = self.state_pages.index_select(0, indices)
        n = rows.shape[0]
        conv = rows[:, : self.conv_numel].reshape(n, self.p.conv_kernel - 1, 3 * self.HK)
        rec = rows[:, self.conv_numel : self.conv_numel + self.rec_numel].reshape(
            n, self.H, self.K, self.V)
        return conv, rec

    def _write_states(self, indices: torch.Tensor, conv: torch.Tensor, rec: torch.Tensor) -> None:
        """Both states back as ONE full-width row per page.

        Not two narrow writes: a per-state column slice is non-contiguous, and an
        in-place write through a strided view of a bound device tensor is what Neuron
        rejects. Straight from ``deltanet._write_states``, including the zero padding.
        """
        n = indices.shape[0]
        parts = [conv.reshape(n, -1).float(), rec.reshape(n, -1).float()]
        pad = self.state_pages.shape[1] - self.conv_numel - self.rec_numel
        if pad:
            parts.append(torch.zeros(n, pad, dtype=self.state_pages.dtype,
                                     device=indices.device))
        self.state_pages.index_copy_(0, indices, torch.cat(parts, dim=1))

    def forward(self, hidden_states, positions, attn_metadata: dict) -> torch.Tensor:
        """Framework entry point: ``[tokens, hidden] -> [tokens, hidden]``.

        State lives in the bound page, one row per request holding ``conv ‖ recurrent``.
        The cached conv state is vLLM's ``[kernel - 1, conv_dim]`` (the ``kda_state_shape``
        that sized the page); ``forward_decode`` works on a ``[conv_dim, kernel]`` window,
        so the decode path prepends a placeholder column (rolled out before it is read)
        and stores the newest ``kernel - 1`` columns back.

        **Prefill starts from a zero state and never reads the page.** The page a fresh
        sequence is handed may hold another group's bytes -- block ids are global, so
        it can be a recycled attention page -- and the plugin does not support chunked
        prefill or prefix caching over recurrent state, so there is no prior state to
        read. Decode reads through ``state_page_indices``, which sends padded rows to
        the runner's zero page and their writes to its sink.

        Returns this rank's partial output: ``o_proj`` is row-parallel under TP and the
        caller reduces.
        """
        if not hasattr(self, "state_pages"):
            raise RuntimeError(
                f"{self.layer_name}: bind_state_pages() has not been called; the "
                f"runner binds state before the first forward"
            )
        metadata = attn_metadata[self.layer_name]
        num_pages = self.state_pages.shape[0]
        if metadata["max_query_len"] <= metadata["decode_token_threshold"]:
            n = metadata["block_table_tensor"].shape[0]
            if hidden_states.shape[0] != n:
                raise NotImplementedError(
                    f"KDA decode expects one token per request, got "
                    f"{hidden_states.shape[0]} tokens for {n} requests; multi-token "
                    f"decode (speculative decoding) needs one recurrent step per draft"
                )
            read_idx, write_idx = self.state_indices(metadata, n)
            conv_prev, rec = self._read_states(read_idx)            # [n, k-1, C], [n,H,K,V]
            window = torch.cat(
                [conv_prev.new_zeros(n, 3 * self.HK, 1), conv_prev.transpose(1, 2)], -1
            ).to(hidden_states.dtype)
            out, (window, rec) = self.forward_decode(hidden_states[:, None], window, rec)
            self._write_states(write_idx, window[..., 1:].transpose(1, 2), rec)
            return out[:, 0]

        tokens = hidden_states.shape[0]
        offsets = torch.arange(tokens, device=positions.device, dtype=positions.dtype)
        real = (positions - positions[0]) == offsets
        out, (window, rec) = self.forward_prefill(hidden_states[None], real=real[None])
        _, write_idx = self.state_indices(metadata, 1)
        self._write_states(write_idx, window[..., 1:].transpose(1, 2), rec)
        return out[0]

    def state_indices(self, metadata: dict, num_reqs: int):
        """``(read, write)`` page per request, from the shared implementation.

        ``kv_cache.state_page_indices`` was lifted out of ``qwen3_5/deltanet.py`` so
        both models use one copy: it carries the PAD_SLOT_ID -> NULL_BLOCK_ID sentinel
        change, an out-of-bound-DMA bound check, and the padded-row read/write split
        whose failure mode is NaN logits spreading across a whole batch tile.
        """
        return state_page_indices(metadata, num_reqs, self.state_pages.shape[0])
