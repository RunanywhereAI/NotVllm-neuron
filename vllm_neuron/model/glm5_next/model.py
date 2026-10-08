# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash text decoder for Neuron: mHC, MoE, the decoder stack and the
``ForCausalLM`` the runner drives.

45 layers, ``layer_types`` read index for index: 34 KDA linear-attention layers
(``kda.py``) and 11 NoPE sparse-MLA layers with the DSA indexer (``mla.py``). Every
layer wraps BOTH sublayers in manifold-constrained hyper-connections over
``hc_mult = 4`` residual streams, and every layer but the first
``first_k_dense_replace`` uses a 288-expert MoE.

Ported from ``personal_reference/glm5_next/reference.py`` -- the oracle, whose mHC, MoE
and decoder layer are cross-checked against transformers 5.17 and vLLM -- and diffed
against it by ``personal_reference/glm5_next/tests/test_plugin_model.py``. Things that
are easy to get wrong and invisible end to end, each guarded below:

* **Sinkhorn runs exactly ``hc_sinkhorn_iters`` (20) iterations.** It does not converge
  at 20 -- columns land at ~1e-6 but rows at ~4e-2, because it ends on a column
  division -- and it keeps converging past it, so more iterations give a *different*
  matrix, not a better one. 20 is truncation, reproduced.
* **mHC's mixing is invisible end to end.** The real ``hc_attn_base`` is a near-identity
  prior, so cross-stream mixing contributes ~1e-3 and any logit check at 1e-2 passes
  identically with mHC present or absent. ``comb``/``post``/``pre`` are capture points.
* **The SwiGLU clamp (``swiglu_limit`` 10.0) is on every MLP**: the three dense layers,
  every routed expert and the shared expert. The oracle once ran 45 of 45 layers
  unclamped. ``limit`` is a required argument, so an unwired call site is a TypeError.
* **Routing**: sigmoid scores; ``e_score_correction_bias`` (float32) only for *choosing*
  experts; the raw scores, normalised over the top-8, times ``routed_scaling_factor``
  2.5 as the weights. The shared expert sees the same layer input, is added after the
  routed sum, and is never multiplied by a routing weight.
* The final stream collapse is an **unweighted mean** (transformers'
  ``Glm5NextTextHyperHead``), unlike DeepSeek-V4.

Routed experts run through the plugin's NKI MoE ops where a kernel can run --
``NF.build_blockwise_mapping`` + ``NF.moe_cte`` for prefill, ``NF.moe_tkg`` for decode,
as ``gpt_oss/model_bf16.py`` -- and through a dense torch contraction over every local
expert everywhere else; the dense path is also the reference they are diffed against.
The router is always torch: no NKI router expresses noaux_tc (see ``Glm5NextRouter``).

Tensor parallelism (``_attach_weight_loaders`` is the one table of what is sharded):

* KDA: q/k/v/b/g_b/f_b projections, ``dt_bias`` and ``A_log`` by head (rows), the three
  depthwise convs by head and re-concatenated per rank, ``o_proj`` by input column;
  ``f_a``/``g_a`` (the low-rank A halves) and ``o_norm`` replicated.
* MLA: ``q_b_proj`` and ``kv_b_proj`` by head (each head's rows are contiguous),
  ``o_proj`` by column; ``q_a``/``kv_a`` (the shared latent) and the whole DSA indexer
  replicated, so every rank selects identically and holds the full latent cache.
* MLPs and experts: tensor-parallel inside every expert -- gate and up rows, down
  columns -- because 288 experts do not divide over 64 ranks (EP would need a hybrid
  layout; a perf item). The router is replicated; routing weights scale partial expert
  outputs, which is exact because the sum over ranks is linear.
* Embedding vocab-sharded and all-reduced, LM head vocab-sharded (``nn``'s
  ``VocabDimShardedEmbedding`` / ``ColumnParallelLinear``), everything else replicated.

Every row-parallel output is all-reduced once, in ``Glm5NextDecoderLayer._reduce``:
after the mixer and after the (routed + shared) MLP.
"""

from __future__ import annotations

import dataclasses
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_neuron.model.kv_cache import KVSpec, LatentLayerSpec, RecurrentLayerSpec
from vllm_neuron.nn.cpl import ColumnParallelLinear
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding

from .cache_layout import latent_page_bytes
from .kda import Glm5NextKDA
from .mla import Glm5NextSparseMLA

try:
    from vllm_neuron.accuracy.tensor_capture import capture_tensor as _capture_tensor
except ImportError:  # pragma: no cover - hosts without vLLM; see kda.py
    def _capture_tensor(name, tensor):  # type: ignore[misc]
        return None

HF_TEXT_PREFIX = "model.language_model"
LINEAR_ATTENTION = "linear_attention"
SPARSE_MLA = "deepseek_sparse_attention"
DENSE_MLP = "dense"
SPARSE_MLP = "sparse"


# ------------------------------------------------------------------------------ norms
class Glm5NextRMSNorm(nn.Module):
    """``weight * normalise(x)``, both in fp32, cast once at the end.

    ``weight``, not Qwen3.5's ``(1 + weight)``. The fp32 multiply is the oracle's form;
    transformers rounds to the input dtype before the multiply, which differs from this
    only in bf16 and by one rounding.
    """

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * normed).to(x.dtype)


def _unweighted_rmsnorm(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps).to(x.dtype)


# -------------------------------------------------------------------------------- mHC
class Glm5NextHyperConnection(nn.Module):
    """Manifold-constrained hyper-connection over ``hc_mult`` residual streams.

    ``streams [T, H, D] -> (post [T, H], comb [T, H, H], collapsed [T, D])``. Parameter
    names are the checkpoint's ``hc_{attn,ffn}_{fn,base,scale}`` suffixes.
    """

    def __init__(self, hc_mult: int, hidden_size: int, sinkhorn_iters: int,
                 hc_eps: float, rms_norm_eps: float, name: str):
        super().__init__()
        H = hc_mult
        self.H, self.iters, self.hc_eps, self.norm_eps = H, sinkhorn_iters, hc_eps, rms_norm_eps
        self.name = name
        mix = (2 + H) * H
        self.fn = nn.Parameter(torch.zeros(mix, H * hidden_size))
        self.base = nn.Parameter(torch.zeros(mix))
        self.scale = nn.Parameter(torch.ones(3))

    def forward(self, streams: torch.Tensor):
        H, eps = self.H, self.hc_eps
        flat = _unweighted_rmsnorm(streams.flatten(-2).float(), self.norm_eps)
        mixw = F.linear(flat, self.fn.float())
        # plain slices, not ``Tensor.split``: split(sizes, -1) miscompiles on Neuron
        # (measured by PR #54, relative error 1.2-1.4 on device)
        pre_w, post_w, comb_w = mixw[..., :H], mixw[..., H:2 * H], mixw[..., 2 * H:]
        base = self.base.float()
        pre_b, post_b, comb_b = base[:H], base[H:2 * H], base[2 * H:]
        scale = self.scale.float()
        pre = torch.sigmoid(pre_w * scale[0] + pre_b) + eps
        post = 2 * torch.sigmoid(post_w * scale[1] + post_b)
        comb = torch.softmax(comb_w.view(*comb_w.shape[:-1], H, H) * scale[2]
                             + comb_b.view(H, H), -1) + eps
        # Sinkhorn-Knopp, EXACTLY ``iters`` iterations: one column normalisation, then
        # (row, column) pairs. It has not converged at 20 and is not meant to be.
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
        for _ in range(self.iters - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + eps)
            comb = comb / (comb.sum(-2, keepdim=True) + eps)
        _capture_tensor(f"{self.name}.pre", pre)
        _capture_tensor(f"{self.name}.post", post)
        _capture_tensor(f"{self.name}.comb", comb)
        collapsed = (pre.unsqueeze(-1) * streams).sum(-2).to(streams.dtype)
        return post, comb, collapsed


def hc_expand(post, comb, sublayer_out, residual_streams):
    """``post ⊗ out + comb^T @ residual`` -> ``[T, H, D]``."""
    dt = residual_streams.dtype
    return (post.to(dt).unsqueeze(-1) * sublayer_out.unsqueeze(-2)
            + torch.matmul(comb.to(dt).transpose(-1, -2), residual_streams))


# -------------------------------------------------------------------------------- MLP
def _clamped_swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """gate clamped to ``(-inf, limit]``, up to ``[-limit, limit]``, THEN silu."""
    return F.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)


class Glm5NextMLP(nn.Module):
    """Clamped SwiGLU. ``limit`` is required: it used to default to the real
    ``swiglu_limit``, which made a call site that forgot it indistinguishable from one
    that wired it."""

    def __init__(self, hidden_size: int, intermediate_size: int, limit: float):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.limit = limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(_clamped_swiglu(self.gate_proj(x), self.up_proj(x), self.limit))


class Glm5NextRouter(nn.Module):
    """noaux_tc top-k over sigmoid scores. ``n_group == 1`` (enforced by the config),
    so group masking is the identity and is omitted, as in the oracle."""

    def __init__(self, n_experts: int, hidden_size: int, top_k: int,
                 routed_scaling_factor: float, norm_topk_prob: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n_experts, hidden_size))
        # float32 in the checkpoint, and it decides WHICH experts run: rounding it to
        # bf16 changes the selection, so the model keeps it in float32.
        self.e_score_correction_bias = nn.Parameter(torch.zeros(n_experts))
        self.top_k, self.scale, self.norm = top_k, routed_scaling_factor, norm_topk_prob

    def forward(self, x: torch.Tensor):
        """``[T, D]`` -> (routing matrix ``[T, E]`` fp32, zero off the top-k; indices
        ``[T, top_k]``).

        Torch, deliberately, on every path. The plugin's ``NF.router``, nkilib's
        ``router_topk`` and the fused ``moe_block_tkg`` all add their bias to the logits
        *before* the activation and select on the biased scores. noaux_tc needs the
        opposite: the correction bias chooses experts and the UNbiased sigmoid scores
        weight them. None of them can express that.
        """
        scores = F.linear(x.float(), self.weight.float()).sigmoid()
        idx = torch.topk(scores + self.e_score_correction_bias.float(), self.top_k, -1,
                         sorted=False).indices
        w = scores.gather(1, idx)
        if self.norm:
            w = w / (w.sum(-1, keepdim=True) + 1e-20)
        return torch.zeros_like(scores).scatter(1, idx, w * self.scale), idx


@dataclasses.dataclass(frozen=True)
class ExpertLayout:
    """Where this rank's routed experts come from: ``ep_degree`` disjoint expert groups,
    each expert's intermediate dim split over ``tp_degree`` ranks.

    ``ep_degree * tp_degree`` is the TP world. Partial outputs from every (expert group,
    intermediate shard) pair are summed by the decoder layer's single all-reduce, so the
    layout changes which rank computes what, never the result.

    Why expert parallelism at all: ``NF.moe_cte``'s kernel guard refuses
    ``I_TP < 128``, and tensor parallelism alone at TP=64 gives ``2048 / 64 = 32``. The
    kernel path therefore needs ``tp_degree <= 16``, i.e. ``ep_degree >= 4`` with
    ``288 % ep_degree == 0``. On device the coordinates come from the plugin's parallel
    state (``get_neuron_ep_rank`` / ``get_neuron_ep_tp_group``), because trn2's 8x8 mesh
    is non-contiguous and ``rank // tp_degree`` is not the expert group there.
    """

    ep_degree: int = 1
    ep_rank: int = 0
    tp_degree: int = 1
    tp_rank: int = 0
    ep_tp_group: object = None     # build_blockwise_mapping's moe_group

    def first_local_expert(self, n_experts: int) -> int:
        return self.ep_rank * (n_experts // self.ep_degree)


# Selects the routed-expert implementation; "auto" uses the NKI MoE ops where a kernel
# can run and the dense reference elsewhere. "nf" forces the NF ops (on CPU: their torch
# fallbacks), "dense" forces the reference.
_MOE_IMPL_ENV = "VLLM_NEURON_GLM5NEXT_MOE"


class Glm5NextMoE(nn.Module):
    """Routed experts plus one shared expert.

    Expert weights are stored in the NKI MoE kernels' layout:
    ``gate_up_proj [E_local, D, 2, I_TP]`` (``[..., 0, :]`` gate, ``[..., 1, :]`` up) and
    ``down_proj [E_local, I_TP, D]`` -- what ``NF.moe_cte`` and ``NF.moe_tkg`` take, so no
    per-step relayout of the expert weights.

    Three implementations of the routed sum, one contract (the router's ``[T, E]``
    affinities, already normalised and scaled by 2.5):

    * ``_routed_dense``: a contraction over every local expert with zero weight for the
      unselected ones. Exact, static-shaped, the reference the others are diffed against;
      unusable at 288 experts.
    * ``_routed_nf_prefill``: ``NF.build_blockwise_mapping`` + ``NF.moe_cte``.
    * ``_routed_nf_decode``: ``NF.moe_tkg`` (kernel only -- it has no CPU path).

    The shared expert sees the same input, is added after the routed sum, and is never
    multiplied by a routing weight. It is tensor-parallel over the whole TP world.
    """

    block_size = 256        # NF.moe_cte tokens per block, as gpt_oss

    def __init__(self, hidden_size: int, moe_intermediate_size: int, n_experts: int,
                 top_k: int, n_shared_experts: int, routed_scaling_factor: float,
                 norm_topk_prob: bool, limit: float, layout: ExpertLayout, world: int):
        super().__init__()
        if n_experts % layout.ep_degree or moe_intermediate_size % layout.tp_degree:
            raise ValueError(
                f"{n_experts} experts x {moe_intermediate_size} intermediate do not "
                f"divide over EP={layout.ep_degree} x TP={layout.tp_degree}")
        self.layout, self.limit, self.top_k = layout, limit, top_k
        self.E, self.E_local = n_experts, n_experts // layout.ep_degree
        self.e0 = layout.first_local_expert(n_experts)
        self.I_tp = moe_intermediate_size // layout.tp_degree
        self.gate = Glm5NextRouter(n_experts, hidden_size, top_k,
                                   routed_scaling_factor, norm_topk_prob)
        self.gate_up_proj = nn.Parameter(torch.zeros(self.E_local, hidden_size, 2, self.I_tp))
        self.down_proj = nn.Parameter(torch.zeros(self.E_local, self.I_tp, hidden_size))
        self.shared_experts = Glm5NextMLP(
            hidden_size, moe_intermediate_size * n_shared_experts // world, limit)

    # -- the routed sum ---------------------------------------------------------------
    def _routed_dense(self, x, local):
        gu = torch.einsum("td,edgi->tegi", x, self.gate_up_proj)          # [T, E_l, 2, I]
        h = _clamped_swiglu(gu[:, :, 0], gu[:, :, 1], self.limit)         # [T, E_l, I]
        y = torch.einsum("tei,eid->ted", h, self.down_proj)               # [T, E_l, D]
        # routing weight applied in fp32, to the ROUTED sum only
        return torch.einsum("te,ted->td", local, y.float()).to(x.dtype)

    def _nf_clamps(self):
        return dict(gate_clamp_upper_limit=self.limit, gate_clamp_lower_limit=None,
                    up_clamp_upper_limit=self.limit, up_clamp_lower_limit=-self.limit)

    def _routed_nf_prefill(self, x, local, real, rank):
        import nki.language as nl
        from nkilib.core.moe.moe_cte.moe_cte import (
            ActFnType, ExpertAffinityScaleMode, MoECTEImplementation)

        from vllm_neuron import functional as NF

        masked, pos_to_id, block_to_expert, conditions = NF.build_blockwise_mapping(
            expert_affinities=local, num_local_experts=self.E_local,
            num_experts_per_token=self.top_k, block_size=self.block_size,
            moe_group=self.layout.ep_tp_group, tp_degree=self.layout.tp_degree,
            padding_mask=real, rank=rank)
        return NF.moe_cte(
            implementation=MoECTEImplementation.shard_on_block, conditions=conditions,
            hidden_states=x, expert_affinities_masked=masked,
            gate_up_proj_weight=self.gate_up_proj, down_proj_weight=self.down_proj,
            activation_function=ActFnType.SiLU, block_size=self.block_size,
            token_position_to_id=pos_to_id.to(torch.int32),
            block_to_expert=block_to_expert.to(torch.int32),
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            skip_token=True, is_tensor_update_accumulating=True,
            compute_dtype=nl.bfloat16, **self._nf_clamps())

    def _routed_nf_decode(self, x, routing, idx):
        from nkilib.core.moe.moe_cte.moe_cte import ActFnType, ExpertAffinityScaleMode

        from vllm_neuron import functional as NF

        all_expert = self.layout.ep_degree > 1
        return NF.moe_tkg(
            hidden_input=x, expert_gate_up_weights=self.gate_up_proj,
            expert_down_weights=self.down_proj, expert_affinities=routing.to(x.dtype),
            expert_index=idx.to(torch.int32), is_all_expert=all_expert,
            rank_id=(torch.tensor([[self.layout.ep_rank]], dtype=torch.int32,
                                  device=x.device) if all_expert else None),
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            activation_fn=ActFnType.SiLU, **self._nf_clamps())

    def _impl(self, x, is_decode: bool) -> str:
        import os

        mode = os.environ.get(_MOE_IMPL_ENV, "auto")
        if mode not in ("auto", "nf", "dense"):
            raise ValueError(f"{_MOE_IMPL_ENV}={mode!r}: expected auto, nf or dense")
        if mode == "dense":
            return "dense"
        try:
            from vllm_neuron.utils.neuron_utils import can_run_kernel
            on_device = can_run_kernel(x)
        except ImportError:                       # the oracle laptop: no plugin runtime
            return "dense"
        if mode == "nf":
            # moe_tkg has no CPU path; off device decode goes through the blockwise path
            return "nf_decode" if (is_decode and on_device) else "nf_prefill"
        if not on_device:
            return "dense"
        # The kernels' shape limits, checked here because the plugin's own guards do
        # not: nkilib moe_tkg asserts H % 128 == 0 at trace time
        # (moe_tkg/mlp_parameters.py), and NF.moe_cte below I_TP 128 falls back to
        # its own dense torch loop over every expert -- the reference with extra steps.
        if self.gate_up_proj.shape[1] % 128:
            return "dense"
        if is_decode:
            return "nf_decode"
        return "nf_prefill" if self.I_tp >= 128 else "dense"

    def forward(self, x, is_decode: bool = False, real=None, rank=None):
        routing, idx = self.gate(x)                                       # [T, E], [T, k]
        local = routing.narrow(1, self.e0, self.E_local)
        impl = self._impl(x, is_decode)
        if impl == "dense":
            routed = self._routed_dense(x, local)
        elif impl == "nf_prefill":
            routed = self._routed_nf_prefill(x, local, real, rank)
        else:
            routed = self._routed_nf_decode(x, routing, idx)
        return routed + self.shared_experts(x)


# ----------------------------------------------------------------------- decoder layer
class Glm5NextDecoderLayer(nn.Module):
    """``attn_hc -> input_layernorm -> mixer -> expand``, then
    ``ffn_hc -> post_attention_layernorm -> MLP/MoE -> expand``."""

    def __init__(self, config, layer_idx: int, tp_size: int = 1, reduce=None,
                 expert_layout: ExpertLayout | None = None):
        super().__init__()
        self.layer_idx = layer_idx
        layer_type = config.layer_types[layer_idx]
        mlp_type = config.mlp_layer_types[layer_idx]
        if layer_type == LINEAR_ATTENTION:
            self.self_attn = Glm5NextKDA(config, layer_idx, tp_size=tp_size)
        elif layer_type == SPARSE_MLA:
            self.self_attn = Glm5NextSparseMLA(config, layer_idx, tp_size=tp_size)
        else:
            raise NotImplementedError(f"layer {layer_idx}: layer type {layer_type!r}")
        self.is_linear_attention = layer_type == LINEAR_ATTENTION
        D = config.hidden_size
        if mlp_type == SPARSE_MLP:
            self.mlp = Glm5NextMoE(
                D, config.moe_intermediate_size, config.n_routed_experts,
                config.num_experts_per_tok, config.n_shared_experts,
                config.routed_scaling_factor, config.norm_topk_prob, config.swiglu_limit,
                expert_layout or ExpertLayout(tp_degree=tp_size), world=tp_size)
        elif mlp_type == DENSE_MLP:
            self.mlp = Glm5NextMLP(D, config.intermediate_size // tp_size, config.swiglu_limit)
        else:
            raise NotImplementedError(f"layer {layer_idx}: mlp type {mlp_type!r}")
        self.input_layernorm = Glm5NextRMSNorm(D, config.rms_norm_eps)
        self.post_attention_layernorm = Glm5NextRMSNorm(D, config.rms_norm_eps)
        hc = functools.partial(Glm5NextHyperConnection, config.hc_mult, D,
                               config.hc_sinkhorn_iters, config.hc_eps, config.rms_norm_eps)
        self.attn_hc = hc(f"model.layers.{layer_idx}.attn_hc")
        self.ffn_hc = hc(f"model.layers.{layer_idx}.ffn_hc")
        self._reduce = reduce if reduce is not None else (lambda t: t)

    @property
    def mixer_name(self) -> str:
        return self.self_attn.layer_name

    def forward(self, streams, positions, attn_metadata, rank=None):
        residual = streams
        post, comb, h = self.attn_hc(streams)
        h = self._reduce(self.self_attn(self.input_layernorm(h), positions, attn_metadata))
        streams = hc_expand(post, comb, h, residual)
        residual = streams
        post, comb, h = self.ffn_hc(streams)
        h = self.post_attention_layernorm(h)
        if isinstance(self.mlp, Glm5NextMoE):
            md = attn_metadata[self.mixer_name]
            is_decode = md["max_query_len"] <= md["decode_token_threshold"]
            real = None
            if not is_decode:              # pads are appended, last position repeated
                offsets = torch.arange(positions.shape[0], device=positions.device,
                                       dtype=positions.dtype)
                real = (positions - positions[0]) == offsets
            h = self.mlp(h, is_decode=is_decode, real=real, rank=rank)
        else:
            h = self.mlp(h)
        return hc_expand(post, comb, self._reduce(h), residual)


class Glm5NextTextModel(nn.Module):
    """Embedding -> ``hc_mult`` copies -> 45 layers -> unweighted stream mean -> norm."""

    def __init__(self, config, tp_size: int = 1, reduce=None, tp_device_group=None,
                 expert_layout: ExpertLayout | None = None):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabDimShardedEmbedding(config.vocab_size, config.hidden_size,
                                                     tp_group=tp_device_group)
        if self.embed_tokens.tp_size != tp_size:
            raise ValueError(f"embedding sees TP={self.embed_tokens.tp_size}, model TP={tp_size}")
        self.layers = nn.ModuleList(
            Glm5NextDecoderLayer(config, i, tp_size=tp_size, reduce=reduce,
                                 expert_layout=expert_layout)
            for i in range(config.num_hidden_layers)
        )
        self.norm = Glm5NextRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids, positions, attn_metadata, rank=None):
        h = self.embed_tokens(input_ids, scatter_tokens=False, rank=rank)
        streams = h.unsqueeze(-2).expand(*h.shape[:-1], self.config.hc_mult, h.shape[-1])
        streams = streams.contiguous()
        for layer in self.layers:
            streams = layer(streams, positions, attn_metadata, rank=rank)
        return self.norm(streams.mean(-2))


def state_dict_from_reference(sd: dict) -> dict:
    """The oracle's / transformers' state dict (stacked experts ``[E, 2I, D]`` gate rows
    first, ``[E, D, I]``) -> this model's at TP=1, EP=1 (kernel layout
    ``[E, D, 2, I]`` / ``[E, I, D]``, ``model.`` prefix). For tests that load reference
    weights directly; checkpoints go through ``load_weights``."""
    out = {}
    for k, v in sd.items():
        if k.endswith("mlp.gate_up_proj") and v.dim() == 3:
            E, two_i, D = v.shape
            v = v.reshape(E, 2, two_i // 2, D).permute(0, 3, 1, 2).contiguous()
        elif k.endswith("mlp.down_proj") and v.dim() == 3:
            v = v.permute(0, 2, 1).contiguous()
        out[k if k == "lm_head.weight" else f"model.{k}"] = v
    return out


# ------------------------------------------------------------------------ ForCausalLM
# Parameters that stay float32 whatever the model dtype. The router bias is F32 in the
# checkpoint and chooses experts; dt_bias / A_log feed exp/sigmoid of the KDA decay.
_KEEP_FP32 = ("e_score_correction_bias", "dt_bias", "A_log")


class Glm5NextForCausalLM(nn.Module):
    """What the runner drives. ``from_configs`` is reached through the registered
    ``Glm5NextForConditionalGeneration`` (``factory.py``)."""

    # KDA and MLA layers share KV-cache buffers across groups, so every layer's data for
    # block ``b`` must stay inside page ``b``: the runner hands each layer one
    # page-major view and reserves a zero page and a write sink. See
    # ``initialize_kv_cache``.
    kv_cache_page_major = True

    def __init__(self, config, tp_group=None, expert_layout: ExpertLayout | None = None):
        """``tp_group``: vLLM's ``GroupCoordinator`` for the TP group (``world_size``,
        ``rank_in_group``, ``all_reduce`` returning its result, ``all_gather``,
        ``device_group``), or anything with that surface; ``None`` means TP=1.
        ``expert_layout``: the routed experts' EP x TP placement; default pure TP."""
        super().__init__()
        text = config.text_config
        self.config, self.text_config = config, text
        self.tp_group = tp_group
        self.world_size = 1 if tp_group is None else tp_group.world_size
        self.rank = 0 if tp_group is None else tp_group.rank_in_group
        device_group = None if tp_group is None else tp_group.device_group
        for name, n in (("linear num_heads", text.linear_num_heads),
                        ("num_attention_heads", text.num_attention_heads),
                        ("intermediate_size", text.intermediate_size),
                        ("moe_intermediate_size", text.moe_intermediate_size),
                        ("vocab_size", text.vocab_size)):
            if n % self.world_size:
                raise ValueError(f"{name}={n} does not divide over TP={self.world_size}")

        nc = getattr(config, "neuron_config", None)
        self.on_device_sampling_config = (
            getattr(nc, "on_device_sampling_config", None) if nc is not None else None
        )
        self.expert_layout = expert_layout or ExpertLayout(
            tp_degree=self.world_size, tp_rank=self.rank, ep_tp_group=tp_group)
        L = self.expert_layout
        if L.ep_degree * L.tp_degree != self.world_size:
            raise ValueError(f"EP={L.ep_degree} x TP={L.tp_degree} != TP world {self.world_size}")
        self.model = Glm5NextTextModel(text, tp_size=self.world_size, reduce=self._all_reduce,
                                       tp_device_group=device_group,
                                       expert_layout=self.expert_layout)
        self.lm_head = ColumnParallelLinear(
            text.hidden_size, text.vocab_size, bias=False,
            gather_output=self.on_device_sampling_config is None, tp_group=device_group)
        if self.lm_head.tp_size != self.world_size:
            raise ValueError(f"lm_head sees TP={self.lm_head.tp_size}, model TP={self.world_size}")
        self._attach_weight_loaders()

        self._gather_logits = nc is not None and (
            getattr(nc, "max_logprobs", 0) != 0 or getattr(nc, "debug_logits_dir", None) is not None
        )
        if self.on_device_sampling_config is not None:
            from vllm_neuron.nn.sampler import Sampler

            self.sampler = Sampler(self.on_device_sampling_config, process_group=device_group)

    def _all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return t
        return self.tp_group.all_reduce(t)

    def set_dtype(self, dtype: torch.dtype) -> "Glm5NextForCausalLM":
        """Cast to the serving dtype, keeping ``_KEEP_FP32`` parameters in float32."""
        self.to(dtype)
        for name, param in self.named_parameters():
            if name.rsplit(".", 1)[-1] in _KEEP_FP32:
                param.data = param.data.float()
        return self

    # -- the KV / state cache --------------------------------------------------------
    def _mixers(self):
        return [layer.self_attn for layer in self.model.layers]

    def get_kv_spec(self) -> KVSpec:
        """Recurrent state for the KDA layers, a folded latent page for the MLA layers.

        Shapes come from vLLM's ``kda_state_shape`` (via the config) and page bytes from
        ``cache_layout``; neither is derived here.
        """
        text = self.text_config
        recurrent, latent = [], []
        for layer in self.model.layers:
            mixer = layer.self_attn
            if layer.is_linear_attention:
                recurrent.append(RecurrentLayerSpec(
                    name=mixer.layer_name,
                    shapes=text.state_shapes(self.world_size),
                    dtypes=text.state_dtypes(),
                ))
            else:
                latent.append(LatentLayerSpec(
                    name=mixer.layer_name,
                    kv_lora_rank=text.kv_lora_rank,
                    dtype=text.torch_dtype,
                    page_bytes_for=functools.partial(latent_page_bytes, text, text.torch_dtype),
                ))
        return KVSpec(layers=[], recurrent_layers=recurrent, latent_layers=latent)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]) -> None:
        for layer in self.model.layers:
            mixer = layer.self_attn
            if mixer.layer_name not in kv_caches:
                raise KeyError(f"cache for layer {mixer.layer_name} not initialized")
            tensors = kv_caches[mixer.layer_name]
            if len(tensors) != 1:
                raise ValueError(
                    f"layer {mixer.layer_name} expected one page-major tensor, got "
                    f"{len(tensors)}"
                )
            if layer.is_linear_attention:
                mixer.bind_state_pages(tensors[0])
            else:
                mixer.bind_latent_pages(tensors[0])

    # -- vLLM's text-generation surface ---------------------------------------------
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        rotary_position_ids: torch.Tensor | None = None,
        attn_metadata: dict | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        # Present because the checkpoint has a vision_config; text-only, never used.
        vision_embedding_blocks: tuple[torch.Tensor, ...] | None = None,
        vision_positions: torch.Tensor | None = None,
    ):
        if spec_decode_metadata is not None:
            raise NotImplementedError("speculative decoding (MTP) is out of scope")
        positions = positions.to(torch.int32)
        hidden = self.model(input_ids, positions, attn_metadata, rank=rank)
        if sampling_positions is not None:
            hidden = torch.index_select(hidden, 0, sampling_positions)
        logits = self.compute_logits(hidden)
        if self.on_device_sampling_config is None:
            return logits
        gathered = None
        if self._gather_logits:
            gathered = (self.tp_group.all_gather(logits, dim=1) if self.world_size > 1
                        else logits)
        # No NaN guard here, deliberately (Qwen3.5 explains why it does not compile);
        # dead rows are kept finite at their source instead -- the zero page.
        sampled = self.sampler(logits, sampling_params, logit_mask=logit_mask, tp_rank=rank)
        return sampled, gathered

    # -- construction and weights ----------------------------------------------------
    @classmethod
    def from_configs(cls, hf_config, text_neuron_config=None, **_):
        from vllm.distributed.parallel_state import get_tp_group

        from .config import Glm5NextConfig

        config = Glm5NextConfig.from_configs(hf_config, text_neuron_config=text_neuron_config)
        # Serve at vLLM's resolved dtype, not the checkpoint's. ``--dtype`` may differ
        # from the checkpoint, and the KV cache is allocated at the serving dtype; the
        # runner refuses a latent page laid out at any other (found by Tier 3, where a
        # float32 tiny checkpoint was served in bf16).
        from vllm.config import get_current_vllm_config

        current = get_current_vllm_config()
        if current is not None and current.model_config is not None:
            config.text_config.torch_dtype = current.model_config.dtype
        tp = get_tp_group()
        layout = None
        if getattr(text_neuron_config, "ep_degree", 1) > 1:
            # [unverified on device] the plugin's EP groups, as gpt_oss reads them
            from vllm_neuron.parallel.neuron_parallel_state import (
                get_neuron_ep_degree, get_neuron_ep_rank, get_neuron_ep_tp_group)

            ep_tp = get_neuron_ep_tp_group()
            layout = ExpertLayout(ep_degree=get_neuron_ep_degree(),
                                  ep_rank=get_neuron_ep_rank(),
                                  tp_degree=ep_tp.world_size, tp_rank=ep_tp.rank_in_group,
                                  ep_tp_group=ep_tp)
        model = cls(config, tp_group=tp, expert_layout=layout)
        return model.set_dtype(config.text_config.torch_dtype)

    def checkpoint_mappings(self) -> dict[str, object]:
        """Parameter name -> checkpoint key(s). Every parameter appears exactly once.

        Three regroupings between checkpoint and model, the same three as
        ``personal_reference/glm5_next/weight_converter.py`` (which is verified against
        all 76,108 real tensor names): the q/k/v depthwise convs concatenate channel-wise;
        the KDA forget gate's four tensors are flat in the checkpoint and nested here;
        288 per-expert tensors stack into ``gate_up_proj`` / ``down_proj``.
        """
        text = self.text_config
        P = HF_TEXT_PREFIX
        m: dict[str, object] = {
            "model.embed_tokens.weight": f"{P}.embed_tokens.weight",
            "model.norm.weight": f"{P}.norm.weight",
            "lm_head.weight": (f"{P}.embed_tokens.weight" if text.tie_word_embeddings
                               else "lm_head.weight"),
        }
        for i, layer in enumerate(self.model.layers):
            hf, ours = f"{P}.layers.{i}", f"model.layers.{i}"
            for norm in ("input_layernorm", "post_attention_layernorm"):
                m[f"{ours}.{norm}.weight"] = f"{hf}.{norm}.weight"
            for site in ("attn", "ffn"):
                for part in ("fn", "base", "scale"):
                    m[f"{ours}.{site}_hc.{part}"] = f"{hf}.hc_{site}_{part}"
            a_hf, a = f"{hf}.self_attn", f"{ours}.self_attn"
            if layer.is_linear_attention:
                for name in ("q_proj", "k_proj", "v_proj", "b_proj", "g_a_proj",
                             "g_b_proj", "o_proj", "o_norm"):
                    m[f"{a}.{name}.weight"] = f"{a_hf}.{name}.weight"
                m[f"{a}.conv1d.weight"] = [f"{a_hf}.{c}_conv1d.weight" for c in "qkv"]
                for name in ("f_a_proj", "f_b_proj"):
                    m[f"{a}.forget_gate.{name}.weight"] = f"{a_hf}.{name}.weight"
                for name in ("dt_bias", "A_log"):
                    m[f"{a}.forget_gate.{name}"] = f"{a_hf}.{name}"
            else:
                for name in ("q_a_proj", "q_a_layernorm", "q_b_proj", "kv_a_proj_with_mqa",
                             "kv_a_layernorm", "kv_b_proj", "o_proj"):
                    m[f"{a}.{name}.weight"] = f"{a_hf}.{name}.weight"
                ix, ix_hf = f"{a}.indexer", f"{a_hf}.indexer"
                for name in ("wq_b", "wk", "weights_proj"):
                    m[f"{ix}.{name}.weight"] = f"{ix_hf}.{name}.weight"
                for name in ("weight", "bias"):
                    m[f"{ix}.k_norm.{name}"] = f"{ix_hf}.k_norm.{name}"
                for name in ("index_kpool_compress_ape", "index_kpool_compress_gate"):
                    m[f"{ix}.{name}"] = f"{ix_hf}.{name}"
            mlp_hf, mlp = f"{hf}.mlp", f"{ours}.mlp"
            if isinstance(layer.mlp, Glm5NextMoE):
                m[f"{mlp}.gate.weight"] = f"{mlp_hf}.gate.weight"
                m[f"{mlp}.gate.e_score_correction_bias"] = f"{mlp_hf}.gate.e_score_correction_bias"
                for name in ("gate_proj", "up_proj", "down_proj"):
                    m[f"{mlp}.shared_experts.{name}.weight"] = f"{mlp_hf}.shared_experts.{name}.weight"
                E = text.n_routed_experts
                m[f"{mlp}.gate_up_proj"] = [
                    f"{mlp_hf}.experts.{e}.{w}.weight" for e in range(E)
                    for w in ("gate_proj", "up_proj")]
                m[f"{mlp}.down_proj"] = [f"{mlp_hf}.experts.{e}.down_proj.weight"
                                         for e in range(E)]
            else:
                for name in ("gate_proj", "up_proj", "down_proj"):
                    m[f"{mlp}.{name}.weight"] = f"{mlp_hf}.{name}.weight"
        return m

    def _attach_weight_loaders(self) -> None:
        """Which parameters are sharded, and how -- set on the modules themselves, not by
        matching names, so a renamed parameter cannot fall through to "replicated".

        Every loader runs at every TP degree (at TP=1 a shard is the whole tensor), so
        the TP=1 oracle comparison exercises the same code that shards at TP=64. A
        parameter left out of this table loads replicated; if its shape is per-rank
        that is a shape error at ``load_state_dict``, not a silent mistake.
        """
        from vllm_neuron.utils.weight_loader import (
            SafetensorsWeightLoader,
            get_shard,
            set_weight_loader,
        )

        n = self.world_size

        def shard(t, dim, rank):
            size = t.get_shape()[dim]
            if size % n:
                raise ValueError(f"dim {dim} of size {size} does not divide over TP={n}")
            return get_shard(t, dim, size // n, n, rank)

        def by(dim):
            return SafetensorsWeightLoader(transform=lambda s, r: shard(s[0], dim, r))

        rows, cols = by(0), by(1)
        # q, k, v depthwise convs: shard each by head, then concatenate per rank
        conv = SafetensorsWeightLoader(
            transform=lambda s, r: torch.cat([shard(x, 0, r) for x in s], 0))
        gate_up, down = self.expert_loaders(self.expert_layout)

        for layer in self.model.layers:
            a = layer.self_attn
            if layer.is_linear_attention:
                for lin in (a.q_proj, a.k_proj, a.v_proj, a.b_proj, a.g_b_proj,
                            a.forget_gate.f_b_proj):
                    set_weight_loader(lin.weight, rows)
                set_weight_loader(a.forget_gate.dt_bias, rows)
                set_weight_loader(a.forget_gate.A_log, rows)
                set_weight_loader(a.conv1d.weight, conv)
                set_weight_loader(a.o_proj.weight, cols)
            else:
                set_weight_loader(a.q_b_proj.weight, rows)
                set_weight_loader(a.kv_b_proj.weight, rows)
                set_weight_loader(a.o_proj.weight, cols)
            mlp = layer.mlp
            if isinstance(mlp, Glm5NextMoE):
                set_weight_loader(mlp.gate_up_proj, gate_up)
                set_weight_loader(mlp.down_proj, down)
                mlp = mlp.shared_experts
            set_weight_loader(mlp.gate_proj.weight, rows)
            set_weight_loader(mlp.up_proj.weight, rows)
            set_weight_loader(mlp.down_proj.weight, cols)

    def expert_loaders(self, layout: ExpertLayout):
        """``(gate_up, down)`` loaders for the routed experts at ``layout``: that expert
        group (EP) and intermediate shard (TP inside the group), into the kernels'
        ``[E_l, D, 2, I_tp]`` / ``[E_l, I_tp, D]`` layout.

        The coordinates are the layout's, not the loader's ``rank`` argument: under EP
        the TP rank is not the intermediate-shard index (trn2's mesh is non-contiguous).
        A function of the layout so a test can build the loaders for a WRONG layout.
        """
        from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader, get_shard

        E_l = self.text_config.n_routed_experts // layout.ep_degree
        e0 = layout.ep_rank * E_l

        def ishard(t, dim):
            size = t.get_shape()[dim] // layout.tp_degree
            return get_shard(t, dim, size, layout.tp_degree, layout.tp_rank)

        # [gate_0, up_0, gate_1, up_1, ...] -> per local expert stack([gate^T, up^T], 1)
        gate_up = SafetensorsWeightLoader(transform=lambda s, r: torch.stack([
            torch.stack([ishard(s[2 * e], 0).T, ishard(s[2 * e + 1], 0).T], 1)
            for e in range(e0, e0 + E_l)]))
        down = SafetensorsWeightLoader(transform=lambda s, r: torch.stack([
            ishard(s[e], 1).T for e in range(e0, e0 + E_l)]))
        return gate_up, down

    def load_weights(self, checkpoint_path: str, device: torch.device,
                     cache_dir: str | None = None) -> None:
        from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

        mappings = self.checkpoint_mappings()
        loaded = SafetensorsCheckpoint(checkpoint_path, cache_dir).load_sharded_pipelined(
            self.rank, self.world_size, self, mappings, device, strict=False,
        ).state_dict
        # strict=False is forced on us (the loader does not know about buffers), which
        # means a parameter with a wrong mapping keeps its uninitialised value and the
        # model produces fluent garbage. Check explicitly, as Qwen3.5 does.
        expected = {name for name, _ in self.named_parameters()}
        unfilled = sorted(expected - set(loaded))
        if unfilled:
            raise RuntimeError(
                f"{len(unfilled)} parameter(s) got no checkpoint tensor: {unfilled[:8]}"
                + (" ..." if len(unfilled) > 8 else "")
            )
        self.load_state_dict(loaded, strict=True, assign=True)


__all__ = ["Glm5NextForCausalLM", "Glm5NextTextModel", "Glm5NextDecoderLayer",
           "Glm5NextHyperConnection", "Glm5NextMoE", "Glm5NextMLP"]
