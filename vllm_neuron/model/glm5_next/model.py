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

The MoE here is the **torch correctness baseline**: a dense contraction over every
expert with a zero routing weight for the unselected ones. Exact, static-shaped and
compilable at a tiny config; hopeless at 288 experts. The device path is the NKI MoE
kernels (``NF.moe_cte`` / ``NF.moe_block_tkg``, as ``gpt_oss/model_bf16.py``), not wired.

Tensor parallelism: the layers size themselves per rank and every row-parallel output
passes through ``Glm5NextDecoderLayer._reduce``, but the weight loaders below are
written for TP=1 only and ``Glm5NextForCausalLM`` refuses anything else. Sharding is
unvalidated, and a wrong shard is silent.
"""

from __future__ import annotations

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_neuron.model.kv_cache import KVSpec, LatentLayerSpec, RecurrentLayerSpec

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
        """``[T, D]`` -> routing matrix ``[T, E]`` fp32, zero off the top-k."""
        scores = F.linear(x.float(), self.weight.float()).sigmoid()
        idx = torch.topk(scores + self.e_score_correction_bias.float(), self.top_k, -1,
                         sorted=False).indices
        w = scores.gather(1, idx)
        if self.norm:
            w = w / (w.sum(-1, keepdim=True) + 1e-20)
        return torch.zeros_like(scores).scatter(1, idx, w * self.scale)


class Glm5NextMoE(nn.Module):
    """Routed experts plus one shared expert.

    ``gate_up_proj [E, 2I, D]`` (gate rows first) and ``down_proj [E, D, I]``, the
    transformers / oracle stacking of the checkpoint's per-expert tensors.
    """

    def __init__(self, hidden_size: int, moe_intermediate_size: int, n_experts: int,
                 top_k: int, n_shared_experts: int, routed_scaling_factor: float,
                 norm_topk_prob: bool, limit: float):
        super().__init__()
        I = moe_intermediate_size
        self.I, self.limit = I, limit
        self.gate = Glm5NextRouter(n_experts, hidden_size, top_k,
                                   routed_scaling_factor, norm_topk_prob)
        self.gate_up_proj = nn.Parameter(torch.zeros(n_experts, 2 * I, hidden_size))
        self.down_proj = nn.Parameter(torch.zeros(n_experts, hidden_size, I))
        self.shared_experts = Glm5NextMLP(hidden_size, I * n_shared_experts, limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        routing = self.gate(x)                                            # [T, E] fp32
        gu = torch.einsum("td,efd->tef", x, self.gate_up_proj)            # [T, E, 2I]
        h = _clamped_swiglu(gu[..., : self.I], gu[..., self.I:], self.limit)
        y = torch.einsum("tei,edi->ted", h, self.down_proj)               # [T, E, D]
        # routing weight applied in fp32, to the ROUTED sum only
        routed = torch.einsum("te,ted->td", routing, y.float()).to(x.dtype)
        return routed + self.shared_experts(x)


# ----------------------------------------------------------------------- decoder layer
class Glm5NextDecoderLayer(nn.Module):
    """``attn_hc -> input_layernorm -> mixer -> expand``, then
    ``ffn_hc -> post_attention_layernorm -> MLP/MoE -> expand``."""

    def __init__(self, config, layer_idx: int, tp_size: int = 1, reduce=None):
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
                D, config.moe_intermediate_size // tp_size, config.n_routed_experts,
                config.num_experts_per_tok, config.n_shared_experts,
                config.routed_scaling_factor, config.norm_topk_prob, config.swiglu_limit)
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

    def forward(self, streams, positions, attn_metadata):
        residual = streams
        post, comb, h = self.attn_hc(streams)
        h = self._reduce(self.self_attn(self.input_layernorm(h), positions, attn_metadata))
        streams = hc_expand(post, comb, h, residual)
        residual = streams
        post, comb, h = self.ffn_hc(streams)
        h = self._reduce(self.mlp(self.post_attention_layernorm(h)))
        return hc_expand(post, comb, h, residual)


class Glm5NextTextModel(nn.Module):
    """Embedding -> ``hc_mult`` copies -> 45 layers -> unweighted stream mean -> norm."""

    def __init__(self, config, tp_size: int = 1, reduce=None):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            Glm5NextDecoderLayer(config, i, tp_size=tp_size, reduce=reduce)
            for i in range(config.num_hidden_layers)
        )
        self.norm = Glm5NextRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids, positions, attn_metadata):
        h = self.embed_tokens(input_ids)
        streams = h.unsqueeze(-2).expand(*h.shape[:-1], self.config.hc_mult, h.shape[-1])
        streams = streams.contiguous()
        for layer in self.layers:
            streams = layer(streams, positions, attn_metadata)
        return self.norm(streams.mean(-2))


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

    def __init__(self, config, tp_group=None):
        super().__init__()
        text = config.text_config
        self.config, self.text_config = config, text
        self.tp_group = tp_group
        self.world_size = 1 if tp_group is None else tp_group.world_size
        if self.world_size != 1:
            raise NotImplementedError(
                f"GLM-5.3-Flash at TP={self.world_size}: the layers size themselves per "
                f"rank, but the weight loaders here shard nothing and no TP>1 run has "
                f"been validated. A wrong shard loads without complaint."
            )
        self.model = Glm5NextTextModel(text, tp_size=self.world_size, reduce=self._all_reduce)
        self.lm_head = nn.Linear(text.hidden_size, text.vocab_size, bias=False)

        nc = getattr(config, "neuron_config", None)
        self.on_device_sampling_config = (
            getattr(nc, "on_device_sampling_config", None) if nc is not None else None
        )
        self._gather_logits = nc is not None and (
            getattr(nc, "max_logprobs", 0) != 0 or getattr(nc, "debug_logits_dir", None) is not None
        )
        if self.on_device_sampling_config is not None:
            from vllm_neuron.nn.sampler import Sampler

            self.sampler = Sampler(
                self.on_device_sampling_config,
                process_group=None if tp_group is None else tp_group.device_group,
            )

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
        hidden = self.model(input_ids, positions, attn_metadata)
        if sampling_positions is not None:
            hidden = torch.index_select(hidden, 0, sampling_positions)
        logits = self.compute_logits(hidden)
        if self.on_device_sampling_config is None:
            return logits
        gathered = logits if self._gather_logits else None
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
        tp = get_tp_group()
        model = cls(config, tp_group=tp if tp.world_size > 1 else None)
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
        """The three regroupings, as loaders on the parameters that need them."""
        from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader, set_weight_loader

        cat0 = SafetensorsWeightLoader(transform=lambda s, _r: torch.cat([x[:] for x in s], 0))

        def _gate_up(slices, _rank):
            pairs = [torch.cat([slices[2 * e][:], slices[2 * e + 1][:]], 0)
                     for e in range(len(slices) // 2)]
            return torch.stack(pairs)

        stack = SafetensorsWeightLoader(transform=lambda s, _r: torch.stack([x[:] for x in s]))
        for name, param in self.named_parameters():
            if name.endswith("self_attn.conv1d.weight"):
                set_weight_loader(param, cat0)
            elif name.endswith("mlp.gate_up_proj"):
                set_weight_loader(param, SafetensorsWeightLoader(transform=_gate_up))
            elif name.endswith("mlp.down_proj"):
                set_weight_loader(param, stack)

    def load_weights(self, checkpoint_path: str, device: torch.device,
                     cache_dir: str | None = None) -> None:
        from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

        self._attach_weight_loaders()
        mappings = self.checkpoint_mappings()
        rank = 0 if self.tp_group is None else self.tp_group.rank_in_group
        loaded = SafetensorsCheckpoint(checkpoint_path, cache_dir).load_sharded_pipelined(
            rank, self.world_size, self, mappings, device, strict=False,
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
