# SPDX-License-Identifier: Apache-2.0
"""Configs for GLM-5.3-Flash (``model_type: glm5_next``).

A hybrid decoder of 45 layers: 34 KDA (Kimi Delta Attention, a linear recurrence
with per-channel forget gates) and 11 NoPE sparse-MLA layers with a DSA indexer,
interleaved ``[linear ×3, sparse-MLA] ×11`` plus a trailing linear layer. Every
layer carries manifold-constrained hyper-connections over 4 residual streams, and
all but the first three use a 288-expert MoE.

**Two model_type values, and both matter.** The top-level config is
``glm5_next``; the nested text config is ``glm5_next_text``. vLLM reads
``hf_text_config.model_type``, so the string that has to be recognised for MLA
detection is the *text* one — see ``vllm_neuron/vllm/patches/mla_detect_patch.py``.

Fields are read from the checkpoint rather than hard-coded, and anything this
implementation has not been validated against raises instead of being silently
approximated. Every constant below was cross-referenced against transformers
5.17.0 ``models/glm5_next/modeling_glm5_next.py`` and the live
``zai-org/GLM-5.3-Flash`` config on 2026-09-25; the audit is
``personal_docs/GLM53_ORACLE_AUDIT.md`` on branch ``oracle-provenance-audit``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # annotations only: the CPU oracle comparison imports this module
    from transformers import PretrainedConfig  # on hosts without transformers

# <-- MODEL-SPECIFIC: the two entries HF uses in text_config.layer_types.
LINEAR_ATTENTION = "linear_attention"
SPARSE_MLA = "deepseek_sparse_attention"
# ... and in mlp_layer_types.
DENSE_MLP = "dense"
SPARSE_MLP = "sparse"


def _dtype_of(cfg: PretrainedConfig, default: torch.dtype) -> torch.dtype:
    raw = getattr(cfg, "dtype", None) or getattr(cfg, "torch_dtype", None)
    if raw is None:
        return default
    if isinstance(raw, torch.dtype):
        return raw
    return getattr(torch, str(raw).replace("torch.", ""))


# checkpoint ``linear_attn_config`` key -> transformers' flattened attribute
_LINEAR_KEYS = {
    "num_heads": "linear_num_heads",
    "head_dim": "linear_head_dim",
    "short_conv_kernel_size": "linear_conv_kernel_dim",
    "gate_lower_bound": "linear_lower_bound",
}


def _linear_attn_fields(text_cfg) -> dict:
    """The four KDA geometry values, from whichever form the config object carries.

    ``config.json`` nests them in ``linear_attn_config``; transformers 5.17's
    ``Glm5NextTextConfig.__post_init__`` copies that dict onto flat ``linear_*``
    attributes, and whether the dict itself survives on the object is transformers'
    business, not ours. Accept either, require every value, and refuse a config whose
    two forms disagree -- no defaults, because a defaulted kernel width or gate bound
    that happens to equal the real one is indistinguishable from a wired one.
    """
    lac = getattr(text_cfg, "linear_attn_config", None)
    out = {}
    for key, flat in _LINEAR_KEYS.items():
        nested = lac.get(key) if isinstance(lac, dict) else None
        attr = getattr(text_cfg, flat, None)
        if nested is not None and attr is not None and nested != attr:
            raise ValueError(
                f"linear_attn_config[{key!r}]={nested!r} disagrees with {flat}={attr!r}"
            )
        value = nested if nested is not None else attr
        if value is None:
            raise ValueError(
                f"GLM-5.3-Flash config has neither linear_attn_config[{key!r}] nor {flat}"
            )
        out[key] = value
    return out


@dataclass
class Glm5NextTextConfig:
    """The text decoder: hybrid KDA + NoPE sparse-MLA, mHC, MoE."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    layer_types: tuple[str, ...]
    mlp_layer_types: tuple[str, ...]

    # sparse-MLA (NoPE)
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int

    # DSA indexer
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    index_kpool: int

    # KDA (linear_attn_config)
    linear_num_heads: int
    linear_head_dim: int
    linear_conv_kernel_dim: int
    linear_lower_bound: float

    # mHC
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float

    # MoE
    n_routed_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    n_shared_experts: int
    routed_scaling_factor: float
    norm_topk_prob: bool
    swiglu_limit: float
    first_k_dense_replace: int

    vocab_size: int
    rms_norm_eps: float
    tie_word_embeddings: bool
    max_position_embeddings: int
    torch_dtype: torch.dtype

    neuron_config: object | None = None

    # ---- derived -----------------------------------------------------------
    @property
    def head_size(self) -> int:
        """The MLA latent width: ``kv_lora_rank + qk_rope_head_dim``.

        **Do not read ``head_dim`` from the checkpoint.** GLM-5.3-Flash ships
        ``text_config.head_dim: 0``, and vLLM's ``get_head_size()`` tests
        ``is not None`` — ``0 is not None`` — so it returns 0 rather than this.
        That is the trap ``mla_detect_patch`` exists to close; this property is
        the value everything sizing a page actually wants.
        """
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def sparse_mla_layers(self) -> tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.layer_types) if t == SPARSE_MLA)

    @property
    def kda_layers(self) -> tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.layer_types) if t == LINEAR_ATTENTION)

    @property
    def conv_dim(self) -> int:
        """Channels the KDA depthwise conv runs over: q, k and v concatenated."""
        return 3 * self.linear_num_heads * self.linear_head_dim

    def state_shapes(self, tp_size: int) -> tuple[tuple[int, ...], ...]:
        """Per-rank ``(conv_state_shape, recurrent_state_shape)`` for one KDA layer.

        Delegated to vLLM's ``MambaStateShapeCalculator.kda_state_shape`` rather
        than derived here, for the same reason PR #54 delegates the GDN shapes:
        vLLM sizes the state *pages* from the same helper, so re-deriving risks a
        layout that disagrees with the pages the planner allocated — a silent
        aliasing bug rather than an error.
        """
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateShapeCalculator,
        )

        return tuple(
            MambaStateShapeCalculator.kda_state_shape(
                tp_world_size=tp_size,
                num_heads=self.linear_num_heads,
                head_dim=self.linear_head_dim,
                # Keyword, not positional: the 4th positional parameter is
                # ``num_k_heads``, so passing the kernel width there would
                # silently size the conv state for 4 key heads instead of 64.
                conv_kernel_size=self.linear_conv_kernel_dim,
                num_spec=0,  # the MTP head is not wired up
            )
        )

    def state_dtypes(self) -> tuple[torch.dtype, torch.dtype]:
        """``(conv_state, recurrent_state)`` dtypes for one KDA layer.

        The conv window holds activations and follows the model dtype; the recurrent
        state accumulates over the whole sequence and is kept in float32 so the delta
        rule does not drift. The registered class reports these to vLLM and the model
        reports them to the runner, from this one place.
        """
        return (self.torch_dtype, torch.float32)

    @classmethod
    def from_hf(cls, text_cfg: PretrainedConfig) -> Glm5NextTextConfig:
        layer_types = tuple(getattr(text_cfg, "layer_types"))
        unknown = set(layer_types) - {LINEAR_ATTENTION, SPARSE_MLA}
        if unknown:
            raise NotImplementedError(
                f"unknown GLM-5.3-Flash layer types {sorted(unknown)}; only "
                f"{LINEAR_ATTENTION!r} and {SPARSE_MLA!r} are implemented."
            )
        if len(layer_types) != text_cfg.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(layer_types)} entries but "
                f"num_hidden_layers is {text_cfg.num_hidden_layers}"
            )

        mlp_layer_types = tuple(getattr(text_cfg, "mlp_layer_types"))
        unknown_mlp = set(mlp_layer_types) - {DENSE_MLP, SPARSE_MLP}
        if unknown_mlp:
            raise NotImplementedError(
                f"unknown GLM-5.3-Flash mlp_layer_types {sorted(unknown_mlp)}"
            )
        if len(mlp_layer_types) != text_cfg.num_hidden_layers:
            raise ValueError(
                f"mlp_layer_types has {len(mlp_layer_types)} entries but "
                f"num_hidden_layers is {text_cfg.num_hidden_layers}"
            )

        # <-- MODEL-SPECIFIC: NoPE. qk_rope_head_dim is 0 in this checkpoint and
        # there is no rotary anywhere in the attention stack; position information
        # comes from the KDA layers. A non-zero value means a different model.
        if getattr(text_cfg, "qk_rope_head_dim", 0) != 0:
            raise NotImplementedError(
                f"qk_rope_head_dim={text_cfg.qk_rope_head_dim} is not implemented; "
                f"this port is NoPE-only (mla_use_nope), which is what the "
                f"released checkpoint ships."
            )
        if not getattr(text_cfg, "mla_use_nope", False):
            raise NotImplementedError(
                "GLM-5.3-Flash with mla_use_nope=False is not implemented."
            )
        if not getattr(text_cfg, "mhc", False):
            raise NotImplementedError(
                "GLM-5.3-Flash without mHC is not implemented; the residual "
                "stream count and the hyper-connection weights depend on it."
            )

        # The router's group logic is the identity only while n_group == 1; the
        # oracle omits it on exactly that basis, so refuse anything else loudly
        # rather than route differently from the reference.
        n_group = getattr(text_cfg, "n_group", 1)
        topk_group = getattr(text_cfg, "topk_group", 1)
        if n_group != 1 or topk_group != 1:
            raise NotImplementedError(
                f"n_group={n_group}, topk_group={topk_group}: expert-group "
                f"masking is not implemented. It is the identity only at 1/1, "
                f"which is what the released checkpoint ships."
            )
        scoring = getattr(text_cfg, "scoring_func", "sigmoid")
        if scoring != "sigmoid":
            raise NotImplementedError(f"scoring_func={scoring!r} is not implemented.")

        # Behaviour the model hardcodes because the released checkpoint fixes it. Each
        # is read with the released value as its default only so that a config which
        # omits the key still loads; a config that says otherwise is refused.
        if not getattr(text_cfg, "index_kpool_compress", True):
            raise NotImplementedError(
                "index_kpool_compress=False is not implemented; the indexer only "
                "scores kpool-compressed pools."
            )
        if not getattr(text_cfg, "index_kpool_always_select_tail", True):
            raise NotImplementedError(
                "index_kpool_always_select_tail=False is not implemented."
            )
        if getattr(text_cfg, "hidden_act", "silu") != "silu":
            raise NotImplementedError(
                f"hidden_act={text_cfg.hidden_act!r}; the clamped SwiGLU and the KDA "
                f"conv activation are silu."
            )
        topk_method = getattr(text_cfg, "topk_method", "noaux_tc")
        if topk_method != "noaux_tc":
            raise NotImplementedError(f"topk_method={topk_method!r} is not implemented.")

        indexer_types = tuple(getattr(text_cfg, "indexer_types", ()) or ())
        if indexer_types and set(indexer_types) != {"full"}:
            raise NotImplementedError(
                f"indexer_types {sorted(set(indexer_types))} is not implemented; "
                f"cross-layer top-k sharing ('shared') needs the indexer to "
                f"propagate selections between layers."
            )

        linear = _linear_attn_fields(text_cfg)

        return cls(
            hidden_size=text_cfg.hidden_size,
            intermediate_size=text_cfg.intermediate_size,
            num_hidden_layers=text_cfg.num_hidden_layers,
            layer_types=layer_types,
            mlp_layer_types=mlp_layer_types,
            num_attention_heads=text_cfg.num_attention_heads,
            q_lora_rank=text_cfg.q_lora_rank,
            kv_lora_rank=text_cfg.kv_lora_rank,
            qk_nope_head_dim=text_cfg.qk_nope_head_dim,
            qk_rope_head_dim=text_cfg.qk_rope_head_dim,
            v_head_dim=text_cfg.v_head_dim,
            index_n_heads=text_cfg.index_n_heads,
            index_head_dim=text_cfg.index_head_dim,
            index_topk=text_cfg.index_topk,
            index_kpool=text_cfg.index_kpool,
            linear_num_heads=linear["num_heads"],
            linear_head_dim=linear["head_dim"],
            linear_conv_kernel_dim=linear["short_conv_kernel_size"],
            linear_lower_bound=linear["gate_lower_bound"],
            hc_mult=text_cfg.hc_mult,
            hc_sinkhorn_iters=text_cfg.hc_sinkhorn_iters,
            hc_eps=text_cfg.hc_eps,
            n_routed_experts=text_cfg.n_routed_experts,
            num_experts_per_tok=text_cfg.num_experts_per_tok,
            moe_intermediate_size=text_cfg.moe_intermediate_size,
            n_shared_experts=text_cfg.n_shared_experts,
            routed_scaling_factor=float(text_cfg.routed_scaling_factor),
            norm_topk_prob=bool(text_cfg.norm_topk_prob),
            swiglu_limit=float(text_cfg.swiglu_limit),
            first_k_dense_replace=text_cfg.first_k_dense_replace,
            vocab_size=text_cfg.vocab_size,
            rms_norm_eps=text_cfg.rms_norm_eps,
            tie_word_embeddings=bool(getattr(text_cfg, "tie_word_embeddings", False)),
            max_position_embeddings=text_cfg.max_position_embeddings,
            torch_dtype=_dtype_of(text_cfg, torch.bfloat16),
        )


@dataclass
class Glm5NextConfig:
    """Top-level config. Text-only scope: the vision tower is not built."""

    text_config: Glm5NextTextConfig
    neuron_config: object | None = None
    extras: dict = field(default_factory=dict)

    @classmethod
    def from_hf(cls, hf_config: PretrainedConfig) -> Glm5NextConfig:
        text_cfg = getattr(hf_config, "text_config", None)
        if text_cfg is None:
            raise ValueError("GLM-5.3-Flash config is missing text_config")
        return cls(text_config=Glm5NextTextConfig.from_hf(text_cfg))

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: object | None = None,
        **_: object,
    ) -> Glm5NextConfig:
        """Entry point used by the factory: HF config plus the runner's NeuronConfig."""
        config = cls.from_hf(hf_config)
        config.text_config.neuron_config = text_neuron_config
        config.neuron_config = text_neuron_config
        return config


__all__ = ["Glm5NextConfig", "Glm5NextTextConfig", "LINEAR_ATTENTION", "SPARSE_MLA"]
