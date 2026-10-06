# SPDX-License-Identifier: Apache-2.0
"""Front-end entry point for GLM-5.3-Flash.

This class exists so vLLM's **front end** can see and accept the model:
``ModelConfig`` validation happens before any worker is created, so the
architecture has to be registered and its hybrid contract satisfied there.

The decoder itself is ``model.py``; ``from_configs`` builds it. Everything else here is
what vLLM asks about *before* it would ever call ``forward``.

The ``IsHybrid`` contract, and why we must satisfy it ourselves
--------------------------------------------------------------
``ModelConfig.is_hybrid`` reads the ``IsHybrid`` protocol's ``is_hybrid``
ClassVar off the **registered class** (``models/registry.py`` →
``interfaces.py``), and vLLM asks the registered class for the state shapes too —
not its own implementation. PR #54's Qwen3.5 port supplies the two classmethods
but never declares ``is_hybrid``, because vLLM's own
``Qwen3_5ForConditionalGeneration`` already declares it and PR #54 inherits it.

There is no such fallback for GLM-5.3-Flash: vLLM 0.24.0 has no ``glm5_next``
entry at all. And the failure is **silent** rather than loud — ``is_hybrid``
defaults False, the hybrid page-size reconciliation early-returns,
``cache_config.mamba_block_size`` stays ``None``, and the error surfaces much
later and further away.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from transformers import PretrainedConfig

from .config import Glm5NextConfig

_FRONT_END_ONLY = (
    "Glm5NextForConditionalGeneration is the front-end registration; the runner builds "
    "the decoder through from_configs, which returns a model.Glm5NextForCausalLM."
)


class Glm5NextForConditionalGeneration(nn.Module):
    """Registered as ``Glm5NextForConditionalGeneration`` in the vLLM front end.

    Subclasses ``nn.Module`` and declares ``forward``/``compute_logits`` because
    vLLM decides a registered architecture supports ``--runner generate`` by
    *inspecting the class* (``ModelRegistry.is_text_generation_model`` ->
    ``interfaces_base``), not by calling it, and the check is structural in four
    parts. ``VllmModel`` needs an ``__init__`` taking a ``vllm_config`` keyword
    (``_check_vllm_model_init``), an ``embed_input_ids`` and a ``forward``;
    ``VllmModelForTextGeneration`` adds ``compute_logits``. Miss any one and
    ``ModelConfig`` validation fails with "This model does not support
    ``--runner generate``" long before anything else here is reached. On this class
    all of them raise when called; only their presence is inspected. The working ones
    are on ``model.Glm5NextForCausalLM``, which ``from_configs`` returns.
    """

    # vLLM's IsHybrid protocol. Declared explicitly: nothing upstream declares it
    # for us, and its absence fails silently rather than loudly.
    is_hybrid: ClassVar[Literal[True]] = True

    @classmethod
    def _text_config(cls, vllm_config) -> "object":
        hf_config = vllm_config.model_config.hf_config
        return Glm5NextConfig.from_hf(hf_config).text_config

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config,
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        """Per-rank ``(conv_state, recurrent_state)`` for one KDA layer.

        Delegated to ``MambaStateShapeCalculator.kda_state_shape`` via the config,
        deliberately: vLLM sizes the state *pages* from that same helper, so a
        second derivation here would be a silent memory-aliasing bug rather than
        a loud disagreement.
        """
        text = cls._text_config(vllm_config)
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        return text.state_shapes(tp_size)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config,
    ) -> tuple[torch.dtype, torch.dtype]:
        """``(conv_dtype, recurrent_dtype)``.

        The conv window holds activations and follows the model dtype; the
        recurrent state accumulates over the whole sequence and is kept in
        float32 so the delta rule does not drift.
        """
        return cls._text_config(vllm_config).state_dtypes()

    @classmethod
    def get_latent_page_bytes_from_config(cls, vllm_config, block_size: int) -> int:
        """Bytes of one folded MLA page (latent + indexer pools + tail ring) at
        ``block_size``.

        The platform's hybrid page alignment calls this through the registered class,
        exactly as vLLM calls ``get_mamba_state_shape_from_config``, and pads the
        recurrent page up to it; the model reports the same number to the runner. Both
        go through ``cache_layout.latent_page_bytes``, so the two cannot disagree.
        """
        from .cache_layout import latent_page_bytes

        text = cls._text_config(vllm_config)
        cache_dtype = vllm_config.cache_config.cache_dtype
        if cache_dtype != "auto":
            raise NotImplementedError(
                f"kv cache dtype {cache_dtype!r}: the folded MLA page is laid out at "
                f"the model dtype; a quantized latent cache is not implemented"
            )
        return latent_page_bytes(text, vllm_config.model_config.dtype, block_size)

    def __init__(self, vllm_config=None, prefix: str = "") -> None:
        """Signature matters: vLLM's ``VllmModel`` protocol checks for a
        ``vllm_config`` keyword here (``_check_vllm_model_init``)."""
        super().__init__()
        self.vllm_config = vllm_config
        self.prefix = prefix

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(_FRONT_END_ONLY)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(_FRONT_END_ONLY)

    def compute_logits(self, *args, **kwargs):
        raise NotImplementedError(_FRONT_END_ONLY)

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: object | None = None,
        **kwargs: object,
    ):
        """Text-only: a ``vision_neuron_config`` is ignored and the vision tower is never
        built (serve with ``limit_mm_per_prompt={"image": 0, "video": 0}``)."""
        from .model import Glm5NextForCausalLM

        return Glm5NextForCausalLM.from_configs(hf_config, text_neuron_config)


__all__ = ["Glm5NextForConditionalGeneration"]
