# SPDX-License-Identifier: Apache-2.0
"""Front-end entry point for GLM-5.3-Flash.

This class exists so vLLM's **front end** can see and accept the model:
``ModelConfig`` validation happens before any worker is created, so the
architecture has to be registered and its hybrid contract satisfied there.

Scope note: the decoder itself (KDA layers, sparse-MLA, the DSA indexer) is not
here. ``from_configs`` raises until that lands. Everything below is the part vLLM
asks about *before* it would ever call ``forward``.

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

from typing import ClassVar, Literal

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from .config import Glm5NextConfig

_NOT_YET = (
    "GLM-5.3-Flash decoder is not implemented yet. This class provides only the "
    "vLLM front-end contract: architecture registration, the IsHybrid protocol "
    "and the KDA state geometry. The KDA and sparse-MLA layers land separately."
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
    ``--runner generate``" long before anything else here is reached. All of them
    raise when called; only their presence is inspected.
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
        text = cls._text_config(vllm_config)
        return (text.torch_dtype, torch.float32)

    def __init__(self, vllm_config=None, prefix: str = "") -> None:
        """Signature matters: vLLM's ``VllmModel`` protocol checks for a
        ``vllm_config`` keyword here (``_check_vllm_model_init``)."""
        super().__init__()
        self.vllm_config = vllm_config
        self.prefix = prefix

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(_NOT_YET)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(_NOT_YET)

    def compute_logits(self, *args, **kwargs):
        raise NotImplementedError(_NOT_YET)

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: object | None = None,
        **kwargs: object,
    ):
        raise NotImplementedError(_NOT_YET)


__all__ = ["Glm5NextForConditionalGeneration"]
