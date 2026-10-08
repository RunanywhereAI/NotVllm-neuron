# SPDX-License-Identifier: Apache-2.0
"""Front-end entry point for DeepSeek-V4.1-Flash.

``DeepseekV41ForCausalLM`` is the class vLLM's front end sees: ``ModelConfig``
validation happens before any worker exists, so the architecture has to be
registered (``NeuronPlatform.pre_register_and_update``) and pass vLLM's structural
model checks there.

It is a SHELL until the decoder (``model.py``, on branch ``dsv41``) lands: vLLM
decides an architecture supports ``--runner generate`` by *inspecting the class*
(``ModelRegistry.is_text_generation_model`` -> ``interfaces_base``), and the check is
structural in four parts -- an ``__init__`` taking a ``vllm_config`` keyword, an
``embed_input_ids``, a ``forward`` and a ``compute_logits``. They raise when called;
only their presence is inspected. ``from_configs`` is what the runner calls to build
the real model.

Not declared hybrid: V4.1 has no recurrent layers. Its compressor carries a
fixed-size partial group across decode steps, which is a cache-design question for
the model, not a vLLM ``IsHybrid`` contract (declaring it would switch on vLLM's
mamba page alignment).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .config import ARCHITECTURE

if TYPE_CHECKING:
    from transformers import PretrainedConfig

_SHELL = (
    f"{ARCHITECTURE} is the front-end registration; the decoder is built by "
    f"from_configs once vllm_neuron/model/deepseek_v41/model.py exists."
)


class DeepseekV41ForCausalLM(nn.Module):
    """Registered as ``DeepseekV41ForCausalLM`` -- the checkpoint's own architecture
    string, so no ``hf_overrides`` is needed."""

    def __init__(self, vllm_config=None, prefix: str = "") -> None:
        """Signature matters: vLLM's ``VllmModel`` protocol checks for a
        ``vllm_config`` keyword here (``_check_vllm_model_init``)."""
        super().__init__()
        self.vllm_config = vllm_config
        self.prefix = prefix

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(_SHELL)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(_SHELL)

    def compute_logits(self, *args, **kwargs):
        raise NotImplementedError(_SHELL)

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, text_neuron_config=None, **kwargs):
        """Text-only. Delegates to ``model.py`` when present."""
        try:
            from . import model
        except ImportError as exc:  # model.py not merged yet
            raise NotImplementedError(_SHELL) from exc
        return model.from_configs(hf_config, text_neuron_config, **kwargs)


__all__ = ["DeepseekV41ForCausalLM"]
