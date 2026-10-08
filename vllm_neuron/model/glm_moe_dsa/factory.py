# SPDX-License-Identifier: Apache-2.0
"""Front-end entry point for GLM-5.3 (``GlmMoeDsaForCausalLM``).

vLLM 0.24 already registers this architecture (its DeepseekV2 subclass), so the front
end validates it unaided; the worker's registry pass (``model/registry.py``) replaces
vLLM's class with this one, and the runner builds the real decoder with
``from_configs``. The methods below exist for vLLM's structural checks only.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_SHELL = "GlmMoeDsaForCausalLM is built by from_configs; this class is the registration"


class GlmMoeDsaForCausalLM(nn.Module):
    def __init__(self, vllm_config=None, prefix: str = "") -> None:
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
    def from_configs(cls, hf_config, text_neuron_config=None, **kwargs):
        from . import model

        return model.from_configs(hf_config, text_neuron_config, **kwargs)


__all__ = ["GlmMoeDsaForCausalLM"]
