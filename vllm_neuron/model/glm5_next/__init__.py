# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash (``model_type: glm5_next``) plugin modules.

The package exports only what vLLM's registry names
(``vllm_neuron.model.glm5_next:Glm5NextForConditionalGeneration``). Those imports
need torch and transformers, never vLLM.

The NKI kernels (``nki_kda_tkg``, ``nki_kda_cte``) and the attention layers must stay
loadable on a host without vLLM; ``personal_reference/glm5_next/sim/simulate_kernels.py``
and the oracle-comparison tests load them **by file path**, which does not execute
this file.
"""
from .config import Glm5NextConfig, Glm5NextTextConfig
from .factory import Glm5NextForConditionalGeneration

__all__ = [
    "Glm5NextConfig",
    "Glm5NextTextConfig",
    "Glm5NextForConditionalGeneration",
]
