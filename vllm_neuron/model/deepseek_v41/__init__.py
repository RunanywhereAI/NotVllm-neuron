# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash (``model_type: deepseek_v41``).

The package exports what vLLM's registry names
(``vllm_neuron.model.deepseek_v41:DeepseekV41ForCausalLM``) and the config classes.
Importing it registers the configs with ``AutoConfig`` (transformers only, no vLLM).
"""
from .config import DeepseekV41Config, DeepseekV41TextArgs, DeepseekV41TextConfig
from .factory import DeepseekV41ForCausalLM

__all__ = [
    "DeepseekV41Config",
    "DeepseekV41TextConfig",
    "DeepseekV41TextArgs",
    "DeepseekV41ForCausalLM",
]
