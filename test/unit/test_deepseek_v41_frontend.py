# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash front end: vLLM 0.24.0 must parse, accept and size the model.

Run where vLLM is installed (the trn2 host), CPU mode is enough:

    DSV41_HF_DIR=/data/models/DeepSeek-V4.1-Flash \\
    VLLM_NEURON_CPU_MODE=1 python -m pytest test/unit/test_deepseek_v41_frontend.py -v -s

What is checked, against vLLM's own ``ModelConfig`` rather than our expectations of it:

* the config parses through vLLM's ``get_config`` (transformers has no class for it);
* the architecture resolves to the plugin's class and passes vLLM's structural
  text-generation check (inspected in a subprocess, so the editable install must point
  at this tree);
* **no MLA patch is needed**: ``use_mla`` is False, head size is the real 512 and the
  KV head count is 1 at TP 1, 8 and 64 -- unlike GLM-5.3-Flash, whose ``head_dim: 0``
  made vLLM compute a head size of 0;
* the served config advertises no quantization, and the ORIGINAL config is refused by
  vLLM in CPU mode -- the reason ``make_served_dir`` exists, pinned so it cannot go
  stale silently.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

vllm = pytest.importorskip("vllm", reason="front-end tests need vLLM installed")
import vllm.config  # noqa: E402,F401

from vllm.config import ModelConfig, ParallelConfig  # noqa: E402

from vllm_neuron.model.deepseek_v41.config import (  # noqa: E402
    DeepseekV41TextArgs,
    make_served_dir,
)
from vllm_neuron.vllm.platform import NeuronPlatform  # noqa: E402

HF_DIR = os.environ.get("DSV41_HF_DIR")
needs_ckpt = pytest.mark.skipif(not HF_DIR, reason="set DSV41_HF_DIR to the HF checkpoint")


@pytest.fixture(scope="module", autouse=True)
def _registered():
    NeuronPlatform.pre_register_and_update()


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    return make_served_dir(HF_DIR, tmp_path_factory.mktemp("dsv41_served"))


def _model_config(path, **kw):
    return ModelConfig(model=str(path), skip_tokenizer_init=True, max_model_len=131072,
                       limit_mm_per_prompt={"image": 0, "video": 0}, **kw)


@needs_ckpt
def test_served_config_has_no_quantization_but_remembers_it(served):
    cfg = json.loads((Path(served) / "config.json").read_text())
    assert "quantization_config" not in cfg
    assert "quantization_config" not in cfg["text_config"]
    assert cfg["original_quantization_config"]["weight_block_size"] == [32, 32]
    assert (Path(served) / "model-00001-of-00048.safetensors").is_symlink()


@needs_ckpt
def test_vllm_parses_and_sizes_the_model(served):
    mc = _model_config(served)
    print(f"\n  arch={mc.architecture} head={mc.get_head_size()} use_mla={mc.use_mla} "
          f"hybrid={mc.is_hybrid} quant={mc.quantization} sliding={mc.get_sliding_window()} "
          f"max_len={mc.max_model_len} dtype={mc.dtype}")
    assert mc.architecture == "DeepseekV41ForCausalLM"
    assert mc.hf_text_config.model_type == "deepseek_v41_text"
    assert mc.use_mla is False
    assert mc.get_head_size() == 512
    for tp in (1, 8, 64):
        assert mc.get_num_kv_heads(ParallelConfig(tensor_parallel_size=tp)) == 1
        assert mc.get_num_attention_heads(ParallelConfig(tensor_parallel_size=tp)) == 64 // tp
    assert mc.quantization is None
    assert mc.is_hybrid is False
    # the typed view the model reads, from what vLLM actually parsed
    args = DeepseekV41TextArgs.from_hf_config(mc.hf_config)
    assert (args.n_layers, args.head_dim, args.dtype, args.expert_dtype) == (40, 512, "fp8", "fp4")


@needs_ckpt
def test_original_fp8_config_is_refused_in_cpu_mode():
    """Non-vacuity for make_served_dir: if vLLM ever accepted the original config,
    the served copy would be an unnecessary indirection and this would say so."""
    if os.environ.get("VLLM_NEURON_CPU_MODE") != "1":
        pytest.skip("the refusal is specific to CPU mode")
    with pytest.raises(Exception, match="(?i)fp8|quantization"):
        _model_config(HF_DIR)
