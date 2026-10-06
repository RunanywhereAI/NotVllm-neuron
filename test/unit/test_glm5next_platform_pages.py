# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash's two page kinds come out equal, through vLLM 0.24.0's own planner.

Run where vLLM is installed (the x86_64 box):

    GLM5NEXT_CONFIG_DIR=<dir with the real config.json> \\
    VLLM_NEURON_CPU_MODE=1 python -m pytest test/unit/test_glm5next_platform_pages.py -v -s

The model folds the DSA indexer's pool keys and tail ring into its MLA page
(``model/glm5_next/cache_layout.py``), so its real attention page is larger than the
``MLAAttentionSpec`` page vLLM's ``_align_hybrid_block_size`` matches the recurrent
state against. ``NeuronPlatform._pad_recurrent_page_to_folded_latent_page`` closes the
gap. These tests run vLLM's real alignment and its real ``unify_kv_cache_spec_page_size``
and grouping on the specs the runner emits, and show the padding step is load-bearing:
without it the planner refuses the model.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

vllm = pytest.importorskip("vllm", reason="needs vLLM installed")
import vllm.config  # noqa: E402,F401

from vllm.config import CacheConfig  # noqa: E402
from vllm.v1.core.kv_cache_utils import (  # noqa: E402
    _get_kv_cache_groups_uniform_page_size,
    unify_kv_cache_spec_page_size,
)
from vllm.v1.kv_cache_interface import MambaSpec, MLAAttentionSpec  # noqa: E402

from vllm_neuron.model.glm5_next import Glm5NextConfig, Glm5NextForConditionalGeneration  # noqa: E402
from vllm_neuron.model.glm5_next.cache_layout import LatentPageLayout  # noqa: E402
from vllm_neuron.vllm.platform import NeuronPlatform  # noqa: E402

CONFIG_DIR = os.environ.get("GLM5NEXT_CONFIG_DIR")
needs_config = pytest.mark.skipif(not CONFIG_DIR, reason="set GLM5NEXT_CONFIG_DIR")


@pytest.fixture(scope="module")
def hf_config():
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(CONFIG_DIR)


@pytest.fixture(autouse=True)
def _registry(monkeypatch):
    """The front end resolves the registered class by architecture; point it at ours
    without a full ModelConfig."""
    from vllm.model_executor.models import ModelRegistry

    monkeypatch.setattr(ModelRegistry, "resolve_model_cls",
                        lambda arch, model_config=None: (Glm5NextForConditionalGeneration, arch))


def _vllm_config(hf_config, tp: int, dtype: torch.dtype):
    text = Glm5NextConfig.from_hf(hf_config).text_config
    model_config = SimpleNamespace(
        hf_config=hf_config, hf_text_config=hf_config.text_config, dtype=dtype,
        architecture="Glm5NextForConditionalGeneration", is_hybrid=True,
        # what mla_detect_patch makes vLLM report (test_glm5_next_frontend pins it)
        use_mla=True, get_num_kv_heads=lambda _pc: 1, get_head_size=lambda: text.head_size,
    )
    return SimpleNamespace(cache_config=CacheConfig(), model_config=model_config,
                           parallel_config=SimpleNamespace(tensor_parallel_size=tp),
                           speculative_config=None), text


def _specs(vc, text, n_kda: int, n_mla: int, dtype):
    """What ``NeuronModelRunner.get_kv_cache_spec`` emits for this model."""
    cc = vc.cache_config
    B = cc.block_size
    lay = LatentPageLayout.from_config(text, B, dtype)
    specs = {}
    for i in range(n_kda):
        specs[f"kda{i}"] = MambaSpec(
            shapes=Glm5NextForConditionalGeneration.get_mamba_state_shape_from_config(vc),
            dtypes=Glm5NextForConditionalGeneration.get_mamba_state_dtype_from_config(vc),
            block_size=4096, page_size_padded=cc.mamba_page_size_padded,
            mamba_cache_mode=cc.mamba_cache_mode)
    for i in range(n_mla):
        specs[f"mla{i}"] = MLAAttentionSpec(block_size=B, num_kv_heads=1,
                                            head_size=text.kv_lora_rank, dtype=dtype,
                                            page_size_padded=lay.total_bytes)
    return specs, lay


@needs_config
def test_real_config_at_tp64_aligns_to_one_page(hf_config):
    vc, text = _vllm_config(hf_config, tp=64, dtype=torch.bfloat16)
    NeuronPlatform._align_hybrid_page_sizes(vc)
    cc = vc.cache_config
    specs, lay = _specs(vc, text, n_kda=34, n_mla=11, dtype=torch.bfloat16)
    print(f"\n  block_size {cc.block_size}; {lay.describe()}; "
          f"recurrent page padded to {cc.mamba_page_size_padded}")
    # arithmetic from the published config: 67840 B of state at TP=64 needs 96 tokens of
    # 1024 B latent; the folded page adds 24 bf16 pool keys and the tail ring
    assert cc.block_size == 96
    assert cc.mamba_page_size_padded == lay.total_bytes == 106496
    sizes = {s.page_size_bytes for s in specs.values()}
    assert sizes == {106496}
    unified = unify_kv_cache_spec_page_size(specs)
    groups = _get_kv_cache_groups_uniform_page_size(unified)
    print(f"  {len(groups)} KV cache groups: "
          + ", ".join(f"{type(g.kv_cache_spec).__name__}x{len(g.layer_names)}" for g in groups))
    assert sum(len(g.layer_names) for g in groups) == 45


@needs_config
def test_without_the_padding_step_vllm_refuses_the_model(hf_config):
    """Non-vacuity: vLLM's alignment alone matches the recurrent page to the LATENT
    page, and the folded page is bigger, so the planner cannot unify them."""
    vc, text = _vllm_config(hf_config, tp=64, dtype=torch.bfloat16)
    super(NeuronPlatform, NeuronPlatform)._align_hybrid_block_size(vc, _alignment_backend())
    specs, lay = _specs(vc, text, n_kda=34, n_mla=11, dtype=torch.bfloat16)
    assert vc.cache_config.mamba_page_size_padded == 96 * 1024 != lay.total_bytes
    with pytest.raises((NotImplementedError, AssertionError)):
        unify_kv_cache_spec_page_size(specs)


def _alignment_backend():
    from vllm.v1.attention.backend import MultipleOf

    class _B:
        @staticmethod
        def get_supported_kernel_block_sizes():
            return [MultipleOf(NeuronPlatform._KERNEL_BLOCK_ALIGNMENT)]

        @staticmethod
        def get_name():
            return "neuron"

    return _B
