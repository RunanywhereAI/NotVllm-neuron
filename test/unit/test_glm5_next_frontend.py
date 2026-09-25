# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash front-end acceptance: vLLM must see and accept the model.

Run where vLLM is installed (the x86_64 box), not on a laptop:

    VLLM_NEURON_CPU_MODE=1 python -m pytest test/unit/test_glm5_next_frontend.py -v

Set ``GLM5NEXT_CONFIG_DIR`` to a directory holding the model's ``config.json``
to run the tests that need the real checkpoint config; they skip otherwise.

The load-bearing one is ``test_upstream_allowlist_still_lacks_glm5_next``. Our
``mla_detect_patch`` exists only because vLLM 0.24.0's ``is_deepseek_mla``
allowlist has no ``glm5_next_text``. If a future vLLM adds it, the patch becomes
redundant — it already steps aside at runtime — but nothing else would tell us,
and a workaround nobody notices has gone stale is how a fork accumulates lies.
"""
from __future__ import annotations

import os

import pytest

vllm = pytest.importorskip("vllm", reason="front-end tests need vLLM installed")
import vllm.config  # noqa: E402,F401  the convertor cannot be imported cold

from vllm.transformers_utils.model_arch_config_convertor import (  # noqa: E402
    ModelArchConfigConvertorBase,
)

from vllm_neuron.model.glm5_next import (  # noqa: E402
    Glm5NextConfig,
    Glm5NextForConditionalGeneration,
)
from vllm_neuron.vllm.patches.mla_detect_patch import (  # noqa: E402
    GLM5_NEXT_TEXT_MODEL_TYPE,
    apply_mla_detect_patch,
    upstream_allowlist_has_glm5_next,
)

CONFIG_DIR = os.environ.get("GLM5NEXT_CONFIG_DIR")
needs_config = pytest.mark.skipif(
    not CONFIG_DIR, reason="set GLM5NEXT_CONFIG_DIR to the model's config directory"
)


@pytest.fixture(scope="module")
def hf_config():
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(CONFIG_DIR)


# ------------------------------------------------------- the staleness guard
def test_upstream_allowlist_still_lacks_glm5_next():
    """If this fails, vLLM now recognises the model and mla_detect_patch must go.

    The patch shadows upstream's logic with ours. That is correct only while
    upstream has no opinion; once it does, ours may diverge from it silently.
    """
    assert not upstream_allowlist_has_glm5_next(), (
        f"vLLM's is_deepseek_mla allowlist now contains "
        f"{GLM5_NEXT_TEXT_MODEL_TYPE!r}. vllm_neuron/vllm/patches/"
        f"mla_detect_patch.py is redundant — delete it and this test."
    )


def test_the_trap_is_still_there():
    """get_head_size() returns head_dim whenever it is `not None` — and 0 is not None.

    If vLLM ever changes that test to a truthiness check, the patch is no longer
    load-bearing for the head size (though it still is for MLAAttentionSpec
    selection), and the reasoning in the patch docstring needs revisiting.
    """
    import inspect

    src = inspect.getsource(ModelArchConfigConvertorBase.get_head_size)
    assert '"head_dim", None) is not None' in src, (
        "vLLM's get_head_size no longer guards head_dim with `is not None`; "
        "re-derive whether head_dim: 0 still reaches the return."
    )


# ------------------------------------------------------------- the patch itself
@needs_config
def test_patch_turns_head_size_from_zero_into_the_latent_width(hf_config):
    text = hf_config.text_config
    assert text.head_dim == 0, "premise changed: the checkpoint no longer ships head_dim 0"
    expected = text.kv_lora_rank + text.qk_rope_head_dim

    convertor = ModelArchConfigConvertorBase(hf_config, text)
    apply_mla_detect_patch()
    patched = ModelArchConfigConvertorBase(hf_config, text)

    assert patched.is_deepseek_mla() is True
    assert patched.get_head_size() == expected == 512
    # and the un-patched value is the thing we are avoiding
    assert convertor.get_head_size.__func__ is patched.get_head_size.__func__


@needs_config
def test_patch_leaves_other_models_alone(hf_config):
    """A model whose text model_type is not ours must be unchanged."""
    apply_mla_detect_patch()

    class _Fake:
        model_type = "llama"
        head_dim = 128

    convertor = ModelArchConfigConvertorBase(hf_config, _Fake())
    assert convertor.is_deepseek_mla() is False
    assert convertor.get_head_size() == 128


# --------------------------------------------------- the four-part vLLM contract
def test_front_end_contract_is_satisfied():
    """vLLM decides `--runner generate` structurally, and is_hybrid is the *fourth* gate.

    ``VllmModel`` needs ``__init__(vllm_config=...)``, ``embed_input_ids`` and
    ``forward``; ``VllmModelForTextGeneration`` adds ``compute_logits``. Only then
    is ``IsHybrid``'s ClassVar consulted. Missing any of the first three fails
    ModelConfig validation with "This model does not support `--runner generate`"
    before is_hybrid is ever read.
    """
    from vllm.model_executor.models import interfaces_base as ib

    cls = Glm5NextForConditionalGeneration
    assert ib._check_vllm_model_init(cls), "__init__ must take a vllm_config keyword"
    assert ib._check_vllm_model_embed_input_ids(cls)
    assert ib._check_vllm_model_forward(cls)
    assert ib.is_vllm_model(cls)
    assert ib.is_text_generation_model(cls), "needs compute_logits"
    assert cls.is_hybrid is True, "IsHybrid ClassVar; nothing upstream declares it for us"


def test_hybrid_state_methods_exist():
    cls = Glm5NextForConditionalGeneration
    assert callable(cls.get_mamba_state_shape_from_config)
    assert callable(cls.get_mamba_state_dtype_from_config)


def test_registered_in_the_plugin_registry():
    from vllm_neuron.model.registry import get_models

    assert "Glm5NextForConditionalGeneration" in dict(get_models())


# --------------------------------------------------------------- the config
@needs_config
def test_config_parses_the_live_checkpoint(hf_config):
    text = Glm5NextConfig.from_hf(hf_config).text_config
    assert text.num_hidden_layers == 45
    assert len(text.kda_layers) == 34 and len(text.sparse_mla_layers) == 11
    assert text.sparse_mla_layers == tuple(range(3, 45, 4))
    assert text.head_size == 512          # derived, never read from head_dim
    assert text.conv_dim == 3 * 64 * 128  # 24576
    assert text.hc_mult == 4 and text.hc_sinkhorn_iters == 20
    assert text.index_topk == 2048 and text.index_kpool == 4
    assert text.swiglu_limit == 10.0 and text.first_k_dense_replace == 3


@needs_config
def test_config_refuses_what_it_has_not_validated(hf_config):
    """Each guard should raise rather than silently approximate."""
    import copy

    for attr, value, match in (
        ("qk_rope_head_dim", 64, "NoPE"),
        ("mla_use_nope", False, "mla_use_nope"),
        ("mhc", False, "mHC"),
        ("n_group", 8, "group"),
        ("scoring_func", "softmax", "scoring_func"),
    ):
        broken = copy.deepcopy(hf_config)
        setattr(broken.text_config, attr, value)
        with pytest.raises(NotImplementedError, match=match):
            Glm5NextConfig.from_hf(broken)


@needs_config
def test_kda_state_shape_is_delegated_not_derived(hf_config):
    """The 4th positional of kda_state_shape is num_k_heads, not conv_kernel_size.

    Passing the kernel width positionally sizes the conv state for 4 key heads
    instead of 64 — a silently wrong allocation. This pins the real answer.
    """
    text = Glm5NextConfig.from_hf(hf_config).text_config
    conv, recurrent = text.state_shapes(64)
    assert conv == (text.linear_conv_kernel_dim - 1, text.conv_dim // 64), conv
    assert recurrent == (1, 128, 128), recurrent
