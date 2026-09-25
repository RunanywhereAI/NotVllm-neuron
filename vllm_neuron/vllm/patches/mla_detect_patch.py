# SPDX-License-Identifier: Apache-2.0
"""Teach vLLM that GLM-5.3-Flash is a DeepSeek-style MLA model.

``ModelConfig.use_mla`` is ``is_deepseek_mla and not VLLM_MLA_DISABLE``. That flag
comes from a **hardcoded model_type allowlist** in
``vllm.transformers_utils.model_arch_config_convertor.ModelArchConfigConvertorBase
.is_deepseek_mla``. At our pin (vLLM 0.24.0) it contains ``deepseek_v2/v3/v32/v4``,
``glm_moe_dsa``, ``glm4_moe_lite``, ``kimi_k2``, ``kimi_linear``, ``longcat_flash``
and others — and **not** ``glm5_next_text``.

Note the string: the top-level config is ``model_type: glm5_next`` while the nested
text config is ``glm5_next_text``, and the allowlist is tested against
``hf_text_config.model_type``.

Why this matters, and why it is not cosmetic
--------------------------------------------
With ``use_mla`` False, ``get_head_size()`` skips its MLA branch — which would
return ``kv_lora_rank + qk_rope_head_dim`` = 512 — and falls through to::

    # NOTE: Some configs may set head_dim=None in the config
    if getattr(self.hf_text_config, "head_dim", None) is not None:
        return self.hf_text_config.head_dim

**GLM-5.3-Flash ships ``text_config.head_dim: 0``.** ``0 is not None``, so
``get_head_size()`` returns **0**, not 512. ``get_num_kv_heads()`` likewise returns
``num_key_value_heads // TP`` = ``64 // TP`` instead of the MLA-correct 1. Every
page-size calculation then starts from zero; vLLM's own
``Platform._align_hybrid_block_size`` divides by ``attn_page_size_1_token`` and
raises ``ZeroDivisionError``.

Why patch the allowlist rather than ``head_dim``
------------------------------------------------
Forcing ``head_dim`` to a plausible non-zero value (via ``hf_overrides`` or
otherwise) trades a loud crash for silently wrong page sizes everywhere — a model
that loads, runs, and corrupts a cache group. Two further reasons this is the
right layer:

* **It is true.** GLM-5.3-Flash genuinely is DeepSeek-style MLA: ``q_lora_rank``
  1536, ``kv_lora_rank`` 512, a ``qk_nope``/``qk_rope`` split. Saying so is
  accurate, where overriding ``model_type`` would misrepresent the checkpoint to
  every other part of the front end that reads it.
* **It selects the structurally correct path.** ``_align_hybrid_block_size``
  (``platforms/interface.py:668``) branches on ``use_mla`` and builds an
  ``MLAAttentionSpec`` rather than a ``FullAttentionSpec``. That is what a latent
  KV cache needs — one 512-wide entry per token, not 64 heads of something.

Everything else that reads ``use_mla`` at our pin was checked and is inert for us:
an nvfp4-KV-cache guard (we use bf16), an ``__repr__`` field, an assert that MLA is
not combined with sliding-window attention (GLM-5.3-Flash has none), vLLM's own
model implementations (the plugin reimplements models), and the disaggregated
KV-transfer paths (no connector configured).

Mechanics: ``ModelConfig.get_model_arch_config`` looks the convertor class up by the
**top-level** ``model_type`` and falls back to ``ModelArchConfigConvertorBase``,
then stores ``convertor.convert()``. Patching the base class's method is therefore
both sufficient and the narrowest option.
"""

import logging

logger = logging.getLogger(__name__)

# The value the allowlist must recognise. This is the *text* config's model_type.
GLM5_NEXT_TEXT_MODEL_TYPE = "glm5_next_text"

_applied = False


def _flatten(consts) -> set:
    """Constants, with tuple literals unpacked one level.

    ``co_consts`` holds a tuple literal as a single entry, so a membership test
    against it silently misses every string inside the allowlist.
    """
    out = set()
    for c in consts:
        if isinstance(c, (tuple, frozenset)):
            out.update(x for x in c if isinstance(x, str))
        elif isinstance(c, str):
            out.add(c)
    return out


def upstream_allowlist_has_glm5_next() -> bool:
    """True once vLLM recognises GLM-5.3-Flash itself, making this patch redundant.

    Read from the function's own constants rather than by calling it, so the check
    does not depend on constructing a config.
    """
    from vllm.transformers_utils import model_arch_config_convertor as conv

    fn = getattr(conv.ModelArchConfigConvertorBase.is_deepseek_mla,
                 "__wrapped_original__",
                 conv.ModelArchConfigConvertorBase.is_deepseek_mla)
    code = getattr(fn, "__code__", None)
    if code is None:
        return False
    # The allowlist is a tuple literal, which CPython stores as ONE constant.
    # Searching co_consts for the string directly never matches — the first
    # version of this function did exactly that and could never fire.
    return GLM5_NEXT_TEXT_MODEL_TYPE in _flatten(code.co_consts)


def apply_mla_detect_patch() -> None:
    global _applied
    if _applied:
        return
    _applied = True

    from vllm.transformers_utils import model_arch_config_convertor as conv

    base = conv.ModelArchConfigConvertorBase
    original = base.is_deepseek_mla

    if upstream_allowlist_has_glm5_next():
        # The premise changed: vLLM now recognises the model on its own. Leaving
        # the patch in would shadow upstream's logic with ours, which may differ.
        logger.warning(
            "vLLM's is_deepseek_mla allowlist now contains %r; "
            "vllm_neuron's mla_detect_patch is redundant and was NOT applied. "
            "Remove it.",
            GLM5_NEXT_TEXT_MODEL_TYPE,
        )
        return

    def is_deepseek_mla(self) -> bool:
        model_type = getattr(self.hf_text_config, "model_type", None)
        if model_type == GLM5_NEXT_TEXT_MODEL_TYPE:
            return True
        return original(self)

    is_deepseek_mla.__wrapped_original__ = original
    base.is_deepseek_mla = is_deepseek_mla
    logger.info(
        "MLA-detect patch applied (%s recognised as DeepSeek-style MLA)",
        GLM5_NEXT_TEXT_MODEL_TYPE,
    )
