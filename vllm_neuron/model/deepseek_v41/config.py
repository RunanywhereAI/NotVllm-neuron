# SPDX-License-Identifier: Apache-2.0
"""Configs for DeepSeek-V4.1-Flash (``model_type: deepseek_v41``).

Three things live here:

* **``DeepseekV41Config`` / ``DeepseekV41TextConfig``** -- ``PretrainedConfig``
  subclasses so vLLM's front end can parse ``config.json`` at all. transformers 5.x
  ships no config class for this model and vLLM 0.24.0's ``_CONFIG_REGISTRY`` has
  ``deepseek_v4`` but not ``deepseek_v41``; without these, ``ModelConfig`` fails before
  any plugin code runs. Registered with ``AutoConfig`` at import and with vLLM by
  ``register_with_vllm`` (called from ``NeuronPlatform.pre_register_and_update``).
* **``DeepseekV41TextArgs``** -- the validated, typed view the model reads, in DeepSeek's
  reference field names (``ref/model.py``'s ``ModelArgs``). The HF config and
  DeepSeek's own inference config spell the same constants differently (``hidden_size``
  vs ``dim``, ``sliding_window`` vs ``window_size``, ...); ``to_reference_args`` maps one
  onto the other, and a test checks it key by key against the inference config DeepSeek
  ships, which is an independent source.
* **``converted_config``** -- the served ``config.json`` without ``quantization_config``.
  The checkpoint is FP8 (32x32 e8m0 blocks) with MXFP4 experts; the plugin dequantizes at
  load (``weights.py``), and vLLM's front end refuses an fp8-advertising config in CPU
  mode before any plugin code runs.

Unlike GLM-5.3-Flash, no MLA-detection patch is needed: ``head_dim`` (512) is real and
``num_key_value_heads`` is 1, and ``deepseek_v41_text`` is not in vLLM's
``is_deepseek_mla`` allowlist, so vLLM reports head size 512 and one KV head at any TP.
Checked against vLLM 0.24.0 by ``test/unit/test_deepseek_v41_frontend.py``.

Imports transformers but not vLLM, so the oracle-side tests can import it on a laptop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

from transformers import AutoConfig, PretrainedConfig

TEXT_MODEL_TYPE = "deepseek_v41_text"
MODEL_TYPE = "deepseek_v41"
ARCHITECTURE = "DeepseekV41ForCausalLM"
QUANT_KEY = "quantization_config"


class DeepseekV41TextConfig(PretrainedConfig):
    """``text_config``. Keeps every key of the checkpoint as an attribute; the typed,
    validated view is ``DeepseekV41TextArgs.from_hf``."""

    model_type = TEXT_MODEL_TYPE
    base_config_key = "text_config"

    def __init__(self, max_position_embeddings: int = 1048576, rope_scaling=None,
                 rope_parameters=None, rope_theta: float = 10000.0, **kwargs):
        # transformers 5.x standardizes RoPE inside PretrainedConfig.__init__ and needs
        # these set first; same shape as vLLM 0.24.0's own DeepseekV4Config.
        self.max_position_embeddings = max_position_embeddings
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta
        self.rope_parameters = rope_scaling or rope_parameters
        super().__init__(**kwargs)


class DeepseekV41Config(PretrainedConfig):
    """Top level: ``text_config`` plus an unused ``vision_config`` (text-only scope)."""

    model_type = MODEL_TYPE
    sub_configs = {"text_config": DeepseekV41TextConfig}

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        if isinstance(text_config, dict):
            text_config = DeepseekV41TextConfig(**text_config)
        self.text_config = text_config if text_config is not None else DeepseekV41TextConfig()
        # Kept as a plain dict: the vision tower is not built, and vLLM only needs to
        # see that it exists (multimodal limits of 0 then serve text-only).
        self.vision_config = vision_config
        super().__init__(**kwargs)

    def get_text_config(self, *args, **kwargs):
        return self.text_config


AutoConfig.register(MODEL_TYPE, DeepseekV41Config, exist_ok=True)
AutoConfig.register(TEXT_MODEL_TYPE, DeepseekV41TextConfig, exist_ok=True)


def register_with_vllm() -> None:
    """Make vLLM's ``get_config`` build ``DeepseekV41Config`` for this model_type.

    vLLM consults its own ``_CONFIG_REGISTRY`` before ``AutoConfig``. The registry is a
    ``LazyConfigDict`` of class *names*; a class object is accepted as well.
    """
    from vllm.transformers_utils import config as vllm_config

    vllm_config._CONFIG_REGISTRY[MODEL_TYPE] = DeepseekV41Config
    AutoConfig.register(MODEL_TYPE, DeepseekV41Config, exist_ok=True)
    AutoConfig.register(TEXT_MODEL_TYPE, DeepseekV41TextConfig, exist_ok=True)


# --------------------------------------------------------------------- typed view
# HF text_config key -> DeepSeek reference (ModelArgs) key, for scalar fields that map
# one to one. Nested / derived fields are handled in ``from_hf``.
_HF_TO_REF = {
    "vocab_size": "vocab_size",
    "hidden_size": "dim",
    "moe_intermediate_size": "moe_inter_dim",
    "num_hidden_layers": "n_layers",
    "num_nextn_predict_layers": "n_mtp_layers",
    "num_attention_heads": "n_heads",
    "n_routed_experts": "n_routed_experts",
    "n_shared_experts": "n_shared_experts",
    "num_experts_per_tok": "n_activated_experts",
    "scoring_func": "score_func",
    "norm_topk_prob": "norm_topk_prob",
    "routed_scaling_factor": "route_scale",
    "swiglu_limit": "swiglu_limit",
    "q_lora_rank": "q_lora_rank",
    "head_dim": "head_dim",
    "qk_rope_head_dim": "rope_head_dim",
    "rms_norm_eps": "norm_eps",
    "o_groups": "o_groups",
    "o_lora_rank": "o_lora_rank",
    "sliding_window": "window_size",
    "compress_ratios": "compress_ratios",
    "kv_source_layer_ids": "kv_source_layers",
    "index_source_layer_ids": "index_source_layers",
    "compress_rope_theta": "compress_rope_theta",
    "rope_theta": "rope_theta",
    "index_n_heads": "index_n_heads",
    "index_head_dim": "index_head_dim",
    "index_topk": "index_topk",
    "candidate_source_layer_id": "candidate_source_layer",
    "candidate_topk_blocks": "candidate_topk_blocks",
    "candidate_block_size": "candidate_block_size",
    "hc_mult": "hc_mult",
    "hc_sinkhorn_iters": "hc_sinkhorn_iters",
    "hc_eps": "hc_eps",
    "engram_layer_ids": "engram_layer_ids",
    "engram_num_embeddings": "engram_num_embeddings",
    "engram_max_ngram_size": "engram_max_ngram_size",
    "engram_vocab_size": "engram_vocab_size",
    "engram_n_heads": "engram_n_heads",
    "engram_head_dim": "engram_head_dim",
    "engram_pad_token_id": "engram_pad_id",
    "engram_compressed_vocab_size": "engram_compressed_vocab_size",
    "dspark_block_size": "dspark_block_size",
    "dspark_noise_token_id": "dspark_noise_token_id",
    "dspark_target_layer_ids": "dspark_target_layer_ids",
    "dspark_markov_rank": "dspark_markov_rank",
    "dspark_n_routed_experts": "dspark_n_routed_experts",
    "dspark_num_experts_per_tok": "dspark_n_activated_experts",
}
# rope_scaling (YaRN) -> reference keys
_ROPE_TO_REF = {
    "factor": "rope_factor",
    "beta_fast": "beta_fast",
    "beta_slow": "beta_slow",
    "original_max_position_embeddings": "original_seq_len",
}


def _get(cfg, key):
    return cfg.get(key) if isinstance(cfg, dict) else getattr(cfg, key, None)


@dataclass(frozen=True)
class DeepseekV41TextArgs:
    """The text decoder's constants, in DeepSeek reference names. Every field is
    required: a defaulted constant that happens to equal the real one makes an unwired
    call site indistinguishable from a wired one."""

    vocab_size: int
    dim: int
    moe_inter_dim: int
    n_layers: int
    n_mtp_layers: int
    n_heads: int
    n_routed_experts: int
    n_shared_experts: int
    n_activated_experts: int
    score_func: str
    norm_topk_prob: bool
    route_scale: float
    swiglu_limit: float
    q_lora_rank: int
    head_dim: int
    rope_head_dim: int
    norm_eps: float
    o_groups: int
    o_lora_rank: int
    window_size: int
    compress_ratios: tuple[int, ...]
    kv_source_layers: tuple[int, ...]
    index_source_layers: tuple[int, ...]
    compress_rope_theta: float
    rope_theta: float
    rope_factor: float
    beta_fast: int
    beta_slow: int
    original_seq_len: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    candidate_source_layer: int
    candidate_topk_blocks: int
    candidate_block_size: int
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    engram_layer_ids: tuple[int, ...]
    engram_num_embeddings: tuple[int, ...]
    engram_max_ngram_size: int
    engram_vocab_size: int
    engram_n_heads: int
    engram_head_dim: int
    engram_pad_id: int
    engram_compressed_vocab_size: int
    dspark_block_size: int
    dspark_noise_token_id: int
    dspark_target_layer_ids: tuple[int, ...]
    dspark_markov_rank: int
    dspark_n_routed_experts: int
    dspark_n_activated_experts: int
    # quantization as stored (the plugin dequantizes at load; see weights.py)
    dtype: str
    expert_dtype: str | None
    weight_block_size: tuple[int, int] | None
    max_position_embeddings: int

    @classmethod
    def from_hf(cls, text_cfg, quantization_config=None) -> "DeepseekV41TextArgs":
        """From the HF ``text_config`` (dict or config object) and the top-level
        ``quantization_config`` (``None`` for an already-dequantized checkpoint).

        Refuses what the plugin does not implement rather than approximating it.
        """
        out = {}
        for hf_key, ref_key in _HF_TO_REF.items():
            value = _get(text_cfg, hf_key)
            if value is None:
                raise ValueError(f"DeepSeek-V4.1 text_config is missing {hf_key!r}")
            out[ref_key] = tuple(value) if isinstance(value, list) else value
        rope = _get(text_cfg, "rope_scaling") or _get(text_cfg, "rope_parameters")
        if not rope or rope.get("rope_type", rope.get("type")) != "yarn":
            raise NotImplementedError(f"rope_scaling {rope!r}: only YaRN is implemented")
        for rope_key, ref_key in _ROPE_TO_REF.items():
            if rope.get(rope_key) is None:
                raise ValueError(f"rope_scaling is missing {rope_key!r}")
            out[ref_key] = rope[rope_key]
        if out["score_func"] != "sqrtsoftplus":
            raise NotImplementedError(f"scoring_func={out['score_func']!r}")
        topk_method = _get(text_cfg, "topk_method")
        if topk_method not in (None, "noaux_tc"):
            raise NotImplementedError(f"topk_method={topk_method!r}")
        if _get(text_cfg, "hidden_act") not in (None, "silu"):
            raise NotImplementedError(f"hidden_act={_get(text_cfg, 'hidden_act')!r}")
        if _get(text_cfg, "num_key_value_heads") not in (None, 1):
            raise NotImplementedError("DeepSeek-V4.1 attention is MQA with one KV head")
        n_total = out["n_layers"] + out["n_mtp_layers"]
        if len(out["compress_ratios"]) != n_total:
            raise ValueError(f"compress_ratios has {len(out['compress_ratios'])} entries, "
                             f"expected n_layers + n_mtp_layers = {n_total}")
        q = quantization_config or {}
        if q and q.get("quant_method") != "fp8":
            raise NotImplementedError(f"quant_method={q.get('quant_method')!r}")
        out["dtype"] = "fp8" if q else "bf16"
        out["expert_dtype"] = q.get("expert_dtype") if q else None
        wbs = q.get("weight_block_size") if q else None
        out["weight_block_size"] = tuple(wbs) if wbs else None
        out["max_position_embeddings"] = _get(text_cfg, "max_position_embeddings")
        return cls(**out)

    @classmethod
    def from_hf_config(cls, hf_config) -> "DeepseekV41TextArgs":
        """From the top-level HF config (object or dict)."""
        text = _get(hf_config, "text_config")
        quant = (_get(hf_config, QUANT_KEY) or _get(text, QUANT_KEY)
                 or _get(hf_config, "original_quantization_config"))   # make_served_dir
        return cls.from_hf(text, quant)

    def to_reference_args(self) -> dict:
        """Keys and values as DeepSeek's inference ``config.json`` spells them."""
        out = {f.name: getattr(self, f.name) for f in fields(self)
               if f.name not in ("weight_block_size", "max_position_embeddings")}
        return {k: list(v) if isinstance(v, tuple) else v for k, v in out.items()}

    @property
    def n_total_layers(self) -> int:
        return self.n_layers + self.n_mtp_layers


# ------------------------------------------------------------------ served config
def converted_config(hf_dir: str | Path) -> dict:
    """The checkpoint's ``config.json`` with ``quantization_config`` removed at every
    level. vLLM's ``ModelConfig`` refuses an fp8-advertising config in CPU mode
    (``"fp8 quantization is currently not supported in cpu"``) before any plugin code
    runs; the plugin reads the stored formats from the tensors themselves."""
    cfg = json.loads((Path(hf_dir) / "config.json").read_text())
    cfg.pop(QUANT_KEY, None)
    for sub in ("text_config", "vision_config"):
        if isinstance(cfg.get(sub), dict):
            cfg[sub].pop(QUANT_KEY, None)
    assert not any(QUANT_KEY in (c or {}) for c in (cfg, cfg.get("text_config"),
                                                     cfg.get("vision_config")))
    return cfg


def make_served_dir(hf_dir: str | Path, out_dir: str | Path) -> Path:
    """A directory vLLM can serve the checkpoint from: ``converted_config`` as
    ``config.json`` (with the original quantization recorded under
    ``original_quantization_config`` for the weight loader) and every other file
    symlinked. Nothing is copied -- the shards are ~510 GB."""
    hf_dir, out_dir = Path(hf_dir).resolve(), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    original = json.loads((hf_dir / "config.json").read_text()).get(QUANT_KEY)
    cfg = converted_config(hf_dir)
    if original is not None:
        cfg["original_quantization_config"] = original
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    for src in hf_dir.iterdir():
        if src.name == "config.json":
            continue
        dst = out_dir / src.name
        if not dst.exists():
            dst.symlink_to(src)
    return out_dir


__all__ = ["DeepseekV41Config", "DeepseekV41TextConfig", "DeepseekV41TextArgs",
           "register_with_vllm", "converted_config", "make_served_dir",
           "ARCHITECTURE", "MODEL_TYPE", "TEXT_MODEL_TYPE"]
