# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 (``model_type: glm_moe_dsa``) -> the fields the Neuron model reads.

transformers >= 5.15 ships ``GlmMoeDsaConfig`` and vLLM 0.24 already knows the
architecture, so no config class is registered here: this is only a typed view that
accepts the transformers config object or the raw ``config.json`` dict.

Two fields are *derived*, and transformers derives them in ``__post_init__``; a raw dict
does not have them, so the same derivation is repeated here (and checked against the
stored list when one is present):

* ``indexer_types``: ``"full"`` layers run the DSA indexer, ``"shared"`` layers reuse the
  previous full layer's top-k. From ``index_topk_pattern`` or
  ``index_topk_freq`` / ``index_skip_topk_offset``.
* ``mlp_layer_types``: the first ``first_k_dense_replace`` layers are dense.

Served dir: ``make_served_dir`` writes ``config.json`` without ``quantization_config`` (vLLM
refuses an fp8 config here before plugin code runs), recorded as
``original_quantization_config``, and symlinks every other file.

``head_dim`` in ``config.json`` (192) is NOT the rotary width: transformers overwrites
it with ``qk_rope_head_dim`` (64). The model reads ``qk_rope_head_dim`` only.
"""

from __future__ import annotations

import dataclasses

ARCHITECTURE = "GlmMoeDsaForCausalLM"
MODEL_TYPE = "glm_moe_dsa"


def _get(cfg, key, default=dataclasses.MISSING):
    if isinstance(cfg, dict):
        if key in cfg:
            return cfg[key]
    elif hasattr(cfg, key):
        return getattr(cfg, key)
    if default is dataclasses.MISSING:
        raise KeyError(f"config has no {key!r}")
    return default


def derive_indexer_types(n_layers: int, pattern=None, freq: int = 1, offset: int = 2) -> tuple[str, ...]:
    """transformers ``GlmMoeDsaConfig.__post_init__``, verbatim in effect."""
    if pattern is not None:
        if isinstance(pattern, str):
            return tuple({"F": "full", "S": "shared"}[c] for c in pattern)
        return tuple(pattern)
    freq = max(freq, 1)
    return tuple("full" if (max(i - offset + 1, 0) % freq) == 0 else "shared" for i in range(n_layers))


def make_served_dir(hf_dir, out_dir):
    import json
    from pathlib import Path

    hf_dir, out_dir = Path(hf_dir).resolve(), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((hf_dir / "config.json").read_text())
    q = cfg.pop("quantization_config", None)
    if q is not None:
        cfg["original_quantization_config"] = q
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    for src in hf_dir.iterdir():
        dst = out_dir / src.name
        if src.name != "config.json" and not dst.exists():
            dst.symlink_to(src)
    return out_dir


@dataclasses.dataclass(frozen=True)
class GlmMoeDsaArgs:
    vocab_size: int
    dim: int
    inter_dim: int
    moe_inter_dim: int
    n_layers: int
    n_heads: int
    n_routed_experts: int
    n_shared_experts: int
    n_activated_experts: int
    route_scale: float
    norm_topk_prob: bool
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    rope_theta: float
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    indexer_types: tuple
    mlp_layer_types: tuple
    norm_eps: float
    fp8_block: tuple | None      # (128, 128) for the released checkpoint; None = unquantized

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def latent_dim(self) -> int:
        """One cached row per token per layer: the normed latent, then the RoPE'd key."""
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def index_layers(self) -> tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.indexer_types) if t == "full")

    @classmethod
    def from_hf(cls, cfg, n_layers: int | None = None) -> "GlmMoeDsaArgs":
        """``n_layers`` truncates the decoder (compile probes); every per-layer list is cut
        to match."""
        L = _get(cfg, "num_hidden_layers")
        rope = _get(cfg, "rope_parameters", None) or _get(cfg, "rope_scaling", None) or {}
        rope_type = rope.get("rope_type", rope.get("type", "default"))
        if rope_type != "default":
            raise NotImplementedError(f"rope_type {rope_type!r}: only default RoPE is implemented")
        theta = rope.get("rope_theta", _get(cfg, "rope_theta", None))
        if _get(cfg, "scoring_func", "sigmoid") != "sigmoid":
            raise NotImplementedError("only sigmoid routing is implemented")
        if _get(cfg, "n_group", 1) != 1 or _get(cfg, "topk_group", 1) != 1:
            raise NotImplementedError("group-limited routing (n_group > 1) is not implemented")
        if not _get(cfg, "rope_interleave", True) or not _get(cfg, "indexer_rope_interleave", True):
            raise NotImplementedError("only interleaved RoPE (rope_interleave=True) is implemented")
        if _get(cfg, "attention_bias", False):
            raise NotImplementedError("attention_bias")
        if _get(cfg, "q_lora_rank", None) is None:
            raise NotImplementedError("q_lora_rank=None")
        it = _get(cfg, "indexer_types", None)
        derived = derive_indexer_types(L, _get(cfg, "index_topk_pattern", None),
                                       _get(cfg, "index_topk_freq", 1),
                                       _get(cfg, "index_skip_topk_offset", 2))
        if it is not None and tuple(it) != derived:
            # the stored list wins (it is what transformers and vLLM read), but say so
            import logging
            logging.getLogger(__name__).warning("indexer_types differs from the freq/offset schedule")
        it = tuple(it) if it is not None else derived
        mt = _get(cfg, "mlp_layer_types", None)
        if mt is None:
            k = min(_get(cfg, "first_k_dense_replace", 1), L)
            mt = ("dense",) * k + ("sparse",) * (L - k)
        mt = tuple(mt)
        if it[0] != "full":
            raise ValueError("layer 0 must own an indexer (a shared layer reuses a previous one)")
        # a served dir strips quantization_config (vLLM refuses fp8 configs on this
        # platform before any plugin code runs) and keeps it as original_quantization_config
        q = _get(cfg, "quantization_config", None) or _get(cfg, "original_quantization_config", None)
        block = None
        if q:
            qd = q if isinstance(q, dict) else q.to_dict()
            if qd.get("quant_method") != "fp8" or qd.get("fmt", "e4m3") != "e4m3":
                raise NotImplementedError(f"quantization {qd.get('quant_method')}/{qd.get('fmt')}")
            block = tuple(qd["weight_block_size"])
        n = L if n_layers is None else min(n_layers, L)
        return cls(
            vocab_size=_get(cfg, "vocab_size"),
            dim=_get(cfg, "hidden_size"),
            inter_dim=_get(cfg, "intermediate_size"),
            moe_inter_dim=_get(cfg, "moe_intermediate_size"),
            n_layers=n,
            n_heads=_get(cfg, "num_attention_heads"),
            n_routed_experts=_get(cfg, "n_routed_experts"),
            n_shared_experts=_get(cfg, "n_shared_experts"),
            n_activated_experts=_get(cfg, "num_experts_per_tok"),
            route_scale=float(_get(cfg, "routed_scaling_factor")),
            norm_topk_prob=bool(_get(cfg, "norm_topk_prob")),
            q_lora_rank=_get(cfg, "q_lora_rank"),
            kv_lora_rank=_get(cfg, "kv_lora_rank"),
            qk_nope_head_dim=_get(cfg, "qk_nope_head_dim"),
            qk_rope_head_dim=_get(cfg, "qk_rope_head_dim"),
            v_head_dim=_get(cfg, "v_head_dim"),
            rope_theta=float(theta),
            index_n_heads=_get(cfg, "index_n_heads"),
            index_head_dim=_get(cfg, "index_head_dim"),
            index_topk=_get(cfg, "index_topk"),
            indexer_types=it[:n],
            mlp_layer_types=mt[:n],
            norm_eps=float(_get(cfg, "rms_norm_eps")),
            fp8_block=block,
        )
