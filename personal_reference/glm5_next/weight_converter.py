# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash FP8 checkpoint -> the oracle's state_dict.

Maps ``zai-org/GLM-5.3-Flash`` (76,108 tensors, 328.3 GB, FP8 e4m3 block-scaled)
onto ``reference.py``'s module names, dequantizing FP8 to BF16 on the way.

Design decisions worth knowing before changing anything here:

**Nothing is passed through silently.** Every tensor name is matched against an
explicit rule and is either mapped, or dropped with a recorded reason, or raises
``ConversionError``. A converter that forwards names it does not recognise is how
a subtly wrong checkpoint ships — it succeeds, and the error surfaces later as
"quality is a bit off".

**FP8-ness is read from the checkpoint, never guessed from the name.** A tensor is
FP8 iff a sibling ``<name>_scale_inv`` exists in the index. Do not reintroduce a
name-pattern allowlist: ``self_attn.o_proj`` is FP8 on the sparse-MLA layers and
BF16 on the KDA layers, so the same name needs both answers. Reading
``quantization_config.modules_to_not_convert`` instead is also a trap — it has
1509 entries, most of them per-layer, and collapsing them to patterns produces
exactly the wrong conclusion.

Verified against the live index and safetensors headers on 2026-09-25:

* prefix ``model.language_model.``; ``lm_head.weight`` is top level.
* Four layer signatures: layers 0-2 KDA + dense MLP (29 kinds); layers
  3,7,...,43 sparse-MLA + MoE (40 kinds); the remaining 31 KDA + MoE (37 kinds);
  layer 45 MTP (38 kinds). 288 experts, indices 0-287.
* FP8 tensors are ``F8_E4M3`` with an ``F32`` ``_scale_inv`` of shape
  ``ceil(dim / 128)`` per axis — checked exactly, e.g. weight (16384, 1536) ->
  scale (128, 12).
* FP8: all MLP/expert/shared-expert weights (including the three dense layers),
  plus ``q_a_proj``, ``q_b_proj``, ``kv_a_proj_with_mqa`` and ``o_proj`` on the 11
  sparse-MLA layers and the MTP layer.
* BF16: everything in the 34 KDA layers, **``kv_b_proj``** (note the asymmetry
  with ``q_b_proj``), the whole indexer (7 tensors per sparse-MLA layer, mapped
  under their own names), every norm, every ``hc_*``,
  ``embed_tokens``, ``lm_head``, ``mlp.gate.weight``.
  ``mlp.gate.e_score_correction_bias`` is F32.

Three structural regroupings between checkpoint and oracle:

1. ``{q,k,v}_conv1d`` are three depthwise convs; the oracle has one over
   ``cat(q, k, v)``, so they concatenate along the channel axis in q, k, v order.
2. ``self_attn.{f_a_proj,f_b_proj,dt_bias,A_log}`` are flat in the checkpoint and
   nested under ``self_attn.forget_gate`` in the oracle.
3. 288 separate ``experts.{e}.{gate,up,down}_proj`` become two stacked
   parameters: ``mlp.gate_up_proj`` of shape [E, 2I, D] with gate rows first, and
   ``mlp.down_proj`` of shape [E, D, I].
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

PREFIX = "model.language_model."
BLOCK = 128
NUM_EXPERTS = 288
MTP_LAYER = 45


class ConversionError(Exception):
    """A checkpoint tensor matched no rule. Never swallow this."""


# Reasons a recognised tensor is deliberately not loaded.
DROP_VISION = "vision tower (scope is text-only)"
DROP_MTP = f"MTP draft layer {MTP_LAYER} (not wired in the oracle)"


@dataclass
class Plan:
    """What to do with every tensor in the checkpoint."""

    simple: dict[str, str] = field(default_factory=dict)          # target <- source
    conv: dict[str, dict[str, str]] = field(default_factory=dict)  # target <- {q,k,v: source}
    expert_gate_up: dict[str, dict[int, dict[str, str]]] = field(default_factory=dict)
    expert_down: dict[str, dict[int, str]] = field(default_factory=dict)
    scales: dict[str, str] = field(default_factory=dict)          # weight source -> scale source
    dropped: dict[str, str] = field(default_factory=dict)         # source -> reason

    @property
    def targets(self) -> set[str]:
        return (set(self.simple) | set(self.conv)
                | set(self.expert_gate_up) | set(self.expert_down))


def is_scale(name: str) -> bool:
    return name.endswith("_scale_inv")


def drop_reason(name: str) -> str | None:
    """Why this tensor is recognised but not loaded, or None if it should be."""
    if name.startswith("model.visual") or name.startswith("visual."):
        return DROP_VISION
    if re.search(rf"\.layers\.{MTP_LAYER}\.", name):
        return DROP_MTP
    return None


# (pattern, target template). Applied to the de-prefixed name. Order matters only
# in that the conv1d and expert patterns are handled separately below.
_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^embed_tokens\.weight$"), "embed_tokens.weight"),
    (re.compile(r"^norm\.weight$"), "norm.weight"),
    (re.compile(r"^(layers\.\d+)\.(input_layernorm|post_attention_layernorm)\.weight$"),
     r"\1.\2.weight"),
    # mHC: hc_attn_* -> attn_hc.*, hc_ffn_* -> ffn_hc.*
    (re.compile(r"^(layers\.\d+)\.hc_attn_(fn|base|scale)$"), r"\1.attn_hc.\2"),
    (re.compile(r"^(layers\.\d+)\.hc_ffn_(fn|base|scale)$"), r"\1.ffn_hc.\2"),
    # KDA forget gate: flat in the checkpoint, nested in the oracle
    (re.compile(r"^(layers\.\d+)\.self_attn\.(f_a_proj|f_b_proj)\.weight$"),
     r"\1.self_attn.forget_gate.\2.weight"),
    (re.compile(r"^(layers\.\d+)\.self_attn\.(dt_bias|A_log)$"),
     r"\1.self_attn.forget_gate.\2"),
    # KDA, rest
    (re.compile(r"^(layers\.\d+)\.self_attn\.(q_proj|k_proj|v_proj|b_proj|g_a_proj|g_b_proj|o_proj)\.weight$"),
     r"\1.self_attn.\2.weight"),
    (re.compile(r"^(layers\.\d+)\.self_attn\.o_norm\.weight$"), r"\1.self_attn.o_norm.weight"),
    # sparse-MLA
    (re.compile(r"^(layers\.\d+)\.self_attn\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj)\.weight$"),
     r"\1.self_attn.\2.weight"),
    (re.compile(r"^(layers\.\d+)\.self_attn\.(q_a_layernorm|kv_a_layernorm)\.weight$"),
     r"\1.self_attn.\2.weight"),
    # DSA indexer: the oracle keeps the checkpoint's names, so this is the identity
    (re.compile(r"^(layers\.\d+)\.self_attn\.indexer\.(wq_b|wk|weights_proj)\.weight$"),
     r"\1.self_attn.indexer.\2.weight"),
    (re.compile(r"^(layers\.\d+)\.self_attn\.indexer\.k_norm\.(weight|bias)$"),
     r"\1.self_attn.indexer.k_norm.\2"),
    (re.compile(r"^(layers\.\d+)\.self_attn\.indexer\.(index_kpool_compress_ape|index_kpool_compress_gate)$"),
     r"\1.self_attn.indexer.\2"),
    # dense MLP (layers 0-2)
    (re.compile(r"^(layers\.\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$"),
     r"\1.mlp.\2.weight"),
    # MoE router and shared expert
    (re.compile(r"^(layers\.\d+)\.mlp\.gate\.weight$"), r"\1.mlp.gate.weight"),
    (re.compile(r"^(layers\.\d+)\.mlp\.gate\.e_score_correction_bias$"),
     r"\1.mlp.gate.e_score_correction_bias"),
    (re.compile(r"^(layers\.\d+)\.mlp\.shared_experts\.(gate_proj|up_proj|down_proj)\.weight$"),
     r"\1.mlp.shared_experts.\2.weight"),
]

_CONV = re.compile(r"^(layers\.\d+)\.self_attn\.([qkv])_conv1d\.weight$")
_EXPERT = re.compile(r"^(layers\.\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


def strip_prefix(name: str) -> str:
    return name[len(PREFIX):] if name.startswith(PREFIX) else name


def convert_name(name: str) -> str | None:
    """Oracle-side name for one checkpoint tensor, or None if dropped.

    Raises ``ConversionError`` on anything unrecognised. Expert tensors return the
    *stacked* target they contribute to; conv1d tensors return the shared conv
    target. Use ``plan_from_index`` when you need the slot detail.
    """
    if is_scale(name):
        base = convert_name(name[: -len("_scale_inv")])
        return None if base is None else base
    if (why := drop_reason(name)) is not None:
        return None if why else None
    n = strip_prefix(name)
    if name == "lm_head.weight":
        return "lm_head.weight"
    if (m := _CONV.match(n)) is not None:
        return f"{m.group(1)}.self_attn.conv1d.weight"
    if (m := _EXPERT.match(n)) is not None:
        which = m.group(3)
        return f"{m.group(1)}.mlp." + ("down_proj" if which == "down_proj" else "gate_up_proj")
    for pat, repl in _RULES:
        if pat.match(n):
            return pat.sub(repl, n)
    raise ConversionError(
        f"unrecognised checkpoint tensor {name!r}. Add an explicit rule or an "
        f"explicit drop reason — do not let it pass through unmapped."
    )


def plan_from_index(weight_map: dict[str, str]) -> Plan:
    """Build the full conversion plan from ``model.safetensors.index.json``.

    Needs no weight data, so the entire mapping is testable against all 76,108
    real tensor names offline.
    """
    plan = Plan()
    for name in sorted(weight_map):
        if (why := drop_reason(name)) is not None:
            plan.dropped[name] = why
            continue
        if is_scale(name):
            plan.scales[name[: -len("_scale_inv")]] = name
            continue
        n = strip_prefix(name)
        if name == "lm_head.weight":
            plan.simple["lm_head.weight"] = name
            continue
        if (m := _CONV.match(n)) is not None:
            tgt = f"{m.group(1)}.self_attn.conv1d.weight"
            plan.conv.setdefault(tgt, {})[m.group(2)] = name
            continue
        if (m := _EXPERT.match(n)) is not None:
            layer, e, which = m.group(1), int(m.group(2)), m.group(3)
            if which == "down_proj":
                plan.expert_down.setdefault(f"{layer}.mlp.down_proj", {})[e] = name
            else:
                slot = plan.expert_gate_up.setdefault(f"{layer}.mlp.gate_up_proj", {})
                slot.setdefault(e, {})[which] = name
            continue
        matched = False
        for pat, repl in _RULES:
            if pat.match(n):
                tgt = pat.sub(repl, n)
                if tgt in plan.simple:
                    raise ConversionError(
                        f"two checkpoint tensors map to {tgt!r}: "
                        f"{plan.simple[tgt]!r} and {name!r}"
                    )
                plan.simple[tgt] = name
                matched = True
                break
        if not matched:
            raise ConversionError(
                f"unrecognised checkpoint tensor {name!r}. Add an explicit rule "
                f"or an explicit drop reason — do not pass it through unmapped."
            )
    _validate(plan)
    return plan


def _validate(plan: Plan) -> None:
    """Structural checks that must hold before any weight is read."""
    for tgt, parts in plan.conv.items():
        if set(parts) != {"q", "k", "v"}:
            raise ConversionError(f"{tgt}: expected q/k/v conv parts, got {sorted(parts)}")
    for tgt, experts in plan.expert_gate_up.items():
        if sorted(experts) != list(range(NUM_EXPERTS)):
            raise ConversionError(
                f"{tgt}: expected experts 0-{NUM_EXPERTS - 1}, got {len(experts)}"
            )
        for e, halves in experts.items():
            if set(halves) != {"gate_proj", "up_proj"}:
                raise ConversionError(f"{tgt}[{e}]: expected gate+up, got {sorted(halves)}")
    for tgt, experts in plan.expert_down.items():
        if sorted(experts) != list(range(NUM_EXPERTS)):
            raise ConversionError(
                f"{tgt}: expected experts 0-{NUM_EXPERTS - 1}, got {len(experts)}"
            )
    # A scale whose weight was dropped is fine; a scale with no weight at all is not.
    known = set(plan.scales)
    sources = set(plan.simple.values())
    for parts in plan.conv.values():
        sources |= set(parts.values())
    for experts in plan.expert_gate_up.values():
        for halves in experts.values():
            sources |= set(halves.values())
    for experts in plan.expert_down.values():
        sources |= set(experts.values())
    orphan = known - sources - set(plan.dropped)
    if orphan:
        raise ConversionError(f"{len(orphan)} scale(s) with no matching weight: {sorted(orphan)[:5]}")


def expected_scale_shape(weight_shape: tuple[int, ...], block: int = BLOCK) -> tuple[int, ...]:
    """Block-scale shape for an FP8 weight: ceil(dim / block) on the last two axes."""
    return tuple(math.ceil(d / block) for d in weight_shape)


def dequant_block_fp8(w, scale_inv, block: int = BLOCK):
    """DeepSeek-V3 style blockwise dequant: ``w[i, j] * scale_inv[i // 128, j // 128]``.

    ``w`` is FP8 e4m3, ``scale_inv`` F32 of shape ``ceil(dim / block)`` per axis.
    Returns BF16. The expand is done with ``repeat_interleave`` then truncated,
    which handles a final partial block correctly.
    """
    import torch

    if scale_inv.shape != expected_scale_shape(tuple(w.shape), block):
        raise ConversionError(
            f"scale shape {tuple(scale_inv.shape)} does not match weight "
            f"{tuple(w.shape)} at block {block}; expected "
            f"{expected_scale_shape(tuple(w.shape), block)}"
        )
    wf = w.to(torch.float32)
    r, c = wf.shape[-2], wf.shape[-1]
    s = scale_inv.to(torch.float32)
    s = s.repeat_interleave(block, -2)[..., :r, :].repeat_interleave(block, -1)[..., :c]
    return (wf * s).to(torch.bfloat16)


def build_state_dict(hf_dir: str | Path, plan: Plan | None = None) -> dict:
    """Materialise the oracle state_dict, dequantizing FP8 to BF16.

    Reads shards with safetensors. Host RAM is roughly the BF16 size (~640 GB for
    the full model), so this is for a large host or a sliced-down plan, not a
    laptop. Every FP8 tensor is dequantized: the oracle is pure BF16 torch.
    """
    import torch
    from safetensors import safe_open

    hf_dir = Path(hf_dir)
    weight_map = json.loads((hf_dir / "model.safetensors.index.json").read_text())["weight_map"]
    plan = plan or plan_from_index(weight_map)
    cache: dict[str, object] = {}

    def get(name: str):
        if name not in cache:
            with safe_open(str(hf_dir / weight_map[name]), "pt") as f:
                cache[name] = f.get_tensor(name)
        return cache[name]

    def weight(name: str):
        w = get(name)
        scale = plan.scales.get(name)
        return dequant_block_fp8(w, get(scale)) if scale else w.to(torch.bfloat16)

    out: dict[str, object] = {}
    for tgt, src in plan.simple.items():
        # e_score_correction_bias is F32 in the checkpoint and read in F32 by the
        # router; keep it rather than rounding to BF16.
        out[tgt] = get(src) if tgt.endswith("e_score_correction_bias") else weight(src)
    for tgt, parts in plan.conv.items():
        out[tgt] = torch.cat([weight(parts[c]) for c in ("q", "k", "v")], dim=0)
    for tgt, experts in plan.expert_gate_up.items():
        out[tgt] = torch.stack([
            torch.cat([weight(experts[e]["gate_proj"]), weight(experts[e]["up_proj"])], dim=0)
            for e in range(NUM_EXPERTS)
        ])
    for tgt, experts in plan.expert_down.items():
        out[tgt] = torch.stack([weight(experts[e]) for e in range(NUM_EXPERTS)])
    return out


if __name__ == "__main__":
    import sys

    wm = json.loads(Path(sys.argv[1]).read_text())["weight_map"]
    p = plan_from_index(wm)
    print(f"checkpoint tensors : {len(wm)}")
    print(f"targets            : {len(p.targets)}")
    print(f"  simple           : {len(p.simple)}")
    print(f"  conv1d concat    : {len(p.conv)} (x3 sources)")
    print(f"  expert gate_up   : {len(p.expert_gate_up)} (x{NUM_EXPERTS} x2)")
    print(f"  expert down      : {len(p.expert_down)} (x{NUM_EXPERTS})")
    print(f"FP8 (has scale)    : {len(p.scales)}")
    print(f"dropped            : {len(p.dropped)}")
    for reason in sorted(set(p.dropped.values())):
        print(f"  {sum(1 for r in p.dropped.values() if r == reason):6d}  {reason}")
