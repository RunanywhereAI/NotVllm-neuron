# SPDX-License-Identifier: Apache-2.0
"""Weight converter invariants.

Run: python3 -m pytest personal_reference/glm5_next/tests/test_weight_converter.py -v

Carries the seven invariants from the NxDI fork's
``contrib/models/GLM-5.3-Flash/test/unit/test_weight_converter.py`` and extends
them. The extensions that matter:

* **Nothing passes through unmapped.** An unknown name must raise.
* **The q_b_proj / kv_b_proj asymmetry.** ``q_b_proj`` is FP8, ``kv_b_proj`` is
  BF16. Symmetry is the natural assumption and it is wrong.
* **``o_proj`` is FP8 on sparse-MLA layers and BF16 on KDA layers** — the same
  name needs both answers, which is why FP8-ness must come from the presence of a
  sibling ``_scale_inv`` and never from a name pattern.
* **Expert stacking**: 288 x 3 tensors per MoE layer collapse to two parameters.

The synthetic index below is built from the four layer signatures verified against
the live checkpoint on 2026-09-25, and ``test_synthetic_index_reproduces_real_totals``
asserts it reproduces the real tensor count exactly (76,108). That keeps these
tests hermetic while still pinning them to reality. Set
``GLM53F_INDEX=/path/to/model.safetensors.index.json`` to additionally run the
whole plan against the real index.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from glm5_next import weight_converter as WC  # noqa: E402
from glm5_next.weight_converter import ConversionError, PREFIX  # noqa: E402

E = WC.NUM_EXPERTS
MLA_LAYERS = [i for i in range(45) if i % 4 == 3]
KDA_LAYERS = [i for i in range(45) if i % 4 != 3]
DENSE_MLP_LAYERS = [0, 1, 2]

COMMON = ["hc_attn_base", "hc_attn_fn", "hc_attn_scale",
          "hc_ffn_base", "hc_ffn_fn", "hc_ffn_scale",
          "input_layernorm.weight", "post_attention_layernorm.weight"]

KDA_ATTN = ["self_attn.A_log", "self_attn.b_proj.weight", "self_attn.dt_bias",
            "self_attn.f_a_proj.weight", "self_attn.f_b_proj.weight",
            "self_attn.g_a_proj.weight", "self_attn.g_b_proj.weight",
            "self_attn.k_conv1d.weight", "self_attn.k_proj.weight",
            "self_attn.o_norm.weight", "self_attn.o_proj.weight",
            "self_attn.q_conv1d.weight", "self_attn.q_proj.weight",
            "self_attn.v_conv1d.weight", "self_attn.v_proj.weight"]

MLA_ATTN = ["self_attn.indexer.index_kpool_compress_ape",
            "self_attn.indexer.index_kpool_compress_gate",
            "self_attn.indexer.k_norm.bias", "self_attn.indexer.k_norm.weight",
            "self_attn.indexer.weights_proj.weight", "self_attn.indexer.wk.weight",
            "self_attn.indexer.wq_b.weight",
            "self_attn.kv_a_layernorm.weight", "self_attn.kv_a_proj_with_mqa.weight",
            "self_attn.kv_b_proj.weight", "self_attn.o_proj.weight",
            "self_attn.q_a_layernorm.weight", "self_attn.q_a_proj.weight",
            "self_attn.q_b_proj.weight"]
# The only FP8 attention tensors, and only on MLA/MTP layers.
MLA_FP8 = ["self_attn.kv_a_proj_with_mqa.weight", "self_attn.o_proj.weight",
           "self_attn.q_a_proj.weight", "self_attn.q_b_proj.weight"]

DENSE_MLP = ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]
MOE = ["mlp.gate.weight", "mlp.gate.e_score_correction_bias",
       "mlp.shared_experts.gate_proj.weight", "mlp.shared_experts.up_proj.weight",
       "mlp.shared_experts.down_proj.weight"]
MOE_FP8 = ["mlp.shared_experts.gate_proj.weight", "mlp.shared_experts.up_proj.weight",
           "mlp.shared_experts.down_proj.weight"]
MTP_EXTRA = ["eh_proj.weight", "enorm.weight", "hnorm.weight", "shared_head.norm.weight",
             "input_layernorm.weight", "post_attention_layernorm.weight"]


def _synthetic_weight_map() -> dict[str, str]:
    """All 76,108 checkpoint tensor names, from the verified layer signatures."""
    names: list[str] = ["lm_head.weight", PREFIX + "embed_tokens.weight", PREFIX + "norm.weight"]

    def add(layer: int, kinds: list[str], fp8: list[str]) -> None:
        for k in kinds:
            names.append(f"{PREFIX}layers.{layer}.{k}")
            if k in fp8:
                names.append(f"{PREFIX}layers.{layer}.{k}_scale_inv")

    def add_experts(layer: int) -> None:
        for e in range(E):
            for which in ("gate_proj", "up_proj", "down_proj"):
                base = f"{PREFIX}layers.{layer}.mlp.experts.{e}.{which}.weight"
                names.extend([base, base + "_scale_inv"])

    for i in range(45):
        mla = i in MLA_LAYERS
        add(i, COMMON, [])
        add(i, MLA_ATTN if mla else KDA_ATTN, MLA_FP8 if mla else [])
        if i in DENSE_MLP_LAYERS:
            add(i, DENSE_MLP, DENSE_MLP)
        else:
            add(i, MOE, MOE_FP8)
            add_experts(i)
    # MTP layer 45: sparse-MLA + MoE, no mHC, plus its own projections
    add(45, MTP_EXTRA, [])
    add(45, MLA_ATTN, MLA_FP8)
    add(45, MOE, MOE_FP8)
    add_experts(45)
    # vision tower
    names += [f"model.visual.blocks.{i}.stub{j}.weight" for i in range(25) for j in range(13)]
    names += [f"model.visual.extra{j}.weight" for j in range(22)]
    return {n: "model-00001-of-00062.safetensors" for n in names}


@pytest.fixture(scope="module")
def wm():
    return _synthetic_weight_map()


@pytest.fixture(scope="module")
def plan(wm):
    return WC.plan_from_index(wm)


def test_synthetic_index_reproduces_real_totals(wm):
    """The structural model must account for every tensor in the real checkpoint.

    76,108 keys and 37,338 ``weight_scale_inv`` were measured from the live index.
    If this drifts, the signatures below no longer describe the checkpoint and
    every other test here is testing fiction.
    """
    assert len(wm) == 76108, f"synthetic index has {len(wm)} names, real has 76108"
    assert sum(1 for k in wm if k.endswith("_scale_inv")) == 37338
    assert sum(1 for k in wm if k.startswith("model.visual")) == 347


# ------------------------------------------------- the seven carried invariants
def test_vision_and_mtp_dropped(plan):
    assert all(".visual." in k or f".layers.{WC.MTP_LAYER}." in k for k in plan.dropped)
    assert any(".visual." in k for k in plan.dropped)
    assert any(f".layers.{WC.MTP_LAYER}." in k for k in plan.dropped)
    assert plan.dropped[f"{PREFIX}layers.45.enorm.weight"] == WC.DROP_MTP


def test_no_language_prefix_survives(plan):
    assert not any(t.startswith(PREFIX) for t in plan.targets)
    assert not any(t.startswith("model.") for t in plan.targets)


def test_conv1d_collapses_to_one_param(plan):
    assert len(plan.conv) == len(KDA_LAYERS) == 34
    for li in (0, 1, 2, 4, 44):
        tgt = f"layers.{li}.self_attn.conv1d.weight"
        assert set(plan.conv[tgt]) == {"q", "k", "v"}
        for c in "qkv":
            assert plan.conv[tgt][c] == f"{PREFIX}layers.{li}.self_attn.{c}_conv1d.weight"


def test_forget_gate_nested():
    assert WC.convert_name(f"{PREFIX}layers.0.self_attn.A_log") == \
        "layers.0.self_attn.forget_gate.A_log"
    assert WC.convert_name(f"{PREFIX}layers.0.self_attn.dt_bias") == \
        "layers.0.self_attn.forget_gate.dt_bias"
    assert WC.convert_name(f"{PREFIX}layers.0.self_attn.f_a_proj.weight") == \
        "layers.0.self_attn.forget_gate.f_a_proj.weight"
    assert WC.convert_name(f"{PREFIX}layers.0.self_attn.f_b_proj.weight") == \
        "layers.0.self_attn.forget_gate.f_b_proj.weight"


def test_mhc_renamed():
    assert WC.convert_name(f"{PREFIX}layers.3.hc_ffn_scale") == "layers.3.ffn_hc.scale"
    assert WC.convert_name(f"{PREFIX}layers.3.hc_attn_fn") == "layers.3.attn_hc.fn"
    assert WC.convert_name(f"{PREFIX}layers.3.hc_attn_base") == "layers.3.attn_hc.base"


def test_sparse_layers_are_every_fourth(plan):
    sparse = sorted(int(k.split(".")[3]) for k in plan.simple.values()
                    if "kv_b_proj" in k and f".layers.{WC.MTP_LAYER}." not in k)
    assert sparse == MLA_LAYERS == [i for i in range(45) if i % 4 == 3]


def test_mapped_names_unique(plan):
    """plan_from_index raises on a collision, so reaching here proves uniqueness."""
    all_targets = list(plan.simple) + list(plan.conv) + \
        list(plan.expert_gate_up) + list(plan.expert_down)
    assert len(all_targets) == len(set(all_targets))


# ------------------------------------------------------------ fail loudly
def test_unrecognised_tensor_raises():
    """A silent passthrough is how a subtly wrong checkpoint ships."""
    with pytest.raises(ConversionError, match="unrecognised"):
        WC.convert_name(f"{PREFIX}layers.0.self_attn.mystery_proj.weight")
    with pytest.raises(ConversionError, match="unrecognised"):
        WC.convert_name(f"{PREFIX}layers.0.brand_new_thing")


def test_unrecognised_tensor_in_index_raises(wm):
    bad = dict(wm)
    bad[f"{PREFIX}layers.0.self_attn.surprise.weight"] = "shard"
    with pytest.raises(ConversionError, match="unrecognised"):
        WC.plan_from_index(bad)


def test_incomplete_conv_triplet_raises(wm):
    bad = {k: v for k, v in wm.items()
           if k != f"{PREFIX}layers.0.self_attn.v_conv1d.weight"}
    with pytest.raises(ConversionError, match="expected q/k/v"):
        WC.plan_from_index(bad)


def test_missing_expert_raises(wm):
    bad = {k: v for k, v in wm.items()
           if not k.startswith(f"{PREFIX}layers.4.mlp.experts.287.")}
    with pytest.raises(ConversionError, match=f"expected experts 0-{E - 1}"):
        WC.plan_from_index(bad)


# --------------------------------------------------- FP8 layout, incl. asymmetry
def test_q_b_proj_is_fp8_but_kv_b_proj_is_not(plan):
    """The asymmetry a symmetry assumption gets wrong.

    Real shapes: q_b_proj F8_E4M3 (16384, 1536) with an F32 (128, 12) scale;
    kv_b_proj BF16 (32768, 512) with no scale at all.
    """
    for li in MLA_LAYERS:
        q_b = f"{PREFIX}layers.{li}.self_attn.q_b_proj.weight"
        kv_b = f"{PREFIX}layers.{li}.self_attn.kv_b_proj.weight"
        assert q_b in plan.scales, f"layer {li}: q_b_proj should be FP8"
        assert kv_b not in plan.scales, f"layer {li}: kv_b_proj should be BF16"


def test_o_proj_is_fp8_on_mla_layers_and_bf16_on_kda_layers(plan):
    """One name, two answers — so FP8-ness cannot come from a name pattern."""
    for li in MLA_LAYERS:
        assert f"{PREFIX}layers.{li}.self_attn.o_proj.weight" in plan.scales
    for li in KDA_LAYERS:
        assert f"{PREFIX}layers.{li}.self_attn.o_proj.weight" not in plan.scales


def test_kda_attention_is_entirely_bf16(plan):
    """No KDA-layer tensor carries a scale — the whole linear-attention stack is BF16."""
    for li in KDA_LAYERS:
        for kind in KDA_ATTN:
            assert f"{PREFIX}layers.{li}.{kind}" not in plan.scales


def test_indexer_and_hc_are_bf16_and_indexer_is_mapped(plan):
    """The oracle runs the real DSA indexer, so its 7 tensors x 11 layers are loaded
    under their own names; before it did, they were dropped."""
    n = 0
    for li in MLA_LAYERS:
        for kind in MLA_ATTN:
            if ".indexer." in kind:
                src = f"{PREFIX}layers.{li}.{kind}"
                assert src not in plan.scales
                assert src not in plan.dropped
                assert plan.simple[f"layers.{li}.{kind}"] == src
                n += 1
        for kind in ("hc_attn_fn", "hc_ffn_fn", "hc_attn_scale"):
            assert f"{PREFIX}layers.{li}.{kind}" not in plan.scales
    assert n == 77


def test_expert_and_mlp_weights_are_all_fp8(plan):
    for li in DENSE_MLP_LAYERS:
        for kind in DENSE_MLP:
            assert f"{PREFIX}layers.{li}.{kind}" in plan.scales
    for li in (4, 44):
        for kind in MOE_FP8:
            assert f"{PREFIX}layers.{li}.{kind}" in plan.scales
        for e in (0, E - 1):
            for which in ("gate_proj", "up_proj", "down_proj"):
                assert f"{PREFIX}layers.{li}.mlp.experts.{e}.{which}.weight" in plan.scales
    # mlp.gate (the router) is never quantized
    assert f"{PREFIX}layers.4.mlp.gate.weight" not in plan.scales


def test_embeddings_and_lm_head_are_bf16(plan):
    for k in ("lm_head.weight", PREFIX + "embed_tokens.weight", PREFIX + "norm.weight"):
        assert k not in plan.scales


# ------------------------------------------------------------ expert stacking
def test_experts_stack_into_two_parameters(plan):
    """288 x 3 tensors per MoE layer become gate_up_proj and down_proj."""
    moe_layers = [i for i in range(45) if i not in DENSE_MLP_LAYERS]
    assert len(plan.expert_gate_up) == len(plan.expert_down) == len(moe_layers) == 42
    gu = plan.expert_gate_up["layers.4.mlp.gate_up_proj"]
    assert sorted(gu) == list(range(E))
    assert set(gu[0]) == {"gate_proj", "up_proj"}
    assert sorted(plan.expert_down["layers.4.mlp.down_proj"]) == list(range(E))


def test_targets_are_exactly_the_oracles_parameters(plan):
    """Every oracle parameter is filled and no target is invented: the plan's targets
    equal the real-config FlashTextModel's state_dict keys. Built on the meta device,
    so no memory is allocated."""
    from glm5_next import reference as R
    with torch.device("meta"):
        keys = set(R.FlashTextModel(R.FlashCfg()).state_dict())
    assert not keys - plan.targets, f"oracle parameters with no source: {sorted(keys - plan.targets)[:5]}"
    assert not plan.targets - keys, f"targets the oracle does not have: {sorted(plan.targets - keys)[:5]}"
    assert len(keys) == 1262


def test_plan_totals_reconcile(wm, plan):
    """1144 simple + 34x3 conv + 42x288x2 gate_up + 42x288 down + 36467 scales + 2107 dropped."""
    assert (len(plan.simple), len(plan.conv), len(plan.scales), len(plan.dropped)) == (1144, 34, 36467, 2107)
    assert 1144 + 34 * 3 + 42 * 288 * 2 + 42 * 288 + 36467 + 2107 == len(wm) == 76108


def test_every_source_is_accounted_for_exactly_once(wm, plan):
    """No tensor may be silently lost or used twice."""
    seen: list[str] = list(plan.simple.values()) + list(plan.scales.values()) \
        + list(plan.dropped)
    for parts in plan.conv.values():
        seen += list(parts.values())
    for experts in plan.expert_gate_up.values():
        for halves in experts.values():
            seen += list(halves.values())
    for experts in plan.expert_down.values():
        seen += list(experts.values())
    assert len(seen) == len(set(seen)), "a source tensor was consumed twice"
    assert set(seen) == set(wm), (
        f"{len(set(wm) - set(seen))} sources unaccounted for, "
        f"{len(set(seen) - set(wm))} invented"
    )


# ------------------------------------------------------------ FP8 dequantization
def test_expected_scale_shape():
    """Measured against real headers: (16384, 1536) -> (128, 12)."""
    assert WC.expected_scale_shape((16384, 1536)) == (128, 12)
    assert WC.expected_scale_shape((2048, 4096)) == (16, 32)
    assert WC.expected_scale_shape((4096, 2048)) == (32, 16)
    assert WC.expected_scale_shape((12288, 4096)) == (96, 32)
    # partial trailing block must round up
    assert WC.expected_scale_shape((129, 127)) == (2, 1)


def test_dequant_applies_the_right_block_to_each_element():
    """Element [i, j] must be scaled by scale[i // 128, j // 128]."""
    w = torch.ones(256, 384)
    scale = torch.arange(2 * 3, dtype=torch.float32).reshape(2, 3) + 1.0
    got = WC.dequant_block_fp8(w, scale).float()
    for bi in range(2):
        for bj in range(3):
            block = got[bi * 128:(bi + 1) * 128, bj * 128:(bj + 1) * 128]
            assert torch.allclose(block, scale[bi, bj].expand_as(block)), (bi, bj)


def test_dequant_handles_a_partial_trailing_block():
    w = torch.ones(130, 129)
    scale = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    got = WC.dequant_block_fp8(w, scale).float()
    assert got.shape == (130, 129)
    assert got[0, 0] == 1.0 and got[0, 128] == 2.0
    assert got[129, 0] == 3.0 and got[129, 128] == 4.0


def test_dequant_rejects_a_mismatched_scale():
    with pytest.raises(ConversionError, match="does not match"):
        WC.dequant_block_fp8(torch.ones(256, 384), torch.ones(2, 2))


def test_dequant_round_trips_real_fp8_dtype():
    """Exercise the actual e4m3 dtype, not a float stand-in."""
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch build without float8_e4m3fn")
    w = torch.randn(256, 256).to(torch.float8_e4m3fn)
    scale = torch.full((2, 2), 2.0)
    got = WC.dequant_block_fp8(w, scale)
    assert got.dtype == torch.bfloat16
    expect = (w.to(torch.float32) * 2.0).to(torch.bfloat16)
    torch.testing.assert_close(got, expect)


# ------------------------------------------------------- optional: real index
@pytest.mark.skipif(not os.environ.get("GLM53F_INDEX"),
                    reason="set GLM53F_INDEX to the real model.safetensors.index.json")
def test_plan_against_real_index():
    wm = json.loads(pathlib.Path(os.environ["GLM53F_INDEX"]).read_text())["weight_map"]
    p = WC.plan_from_index(wm)
    assert len(wm) == 76108
    assert len(p.scales) == 36467          # 37,338 total minus the MTP layer's 871
    assert len(p.dropped) == 2107          # 1760 MTP + 347 vision; the 77 indexer tensors are now mapped
    assert len(p.simple) == 1144 and len(p.targets) == 1262
    assert len(p.conv) == 34
    assert len(p.expert_gate_up) == len(p.expert_down) == 42
