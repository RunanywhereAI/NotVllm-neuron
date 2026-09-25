# SPDX-License-Identifier: Apache-2.0
"""Every layer must pick its mixer and its FFN from the config, index for index.

Run: python3 -m pytest personal_reference/glm5_next/tests/test_layer_dispatch.py -v

GLM-5.3-Flash is `[linear_attention ×3, deepseek_sparse_attention] ×11` plus a
trailing linear layer, with the first `first_k_dense_replace` FFNs dense and the
rest MoE. Getting either sequence wrong produces a model that **runs and fails
nothing**: every shape still matches, every kernel still executes, and the output
is plausible and wrong.

## What this replaces

Before this file, `layer_types` appeared **zero times** in the whole test
directory — nothing asserted that any layer picked the right attention kind. The
FFN side was checked, but only by sampling two indices
(`test_mlp_moe.py::test_both_mlp_call_sites_carry_the_limit`, which pins index 0
as dense and index 3 as MoE).

Sampling cannot pin a sequence, and the gap is not hypothetical. With
`mlp_layer_types = [dense, dense, dense, sparse, sparse, …]`, shift the list
**left by one**:

| index | correct | shifted | sampled? |
|---|---|---|---|
| 0 | dense | dense | ✓ checked — passes |
| 1 | dense | dense | not checked |
| 2 | dense | **sparse** | not checked — **wrong** |
| 3 | sparse | sparse | ✓ checked — passes |

Both sampled assertions hold while layer 2 is wrong.
`test_left_shift_defeats_sampling_but_not_this_file` builds exactly that mutation
and asserts the old approach passes and this one does not — the mutation is the
proof that these tests work, not a description of a weakness.

Asserting the **full sequence** is the whole point; do not reduce these to spot
checks.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from glm5_next import reference as R  # noqa: E402

LINEAR = "linear_attention"
SPARSE_MLA = "deepseek_sparse_attention"
DENSE = "dense"
MOE = "sparse"


def _small(**kw):
    """Tiny but structurally complete; layer count is what these tests vary."""
    base = dict(hidden_size=64, linear_num_heads=2, linear_head_dim=32,
                num_attention_heads=2, q_lora_rank=32, kv_lora_rank=32,
                qk_nope_head_dim=32, v_head_dim=32, n_routed_experts=4,
                num_experts_per_tok=2, moe_intermediate_size=32,
                intermediate_size=64, index_n_heads=2, index_head_dim=16,
                index_topk=16)
    base.update(kw)
    return R.tiny_cfg(**base)


def _attention_mismatches(cfg) -> list[tuple[int, str, str]]:
    """(index, expected layer_type, actual class) for every layer that disagrees."""
    bad = []
    for i, want in enumerate(cfg.layer_types):
        layer = R.DecoderLayer(cfg, i)
        got = type(layer.self_attn).__name__
        ok = (isinstance(layer.self_attn, R.LinearAttention) if want == LINEAR
              else isinstance(layer.self_attn, R.SparseMLAttention))
        if not ok:
            bad.append((i, want, got))
    return bad


def _mlp_mismatches(cfg) -> list[tuple[int, str, str]]:
    bad = []
    for i, want in enumerate(cfg.mlp_layer_types):
        layer = R.DecoderLayer(cfg, i)
        got = type(layer.mlp).__name__
        ok = (isinstance(layer.mlp, R.MoE) if want == MOE
              else isinstance(layer.mlp, R.MLP) and not isinstance(layer.mlp, R.MoE))
        if not ok:
            bad.append((i, want, got))
    return bad


# ------------------------------------------------------- the config sequences
def test_layer_types_sequence_matches_the_live_config():
    """45 layers: [linear ×3, sparse-MLA] ×11 + linear. Sparse-MLA at 3, 7, …, 43."""
    cfg = R.FlashCfg()
    assert cfg.num_hidden_layers == 45
    expected = [LINEAR if i % 4 != 3 else SPARSE_MLA for i in range(45)]
    assert list(cfg.layer_types) == expected
    assert [i for i, t in enumerate(cfg.layer_types) if t != LINEAR] == list(range(3, 45, 4))
    assert sum(1 for t in cfg.layer_types if t == LINEAR) == 34
    assert sum(1 for t in cfg.layer_types if t == SPARSE_MLA) == 11


def test_mlp_layer_types_sequence_matches_first_k_dense_replace():
    cfg = R.FlashCfg()
    assert cfg.first_k_dense_replace == 3
    assert list(cfg.mlp_layer_types) == [DENSE] * 3 + [MOE] * 42


# ----------------------------------------------- dispatch, index for index
@pytest.mark.parametrize("n_layers", [8, 12, 16])
def test_attention_dispatch_matches_layer_types_index_for_index(n_layers):
    """Every layer, not a sample. Nothing asserted this before this file existed."""
    cfg = _small(num_hidden_layers=n_layers)
    bad = _attention_mismatches(cfg)
    assert not bad, f"attention dispatch disagrees with layer_types at {bad}"
    # guard against a vacuous pass: the config must actually contain both kinds
    kinds = set(cfg.layer_types)
    assert kinds == {LINEAR, SPARSE_MLA}, f"only {kinds} present; test proves nothing"


@pytest.mark.parametrize("n_layers", [8, 12, 16])
def test_mlp_dispatch_matches_mlp_layer_types_index_for_index(n_layers):
    cfg = _small(num_hidden_layers=n_layers)
    bad = _mlp_mismatches(cfg)
    assert not bad, f"MLP dispatch disagrees with mlp_layer_types at {bad}"
    assert set(cfg.mlp_layer_types) == {DENSE, MOE}


# --------------------------------------------------------- the mutation proof
def test_left_shift_defeats_sampling_but_not_this_file():
    """The specific mutation that the previous coverage could not see.

    Shifting ``mlp_layer_types`` left by one leaves index 0 dense and index 3 MoE
    — the two indices the old test sampled — while layer 2 becomes MoE when it
    should be dense. This asserts both halves: the sampled checks still pass, and
    the index-for-index check does not.
    """
    cfg = _small(num_hidden_layers=8)
    correct = list(cfg.mlp_layer_types)
    shifted = correct[1:] + [correct[-1]]
    assert shifted != correct and shifted[0] == correct[0] and shifted[3] == correct[3], (
        "the mutation no longer has the property that makes this test meaningful"
    )

    cfg.mlp_layer_types = shifted

    # 1. the old sampling approach — indices 0 and 3 — still passes
    assert isinstance(R.DecoderLayer(cfg, 0).mlp, R.MLP)
    assert isinstance(R.DecoderLayer(cfg, 3).mlp, R.MoE)

    # 2. index-for-index against the TRUE sequence catches it
    cfg_true = _small(num_hidden_layers=8)
    bad = [i for i, want in enumerate(cfg_true.mlp_layer_types)
           if (want == MOE) != isinstance(R.DecoderLayer(cfg, i).mlp, R.MoE)]
    assert bad == [2], f"expected layer 2 to be the detectable break, got {bad}"


def test_attention_left_shift_is_caught():
    """Same mutation on layer_types. Sparse-MLA moves from 3 to 2."""
    cfg = _small(num_hidden_layers=8)
    correct = list(cfg.layer_types)
    cfg.layer_types = correct[1:] + [correct[-1]]

    cfg_true = _small(num_hidden_layers=8)
    bad = [i for i, want in enumerate(cfg_true.layer_types)
           if (want == LINEAR) != isinstance(R.DecoderLayer(cfg, i).self_attn, R.LinearAttention)]
    assert bad, "a shifted layer_types produced no dispatch disagreement at all"
    assert 2 in bad and 3 in bad, f"expected the break around the boundary, got {bad}"


def test_inverted_dispatch_is_caught():
    """The coarsest possible error: every layer gets the wrong mixer."""
    cfg = _small(num_hidden_layers=8)
    cfg.layer_types = [SPARSE_MLA if t == LINEAR else LINEAR for t in cfg.layer_types]
    cfg_true = _small(num_hidden_layers=8)
    bad = [i for i, want in enumerate(cfg_true.layer_types)
           if (want == LINEAR) != isinstance(R.DecoderLayer(cfg, i).self_attn, R.LinearAttention)]
    assert len(bad) == cfg.num_hidden_layers, f"only {len(bad)} of 8 layers flagged"


# ------------------------------------------------- the model builds them in order
def test_model_layers_follow_the_config_in_order():
    """FlashTextModel must build DecoderLayer(cfg, i) for ascending i, not reuse one."""
    torch.manual_seed(0)
    cfg = _small(num_hidden_layers=8)
    model = R.FlashTextModel(cfg, vocab=64)
    assert len(model.layers) == cfg.num_hidden_layers
    for i, layer in enumerate(model.layers):
        want_linear = cfg.layer_types[i] == LINEAR
        assert isinstance(layer.self_attn, R.LinearAttention) is want_linear, (
            f"model.layers[{i}] has {type(layer.self_attn).__name__} but "
            f"layer_types[{i}] is {cfg.layer_types[i]}"
        )
        assert isinstance(layer.mlp, R.MoE) is (cfg.mlp_layer_types[i] == MOE)
    # distinct module instances, not one layer repeated
    assert len({id(l) for l in model.layers}) == cfg.num_hidden_layers
