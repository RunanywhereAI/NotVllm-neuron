# SPDX-License-Identifier: Apache-2.0
"""Tensor parallelism: N in-process ranks over gloo must reproduce the TP=1 model.

Run: python3 -m pytest personal_reference/glm5_next/tests/test_plugin_tp.py -v -s

Each rank is its own process: ``torch.distributed`` over gloo, the model built with a
``GroupCoordinator``-shaped TP group, ``load_weights`` reading only that rank's shards of
a tiny checkpoint through the plugin's own ``SafetensorsCheckpoint``, then prefill and
batched decode through ``FakeRunner`` with every collective real. Inputs are
teacher-forced on the oracle's greedy tokens, so every degree sees the same sequence.

The chain of evidence: TP=1 matches the oracle (``test_plugin_model.py``, and the
reference check here); TP=2 and TP=4 match TP=1. A sharded run can match for the wrong
reason -- a parameter accidentally replicated would too -- so each of four parameters
(the q/k/v conv's segment shard, an MLA head-sharded projection, an expert's gate/up
shard, a per-head decay) is loaded from the WRONG rank's slice on rank 1, at the right
shape, and the comparison must fail.

Needs the released ``config.json`` (for the checkpoint's top-level structure) in the
HF cache or at ``GLM53F_BF16_CONFIG``; skips otherwise.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from glm5_next.tests import plugin_harness as H  # noqa: E402
from glm5_next.tier2 import make_tiny  # noqa: E402

FOUR_HEADS = {"num_attention_heads": 4,
              "linear_attn_config": {"num_heads": 4, "head_dim": 16,
                                     "short_conv_kernel_size": 4, "gate_lower_bound": -5.0}}


def _real_config():
    env = os.environ.get("GLM53F_BF16_CONFIG")
    if env and pathlib.Path(env).is_file():
        return pathlib.Path(env)
    cached = sorted(pathlib.Path.home().glob(
        ".cache/huggingface/hub/models--zai-org--GLM-5.3-Flash-BF16/snapshots/*/config.json"))
    if not cached:
        pytest.skip("released GLM-5.3-Flash config.json not available")
    return cached[0]


@pytest.fixture(scope="module")
def ckpts(tmp_path_factory):
    cfg = _real_config()
    two = tmp_path_factory.mktemp("tiny2")
    four = tmp_path_factory.mktemp("tiny4")
    make_tiny.main(two, cfg)
    make_tiny.main(four, cfg, FOUR_HEADS)
    return {2: two, 4: four}


_CACHE: dict = {}


def _run(world, ckpt, tmp_path, sabotage=None):
    key = (world, str(ckpt), sabotage)
    if key not in _CACHE:
        _CACHE[key] = H.run_tp(world, ckpt, tmp_path / f"tp{world}_{sabotage}.pt", sabotage)
    return _CACHE[key]


def _max_rel(a, b):
    """Worst relative error over every prefill and decode logit row (live rows only)."""
    worst = 0.0
    for x, y in zip(a["prefill"], b["prefill"]):
        worst = max(worst, ((x - y).abs().max() / y.abs().max()).item())
    for x, y in zip(a["decode"], b["decode"]):
        n = x.shape[0] - 1                          # last row is padding
        worst = max(worst, ((x[:n] - y[:n]).abs().max() / y[:n].abs().max()).item())
    return worst


def test_tp1_reproduces_the_oracle_reference(ckpts, tmp_path):
    """Ties this file's TP=1 baseline to the oracle: the log-probability of each
    teacher-forced greedy token matches ``oracle_reference.json``."""
    for heads, ckpt in ckpts.items():
        res = _run(1, ckpt, tmp_path)
        ref = json.loads((ckpt / "oracle_reference.json").read_text())
        worst = 0.0
        for r, p in enumerate(ref["prompts"]):
            rows = [res["prefill"][r]] + [d[r] for d in res["decode"]]
            for logits, tok, want in zip(rows, p["greedy"], p["logprob"]):
                lp = torch.log_softmax(logits, -1)
                assert int(lp.argmax()) == tok
                worst = max(worst, abs(float(lp[tok]) - want))
        print(f"\n  {heads}-head checkpoint, TP=1 vs oracle: worst |dlogprob| {worst:.2e}")
        assert worst < 1e-4


@pytest.mark.parametrize("world,heads", [(2, 2), (2, 4), (4, 4)])
def test_tp_matches_tp1(ckpts, tmp_path, world, heads):
    base = _run(1, ckpts[heads], tmp_path)
    got = _run(world, ckpts[heads], tmp_path)
    err = _max_rel(got, base)
    print(f"\n  TP={world} vs TP=1 ({heads} heads): max rel {err:.2e}")
    assert err < 1e-4


@pytest.mark.parametrize("sabotage", sorted(H.SABOTAGE))
def test_a_wrong_shard_is_caught(ckpts, tmp_path, sabotage):
    """Rank 1 loads one parameter from rank 0's slice -- right shape, wrong values."""
    base = _run(1, ckpts[2], tmp_path)
    got = _run(2, ckpts[2], tmp_path, sabotage)
    err = _max_rel(got, base)
    print(f"\n  {sabotage}: TP=2 with one wrong shard, max rel {err:.2e}")
    assert err > 1e-3, f"a wrong {sabotage} shard went unnoticed"
    assert all(torch.isfinite(t).all() for t in got["decode"])
