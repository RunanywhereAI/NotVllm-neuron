# SPDX-License-Identifier: Apache-2.0
"""Is running the 11 sparse-MLA layers dense exact, and up to what length?

Run: python3 -m pytest personal_reference/glm5_next/tests/test_dense_mla_exactness.py -v

**Answer: yes, and the ceiling is seq_len <= index_topk + index_kpool - 1 == 2051.**
Above that, the dense path is not approximate. It is *wrong*: it attends to tokens
the model's indexer would have excluded.

Two numbers, and they are both right
------------------------------------
* **2051 is the true ceiling**: the largest seq_len at which the sparse path still
  selects every token.
* **2048 is vLLM's shortcut gate** (``seq_len <= topk_tokens``, with
  ``topk_tokens = index_topk``). Below it vLLM skips scoring and fills the top-k
  buffer with every causal index. The gate is correct but not tight: from 2049 to
  2051 vLLM runs the sparse path, and that path also selects every token.

Do not "fix" either number to match the other. They answer different questions.

Why they differ: floor plus tail, not ceil
------------------------------------------
For a query that sees ``L`` tokens, the indexer scores only the **complete** pools,
``L // index_kpool`` of them (vLLM: ``ke = (start_pos + 1 + offset) //
COMPRESS_RATIO`` in prefill, ``seq_lens //= compress_ratio`` in decode;
transformers: a pool is a candidate only if its final token is visible). It
selects ``index_topk // index_kpool`` of them and expands each back to its
tokens. ``index_kpool_always_select_tail`` then appends the ``L % index_kpool``
tokens of the incomplete pool as raw indices. So::

    covered(L) = min(L // kpool, topk // kpool) * kpool + L % kpool

which equals ``L`` exactly when ``L // kpool <= topk // kpool``, i.e.
``L <= topk + kpool - 1``.

An earlier version of this file reasoned with ``ceil(L / kpool)`` pools, which
treats the incomplete pool as a scoring candidate. It is not one, since it is
always covered by the tail, so that version put the ceiling at 2048 and claimed
tokens were excluded at 2049. That was an error in modelling the implementation,
not in the arithmetic.

What else was checked
---------------------
The risk was never the arithmetic. It was the meaning of ``index_kpool_compress``:
if attention ran over *compressed pool representatives*, selecting every pool
would still be lossy. It does not. vLLM keeps three separate caches
(`vllm/models/glm5next/common/attention.py`):

* ``Glm5NextIndexerCache``: kpool-compressed entries, used **only to score**.
* ``Glm5NextTailCache``: the trailing incomplete pool's **raw K** plus gate score,
  one block of ``index_kpool`` slots per request, overwritten by ``pos % kpool``.
* the ordinary MLA latent KV cache: full fidelity, and what attention gathers.

The indexer's output is ``topk_indices_buffer``, which holds **token** indices.
Compression never reaches attention. transformers 5.17's plain-torch
``Glm5NextTextIndexer`` agrees on all of the above, and it has no short-sequence
shortcut: it always runs the sparse path.

The tests below pin each dependency, so a config change trips one of them:

1. ``index_topk == 2048`` and ``index_kpool == 4``.
2. ``index_topk % index_kpool == 0``. vLLM asserts it
   (``history_group_budget_for_topk``), and ``topk + kpool - 1`` relies on it.
3. ``index_kpool_compress`` stays confined to the indexer's scoring cache.
4. ``index_kpool_always_select_tail`` keeps appending raw tail tokens. Without it
   the ceiling drops to ``topk`` and every ``L`` not divisible by ``kpool`` loses
   its newest tokens.
5. ``indexer_types`` is ``"full"`` on all 45 layers.

A real indexer in the oracle has to reduce to the dense path at every length up
to 2051. This file pins the arithmetic behind that requirement.
"""
from __future__ import annotations

import json
import os
import pathlib

import pytest

# Values verified against the live zai-org/GLM-5.3-Flash config.json on 2026-09-25.
INDEX_TOPK = 2048
INDEX_KPOOL = 4
NUM_LAYERS = 45
DENSE_EXACT_MAX_SEQ_LEN = INDEX_TOPK + INDEX_KPOOL - 1    # 2051, the true ceiling
VLLM_SHORTCUT_MAX_SEQ_LEN = INDEX_TOPK                    # 2048, vLLM's gate


def _live_config():
    """The real config, if a copy is available; otherwise the verified constants."""
    path = os.environ.get("GLM53F_CONFIG")
    if path:
        return json.loads(pathlib.Path(path).read_text())["text_config"]
    return {
        "index_topk": INDEX_TOPK,
        "index_kpool": INDEX_KPOOL,
        "index_kpool_compress": True,
        "index_kpool_always_select_tail": True,
        "indexer_types": ["full"] * NUM_LAYERS,
        "num_hidden_layers": NUM_LAYERS,
    }


def covered(L: int, topk: int, kpool: int, tail: bool = True) -> int:
    """Tokens a query that sees L tokens attends to: selected complete pools + tail."""
    pools = min(L // kpool, topk // kpool) * kpool
    return pools + (L % kpool if tail else 0)


def test_constants_match_config():
    cfg = _live_config()
    assert cfg["index_topk"] == INDEX_TOPK
    assert cfg["index_kpool"] == INDEX_KPOOL
    assert VLLM_SHORTCUT_MAX_SEQ_LEN == cfg["index_topk"]


def test_index_topk_divides_by_index_kpool():
    cfg = _live_config()
    topk, kpool = cfg["index_topk"], cfg["index_kpool"]
    assert topk % kpool == 0, (
        f"index_topk {topk} no longer divides by index_kpool {kpool}; vLLM asserts "
        f"this and the topk + kpool - 1 ceiling depends on it. Re-derive."
    )


def test_true_ceiling_is_topk_plus_kpool_minus_one():
    """Scan every L: the largest L with covered(L) == L is 2051, not 2048."""
    cfg = _live_config()
    topk, kpool = cfg["index_topk"], cfg["index_kpool"]
    exact = [L for L in range(1, 3 * topk) if covered(L, topk, kpool) == L]
    assert exact == list(range(1, DENSE_EXACT_MAX_SEQ_LEN + 1))
    assert max(exact) == topk + kpool - 1 == 2051


@pytest.mark.parametrize("L", [1, 2, 3, 4, 5, 64, 2047, 2048, 2049, 2050, 2051])
def test_every_token_selected_at_or_below_the_ceiling(L):
    cfg = _live_config()
    assert covered(L, cfg["index_topk"], cfg["index_kpool"]) == L


@pytest.mark.parametrize("L", [2049, 2050, 2051])
def test_vllm_gate_is_conservative_not_tight(L):
    """Above vLLM's 2048 gate it runs the sparse path, which still selects everything."""
    cfg = _live_config()
    assert L > VLLM_SHORTCUT_MAX_SEQ_LEN
    assert covered(L, cfg["index_topk"], cfg["index_kpool"]) == L


@pytest.mark.parametrize("L", [2052, 2053, 4096, 1 << 20])
def test_above_the_ceiling_the_dense_path_is_wrong_not_approximate(L):
    cfg = _live_config()
    topk, kpool = cfg["index_topk"], cfg["index_kpool"]
    assert covered(L, topk, kpool) < L
    # the first lossy length drops exactly one whole pool
    if L == DENSE_EXACT_MAX_SEQ_LEN + 1:
        assert L - covered(L, topk, kpool) == kpool


def test_the_ceil_derivation_is_the_wrong_model():
    """Recorded so the old argument cannot quietly come back.

    Counting ceil(L / kpool) pools puts the ceiling at 2048 and says L=2049 loses
    a pool. It does not: pool 512 is incomplete at L=2049, so it is never scored,
    and its one token comes back as tail.
    """
    cfg = _live_config()
    topk, kpool = cfg["index_topk"], cfg["index_kpool"]
    ceil_pools = -(-2049 // kpool)
    assert ceil_pools > topk // kpool           # what the ceil model concludes
    assert covered(2049, topk, kpool) == 2049   # what the implementation does


def test_the_tail_is_what_moves_the_ceiling():
    """Without always_select_tail the ceiling is topk and non-aligned L lose tokens."""
    cfg = _live_config()
    topk, kpool = cfg["index_topk"], cfg["index_kpool"]
    assert covered(2051, topk, kpool, tail=False) == 2048
    assert covered(7, topk, kpool, tail=False) == 4


def test_compress_and_tail_flags_are_as_assumed():
    """The argument depends on what these two flags mean, not just their value.

    ``index_kpool_compress`` must remain confined to the indexer's scoring cache
    (vLLM: ``Glm5NextIndexerCache``), and ``index_kpool_always_select_tail`` must
    append raw tail tokens (vLLM: ``Glm5NextTailCache`` stores "raw bf16 K ... not
    the fp8-compressed entry"). If either changes, exactness has to be re-argued.
    """
    cfg = _live_config()
    assert cfg["index_kpool_compress"] is True
    assert cfg["index_kpool_always_select_tail"] is True


def test_all_layers_use_the_full_indexer():
    """A "shared" layer reuses the previous layer's selection (transformers
    ``skip_topk``), which the oracle does not implement."""
    cfg = _live_config()
    types = cfg["indexer_types"]
    assert len(types) == cfg["num_hidden_layers"] == NUM_LAYERS
    assert set(types) == {"full"}, f"unexpected indexer types: {sorted(set(types))}"


def test_dense_cannot_cover_the_million_token_target():
    """Milestone 2 wants 1M context; skipping the indexer is exact only to 2051."""
    assert DENSE_EXACT_MAX_SEQ_LEN < (1 << 20)
