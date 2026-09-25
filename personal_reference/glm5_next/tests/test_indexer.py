# SPDX-License-Identifier: Apache-2.0
"""The DSA indexer: cross-referenced, continuous with dense, and able to fail.

Run: python3 -m pytest personal_reference/glm5_next/tests/test_indexer.py -v -s
(``-s`` prints the mutant-margin table.)

The bug most likely to ship here is an indexer that degenerates to dense. Up to
2051 tokens, dense is *exact* (``test_dense_mla_exactness.py``), so every check
at or below that length passes whether the indexer works or not. Everything that
has to be able to fail therefore runs **above** the ceiling.

What this file establishes:

1. **Two independent references agree with the oracle, row for row.**
   transformers 5.17's ``Glm5NextTextIndexer`` and vLLM's prefill path (both
   vendored in ``indexer_refs.py``) select the same token set as the oracle on
   every row. This uses the real indexer geometry (32 heads x 128, topk 2048,
   kpool 4) at S = 2052 / 2101 / 2562 / 3003, which covers every S % 4, with
   hundreds of rows actually dropping tokens.
2. **Continuity.** At S <= 2051 the sparse path, with no shortcut, selects
   exactly the causal set, and the layer output is bit-identical to the dense
   path. At 2052 exactly one pool drops out of exactly one row.
3. **Discrimination.** Eleven wrong indexers — select-all, random, bottom-k,
   recency window, off-by-one expansion, mean-pool, no ReLU, unweighted heads,
   scoring the incomplete pool, no tail, force-include-self — are each rejected
   above the ceiling. Their output shift is reported as a multiple of the
   agreement floor.
4. **A planted needle** is selected from far back and a planted dud is dropped
   from the most recent pool. A recency window fails both.
5. **Self-exclusion is correct.** Above the ceiling with ``L % 4 == 0``, a query
   often does not attend to itself. Both references agree, so forcing self in is
   a bug.
6. **Causality.** Future tokens never influence an earlier row's selection. A
   version that scores the incomplete pool leaks.
7. **Tail cache.** Prefill-then-decode equals one-shot prefill, for every
   ``S0 % 4``, across pool completions and across the 2051 ceiling. A stale
   ring (stash only on completion) and an unseeded tail are both caught.
8. **Injected indices.** ``SparseMLAttention(topk_indices=...)`` with the oracle's
   own indices is bit-identical, through decode as well. Given a device's
   noisy-score selection, it separates a 0.5% softmax-scale bug (>1000x the floor)
   from selection divergence, which otherwise swallows the bug completely.

Hidden size and ``q_lora_rank`` are shrunk in the real-geometry tests. They only
size the input projections; nothing about selection depends on them.
Parameters are randomised at scales where the gate softmax is far from uniform
(gate scores ~N(0, 4)). The real checkpoint is milder: gate-score std ~0.23-0.27
and the largest pool slot weight has a median of ~0.33, against 0.25 for a plain
mean. Even so, a mean-pool indexer on the real weights of layers 3/23/43 still
changes the selection on 99-100% of lossy rows at S=3003 (pool overlap ~0.956).
That is measured in dev/progress, not here.
"""
from __future__ import annotations

import os
import json
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from glm5_next import reference as R  # noqa: E402
import indexer_refs as X  # noqa: E402

TOPK, KPOOL, HI, HD = 2048, 4, 32, 128           # real indexer geometry
CEIL = TOPK + KPOOL - 1                           # 2051
HID, QL = 512, 256                                # shrunk: projection inputs only
LONG = [2052, 2101, 2562, 3003]                   # every S % 4, all above the ceiling


# ------------------------------------------------------------------------- helpers
def _randomize(ix: R.Indexer, seed: int):
    g = torch.Generator().manual_seed(seed)
    r = lambda shape: torch.randn(shape, generator=g)
    with torch.no_grad():
        for lin in (ix.wq_b, ix.wk, ix.weights_proj):
            lin.weight.copy_(r(lin.weight.shape) * lin.in_features ** -0.5)
        gate = ix.index_kpool_compress_gate
        gate.copy_(r(gate.shape) * 2 * gate.shape[1] ** -0.5)       # gate scores ~ N(0, 4)
        ix.index_kpool_compress_ape.copy_(r(ix.index_kpool_compress_ape.shape))
        ix.k_norm.weight.copy_(1 + 0.1 * r(ix.k_norm.weight.shape))
        ix.k_norm.bias.copy_(0.1 * r(ix.k_norm.bias.shape))
    return ix


def _inputs(S, hid, ql, B=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, S, hid, generator=g), torch.randn(B, S, ql, generator=g)


def _mask(idx, L):
    return R.indices_to_mask(idx.long(), L)


def _causal(S, L=None):
    L = S if L is None else L
    return torch.ones(S, L, dtype=torch.bool).tril(L - S)


def _covered(L, topk=TOPK, kpool=KPOOL):
    return min(L // kpool, topk // kpool) * kpool + L % kpool


@pytest.fixture(scope="module")
def real_ix():
    return _randomize(R.Indexer(R.FlashCfg(hidden_size=HID, q_lora_rank=QL)), seed=1).eval()


@pytest.fixture(scope="module")
def hf(real_ix):
    m = X.HFIndexer(HID, HI, HD, QL, TOPK, KPOOL)
    m.load_state_dict(real_ix.state_dict())
    return m.eval()


def _oracle(ix, x, qc):
    with torch.no_grad():
        return ix(x, qc)[0]


def _hf(hf, x, qc):
    return hf(x, qc, torch.ones(x.shape[:2], dtype=torch.bool))


def _vllm(ix, x, qc):
    return X.vllm_select(ix.wq_b.weight, ix.wk.weight, ix.k_norm.weight, ix.k_norm.bias,
                         ix.weights_proj.weight, ix.index_kpool_compress_gate,
                         ix.index_kpool_compress_ape, x[0], qc[0], ix.topk, ix.kpool)[None]


# ------------------------------------------------------------------------- mutants
_orig_select, _orig_scores = R.select_tokens, R.index_scores
_orig_compress, _orig_prefill, _orig_decode = R.kpool_compress, R.indexer_prefill, R.indexer_decode


def _m_select_all(scores, lens, topk, kpool, tail=True):
    L = int(lens.max())
    idx = torch.arange(L).expand(scores.shape[0], lens.shape[0], L)
    return idx.masked_fill(torch.arange(L)[None, :] >= lens[:, None], -1)


def _m_random(scores, lens, topk, kpool, tail=True):
    return _orig_select(torch.rand(scores.shape, generator=torch.Generator().manual_seed(7)), lens, topk, kpool, tail)


def _m_bottom_k(scores, lens, topk, kpool, tail=True):
    return _orig_select(-scores, lens, topk, kpool, tail)


def _m_recency(scores, lens, topk, kpool, tail=True):
    return _orig_select(torch.arange(scores.shape[-1]).float().expand_as(scores), lens, topk, kpool, tail)


def _m_off_by_one(scores, lens, topk, kpool, tail=True):
    idx = _orig_select(scores, lens, topk, kpool, tail)
    hist = torch.where(idx[..., :topk] >= 0, idx[..., :topk] + 1, -1)
    hist = hist.masked_fill(hist >= lens[:, None], -1)
    return torch.cat([hist, idx[..., topk:]], -1)


def _m_no_tail(scores, lens, topk, kpool, tail=True):
    return _orig_select(scores, lens, topk, kpool, tail=False)


def _m_force_self(scores, lens, topk, kpool, tail=True):
    idx = _orig_select(scores, lens, topk, kpool, tail)
    return torch.cat([idx, (lens - 1).expand(idx.shape[0], -1)[..., None]], -1)


def _m_incomplete_candidate(scores, lens, topk, kpool, tail=True):
    """ceil(L/4) candidates: the pool holding the query is scored even though, in a
    prefill, its compressed key contains future tokens. Future tokens are then
    dropped from the output, so the leak is in *scoring* only."""
    B, S, P = scores.shape
    cand = torch.arange(P)[None, :] < ((lens + kpool - 1) // kpool)[:, None]
    top = scores.masked_fill(~cand, float("-inf")).topk(min(topk // kpool, P), -1).indices
    ok = cand.expand(B, S, P).gather(-1, top)
    tok = (top[..., None] * kpool + torch.arange(kpool)).masked_fill(~ok[..., None], -1).flatten(-2)
    tok = tok.masked_fill(tok >= lens[:, None], -1)
    t = _orig_select(scores, lens, topk, kpool, tail)[..., topk:]
    return torch.cat([torch.nn.functional.pad(tok, (0, topk - tok.shape[-1]), value=-1), t], -1)


def _m_no_relu(q, w, pool_k, scale, rows=256):
    kT = pool_k.float().transpose(-1, -2).unsqueeze(1)
    return (w[:, :, None, :].float() @ ((q.float() @ kT) * scale)).squeeze(-2)


def _m_unweighted(q, w, pool_k, scale, rows=256):
    return _orig_scores(q, torch.ones_like(w), pool_k, scale, rows)


def _m_mean_pool(k, gate, ape):
    return k.float().mean(-2)


SELECTION_MUTANTS = {
    "select-all (dense)": ("select_tokens", _m_select_all),
    "random-k": ("select_tokens", _m_random),
    "bottom-k": ("select_tokens", _m_bottom_k),
    "recency window": ("select_tokens", _m_recency),
    "off-by-one expansion": ("select_tokens", _m_off_by_one),
    "no tail": ("select_tokens", _m_no_tail),
    "force-include self": ("select_tokens", _m_force_self),
    "scores incomplete pool": ("select_tokens", _m_incomplete_candidate),
    "no relu": ("index_scores", _m_no_relu),
    "unweighted heads": ("index_scores", _m_unweighted),
    "mean-pool (no gate/ape)": ("kpool_compress", _m_mean_pool),
}


def _m_stale_ring(state, k, gate, ape, kpool):
    """The bug vLLM's decode kernel records fixing: stash gated on pool completion."""
    new = _orig_decode(state, k, gate, ape, kpool)
    if state.length % kpool != kpool - 1:
        return R.IndexerState(new.pool_k, state.tail_k, state.tail_gate, new.length)
    return new


def _m_unseeded(state, k, gate, ape, kpool):
    new = _orig_prefill(state, k, gate, ape, kpool)
    return R.IndexerState(new.pool_k, torch.zeros_like(new.tail_k), torch.zeros_like(new.tail_gate), new.length)


# ================================================================ 1. cross-reference
def _scores64(ix, x, qc):
    """The oracle's pool scores recomputed in float64 -> [B, S, P], the arbiter for ties."""
    with torch.no_grad():
        B, S, _ = x.shape
        q = ix.wq_b(qc).view(B, S, ix.Hi, ix.D).double()
        k, gate = ix.k_norm(ix.wk(x)).double(), torch.nn.functional.linear(x, ix.index_kpool_compress_gate).double()
        w = ix.weights_proj(x).double() * ix.Hi ** -0.5
        pool_k = _orig_prefill(None, k, gate, ix.index_kpool_compress_ape.double(), ix.kpool).pool_k.double()
        return (w[:, :, None, :] @ torch.relu((q @ pool_k.transpose(-1, -2).unsqueeze(1)) * ix.D ** -0.5)).squeeze(-2)


TIE_TOL = 1e-5      # ~10x the fp32 score noise measured on these inputs (<= 1.1e-6)


def _selection_diff(ma, mb, s64, topk=TOPK, kpool=KPOOL):
    """Compare two selections [B,S,L]. -> (bad, tied): rows that differ for a real reason,
    and rows that differ only by swapping pools whose float64 scores tie the top-k cut
    within TIE_TOL. A tie is a legitimate disagreement: either choice is correct."""
    bad = tied = 0
    k = topk // kpool
    for b, p in (ma != mb).any(-1).nonzero().tolist():
        nc = (p + 1) // kpool
        diff = (ma[b, p] ^ mb[b, p]).nonzero().flatten()
        if (diff >= nc * kpool).any() or nc <= k:       # tail, or a row that must select everything
            bad += 1
            continue
        row = s64[b, p, :nc]
        cut = row.sort(descending=True).values[k - 1:k + 1]
        pools = (diff // kpool).unique()
        if ((row[pools, None] - cut[None, :]).abs().min(-1).values <= TIE_TOL).all():
            tied += 1
        else:
            bad += 1
    return bad, tied


def _cut_gaps(s64, S, topk=TOPK, kpool=KPOOL):
    """float64 gap between the last selected and first rejected pool, per lossy row."""
    k = topk // kpool
    return torch.tensor([float(v[k - 1] - v[k]) for v in
                         (s64[0, p, : (p + 1) // kpool].sort(descending=True).values
                          for p in range(topk + kpool - 1, S))])


@pytest.mark.parametrize("S", LONG)
def test_oracle_matches_transformers_and_vllm_above_the_ceiling(real_ix, hf, S):
    x, qc = _inputs(S, HID, QL, seed=S)
    a, b, c = _oracle(real_ix, x, qc), _hf(hf, x, qc), _vllm(real_ix, x, qc)
    ma, mb, mc = _mask(a, S), _mask(b, S), _mask(c, S)
    s64 = _scores64(real_ix, x, qc)
    bad_hf, tol_hf = _selection_diff(ma, mb, s64)
    bad_vllm, tol_vllm = _selection_diff(ma, mc, s64)
    assert bad_hf == 0, f"oracle vs transformers: {bad_hf} rows differ beyond a tie"
    assert bad_vllm == 0, f"oracle vs vLLM: {bad_vllm} rows differ beyond a tie"
    # vLLM's exact layout: 2048 history columns then kpool-1 tail columns, tail identical
    assert a.shape[-1] == c.shape[-1] == TOPK + KPOOL - 1
    assert torch.equal(a[..., TOPK:].int(), c[..., TOPK:])
    # non-vacuous: rows really do drop tokens, and exactly as many as the arithmetic says
    counts = ma.sum(-1)[0]
    L = torch.arange(1, S + 1)
    assert torch.equal(counts, torch.tensor([_covered(int(l)) for l in L]))
    assert (counts < L).sum() == S - CEIL
    gaps = _cut_gaps(s64, S)
    print(f"\n  S={S}: {S - CEIL} lossy rows; near-tie rows tolerated vs transformers {tol_hf}, "
          f"vs vLLM {tol_vllm}; cut gap min {gaps.min():.2e}, median {gaps.median():.2e}, "
          f"{int((gaps < TIE_TOL).sum())} rows inside {TIE_TOL:.0e}")


@pytest.mark.parametrize("S", [1, 2, 3, 4, 5, 7, 8, 19, 20, 21, 33, 64])
def test_oracle_matches_transformers_at_tiny_topk_batched(S):
    """topk 16: the lossy regime starts at 20. B=2 so batch handling is covered."""
    cfg = R.tiny_cfg()
    ix = _randomize(R.Indexer(cfg), seed=3).eval()
    m = X.HFIndexer(cfg.hidden_size, cfg.index_n_heads, cfg.index_head_dim, cfg.q_lora_rank,
                    cfg.index_topk, cfg.index_kpool)
    m.load_state_dict(ix.state_dict())
    x, qc = _inputs(S, cfg.hidden_size, cfg.q_lora_rank, B=2, seed=S)
    bad, _ = _selection_diff(_mask(_oracle(ix, x, qc), S), _mask(_hf(m, x, qc), S), _scores64(ix, x, qc),
                             cfg.index_topk, cfg.index_kpool)
    assert bad == 0


def test_vllm_shortcut_equals_the_sparse_path_below_its_gate(real_ix):
    """vLLM's short-prefill fill (S <= 2048) and its own sparse path agree there."""
    S = TOPK
    x, qc = _inputs(S, HID, QL, seed=11)
    short = X.vllm_short_prefill(S, S)
    assert torch.equal(_mask(short[None], S), _mask(_vllm(real_ix, x, qc), S))


# ===================================================================== 2. continuity
@pytest.mark.parametrize("S", [1, 3, 4, 5, 2047, 2048, 2049, 2050, 2051])
def test_sparse_path_selects_exactly_the_causal_set_up_to_2051(real_ix, S):
    x, qc = _inputs(S, HID, QL, seed=S)
    assert torch.equal(_mask(_oracle(real_ix, x, qc), S)[0], _causal(S))


def test_2052_drops_exactly_one_pool_from_exactly_one_row(real_ix):
    S = CEIL + 1
    x, qc = _inputs(S, HID, QL, seed=5)
    m = _mask(_oracle(real_ix, x, qc), S)[0]
    missing = _causal(S) & ~m
    assert not (m & ~_causal(S)).any()
    assert missing[:-1].sum() == 0
    gone = missing[-1].nonzero().flatten()
    assert len(gone) == KPOOL and gone[0] % KPOOL == 0 and torch.equal(gone, gone[0] + torch.arange(KPOOL))


def _real_topk_mla(seed=0):
    cfg = R.tiny_cfg(index_n_heads=HI, index_head_dim=HD, index_topk=TOPK)
    torch.manual_seed(seed)
    m = R.SparseMLAttention(cfg).eval()
    _randomize(m.indexer, seed)
    return m


def _layer(m, x, dense=False, state=None):
    prev, m.dense = m.dense, dense
    try:
        with torch.no_grad():
            return m(x, state)
    finally:
        m.dense = prev


def test_layer_output_is_bit_identical_to_dense_at_2051():
    m = _real_topk_mla()
    x = torch.randn(1, CEIL, m.q_a_proj.in_features)
    assert torch.equal(_layer(m, x)[0], _layer(m, x, dense=True)[0])


def test_layer_output_diverges_from_dense_at_2052_on_the_last_row_only():
    m = _real_topk_mla()
    x = torch.randn(1, CEIL + 1, m.q_a_proj.in_features)
    sparse, dense = _layer(m, x)[0], _layer(m, x, dense=True)[0]
    assert torch.equal(sparse[:, :-1], dense[:, :-1])
    assert (sparse[:, -1] - dense[:, -1]).abs().max() > 1e-4


@pytest.mark.parametrize("S", [5, 13, 19])
def test_whole_model_matches_dense_up_to_the_tiny_ceiling(S):
    """tiny_cfg: topk 16, so the ceiling is 19. Both sparse-MLA layers take the real path."""
    torch.manual_seed(0)
    model = R.FlashTextModel(R.tiny_cfg(), vocab=500).eval()
    ids = torch.randint(0, 500, (2, S))
    mla = [l.self_attn for l in model.layers if not l.linear]
    assert len(mla) == 2
    with torch.no_grad():
        sparse = model(ids)
        for a in mla: a.dense = True
        dense = model(ids)
    assert torch.equal(sparse, dense)


def test_whole_model_diverges_from_dense_past_the_tiny_ceiling():
    torch.manual_seed(0)
    model = R.FlashTextModel(R.tiny_cfg(), vocab=500).eval()
    ids = torch.randint(0, 500, (2, 20))
    mla = [l.self_attn for l in model.layers if not l.linear]
    with torch.no_grad():
        sparse = model(ids)
        for a in mla: a.dense = True
        dense = model(ids)
    torch.testing.assert_close(sparse[:, :-1], dense[:, :-1], rtol=0, atol=1e-5)
    assert (sparse[:, -1] - dense[:, -1]).abs().max() > 1e-4


# ================================================================== 3. discrimination
@pytest.fixture(scope="module")
def long_case(real_ix, hf):
    S = LONG[-1]
    x, qc = _inputs(S, HID, QL, seed=S)
    return S, x, qc, _mask(_hf(hf, x, qc), S)


def _step_decode(m, x, start):
    """Prefill x[:, :start], then decode one token at a time through the layer's own state."""
    out, st = _layer(m, x[:, :start])
    outs = [out]
    for t in range(start, x.shape[1]):
        o, st = _layer(m, x[:, t:t + 1], state=st)
        outs.append(o)
    return torch.cat(outs, 1)


@pytest.fixture(scope="module")
def layer_case():
    """Real-topk sparse-MLA layer at S=3003, and its agreement floor: one-shot prefill vs
    prefill-then-decode for the last 8 rows, two correct computations of the same thing."""
    m = _real_topk_mla()
    S = LONG[-1]
    x = torch.randn(1, S, m.q_a_proj.in_features, generator=torch.Generator().manual_seed(2))
    ref = _layer(m, x)[0]
    floor = (ref - _step_decode(m, x, S - 8)).abs().max().item()
    assert 0 < floor < 1e-6
    return m, x, ref, floor


MARGINS: dict[str, str] = {}


@pytest.mark.parametrize("name", list(SELECTION_MUTANTS))
def test_mutant_is_rejected_above_the_ceiling(real_ix, long_case, layer_case, monkeypatch, name):
    fn, mutant = SELECTION_MUTANTS[name]
    S, x, qc, ref_mask = long_case
    m, lx, ref_out, floor = layer_case
    monkeypatch.setattr(R, fn, mutant)
    rows = int((_mask(_oracle(real_ix, x, qc), S) != ref_mask).any(-1).sum())
    out = _layer(m, lx)[0]
    monkeypatch.undo()
    assert rows > 0, f"{name}: selection identical to transformers on all {S} rows"
    nan_rows = int(torch.isnan(out).any(-1).sum())
    finite = ~torch.isnan(out).any(-1)
    shift = (out - ref_out)[finite].abs().max().item()
    ratio = shift / floor
    MARGINS[name] = f"{rows:5d} rows  {shift:.3e}  {ratio:>12,.0f}x" + (f"  (+{nan_rows} NaN rows)" if nan_rows else "")
    assert ratio > 1e3, f"{name}: output moves only {ratio:.0f}x the agreement floor"
    if name == "no tail":   # rows seeing < kpool tokens have no complete pool and select nothing
        assert nan_rows == KPOOL - 1


def test_select_all_is_invisible_at_or_below_the_ceiling(monkeypatch):
    """Why the test above runs at 3003: at 2051 a dense 'indexer' is bit-identical."""
    m = _real_topk_mla()
    x = torch.randn(1, CEIL, m.q_a_proj.in_features)
    ref = _layer(m, x)[0]
    monkeypatch.setattr(R, "select_tokens", _m_select_all)
    assert torch.equal(_layer(m, x)[0], ref)


def test_zz_report_mutant_margins(layer_case):
    """Printed with -s. Rows = rows whose selection differs from transformers at S=3003;
    shift = max |output - oracle output| of the real-topk layer; x = shift / floor."""
    floor = layer_case[3]
    print(f"\n  agreement floor (prefill vs decode, real topk, S=3003): {floor:.3e}")
    for name, line in MARGINS.items():
        print(f"  {name:26s} {line}")
    assert len(MARGINS) == len(SELECTION_MUTANTS)


# ============================================================== 4. needle, 5. self, 6. causality
def _pipeline(q, k, gate, w, ape, lens, select=None):
    st = R.indexer_prefill(None, k, gate, ape, KPOOL)
    return (select or R.select_tokens)(R.index_scores(q, w, st.pool_k, HD ** -0.5), lens, TOPK, KPOOL)


def _crafted(S, seed=0):
    g = torch.Generator().manual_seed(seed)
    q, k = torch.randn(1, S, HI, HD, generator=g), torch.randn(1, S, HD, generator=g)
    gate, w = torch.randn(1, S, HD, generator=g), torch.randn(1, S, HI, generator=g).abs()
    return q, k, gate, w, torch.randn(KPOOL, HD, generator=g)


def _direction(q, w, p):
    u = (w[0, p, :, None] * q[0, p]).sum(0)
    return u / u.norm()


def test_planted_needle_is_found_and_a_recent_dud_is_dropped():
    """Pool 5 (tokens 20-23) is made maximally relevant to the last query; the most
    recent complete pool is zeroed so it scores 0, the minimum under ReLU with w > 0."""
    S = LONG[-1]
    q, k, gate, w, ape = _crafted(S)
    p = S - 1
    k[0, 20:24] = 10 * _direction(q, w, p)
    last = (S // KPOOL - 1) * KPOOL
    k[0, last:last + KPOOL] = 0
    lens = torch.arange(1, S + 1)
    row = _mask(_pipeline(q, k, gate, w, ape, lens), S)[0, p]
    assert row[20:24].all(), "needle 1,000 pools back was not selected"
    assert not row[last:last + KPOOL].any(), "a zero-score recent pool was selected"
    assert row[S - S % KPOOL:].all(), "tail missing"
    rec = _mask(_pipeline(q, k, gate, w, ape, lens, select=_m_recency), S)[0, p]
    assert not rec[20:24].any() and rec[last:last + KPOOL].all(), "recency mutant should fail both"


def test_query_does_not_always_attend_to_itself(real_ix, long_case):
    """Above the ceiling with L % 4 == 0 the tail is empty, so the query's own token is
    selected only if its pool wins top-k. Both references agree; forcing self in is a bug."""
    S, x, qc, ref_mask = long_case
    mine = _mask(_oracle(real_ix, x, qc), S)[0]
    rows = [p for p in range(CEIL, S) if (p + 1) % KPOOL == 0]
    excluded = [p for p in rows if not mine[p, p]]
    print(f"\n  S={S}: query excludes itself on {len(excluded)}/{len(rows)} rows with L % 4 == 0")
    assert len(excluded) > 0
    for p in excluded:
        assert not ref_mask[0, p, p], f"row {p}: transformers attends to self, oracle does not"
    # below the ceiling, and whenever L % 4 != 0, self is always present (tail or all-pools)
    others = [p for p in range(S) if p < CEIL or (p + 1) % KPOOL != 0]
    assert all(mine[p, p] for p in others)


def test_future_tokens_never_move_an_earlier_selection(real_ix):
    S, t = LONG[-1], 2502                       # t mid-pool: pool 625 = 2500..2503 straddles it
    x, qc = _inputs(S, HID, QL, seed=21)
    x2, qc2 = x.clone(), qc.clone()
    x2[:, t:] += 3 * torch.randn_like(x2[:, t:]); qc2[:, t:] += 3 * torch.randn_like(qc2[:, t:])
    a, b = _mask(_oracle(real_ix, x, qc), S), _mask(_oracle(real_ix, x2, qc2), S)
    assert torch.equal(a[:, :t], b[:, :t])
    assert not torch.equal(a[:, t:], b[:, t:]), "perturbation had no effect at all"


def test_scoring_the_incomplete_pool_leaks_the_future():
    """Plant a needle in future tokens 2502-2503: row 2501 must not react. It does under
    the ceil-candidates mutant, whose pool 625 compresses 2500..2503."""
    S, p = LONG[-1], 2501
    q, k, gate, w, ape = _crafted(S, seed=4)
    lens = torch.arange(1, S + 1)
    k[0, 2500:2504] = 0                          # base: pool 625 scores 0, so it starts unselected
    k2, gate2 = k.clone(), gate.clone()
    k2[0, p + 1:p + 3] = 50 * _direction(q, w, p)
    gate2[0, p + 1:p + 3] = 20.0
    ok = [_mask(_pipeline(q, kk, gg, w, ape, lens), S)[0, p] for kk, gg in ((k, gate), (k2, gate2))]
    bad = [_mask(_pipeline(q, kk, gg, w, ape, lens, select=_m_incomplete_candidate), S)[0, p]
           for kk, gg in ((k, gate), (k2, gate2))]
    assert torch.equal(ok[0], ok[1])
    assert not torch.equal(bad[0], bad[1]), "mutant did not react to the future needle"


# ===================================================================== 7. tail cache
def _tiny_ix(seed=0):
    cfg = R.tiny_cfg()
    return cfg, _randomize(R.Indexer(cfg), seed).eval()


STATE_TOL = 1e-5    # one-token vs batched projections differ by ~1e-6 in fp32


def _one_shot(ix, x, qc):
    with torch.no_grad():
        idx, st = ix(x, qc)
    return _mask(idx, x.shape[1]), st, _scores64(ix, x, qc)


def _decode_matches_prefill(ix, x, qc, S0, S1, chunk=None, reference=None):
    """Prefill S0 (optionally continue with a multi-token chunk), decode the rest.
    -> (rows whose selection differs from one-shot prefill(S1) beyond a tie, tied rows,
    final-state max diff). Pass ``reference`` (from _one_shot) when a mutant is patched
    in, so the reference stays clean."""
    full_m, full, s64 = reference or _one_shot(ix, x[:, :S1], qc[:, :S1])
    got = torch.zeros_like(full_m)
    with torch.no_grad():
        idx, st = ix(x[:, :S0], qc[:, :S0])
        got[:, :S0] = _mask(idx, S1)
        t = S0
        if chunk:
            idx, st = ix(x[:, t:t + chunk], qc[:, t:t + chunk], st)
            got[:, t:t + chunk] = _mask(idx, S1)
            t += chunk
        for t in range(t, S1):
            before = (st.pool_k.clone(), st.tail_k.clone(), st.tail_gate.clone(), st.length)
            idx, st_new = ix(x[:, t:t + 1], qc[:, t:t + 1], st)
            assert torch.equal(st.pool_k, before[0]) and torch.equal(st.tail_k, before[1]) \
                and torch.equal(st.tail_gate, before[2]) and st.length == before[3], "state mutated in place"
            st = st_new
            got[:, t] = _mask(idx, S1)[:, 0]
    assert st.length == full.length == S1
    diff = max((st.pool_k - full.pool_k).abs().max().item() if st.pool_k.shape == full.pool_k.shape else float("inf"),
               (st.tail_k - full.tail_k).abs().max().item(), (st.tail_gate - full.tail_gate).abs().max().item())
    bad, tied = _selection_diff(got, full_m, s64, ix.topk, ix.kpool)
    return bad, tied, diff


@pytest.mark.parametrize("S0", [28, 29, 30, 31])
def test_prefill_then_decode_equals_one_shot_prefill(S0):
    """Every S0 % 4, 13 decode steps = 3+ pool completions, all in the lossy regime (topk 16)."""
    cfg, ix = _tiny_ix()
    x, qc = _inputs(S0 + 13, cfg.hidden_size, cfg.q_lora_rank, B=2, seed=S0)
    bad, _, diff = _decode_matches_prefill(ix, x, qc, S0, S0 + 13)
    assert bad == 0 and diff < STATE_TOL


@pytest.mark.parametrize("S0", [28, 29, 30, 31])
def test_continuation_prefill_equals_one_shot_prefill(S0):
    cfg, ix = _tiny_ix()
    x, qc = _inputs(S0 + 20, cfg.hidden_size, cfg.q_lora_rank, B=2, seed=S0)
    bad, _, diff = _decode_matches_prefill(ix, x, qc, S0, S0 + 20, chunk=7)
    assert bad == 0 and diff < STATE_TOL


def test_decode_across_the_real_ceiling(real_ix):
    """Real geometry: prefill 2049, decode to 2057, crossing 2051 -> lossy rows via decode."""
    x, qc = _inputs(2057, HID, QL, seed=9)
    bad, _, diff = _decode_matches_prefill(real_ix, x, qc, 2049, 2057)
    assert bad == 0 and diff < STATE_TOL
    assert _covered(2056) < 2056                # the decoded tail end really is lossy


@pytest.mark.parametrize("mutant", ["stale ring", "unseeded tail"])
def test_broken_tail_cache_is_caught(monkeypatch, mutant):
    cfg, ix = _tiny_ix()
    cases = {S0: _inputs(S0 + 13, cfg.hidden_size, cfg.q_lora_rank, B=2, seed=S0) for S0 in (28, 29, 30, 31)}
    refs = {S0: _one_shot(ix, *cases[S0]) for S0 in cases}
    if mutant == "stale ring": monkeypatch.setattr(R, "indexer_decode", _m_stale_ring)
    else: monkeypatch.setattr(R, "indexer_prefill", _m_unseeded)
    results = {S0: _decode_matches_prefill(ix, *cases[S0], S0, S0 + 13, reference=refs[S0]) for S0 in cases}
    print(f"\n  {mutant}: S0 -> (wrong-selection rows, tied rows, state diff): "
          + ", ".join(f"{k}: ({b}, {t}, {d:.2e})" for k, (b, t, d) in results.items()))
    if mutant == "stale ring":
        assert all(d > 1e-2 for _, _, d in results.values())
    else:
        # seeding matters exactly when prefill leaves an incomplete pool behind
        assert results[28][0] == 0 and results[28][2] < STATE_TOL
        assert all(results[s][2] > 1e-2 for s in (29, 30, 31))
    assert sum(b for b, _, _ in results.values()) > 0, "state corrupted but no selection ever changed"


def test_layer_decode_matches_prefill_through_the_mla_state():
    """SparseMLAttention carries (latent, IndexerState) through its own decode path."""
    cfg = R.tiny_cfg()
    torch.manual_seed(0)
    m = R.SparseMLAttention(cfg).eval()
    _randomize(m.indexer, 0)
    x = torch.randn(1, 64, cfg.hidden_size)
    ref = _layer(m, x)[0]
    got = _step_decode(m, x, 1)
    assert (ref - got).abs().max() < 1e-5


def test_state_out_of_step_with_kv_cache_is_refused():
    cfg = R.tiny_cfg()
    m = R.SparseMLAttention(cfg).eval()
    x = torch.randn(1, 8, cfg.hidden_size)
    latent, _ = _layer(m, x)[1]
    with pytest.raises(AssertionError, match="out of step"):
        _layer(m, x[:, :1], state=(latent, None))


# ======================================================================= config pins
def test_indexer_config_is_the_verified_one():
    cfg = R.FlashCfg()
    assert (cfg.index_n_heads, cfg.index_head_dim, cfg.index_topk, cfg.index_kpool) == (HI, HD, TOPK, KPOOL)
    assert cfg.index_kpool_compress and cfg.index_kpool_always_select_tail
    assert not cfg.sparse_mla_dense
    path = os.environ.get("GLM53F_CONFIG")
    if path:
        live = json.loads(pathlib.Path(path).read_text())["text_config"]
        for key in ("index_n_heads", "index_head_dim", "index_topk", "index_kpool",
                    "index_kpool_compress", "index_kpool_always_select_tail", "qk_rope_head_dim"):
            assert live[key] == getattr(cfg, key), key


def test_no_rope_in_the_indexer_because_qk_rope_head_dim_is_zero():
    """Both references split off qk_rope_head_dim dims for RoPE; it is 0, so none is
    applied and indexer_rope_interleave is inert. If this changes, the indexer needs RoPE."""
    assert R.FlashCfg().qk_rope_head_dim == 0


# ================================================================ 8. injected indices
# SparseMLAttention(topk_indices=...) replaces the indexer's selection for one call, so
# attention can be validated against a device *given the device's own selection*. FP8
# scoring legitimately picks different pools, and without this an attention bug hides
# behind that divergence (test_hook_separates_an_attention_bug_from_selection_divergence).
def _own_indices(m, x, idx_state=None):
    with torch.no_grad():
        return m.indexer(x, m.q_a_layernorm(m.q_a_proj(x)), idx_state)[0]


def _layer_inj(m, x, idx, state=None):
    with torch.no_grad():
        return m(x, state, topk_indices=idx)


def test_injecting_own_indices_is_bit_identical(layer_case):
    """The hook perturbs nothing: output and returned state match bit for bit (S=3003, lossy)."""
    m, x, ref, _ = layer_case
    out, (latent, st) = _layer_inj(m, x, _own_indices(m, x))
    ref_out, (ref_latent, ref_st) = _layer(m, x)
    assert torch.equal(out, ref) and torch.equal(out, ref_out)
    assert torch.equal(latent, ref_latent) and st.length == ref_st.length
    assert torch.equal(st.pool_k, ref_st.pool_k) and torch.equal(st.tail_k, ref_st.tail_k)


def test_injecting_own_indices_is_bit_identical_through_decode():
    cfg = R.tiny_cfg()
    torch.manual_seed(0)
    m = R.SparseMLAttention(cfg).eval()
    _randomize(m.indexer, 0)
    x = torch.randn(1, 40, cfg.hidden_size)
    a, sa = _layer(m, x[:, :25])
    b, sb = _layer_inj(m, x[:, :25], _own_indices(m, x[:, :25]))
    assert torch.equal(a, b)
    for t in range(25, 40):
        own = _own_indices(m, x[:, t:t + 1], sb[1])
        a, sa = _layer(m, x[:, t:t + 1], state=sa)
        b, sb = _layer_inj(m, x[:, t:t + 1], own, state=sb)
        assert torch.equal(a, b), f"step {t}"
        assert torch.equal(sa[1].pool_k, sb[1].pool_k) and torch.equal(sa[0], sb[0])


def test_injection_is_order_and_padding_invariant(layer_case):
    """A captured buffer is a set: column order and -1 padding width must not matter."""
    m, x, ref, _ = layer_case
    own = _own_indices(m, x)
    perm = own[..., torch.randperm(own.shape[-1], generator=torch.Generator().manual_seed(0))]
    padded = torch.nn.functional.pad(perm, (0, 37), value=-1)
    assert torch.equal(_layer_inj(m, x, padded)[0], ref)


def test_injecting_the_causal_set_reproduces_dense(layer_case):
    """Proves the injected indices really drive the mask: select-all above the ceiling == dense."""
    m, x, ref, _ = layer_case
    S = x.shape[1]
    causal = torch.arange(S).expand(1, S, S).masked_fill(~_causal(S), -1)
    got = _layer_inj(m, x, causal)[0]
    assert torch.equal(got, _layer(m, x, dense=True)[0])
    assert not torch.equal(got, ref)


def test_injection_refuses_selections_no_indexer_can_emit():
    cfg = R.tiny_cfg()
    m = R.SparseMLAttention(cfg).eval()
    x = torch.randn(1, 6, cfg.hidden_size)
    ok = torch.arange(6).expand(1, 6, 6).masked_fill(~_causal(6), -1)
    _layer_inj(m, x, ok)
    future = ok.clone(); future[0, 2, 5] = 3
    dup = ok.clone(); dup[0, 4, 1] = 0
    for bad, msg in ((future, "future"), (dup, "duplicate"), (ok.clamp(min=-2) - 1, "out of range"),
                     (ok + 6 * (ok >= 0), "out of range")):
        with pytest.raises(ValueError, match=msg):
            _layer_inj(m, x, bad)
    with pytest.raises(AssertionError, match=r"\[B, S, W\]"):
        _layer_inj(m, x, ok[:, :5])


def _gather_attention(m, x, idx, scale_mult=1.0):
    """An independent sparse MLA, shaped like FlashMLA's sparse kernel: per query row,
    gather the selected tokens' K/V and softmax over them. No mask. Stands in for a device."""
    with torch.no_grad():
        B, S, _ = x.shape
        q = m.q_b_proj(m.q_a_layernorm(m.q_a_proj(x))).view(B, S, m.Hh, m.qk)
        kv = m.kv_b_proj(m.kv_a_layernorm(m.kv_a_proj_with_mqa(x)[..., :m.kvr])).view(B, S, m.Hh, m.qk + m.vd)
        k, v = kv.split([m.qk, m.vd], -1)
        out = torch.empty(B, S, m.Hh, m.vd)
        for b in range(B):
            for p in range(S):
                sel = idx[b, p][idx[b, p] >= 0].long()
                s = torch.einsum("hd,nhd->hn", q[b, p], k[b, sel]) * m.scaling * scale_mult
                out[b, p] = torch.einsum("hn,nhd->hd", s.softmax(-1), v[b, sel])
        return m.o_proj(out.reshape(B, S, -1))


def test_hook_separates_an_attention_bug_from_selection_divergence(layer_case, monkeypatch):
    """A 'device' selects with noisy scores (sigma 1e-2 ~ the bf16 score error measured on
    random weights) and runs its own gather attention. Two devices: correct, and one with a
    0.5% softmax-scale bug.

    Without the hook both sit ~equally far from the oracle: the bug is invisible inside
    selection divergence. With the device's indices injected, the correct device agrees to
    the floor and the bug stands out by >1000x."""
    m, x, oracle, _ = layer_case
    S = x.shape[1]
    g = torch.Generator().manual_seed(0)
    monkeypatch.setattr(R, "index_scores", lambda q, w, pk, sc, rows=256: _orig_scores(q, w, pk, sc, rows)
                        + 1e-2 * torch.randn(q.shape[0], q.shape[1], pk.shape[1], generator=g))
    dev_idx = _own_indices(m, x)
    monkeypatch.undo()
    rows = int((_mask(dev_idx, S) != _mask(_own_indices(m, x), S)).any(-1).sum())
    dev_ok, dev_bug = _gather_attention(m, x, dev_idx), _gather_attention(m, x, dev_idx, 1.005)
    inj = _layer_inj(m, x, dev_idx)[0]
    e = lambda a, b: (a - b).abs().max().item()
    print(f"\n  device selects differently on {rows} rows. without hook: correct {e(oracle, dev_ok):.2e}, "
          f"0.5% bug {e(oracle, dev_bug):.2e} | with hook: correct {e(inj, dev_ok):.2e}, "
          f"0.5% bug {e(inj, dev_bug):.2e}")
    assert rows > 100
    assert e(oracle, dev_ok) > 1e-3                      # selection divergence is large
    assert e(oracle, dev_bug) < 1.1 * e(oracle, dev_ok)  # ...and swallows the bug
    assert e(inj, dev_ok) < 1e-6                         # hook: the correct device agrees to the floor
    assert e(inj, dev_bug) > 1e3 * e(inj, dev_ok)        # ...and the bug is plain
