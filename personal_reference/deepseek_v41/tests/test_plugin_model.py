"""The plugin's DeepSeek-V4.1 model against the oracle, through a fake runner.

One scenario covers every stateful seam the serving path crosses:

* A: prefill of an odd length (an open ratio-2 group carried into decode), padded to a
  bucket, then 23 decode steps.
* B: a prefix-cache hit on A's first compressed block, prefilled from position 32 --
  window history, compressor entries and index keys all read back from A's pages.
* A and B decoded together at different positions, with dead pad rows.

All against the oracle's exact mode, one-shot prefill of each full sequence. Pages are
hostile (0xFF = NaN) wherever the model did not write, including the zero page.
"""

import functools
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from personal_reference.deepseek_v41.tests import harness  # noqa: E402
from personal_reference.deepseek_v41.tests import plugin_harness as ph  # noqa: E402

M = ph.import_plugin("vllm_neuron.model.deepseek_v41.model")
TOL = 1e-5


def _oracle_logits(seq):
    ref, _, _ = harness.build("exact", 0)
    ref.head.forward = functools.partial(type(ref.head).forward, ref.head, full_logits=True)
    with torch.no_grad():
        _, logits, _ = ref(torch.tensor([seq]), 0)
    return logits[0]


def _rel(a, b):
    return ((a - b).norm(dim=-1) / b.norm(dim=-1)).max().item()


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(1)
    ref, args, _ = harness.build("exact", 0)
    V = args.vocab_size
    A = torch.randint(0, V, (70,)).tolist()
    B = A[:32] + torch.randint(0, V, (38,)).tolist()
    return ref, args, A, B, _oracle_logits(A[:60]), _oracle_logits(B[:68])


def _scenario(setup):
    ref, args, A, B, OA, OB = setup
    plug = ph.plugin_from_reference(ref, args, block_size=8)
    with torch.no_grad():
        run = ph.FakeRunner(plug)
        assert run.bs(run.C) == 32, "the prefix share below assumes one 32-token block"
        la = [run.prefill("a", A[:37], 0, bucket=48)]
        run.share_prefix("a", "b", 32)
        lb = [run.prefill("b", B[32:45], 32, bucket=16)]
        pa, pb = 37, 45
        while pa < 60:
            out = run.decode(["a", "b"], [A[pa], B[pb]], [pa, pb], pad_rows=2)
            la.append(out[:1])
            lb.append(out[1:])
            pa, pb = pa + 1, pb + 1
    la, lb = torch.cat(la), torch.cat(lb)
    return {"A prefill": _rel(la[:37], OA[:37]), "A decode": _rel(la[37:], OA[37:60]),
            "B cached prefill": _rel(lb[:13], OB[32:45]), "B decode": _rel(lb[13:], OB[45:68])}


@pytest.fixture
def deferred_writes(monkeypatch):
    """Nothing written during a step is visible until the step ends: the device's aliasing
    pass is allowed to order writes after reads, and the model must not care."""
    queue, real = [], M.write_cache_rows

    def write(cache, rows, idx):
        queue.append((cache, rows.clone(), idx.clone()))
        return cache

    fwd = M.DeepseekV41Model.forward

    def forward(self, *a, **k):
        out = fwd(self, *a, **k)
        for c, r, i in queue:
            real(c, r, i)
        queue.clear()
        return out

    monkeypatch.setattr(M, "write_cache_rows", write)
    monkeypatch.setattr(M.DeepseekV41Model, "forward", forward)


def test_matches_oracle(setup):
    errs = _scenario(setup)
    assert max(errs.values()) < TOL, errs


def test_matches_oracle_with_deferred_writes(setup, deferred_writes):
    errs = _scenario(setup)
    assert max(errs.values()) < TOL, errs


# -- the scenario must be able to fail: each mutation breaks one seam ---------------------
def test_detects_missing_fresh_substitution(setup, deferred_writes, monkeypatch):
    monkeypatch.setattr(M, "_substitute", lambda g, f, j, v: g)
    errs = _scenario(setup)
    assert not all(e < TOL for e in errs.values()), errs


def test_detects_window_history_off_by_one(setup, monkeypatch):
    real = M.build_step

    def build(*a, **k):
        st = real(*a, **k)
        st.h_pos = st.h_pos - 1
        return st

    monkeypatch.setattr(M, "build_step", build)
    errs = _scenario(setup)
    assert errs["A prefill"] < TOL  # no history at s = 0
    assert min(errs["A decode"], errs["B cached prefill"], errs["B decode"]) > 0.1, errs


def test_detects_ignored_candidates(setup, monkeypatch):
    real = M.Indexer.forward

    def forward(self, *a, **k):
        keep, self.uses_candidates = self.uses_candidates, False
        try:
            return real(self, *a, **k)
        finally:
            self.uses_candidates = keep

    monkeypatch.setattr(M.Indexer, "forward", forward)
    errs = _scenario(setup)
    assert min(errs.values()) > 1e-2, errs


def test_detects_lost_compressor_state(setup, monkeypatch):
    real = M.Compressor.forward

    def forward(self, x, step, caches):
        if step.decode and self.ratio > 1:
            step = M.dataclasses.replace(step, h_page=torch.full_like(step.h_page, 1))
        return real(self, x, step, caches)

    monkeypatch.setattr(M.Compressor, "forward", forward)
    errs = _scenario(setup)
    assert errs["A prefill"] < TOL and errs["A decode"] > 1e-3, errs
