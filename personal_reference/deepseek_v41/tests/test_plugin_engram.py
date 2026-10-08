"""Engram: the host hasher against the reference's NgramHashState, and the plugin model
with Engram layers against the oracle through the same prefix-hit and decode scenario.

The released table needs a tokenizer to build its compressed-token map; here the map
is synthetic (``t % 300``), patched into the reference so both sides use it.
"""

import functools
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from personal_reference.deepseek_v41 import oracle  # noqa: E402
from personal_reference.deepseek_v41.tests import harness  # noqa: E402
from personal_reference.deepseek_v41.tests import plugin_harness as ph  # noqa: E402

E = ph.import_plugin("vllm_neuron.model.deepseek_v41.engram")
CV = 300
ENGRAM = dict(engram_layer_ids=(1, 4), engram_num_embeddings=(400, 600), engram_max_ngram_size=4,
              engram_vocab_size=50, engram_n_heads=2, engram_head_dim=32, engram_pad_id=2,
              engram_compressed_vocab_size=CV)
TOKEN_MAP = [t % CV for t in range(harness.SMALL["vocab_size"])]


@pytest.fixture(scope="module", autouse=True)
def synthetic_token_map():
    oracle.load_reference("exact")
    eng = sys.modules["engram"]
    real = eng.build_compressed_token_map
    eng.build_compressed_token_map = lambda tok: (TOKEN_MAP, CV)
    yield
    eng.build_compressed_token_map = real


def _build():
    return harness.build("exact", 0, **ENGRAM)


def test_hasher_matches_reference_ngram_state():
    ref, args, _ = _build()
    torch.manual_seed(3)
    seq = torch.randint(0, args.vocab_size, (1, 50))
    want_prefill = ref.engram_hash(seq[:, :40], 0)[0]
    want_decode = torch.cat([ref.engram_hash(seq[:, p:p + 1], p)[0] for p in range(40, 50)])
    h = E.EngramHasher(args, TOKEN_MAP, CV)
    assert torch.equal(h(seq[0].tolist(), 0, 40), want_prefill)
    assert torch.equal(h(seq[0].tolist(), 40, 10), want_decode)
    assert int(want_prefill.max()) >= 300, "hash rows should reach well into the table"


def test_hasher_rejects_wrong_compressed_vocab():
    _, args, _ = _build()
    with pytest.raises(ValueError):
        E.EngramHasher(args, TOKEN_MAP, CV + 1)


def _oracle_logits(seq):
    ref, _, _ = _build()
    ref.head.forward = functools.partial(type(ref.head).forward, ref.head, full_logits=True)
    with torch.no_grad():
        _, logits, _ = ref(torch.tensor([seq]), 0)
    return logits[0]


def _rel(a, b):
    return ((a - b).norm(dim=-1) / b.norm(dim=-1)).max().item()


def _scenario(blank_engram=False):
    ref, args, _ = _build()
    hasher = E.EngramHasher(args, TOKEN_MAP, CV)

    def ids(seq, start, count):
        out = hasher(seq, start, count)
        return torch.zeros_like(out) if blank_engram else out

    torch.manual_seed(1)
    A = torch.randint(0, args.vocab_size, (60,)).tolist()
    B = A[:32] + torch.randint(0, args.vocab_size, (28,)).tolist()
    plug = ph.plugin_from_reference(ref, args, block_size=8)
    with torch.no_grad():
        run = ph.FakeRunner(plug)
        la = [run.prefill("a", A[:37], 0, bucket=48, engram_ids=ids(A, 0, 37))]
        run.share_prefix("a", "b", 32)
        lb = [run.prefill("b", B[32:45], 32, bucket=16, engram_ids=ids(B, 32, 13))]
        for pa, pb in zip(range(37, 50), range(45, 58)):
            e = torch.cat([ids(A, pa, 1), ids(B, pb, 1)])
            out = run.decode(["a", "b"], [A[pa], B[pb]], [pa, pb], pad_rows=1, engram_ids=e)
            la.append(out[:1])
            lb.append(out[1:])
    la, lb = torch.cat(la), torch.cat(lb)
    OA, OB = _oracle_logits(A[:50]), _oracle_logits(B[:58])
    return {"A": _rel(la, OA), "B cached prefill": _rel(lb[:13], OB[32:45]),
            "B decode": _rel(lb[13:], OB[45:58])}


def test_model_with_engram_matches_oracle():
    errs = _scenario()
    assert max(errs.values()) < 1e-5, errs


def test_engram_contribution_is_visible():
    # non-vacuity: wrong hash rows must show, or the comparison above proves nothing
    errs = _scenario(blank_engram=True)
    assert min(errs.values()) > 1e-2, errs
