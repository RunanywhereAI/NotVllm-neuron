"""Prefill-then-decode must equal one-shot prefill, on the reference itself.

Everything stateful in V4.1 crosses this seam: the sliding-window ring, the compressor's
partial group, the compressed KV and index-key caches, the top-k published by index
sources, and the two-level candidate mask.

Where the two paths select the same compressed positions they must agree to fp32 rounding.
Where they select differently, it is a near-tie in index scores resolved differently by
batched and per-token arithmetic; those steps are counted, not asserted on, and most steps
must be tie-free so the comparison keeps its teeth.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from personal_reference.deepseek_v41.tests import harness  # noqa: E402

PREFILL, END = 40, 60  # window 16, index_topk 8: well past both, so selection is sparse
VOCAB = harness.SMALL["vocab_size"]


def _with_pick_capture(model):
    """Per index-source layer: the compressed positions each row selected, by absolute row."""
    rec = {}

    def make(li):
        def hook(mod, inputs, out):
            start, offset = inputs[3], inputs[4]
            for r in range(out.size(1)):
                rec[(li, start + r)] = [sorted(v - offset for v in row.tolist() if v >= 0) for row in out[:, r]]

        return hook

    for li, layer in enumerate(model.layers):
        if layer.attn.indexer is not None:
            layer.attn.indexer.register_forward_hook(make(li))
    return rec


def _compare(mode, fix=True, x=None):
    """Per decode step: (position, relative logit error vs fresh prefill, did every row of every
    index source select the same positions on both paths)."""
    if x is None:
        torch.manual_seed(1)
        x = torch.randint(0, VOCAB, (2, END))
    dec, _, _ = harness.build(mode, 0, fix_reference_bug=fix)
    rec_dec = _with_pick_capture(dec)
    dec(x[:, :PREFILL], 0)
    steps = []
    for p in range(PREFILL, END):
        _, ld, _ = dec(x[:, p : p + 1], p)
        ref, _, _ = harness.build(mode, 0, fix_reference_bug=fix)
        rec_ref = _with_pick_capture(ref)
        _, lr, _ = ref(x[:, : p + 1], 0)
        err = ((ld - lr).norm() / lr.norm()).item()
        same = all(rec_dec[k] == v for k, v in rec_ref.items())
        steps.append((p, err, same))
    return steps


def test_exact_decode_matches_prefill_wherever_selection_agrees():
    steps = _compare("exact")
    agreeing = [(p, e) for p, e, same in steps if same]
    assert len(agreeing) >= len(steps) // 2, f"too few tie-free steps to mean anything: {steps}"
    worst = max(e for _, e in agreeing)
    assert worst < 1e-5, f"decode != prefill with identical selection: {agreeing}"


def test_faithful_mode_is_deterministic():
    # fake quantization makes faithful prefill-vs-decode legitimately discontinuous (a tiny
    # difference that crosses an fp4/fp8 rounding boundary moves a whole step), so the
    # continuity bound lives in exact mode; faithful must at least be reproducible
    torch.manual_seed(1)
    x = torch.randint(0, VOCAB, (2, END))
    outs = []
    for _ in range(2):
        model, _, _ = harness.build("faithful", 0)
        _, logits, _ = model(x, 0)
        outs.append(logits)
    assert torch.isfinite(outs[0]).all() and torch.equal(outs[0], outs[1])


def test_reference_bug_stale_index_keys_is_real():
    # As shipped, ref/model.py publishes an owner's index keys only when its compressor
    # completes a group. If DeepSeek fixes this upstream, this test fails: drop the fix then.
    model, args, m = harness.build("exact", 0, fix_reference_bug=False)
    torch.manual_seed(1)
    x = torch.randint(0, VOCAB, (2, PREFILL + 1))
    model(x[:, :PREFILL], 0)
    seen = {}
    ix1 = model.layers[1].attn.indexer  # ratio 2: step PREFILL (even) completes no group
    ix1.register_forward_pre_hook(lambda mod, inp: seen.__setitem__("k", m.shared_attn.index_k))
    model(x[:, PREFILL : PREFILL + 1], PREFILL)
    assert seen["k"] is not ix1.k_cache
    assert seen["k"] is model.layers[3].attn.indexer.k_cache
    unfixed = _compare("exact", fix=False)
    assert max(e for _, e, _ in unfixed) > 0.1


def test_selection_is_actually_sparse_at_these_lengths():
    # non-vacuity: if every compressed position were selected, the indexer could be wrong
    # and the continuity tests would still pass
    model, args, m = harness.build("exact", 0)
    torch.manual_seed(1)
    x = torch.randint(0, VOCAB, (1, END))
    model(x, 0)
    idx = m.shared_attn.topk_idxs  # published by the last index source (ratio 1)
    reachable = END // args.compress_ratios[args.index_source_layers[-1]]
    picked = (idx[0, -1] >= 0).sum().item()
    assert picked == args.index_topk < reachable


def test_one_shot_prefill_is_sensitive_to_history():
    # the comparison must be able to tell a decode that forgot its history from one that did not
    torch.manual_seed(1)
    x = torch.randint(0, VOCAB, (2, END))
    a, _, _ = harness.build("exact", 0)
    b, _, _ = harness.build("exact", 0)
    _, la, _ = a(x, 0)
    y = x.clone()
    y[:, :20] = torch.randint(0, VOCAB, (2, 20))  # change only distant tokens
    _, lb, _ = b(y, 0)
    assert ((la - lb).norm() / la.norm()).item() > 0.1
