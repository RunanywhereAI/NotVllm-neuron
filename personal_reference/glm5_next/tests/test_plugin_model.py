# SPDX-License-Identifier: Apache-2.0
"""The plugin's whole GLM-5.3-Flash decoder against the oracle -- Tier 1.

Run: python3 -m pytest personal_reference/glm5_next/tests/test_plugin_model.py -v -s

``vllm_neuron/model/glm5_next/model.py`` runs exactly as written (``plugin_harness``
imports it without executing the vLLM-dependent package ``__init__`` files), on a tiny
config that keeps every structural feature, through a fake of the runner's shared
page-major cache. The oracle is ``reference.py``'s ``FlashTextModel`` carrying the same
weights, run one-shot over each whole sequence.

What would make these pass with the model wrong, and what stops it:

* **Prefill-only validation.** A layer with carried state is not validated until
  prefill-then-decode equals one-shot prefill -- the MLA decode path once passed every
  prefill test while a decode token attended to itself alone. Every decode logit here
  is compared with the one-shot oracle at that position.
* **Batch 1.** Cache-group hazards only appear with padded rows, so decode runs at a
  batch bucket larger than the live batch from the first test, with padded rows
  pointing at a live request's blocks.
* **Unreachable behaviour.** Sequences cross the indexer's dense-exact ceiling (11 at
  this config) and pool boundaries and a 32-token block boundary; MLP weights are
  scaled so the SwiGLU clamp engages; mHC's base is random so its mixing is neither
  identity nor uniform. Each is asserted to be live, not assumed.
* **A forgiving cache.** Every byte the model did not write is NaN, so a read that is
  multiplied away rather than selected away poisons the logits.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from glm5_next import reference as R  # noqa: E402
from glm5_next.tests import plugin_harness as H  # noqa: E402

CFG = H.import_plugin("vllm_neuron.model.glm5_next.config")
M = H.import_plugin("vllm_neuron.model.glm5_next.model")
MLA = H.import_plugin("vllm_neuron.model.glm5_next.mla")

BLOCK = 32
TOL = dict(rtol=2e-4, atol=2e-4)


def _pair(seed=0, **over):
    text = CFG.Glm5NextTextConfig.from_hf(H.hf_text_config(**over))
    torch.manual_seed(seed)
    oracle = R.FlashTextModel(H.oracle_cfg(text, R), vocab=text.vocab_size).eval()
    H.randomize_(oracle, seed=seed)
    plugin = M.Glm5NextForCausalLM(CFG.Glm5NextConfig(text_config=text)).eval()
    plugin.load_state_dict(H.plugin_state_from_oracle(oracle.state_dict()), strict=True)
    return plugin, oracle


def _ids(n, seed):
    return torch.randint(1, H.TINY_HF["vocab_size"], (n,),
                         generator=torch.Generator().manual_seed(seed)).tolist()


def _oracle_logits(oracle, ids):
    with torch.no_grad():
        return oracle(torch.tensor([ids]))[0]


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


# ----------------------------------------------------------------------- prefill
@pytest.mark.parametrize("n,bucket", [(5, 8), (11, 16), (12, 16), (23, 32), (40, 64)])
def test_prefill_matches_the_oracle(n, bucket):
    """Every real position's logits, including padded buckets and lengths both under
    and over the dense-exact ceiling (11)."""
    plugin, oracle = _pair()
    run = H.FakeRunner(plugin, BLOCK, num_blocks=16)
    ids = _ids(n, seed=n)
    with torch.no_grad():
        got = run.prefill(0, ids, bucket)
    want = _oracle_logits(oracle, ids)
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, want, **TOL)
    print(f"\n  n={n:3d} bucket={bucket:3d}  max rel {_rel(got, want):.2e}")


# ---------------------------------------------- prefill, then decode, batch > 1
def _continuity(seed, lengths, steps, batch_bucket, nb=None):
    plugin, oracle = _pair(seed=seed)
    run = H.FakeRunner(plugin, BLOCK, num_blocks=40)
    seqs = [_ids(n + steps, seed=100 * seed + r) for r, n in enumerate(lengths)]
    worst = 0.0
    with torch.no_grad():
        for r, n in enumerate(lengths):          # the plugin prefills one request at a time
            got = run.prefill(r, seqs[r][:n], bucket=-(-n // 16) * 16)
            want = _oracle_logits(oracle, seqs[r][:n])
            torch.testing.assert_close(got, want, **TOL)
        wants = [_oracle_logits(oracle, s) for s in seqs]
        for step in range(steps):
            rows = [(r, seqs[r][n + step], n + step) for r, n in enumerate(lengths)]
            got = run.decode(rows, batch_bucket, nb=nb)
            assert torch.isfinite(got).all(), f"step {step}: non-finite logits (padded rows too)"
            for r, n in enumerate(lengths):
                want = wants[r][n + step]
                torch.testing.assert_close(got[r], want, **TOL,
                                           msg=lambda m: f"req {r} step {step}: {m}")
                worst = max(worst, _rel(got[r], want))
    return worst


@pytest.mark.parametrize("seed", [0, 1])
def test_prefill_then_decode_equals_one_shot_prefill_at_batch_3_of_5(seed):
    """Three live requests and two padded rows (one pointing at a live request's
    blocks, one at a never-written page). Lengths chosen so decode crosses pool
    completions, the dense-exact ceiling (req 0 starts below it at 9) and a 32-token
    block boundary (req 2: 29 -> 42)."""
    worst = _continuity(seed, lengths=(9, 17, 29), steps=13, batch_bucket=5)
    print(f"\n  seed {seed}: worst decode rel err {worst:.2e}")


def test_decode_with_a_trimmed_block_table():
    """Decode-context bucketing hands the model a block table narrower than
    ``max_model_len`` -- here exactly the 2 blocks the longest request needs."""
    _continuity(0, lengths=(20, 30), steps=6, batch_bucket=3, nb=2)


# ---------------------------------------------------- the behaviour is reachable
def test_the_regimes_under_test_are_actually_reached(monkeypatch):
    """Non-vacuity: the clamp engages, the indexer drops tokens, mHC mixes."""
    plugin, _ = _pair()
    captured: dict[str, list] = {}

    def rec(name, t):
        captured.setdefault(name, []).append(t.detach().clone())

    monkeypatch.setattr(MLA, "_capture_tensor", rec)
    monkeypatch.setattr(M, "_capture_tensor", rec)
    clamped = []

    def hook(mod, inp, out):
        clamped.append((out.abs() > mod.limit if hasattr(mod, "limit") else out).float().mean())

    run = H.FakeRunner(plugin, BLOCK, num_blocks=8)
    ids = _ids(30, seed=7)
    swiglu = M._clamped_swiglu
    seen = {"over": 0, "total": 0}

    def counting(gate, up, limit):
        seen["over"] += int((gate > limit).sum() + (up.abs() > limit).sum())
        seen["total"] += gate.numel() + up.numel()
        return swiglu(gate, up, limit)

    monkeypatch.setattr(M, "_clamped_swiglu", counting)
    with torch.no_grad():
        run.prefill(0, ids, 32)
    frac = seen["over"] / seen["total"]
    print(f"\n  SwiGLU inputs beyond the clamp: {100 * frac:.1f}%")
    assert frac > 0.02, "the clamp never engages; a missing clamp would pass"

    idx = [t for k, v in captured.items() if k.endswith("indexer.topk_indices") for t in v]
    selected = (idx[0][0] >= 0).sum(-1)                      # per query row
    print(f"  tokens selected at row 29: {selected[29].item()} of 30")
    assert selected[29] < 30 and selected[29] <= 8 + 3, "the indexer selected everything"

    comb = [t for k, v in captured.items() if k.endswith("attn_hc.comb") for t in v][0]
    eye = torch.eye(4)
    off = (comb * (1 - eye)).sum((-1, -2)) / comb.sum((-1, -2))
    print(f"  mHC comb off-diagonal mass: {off.mean():.3f}")
    assert off.mean() > 0.2 and (comb - 0.25).abs().max() > 0.05, \
        "comb is near identity or near uniform; mixing errors would hide"


def test_hostile_bytes_are_live_in_the_cache():
    """Non-vacuity for the NaN fill: unwritten pages really are NaN, so the finiteness
    of every logit above means the model selected them away."""
    plugin, _ = _pair()
    run = H.FakeRunner(plugin, BLOCK, num_blocks=8)
    pages = next(iter(run.kv.values()))[0]
    assert torch.isnan(pages[1]).all() and (pages[run.zero_page] == 0).all()


def _hc_pair(prior, seed=0):
    """A plugin and an oracle hyper-connection carrying the same weights.

    ``prior="real"``: the checkpoint's comb prior -- 0 on the diagonal, exactly -8.0 off
    it (measured from layer 0, see test_mhc.py). ``prior="random"``: base N(0, 0.5).
    ``fn`` is scaled so the comb logits have std ~2 whatever the hidden size; at the
    real prior that leaves Sinkhorn unconverged at 20 iterations (rows off by ~3e-2,
    the regime AGENTS.md records for the real weights), while at small logits it
    converges to ~1e-6 and the iteration count stops being observable at all.
    """
    text = CFG.Glm5NextTextConfig.from_hf(H.hf_text_config())
    o = R.HyperConnection(H.oracle_cfg(text, R))
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        o.fn.copy_(torch.randn(o.fn.shape, generator=g) * 2.0 / o.fn.shape[1] ** 0.5)
        base = torch.randn(o.base.shape, generator=g) * 0.5
        if prior == "real":
            base[2 * 4:] = torch.full((4, 4), -8.0).fill_diagonal_(0.0).flatten()
        o.base.copy_(base)
        o.scale.copy_(torch.tensor([0.5, 0.25, 1.0]))
    p = M.Glm5NextHyperConnection(text.hc_mult, text.hidden_size, text.hc_sinkhorn_iters,
                                  text.hc_eps, text.rms_norm_eps, "t")
    p.load_state_dict(o.state_dict())
    return p, o


@pytest.mark.parametrize("prior", ["real", "random"])
def test_mhc_matches_the_oracle_component_wise(prior):
    """mHC is validated at the component -- the captured ``pre``/``post``/``comb`` --
    not end to end: at the real near-identity prior cross-stream mixing is ~1e-3 of the
    output, and in this file's model tests one extra Sinkhorn iteration moved the
    logits by only 2e-6, below any sane end-to-end tolerance."""
    p, o = _hc_pair(prior)
    streams = torch.randn(3, 7, 4, H.TINY_HF["hidden_size"],
                          generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        o_post, o_comb, o_col = o(streams)
        g_post, g_comb, g_col = p(streams.flatten(0, 1))
    torch.testing.assert_close(g_comb, o_comb.flatten(0, 1), rtol=0, atol=1e-6)
    torch.testing.assert_close(g_post, o_post.flatten(0, 1), rtol=0, atol=1e-6)
    torch.testing.assert_close(g_col, o_col.flatten(0, 1), rtol=0, atol=1e-5)


def test_the_component_check_can_tell_20_sinkhorn_iterations_from_21():
    """Non-vacuity for the above: at the real prior, where Sinkhorn is unconverged at
    20, one more iteration must move ``comb`` well past the 1e-6 the comparison allows."""
    p, _ = _hc_pair("real")
    streams = torch.randn(21, 4, H.TINY_HF["hidden_size"],
                          generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        _, comb20, _ = p(streams)
        p.iters = 21
        _, comb21, _ = p(streams)
    gap = (comb21 - comb20).abs().max().item()
    row_err = (comb20.sum(-1) - 1).abs().max().item()
    print(f"\n  rows off by {row_err:.2e} at 20 iterations; 21 moves comb by {gap:.2e}")
    assert gap > 1e-4


# ------------------------------------------------------------------------- bf16
def test_bf16_serving_path_runs_finite_and_close_to_fp32():
    """The serving dtype: bf16 MLA pages and fp32 KDA pages viewed over one buffer, NaN
    (0xFFFF) wherever unwritten, prefill then decode at batch 2 of 3.

    No oracle comparison is possible in bf16 (the oracle is fp32 by design), and
    discrete choices -- expert routing, indexer selection -- legitimately flip under
    bf16, so this compares the plugin with ITSELF in fp32 at a config with neither:
    every expert routed and the indexer above the sequence length. Measured 2026-10-06:
    ~4% mean-relative at the logits after 8 layers, growing ~0.5%/layer, KDA the largest
    contributor. That is a measurement on random weights, not a budget. The bound below
    catches gross bf16 defects (a page read at the wrong dtype, a missing cast) only.
    """
    over = dict(index_topk=64, num_experts_per_tok=8)
    lengths, steps = (13, 21), 6
    outs = {}
    for dtype in (torch.float32, torch.bfloat16):
        plugin, _ = _pair(seed=0, **over)
        plugin.set_dtype(dtype)
        run = H.FakeRunner(plugin, 64, num_blocks=16, dtype=dtype)
        seqs = [_ids(n + steps, seed=40 + r) for r, n in enumerate(lengths)]
        got = []
        with torch.no_grad():
            for r, n in enumerate(lengths):
                got.append(run.prefill(r, seqs[r][:n], bucket=32).float())
            for step in range(steps):
                rows = [(r, seqs[r][n + step], n + step) for r, n in enumerate(lengths)]
                d = run.decode(rows, 3).float()
                assert torch.isfinite(d).all(), f"{dtype} step {step}: non-finite"
                got.append(d[: len(lengths)])
        outs[dtype] = torch.cat(got)
    a, b = outs[torch.float32], outs[torch.bfloat16]
    mean_rel = ((a - b).abs().mean() / a.abs().mean()).item()
    print(f"\n  bf16 vs fp32 (plugin): mean rel {mean_rel:.2e}")
    assert mean_rel < 0.15
