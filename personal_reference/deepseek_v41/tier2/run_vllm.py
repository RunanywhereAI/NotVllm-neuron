# SPDX-License-Identifier: Apache-2.0
"""Tier 2: the tiny checkpoint through vLLM in ``VLLM_NEURON_CPU_MODE=1``, against the
exact-mode oracle.

Run on the Linux host, with the interpreter whose editable install is the tree under
test. vLLM inspects model classes in a subprocess, so PYTHONPATH is not enough. On the
trn2 host, call ``/data/venv-fork/bin/python`` directly: sourcing its ``activate``
activates the stock venv.

    rm -rf ~/.cache/vllm/modelinfos
    VLLM_NEURON_CPU_MODE=1 python run_vllm.py TINY_OUT_DIR [TP] [NEW_TOKENS] [ORACLE_DIR]

``ORACLE_DIR`` (default ``TINY_OUT_DIR``) is where the oracle loads its weights from. Point
vLLM at a deliberately perturbed copy and the oracle at the original, and the run must fail.
That shows the tolerance is not vacuous.

``TINY_OUT_DIR`` comes from ``make_tiny.py``. The prefix-hit granularity is the
compressed block, from ``CacheLayout`` at vLLM's cache block size and the served dtype.
There are two phases, with prefix caching on:

1. Sequential. A cold prompt runs, then one sharing its first two compressed blocks,
   which must hit them. The runner takes one prefill per step.
2. Batched. Three prompts go in one ``generate`` call: one hitting the phase-1 prefix
   and two cold ones of other lengths. They prefill one per step, then decode together
   at different positions. The only decode bucket is 4, so every decode step carries a
   padded row.

Hits are asserted from ``num_cached_tokens``.

Greedy, synchronous scheduling (Engram hashes the newest tokens on the host), on-device
sampling off so vLLM reports log-probabilities. The oracle is then teacher-forced on
exactly the tokens vLLM produced: each generated token must be the oracle's argmax
given vLLM's own history, and every reported top-k log-probability must match the
oracle's to ``TOL``.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

ROOT = Path(__file__).resolve().parents[3]
TOL = 1e-3
TOP_K = 8
DECODE_BUCKET = 4


def _compare(ref, oracle_side, label, prompt, out):
    """vLLM's tokens against the oracle's teacher-forced argmax, and vLLM's top-k
    log-probabilities against the oracle's. Returns (ok, worst |dlogprob|)."""
    got = list(out.outputs[0].token_ids)
    lp = oracle_side.teacher_forced_logprobs(ref, prompt + got, len(prompt))
    steps = out.outputs[0].logprobs or []
    assert len(steps) == len(got), (len(steps), len(got))
    argmax = lp[:-1].argmax(-1).tolist()
    top2 = lp[:-1].topk(2, dim=-1).values
    gaps = (top2[:, 0] - top2[:, 1]).tolist()
    worst = 0.0
    for step, entries in enumerate(steps):
        for tok, e in entries.items():
            worst = max(worst, abs(e.logprob - float(lp[step, tok])))
    mism = [(s, got[s], argmax[s], round(gaps[s], 5)) for s in range(len(got)) if got[s] != argmax[s]]
    cached = out.num_cached_tokens or 0
    print(f"{label} (len {len(prompt)}, cached {cached}): tokens {len(got) - len(mism)}/{len(got)} "
          f"match the oracle's teacher-forced argmax; max |dlogprob| {worst:.2e} over top-{TOP_K}; "
          f"min top1-top2 gap {min(gaps):.2e}" + (f"; mismatches (step, vllm, oracle, gap) {mism}" if mism else ""),
          flush=True)
    return not mism and worst < TOL, worst


def main(tiny_dir: str, tp: int = 1, new_tokens: int = 12, oracle_dir: str | None = None) -> int:
    import torch
    from vllm import LLM, SamplingParams

    from vllm_neuron.model.deepseek_v41.cache_layout import CacheLayout

    tiny = Path(tiny_dir)
    args = json.loads((tiny / "reference_args.json").read_text())
    t0 = time.time()
    llm = LLM(
        model=str(tiny / "served"),
        skip_tokenizer_init=True,
        dtype="float32",
        max_model_len=2048,
        max_num_seqs=4,
        block_size=32,                   # the device layout: window block 32
        tensor_parallel_size=tp,
        enable_prefix_caching=True,
        max_num_batched_tokens=1024,     # APC needs segmented prefill: a supported size below max_model_len
        async_scheduling=False,
        enforce_eager=True,
        num_gpu_blocks_override=256,
        limit_mm_per_prompt={"image": 0, "video": 0},
        max_logprobs=TOP_K,
        # one decode bucket of 4: every decode step carries padded rows
        additional_config={"neuron_config": {"on_device_sampling_config": None,
                                             "num_seqs_buckets": [DECODE_BUCKET]}},
    )
    block = llm.llm_engine.vllm_config.cache_config.block_size
    layout = CacheLayout.build(SimpleNamespace(**args), window_block=block, comp_dtype=torch.float32)
    prefix_len = 2 * layout.comp_block
    print(f"engine up in {time.time() - t0:.0f}s; cache block {block}, compressed block "
          f"{layout.comp_block}, shared prefix {prefix_len}, decode bucket {DECODE_BUCKET}", flush=True)

    sys.path.insert(0, str(ROOT))
    from personal_reference.deepseek_v41.tier2 import oracle_side

    sp = SamplingParams(max_tokens=new_tokens, temperature=0.0, logprobs=TOP_K,
                        ignore_eos=True, detokenize=False)
    # phase 1, one request at a time: a cold prompt, then one that hits its prefix
    seq = oracle_side.shared_prefix_prompts(args["vocab_size"], prefix_len)
    seq_outs = [llm.generate([{"prompt_token_ids": p}], sp)[0] for p in seq]
    # phase 2, three requests in one call: they prefill one per step and then decode
    # together at different positions in a bucket of 4 (one padded row); the first hits
    # the phase-1 prefix
    fresh = oracle_side.shared_prefix_prompts(args["vocab_size"], 0, suffix_lens=(150, 77), seed=99)
    batch = [seq[0][:prefix_len] + oracle_side.shared_prefix_prompts(
        args["vocab_size"], 0, suffix_lens=(29,), seed=7)[0], *fresh]
    batch_outs = llm.generate([{"prompt_token_ids": p} for p in batch], sp)
    cached = [o.num_cached_tokens or 0 for o in (*seq_outs, *batch_outs)]
    print(f"num_cached_tokens: sequential {cached[:2]}, batched {cached[2:]}", flush=True)

    ref = oracle_side.load(Path(oracle_dir) if oracle_dir else tiny)
    ok = cached[0] == 0 and cached[1] >= prefix_len and cached[2] >= prefix_len
    worst_all = 0.0
    labels = ["sequential 0", "sequential 1 (prefix hit)", "batched 0 (prefix hit)", "batched 1", "batched 2"]
    for label, p, out in zip(labels, (*seq, *batch), (*seq_outs, *batch_outs)):
        good, worst = _compare(ref, oracle_side, label, p, out)
        ok &= good
        worst_all = max(worst_all, worst)
    print(f"TP={tp}: {'PASS' if ok else 'FAIL'}; worst |dlogprob| {worst_all:.2e}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1,
                  int(sys.argv[3]) if len(sys.argv) > 3 else 12,
                  sys.argv[4] if len(sys.argv) > 4 else None))
