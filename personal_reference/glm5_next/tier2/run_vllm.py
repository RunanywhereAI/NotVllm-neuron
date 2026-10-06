# SPDX-License-Identifier: Apache-2.0
"""Tier 2: the tiny checkpoint through vLLM in ``VLLM_NEURON_CPU_MODE=1``, vs the oracle.

Run on a Linux host with vLLM 0.24.0 and the plugin installed EDITABLE from the tree
under test (vLLM inspects model classes in a subprocess, so PYTHONPATH is not enough):

    rm -rf ~/.cache/vllm/modelinfos      # stale class info survives edits otherwise
    VLLM_NEURON_CPU_MODE=1 python run_vllm.py CKPT_DIR [max_num_seqs]

``CKPT_DIR`` comes from ``make_tiny.py``. All prompts are submitted at once with
``max_num_seqs > 1``, so decode runs batched with padded rows -- the cache-group
hazards only appear there. On-device sampling is disabled so the runner returns
logits and vLLM can report each sampled token's log-probability; greedy tokens must
equal the oracle's and the log-probabilities must agree to fp32 noise.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import sys

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")


def main(ckpt: str, max_num_seqs: int) -> int:
    from vllm import LLM, SamplingParams

    ref = json.loads((pathlib.Path(ckpt) / "oracle_reference.json").read_text())
    llm = LLM(
        model=ckpt,
        skip_tokenizer_init=True,
        dtype="float32",
        max_model_len=128,
        max_num_seqs=max_num_seqs,
        tensor_parallel_size=1,
        enable_prefix_caching=False,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
        max_logprobs=1,
        additional_config={"neuron_config": {"on_device_sampling_config": None}},
    )
    sp = SamplingParams(max_tokens=ref["new_tokens"], temperature=0.0, logprobs=1,
                        ignore_eos=True, detokenize=False)
    outs = llm.generate([{"prompt_token_ids": p["prompt"]} for p in ref["prompts"]], sp)
    tok_ok = tok_total = 0
    worst_lp = 0.0
    for p, out in zip(ref["prompts"], outs):
        got = list(out.outputs[0].token_ids)
        match = sum(a == b for a, b in zip(got, p["greedy"]))
        prefix = next((i for i, (a, b) in enumerate(zip(got, p["greedy"])) if a != b), len(got))
        tok_ok += match
        tok_total += len(p["greedy"])
        lps = out.outputs[0].logprobs or []
        diffs = []
        for step, (tok, want) in enumerate(zip(got[:prefix], p["logprob"][:prefix])):
            entry = lps[step].get(tok) if step < len(lps) else None
            if entry is not None:
                diffs.append(abs(entry.logprob - want))
        d = max(diffs) if diffs else math.nan
        worst_lp = max(worst_lp, d) if diffs else worst_lp
        print(f"prompt len {len(p['prompt']):3d}: tokens {match}/{len(p['greedy'])}, "
              f"exact prefix {prefix}, max |dlogprob| {d:.2e} over {len(diffs)} steps")
    print(f"TOTAL tokens {tok_ok}/{tok_total}; worst |dlogprob| {worst_lp:.2e}")
    return 0 if tok_ok == tok_total and worst_lp < 1e-3 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4))
