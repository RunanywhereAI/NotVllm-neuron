# SPDX-License-Identifier: Apache-2.0
"""The tiny checkpoint through transformers' own ``Glm5NextForConditionalGeneration``.

    <python with torch + transformers>=5.17> check_transformers.py CKPT_DIR

A second, independent reference for ``make_tiny.py``'s ``oracle_reference.json``:
transformers loads the checkpoint with ITS key mapping (which also checks the
checkpoint writer's names and layouts) and runs its own composition -- decoder layer,
both hyper-connections, hc_head, the indexer -- one-shot over each growing sequence.
Greedy tokens must equal the oracle's and the chosen-token log-probabilities agree to
fp32 noise. transformers falls back to torch for KDA when the ``fla`` hub kernels are
unavailable, which on a CPU host they are.
"""
from __future__ import annotations

import json
import pathlib
import sys

import torch


def main(ckpt: str) -> int:
    from transformers import AutoConfig, Glm5NextForConditionalGeneration

    ref = json.loads((pathlib.Path(ckpt) / "oracle_reference.json").read_text())
    cfg = AutoConfig.from_pretrained(ckpt)
    model, info = Glm5NextForConditionalGeneration.from_pretrained(
        ckpt, config=cfg, dtype=torch.float32, output_loading_info=True)
    model.eval()
    text_missing = [k for k in info["missing_keys"] if "visual" not in k]
    print(f"missing (non-vision): {text_missing}; unexpected: {info['unexpected_keys']}")
    bad = bool(text_missing or info["unexpected_keys"])
    worst = 0.0
    for p in ref["prompts"]:
        seq = list(p["prompt"])
        toks, diffs = [], []
        with torch.no_grad():
            for want_tok, want_lp in zip(p["greedy"], p["logprob"]):
                logits = model(input_ids=torch.tensor([seq])).logits[0, -1].float()
                lp = torch.log_softmax(logits, -1)
                tok = int(lp.argmax())
                toks.append(tok)
                diffs.append(abs(float(lp[want_tok]) - want_lp))
                seq.append(want_tok)                     # teacher-force the oracle's path
        match = sum(a == b for a, b in zip(toks, p["greedy"]))
        worst = max(worst, max(diffs))
        bad |= match != len(p["greedy"])
        print(f"prompt len {len(p['prompt']):3d}: tokens {match}/{len(p['greedy'])}, "
              f"max |dlogprob| {max(diffs):.2e}")
    print(f"worst |dlogprob| {worst:.2e}")
    return 1 if bad or worst > 1e-3 else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
