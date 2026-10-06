# SPDX-License-Identifier: Apache-2.0
"""The tiny checkpoint through the PLUGIN model, without vLLM, vs ``oracle_reference.json``.

    python check_plugin.py CKPT_DIR          # torch + safetensors only

``Glm5NextForCausalLM.load_weights`` reads the checkpoint, and greedy decoding runs
through ``plugin_harness.FakeRunner`` -- shared page-major buffers, NaN-filled hostile
pages, one prefill per request, then batched decode with padded rows (live + 1 in a
bucket of live + 1). The laptop-side counterpart of ``run_vllm.py``.
"""
from __future__ import annotations

import json
import pathlib
import socket
import sys
from types import SimpleNamespace

import torch

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
from glm5_next.tests import plugin_harness as H  # noqa: E402


def main(ckpt: str) -> int:
    import torch.distributed as dist

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)
    CFG = H.import_plugin("vllm_neuron.model.glm5_next.config")
    M = H.import_plugin("vllm_neuron.model.glm5_next.model")
    tc = json.loads((pathlib.Path(ckpt) / "config.json").read_text())["text_config"]
    tc["dtype"] = torch.float32
    text = CFG.Glm5NextTextConfig.from_hf(SimpleNamespace(**tc))
    model = M.Glm5NextForCausalLM(CFG.Glm5NextConfig(text_config=text)).eval()
    model.load_weights(ckpt, torch.device("cpu"))
    ref = json.loads((pathlib.Path(ckpt) / "oracle_reference.json").read_text())
    run = H.FakeRunner(model, H.aligned_block_size(model), num_blocks=64)
    seqs = [list(p["prompt"]) for p in ref["prompts"]]
    lps = [[] for _ in seqs]
    with torch.no_grad():
        for r, s in enumerate(seqs):
            lp = torch.log_softmax(run.prefill(r, s, bucket=-(-len(s) // 16) * 16)[-1], -1)
            lps[r].append(float(lp.max()))
            s.append(int(lp.argmax()))
        for _ in range(ref["new_tokens"] - 1):
            rows = [(r, s[-1], len(s) - 1) for r, s in enumerate(seqs)]
            out = run.decode(rows, len(seqs) + 1)
            assert torch.isfinite(out).all()
            for r, s in enumerate(seqs):
                lp = torch.log_softmax(out[r], -1)
                lps[r].append(float(lp.max()))
                s.append(int(lp.argmax()))
    bad, worst = False, 0.0
    for p, s, lp in zip(ref["prompts"], seqs, lps):
        got = s[len(p["prompt"]):]
        match = sum(a == b for a, b in zip(got, p["greedy"]))
        d = max(abs(a - b) for a, b in zip(lp, p["logprob"]))
        worst = max(worst, d)
        bad |= match != len(p["greedy"])
        print(f"prompt len {len(p['prompt']):3d}: tokens {match}/{len(p['greedy'])}, "
              f"max |dlogprob| {d:.2e}")
    print(f"worst |dlogprob| {worst:.2e}")
    return 1 if bad or worst > 1e-3 else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
