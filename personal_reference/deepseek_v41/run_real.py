"""Run DeepSeek-V4.1-Flash's reference on CPU with the real weights, through the oracle.

    python -m personal_reference.deepseek_v41.run_real \
        --ckpt /data/models/dsv41-ref-mp1 --hf /data/models/DeepSeek-V4.1-Flash \
        --mode faithful --max-new-tokens 32 --prompt "..."

``--ckpt`` is the output of ``ref/convert.py --model-parallel 1 --expert-dtype fp4``;
``--hf`` is the HF snapshot, needed only for its ``encoding/`` chat formatter. Greedy.
Needs roughly the checkpoint size in host RAM (~512 GB).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_model

from . import oracle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--mode", default="faithful", choices=["faithful", "exact"])
    ap.add_argument("--prompt", default="What is the capital of France? Answer in one sentence.")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--dump", default="", help="write prompt ids, generated ids and per-step logits here")
    a = ap.parse_args()

    if a.threads:
        torch.set_num_threads(a.threads)
    sys.path.insert(0, str(Path(a.hf) / "encoding"))  # encoding/encoding.py, as run.sh arranges
    from encoding import encode_messages  # noqa: E402
    from transformers import AutoTokenizer  # noqa: E402

    m = oracle.load_reference(a.mode)
    with open(oracle.REF_DIR / "config.json") as f:
        args = m.ModelArgs(**json.load(f))
    args.max_batch_size, args.max_seq_len, args.temperature = 1, a.max_seq_len, 0.0
    tokenizer = AutoTokenizer.from_pretrained(a.ckpt)

    t0 = time.time()
    model = oracle.build(m, args, init=None, tokenizer=tokenizer)
    print(f"built in {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    missing, unexpected = load_model(model, str(Path(a.ckpt) / "model0-mp1.safetensors"), strict=False)
    print(f"loaded in {time.time() - t0:.0f}s; missing {len(missing)} unexpected {len(unexpected)}", flush=True)
    if missing or unexpected:
        print("missing:", missing[:10], "unexpected:", unexpected[:10], flush=True)

    text = encode_messages([{"role": "user", "content": a.prompt}], thinking_mode="chat")
    ids = tokenizer.encode(text)
    print(f"prompt: {len(ids)} tokens", flush=True)
    tokens = torch.tensor([ids])
    out, logits_seq = [], []
    t0 = time.time()
    nxt, logits, _ = model(tokens, 0)
    print(f"prefill {time.time() - t0:.1f}s", flush=True)
    pos = len(ids)
    for _ in range(a.max_new_tokens):
        tok = int(nxt[0])
        out.append(tok)
        logits_seq.append(logits[0].float())
        if tok == tokenizer.eos_token_id:
            break
        t1 = time.time()
        nxt, logits, _ = model(torch.tensor([[tok]]), pos)
        pos += 1
        print(f"  step {len(out)}: {time.time() - t1:.1f}s  {tokenizer.decode(out)!r}", flush=True)
    print("COMPLETION:", tokenizer.decode(out), flush=True)
    if a.dump:
        torch.save({"prompt": ids, "generated": out, "logits": torch.stack(logits_seq)}, a.dump)


if __name__ == "__main__":
    main()
