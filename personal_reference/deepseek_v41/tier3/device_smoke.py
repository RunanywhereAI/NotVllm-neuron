"""Tier 3 smoke: the tiny model on one NeuronCore, compiled, against the same model on CPU.

    NEURON_RT_VISIBLE_CORES=0 python -m personal_reference.deepseek_v41.tier3.device_smoke

Runs the test_plugin_model scenario (padded prefill, a prefix hit, interleaved batched
decode with pad rows) through ``torch.compile(backend="neuron_libtorch")``, one graph
per shape, and reports the relative logit error against the identical CPU run. Its job
is to find out which ops neuronx-cc refuses, before vLLM is in the way.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from personal_reference.deepseek_v41.tests import harness  # noqa: E402
from personal_reference.deepseek_v41.tests import plugin_harness as ph  # noqa: E402


def scenario(run, A, B, steps):
    la = [run.prefill("a", A[:37], 0, bucket=48)]
    run.share_prefix("a", "b", 32)
    lb = [run.prefill("b", B[32:45], 32, bucket=16)]
    for pa, pb in zip(range(37, 37 + steps), range(45, 45 + steps)):
        t0 = time.time()
        out = run.decode(["a", "b"], [A[pa], B[pb]], [pa, pb], pad_rows=2)
        print(f"  decode step {pa - 37}: {time.time() - t0:.2f}s", flush=True)
        la.append(out[:1])
        lb.append(out[1:])
    return torch.cat(la), torch.cat(lb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--steps", type=int, default=6)
    a = ap.parse_args()
    dtype = getattr(torch, a.dtype)

    import libtorch_neuronx_lite  # noqa: F401  registers the device and backend
    dev = torch.device("neuron:0")

    torch.manual_seed(1)
    ref, args, _ = harness.build("exact", 0)
    A = torch.randint(0, args.vocab_size, (70,)).tolist()
    B = A[:32] + torch.randint(0, args.vocab_size, (38,)).tolist()

    cpu_model = ph.plugin_from_reference(ref, args, block_size=8, dtype=dtype)
    with torch.no_grad():
        ca, cb = scenario(ph.FakeRunner(cpu_model), A, B, a.steps)

    dev_model = ph.plugin_from_reference(ref, args, block_size=8, dtype=dtype).to(dev)
    compiled = torch.compile(lambda i, p, m, e: dev_model.hidden_states(i, p, m, engram_ids=e),
                             backend="neuron_libtorch", fullgraph=True, dynamic=False)
    t0 = time.time()
    with torch.no_grad():
        run = ph.FakeRunner(dev_model, device=dev, forward=compiled)
        da, db = scenario(run, A, B, a.steps)
    print(f"device scenario incl. compiles: {time.time() - t0:.0f}s")

    def rel(x, y):
        return ((x - y).norm(dim=-1) / y.norm(dim=-1)).max().item()

    print(f"A: {rel(da, ca):.2e}   B: {rel(db, cb):.2e}   finite: {torch.isfinite(da).all().item()}")


if __name__ == "__main__":
    main()
