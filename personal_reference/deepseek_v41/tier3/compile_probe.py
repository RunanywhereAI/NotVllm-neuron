"""Compile-only probe of DeepSeek-V4.1's prefill and decode graphs at REAL dims, no device.

    NEURON_LIBTORCH_CPU_COMPILE=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2 PYTHONPATH=/data/dsv41-wt \
        python compile_probe.py OUT_DIR [prefill|decode ...]

A truncated model, at the server's coordinates:
- real dim, head and expert sizes;
- TP=64 with one query head per rank, EP=64 with 6 local experts;
- seven layers, one of every kind:

  | layer | kind |
  |---|---|
  | 0 | window only |
  | 1 | Engram |
  | 2, 3, 4 | ratio-2 kv+index sources |
  | 5 | ratio-1 candidate source |
  | 6 | ratio-1 index layer using candidates |

The window layer count is reported to CacheLayout as the real 40, so the page layout is
the server's exactly: 50 window fields, a compressed block of 1024 tokens, window pages
(402, 819200) fp32 and compressed pages (402, 1638400) bf16. The input dtypes are taken
from the server's own example_inputs.txt.

Each graph is captured with the runner's ``neuron_libtorch_graph_capture`` backend (HLO
only), then compiled with neuronx-cc using the runner's flags plus ``--lnc 2``. The first
error per graph is printed.
"""
import dataclasses
import glob
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("NEURON_LIBTORCH_CPU_COMPILE", "1")
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

import torch  # noqa: E402

import libtorch_neuronx_lite  # noqa: E402,F401  registers the backends

from vllm_neuron.model.deepseek_v41 import cache_layout as CL  # noqa: E402
from vllm_neuron.model.deepseek_v41 import model as M  # noqa: E402
from vllm_neuron.model.deepseek_v41.config import DeepseekV41TextArgs  # noqa: E402

OUT = os.path.abspath(sys.argv[1])
COLLECTIVES = "--collectives" in sys.argv
WHICH = [a for a in sys.argv[2:] if not a.startswith("--")] or ["prefill", "decode"]
TP, BLOCK, MAX_LEN, NUM_BLOCKS, T_PREFILL, N_DECODE = 64, 32, 8192, 400, 1024, 16
FLAGS = ["--framework", "XLA", "--target", "trn2", "--lnc", "2", "--auto-cast=none", "--verbose=35", "-O1",
         "--internal-hlo2tensorizer-options=--modular-flow-mac-threshold=10 --experimental-unsafe-fp8e4m3fn-as-fp8e4m3",
         "--internal-backend-options=--enable-verifier=false --enable-nested-dynamic-loop"]
META = torch.device("meta")


class StubGroup:
    """Shapes only: all_reduce is identity, all_gather tiles ``tp`` times."""

    def all_reduce(self, t):
        return t

    def all_gather(self, t, dim=-1):
        return torch.cat([t] * TP, dim=dim)


class FuncolGroup:
    """Real collective ops in the graph: ``_c10d_functional.all_reduce`` /
    ``all_gather_into_tensor`` over a 64-rank fake process group, the ops the server's
    graph contains. The plugin's XLA impls lower them to HLO collectives with replica
    groups resolved from the group."""

    def __init__(self):
        import torch.distributed as dist
        from torch.testing._internal.distributed.fake_pg import FakeStore

        if not dist.is_initialized():
            dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=TP)
        self.pg = dist.group.WORLD
        # the capture backend names per-rank workdirs from vLLM's TP group, which a bare
        # fake process group does not create; one workdir is enough here
        from libtorch_neuronx_lite.compile import capture_backend as cb
        orig = cb.setup_workdir_common
        cb.setup_workdir_common = lambda *a, per_rank=True, **k: orig(*a, per_rank=False, **k)

    def all_reduce(self, t):
        import torch.distributed._functional_collectives as funcol
        return funcol.all_reduce(t, "sum", self.pg)

    def all_gather(self, t, dim=-1):
        import torch.distributed._functional_collectives as funcol
        return funcol.all_gather_tensor(t, gather_dim=dim, group=self.pg)


def build_model():
    full = DeepseekV41TextArgs.from_hf_config(json.load(open("/data/dsv41-served/config.json")))
    args = dataclasses.replace(
        full, n_layers=7, n_mtp_layers=0, compress_ratios=(0, 0, 2, 2, 2, 1, 1),
        kv_source_layers=(2, 3, 4, 5), index_source_layers=(2, 3, 4, 5, 6), candidate_source_layer=5,
        engram_layer_ids=(1,), engram_num_embeddings=(full.engram_num_embeddings[0],))
    real_build = CL.CacheLayout.build.__func__

    def build(cls, a, window_block, comp_dtype, max_k=4096):
        # the server's layout: CacheLayout counts one window field per backbone layer
        proxy = SimpleNamespace(**{f.name: getattr(a, f.name) for f in dataclasses.fields(a)})
        proxy.n_layers = full.n_layers
        return real_build(cls, proxy, window_block, comp_dtype, max_k)

    CL.CacheLayout.build = classmethod(build)
    par = M.Parallel(tp=TP, rank=0, ep=TP, ep_rank=0, etp=1, etp_rank=0, group=FuncolGroup() if COLLECTIVES else StubGroup())
    with torch.device("meta"):
        model = M.DeepseekV41Model(args, block_size=BLOCK, cache_dtype=torch.bfloat16, par=par)
    model.set_dtype(torch.bfloat16)
    model.to(META)            # host-built buffers (the RoPE table) follow the model, as load_weights does
    model.attach_sampler(None)
    lay = model.layout
    print(f"layout: n_fields {lay.n_fields}, comp_block {lay.comp_block}, window page "
          f"{lay.window_page_elems}, comp page {lay.comp_page_elems}", flush=True)
    pages = NUM_BLOCKS + 2
    model.bind_kv_cache({
        CL.WINDOW_LAYER: [torch.empty(pages, lay.window_page_elems, dtype=torch.float32, device=META)],
        CL.COMPRESSED_LAYER: [torch.empty(pages, lay.comp_page_elems, dtype=torch.bfloat16, device=META)],
    })
    return model, args


def ints(shape, dtype):
    return torch.zeros(shape, dtype=dtype, device=META)


def prefill_inputs(model, args):
    T, cb = T_PREFILL, model.layout.comp_block
    md = {
        CL.WINDOW_LAYER: dict(block_table_tensor=ints((1, MAX_LEN // BLOCK), torch.int32),
                              slot_mapping=ints((T,), torch.int64), max_query_len=T,
                              decode_token_threshold=1, block_size=BLOCK),
        CL.COMPRESSED_LAYER: dict(block_table_tensor=ints((1, MAX_LEN // cb), torch.int32),
                                  slot_mapping=ints((T,), torch.int64), max_query_len=T,
                                  decode_token_threshold=1, block_size=cb),
    }
    return dict(input_ids=ints((T,), torch.int32), positions=ints((T,), torch.int64), attn_metadata=md,
                rank=ints((), torch.int32), engram_ids=ints((T, 1, model.engram_cols()), torch.int32),
                sampling_positions=ints((1,), torch.int64))


def decode_inputs(model, args):
    n, cb = N_DECODE, model.layout.comp_block
    win = model.layout.window_size
    nbw = min(((win // BLOCK + 1) + (128 // BLOCK) - 1) // (128 // BLOCK) * (128 // BLOCK), MAX_LEN // BLOCK)
    md = {
        CL.WINDOW_LAYER: dict(block_table_tensor=ints((n, nbw), torch.int32),
                              slot_mapping=ints((n,), torch.int64), max_query_len=1,
                              decode_token_threshold=1, block_size=BLOCK,
                              swa_kv_pos_offset=ints((n,), torch.int32)),
        CL.COMPRESSED_LAYER: dict(block_table_tensor=ints((n, MAX_LEN // cb), torch.int32),
                                  slot_mapping=ints((n,), torch.int64), max_query_len=1,
                                  decode_token_threshold=1, block_size=cb),
    }
    print(f"decode: window table width {nbw}", flush=True)
    return dict(input_ids=ints((n,), torch.int32), positions=ints((n,), torch.int64), attn_metadata=md,
                rank=ints((), torch.int32), engram_ids=ints((n, 1, model.engram_cols()), torch.int32),
                sampling_positions=ints((n,), torch.int64))


def capture(model, name, kwargs):
    workdir = os.path.join(OUT, name)
    os.makedirs(workdir, exist_ok=True)
    before = set(glob.glob(f"{workdir}/**/graph.hlo", recursive=True))
    compiled = torch.compile(model, backend="neuron_libtorch_graph_capture", fullgraph=True,
                             dynamic=False, options={"alias_meta_to_neuron": True,
                                                     "compiler_workdir": workdir})
    t0 = time.time()
    from libtorch_neuronx_lite.compile.capture_backend import CaptureComplete
    try:
        with torch.no_grad():
            compiled(**kwargs)
    except CaptureComplete:       # the capture backend's normal exit: HLO written, nothing run
        pass
    hlos = sorted(set(glob.glob(f"{workdir}/**/graph.hlo", recursive=True)) - before)
    print(f"{name}: captured {len(hlos)} graph(s) in {time.time() - t0:.0f}s: {hlos}", flush=True)
    torch._dynamo.reset()
    return hlos


def neuronx_cc(name, hlo):
    d = os.path.dirname(hlo)
    t0 = time.time()
    r = subprocess.run(["/data/venv-fork/bin/neuronx-cc", "compile", hlo, *FLAGS, "--output", f"{d}/graph.neff",
                        "--logfile", f"{d}/log-neuron-cc.txt"], cwd=d, capture_output=True, text=True)
    lines = (r.stdout + r.stderr).splitlines()
    err = [l for l in lines if "[INTERNAL_ERROR]" in l or "ERROR" in l or "NCC_" in l]
    print(f"{name}: neuronx-cc rc={r.returncode} in {time.time() - t0:.0f}s"
          + ("" if r.returncode == 0 else f"\n  first error: {err[0][:600] if err else lines[-5:]}"), flush=True)
    return r.returncode


def main():
    model, args = build_model()
    jobs = []
    if "prefill" in WHICH:
        jobs += [("prefill", h) for h in capture(model, "prefill", prefill_inputs(model, args))]
    if "decode" in WHICH:
        jobs += [("decode", h) for h in capture(model, "decode", decode_inputs(model, args))]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(len(jobs) or 1) as ex:
        list(ex.map(lambda j: neuronx_cc(*j), jobs))


if __name__ == "__main__":
    main()
