"""Compile-only probe of GLM-5.3-Flash's prefill and decode graphs at REAL dims, no device.

    NEURON_LIBTORCH_CPU_COMPILE=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \\
    PYTHONPATH=/data/glm53f-wt:/data/pylib-tf517 python compile_probe.py OUT [prefill|decode] [--compile]

Truncated model, server coordinates:
- the released config.json with ``text_config`` cut to 4 layers, one of each kind:

  | layer | attention | MLP |
  |---|---|---|
  | 0 | KDA | dense |
  | 1, 2 | KDA | MoE |
  | 3 | MLA + DSA indexer | MoE |

- every dim is real: hidden 4096, 64 KDA and MLA heads, 288 experts, vocab 154880;
- TP=64 (one head per rank); routed experts as EP=32 x expert-TP=2 (9 per rank, I_TP=1024);
- the collectives are real ``_c10d_functional`` ops over a 64-rank fake process group,
  so the HLO carries the server's all-reduces and all-gathers;
- pages and metadata are built the way ``tests/plugin_harness.FakeRunner`` builds them
  (vLLM 0.24's grouping), on the meta device.

Before capture, an FX lint lists every op the Neuron device path is known to refuse, with
its source line:
- top-k or sort (an HLO ``sort``, NCC_EVRF029);
- integer floor-div or mod (through f64);
- a compare against a Python float (f64);
- ``copysign`` (a custom call);
- ``.to(device)`` in the graph;
- fp8, e8m0 or f64 tensors.

The captured HLO is linted for f64, sort and non-Neuron custom calls. ``--compile`` then
runs neuronx-cc with the runner's flags plus ``--lnc 2``.
"""
from __future__ import annotations

import functools
import glob
import json
import operator
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("NEURON_LIBTORCH_CPU_COMPILE", "1")
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
os.environ.pop("VLLM_NEURON_CPU_MODE", None)

import torch  # noqa: E402

import libtorch_neuronx_lite  # noqa: E402,F401
import torch.distributed as dist  # noqa: E402
import torch.distributed._functional_collectives as funcol  # noqa: E402

GLM_DIR = Path(os.environ.get("GLM_DIR", "/data/models/GLM-5.3-Flash-BF16"))
OUT = Path(sys.argv[1]).resolve()
ARGS = [a for a in sys.argv[2:] if not a.startswith("--")]
WHICH = ARGS or ["prefill", "decode"]
COMPILE = "--compile" in sys.argv
TP, EP, ETP = 64, 32, 2
RANK = int(os.environ.get("GLM_RANK", "0"))
# the runner's compile options (neuron_model_runner.py, "Build compile options"), so the
# capture directory name is the cache key the server computes
RUNNER_OPTIONS = {"alias_meta_to_neuron": True, "compiler_args": [
    "--auto-cast=none", "--verbose=35", "-O1",
    "--internal-hlo2tensorizer-options=--modular-flow-mac-threshold=10",
    "--internal-backend-options=--enable-verifier=false --enable-nested-dynamic-loop"]}
# the serve script's shapes (serve_glm53f.sh): single-shot prefill bucket == max_model_len
MAX_LEN = int(os.environ.get("GLM_MAX_LEN", "2048"))
T_PREFILL = int(os.environ.get("GLM_T_PREFILL", str(MAX_LEN)))
N_DECODE = int(os.environ.get("GLM_SEQS", "16"))
NUM_BLOCKS = int(os.environ.get("GLM_BLOCKS", str((MAX_LEN // 64 + 4) * N_DECODE + 64)))
FULL_DEPTH = os.environ.get("GLM_LAYERS", "4") == "all"
META = torch.device("meta")
FLAGS = ["--framework", "XLA", "--target", "trn2", "--lnc", "2", "--auto-cast=none", "--verbose=35", "-O1",
         "--internal-hlo2tensorizer-options=--modular-flow-mac-threshold=10 --experimental-unsafe-fp8e4m3fn-as-fp8e4m3",
         "--internal-backend-options=--enable-verifier=false --enable-nested-dynamic-loop"]


# ------------------------------------------------------------------ process groups
def init_groups():
    from torch.testing._internal.distributed.fake_pg import FakeStore
    if not dist.is_initialized():
        dist.init_process_group("fake", store=FakeStore(), rank=RANK, world_size=TP)
    world = dist.group.WORLD
    # every rank creates every EP-TP row group, in order, as the plugin's parallel state does
    rows = [dist.new_group(ranks=list(range(r * ETP, (r + 1) * ETP))) for r in range(TP // ETP)]
    etp = rows[RANK // ETP]
    # the capture backend names per-rank workdirs from vLLM's TP group; one is enough
    from libtorch_neuronx_lite.compile import capture_backend as cb
    orig = cb.setup_workdir_common
    cb.setup_workdir_common = lambda *a, per_rank=True, **k: orig(*a, per_rank=False, **k)
    return world, etp


class Coordinator:
    """The slice of vLLM's GroupCoordinator the GLM model uses, over a fake group."""

    def __init__(self, pg, world_size, rank_in_group=0):
        self.device_group, self.world_size, self.rank_in_group = pg, world_size, rank_in_group

    def all_reduce(self, t):
        return funcol.all_reduce(t, "sum", self.device_group)

    def all_gather(self, t, dim=-1):
        return funcol.all_gather_tensor(t, gather_dim=dim, group=self.device_group)


# ------------------------------------------------------------------------- model
def truncated_hf_config(n_layers=4):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(str(GLM_DIR))
    if FULL_DEPTH:
        return cfg
    t = cfg.text_config
    full = t.num_hidden_layers
    layer_types = list(t.layer_types)
    keep = [0, 1, 2, 3]                         # KDA, KDA, KDA, MLA in the real pattern
    assert [layer_types[i] for i in keep] == [layer_types[0]] * 3 + [layer_types[3]], layer_types[:4]
    t.num_hidden_layers = n_layers
    t.layer_types = [layer_types[i] for i in keep]
    mlp = list(t.mlp_layer_types)
    t.mlp_layer_types = [mlp[0], mlp[3], mlp[3], mlp[3]]   # dense, then MoE
    t.first_k_dense_replace = 1
    for name in ("indexer_types",):
        v = getattr(t, name, None)
        if isinstance(v, (list, tuple)) and len(v) == full:
            setattr(t, name, [v[i] for i in keep])
    lac = dict(t.linear_attn_config)
    lac["kda_layers"] = [i for i, k in enumerate(t.layer_types) if k == layer_types[0]]
    lac["full_attn_layers"] = [i for i, k in enumerate(t.layer_types) if k == layer_types[3]]
    t.linear_attn_config = lac
    return cfg


def build_model():
    world, etp = init_groups()
    from vllm_neuron.model.glm5_next import model as M
    from vllm_neuron.model.glm5_next.config import Glm5NextConfig

    hf = truncated_hf_config()
    config = Glm5NextConfig.from_configs(hf, text_neuron_config=None)
    config.text_config.torch_dtype = torch.bfloat16
    layout = M.ExpertLayout(ep_degree=EP, ep_rank=RANK // ETP, tp_degree=ETP, tp_rank=RANK % ETP,
                            ep_tp_group=Coordinator(etp, ETP, RANK % ETP))
    with torch.device("meta"):
        model = M.Glm5NextForCausalLM(config, tp_group=Coordinator(world, TP, RANK), expert_layout=layout)
    model.set_dtype(torch.bfloat16)
    model.to(META)
    return model


# ------------------------------------------------------------- pages and metadata
def aligned_block_size(model, dtype=torch.bfloat16, alignment=32):
    from vllm_neuron.model.glm5_next.cache_layout import LatentPageLayout
    kda = next(l.self_attn for l in model.model.layers if l.is_linear_attention)
    need = 4 * (kda.conv_numel + kda.rec_numel)
    B = alignment
    while LatentPageLayout.from_config(model.text_config, B, dtype).total_bytes < need:
        B += alignment
    return B


class MetaRunner:
    """FakeRunner's grouping and metadata, every tensor on meta."""

    def __init__(self, model, B, dtype=torch.bfloat16):
        from vllm_neuron.model.glm5_next.cache_layout import LatentPageLayout
        self.model, self.B = model, B
        self.layout = LatentPageLayout.from_config(model.text_config, B, dtype)
        page = self.layout.total_bytes
        kda = [l.self_attn for l in model.model.layers if l.is_linear_attention]
        mla = [l.self_attn for l in model.model.layers if not l.is_linear_attention]
        n = min(len(kda), len(mla))
        kda_groups = [kda[i::-(-len(kda) // n)] for i in range(-(-len(kda) // n))]
        self.groups = [("mla", mla)] + [("kda", g) for g in kda_groups]
        self.num_pages = NUM_BLOCKS + 2
        kv = {}
        for kind, layers in self.groups:
            for layer in layers:
                dt = torch.float32 if kind == "kda" else dtype
                kv[layer.layer_name] = [torch.empty(self.num_pages, page // dt.itemsize, dtype=dt, device=META)]
        model.bind_kv_cache(kv)
        print(f"block_size {B}, page {page} B, {len(self.groups)} groups "
              f"({[(k, len(l)) for k, l in self.groups]}), pages {tuple(next(iter(kv.values()))[0].shape)}",
              flush=True)

    def _md(self, rows, q, slots_len):
        md = {}
        for kind, layers in self.groups:
            if kind == "kda":
                bt, bs = torch.zeros(rows, 1, dtype=torch.int32, device=META), MAX_LEN
            else:
                bt, bs = torch.zeros(rows, -(-MAX_LEN // self.B), dtype=torch.int32, device=META), self.B
            meta = {"block_table_tensor": bt, "slot_mapping": torch.zeros(slots_len, dtype=torch.int64, device=META),
                    "max_query_len": q, "decode_token_threshold": 1, "block_size": bs}
            for layer in layers:
                md[layer.layer_name] = meta
        return md

    def prefill_inputs(self):
        T = T_PREFILL
        return dict(input_ids=torch.zeros(T, dtype=torch.int32, device=META),
                    positions=torch.zeros(T, dtype=torch.int64, device=META),
                    attn_metadata=self._md(1, T, T), rank=torch.zeros((), dtype=torch.int32, device=META),
                    sampling_positions=torch.zeros(1, dtype=torch.int64, device=META))

    def decode_inputs(self):
        n = N_DECODE
        return dict(input_ids=torch.zeros(n, dtype=torch.int32, device=META),
                    positions=torch.zeros(n, dtype=torch.int64, device=META),
                    attn_metadata=self._md(n, 1, n), rank=torch.zeros((), dtype=torch.int32, device=META),
                    sampling_positions=torch.zeros(n, dtype=torch.int64, device=META))


# ------------------------------------------------------------------------ linting
_SORTS = {torch.topk, torch.sort, torch.argsort, torch.kthvalue, "topk", "sort", "argsort", "kthvalue"}
_INTDIV = {operator.floordiv, operator.mod, torch.floor_divide, torch.remainder, "floor_divide",
           "remainder", "__floordiv__", "__mod__", "__rfloordiv__", "__rmod__"}
_CMP = {operator.lt, operator.le, operator.gt, operator.ge, operator.eq, operator.ne,
        torch.lt, torch.le, torch.gt, torch.ge, torch.eq, torch.ne, "lt", "le", "gt", "ge", "eq", "ne"}
_BAD_DTYPES = {torch.float64, torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e8m0fnu}


def _ev(x):
    return x.meta.get("example_value") if isinstance(x, torch.fx.Node) else None


def _where(node):
    st = node.meta.get("stack_trace") or ""
    frames = [l.strip() for l in st.splitlines() if l.strip().startswith("File ")]
    mine = [f for f in frames if "glm5_next" in f or "vllm_neuron" in f]
    return (mine or frames or ["?"])[-1].replace("File ", "")


def fx_lint(gm):
    found = []
    for node in gm.graph.nodes:
        if node.op not in ("call_function", "call_method"):
            continue
        t = node.target
        name = getattr(t, "__name__", str(t))
        args = list(node.args) + list(node.kwargs.values())
        tensors = [a for a in args if isinstance(a, torch.fx.Node)]
        if t in _SORTS:
            found.append(("sort/top-k -> HLO sort (NCC_EVRF029)", name, _where(node)))
        if t in _INTDIV or (t in (torch.div, "div") and node.kwargs.get("rounding_mode") == "floor"):
            if any((ev := _ev(a)) is not None and not ev.dtype.is_floating_point for a in tensors):
                found.append(("integer floor-div/mod -> f64", name, _where(node)))
        if t in _CMP and any(isinstance(a, float) for a in args):
            found.append(("compare with a Python float -> f64", name, _where(node)))
        if t in (torch.copysign, "copysign"):
            found.append(("copysign -> custom call", name, _where(node)))
        if t in ("to", "cuda", "cpu") and any(isinstance(a, (torch.device, str)) and "neuron" in str(a) or
                                             isinstance(a, torch.device) for a in node.args[1:]):
            found.append(("device copy in graph", name, _where(node)))
        ev = node.meta.get("example_value")
        for v in (ev if isinstance(ev, (tuple, list)) else [ev]):
            if isinstance(v, torch.Tensor) and v.dtype in _BAD_DTYPES:
                found.append((f"{v.dtype} tensor in graph", name, _where(node)))
    return found


def hlo_lint(path):
    from libtorch_neuronx_lite.pyhlo.service import hlo_pb2
    m = hlo_pb2.HloModuleProto()
    m.ParseFromString(open(path, "rb").read())
    ops, f64, cc = Counter(), 0, Counter()
    for comp in m.computations:
        for i in comp.instructions:
            ops[i.opcode] += 1
            if i.shape.element_type == 12:          # F64
                f64 += 1
            if i.opcode == "custom-call":
                cc[i.custom_call_target] += 1
    bad_cc = {k: v for k, v in cc.items() if not k.startswith("AwsNeuron")}
    return {"sort": ops.get("sort", 0), "f64": f64, "custom_calls": dict(cc), "non_neuron_custom_calls": bad_cc,
            "all-reduce": ops.get("all-reduce", 0), "all-gather": ops.get("all-gather", 0),
            "instructions": sum(ops.values())}


# ------------------------------------------------------------------------ capture
def capture(model, name, kwargs):
    from libtorch_neuronx_lite.compile.capture_backend import CaptureComplete, capture as cap
    workdir = OUT / name
    workdir.mkdir(parents=True, exist_ok=True)
    lint = []

    def backend(gm, example_inputs):
        found = fx_lint(gm)
        lint.extend(found)
        uniq = Counter(found)
        for (why, op, where), k in sorted(uniq.items(), key=lambda x: x[0][2]):
            print(f"  FX-LINT {why}: {op} x{k} at {where}", flush=True)
        if not uniq:
            print("  FX-LINT clean", flush=True)
        return cap(gm, example_inputs, {**RUNNER_OPTIONS, "compiler_workdir": str(workdir)})

    before = set(glob.glob(f"{workdir}/**/graph.hlo", recursive=True))
    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    t0 = time.time()
    try:
        compiled(**kwargs)        # grad enabled outside, as the runner calls it
    except CaptureComplete:
        pass
    hlos = sorted(set(glob.glob(f"{workdir}/**/graph.hlo", recursive=True)) - before)
    print(f"{name}: captured {len(hlos)} graph(s) in {time.time() - t0:.0f}s", flush=True)
    import hashlib
    for h in hlos:
        print(f"  KEY {name} rank {RANK}: {Path(h).parent.name}  hlo_md5 "
              f"{hashlib.md5(open(h, 'rb').read()).hexdigest()}", flush=True)
        print(f"  HLO {h}: {json.dumps(hlo_lint(h))}", flush=True)
    torch._dynamo.reset()
    return hlos


def neuronx_cc(name, hlo):
    d = os.path.dirname(hlo)
    t0 = time.time()
    r = subprocess.run(["nice", "-n", "10", "/data/venv-fork/bin/neuronx-cc", "compile", hlo, *FLAGS, "--output", f"{d}/graph.neff",
                        "--logfile", f"{d}/log-neuron-cc.txt"], cwd=d, capture_output=True, text=True)
    lines = (r.stdout + r.stderr).splitlines()
    err = [l for l in lines if "[INTERNAL_ERROR]" in l or "ERROR" in l or "NCC_" in l]
    hbm = ""
    try:
        log = open(f"{d}/log-neuron-cc.txt", errors="replace").read()
        hits = [l for l in log.splitlines() if "HBM usage is" in l or "peak HBM" in l or "NCC_EOOM" in l]
        hbm = hits[-1].split("]: ", 1)[-1][:300] if hits else ""
    except OSError:
        pass
    print(f"{name}: neuronx-cc rc={r.returncode} in {time.time() - t0:.0f}s; {hbm}"
          + ("" if r.returncode == 0 else f"\n  first error: {err[0][:600] if err else lines[-5:]}"), flush=True)


def main():
    model = build_model()
    print(f"layers {len(model.model.layers)}, prefill T={T_PREFILL}, max_len {MAX_LEN}, decode n={N_DECODE}, "
          f"blocks {NUM_BLOCKS}", flush=True)
    runner = MetaRunner(model, aligned_block_size(model))
    jobs = []
    if "prefill" in WHICH:
        jobs += [("prefill", h) for h in capture(model, "prefill", runner.prefill_inputs())]
    if "decode" in WHICH:
        jobs += [("decode", h) for h in capture(model, "decode", runner.decode_inputs())]
    if COMPILE and jobs:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(len(jobs)) as ex:
            list(ex.map(lambda j: neuronx_cc(*j), jobs))


if __name__ == "__main__":
    main()
