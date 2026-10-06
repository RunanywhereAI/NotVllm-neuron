# SPDX-License-Identifier: Apache-2.0
"""Run the plugin's GLM-5.3-Flash model on a laptop, against the oracle.

Two pieces, both torch-only:

* ``import_plugin`` imports a module of ``vllm_neuron`` **without executing the package
  ``__init__`` files**, which pull in vLLM (manylinux wheels only). It registers the
  intermediate packages as bare namespace modules pointing at their directories, so
  the plugin's own relative and absolute imports resolve to the real files. Unlike
  loading single files by path, this runs ``model.py`` exactly as written -- including
  its imports of ``kda``, ``mla``, ``cache_layout``, ``kv_cache`` and the vendored row
  scatter.
* ``FakeRunner`` stands in for ``NeuronModelRunner.initialize_kv_cache`` and the
  per-group ``attn_metadata``, reproducing the parts that have caused bugs:
  vLLM's grouping (``layers[i::n_groups]``, buffer ``i`` shared by the i-th layer of
  every group), page-major views at two different dtypes over one buffer, the two
  private pages past ``num_blocks`` (zero page, sink), block ids that are global
  across groups, padded prefill (pads appended, last position repeated), padded
  decode rows (slot 0, stale block ids) and **hostile pages**: every byte the model
  did not write is ``0xFF`` -- NaN in fp32 and bf16 -- so anything read without being
  selected away poisons the output loudly.
"""
from __future__ import annotations

import dataclasses
import importlib
import math
import pathlib
import sys
import types
from types import SimpleNamespace

import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
_PACKAGES = ("vllm_neuron", "vllm_neuron.model", "vllm_neuron.model.glm5_next",
             "vllm_neuron.functional", "vllm_neuron.functional.vendored_kernels",
             "vllm_neuron.utils", "vllm_neuron.accuracy", "vllm_neuron.nn")


def import_plugin(dotted: str):
    for pkg in _PACKAGES:
        if pkg not in sys.modules:
            mod = types.ModuleType(pkg)
            mod.__path__ = [str(ROOT.joinpath(*pkg.split(".")))]
            sys.modules[pkg] = mod
    return importlib.import_module(dotted)


# ------------------------------------------------------------------ a tiny config
# HF attribute names, as ``config.json``'s text_config spells them. Small, with every
# structural feature: [KDA x3, MLA] x2 (both layer types, the MLA layers at i % 4 == 3),
# first_k_dense_replace 3 (both MLP kinds), index_topk 8 with kpool 4 -> the dense-exact
# ceiling is 8 + 4 - 1 = 11 tokens, so every sequence below crosses it.
TINY_HF = dict(
    hidden_size=64, intermediate_size=96, num_hidden_layers=8,
    num_attention_heads=2, q_lora_rank=32, kv_lora_rank=32, qk_nope_head_dim=16,
    qk_rope_head_dim=0, v_head_dim=16,
    index_n_heads=32, index_head_dim=16, index_topk=8, index_kpool=4,
    linear_attn_config={"num_heads": 2, "head_dim": 16, "short_conv_kernel_size": 4,
                        "gate_lower_bound": -5.0},
    hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
    n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=24,
    n_shared_experts=1, routed_scaling_factor=2.5, norm_topk_prob=True,
    swiglu_limit=10.0, first_k_dense_replace=3, n_group=1, topk_group=1,
    vocab_size=96, rms_norm_eps=1e-5, tie_word_embeddings=False,
    max_position_embeddings=4096, mla_use_nope=True, mhc=True,
    scoring_func="sigmoid", topk_method="noaux_tc", hidden_act="silu",
    index_kpool_compress=True, index_kpool_always_select_tail=True,
)


def hf_text_config(dtype=torch.float32, **over):
    d = dict(TINY_HF, **over)
    n = d["num_hidden_layers"]
    d.setdefault("layer_types", ["linear_attention" if i % 4 != 3 else
                                 "deepseek_sparse_attention" for i in range(n)])
    d.setdefault("mlp_layer_types", ["dense" if i < d["first_k_dense_replace"] else
                                     "sparse" for i in range(n)])
    d.setdefault("indexer_types", ["full"] * n)
    return SimpleNamespace(dtype=dtype, **d)


def oracle_cfg(text, R):
    """The oracle's ``FlashCfg`` for a plugin ``Glm5NextTextConfig``."""
    return R.FlashCfg(
        hidden_size=text.hidden_size, num_hidden_layers=text.num_hidden_layers,
        rms_norm_eps=text.rms_norm_eps, hc_mult=text.hc_mult,
        hc_sinkhorn_iters=text.hc_sinkhorn_iters, hc_eps=text.hc_eps,
        linear_num_heads=text.linear_num_heads, linear_head_dim=text.linear_head_dim,
        linear_conv_kernel_dim=text.linear_conv_kernel_dim,
        linear_lower_bound=text.linear_lower_bound,
        num_attention_heads=text.num_attention_heads, q_lora_rank=text.q_lora_rank,
        kv_lora_rank=text.kv_lora_rank, qk_nope_head_dim=text.qk_nope_head_dim,
        qk_rope_head_dim=text.qk_rope_head_dim, v_head_dim=text.v_head_dim,
        index_n_heads=text.index_n_heads, index_head_dim=text.index_head_dim,
        index_topk=text.index_topk, index_kpool=text.index_kpool,
        n_routed_experts=text.n_routed_experts, num_experts_per_tok=text.num_experts_per_tok,
        moe_intermediate_size=text.moe_intermediate_size,
        n_shared_experts=text.n_shared_experts,
        routed_scaling_factor=text.routed_scaling_factor,
        norm_topk_prob=text.norm_topk_prob, swiglu_limit=text.swiglu_limit,
        first_k_dense_replace=text.first_k_dense_replace,
        intermediate_size=text.intermediate_size,
        layer_types=list(text.layer_types), mlp_layer_types=list(text.mlp_layer_types),
    )


def randomize_(model, seed=0, mlp_gain=6.0):
    """Weights hard enough to reach the behaviour under test.

    The oracle's own init leaves mHC uniform (base 0, fn ~0) and the SwiGLU far below
    its clamp (std 0.02). Here: unit-variance projections, MLP/expert gate and up
    projections at ``mlp_gain`` x unit variance so a large fraction of activations
    exceeds ``swiglu_limit`` 10, a random mHC base so ``comb`` is far from both the
    identity and the uniform matrix, and random gate parameters.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            leaf = name.rsplit(".", 1)[-1]
            r = torch.randn(p.shape, generator=g)
            if "norm" in name and leaf in ("weight",):
                p.copy_(1 + 0.1 * r)
            elif leaf == "bias":                                   # indexer k_norm
                p.copy_(0.1 * r)
            elif leaf == "fn":
                p.copy_(r * 0.3 / math.sqrt(p.shape[1]))
            elif leaf == "base":
                p.copy_(r)
            elif leaf == "scale":
                p.copy_(1 + 0.1 * r)
            elif leaf in ("A_log", "dt_bias"):
                p.copy_(0.5 * r)
            elif leaf == "e_score_correction_bias":
                p.copy_(0.1 * r)
            elif leaf == "index_kpool_compress_ape":
                p.copy_(0.5 * r)
            elif name.endswith("conv1d.weight"):
                p.copy_(0.5 * r)
            elif name.endswith("embed_tokens.weight"):
                p.copy_(r)
            elif ("gate_proj" in name or "up_proj" in name) and "mlp" in name:
                p.copy_(r * mlp_gain / math.sqrt(p.shape[-1]))
            elif leaf == "gate_up_proj":
                p.copy_(r * mlp_gain / math.sqrt(p.shape[-1]))
            else:
                p.copy_(r / math.sqrt(p.shape[-1]))
    return model


def plugin_state_from_oracle(oracle_sd):
    return import_plugin("vllm_neuron.model.glm5_next.model").state_dict_from_reference(oracle_sd)


# ------------------------------------------------------------------ the fake runner
def aligned_block_size(model, dtype=torch.float32, alignment: int = 32) -> int:
    """The smallest multiple of ``alignment`` whose folded MLA page holds one KDA layer's
    page-major state -- what the platform's hybrid alignment arrives at."""
    CL = import_plugin("vllm_neuron.model.glm5_next.cache_layout")
    kda = next(l.self_attn for l in model.model.layers if l.is_linear_attention)
    need = 4 * (kda.conv_numel + kda.rec_numel)              # page-major view is fp32
    B = alignment
    while CL.LatentPageLayout.from_config(model.text_config, B, dtype).total_bytes < need:
        B += alignment
    return B


class FakeRunner:
    """Allocation, binding and metadata as ``NeuronModelRunner`` does them.

    Groups follow vLLM 0.24's ``_get_kv_cache_groups_uniform_page_size``: the smallest
    layer kind sets the group size, larger kinds split into ``layers[i::n]`` groups, and
    buffer ``i`` is shared by the i-th layer of every group. Every group has its own
    block ids, drawn from one global pool -- the property that makes a recycled page
    hostile.
    """

    def __init__(self, model, block_size: int, num_blocks: int, dtype=torch.float32,
                 max_model_len: int = 4096):
        CL = import_plugin("vllm_neuron.model.glm5_next.cache_layout")
        self.model, self.B, self.dtype, self.max_len = model, block_size, dtype, max_model_len
        text = model.text_config
        self.layout = CL.LatentPageLayout.from_config(text, block_size, dtype)
        page = self.layout.total_bytes
        kda = [l.self_attn for l in model.model.layers if l.is_linear_attention]
        mla = [l.self_attn for l in model.model.layers if not l.is_linear_attention]
        state_bytes = 4 * (kda[0].conv_numel + kda[0].rec_numel)
        assert page >= state_bytes, (page, state_bytes)
        self.page_bytes = page
        n = min(len(kda), len(mla))
        kda_groups = [kda[i::math.ceil(len(kda) / n)] for i in range(math.ceil(len(kda) / n))]
        self.groups = [("mla", mla)] + [("kda", g) for g in kda_groups]
        self.num_pages = num_blocks + 2
        self.zero_page, self.sink = self.num_pages - 2, self.num_pages - 1
        kv = {}
        for i in range(n):
            raw = torch.full((self.num_pages * page,), -1, dtype=torch.int8)   # 0xFF
            raw.view(self.num_pages, page)[self.zero_page] = 0
            for kind, layers in self.groups:
                if i >= len(layers):
                    continue
                view = (raw.view(torch.float32) if kind == "kda" else raw.view(dtype))
                kv[layers[i].layer_name] = [view.view(self.num_pages, -1)]
        for kind, layers in self.groups:
            for layer in layers[n:]:                   # an unshared tail, as vLLM allows
                raw = torch.full((self.num_pages * page,), -1, dtype=torch.int8)
                raw.view(self.num_pages, page)[self.zero_page] = 0
                view = raw.view(torch.float32) if kind == "kda" else raw.view(dtype)
                kv[layer.layer_name] = [view.view(self.num_pages, -1)]
        model.bind_kv_cache(kv)
        self.kv = kv
        # global block pool; hand out from the top so ids differ visibly across groups
        self.free = list(range(num_blocks - 1, 0, -1))
        self.blocks: dict[tuple[int, int], list[int]] = {}       # (req, group) -> ids

    def _alloc(self, req, g, upto_tokens):
        ids = self.blocks.setdefault((req, g), [])
        kind = self.groups[g][0]
        need = 1 if kind == "kda" else -(-upto_tokens // self.B)
        while len(ids) < need:
            ids.append(self.free.pop())
        return ids

    def _meta(self, g, block_table, slots, max_query_len, block_size):
        return {"block_table_tensor": block_table, "slot_mapping": slots,
                "max_query_len": max_query_len, "decode_token_threshold": 1,
                "block_size": block_size}

    def _fan_out(self, per_group):
        md = {}
        for (kind, layers), meta in zip(self.groups, per_group):
            for layer in layers:
                md[layer.layer_name] = meta
        return md

    def prefill(self, req, ids, bucket):
        """One request, padded to ``bucket`` -> logits for every real position."""
        n = len(ids)
        assert n <= bucket
        positions = torch.tensor(list(range(n)) + [n - 1] * (bucket - n))
        input_ids = torch.tensor(list(ids) + [0] * (bucket - n))
        per_group = []
        for g, (kind, _) in enumerate(self.groups):
            blk = self._alloc(req, g, n)
            if kind == "kda":
                bt = torch.tensor([blk], dtype=torch.int32)
                slots = [blk[0] * self.max_len + t for t in range(n)]
                bs = self.max_len
            else:
                width = -(-self.max_len // self.B)
                bt = torch.tensor([blk + [0] * (width - len(blk))], dtype=torch.int32)
                slots = [blk[t // self.B] * self.B + t % self.B for t in range(n)]
                bs = self.B
            slots = torch.tensor(slots + [0] * (bucket - n), dtype=torch.int64)
            per_group.append(self._meta(g, bt, slots, bucket, bs))
        out = self.model(input_ids=input_ids, positions=positions,
                         attn_metadata=self._fan_out(per_group),
                         sampling_positions=torch.arange(n))
        return out

    def decode(self, rows, batch_bucket, nb=None):
        """``rows``: list of ``(req, token_id, position)`` -> logits ``[batch_bucket, V]``
        (padded rows included: their logits are not discarded on device -- the sampler's
        argmax reduces across the whole tile -- so they must be finite too).

        Rows past ``len(rows)`` are padding: slot 0, position 0 and a STALE block id.
        Alternate padded rows point at a live request's own blocks (a padded row that
        wrote anywhere but the sink would corrupt it) and at a never-written page (a
        padded row that read anything but the zero page would read NaN).
        """
        n = len(rows)
        assert n <= batch_bucket
        nb = nb or -(-self.max_len // self.B)
        positions = torch.tensor([p for _, _, p in rows] + [0] * (batch_bucket - n))
        input_ids = torch.tensor([t for _, t, _ in rows] + [0] * (batch_bucket - n))
        per_group = []
        for g, (kind, _) in enumerate(self.groups):
            bt, slots = [], []
            for req, _, p in rows:
                blk = self._alloc(req, g, p + 1)
                if kind == "kda":
                    bt.append([blk[0]])
                    slots.append(blk[0] * self.max_len + p)
                else:
                    bt.append(blk + [0] * (nb - len(blk)))
                    slots.append(blk[p // self.B] * self.B + p % self.B)
            hostile = self.free[0]                      # never handed out yet
            for k in range(batch_bucket - n):
                bt.append(list(bt[0]) if k % 2 == 0 else [hostile] * len(bt[0]))
                slots.append(0)
            per_group.append(self._meta(
                g, torch.tensor(bt, dtype=torch.int32), torch.tensor(slots, dtype=torch.int64),
                1, self.max_len if kind == "kda" else self.B))
        return self.model(input_ids=input_ids, positions=positions,
                          attn_metadata=self._fan_out(per_group),
                          sampling_positions=torch.arange(batch_bucket))


# ------------------------------------------------------------ a checkpoint on disk
def hf_checkpoint_from_oracle(oracle_sd: dict, n_experts: int,
                              dtype: torch.dtype = torch.bfloat16) -> dict:
    """The oracle's state_dict in the checkpoint's own names and layout: the inverse of
    ``weight_converter.py`` (q/k/v convs split, the forget gate flattened, experts
    unstacked). BF16 like ``zai-org/GLM-5.3-Flash-BF16``, except the router bias, which
    ships F32. Validated by round-tripping through the converter, never trusted alone.
    """
    P = "model.language_model."
    out = {}
    for k, v in oracle_sd.items():
        v = v.detach().clone()
        if k == "lm_head.weight":
            out[k] = v
            continue
        k2 = (k.replace(".attn_hc.", ".hc_attn_").replace(".ffn_hc.", ".hc_ffn_")
               .replace(".self_attn.forget_gate.", ".self_attn."))
        if k2.endswith("self_attn.conv1d.weight"):
            for c, part in zip("qkv", v.chunk(3, 0)):
                out[P + k2.replace("conv1d", f"{c}_conv1d")] = part.contiguous()
        elif k2.endswith("mlp.gate_up_proj"):
            base = P + k2[: -len("gate_up_proj")]
            for e in range(n_experts):
                g, u = v[e].chunk(2, 0)
                out[f"{base}experts.{e}.gate_proj.weight"] = g.contiguous()
                out[f"{base}experts.{e}.up_proj.weight"] = u.contiguous()
        elif k2.endswith("mlp.down_proj") and v.dim() == 3:
            base = P + k2[: -len("down_proj")]
            for e in range(n_experts):
                out[f"{base}experts.{e}.down_proj.weight"] = v[e].contiguous()
        else:
            out[P + k2] = v
    return {k: (v.float() if k.endswith("e_score_correction_bias") else v.to(dtype))
            for k, v in out.items()}


def write_checkpoint(tensors: dict, directory, shards: int = 3):
    """``model-0000i-of-0000n.safetensors`` plus ``model.safetensors.index.json``."""
    import json

    from safetensors.torch import save_file

    directory = pathlib.Path(directory)
    names = sorted(tensors)
    weight_map = {}
    for i in range(shards):
        fname = f"model-{i + 1:05d}-of-{shards:05d}.safetensors"
        part = {n: tensors[n].contiguous() for n in names[i::shards]}
        save_file(part, str(directory / fname))
        weight_map.update({n: fname for n in part})
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}))


# ------------------------------------------------------- tensor parallelism on a laptop
class GlooTP:
    """The slice of vLLM's ``GroupCoordinator`` the model uses, over a gloo group.

    ``all_reduce`` returns its result (vLLM's is out-of-place), and ``device_group`` is
    what the plugin's ``VocabDimShardedEmbedding`` / ``ColumnParallelLinear`` take.
    """

    def __init__(self, world_size: int, rank: int):
        import torch.distributed as dist

        self.dist, self.world_size, self.rank_in_group = dist, world_size, rank
        self.device_group = dist.group.WORLD

    def all_reduce(self, t):
        t = t.clone()
        self.dist.all_reduce(t, group=self.device_group)
        return t

    def all_gather(self, t, dim=-1):
        parts = [torch.empty_like(t) for _ in range(self.world_size)]
        self.dist.all_gather(parts, t.contiguous(), group=self.device_group)
        return torch.cat(parts, dim)


SABOTAGE = {
    # parameter -> loaded from the NEXT rank's slice on rank 1 (wrong shard, right shape)
    "kda_conv": "model.layers.0.self_attn.conv1d.weight",
    "mla_kv_b": "model.layers.3.self_attn.kv_b_proj.weight",
    "kda_A_log": "model.layers.5.self_attn.forget_gate.A_log",
    # routed experts take their shard from the expert layout, not the loader's rank, so
    # these rebuild the loaders from a layout with one coordinate wrong
    "expert_gate_up": "model.layers.7.mlp.gate_up_proj",
    "expert_down": "model.layers.4.mlp.down_proj",
}
_EXPERT_SABOTAGE = {"expert_gate_up": 0, "expert_down": 1}     # which of the two loaders


def expert_layout_for(rank, world, ep_degree, M):
    """Contiguous EP x TP placement for the laptop harness: expert group ``rank // tp``,
    intermediate shard ``rank % tp``. (On trn2 the plugin's mesh decides; see
    ``ExpertLayout``.)"""
    if ep_degree == 1:
        return None
    tp = world // ep_degree
    return M.ExpertLayout(ep_degree=ep_degree, ep_rank=rank // tp, tp_degree=tp,
                          tp_rank=rank % tp)


def _wrong(layout, world, M):
    """The same layout with the coordinate that varies across ranks moved by one."""
    L = layout or M.ExpertLayout(tp_degree=world, tp_rank=0)
    if L.tp_degree > 1:
        return dataclasses.replace(L, tp_rank=(L.tp_rank + 1) % L.tp_degree)
    return dataclasses.replace(L, ep_rank=(L.ep_rank + 1) % L.ep_degree)


def tp_worker(rank, world, port, ckpt, out_path, sabotage, ep_degree=1):
    """One TP rank: load this rank's shards from ``ckpt``, then run every prompt of
    ``oracle_reference.json`` through ``FakeRunner`` -- one prefill per request, then
    batched decode with a padded row -- teacher-forced on the oracle's greedy tokens so
    every TP degree sees identical inputs. Rank 0 saves the logits."""
    import json
    from types import SimpleNamespace

    import torch.distributed as dist

    torch.manual_seed(0)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                            world_size=world)
    try:
        CFG = import_plugin("vllm_neuron.model.glm5_next.config")
        M = import_plugin("vllm_neuron.model.glm5_next.model")
        WL = import_plugin("vllm_neuron.utils.weight_loader")
        tc = json.loads((pathlib.Path(ckpt) / "config.json").read_text())["text_config"]
        tc["dtype"] = torch.float32
        text = CFG.Glm5NextTextConfig.from_hf(SimpleNamespace(**tc))
        layout = expert_layout_for(rank, world, ep_degree, M)
        model = M.Glm5NextForCausalLM(CFG.Glm5NextConfig(text_config=text),
                                      tp_group=GlooTP(world, rank),
                                      expert_layout=layout).eval()
        if sabotage and rank == 1:
            param = dict(model.named_parameters())[SABOTAGE[sabotage]]
            if sabotage in _EXPERT_SABOTAGE:
                if layout is None:
                    layout = M.ExpertLayout(tp_degree=world, tp_rank=rank)
                bad = model.expert_loaders(_wrong(layout, world, M))[_EXPERT_SABOTAGE[sabotage]]
                WL.set_weight_loader(param, bad)
            else:
                inner = WL.get_weight_loader(param)
                WL.set_weight_loader(param, WL.SafetensorsWeightLoader(
                    transform=lambda s, r: inner.transform(s, (r + 1) % world)))
        model.load_weights(ckpt, torch.device("cpu"))
        ref = json.loads((pathlib.Path(ckpt) / "oracle_reference.json").read_text())
        run = FakeRunner(model, aligned_block_size(model), num_blocks=64)
        prompts = ref["prompts"]
        out = {"prefill": [], "decode": []}
        with torch.no_grad():
            for r, p in enumerate(prompts):
                ids = p["prompt"]
                out["prefill"].append(run.prefill(r, ids, bucket=-(-len(ids) // 16) * 16)[-1])
            for step in range(ref["new_tokens"] - 1):
                rows = [(r, p["greedy"][step], len(p["prompt"]) + step)
                        for r, p in enumerate(prompts)]
                out["decode"].append(run.decode(rows, len(prompts) + 1))
        if rank == 0:
            torch.save(out, out_path)
    finally:
        dist.destroy_process_group()


def run_tp(world: int, ckpt, out_path, sabotage=None, ep_degree=1):
    import socket

    import torch.multiprocessing as mp

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(tp_worker, args=(world, port, str(ckpt), str(out_path), sabotage, ep_degree),
             nprocs=world, join=True)
    return torch.load(out_path)
