# SPDX-License-Identifier: Apache-2.0
"""Run the plugin's DeepSeek-V4.1 model on a laptop, against the oracle.

* ``import_plugin`` imports a ``vllm_neuron`` module without running the package
  ``__init__`` files, which pull in vLLM (manylinux only). Same mechanism as the GLM
  harness.
* ``plugin_from_reference`` builds the plugin model with the reference model's weights,
  dequantized to fp32 the way exact mode computes them.
* ``FakeRunner`` stands in for ``NeuronModelRunner``: one int8 buffer shared by the two
  pseudo-layers (vLLM's grouping), page-major views at two dtypes, the zero page and
  sink past ``num_blocks``, global block ids drawn from one free list, padded prefill
  (pads repeat the last position, slot 0), padded decode rows (slot 0, stale block ids),
  the runner's SWA decode trimming, and **hostile pages**: every byte the model did not
  write is 0xFF, NaN in fp32 and bf16, so an unselected read poisons the output.
"""

from __future__ import annotations

import importlib
import math
import pathlib
import sys
import types

import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
_PACKAGES = ("vllm_neuron", "vllm_neuron.model", "vllm_neuron.model.deepseek_v41",
             "vllm_neuron.functional", "vllm_neuron.functional.vendored_kernels",
             "vllm_neuron.utils", "vllm_neuron.accuracy", "vllm_neuron.nn")


def import_plugin(dotted: str):
    for pkg in _PACKAGES:
        if pkg not in sys.modules:
            mod = types.ModuleType(pkg)
            mod.__path__ = [str(ROOT.joinpath(*pkg.split(".")))]
            sys.modules[pkg] = mod
    return importlib.import_module(dotted)


def dequantized_state_dict(model) -> dict:
    """Reference state dict with every FP8/FP4 Linear weight dequantized to fp32 and its
    ``.scale`` dropped. Engram tables keep their FP8 storage and scale."""
    from personal_reference.deepseek_v41 import kernel_torch as kt

    sd = model.state_dict()
    out = {}
    for k, v in sd.items():
        if k.endswith(".scale") and ".engram.embed." not in k:
            continue
        if k.endswith(".weight") and ".engram.embed." not in k:
            s = sd.get(k[: -len("weight")] + "scale")
            if v.dtype == torch.float8_e4m3fn:
                v = kt.dequant_fp8_weight(v, s, 32)
            elif v.dtype == torch.float4_e2m1fn_x2:
                v = kt.dequant_fp4_weight(v, s)
        out[k] = v
    return out


def plugin_from_reference(ref_model, args, block_size: int = 8, dtype=torch.float32):
    mod = import_plugin("vllm_neuron.model.deepseek_v41.model")
    model = mod.DeepseekV41Model(args, block_size=block_size, cache_dtype=dtype).set_dtype(dtype)
    model.load_reference_state_dict(dequantized_state_dict(ref_model))
    return model.eval()


class FakeRunner:
    """Block bookkeeping and per-group ``attn_metadata`` for the two pseudo-layers."""

    def __init__(self, model, num_blocks: int = 256, max_model_len: int = 512):
        from vllm_neuron.model.deepseek_v41.cache_layout import COMPRESSED_LAYER, WINDOW_LAYER

        self.W, self.C = WINDOW_LAYER, COMPRESSED_LAYER
        self.model = model
        spec = {s.name: s for s in model.get_kv_spec().paged_layers}
        self.spec = spec
        page = spec[self.W].page_elems * 4
        assert page == spec[self.C].page_elems * spec[self.C].dtype.itemsize
        self.num_pages = num_blocks + 2
        raw = torch.full((self.num_pages * page,), -1, dtype=torch.int8)        # 0xFF
        model.bind_kv_cache({
            self.W: [raw.view(torch.float32).view(self.num_pages, -1)],
            self.C: [raw.view(spec[self.C].dtype).view(self.num_pages, -1)],
        })
        self.free = list(range(1, num_blocks))     # 0 is vLLM's null block
        self.max_model_len = max_model_len
        self.blocks = {}                           # req -> {group: [ids]}

    def bs(self, g):
        return self.spec[g].block_size

    def _ensure(self, req, upto):
        """Blocks covering positions ``[0, upto)`` in both groups. Ids are drawn from one
        pool and interleaved across groups, as vLLM's shared pool does."""
        tables = self.blocks.setdefault(req, {self.W: [], self.C: []})
        for g in (self.W, self.C):
            while len(tables[g]) * self.bs(g) < upto:
                tables[g].append(self.free.pop(0))

    def share_prefix(self, src, dst, tokens):
        """``dst`` starts with ``src``'s first ``tokens`` positions cached (block-aligned)."""
        for g in (self.W, self.C):
            assert tokens % self.bs(g) == 0
        self.blocks[dst] = {g: list(self.blocks[src][g][: tokens // self.bs(g)])
                            for g in (self.W, self.C)}

    def _slots(self, req, g, pos):
        t = self.blocks[req][g]
        return torch.tensor([t[p // self.bs(g)] * self.bs(g) + p % self.bs(g) for p in pos])

    def _table(self, req, g, width):
        t = self.blocks[req][g] if req is not None else []
        row = t + [0] * (width - len(t))
        return torch.tensor(row[:width])

    def prefill(self, req, tokens, start: int, bucket: int, engram_ids=None):
        """``tokens`` real ids at positions ``start ..``; padded to ``bucket``."""
        n = len(tokens)
        self._ensure(req, start + n)
        pos = list(range(start, start + n)) + [start + n - 1] * (bucket - n)
        md = {}
        for g in (self.W, self.C):
            slots = torch.cat([self._slots(req, g, pos[:n]), torch.zeros(bucket - n, dtype=torch.long)])
            width = math.ceil(self.max_model_len / self.bs(g))
            md[g] = dict(block_table_tensor=self._table(req, g, width).view(1, -1),
                         slot_mapping=slots, max_query_len=bucket, decode_token_threshold=1,
                         block_size=self.bs(g))
        ids = torch.tensor(list(tokens) + [tokens[-1]] * (bucket - n))
        if engram_ids is not None:
            engram_ids = torch.cat([engram_ids, engram_ids[-1:].expand(bucket - n, *engram_ids.shape[1:])])
        h = self.model(ids, torch.tensor(pos), md, engram_ids=engram_ids)
        return self.model.compute_logits(h)[:n]

    def _swa_blocks(self):
        bs, win = self.bs(self.W), self.spec[self.W].sliding_window
        P_MAX = 128
        min_blocks = win // bs + 1
        per = max(P_MAX // bs, 1)
        nb = (min_blocks + per - 1) // per * per
        return min(nb, math.ceil(self.max_model_len / bs))

    def decode(self, reqs, tokens, positions, pad_rows: int = 0, engram_ids=None):
        """One token per request; ``pad_rows`` dead rows appended (slot 0, stale tables)."""
        for r, p in zip(reqs, positions):
            self._ensure(r, p + 1)
        n = len(reqs) + pad_rows
        md = {}
        nbw = self._swa_blocks()
        bs = self.bs(self.W)
        starts = [max(p // bs - nbw + 1, 0) for p in positions] + [0] * pad_rows
        full = [self._table(r, self.W, math.ceil(self.max_model_len / bs)) for r in reqs]
        # stale ids on padded rows: whatever block the first request owns
        full += [full[0]] * pad_rows
        table = torch.stack([f[s0:s0 + nbw] for f, s0 in zip(full, starts)])
        slots = torch.cat([torch.cat([self._slots(r, self.W, [p]) for r, p in zip(reqs, positions)]),
                           torch.zeros(pad_rows, dtype=torch.long)])
        md[self.W] = dict(block_table_tensor=table, slot_mapping=slots, max_query_len=1,
                          decode_token_threshold=1, block_size=bs,
                          swa_kv_pos_offset=torch.tensor([s0 * bs for s0 in starts], dtype=torch.int32))
        wc = math.ceil(self.max_model_len / self.bs(self.C))
        tc = [self._table(r, self.C, wc) for r in reqs] + [self._table(reqs[0], self.C, wc)] * pad_rows
        slots_c = torch.cat([torch.cat([self._slots(r, self.C, [p]) for r, p in zip(reqs, positions)]),
                             torch.zeros(pad_rows, dtype=torch.long)])
        md[self.C] = dict(block_table_tensor=torch.stack(tc), slot_mapping=slots_c, max_query_len=1,
                          decode_token_threshold=1, block_size=self.bs(self.C))
        ids = torch.tensor(list(tokens) + [0] * pad_rows)
        pos = torch.tensor(list(positions) + [0] * pad_rows)
        if engram_ids is not None:
            engram_ids = torch.cat([engram_ids, torch.zeros(pad_rows, *engram_ids.shape[1:],
                                                            dtype=engram_ids.dtype)])
        h = self.model(ids, pos, md, engram_ids=engram_ids)
        return self.model.compute_logits(h)[: len(reqs)]
