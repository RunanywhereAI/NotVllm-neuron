# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 plugin model vs transformers ``GlmMoeDsaForCausalLM`` at a tiny config, fp32.

Run with a transformers >= 5.15 that has ``indexer_types`` (cross-layer top-k sharing):

    <venv with transformers 5.17>/bin/python personal_reference/glm_moe_dsa/tests/test_oracle.py

* weights: random, every linear the real checkpoint stores in FP8 is block-quantized
  (16x16 here, scale = amax/448 so blocks reach 448 as in the checkpoint), passed through
  ``fp8_le240`` and dequantized for the oracle; the plugin gets the raw FP8 + scales and
  does its own ``fp8_le240``, so both compute with identical weights;
* a fake runner: page-major fp32 buffer with every unwritten byte NaN except the zero
  page; bucket-padded prefill (pads repeat the last position, slot 0); a prefix-continuing
  second prefill chunk; batched decode with a dead padded row (slot 0, stale block ids);
* ``index_topk`` < context so the indexer's top-k actually selects, and a second run with
  ``index_topk`` >= context for the no-top-k shortcut.
"""
from __future__ import annotations

import importlib
import pathlib
import sys
import types

import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
_PKGS = ("vllm_neuron", "vllm_neuron.model", "vllm_neuron.model.glm_moe_dsa",
         "vllm_neuron.functional", "vllm_neuron.functional.vendored_kernels",
         "vllm_neuron.utils", "vllm_neuron.accuracy", "vllm_neuron.nn")


def import_plugin(dotted: str):
    for pkg in _PKGS:
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = [str(ROOT.joinpath(*pkg.split(".")))]
            sys.modules[pkg] = m
    return importlib.import_module(dotted)


M = import_plugin("vllm_neuron.model.glm_moe_dsa.model")
from vllm_neuron.model.glm_moe_dsa.config import GlmMoeDsaArgs  # noqa: E402

from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM  # noqa: E402

BLOCK = (16, 16)
FP8_LINEARS = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj",
               "indexer.wq_b", "indexer.wk", "gate_proj", "up_proj", "down_proj")


def tiny_config(index_topk: int, n_layers: int = 6):
    types_ = ["full", "shared", "full", "shared", "shared", "full"][:n_layers]
    cfg = dict(
        vocab_size=256, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
        num_hidden_layers=n_layers, num_attention_heads=4, num_key_value_heads=4,
        n_shared_experts=1, n_routed_experts=8, routed_scaling_factor=2.5, kv_lora_rank=32,
        q_lora_rank=32, qk_rope_head_dim=16, v_head_dim=16, qk_nope_head_dim=16, n_group=1,
        topk_group=1, num_experts_per_tok=2, norm_topk_prob=True, hidden_act="silu",
        max_position_embeddings=4096, rms_norm_eps=1e-5, first_k_dense_replace=1,
        index_topk=index_topk, index_head_dim=32, index_n_heads=8, indexer_types=types_,
        rope_parameters={"rope_theta": 8000000.0, "rope_type": "default"},
        rope_interleave=True, indexer_rope_interleave=True, tie_word_embeddings=False,
        pad_token_id=0, scoring_func="sigmoid",
        quantization_config={"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
                             "weight_block_size": list(BLOCK)},
    )
    return cfg


def quant_block(w: torch.Tensor):
    bn, bk = BLOCK
    N, K = w.shape
    v = w.view(N // bn, bn, K // bk, bk)
    s = v.abs().amax(dim=(1, 3)) / 448.0
    q = (v / s[:, None, :, None]).to(torch.float8_e4m3fn).view(N, K)
    return q, s


def build(index_topk: int, seed: int = 0, n_layers: int = 6):
    torch.manual_seed(seed)
    raw = tiny_config(index_topk, n_layers)
    hf_cfg = dict(raw)
    hf_cfg.pop("quantization_config")
    config = GlmMoeDsaConfig(**hf_cfg)
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"   # the reference loop, not grouped_mm
    ref = GlmMoeDsaForCausalLM(config).float().eval()
    with torch.no_grad():
        for name, p in ref.named_parameters():
            p.normal_(0, 0.08 if p.dim() > 1 else 0.3)
            if name.endswith("layernorm.weight") or name.endswith("norm.weight"):
                p.copy_(1 + 0.2 * torch.randn_like(p))
            if name.endswith("indexer.weights_proj.weight"):
                # positive head weights: a score is exactly 0 only if every head's ReLU is,
                # and exact-zero ties at the top-k boundary are broken arbitrarily by both
                # sides (a reference ambiguity, not a defect)
                p.abs_()
        for name, b in ref.named_buffers():
            if name.endswith("e_score_correction_bias"):
                b.copy_(0.05 * torch.randn_like(b))
    sd = ref.state_dict()
    ckpt, oracle_sd = {}, {}
    I = config.moe_intermediate_size
    for name, t in sd.items():
        if name.endswith("experts.gate_up_proj"):
            pre = name[: -len("gate_up_proj")]
            for e in range(t.shape[0]):
                ckpt[f"{pre}{e}.gate_proj.weight"] = t[e, :I]
                ckpt[f"{pre}{e}.up_proj.weight"] = t[e, I:]
            continue
        if name.endswith("experts.down_proj"):
            pre = name[: -len("down_proj")]
            for e in range(t.shape[0]):
                ckpt[f"{pre}{e}.down_proj.weight"] = t[e]
            continue
        if "rotary_emb" in name:
            continue
        ckpt[name] = t
    # quantize what the checkpoint stores in FP8
    out = {}
    deq = {}
    for name, t in ckpt.items():
        if name.endswith(".weight") and t.dim() == 2 and any(f".{k}.weight" in name for k in FP8_LINEARS):
            q, s = quant_block(t.float())
            out[name] = q
            out[name[: -len("weight")] + "weight_scale_inv"] = s
            q2, s2 = M.fp8_le240(q, s, BLOCK)
            deq[name] = M.dequant_block(q2, s2, BLOCK, torch.float32)
        else:
            out[name] = t
            deq[name] = t
    # load the oracle with exactly the plugin's weights
    for name, t in sd.items():
        if name.endswith("experts.gate_up_proj"):
            pre = name[: -len("gate_up_proj")]
            oracle_sd[name] = torch.stack([torch.cat([deq[f"{pre}{e}.gate_proj.weight"],
                                                      deq[f"{pre}{e}.up_proj.weight"]]) for e in range(t.shape[0])])
        elif name.endswith("experts.down_proj"):
            pre = name[: -len("down_proj")]
            oracle_sd[name] = torch.stack([deq[f"{pre}{e}.down_proj.weight"] for e in range(t.shape[0])])
        elif name in deq:
            oracle_sd[name] = deq[name]
        else:
            oracle_sd[name] = t
    ref.load_state_dict(oracle_sd, strict=True)
    # transformers builds q_a_layernorm / kv_a_layernorm with RMSNorm's default eps 1e-6;
    # vLLM's deepseek_v2 (GLM's serving reference) passes config.rms_norm_eps (1e-5), and
    # so does the plugin. The two references disagree; follow vLLM, and align the oracle
    # so this test checks everything else exactly. (Effect: ~1e-5 relative on logits.)
    for layer in ref.model.layers:
        layer.self_attn.q_a_layernorm.variance_epsilon = config.rms_norm_eps
        layer.self_attn.kv_a_layernorm.variance_epsilon = config.rms_norm_eps
    return raw, ref, out


class FakeRunner:
    def __init__(self, model, block_size: int, num_blocks: int, max_blocks: int):
        self.m, self.B, self.nb = model, block_size, max_blocks
        pe = model.layout.page_elems
        pages = torch.full((num_blocks + 2, pe), float("nan"))
        zero_page, _ = M.reserved_pages(num_blocks + 2)
        pages[zero_page] = 0
        model.bind_kv_cache({M.CACHE_LAYER: [pages]})
        self.free = list(range(1, num_blocks))   # block 0 is vLLM's null block
        torch.manual_seed(1)
        self.free = [self.free[i] for i in torch.randperm(len(self.free)).tolist()]
        self.tables: dict[int, list[int]] = {}

    def _slots(self, rid, positions):
        tab = self.tables.setdefault(rid, [])
        out = []
        for p in positions:
            while len(tab) <= p // self.B:
                tab.append(self.free.pop())
            out.append(tab[p // self.B] * self.B + p % self.B)
        return out

    def _bt(self, rid):
        tab = self.tables.get(rid, [])
        return tab + [0] * (self.nb - len(tab))

    def prefill(self, rid, tokens, start, bucket):
        L = len(tokens)
        pos = list(range(start, start + L))
        slots = self._slots(rid, pos)
        pad = bucket - L
        ids = torch.tensor(tokens + [0] * pad)
        positions = torch.tensor(pos + [pos[-1]] * pad)
        slot = torch.tensor(slots + [0] * pad)
        md = {M.CACHE_LAYER: {"block_table_tensor": torch.tensor([self._bt(rid)]),
                              "slot_mapping": slot, "max_query_len": bucket,
                              "decode_token_threshold": 1}}
        return self.m(ids, positions, attn_metadata=md)[:L]

    def decode(self, reqs, n_rows):
        """reqs: list of (rid, token, position); padded to ``n_rows`` with dead rows."""
        ids, pos, slot, bt = [], [], [], []
        for rid, tok, p in reqs:
            ids.append(tok)
            pos.append(p)
            slot.append(self._slots(rid, [p])[0])
            bt.append(self._bt(rid))
        for _ in range(n_rows - len(reqs)):
            ids.append(0); pos.append(0); slot.append(0)
            bt.append([self.tables[reqs[0][0]][0]] * self.nb)        # stale ids
        md = {M.CACHE_LAYER: {"block_table_tensor": torch.tensor(bt), "slot_mapping": torch.tensor(slot),
                              "max_query_len": 1, "decode_token_threshold": 1}}
        return self.m(torch.tensor(ids), torch.tensor(pos), attn_metadata=md)[: len(reqs)]


def run(index_topk: int, block_size: int = 4, max_blocks: int = 10, n_layers: int = 6):
    raw, ref, ckpt = build(index_topk, n_layers=n_layers)
    args = GlmMoeDsaArgs.from_hf(raw)
    model = M.GlmMoeDsaModel(args, block_size=block_size, cache_dtype=torch.float32)
    model.set_dtype(torch.float32)
    model.load_from(M.DictSource(ckpt, BLOCK))
    model.eval()
    fr = FakeRunner(model, block_size, num_blocks=40, max_blocks=max_blocks)
    torch.manual_seed(3)
    seqA = torch.randint(1, 256, (30,)).tolist()
    seqB = torch.randint(1, 256, (34,)).tolist()
    with torch.no_grad():
        refA = ref(torch.tensor([seqA])).logits[0]
        refB = ref(torch.tensor([seqB])).logits[0]
    errs = []

    def cmp(tag, got, want):
        e = (got - want).abs().max().item() / want.abs().max().item()
        errs.append(e)
        print(f"  {tag:34s} rel max err {e:.2e}")

    # A: one-shot prefill of 22 tokens in a 24 bucket
    cmp("A prefill 22 (bucket 24)", fr.prefill(0, seqA[:22], 0, 24), refA[:22])
    # B: prefill 13 in 16, then a prefix-continuing chunk of 10 in 16 (starts mid-block)
    cmp("B prefill 13 (bucket 16)", fr.prefill(1, seqB[:13], 0, 16), refB[:13])
    cmp("B continue 13..22 (bucket 16)", fr.prefill(1, seqB[13:23], 13, 16), refB[13:23])
    # batched decode, with a dead padded row
    for k in range(8):
        a, b = 22 + k, 23 + k
        got = fr.decode([(0, seqA[a], a), (1, seqB[b], b)], n_rows=3)
        cmp(f"decode step {k} (A@{a}, B@{b})", got, torch.stack([refA[a], refB[b]]))
    return max(errs)


if __name__ == "__main__":
    worst = 0.0
    for nl in (1, 3):
        print(f"index_topk=8, n_layers={nl}")
        run(8, n_layers=nl)
    for topk in (8, 64):
        print(f"index_topk={topk} (context the block table addresses: 40)")
        worst = max(worst, run(topk))
    print(f"WORST {worst:.2e}")
    assert worst < 1e-5, worst
    print("PASS")
