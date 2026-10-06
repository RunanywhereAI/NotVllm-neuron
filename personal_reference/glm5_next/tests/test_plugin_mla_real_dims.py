# SPDX-License-Identifier: Apache-2.0
"""One sparse-MLA layer at the RELEASED dimensions, through the paged framework path.

Run: python3 -m pytest personal_reference/glm5_next/tests/test_plugin_mla_real_dims.py -v -s

The model-level tests run a tiny config, which could hide a shape-specific defect: the
real page has two row widths (512 latent, 128 index) that must both tile it, a block of
96 tokens (not a power of two), 24 pools per page, and an indexer whose dense-exact
ceiling is 2051, not 11. Here: 64 heads, kv_lora_rank 512, indexer 32 x 128, topk 2048,
kpool 4, block 96 -- prefill 2045 tokens in a 2048 bucket, then decode across 2051 at
batch 1 of 2 (one padded row), every unwritten page byte NaN. ~2 s on a laptop.
"""
from __future__ import annotations

import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from glm5_next import reference as R  # noqa: E402
from glm5_next.tests import plugin_harness as H  # noqa: E402

MLA = H.import_plugin("vllm_neuron.model.glm5_next.mla")
CL = H.import_plugin("vllm_neuron.model.glm5_next.cache_layout")

KEYS = ("num_attention_heads q_lora_rank kv_lora_rank qk_nope_head_dim qk_rope_head_dim "
        "v_head_dim hidden_size rms_norm_eps index_topk index_kpool index_n_heads "
        "index_head_dim").split()


def test_real_dims_prefill_then_decode_across_the_2051_ceiling(monkeypatch):
    ocfg = R.FlashCfg()
    o = R.SparseMLAttention(ocfg).eval()
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for n, p in o.named_parameters():
            r = torch.randn(p.shape, generator=g)
            if "norm" in n and n.endswith("weight"):
                p.copy_(1 + 0.1 * r)
            elif n.endswith("bias"):
                p.copy_(0.1 * r)
            elif n.endswith("ape"):
                p.copy_(0.5 * r)
            else:
                p.copy_(r / math.sqrt(p.shape[-1]))
    cfg = type("C", (), {k: getattr(ocfg, k) for k in KEYS})()
    m = MLA.Glm5NextSparseMLA(cfg, layer_idx=3).eval()
    m.load_state_dict(o.state_dict())

    B, S0, steps, T = 96, 2045, 10, 2048
    lay = CL.LatentPageLayout.from_config(cfg, B, torch.float32)
    assert lay.total_elems % 512 == 0 and lay.total_elems % 128 == 0
    nb = math.ceil((S0 + steps) / B) + 1
    pages = torch.full((nb + 3, lay.total_elems), float("nan"))
    pages[nb + 1] = 0                                           # the zero page
    m.bind_latent_pages(pages)
    blocks = list(range(1, nb + 1))
    width = nb + 4
    x = torch.randn(1, S0 + steps, ocfg.hidden_size, generator=g)
    sel = []
    monkeypatch.setattr(MLA, "_capture_tensor",
                        lambda name, t: sel.append(t) if name.endswith("topk_indices") else None)
    with torch.no_grad():
        want, _ = o(x)
        pos = torch.tensor(list(range(S0)) + [S0 - 1] * (T - S0))
        slots = torch.tensor([blocks[t // B] * B + t % B for t in range(S0)] + [0] * (T - S0))
        md = {m.layer_name: dict(block_table_tensor=torch.tensor([blocks + [0] * 4]),
                                 slot_mapping=slots, max_query_len=T,
                                 decode_token_threshold=1, block_size=B)}
        xp = torch.cat([x[0, :S0], torch.zeros(T - S0, ocfg.hidden_size)])
        got = m(xp, pos, md)[:S0]
        torch.testing.assert_close(got, want[0, :S0], rtol=1e-4, atol=1e-4)
        for s in range(steps):
            t = S0 + s
            md = {m.layer_name: dict(
                block_table_tensor=torch.tensor([blocks + [0] * 4, [blocks[0]] * width]),
                slot_mapping=torch.tensor([blocks[t // B] * B + t % B, 0]),
                max_query_len=1, decode_token_threshold=1, block_size=B)}
            out = m(torch.stack([x[0, t], torch.zeros(ocfg.hidden_size)]),
                    torch.tensor([t, 0]), md)
            assert torch.isfinite(out).all(), f"step {s}: non-finite (padded row too)"
            torch.testing.assert_close(out[0], want[0, t], rtol=1e-4, atol=1e-4)
    # non-vacuity: the last decode step (L = 2055) really is sparse
    last = sel[-1][0, 0]                                       # live row
    kept = int((last >= 0).sum())
    print(f"\n  L={S0 + steps}: {kept} tokens attended of {S0 + steps}")
    assert kept < S0 + steps
