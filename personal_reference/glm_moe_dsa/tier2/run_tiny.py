# SPDX-License-Identifier: Apache-2.0
"""Tier 2: a tiny block-FP8 GLM-5.3 checkpoint through vLLM in VLLM_NEURON_CPU_MODE=1,
greedy tokens and logprobs vs transformers (the oracle aligned as in tests/test_oracle.py).

    PYTHONPATH=/data/glm53-wt VLLM_NEURON_CPU_MODE=1 python run_tiny.py make DIR
    PYTHONPATH=/data/glm53-wt VLLM_NEURON_CPU_MODE=1 python run_tiny.py run DIR [TP]

``make`` writes the checkpoint the way the released one is laid out (per-expert
``gate/up/down_proj``, ``weight_scale_inv`` per 16x16 block, scale = amax/448) and a
served ``config.json`` (``original_quantization_config``), plus oracle tokens/logprobs.
``run`` serves it with TP (and EP = TP) and compares.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import sys

import torch

BLOCK = (16, 16)
FP8_LINEARS = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj",
               "indexer.wq_b", "indexer.wk", "gate_proj", "up_proj", "down_proj")
NEW = 12


def make(out: str):
    from safetensors.torch import save_file
    from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM

    from vllm_neuron.model.glm_moe_dsa.model import dequant_block, fp8_le240

    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    cfg = dict(
        architectures=["GlmMoeDsaForCausalLM"], model_type="glm_moe_dsa",
        vocab_size=256, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
        num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=4, head_dim=16,
        n_shared_experts=1, n_routed_experts=8, routed_scaling_factor=2.5, kv_lora_rank=32,
        q_lora_rank=32, qk_rope_head_dim=16, v_head_dim=16, qk_nope_head_dim=16, n_group=1,
        topk_group=1, num_experts_per_tok=2, norm_topk_prob=True, hidden_act="silu",
        max_position_embeddings=4096, rms_norm_eps=1e-5, first_k_dense_replace=1,
        index_topk=16, index_head_dim=32, index_n_heads=8,
        indexer_types=["full", "shared", "full", "shared", "shared", "full"],
        rope_parameters={"rope_theta": 8000000.0, "rope_type": "default"},
        rope_interleave=True, indexer_rope_interleave=True, tie_word_embeddings=False,
        pad_token_id=0, bos_token_id=0, eos_token_id=1, scoring_func="sigmoid", topk_method="noaux_tc",
        dtype="float32", num_nextn_predict_layers=0,
    )
    hf = {k: v for k, v in cfg.items() if k not in ("architectures", "model_type", "head_dim")}
    config = GlmMoeDsaConfig(**hf)
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    ref = GlmMoeDsaForCausalLM(config).float().eval()
    with torch.no_grad():
        for name, p in ref.named_parameters():
            p.normal_(0, 0.08 if p.dim() > 1 else 0.3)
            if name.endswith("norm.weight"):
                p.copy_(1 + 0.2 * torch.randn_like(p))
            if name.endswith("indexer.weights_proj.weight"):
                p.abs_()
        for name, b in ref.named_buffers():
            if name.endswith("e_score_correction_bias"):
                b.copy_(0.05 * torch.randn_like(b))
    I = config.moe_intermediate_size
    ckpt = {}
    for name, t in ref.state_dict().items():
        if "rotary_emb" in name:
            continue
        if name.endswith("experts.gate_up_proj"):
            pre = name[: -len("gate_up_proj")]
            for e in range(t.shape[0]):
                ckpt[f"{pre}{e}.gate_proj.weight"] = t[e, :I]
                ckpt[f"{pre}{e}.up_proj.weight"] = t[e, I:]
        elif name.endswith("experts.down_proj"):
            pre = name[: -len("down_proj")]
            for e in range(t.shape[0]):
                ckpt[f"{pre}{e}.down_proj.weight"] = t[e]
        else:
            ckpt[name] = t
    stored, deq = {}, {}
    for name, t in ckpt.items():
        if name.endswith(".weight") and t.dim() == 2 and any(f".{k}.weight" in name for k in FP8_LINEARS):
            bn, bk = BLOCK
            N, K = t.shape
            v = t.float().view(N // bn, bn, K // bk, bk)
            s = v.abs().amax(dim=(1, 3)) / 448.0
            q = (v / s[:, None, :, None]).to(torch.float8_e4m3fn).view(N, K)
            stored[name], stored[name[:-6] + "weight_scale_inv"] = q, s
            q2, s2 = fp8_le240(q, s, BLOCK)
            deq[name] = dequant_block(q2, s2, BLOCK, torch.float32)
        else:
            stored[name] = t.contiguous()
            deq[name] = t
    # oracle weights = what the plugin computes with
    sd = {}
    for name, t in ref.state_dict().items():
        if name.endswith("experts.gate_up_proj"):
            pre = name[: -len("gate_up_proj")]
            sd[name] = torch.stack([torch.cat([deq[f"{pre}{e}.gate_proj.weight"], deq[f"{pre}{e}.up_proj.weight"]])
                                    for e in range(t.shape[0])])
        elif name.endswith("experts.down_proj"):
            pre = name[: -len("down_proj")]
            sd[name] = torch.stack([deq[f"{pre}{e}.down_proj.weight"] for e in range(t.shape[0])])
        else:
            sd[name] = deq.get(name, t)
    ref.load_state_dict(sd, strict=True)
    for layer in ref.model.layers:           # vLLM's eps for the latent norms (see test_oracle)
        layer.self_attn.q_a_layernorm.variance_epsilon = config.rms_norm_eps
        layer.self_attn.kv_a_layernorm.variance_epsilon = config.rms_norm_eps
    names = sorted(stored)
    half = len(names) // 2
    shards = {"model-00001-of-00002.safetensors": names[:half], "model-00002-of-00002.safetensors": names[half:]}
    wm = {}
    for f, ns in shards.items():
        save_file({n: stored[n] for n in ns}, str(out / f))
        wm.update({n: f for n in ns})
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    cfg["original_quantization_config"] = {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
                                           "weight_block_size": list(BLOCK)}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    torch.manual_seed(5)
    prompts = []
    for L in (5, 20, 37, 600):   # 600 > the 512 bucket: a continuing segment
        p = torch.randint(2, 256, (L,)).tolist()
        ids = list(p)
        lps = []
        with torch.no_grad():
            for _ in range(NEW):
                logits = ref(torch.tensor([ids])).logits[0, -1]
                lp = torch.log_softmax(logits.double(), -1)
                t = int(lp.argmax())
                lps.append(float(lp[t]))
                ids.append(t)
        prompts.append({"prompt": p, "greedy": ids[L:], "logprob": lps})
        print(f"oracle prompt {L}: {ids[L:]}")
    (out / "oracle_reference.json").write_text(json.dumps({"new_tokens": NEW, "prompts": prompts}))


def run(ckpt: str, tp: int, seqs: int = 4):
    from vllm import LLM, SamplingParams

    ref = json.loads((pathlib.Path(ckpt) / "oracle_reference.json").read_text())
    llm = LLM(model=ckpt, skip_tokenizer_init=True, dtype="float32", max_model_len=1024,
              max_num_seqs=seqs, tensor_parallel_size=tp, enable_expert_parallel=tp > 1,
              enable_prefix_caching=False, enforce_eager=True, max_logprobs=1, block_size=16,
              max_num_batched_tokens=512,
              additional_config={"neuron_config": {"on_device_sampling_config": None, "ep_degree": tp}},
              async_scheduling=False, num_gpu_blocks_override=128)
    sp = SamplingParams(max_tokens=ref["new_tokens"], temperature=0.0, logprobs=1, ignore_eos=True,
                        detokenize=False)
    outs = llm.generate([{"prompt_token_ids": p["prompt"]} for p in ref["prompts"]], sp)
    ok = tot = 0
    worst = 0.0
    for p, out in zip(ref["prompts"], outs):
        got = list(out.outputs[0].token_ids)
        prefix = next((i for i, (a, b) in enumerate(zip(got, p["greedy"])) if a != b), len(got))
        ok += sum(a == b for a, b in zip(got, p["greedy"]))
        tot += len(p["greedy"])
        lps = out.outputs[0].logprobs or []
        d = [abs(lps[i][t].logprob - p["logprob"][i]) for i, t in enumerate(got[:prefix]) if i < len(lps) and t in lps[i]]
        worst = max([worst] + d)
        print(f"prompt {len(p['prompt']):3d}: tokens match {prefix}/{len(p['greedy'])}; max |dlogprob| "
              f"{max(d) if d else math.nan:.2e}; got {got}")
        print("   per-step |dlogprob|", " ".join(f"{x:.1e}" for x in d))
    print(f"TOTAL {ok}/{tot}; worst |dlogprob| {worst:.2e}")
    return 0 if ok == tot and worst < 1e-3 else 1


if __name__ == "__main__":
    os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")
    if sys.argv[1] == "make":
        make(sys.argv[2])
    else:
        sys.exit(run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1,
                     int(sys.argv[4]) if len(sys.argv) > 4 else 4))
