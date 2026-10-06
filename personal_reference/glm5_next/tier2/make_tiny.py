# SPDX-License-Identifier: Apache-2.0
"""Write a tiny GLM-5.3-Flash checkpoint plus the oracle's greedy reference.

Run anywhere with torch + safetensors (no vLLM, no transformers):

    python personal_reference/glm5_next/tier2/make_tiny.py OUT_DIR [REAL_CONFIG_JSON]

``config.json`` is the released config (``zai-org/GLM-5.3-Flash-BF16``, found in the HF
cache if not given) with ``text_config`` shrunk to ``plugin_harness.TINY_HF`` and
``quantization_config`` absent, so vLLM and transformers parse it exactly as they parse
the real one. Weights are the oracle's, randomised hard enough to engage the SwiGLU
clamp, the indexer's sparse regime and non-trivial mHC mixing, and written under the
checkpoint's own names (round-trip-tested against ``weight_converter.py``). Served in
float32, so the comparison with the fp32 oracle is tight.

``oracle_reference.json``: for each prompt, the oracle's greedy continuation and the
log-probability of each chosen token, from a one-shot forward over the whole prefix --
the prefill-then-decode-equals-one-shot criterion, applied end to end.
"""
from __future__ import annotations

import json
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
from glm5_next import reference as R  # noqa: E402
from glm5_next.tests import plugin_harness as H  # noqa: E402

PROMPT_LENS = (9, 17, 29, 6)     # below/above the 11-token dense ceiling; 29 -> crosses a block
NEW_TOKENS = 16
SEED = 0


def main(out: pathlib.Path, real_config: pathlib.Path | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if real_config is None:
        real_config = sorted(pathlib.Path.home().glob(
            ".cache/huggingface/hub/models--zai-org--GLM-5.3-Flash-BF16/snapshots/*/config.json"))[0]
    cfg = json.loads(pathlib.Path(real_config).read_text())
    assert "quantization_config" not in cfg and "quantization_config" not in cfg["text_config"]

    text_ns = H.hf_text_config(dtype=torch.float32)
    n = text_ns.num_hidden_layers
    tiny = dict(cfg["text_config"])
    for k, v in vars(text_ns).items():
        if k != "dtype":
            tiny[k] = v
    tiny["dtype"] = "float32"
    tiny["num_key_value_heads"] = text_ns.num_attention_heads
    tiny["qk_head_dim"] = text_ns.qk_nope_head_dim
    tiny["pad_token_id"] = 0
    tiny["eos_token_id"] = [text_ns.vocab_size - 1]
    tiny["num_nextn_predict_layers"] = 0
    lac = dict(text_ns.linear_attn_config)
    lac["kda_layers"] = [i for i in range(n) if i % 4 != 3]
    lac["full_attn_layers"] = [i for i in range(n) if i % 4 == 3]
    tiny["linear_attn_config"] = lac
    cfg["text_config"] = tiny
    cfg["tie_word_embeddings"] = False
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    # the oracle, at the plugin config's values
    CFG = H.import_plugin("vllm_neuron.model.glm5_next.config")
    text = CFG.Glm5NextTextConfig.from_hf(H.hf_text_config(dtype=torch.float32))
    oracle = R.FlashTextModel(H.oracle_cfg(text, R), vocab=text.vocab_size).eval()
    H.randomize_(oracle, seed=SEED)
    # float32 on disk: the reference below runs these exact weights, unrounded
    tensors = H.hf_checkpoint_from_oracle(oracle.state_dict(), text.n_routed_experts,
                                          dtype=torch.float32)
    H.write_checkpoint(tensors, out)

    g = torch.Generator().manual_seed(1234)
    ref = {"seed": SEED, "new_tokens": NEW_TOKENS, "prompts": []}
    with torch.no_grad():
        for L in PROMPT_LENS:
            ids = torch.randint(1, text.vocab_size - 1, (L,), generator=g).tolist()
            seq, chosen_lp, top_gap = list(ids), [], []
            for _ in range(NEW_TOKENS):
                logits = oracle(torch.tensor([seq]))[0, -1]
                lp = torch.log_softmax(logits.float(), -1)
                top2 = lp.topk(2)
                tok = int(top2.indices[0])
                chosen_lp.append(float(top2.values[0]))
                top_gap.append(float(top2.values[0] - top2.values[1]))
                seq.append(tok)
            ref["prompts"].append({"prompt": ids, "greedy": seq[L:], "logprob": chosen_lp,
                                   "top1_top2_gap": top_gap})
    (out / "oracle_reference.json").write_text(json.dumps(ref, indent=1))
    print(f"wrote {out}: {len(tensors)} tensors; min top1-top2 logprob gap "
          f"{min(min(p['top1_top2_gap']) for p in ref['prompts']):.3e}")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else None)
