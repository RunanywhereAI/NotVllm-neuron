# DeepSeek-V4.1-Flash on Trainium — scope

Written 2026-10-06. Sources: the live `config.json`, all 48 safetensors shard
headers (parameter counts below are counted, not estimated), DeepSeek's reference
`inference/model.py` (1,309 lines), vLLM `vllm/models/deepseek_v41/`, NotSglang
`models/deepseek_v4.py`, and this plugin at `dev` @ 2aad4b1.

## Verdict

Feasible, and **less risky than GLM-5.3-Flash on the kernel side**: there is no
linear attention, so nothing like the KDA kernels is needed, and the attention
shape is one we are already building for GLM. The two hard problems are
**capacity on Trn2** and **building a CPU oracle**. It should be built **after**
the GLM integration lands, because it reuses GLM's mHC, its 512-wide single-KV
attention path and its latent cache-write kernel.

## This corrects earlier notes

AGENTS.md and STRATEGY.md filed V4.1 as "hardest, no reference, deprioritize",
and claimed every DeepSeek-family model has a 576-wide attention head that fails
nkilib's decode kernel. **Both were wrong for V4.1.**

- **Head dim is 512, not 576.** `head_dim: 512` *includes* the 64 rotary dims
  (`qk_nope_head_dim = head_dim - qk_rope_head_dim` in both NotSglang and the
  reference). 576 was DeepSeek-V3/V3.2 and GLM-5.3 (`kv_lora_rank 512 + rope 64`).
  512 passes both nkilib TKG asserts (`_MAX_D_HEAD = 512`; multiple of 128).
- **A reference exists.** DeepSeek's own `inference/model.py` is readable torch.
  It is CUDA-bound only through five tilelang kernels in `kernel.py`
  (`act_quant`, `fp4_act_quant`, `fp8_gemm`, `sparse_attn`, `hc_split_sinkhorn`),
  each with simple semantics that a torch shim can reproduce.
- **"Causal encoder-decoder" is cross-layer KV sharing**, not an extra
  architecture. Only 4 layers (`kv_source_layer_ids [2,8,14,20]`) compute
  compressed KV; the other 36 read those caches.

## Architecture, as the reference computes it

40 layers, hidden 5120, 64 query heads. Per layer:

- **Attention — MQA, one 512-wide KV head, K and V are the same tensor.**
  Low-rank Q (`q_lora_rank 1280`). RoPE on the last 64 dims of q and kv; inverse
  RoPE on the last 64 dims of the *output* (V carries rotated dims). A learned
  per-head attention sink. Grouped low-rank output: `o_groups 8`,
  `o_lora_rank 1024`, block-diagonal einsum then `wo_b`.
- **Each query attends to ≤ 640 KV entries**: a 128-token sliding window (every
  layer has a ring buffer) plus up to `index_topk = 512` compressed positions.
- **Compressor** (source layers only): softmax-gated pooling of `compress_ratio`
  consecutive tokens into one 512-wide latent. Ratio 2 for layers 2–19, ratio 1
  (no pooling) for 20–39, none (window only) for 0–1. Carries a partial group across
  decode steps.
- **Indexer** (`index_source_layer_ids [2,8,14,20,24,28,32,36]`; others reuse the
  published top-k): `relu(q·k)` weighted over 32 heads × 128 dims, top-512.
  **Two-level**: layer 20 picks the top 2,048 blocks of 8 positions, and later index
  layers search only inside them.
- **MoE on every layer**: 384 routed experts top-6 + 1 shared, `moe_intermediate
  2304`, score `sqrt(softplus(x))`, bias selects but does not scale, normalised top-k
  times `routed_scaling_factor 1.5`, SwiGLU clamp 10.0.
- **mHC**: `hc_mult 4`, `hc_sinkhorn_iters 20`, `hc_eps 1e-6` — the same config as
  GLM-5.3-Flash. vLLM implements both models on the same `kernels/mhc` family, so
  GLM's implementation should transfer **[unverified until diffed against the
  oracle]**.
- **Engram** at layers 1 and 14: n-gram hashing over token ids (up to 4-grams,
  8 heads), lookups into two ~384M-row × 256 tables, added into the residual.
- `rms_norm_eps 1e-20`; YaRN ×16 on compressed layers, base RoPE on window-only ones.
- Descoped for first bring-up: vision, MTP (3 layers), DSpark speculative decoding.

## Parameters and capacity (counted from shard headers)

| Part | Params | Stored as | BF16 size |
|---|---:|---|---:|
| Routed experts | 557.2 B | FP4 (e2m1 in int8, e8m0 scale per 32 = MXFP4) | 1,114 GB |
| Engram tables | 196.9 B | FP8 e4m3 + e8m0 per 32 | 394 GB |
| Attention + compressor + indexer | 5.5 B | FP8 e4m3, 32×32 blocks, e8m0 | 11 GB |
| Shared experts / embed+head / other | 3.1 B | FP8 / BF16 | 6 GB |
| **Total (text)** | **763 B** | ~510 GB on disk | ~1.53 TB |

**The plugin cannot load FP4 experts on Trn2** — `gpt_oss/factory.py`:
"quantization='mxfp4' is not supported on TRN2". MXFP4/MXFP8 are Trn3-only. So on
Trn2 the experts must be dequantized to BF16.

**trn2.48xlarge** (64 ranks × ~25.8 GB) **[estimate, arithmetic only]**:
- BF16 experts 17.4 GB/rank + Engram kept FP8 ~3.2 GB/rank + the rest ~0.3 GB/rank
  ≈ **20.9 GB/rank, leaving ~4.9 GB** for KV, activations, NEFFs and runtime.
  Engram is a gather, not a matmul, so it can stay FP8 on device and be scaled
  after lookup.
- KV is small: V4.1's whole design is KV compression. In BF16 one sequence costs
  ~0.42 GB/rank at 128K and ~3.4 GB/rank at 1M — versus GLM-5.3-Flash's
  ~11.5 GB/rank at 1M. So the binding constraint is weights, not KV: roughly batch
  8–10 at 128K.
- If 4.9 GB proves too tight, move Engram to host DRAM (2 TiB on the instance) for
  ~3.2 GB/rank more.
- **Trn3 is the natural target**: native MXFP4 experts (~280 GB) and MXFP8 Engram,
  roughly the release checkpoint's ~510 GB. **[unverified]** — trn3 instance specs
  were not checked.

## What it will take

Reused from the GLM work (in flight or done): mHC; the 512-wide single-KV-head
attention path on nkilib TKG, which is already validated at q_head = 64 — V4.1 also
has 64 query heads; one tensor serving as both K and V (bit-identical, proven); the
latent-only cache-write kernel; the FP8 block-dequant converter pattern; the
front-end registration pattern; captured-intermediate validation.

New work:
1. **Oracle.** Torch shims for the five tilelang kernels, so DeepSeek's reference
   runs on CPU; cross-checked against a second implementation (the closed
   community transformers PR #48721, +5,990 lines, which claims CPU verification
   against the reference). Everything else is validated against this, so it comes
   first. Independent of the plugin, so it can start now.
2. **Converter.** FP8 32×32/e8m0 → BF16; MXFP4 → BF16 for experts; Engram stays FP8
   plus scales. Drop `quantization_config` (vLLM's front end refuses fp8 configs in
   CPU mode). Stream shard by shard: the BF16 output is ~1.1 TB.
3. **Front end.** No transformers config class exists, so register our own;
   registration in `pre_register_and_update`. Unlike GLM, `head_dim` is real and
   `num_key_value_heads` is 1, so no MLA allowlist patch is expected
   **[unverified]**.
4. **Caches.** Sliding-window ring per layer (the plugin has `SlidingWindowSpec`);
   compressed KV and index-K caches on 4 source layers shared by the other 36;
   compressor partial-group state (fixed size, like recurrent state).
5. **Model.** Compressor, the two-level indexer, sparse attention over window plus
   top-512 with a sink and inverse RoPE, grouped low-rank O, sqrt-softplus MoE,
   Engram hashing and sharded lookup, mHC.
6. **Validation.** Same tiers as GLM: oracle on the Mac, CPU mode through vLLM on
   the Linux box at batch > 1, CPU compile, then device.

Relative size: roughly the GLM port **minus its hardest part (the KDA kernels)**,
**plus Engram and the converter**. It reuses GLM's attention and cache plumbing,
so it cannot be finished before those land.

## Risks

- **Capacity on Trn2** — ~4.9 GB/rank of headroom is an estimate; runtime and
  NEFF overheads are unmeasured.
- **Quantization regime.** The reference quantizes window KV to FP8, compressed KV
  and index keys to FP4 — and the model was trained with fake quantization (the
  Unsloth PR keeps "QAT fake-quant" out of `torch.compile`). Running BF16 KV
  departs from the trained regime. Probably harmless, possibly beneficial —
  **[unverified]**, needs a quality eval, not just a numerics diff.
- **Sparse prefill.** Each prefill query gathers a different index set.
  `mla_sparse_attention_cte` is installed (`nkilib/experimental/mla/deepseek/`),
  but whether it accepts a 512-wide head with no separate rope part is
  **[unverified]**; the torch path works regardless.
- **Same compiler risk as Qwen3.8-27B** (`NCC_ISMP902`) until a vendor toolchain
  compiles something.
