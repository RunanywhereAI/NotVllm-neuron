# Serving frontier models on Trainium — strategy and findings

Written 2026-09-25. Fork point: upstream `vllm-project/vllm-neuron` @ `f8abae6`,
branch `release-0.24.0.1.1.0` (vLLM 0.24.0, Neuron SDK 2.32).

**Goal:** serve GLM-5.3, GLM-5.3-Flash, DeepSeek-V4.1-Flash, Qwen3.8-27B and
MiMo-V2.6-Pro on Trainium2/Trainium3.

Everything below is either verified against a repo in this workspace, against a
live config on HuggingFace, or against AWS documentation. Claims that are
estimates or unconfirmed are marked **[unverified]**.

---

## 1. The decision

**Build on the vLLM Neuron plugin.** Two alternatives were evaluated and rejected.

### Rejected: NxD Inference (`neuronx-distributed-inference`)

NxDI is in maintenance mode. AWS's own statement, in
[vllm-neuron issue #32](https://github.com/vllm-project/vllm-neuron/issues/32):

> Starting with Neuron 2.30.0, NxD Inference enters maintenance mode and will only
> receive critical bug fixes and security updates. We are building an enhanced
> vLLM-Neuron plugin.

Neuron **2.32.0 (2026-08-17) removed NxDI from the DLAMIs and DLCs entirely**. The
plugin's own compatibility matrix labels the NxDI-based line (`0.5.3`)
*Maintenance* and the current line *Not required* for NxDI.

Practical consequence: staying on NxDI pins you to Neuron SDK 2.31 permanently —
no new `nkilib` kernels, no compiler fixes, **no Trainium3**.

Neither repo carries a deprecation banner, and the upstream NxDI docs page still
reads as actively developed. This is worth writing down precisely because it is
not discoverable from the code.

### Rejected (for now): an SGLang Neuron backend

Technically feasible, and more so than first assumed. SGLang has a real
out-of-tree platform plugin system (`python/sglang/srt/platforms/`, entry-point
group `sglang.srt.platforms`), four non-CUDA precedents in-tree, and the Ascend
NPU graph runner already does `torch.compile(fullgraph=True, dynamic=False,
backend=<vendor AOT backend>)` over **unmodified** SGLang models. Intel's XPU
backend is 13 files.

Rejected on opportunity cost, not feasibility:

- The decode path does not map onto SGLang's attention interface. `NF.attention_decode`
  takes **hidden states** and fuses RMSNorm + QKV projection + RoPE + QK-norm +
  attention + paged KV write + output projection into one kernel. SGLang's
  `AttentionBackend.forward_decode(q, k, v, layer, forward_batch)` hands you
  already-projected q/k/v against a token-flat pool. You either restructure the
  model layers so the kernel owns the block (which is why AWS rewrote every model
  rather than reusing vLLM's), or use unfused kernels and lose the fusion.
  Prefill (`NF.flash_attention(q, k, v, ...)`) *does* map cleanly.
- No Triton on Trainium: 51 top-level Triton imports in `srt/`, 4 inside `mem_cache/`.
- Everything gained is already available here today.

**[unverified]** Effort estimate for a competent engineer to reach a good
Llama-class SGLang Neuron backend: roughly a quarter, with decode performance the
open risk. This estimate has not been tested by a spike.

Revisit if Trainium proves out economically and owning the scheduler becomes
worth it.

### Deployment topology

Run both runtimes behind `sgl-model-gateway`, which already routes to
OpenAI-compatible backends and has multi-model IGW mode: NotSglang on B200 for
production, `vllm-neuron` workers on trn2. One control plane, no porting, and
Trainium can be evaluated without betting the serving stack on it.

**Trainium is a margin play on a multi-quarter runway, not a serve-customers-today
move.**

---

## 2. What this plugin is

- **No NxDI dependency.** `requirements/core.txt` pulls `vllm==0.24.0`,
  `libtorch-neuronx-lite`, `transformers>=5.5.1,<6.0.0`. That's it.
- **Models are reimplemented Neuron-natively.** Upstream vLLM's
  `model_executor/models/` code is *not* reused. AWS's migration guide states there
  is "no direct port of NxDI modeling classes." So "vLLM supports model X" does not
  transfer — it has to be written against this plugin's building blocks.
- **Status: Beta.** The registry (`vllm_neuron/model/registry.py`) is a hardcoded
  list of **five** entries: `LlamaForCausalLM`, `GptOssForCausalLM`,
  `Eagle3LlamaForCausalLM`, `Qwen3ForCausalLM`, `Qwen3VLForConditionalGeneration`.

### Cost per model, measured from the tree

| Model | LOC | Notes |
|---|---:|---|
| `qwen3/` | ~1.4k | dense, single BF16 path — the floor |
| `gpt_oss/` | ~6.6k | MoE, BF16 + MXFP4 paths; the canonical reference |
| `qwen3_vl/` | ~6k | vision tower + MXFP8 |
| `llama3/` | ~7.5k | all quant paths + Eagle3 |

Budget ~1.5–3k LOC for a dense model reusing existing blocks, more for MoE and per
quantization format.

### Out-of-tree registration

Documented (`docs/design/framework/model_factory_design.md`) but second-class.
`vllm_neuron/utils/vision_utils.py` and `vllm/spec_decode/eagle.py` resolve model
classes via `dict(get_models())` only, so an out-of-tree VLM silently loses
merge-factor and max-pixels handling, and out-of-tree EAGLE3 targets won't
resolve. **For our purposes: add models in-tree in this fork.**

---

## 3. Hard constraints

These are the things that bite, roughly in order of how often.

### 3.1 Ahead-of-time compilation, static shapes

`neuronx-cc` compiles the forward pass to a NEFF binary before serving. Per AWS's
compiler docs, *"models with control-flow and dynamic shapes are not currently
supported."* Change batch size or sequence length by one token and it is a
different binary — hence bucketing.

In practice, inside `forward()`: no `.item()`, no tensor-valued control flow, no
dynamic slicing, no graph breaks. Every conditional branch becomes its own NEFF.
After warmup `fail_on_recompile` is armed, so an uncovered shape is a **hard
failure**, not a slow path.

### 3.2 `head_dim` is capped at 128 — **in prefill only**

`MAX_HEAD_DIM = 128` in `vllm_neuron/functional/attention/attention_cte.py` and
`_MAX_HEAD_DIM = 128` in `attention_segmented_cte.py`. This is `P_MAX`, the SBUF
partition dimension — a hardware property, not a software limit.

**Both of those files are `*_cte` — context encoding, i.e. prefill. The cap does
not apply to decode.** Established by dev3 from `nkilib` source and verified
independently here on 2026-09-25:

- `MAX_HEAD_DIM` appears **nowhere** outside those two prefill files.
- `attention_decode.py` has no head-dim ceiling at all. Its eligibility guard
  `_can_use_attention_block_kernel` (line 1211) rejects only an **odd** `d_head`
  (line 1255, `if d_head % 2 != 0`).
- `nkilib/core/attention/attention_tkg.py:52` sets `_MAX_D_HEAD = 512`, asserted
  at line 1222 as `0 < cfg.d_head <= _MAX_D_HEAD`. TKG is token generation, i.e.
  decode.

So a model with `head_dim` in (128, 512] loses the fused kernel on **prefill**
and keeps it on **decode**. That is a much narrower problem than "falls off the
fused attention path entirely", which is what this section used to say.

Models above it fall off the fused attention path entirely. PR #40 (MiMo-V2.5,
192-wide Q/K) had to run **eager attention with fp32 scores**. The NxDI Qwen3.5
port worked around it by hot-swapping `sys.modules` to a forked `nkilib` with
`_MAX_HEAD_DIM=256`.

**Not a concern for Qwen3.8-27B** (`head_dim: 256`), verified 2026-09-25 against
PR #54 as merged. Its full-attention layers never enter the fused path at all:
there is no `NF.`/`flash_attention`/`segmented_attention` call anywhere in
`model/qwen3_5/model.py` — the sole mention of `NF.flash_attention` is inside a
docstring. Both `forward_prefill` and `forward_decode` compute attention inline
in fp32 (`q.float() @ k.float().transpose(-1, -2)` then `torch.softmax`), so
`MAX_HEAD_DIM` is never reached and there is nothing to work around. The
`sys.modules` hot-swap should **not** be attempted here. PR #54 measures the cost
of the eager path at 6–8% of prefill, i.e. a **performance** item, not a
correctness gate.

Still relevant to **MiMo-V2.6-Pro** (192), which has no such port yet — but per
the above, only on its prefill path. Its decode would keep the fused kernel,
since 192 is well inside `_MAX_D_HEAD = 512`.

### 3.3 No recurrent-state cache (as of this fork point)

`vllm_neuron/model/kv_cache.py` is 33 lines: `LayerSpec(num_kv_heads, head_size,
dtype, sliding_window_size, chunk_size)`. `initialize_kv_cache` handles only
`FullAttentionSpec | SlidingWindowSpec`; anything else raises `NotImplementedError`.
The runner says so at `vllm/worker/neuron_model_runner.py:8189`:

> `# Neuron does not support hybrid models yet, so this is a no-op.`

This blocks every linear-attention model (gated DeltaNet, KDA). **See §5 — PR #54
fixes it.**

### 3.4 No MLA

`kv_lora_rank` appears nowhere in the plugin. The
`(2, num_blocks, num_kv_heads, block_size, head_size)` allocation wastes the V half
for a latent cache. Blocks GLM-5.3 and DeepSeek-V4.1-Flash.

`nkilib` has MLA kernels but they are **prefill only** — every file under
`experimental/mla/deepseek/` is `*_cte` (`mla_qkv_cte`, `mla_sparse_attention_cte`,
`mla_vup_oproj_cte`). **There is no MLA decode kernel.** This is the deepest hole
on our roadmap.

### 3.5 Chunked prefill is unsupported

Mixing prefill and decode in one batch is ❌ in the feature table — ragged shapes
defeat bucketing.

**Prefix caching, by contrast, works.** APC is supported; it just requires
segmented prefill (`neuron_model_runner.py:680`), and `NF.segmented_attention`
handles per-chunk APC offsets. Do not confuse the two.

### 3.6 Compiler pathologies eat schedule

From PR #40's write-up, all real and none of it model math:

- `Tensor.index_put_` on a KV cache is **not** an in-place write under `neuronx-cc`.
  XLA lowers it to a full-pool `scatter`; `AliasingOutputRewritePass` only aliases
  the last write per placeholder, so with 48 layers sharing 18 placeholders, 78
  intermediate full-pool copies survived — ~124 GB of traffic per decode step.
  Fixed by a NKI scatter kernel that *returns* the cache tensors. TPOT 199 → 128.5 ms.
- `bwmm_shard_on_block` faults (`indirect memory copy via vector DGE out-of-bound`)
  once MoE block count passes ~11.
- `--modular-flow-mac-threshold=10`, passed unconditionally, breaks codegen on
  DeltaNet decode.

---

## 4. Kernels: less work than assumed

`nkilib` is Apache-2.0, full source at
[aws-neuron/nki-library](https://github.com/aws-neuron/nki-library). It ships
bundled inside `neuronx-cc`. **There is no separate AWS `nki-library` pip
package**, and the way it is missing is a trap rather than an error (checked
2026-09-25):

- On the Neuron index, `https://pip.repos.neuron.amazonaws.com/nki-library/`
  returns **404**.
- On **PyPI**, `nki-library` **exists** — version 0.0.2, no summary, no author,
  no homepage, two files, releases 0.0.1 and 0.0.2. That is a placeholder, not
  AWS's library.

So `pip install nki-library` — which an earlier draft of this section
recommended — **succeeds and installs a stub**. It does not fail loudly. Whatever
`neuronx-cc` bundles is what you actually get.

> **Correction, 2026-09-25.** The table below was built from a *source checkout*
> of nki-library. It does **not** describe the installed package. Verified
> against the `nkilib` bundled with `neuronx-cc 2.27.5334.0` — version
> `0.0.0.0dev0+3b542be2`, built Jul 15 2026 — on the dev box:
>
> `experimental/` contains: `attention`, `attention_mxfp8`, `benchmark`,
> `collectives`, `conv`, `deformable_attention`, `dynamic_shapes`, `foreach`,
> `loss`, `matmul_mxfp8`, `misc`, `mla`, `mlp_mxfp8`, `moe`, `moe_block`,
> `moe_mxfp8`, `mxfp_subkernels`, `mxfp_utils`.
>
> **Absent: `gdn/`, `sparse_attention_indexer/`, `deepseekv32_mlp/`, `scan/`,
> `transformer/`.** `gdn` does not appear anywhere in the installed tree.
> Found by dev2; independently confirmed here.

| Directory | Contents | Relevant to | Installed? |
|---|---|---|---|
| `gdn/` | `gdn_tkg.py`, `gdn_cte.py`, `gdn_conv1d.py`, `gdn_block_tkg.py`, + `_torch` refs | Qwen3.8-27B; base for GLM-5.3-Flash KDA | **NO** |
| `mla/deepseek/` | `mla_qkv_cte`, `mla_sparse_attention_cte`, `mla_vup_oproj_cte`, `mla_common_cte`, + `_torch` refs — **prefill only** | GLM-5.3, DeepSeek-V4.1 | yes |
| `sparse_attention_indexer/` | DeepSeek sparse-attention indexer, top-k | GLM-5.3 DSA | **NO** |
| `moe_block/`, `moe_mxfp8/`, `moe/` | MoE + EP + MXFP8 | all MoE targets | yes |
| `deepseekv32_mlp/` | DeepSeek V3.2 MLP | MoE targets | **NO** |
| `scan/` | linear scan, selective scan (Mamba), SSD (Mamba-2) | linear-attention fallbacks | **NO** |
| `attention/`, `attention_mxfp8/` | flash CTE/TKG, SWA fused | everything | yes |
| `transformer/` | megakernels | everything | **NO** |

The `mla/deepseek/` listing **confirms §3.4 against the shipped package**: every
file is `*_cte`, plus `mla_validate_params.py`. There is no decode kernel in the
installed nkilib either, not just in the checkout.

Every kernel does have a `*_torch.py` reference next to it, which is what makes
CPU-mode validation possible — for the kernels that ship.

**Revised view, twice over.** Kernel *authoring* may still be 10–20% of a port,
but the kernels this roadmap leans on hardest — `gdn` for Qwen3.8-27B and as the
base for GLM-5.3-Flash's KDA, and `sparse_attention_indexer` for GLM-5.3's DSA —
**are not in the shipped library**. They exist in the GitHub source, so the work
is vendoring rather than writing, with
`functional/vendored_kernels/rotational_topk/` as the in-tree precedent. But that
adds a provenance and version-skew problem that "they already exist" concealed:
the vendored copy and the bundled `nkilib` will drift, and nothing checks it.

Before relying on any `nkilib/experimental/` module, **check it is installed**.
For these the question is absence, not version skew.

And check what it is tested against. dev2 found that `gdn_tkg`'s entire upstream
test table is a **single case** — `test_gdn_tkg.py:88`,
`test_cases_basic = [(24, 128, 128, bfloat16)]`, which at `NUM_V_HEADS=12` is
batch 2. `gdn_cte` has both a basic and a large table. So the **decode** kernel
this roadmap leans on has one test point, and GLM-5.3-Flash needs 64 heads
against that point's 24.

### How kernels are wired

Models never call NKI directly — they call `vllm_neuron.functional` (`NF`). The
pattern, four parts:

1. `@nki.jit` torch-compatible shim taking only tensors and primitives, rebuilding
   `nkilib` dataclasses *inside* the jit boundary so FX tracing never sees them.
2. A pure-PyTorch fallback (this is what makes CPU mode work).
3. A `_can_use_kernel(...)` guard — see `utils/neuron_utils.py:16`.
4. `wrap_nki(...)` from `libtorch_neuronx_lite.nki.nki_hop`, indexed by LNC grid:
   `wrapped[2](...)`.

`vllm_neuron/nki/` is empty (SPDX header only). Don't look there.
`functional/vendored_kernels/rotational_topk/` is the precedent for pulling a
newer `nkilib` kernel in-tree with a provenance note.

---

## 5. In-flight upstream PRs that matter

Worth tracking and, where possible, collaborating on rather than re-deriving.

| PR | What | Why it matters |
|---|---|---|
| [#54](https://github.com/vllm-project/vllm-neuron/pull/54) | Qwen3.5 dense (2B, 27B), hybrid gated-DeltaNet + attention | **The unblock for §3.3.** Adds `RecurrentLayerSpec` → vLLM `MambaSpec`, two KV cache groups, `_align_hybrid_page_sizes`. ~330 lines of framework plumbing outside the model dir. Benchmarked on a **trn2.3xlarge at TP=4**. |
| [#55](https://github.com/vllm-project/vllm-neuron/pull/55) | Sparse Qwen3.5 (35B-A3B, 397B-A17B) | Expert-parallel path stacked on #54. Author notes 397B is untested and needs experts sharded across DP replicas. |
| [#40](https://github.com/vllm-project/vllm-neuron/pull/40) | MiMo-V2.5 text decoder (BF16, TP64/EP64) | Adds `NF.write_paged_kv_cache`, a shared scatter kernel for head_dim > 128 models. Documents the aliasing pathology in §3.6. |
| [#56](https://github.com/vllm-project/vllm-neuron/pull/56) | Qwen3-MoE with YaRN | — |
| [#47–#50](https://github.com/vllm-project/vllm-neuron/pull/47) | DFlash speculative decoding + recipes | — |

PR #54 is the single most valuable asset for our roadmap.

---

## 6. Target models

Ordered easiest → hardest to bring up. Architecture verified against live
HuggingFace configs on 2026-09-24.

### Qwen3.8-27B — start here

- `model_type` is **`qwen3_5`** — Qwen reused the Qwen3.5 architecture ID. This is
  why it needed zero new model code in upstream vLLM or transformers, and why
  PR #54's Qwen3.5 work applies directly rather than by analogy.
- **Dense, not MoE.** No expert routing, no EP, no MoE dispatch kernels.
- 64 layers as `[linear_attention ×3, full_attention] ×16` → 48 gated DeltaNet +
  16 gated GQA. Hidden 5120, 24 Q / 4 KV heads.
- **BF16 released** (no `quantization_config`) — no dequant path needed.
- **55.6 GB / 18 shards.** Fits a single Trainium2 chip with room for KV.
- transformers reference: `modeling_qwen3_5.py` ✅

Watch: `partial_rotary_factor: 0.25`, interleaved mRoPE `[11, 11, 10]`, and the
two **separate** gates — `attn_output_gate: true` is a **sigmoid** on the
full-attention output, while `output_gate_type: "swish"` is the activation of the
DeltaNet block's gated RMSNorm (swish ≡ SiLU). Conflating them is easy and wrong.

`head_dim: 256` is **not** a concern here — see §3.2.

All four of those were checked against the `transformers` reference at this
checkpoint's exact dimensions on 2026-09-25 and match bit-exactly (mRoPE cos/sin,
partial rotary on q and k, both RMSNorm variants, the attention gate). Note the
asymmetry the port gets right: the plain RMSNorm scales by `(1 + weight)`, the
gated one by plain `weight`.

Two further findings from that evaluation:

- **PR #54 needs no new model code for this checkpoint.** `Qwen3_5Config.from_hf`
  takes the published config unchanged, and all 851 text-decoder tensors are
  predicted exactly from it (0 mismatched, 0 unexplained, 0 missing).
- The only tensor-level difference from Qwen3.5-27B is that `linear_attn.A_log`
  and `linear_attn.norm.weight` moved F32 → BF16. It is already handled:
  `model.py` coerces `dt_bias`/`A_log` to float32 *before* a
  `load_state_dict(..., assign=True)`, which would otherwise replace the
  parameters without casting and run `A_log.exp()` in bf16.

Full write-up: `dev/progress/2026-09-25-qwen38-27b-pr54-findings.md`.

### MiMo-V2.6-Pro-RL — second

- `model_type: mimo_v2`, **1.02T total / 42B active**, 70 layers = 60 SWA + 10 global.
- Architecturally a *good* fit: plain GQA + SWA + attention sinks + MoE — the
  plugin's most mature subsystems. Sinks are already wired (`GptOssAttention.sinks`),
  SWA is first-class (`LayerSpec.sliding_window_size`, `NF.swa_fused_attention`).
- Demoted by operational weight: **573.5 GB**, needs a trn2.48xlarge; MXFP4-stored
  experts (`store_dtype: mxfp4`); bf16 MoE router; `head_dim 192` / `v_head_dim 128`.
- transformers: **remote code only** (`trust_remote_code`), no native support for Pro.

### GLM-5.3 — third

- `model_type: **glm_moe_dsa**` (not `glm5_next`). 78 layers, **all MLA + DSA**, no
  linear attention, **no mHC**. 256 routed + 1 shared experts.
- Upstream vLLM aliases it straight onto the DeepSeek-V3.2 implementation — which
  is exactly the architecture AWS just shipped `nkilib` kernels for (2.31 and 2.32
  kernel drops were DeepSeek MLA and V3.2 sparse-MLA). We ride that roadmap.
- MLA with partial RoPE: `q_lora_rank 2048`, `kv_lora_rank 512`,
  `qk_nope_head_dim 192` + `qk_rope_head_dim 64`.
- **755.6 GB FP8.** transformers reference: `modeling_glm_moe_dsa.py` ✅
- Gated on MLA decode (§3.4).

### GLM-5.3-Flash — fourth

- `model_type: glm5_next`, **320B total / 18B active**, 45 layers, 1M context.
- 34 KDA (Kimi Delta Attention) layers + 11 NoPE sparse-MLA layers. KDA differs
  from `nkilib`'s GDN in one way that matters: **GDN scales the state by a per-head
  *scalar* gate; KDA scales row k by its own `exp(g[k])`** — a per-channel forget gate.
- **mHC**: `hc_mult 4`, `hc_sinkhorn_iters 20` — 4 parallel residual streams with
  Sinkhorn-normalized mixing. Greenfield, but pure model code, not kernel code.
- DSA indexer: `index_topk 2048`, `index_kpool 4`. Note: **running the sparse layers
  dense is exact, not approximate, for seq_len ≤ 2048**, because every pool gets
  selected — so correctness can be established before any sparse kernel exists.
- FP8 as released (328 GB); a BF16 repo exists (`zai-org/GLM-5.3-Flash-BF16`, 643 GB).
- transformers reference: `modeling_glm5_next.py`, needs **transformers ≥ 5.16.1**
  (within this plugin's `<6.0.0` pin) ✅

### DeepSeek-V4.1-Flash — deprioritize

- **No HuggingFace transformers reference exists.** Both PRs
  ([#48721](https://github.com/huggingface/transformers/pull/48721),
  [#48768](https://github.com/huggingface/transformers/pull/48768)) were closed
  unmerged, and there is no remote-code module. The only oracle is DeepSeek's
  `inference/` folder, which needs a TP-sharded weight conversion and is explicitly
  not a serving engine.
- This plugin's entire validation methodology depends on diffing against a CPU
  reference. Without one, there is no way to localize a numerical bug to a layer.
- Architecturally also the hardest: causal encoder-decoder (decoder KV projected
  from encoder layers 2/8/14/20), CSA2 with three per-layer cache regimes, a
  two-level hierarchical indexer, Single-Pass mHC, **Engram** (196B params of
  n-gram memory over 384M-row tables), DSpark semi-AR drafting, FP8 at 32×32 blocks
  with `ue8m0` scales, FP4 experts, NVFP4 KV cache.

**Do not start this one until something else has shipped.**

---

## 7. Roadmap

### Phase 0 — tooling and hardware (days)

1. Install AWS's agentic porting toolchain:
   ```bash
   pip install --upgrade neuron-agentic-development \
       --extra-index-url https://pip.repos.neuron.amazonaws.com
   deploy-neuron-agentic-development-to-claude
   ```
   Provides `neuron-framework-autoport-vllm-neuron` (11-step port: research →
   generate `config.py`/`factory.py`/`model.py` → register → smoke test → logit
   validation), `neuron-framework-equivalence`, and NKI writer/debugger/profiler
   agents. Has a **`dry-run` mode** that completes research and codegen without
   hardware.
2. Set up CPU development — no device needed:
   ```bash
   VLLM_NEURON_CPU_MODE=1 pytest test/unit -v
   VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 pytest test/vllm_neuron/nki/ -v
   ```
   CPU mode catches wrong weight mappings, shape mismatches, bad collectives, RoPE
   variants and transposition errors. CPU compilation mode emits NEFFs with no device.
3. Get a **trn2.3xlarge**. PR #54 did both Qwen3.5-2B *and* 27B on one. A
   trn2.48xlarge is not needed to start, and the largest current risk is that
   nothing has touched silicon.

### Phase 1 — Qwen3.8-27B

Rebase onto PR #54's branch rather than the release branch, since it carries the
recurrent-state cache work. Engage `whn09` on
[#53](https://github.com/vllm-project/vllm-neuron/issues/53)/#54 — they solved our
hardest framework problem and are looking for review.

Deliverable: coherent greedy output, HF logit match, TTFT/TPOT on trn2.3xlarge TP=4.

### Phase 2 — pick one of

- **MiMo-V2.6-Pro** if a trn2.48xlarge is available and MXFP4 expert loading is
  tractable.
- **MLA decode kernel** — the gate for both GLM-5.3 and DeepSeek-V4.1. Higher
  leverage, higher risk. `nkilib`'s `mla/deepseek/*_cte` kernels give the prefill
  half and a `*_torch.py` reference to validate against.

### Phase 3

GLM-5.3, then GLM-5.3-Flash (KDA per-channel gate + mHC). DeepSeek-V4.1-Flash
only if a reference implementation appears.

---

## 8. Hardware notes

- **trn2.48xlarge**: 16 Trainium2 chips, 96 GiB HBM each (1,536 GiB total),
  46.4 TB/s device bandwidth, 192 vCPU, 2,048 GiB host RAM.
- **trn2.3xlarge**: 1 chip, 12 vCPU, 128 GB host RAM. The dev box.
- **LNC**: Trn2 runs `logical_nc_config=2` by default — 128 physical NeuronCores on
  a 48xlarge present as 64 logical.
- Trn3 is supported by this plugin line (`Trn2, Trn3` in the compat matrix) and
  **not** by NxDI.
- **[unverified]** On-demand pricing was not reliably established; public scrapes
  disagree. Check the AWS pricing page directly before sizing a budget.

---

## 9. Reference

- Onboarding: `docs/model-dev/onboarding-models.md` (1052 lines, procedural)
- Code patterns: `docs/design/framework/model_bringup.md` (1168 lines, written for
  AI-assisted porting; 12-step checklist, test pyramid, tolerance maps)
- Canonical reference model: `vllm_neuron/model/gpt_oss/model_bf16.py`, annotated
  throughout with `# >>> PARALLELISM <<<` (keep) vs `# <-- MODEL-SPECIFIC` (change)
- CPU workflow: `docs/model-dev/cpu-development.md`, `nki_cpu_simulator.md`
- Migration context: `docs/getting-started/migration-nxdi-to-vllm-neuron.md`
- Autoport skill: [Neuron Agentic Development](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-agentic-development/index.html)
- Kernel catalog: [NKI Library reference](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/library/api/index.html)

### Model interface contract

Duck-typed, enforced by `NeuronModelRunner`. Required:
`from_configs(hf_config, neuron_config)` (classmethod), `load_weights(...)`,
`forward(...)`, `get_kv_spec() -> KVSpec`, `bind_kv_cache(...)`.
Optional: `load_weights_lite(...)` (CPU compile), Eagle3 aux-hidden-state hooks,
`embed_multimodal(...)` (VLM).

`attn_metadata` is a dict keyed by layer name holding `block_table_tensor`,
`slot_mapping` (-1 = padding), `max_query_len`, `block_size`,
`decode_token_threshold`. **The model dispatches prefill vs decode itself** on
`max_query_len <= decode_token_threshold`, and **the model owns KV cache writes** —
the runner only allocates and binds.

### Things the plugin does not give you

No shared RMSNorm or RoPE module — every model defines its own (`LlamaRMSNorm`,
`GptOssRMSNorm`, `Qwen3RMSNorm`, …). `vllm_neuron/nn/gqa.py` is a stub, one SPDX line.
