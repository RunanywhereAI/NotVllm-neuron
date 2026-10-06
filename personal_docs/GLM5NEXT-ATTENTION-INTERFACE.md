# GLM-5.3-Flash attention layers — file layout and interfaces

Written for **dev3** (composing these into `model.py`) and **dev1** (config/factory),
by dev2, 2026-09-25. Branch `glm5next-attention`. **Implemented**: `kda.py` and `mla.py` exist and are validated against the
oracle (28 tests). The framework seams — state indices, the latent page layout — are
still open and marked in the code.

> **Status on branch `glm5next`, 2026-10-06.** Both seams are closed. `Glm5NextKDA.forward`
> binds to its state page through `kv_cache.state_page_indices`; `Glm5NextSparseMLA.forward`
> reads and writes the folded page only through `cache_layout.LatentPageLayout` and writes
> through `functional/vendored_kernels/latent_cache_write` (a one-buffer row scatter from PR
> #40). `model.py` composes them with mHC and the MoE, and the whole model matches the oracle
> on prefill and on prefill-then-decode at batch > 1 over hostile pages
> (`personal_reference/glm5_next/tests/test_plugin_model.py`). Two changes to what is
> described below: the indexer's pool keys are stored at the **page dtype**, not vLLM's FP8
> format (nothing in the plugin or the oracle scores FP8); and `KDAParams`/`MLAParams` no
> longer default any field. The NKI KDA kernels are still not dispatched from the layers.

Read §5 first if you only read one part: the capture design is the thing that is hard
to retrofit and easy to omit.

---

## 1. Files

| file | contents |
|---|---|
| `vllm_neuron/model/glm5_next/kda.py` | `Glm5NextKDA` — the 34 linear-attention layers |
| `vllm_neuron/model/glm5_next/mla.py` | `Glm5NextSparseMLA` + `Glm5NextIndexer` — the 11 sparse-MLA layers |
| `vllm_neuron/model/glm5_next/nki_kda_tkg.py` | KDA decode kernel (exists, simulation-validated) |
| `vllm_neuron/model/glm5_next/nki_kda_cte.py` | KDA prefill kernel (exists, simulation-validated) |

The two `nki_*` files follow PR #54's convention exactly — `qwen3_5/` keeps
`nki_deltanet.py` and `nki_deltanet_fused.py` beside `deltanet.py`, so kernels live in
the model directory rather than a shared kernel home.

### Kernel home — settled, and the dispatch is NOT wired

The kernels now live in `vllm_neuron/model/glm5_next/`, following PR #54's
convention, and `sim/simulate_kernels.py` loads them **by file path** so neither it
nor they require vLLM.

> **The plugin layers do not call the kernels.** `kda.py` runs its torch path only;
> the NKI dispatch is deliberately unwired until dev3's seams land. So "the layers
> are validated" and "the kernels are validated" are two true statements about two
> things that **have never run together through the plugin**. Wiring the dispatch
> creates a new composition, and per §8 that needs its own test before it is trusted.

The indexer is torch-only everywhere — no NKI indexer kernel exists. That is a
performance item, not a correctness one, but it is on the decode path for 11 of 45
layers.

### Historical: where they came from

They exist on branch `glm53-indexer` at `personal_reference/glm5_next/nki_kda_*.py`,
which this branch does not have. That directory is also the **oracle's** home, and the
two have different jobs: the oracle is torch-only reference material, the kernels are
production code. My recommendation is to **move** the kernels here rather than copy
them, and have `personal_reference/glm5_next/sim/simulate_kernels.py` import them from
their plugin path. One copy, each in the right place. The alternative — the plugin
importing from `personal_reference/` — puts production code behind a `personal_` path.

**Blocked on:** how they arrive (cherry-pick the two commits, or wait for
`glm53-indexer` to merge). Master's call; I have not moved anything.

---

## 2. `Glm5NextKDA` — the linear-attention layers

Mirrors `qwen3_5/deltanet.py`'s contract, because that is the shape the framework
expects and there is no reason to diverge.

```python
class Glm5NextKDA(nn.Module):
    def __init__(self, config, layer_idx: int): ...
    def forward(self, hidden_states, positions, attn_metadata: dict) -> torch.Tensor
```

* **Returns a plain tensor**, not a tuple. State is *not* passed in or out.
* **State lives in vLLM's KV cache**, bound by `bind_kv_cache`, read and written
  through `state_indices(metadata, num_reqs)` / `_read_states` / `_write_states`.
  Two tensors per layer, as in `deltanet.py`:
  * `conv_state` `[kernel - 1, conv_dim]` — `conv_dim = 3 * 64 * 128 = 24576`,
    `kernel = 4`. (Four independent sources agree on this geometry; see dev3's
    framework doc.)
  * `recurrent_state` `[num_heads, head_k_dim, head_v_dim]` = `[64, 128, 128]` fp32
    per request, sharded to `64 // TP` heads per rank.
* **Dispatch** on `metadata["max_query_len"] <= metadata["decode_token_threshold"]`,
  as `deltanet.py` does — decode path to `nki_kda_tkg`, prefill to `nki_kda_cte`, each
  with a torch fallback validated against the oracle.

### Four things that must not be re-derived

These cost real errors during the kernel work and are easy to get wrong again:

1. **The output gate is `sigmoid`, not `silu`/`swish`.** No config key selects it;
   both transformers and vLLM hardcode sigmoid, and FLA's *default* is the wrong
   branch. Getting this wrong is a 194% error across all 34 layers.
2. **`exp(cg)` does not commute out of the matmul.** Per-channel it must be
   `(q * exp_cg) @ state`, never `(q @ state) * exp_cg`. The latter is a
   per-token-scalar optimisation and gives 124% error here.
3. **Two different masks.** The intra-chunk output keeps `j <= i` **inclusive**; the
   `A` operator keeps `j < i` **strict**. Using one for both leaves the state correct
   and the output 99.8% wrong.
4. **The sub-block size is derived, not a constant.** `16` follows from
   `gate_lower_bound = -5.0`; at `-6.0` it halves to `8`. It is a function of config,
   and `nki_kda_cte.max_safe_subblock()` computes it.

---

## 3. `Glm5NextSparseMLA` — the sparse-MLA layers

```python
class Glm5NextSparseMLA(nn.Module):
    def __init__(self, config, layer_idx: int, tp_size: int = 1): ...
    def forward(self, hidden_states, positions, attn_metadata: dict) -> torch.Tensor
    # the tested path, until dev3's cache seams land:
    def forward_core(self, hidden_states, kv_cache=None, topk_indices=None,
                     indexer_state=None) -> tuple[Tensor, tuple[Tensor, IndexerState]]
```

**The indexer carries state, and getting that wrong is invisible in prefill.** It
returns `(indices, IndexerState)` where the state holds the compressed pools, the raw-K
tail ring and the token count. Before it did, decode re-pooled from the single new
token and used *local* positions, so **a decode token attended to 1 of 21 positions —
itself alone** — while every prefill test passed. On device those three regions live in
the latent page (`cache_layout.py`); `forward_core` takes them explicitly so the decode
path is testable without the cache.

Same outer contract. Per dev3's `MLA-DECODE-GAP.md`:

* **Absorbed latent.** `q` absorbs `W_uk`, so attention runs directly against the
  512-wide latent and never materialises 64×512 K and V per token.
* **One latent tensor serves as both `k_prior` and `v_prior`** — dev3 validated this
  as bit-identical, so the cache is a single tensor, not two.
* **`d_head = 512` passes both of the decode kernel's asserts** (`_MAX_D_HEAD = 512`,
  and 512 is a multiple of 128). DeepSeek's 576 fails both. This is why NoPE
  (`qk_rope_head_dim = 0`) is what makes the layer implementable at all.
* `functional/attention/attention_decode.py` is the wrapper template.

### `Glm5NextIndexer`, a submodule of the above

Its own `nn.Module` so a capture hook can reach it (§5). Emits **token** indices;
kpool compression is scoring-only and never reaches attention.

* `index_topk 2048`, `index_kpool 4`, `index_n_heads 32`, `index_head_dim 128`.
  **`index_n_heads` is 32, not 64** — vLLM's own source carries a `# 64` comment three
  lines from the code that reads the config, and 64 is DeepSeek's value.
* **No RoPE.** `qk_rope_head_dim = 0`, so `indexer_rope_interleave` is inert here.
* Selecting every pool is exact to `seq_len <= 2051` (`index_topk + index_kpool - 1`),
  not 2048; vLLM's own gate at 2048 is conservative. At 128K this path is always live,
  so the shortcut is not reachable in the bring-up configuration.

---

## 4. What I assume from dev1 and dev3

Flagging so a mismatch surfaces now rather than at composition:

* **dev1** provides a config object exposing the `linear_attn_config` fields
  (`num_heads` 64, `head_dim` 128, `short_conv_kernel_size` 4,
  `gate_lower_bound` -5.0), the MLA fields (`q_lora_rank` 1536, `kv_lora_rank` 512,
  `qk_nope_head_dim` 256, `qk_rope_head_dim` 0, `v_head_dim` 256), the `index_*`
  fields, and `rms_norm_eps` 1e-5. I read these; I do not define them.
* **dev3** provides the cache specs and `attn_metadata` keys. I assume
  `max_query_len`, `decode_token_threshold`, `block_table_tensor` and the state-index
  helpers behave as in `deltanet.py`. **If the MLA latent cache spec names differ, tell
  me and I will follow yours.**
* Working assumptions from master: vLLM 0.24.0, TP=64, `max_model_len` 131072, **BF16
  not FP8** — so the converter's FP8 path is out of scope for this bring-up.

---

## 5. Capture design — the part that is hard to retrofit

**Why this is a first-class requirement, not instrumentation.** Three independent
measurements say the natural output boundary cannot see a real class of fault:

| component | what is invisible at the output |
|---|---|
| KDA decode | errors below **~2%** of the output — the bf16 floor is ~0.6% and a 5% input perturbation moves it only 3-6x that |
| DSA indexer | FP8 scoring changes **88-100%** of rows' selections, so exact index equality is unusable as a check |
| mHC | cross-stream mixing lives at **~1e-3**, below any sane end-to-end tolerance |

So a layer that exposes only its final output gives on-device validation **no
instrument**. Both capture mechanisms in `accuracy/tensor_capture.py` are used:

* **hooks** capture a module's output (`output[0]` if it returns a tuple), so anything
  that is naturally a submodule is capturable for free;
* **`capture_tensor(name, tensor)`** is an inline no-op unless capture is active, for
  intermediates that are not module outputs.

### Capture points I will emit

`Glm5NextKDA`, via `capture_tensor`:

| name | why it must be separately visible |
|---|---|
| `layers.{i}.kda.g` | the per-channel gate — substituting a per-head scalar is the single most likely way to get KDA wrong, and it is **invisible in the final output below ~2%** |
| `layers.{i}.kda.core_pre_norm` | the delta-rule output *before* the gated norm; the norm's sigmoid gate attenuates and hides upstream error |
| `layers.{i}.kda.recurrent_state` | fp32 and carried across steps, so a drift shows here before it shows in any single token's output |
| `layers.{i}.kda.conv_window` | the conv handoff — a stale window is silently wrong for exactly one step per pool |

`Glm5NextSparseMLA` / `Glm5NextIndexer`:

| name | why |
|---|---|
| `layers.{i}.indexer.topk_indices` | the selection itself (also reachable by hook, since the indexer is a module) |
| `layers.{i}.indexer.scores` | **the important one.** Exact index equality fails a *correct* FP8 device, so validation needs the scores to tell a near-tie swap from a real selection bug |
| `layers.{i}.mla.latent` | the single cache tensor, before it is consumed as both K and V |
| `layers.{i}.mla.attn_pre_oproj` | attention output before `o_proj` folds heads together |

`Glm5NextSparseMLA.**forward_core**` accepts an optional `topk_indices` override —
**not `forward`**, so it is unreachable from the serving path and cannot be left on by
accident (there is a test asserting it is absent from `forward`'s signature). It lets
attention can be checked **given the device's own selection**. That is the only way to
separate "selected the right tokens" from "attended to them correctly", and the same
hook on the oracle's `SparseMLAttention` isolated a 0.5% attention bug that was
otherwise buried inside selection divergence.

---

## 6. Two hazards for whoever registers these

* **vLLM silently caches model info** to `~/.cache/vllm/modelinfos/<module>-<class>.json`
  and nothing invalidates it on edit. dev1 lost four wrong diagnoses to this: an early
  registration cached `is_text_generation_model: false`, and every later source fix was
  correct and had no effect. If a registration or interface change appears not to work,
  `rm -rf ~/.cache/vllm/modelinfos` **before** debugging anything else. These two
  classes will change shape repeatedly, so this is the likeliest trap here.
* **A converted checkpoint must drop `quantization_config`**, not merely dequantize.
  vLLM's `ModelConfig` refuses an fp8-advertising checkpoint in its front end, before
  any plugin code runs — `"fp8 quantization is currently not supported in cpu"` — with
  no visible connection to the converter. Use `weight_converter.converted_config()`,
  which drops it at every level and asserts none survived.

## 7. Two rules that cost real defects here

**Any component with carried state needs a prefill-then-decode-equals-one-shot-prefill
test before it can be called validated.** Not "remember to check" — apply it
mechanically. Both the KDA conv window and the MLA indexer shipped broken decode paths
with every prefill test passing.

**Any two components that hand state to each other need a composition test, not two
independent ones.** Layer-to-layer, prefill-to-decode, kernel-to-kernel. The two NKI
kernels were each validated against the oracle and their *handoff* was untested; it
turned out sound, but a transposed `[K, V]` state would have passed every shape
assertion we have — `128 x 128` is square — and is visible only numerically, at 20.3%.

### The kernels' input contracts are asymmetric, and only one half bites

| input | `kda_cte` | `kda_tkg` | uniform caller? |
|---|---|---|---|
| `q`, `k` | already l2-normed | RAW | **harmless** — l2norm is idempotent to 5e-7 |
| `beta` | POST-sigmoid | RAW, pre-sigmoid | **WRONG** — ~5% on the output |
| `gate` | per-channel log | same | fine |

A contract asymmetry is only dangerous where the operation is **non-idempotent**. The
`q`/`k` one looks like the obvious hazard and is not; `beta` is, because double-sigmoid
*compresses the range* (0.48–0.70 into 0.62–0.67) rather than producing an obvious
error. Both are asserted in `sim/simulate_kernels.py::run_chain`.

## 8. Order of work

1. This document, and any corrections from dev1/dev3.
2. `kda.py` against the oracle's `LinearAttention`, torch fallback first, then kernels.
3. `mla.py` against the oracle's `SparseMLAttention` + `Indexer`, dense first, then
   the real indexer.
4. Capture points wired and verified to produce tensors, not merely to exist.

Every step diffs against `personal_reference/glm5_next/reference.py`, which is 261
tests deep with three defects removed. The prefill mask bug localised in one run
because there was a reference to diff against; the same will be true here, on a
larger surface.
