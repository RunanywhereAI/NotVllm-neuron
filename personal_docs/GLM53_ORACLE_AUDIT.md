# GLM-5.3-Flash oracle — audit by constant

dev1, 2026-09-25. Branch `oracle-provenance-audit` off `dev`.

Audit only — **no fixes made here.** The oracle lives on `glm53-indexer`, which
dev2 owns; this branch carries the finding, not the change.

## Read this before the tables

**Every row below is a source diff** against `transformers`
`modeling_glm5_next.py` and the live config — **not a numerical comparison**. The
`agrees?` column says what the code says. The `test?` column says whether anything
in the suite would notice if it changed. Only rows where `test?` is yes are backed
by something that runs.

Without that distinction a reader takes 42 rows of "matches" as verification,
which is precisely the trust-in-provenance error that produced three bugs on
2026-09-25.

## Why this is a table of constants, not of components

The KDA output-gate bug is the worked example, and it disqualifies component-level
auditing:

> `RMSNormGated` had an external oracle available the whole time. transformers
> implements it (`Glm5NextTextRMSNormGated`, with `self.activation = "sigmoid"`
> spelled out on its own line) and so does vLLM. The component was never
> unreferenced. It was wrong because **no test was ever pointed at that one
> value**.
>
> Worse, the suite *looked* covered: `test_gate_is_per_channel_not_scalar` has a
> 159,436x discriminating margin and passes. But it discriminates the **decay**
> gate — `ForgetGate`'s per-channel `exp(g)` — while the thing that was wrong was
> the **output** gate, `o_norm`'s activation. Different gate, similar name.
>
> **A component-level audit would have scored `RMSNormGated` green.**

So the question asked below is per constant and per activation choice: *is there a
test whose two sides could disagree about this specific value?* Credit to dev2 for
the reframing.

## Method

Every row diffed against `transformers` 5.17.0
`models/glm5_next/modeling_glm5_next.py` (2,426 lines), which ships a one-to-one
counterpart for every component of the oracle, and against the live
`zai-org/GLM-5.3-Flash` `config.json`. "Test could disagree?" is judged against
the suite as it stands on `glm53-indexer`.

One structural note: **no test imports transformers.** The oracle is deliberately
transformers-free, so external references are *vendored* (`tests/gdn_refs.py` from
nkilib, `tests/indexer_refs.py` from transformers + vLLM). That is a sound design,
but it means external coverage exists only where someone explicitly vendored a
reference — which happened for exactly two areas, KDA and the indexer.

---

## The table

`agrees?` ✓ = matches transformers. ✗ = does not.
`test?` = would the suite **as it stands** fail if this value were wrong.

### Norms and activations

| # | Constant / choice | Oracle | Authority | agrees? | test? |
|---|---|---|---|---|---|
| 1 | `rms_norm_eps` | `cfg.rms_norm_eps` = 1e-5 | config; tf same | ✓ | **no** |
| 2 | RMSNorm scale form | plain `weight` | tf plain `weight` | ✓ | **no** — and this is a Qwen trap: Qwen3.5 uses `(1 + weight)` |
| 3 | RMSNorm cast order | `(w · x_fp32).to(dt)` | tf `w * x.to(input_dtype)` | **✗ deliberate** | **yes** — dev2 pinned it with an fp64 arbiter: oracle 7.75e-03 vs tf 8.89e-03 in bf16, so ours is measurably the better reference |
| 4 | `UnweightedRMSNorm` eps | `cfg.rms_norm_eps` | tf passes `config.rms_norm_eps` too | ✓ | **no** |
| 5 | `RMSNormGated` activation | `sigmoid` | tf `self.activation = "sigmoid"` | ✓ **now** | **yes** (dev2, `b50f62d`) — was `silu` |
| 6 | `o_norm` eps | `cfg.rms_norm_eps` | tf `layer_norm_epsilon` | ✓ | **no** |
| 7 | `l2norm` form | `x * rsqrt(Σ + eps)` | tf `x / sqrt(Σ + eps)` — *"intentionally use sqrt and / to match original triton"* | **✗** | **no** |
| 8 | `l2norm` eps | 1e-6 | tf 1e-6 | ✓ | **no** |
| 9 | conv activation | hardcoded `F.silu` | tf `config.hidden_act` (= `"silu"`) | ✓ value, hardcoded | **no** |

### KDA

| # | Constant / choice | Oracle | Authority | agrees? | test? |
|---|---|---|---|---|---|
| 10 | `beta` | `sigmoid(b_proj(x))` | tf same | ✓ | partial (`gdn_refs`) |
| 11 | `gate_lower_bound` | `cfg` = −5.0 | config; tf same | ✓ | partial |
| 12 | ForgetGate activation | `lower · sigmoid(decay · g)` | tf same | ✓ | partial |
| 13 | softplus branch threshold | 20.0 | tf 20.0 | ✓ | **no** |
| 14 | gate is **per channel** `[B,S,H,K]` | yes | tf same | ✓ | **yes** |
| 15 | recurrence / chunked scan | — | nkilib GDN torch ref | ✓ | **yes** (dev2) |

### mHC

| # | Constant / choice | Oracle | Authority | agrees? | test? |
|---|---|---|---|---|---|
| 16 | `hc_mult` 4 / `hc_sinkhorn_iters` 20 / `hc_eps` 1e-6 | from cfg | config; vLLM | ✓ | **yes** |
| 17 | post multiplier | hardcoded `2 *` | vLLM `mhc_post_mult_value = 2.0` | ✓ | **yes** |
| 18 | mix width | `(2 + H) · H` | vLLM `mix_hc = (2 + n) · n` | ✓ | **yes** |
| 19 | HyperHead collapse | `streams.mean(2)` | tf `Glm5NextTextHyperHead` — unweighted mean | ✓ | **no** |

### Sparse-MLA

| # | Constant / choice | Oracle | Authority | agrees? | test? |
|---|---|---|---|---|---|
| 20 | attention scale | `qk_head_dim ** -0.5` | tf identical (`qk_rope = 0`, so 256) | ✓ | **no** |
| 21 | softmax dtype | ambient (bf16) | tf forces `dtype=torch.float32` | **✗** | **no** |
| 22 | latent slice / NoPE | `[..., :kv_lora_rank]`, no RoPE | config `qk_rope_head_dim = 0` | ✓ | **no** |

### MoE and MLP

| # | Constant / choice | Oracle | Authority | agrees? | test? |
|---|---|---|---|---|---|
| 23 | `swiglu_limit` in routed experts | 10.0, clamped | tf `_apply_gate` identical | ✓ | **no** |
| 24 | `swiglu_limit` in `MLP` | **absent** | tf `Glm5NextTextMLP` clamps | **✗ BUG** | **no — and a naive external test would also pass** |
| 25 | `routed_scaling_factor` | `cfg` = 2.5 | config; tf same | ✓ | **no** |
| 26 | top-k norm denominator | `+ 1e-20` | tf `+ 1e-20` | ✓ | **no** |
| 27 | scoring function | `sigmoid` | config `scoring_func: sigmoid` | ✓ | **no** |
| 28 | group masking (`n_group`) | omitted | tf masks by group | ✓ *only because* `n_group = 1` | **no** — unguarded |
| 29 | `first_k_dense_replace` | `cfg` = 3 | config | ✓ | **no** |

### Wiring — `LinearAttention`, `DecoderLayer`, `FlashTextModel`

Audited 2026-09-25 (second pass; these were recorded as unknown in the first).
**All match.** Recorded because a checked-and-clean row is not the same as an
unchecked one, and that distinction is the point of this table.

| # | Constant / choice | Oracle | Authority | agrees? | test? |
|---|---|---|---|---|---|
| 30 | qkv concat order | `cat([q_proj, k_proj, v_proj], -1)` | tf identical order | ✓ | **no** |
| 31 | conv causal trim | `[..., -S:]` after the conv | tf `mixed_qkv[:, :, -seq_len:]` | ✓ | partial |
| 32 | `forget_gate` input | the **layer input**, pre-conv | tf `self.forget_gate(hidden_states)` | ✓ | **no** |
| 33 | output-gate path input | the **layer input**, pre-conv | tf `g_b_proj(g_a_proj(hidden_states))` | ✓ | **no** |
| 34 | `l2norm` placement | inside the scan, on q and k | tf `use_qk_l2norm_in_kernel=True` | ✓ | partial |
| 35 | final projection order | `o_proj(o_norm(core, gate))` | tf identical | ✓ | **no** |
| 36 | layer order | `attn_hc` → `input_layernorm` → attn → expand → `ffn_hc` → `post_attention_layernorm` → mlp → expand | tf identical | ✓ | **no** |
| 37 | `hc_expand` in the layer | `post·out + combᵀ·residual` | tf inlines the same expression | ✓ | **yes** (via mHC tests) |
| 38 | stream initialisation | `h.unsqueeze(2).expand(-1,-1,hc_mult,-1).contiguous()` | tf byte-identical | ✓ | **no** |
| 39 | final collapse order | `norm(streams.mean(2))` — **mean first** | tf `self.norm(self.hc_head(hidden_states))` | ✓ | **no** |

Rows 32, 33, 38 and 39 are the ones worth noting: each is an ordering or
input-selection choice that would produce plausible, wrong output if reversed, and
none of them has a test.

### Dispatch and residual placement (second pass, re-verified 2026-09-25)

Re-checked against `glm53-indexer` **after** dev2's clamp fix, since the file had
moved (644 → 693 lines). `DecoderLayer.forward`, `LinearAttention` and
`FlashTextModel` are byte-unchanged; only the `swiglu_limit` threading differs, so
rows 30–39 stand. Three further rows, prompted by master:

| # | Constant / choice | Oracle | Authority | agrees? | test? |
|---|---|---|---|---|---|
| 40 | attention dispatch | `cfg.layer_types[i]` → `LinearAttention` / `SparseMLAttention` | tf `config.layer_types[layer_idx]` (:1265) | ✓ | **yes** — closed by `tests/test_layer_dispatch.py` (`6c1f661`) |
| 41 | MLP dispatch | `cfg.mlp_layer_types[i]` → `MoE` / `MLP` | tf `config.mlp_layer_types[layer_idx]` (:1274) | ✓ | **yes** — upgraded from partial by the same file |
| 42 | shared-expert placement | `routed_sum + shared_experts(x)`, shared **not** routing-weighted, applied to the layer input | tf `experts(...) + shared_experts(residuals)`, `residuals` captured pre-flatten | ✓ | **no — `MoE.forward` has no external comparison** |

**Row 40** is the larger gap. Across the whole suite there is no assertion that any
layer picks the right *attention* kind. Inverting the dispatch, or shifting
`layer_types` by one, would produce a model that runs, trains nothing, and fails
nothing.

**Row 41 is covered only partially, and the gap is demonstrable.**
`test_both_mlp_call_sites_carry_the_limit` pins indices 0 (dense) and 3 (MoE).
With `mlp_layer_types = [dense, dense, dense, sparse, sparse, …]`, shifting the
list **left** by one gives index 0 → dense and index 3 → sparse: **both
assertions still pass**. Two sampled indices cannot pin a pattern; an
index-for-index comparison against the config can.

**Row 42:** the MoE tests added with the clamp fix check the *limits* carried by
`MLP` and by the routed path; they do not compare `MoE.forward`'s output to
anything external. So the add position, and the fact that the shared expert is
**not** multiplied by the routing weight, are unverified by test — though both
match transformers on inspection.

**Basis of this audit:** every row is a *source diff* against
`modeling_glm5_next.py` and the live config, not a numerical comparison. The
`agrees?` column says what the code says; the `test?` column says whether anything
in the suite would notice if it changed. No row here is backed by a running test
except where `test?` says yes.

---

## The four findings, in order of severity

### 1. `MLP` omits the swiglu clamp — a real bug on 45 of 45 layers

```python
# oracle
return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
# transformers — the comment is theirs
gate = gate.clamp(min=None, max=self.swiglu_limit)          # 10.0
up   = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
```

`MLP` is used at layers 0–2 (`first_k_dense_replace = 3`) **and as
`MoE.shared_experts` on all 42 MoE layers**, so the shared expert runs unclamped
everywhere. The routed experts *are* clamped and match exactly — the same
one-place-not-the-other shape as the gate bug.

**And an external test would also have missed it.** The clamp is a no-op until
activations exceed ±10:

| gate/up std | fraction clamped | mean rel. diff |
|---|---|---|
| 1.0 | 0.000 | 0.00e+00 |
| 3.0 | 0.002 | 8.8e-05 |
| 5.0 | 0.089 | 9.7e-03 |
| 8.0 | 0.378 | 7.5e-02 |

`tiny_cfg` initialises at `normal_(0, 0.02)`, keeping activations far inside the
limit, so comparing `MLP` against `Glm5NextTextMLP` on those weights **passes with
the bug present**. This row needs an external oracle *and* inputs driven past the
limit. An external reference is necessary but not sufficient.

### 2. `l2norm` uses the Qwen form (row 7)

transformers computes `x / sqrt(Σ + eps)` and comments *"main difference to qwen's
gdn variation: intentionally use sqrt and / to match original triton"*. The oracle
uses `x * rsqrt(Σ + eps)` — precisely the Qwen variant transformers is warning
about. ULP-level, but transformers considered it worth pinning, and this is the
**third** Qwen→GLM carry-over after the output gate and the 2051 ceiling.

### 3. Softmax and cast-order divergences (rows 3, 21)

Sparse-MLA does not force fp32 softmax where transformers does; `RMSNorm` casts in
the other order. Both are precision divergences invisible to self-consistency.

### 4. `n_group` is an unguarded assumption (row 28)

Correct today and the docstring says why, but nothing asserts `n_group == 1`. One
assertion converts correct-by-luck into correct-by-construction.

---

## What the pattern says

Sorted by provenance, the diagnostic holds exactly:

- **Written against an external reference** (indexer, mHC): rows 16–18 and the
  indexer — cross-checked, none wrong.
- **Inherited verbatim from the NxDI port**: 1 confirmed bug (row 5), 1 found here
  (row 24), 3 divergences (rows 3, 7, 21), 1 unguarded assumption (row 28), and
  **30 of 42 rows with no test that could disagree**, including the
  attention-kind dispatch, which nothing asserts at all.

`reference.py` says *"The math is unchanged from that CPU-verified version"*.
"CPU-verified" in the NxDI fork meant its own tests passed — not that it had been
diffed against transformers. **Provenance is not verification.**

## Suggested order of work

1. Row 24, the `MLP` clamp — a real bug with a known blast radius.
2. Rows 7, 3, 21 — decide whether to match transformers exactly or record the
   divergence deliberately. Either is defensible; silently differing is not.
3. Vendor a transformers reference for the norms, MLP, router and MLA the way
   `indexer_refs.py` already does, and **drive inputs hard enough to exercise
   clamps and saturation** or the tests inherit the blindness they remove.
4. Row 28's assertion.
5. ~~Audit `LinearAttention` wiring, `DecoderLayer`, `FlashTextModel`.~~ Done —
   rows 30-39, all match. But 8 of those 10 rows still have no test.
