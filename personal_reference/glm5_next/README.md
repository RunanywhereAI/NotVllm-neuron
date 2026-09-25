# GLM-5.3-Flash oracle and KDA NKI kernels — how to run any of this

Written because none of it was reproducible from the repo. Three of us solved the
"which interpreter has both torch and pytest" problem independently and differently,
and the kernel simulation harnesses existed only in one user's home directory on one
machine. Both are fixed here.

## What is in here

| path | what | runs where |
|---|---|---|
| `reference.py` | the CPU oracle: KDA, mHC, the DSA indexer, sparse-MLA, MoE | anywhere with torch |
| `weight_converter.py` | FP8 checkpoint → the oracle's `state_dict` | anywhere with torch |
| `nki_kda_tkg.py` | KDA **decode** kernel | needs NKI |
| `nki_kda_cte.py` | KDA **chunked-prefill** kernel | needs NKI |
| `tests/` | ~260 tests, torch only | anywhere with torch |
| `sim/simulate_kernels.py` | acceptance harness for both kernels | needs NKI |

## Running the test suite (no special hardware)

The whole suite is torch-only by construction — **no test imports transformers, vllm,
nkilib or nki**. External references are *vendored* (`tests/hf_refs.py`,
`tests/gdn_refs.py`, `tests/indexer_refs.py`) precisely so the suite runs anywhere.

The only real obstacle is finding one interpreter with both torch and pytest:

```bash
uv venv --system-site-packages --python /opt/homebrew/bin/python3.14 /tmp/glm53venv
uv pip install --python /tmp/glm53venv/bin/python pytest
/tmp/glm53venv/bin/python -m pytest personal_reference/glm5_next/tests/ -q
```

`--system-site-packages` **against the Homebrew interpreter** is the load-bearing part:
torch lives in Homebrew's site-packages, and a clean venv will not see it. Use `-s` to
see the measured tables the tests print (agreement floors, clamp no-op table, selection
overlap, sub-block headroom) — several are as informative as the assertions.

### Optional environment variables

| variable | effect |
|---|---|
| `GLM53F_INDEX` | path to the real `model.safetensors.index.json`; runs the converter plan against all 76,108 real tensor names instead of the synthetic index |
| `GLM53F_CONFIG` | path to the real `config.json`; checks the indexer constants against it rather than the recorded values |
| `NKILIB_SRC` | `<nki-library>/src/nkilib_src`; diffs the vendored GDN reference against the live checkout |

Without them those three tests skip; everything else runs.

## Running the kernel simulation (needs the NKI toolchain)

`sim/simulate_kernels.py` runs both kernels under `nki.simulate` and checks them
against the oracle. It exits non-zero on failure, so it is a gate rather than a
printout. It is deliberately **not** a pytest test: pytest lives where the toolchain
does not, and vice versa.

```bash
# on the x86 box
python3 personal_reference/glm5_next/sim/simulate_kernels.py        # or: tkg | cte
```

Three things that will otherwise cost time:

- **Use the venv interpreter that has NKI.** `python3` on PATH may be PyPy, which
  fails in confusing ways rather than cleanly.
- **`nkilib`'s `experimental/` is incomplete in the installed package** — `gdn/`,
  `sparse_attention_indexer/`, `deepseekv32_mlp/`, `scan/` and `transformer/` are all
  absent. That does **not** matter for these kernels: they import only
  `nkilib.core.utils` (`kernel_assert`, `stream_shuffle_broadcast`), which *is*
  installed. It matters if you want to run nkilib's own GDN kernels for comparison,
  which needs the source checkout on `PYTHONPATH`.
- **The API is `nki.simulate`, not `nki.simulate_kernel`.**

The harness only needs this package's files, not the whole repo, so copying
`glm5_next/` to the box is enough.

## What is validated, and what is not

- **The oracle** is cross-referenced against transformers 5.17, vLLM and nkilib's GDN
  reference, all vendored. Four bugs were found this way; see
  `dev/progress/2026-09-25-glm53-oracle-audit.md`.
- **Both kernels** agree with the oracle to ~0.4–0.8% relative under simulation, which
  is the bf16 arithmetic floor, and both reject a per-head scalar gate.
- **Treat the scalar-gate ratio as a distribution, not a measurement.** The harness
  reports it from mean-abs, which spans ~1.2–1.6× across input draws. Earlier notes
  quote 54.4× and 88.6× for the decode kernel; those are two *max-abs* samples of the
  same quantity, and a max-abs ratio — one extreme-value statistic over another —
  spans 2.2–4.4× and degrades with head count, down to 19.8× at BH=32. The numbers do
  not contradict each other; the statistic was the wrong one.
- **Neither kernel has run on a device.** Simulation does not exercise the real DMA
  engine, PSUM bank allocation or SBUF capacity, so the batch-vs-tensor-parallelism
  envelope in `nki_kda_tkg.py` is arithmetic, not measurement.
- The oracle is **fp32 by design** and deliberately diverges from transformers where
  transformers rounds early. An oracle is maximum-precision ground truth, not a replica
  of one deployment's rounding — `tests/test_norms_router.py` pins that and says why,
  so nobody "fixes" it into being less accurate.
