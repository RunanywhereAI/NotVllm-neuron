#!/bin/bash
# GLM-5.3 (full, zai-org/GLM-5.3, glm_moe_dsa) on trn2.48xlarge: TP=64, routed experts EP=64
# (4 per rank, block FP8 dequantized in-graph). Mirrors /data/serve_dsv41.sh.
# Model code: /data/glm53-wt (branch glm53-full).
#
# Prefill continues from cached context (prefix-hit / segmented prefill are both supported:
# every step reads the whole block-table context with its fresh rows substituted; CPU
# oracle covers a continuing chunk), so prefix caching is on and prompts longer than the
# 512 bucket are prefilled in 512-token segments.
#
# KV: one buffer per layer (78), one page per 32 tokens = 576-wide latent + 128-wide index
# key rows (+pad) = 46 KB/layer/page, 3.59 MB per block over all layers, replicated on every
# rank. BLOCKS=1024 -> 3.4 GiB, 32K tokens in flight.
# Compile probe (personal_reference/glm_moe_dsa/tier3/compile_probe.py, full depth,
# T=512 prefill, n=16 decode, max_len 8192), compiler's total estimated HBM per core:
#   512 blocks:  prefill 13.57 GB, decode 14.50 GB (scratch 0.3 / 1.3 GB)
#   1024 blocks: prefill 15.37 GB, decode 16.26 GB (scratch 0.35 / 1.28 GB)
# Decode scratch grows with SEQS x MAX_LEN (every request reads the whole addressable
# context); 16 x 8192 is what was compiled.
#
#   MAX_LEN=8192 SEQS=16 BLOCKS=1024 /data/serve_glm53.sh
#
# vLLM refuses an fp8 config.json on this platform before plugin code runs, so the model
# is served from /data/glm53-served: config.json without quantization_config (kept as
# original_quantization_config for the loader), every other file symlinked.
set -e
MAX_LEN=${MAX_LEN:-8192}
SEQS=${SEQS:-16}
BLOCKS=${BLOCKS:-1024}
BUCKET=${BUCKET:-512}
export PYTHONPATH=/data/glm53-wt
export PATH=/data/venv-fork/bin:$PATH
export VLLM_NEURON_LOG_LEVEL=INFO
export NEURON_SKIP_EFA_AFFINITY=1
export NEURON_LIBTORCH_DISABLE_GRAPH_CAPTURE_BACKEND=1
export NEURON_LIBTORCH_COMPILATION_TIMEOUT=7200
cd /data
/data/venv-fork/bin/python -c "
from vllm_neuron.model.glm_moe_dsa.config import make_served_dir
make_served_dir('/data/models/GLM-5.3', '/data/glm53-served')" 2>&1 | grep -v INFO || true
exec /data/venv-fork/bin/python -m vllm.entrypoints.openai.api_server \
  --model /data/glm53-served --served-model-name glm53 \
  --tensor-parallel-size 64 --enable-expert-parallel --dtype bfloat16 \
  --max-model-len "$MAX_LEN" --max-num-seqs "$SEQS" --max-num-batched-tokens "$BUCKET" \
  --block-size 32 --enable-prefix-caching --no-async-scheduling \
  --num-gpu-blocks-override "$BLOCKS" \
  --additional-config "{\"neuron_config\": {\"ep_degree\": 64, \"num_seqs_buckets\": [$SEQS], \"num_batched_tokens_buckets\": [$BUCKET], \"on_device_sampling_config\": null}}" \
  --port 8000
