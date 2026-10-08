#!/bin/bash
# GLM-5.3-Flash (BF16) on trn2.48xlarge, TP=64 with routed experts as EP=32 x expert-TP=2.
# Mirrors /data/serve_dsv41.sh. Model code: /data/glm53f-wt (branch glm53f-device).
#
# GLM's plugin model only does SINGLE-SHOT prefill from position 0:
# - MLA prefill reads no cache ("every key is in this call");
# - KDA prefill starts from a zero recurrent state.
# So:
# - no prefix caching;
# - max_num_batched_tokens == max_model_len, which the runner treats as "segmented prefill
#   disabled". A segmented or prefix-hit prefill would be computed silently without its
#   earlier context.
# The one prefill bucket is therefore max_model_len wide. Keep it modest, because prefill
# scratch grows with it [unverified HBM above 512].
#
#   MAX_LEN=2048 /data/serve_glm53f.sh        (default 2048; the bench prompts must fit)
#
# transformers 5.15 (DLAMI) has no glm5_next config; 5.17 lives in /data/pylib-tf517.
set -e
MAX_LEN=${MAX_LEN:-2048}
SEQS=${SEQS:-16}
BLOCKS=${BLOCKS:-$(( (MAX_LEN / 64 + 4) * SEQS + 64 ))}     # MLA ceil(len/64) + 4 KDA groups per seq
export PYTHONPATH=/data/glm53f-wt:/data/pylib-tf517
export PATH=/data/venv-fork/bin:$PATH
export VLLM_NEURON_LOG_LEVEL=INFO
export NEURON_SKIP_EFA_AFFINITY=1
export NEURON_LIBTORCH_DISABLE_GRAPH_CAPTURE_BACKEND=1
export NEURON_LIBTORCH_COMPILATION_TIMEOUT=7200
cd /data
exec /data/venv-fork/bin/python -m vllm.entrypoints.openai.api_server \
  --model /data/models/GLM-5.3-Flash-BF16 --served-model-name glm53f \
  --tensor-parallel-size 64 --enable-expert-parallel --dtype bfloat16 \
  --max-model-len "$MAX_LEN" --max-num-seqs "$SEQS" --max-num-batched-tokens "$MAX_LEN" \
  --no-enable-prefix-caching --no-async-scheduling \
  --num-gpu-blocks-override "$BLOCKS" \
  --limit-mm-per-prompt "{\"image\":0,\"video\":0}" \
  --additional-config "{\"neuron_config\": {\"ep_degree\": 32, \"num_seqs_buckets\": [$SEQS], \"num_batched_tokens_buckets\": [$MAX_LEN], \"on_device_sampling_config\": null}}" \
  --port 8000
