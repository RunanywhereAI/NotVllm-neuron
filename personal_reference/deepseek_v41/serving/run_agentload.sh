#!/bin/bash
# agentload (opencode seats) against the dsv41 vLLM server on this host.
# Does NOT start or touch the server. Everything lives under /data.
#
#   SEATS=4 MINUTES=10 /data/run_agentload.sh            # preflight, then stress
#   /data/run_agentload.sh preflight                      # capability check only
#
# Needs the server started with the tool flags (see /data/dsv41_tool_parser.py):
#   --chat-template /data/dsv41-served/chat_template.jinja --enable-auto-tool-choice
#   --tool-parser-plugin /data/dsv41_tool_parser.py --tool-call-parser deepseek_v41
#   --reasoning-parser deepseek_v4 --default-chat-template-kwargs '{"enable_thinking": false}'
# and a context agents can use. opencode's FIRST agent request is 7,302 prompt tokens
# (10 tools) and asks max_tokens 8192, so max_model_len must be >= 32768. This script
# refuses below that.
set -euo pipefail
MODE=${1:-stress}
SEATS=${SEATS:-4}
MINUTES=${MINUTES:-10}
MIN_CONTEXT=${MIN_CONTEXT:-32768}

export AGENTLOAD_HOME=/data/agentload-home
export PYTHONPATH=/data/agentload-src/agentload
export PATH=/data/opencode/bin:$PATH
export XDG_CACHE_HOME=/data/opencode/xdg/cache XDG_CONFIG_HOME=/data/opencode/xdg/config \
       XDG_STATE_HOME=/data/opencode/xdg/state
WS=/data/agentload-ws
cd "$WS"

ctx=$(curl -sf -m 10 http://127.0.0.1:8000/v1/models | python3 -c \
  'import json,sys; d=json.load(sys.stdin)["data"]; print(next((m.get("max_model_len") or 0) for m in d if m["id"]=="dsv41"))' \
  2>/dev/null || echo 0)
if [ "$ctx" -lt "$MIN_CONTEXT" ]; then
  echo "server not up, or max_model_len=$ctx < $MIN_CONTEXT: opencode cannot fit its first turn. Not running." >&2
  exit 2
fi

# seats and duration for this run
sed -i -e "s/^opencode = .*/opencode = $SEATS/" -e "s/^max_duration = [0-9]*/max_duration = $((MINUTES * 60))/" agentload.toml
echo "agentload: $SEATS opencode seats, $MINUTES min, context $ctx, workspace $WS"

python3 -m agentload -c agentload.toml preflight
[ "$MODE" = preflight ] && exit 0
python3 -m agentload -c agentload.toml "$MODE" --allow-dirty-corpus
python3 -m agentload -c agentload.toml runs | tail -3
