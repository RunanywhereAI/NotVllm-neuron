#!/bin/bash
# Poll the dsv41 server every 60 s; when /v1/models answers, start the benchmark detached and exit.
# Exits 2 if the server log shows the engine failed, 3 after the deadline.
deadline=$(( $(date +%s) + ${1:-10800} ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  if curl -sf -m 10 http://localhost:8000/v1/models >/dev/null; then
    echo "SERVER UP $(date -u +%H:%M:%SZ)"
    setsid nohup /usr/bin/python3 /data/bench_dsv41.py > /data/logs/dsv41-bench-run.log 2>&1 < /dev/null &
    echo "bench started pid $!"
    exit 0
  fi
  if grep -q "Engine core initialization failed" /data/logs/dsv41-serve.log 2>/dev/null; then
    echo "SERVER FAILED $(date -u +%H:%M:%SZ)"; grep -m3 -E "Error|error" /data/logs/dsv41-serve.log | cut -c1-300
    exit 2
  fi
  sleep 60
done
echo "DEADLINE"; exit 3
