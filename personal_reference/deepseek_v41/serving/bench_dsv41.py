#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash serving benchmark against a vLLM OpenAI server. Stdlib only.

    python3 /data/bench_dsv41.py                      # run: sanity, warmup, c = 1, 4, 8, 16
    python3 /data/bench_dsv41.py --dry-run            # build prompts, count tokens, no requests
    python3 /data/bench_dsv41.py --levels 1,4 --per-level 4

The workload is agentic: one ~6000-token shared preamble of real source text plus a
unique ~600-token suffix per request (~90% prefix sharing), 64 output tokens
(ignore_eos), greedy. Requests stream from /v1/completions, so TTFT is measured
directly:

* TTFT = first streamed token minus send.
* TPOT = (last token - first token) / (n_out - 1).
* Per-sequence decode tok/s = 1 / TPOT.
* Aggregate input and output tok/s use the server's own usage counts over the level's
  wall time.

Prefix-cache hits are read as deltas of ``vllm:prefix_cache_{hits,queries}`` from
/metrics around each level. Token counts come from the server's /tokenize. In
--dry-run they come from transformers with the served tokenizer, if importable.
Results go to /data/logs/dsv41-bench-<timestamp>.{json,md}.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import http.client
import json
import os
import statistics
import sys
import time
import urllib.parse
from pathlib import Path

SRC_ROOT = Path(os.environ.get("BENCH_SRC_ROOT", "/data/NotVllm-neuron/vllm_neuron"))
PREAMBLE_TOKENS = 6000
SUFFIX_TOKENS = 600
OUT_TOKENS = 64
DEFAULT_LEVELS = (1, 4, 8, 16)


# ------------------------------------------------------------------- http helpers
class Server:
    def __init__(self, url: str, model: str, timeout: float = 3600):
        u = urllib.parse.urlparse(url)
        self.host, self.port, self.model, self.timeout = u.hostname, u.port or 80, model, timeout

    def _conn(self):
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    def get(self, path: str) -> tuple[int, str]:
        c = self._conn()
        try:
            c.request("GET", path)
            r = c.getresponse()
            return r.status, r.read().decode("utf-8", "replace")
        finally:
            c.close()

    def post(self, path: str, body: dict) -> dict:
        c = self._conn()
        try:
            c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
            r = c.getresponse()
            data = r.read().decode("utf-8", "replace")
            if r.status != 200:
                raise RuntimeError(f"POST {path} -> {r.status}: {data[:500]}")
            return json.loads(data)
        finally:
            c.close()

    def count_tokens(self, text: str) -> int:
        return int(self.post("/tokenize", {"model": self.model, "prompt": text,
                                           "add_special_tokens": False})["count"])

    def stream_completion(self, prompt: str, max_tokens: int) -> dict:
        """One streamed request. Returns timings, usage and text."""
        body = {"model": self.model, "prompt": prompt, "max_tokens": max_tokens,
                "temperature": 0.0, "stream": True, "ignore_eos": True,
                "stream_options": {"include_usage": True}}
        c = self._conn()
        t0 = time.perf_counter()
        first = last = None
        chunks, text, usage = 0, [], None
        try:
            c.request("POST", "/v1/completions", json.dumps(body), {"Content-Type": "application/json"})
            r = c.getresponse()
            if r.status != 200:
                raise RuntimeError(f"/v1/completions -> {r.status}: {r.read()[:500]!r}")
            while True:
                line = r.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                msg = json.loads(payload)
                if msg.get("usage"):
                    usage = msg["usage"]
                for ch in msg.get("choices") or []:
                    if ch.get("text") is not None:
                        now = time.perf_counter()
                        if first is None:
                            first = now
                        last = now
                        chunks += 1
                        text.append(ch["text"])
        finally:
            c.close()
        t1 = time.perf_counter()
        n_out = (usage or {}).get("completion_tokens") or chunks
        n_in = (usage or {}).get("prompt_tokens")
        cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
        return {"t_send": t0, "t_end": t1, "ttft": (first - t0) if first else None,
                "tpot": ((last - first) / (n_out - 1)) if first and n_out > 1 else None,
                "e2e": t1 - t0, "n_out": n_out, "n_in": n_in, "cached_tokens": cached,
                "chunks": chunks, "text": "".join(text)}

    def metrics(self) -> dict[str, float]:
        """Summed vLLM counters we care about (all label sets)."""
        try:
            status, body = self.get("/metrics")
        except OSError:
            return {}
        if status != 200:
            return {}
        out: dict[str, float] = {}
        for line in body.splitlines():
            if line.startswith("#") or not line.startswith("vllm:"):
                continue
            name = line.split("{", 1)[0].split(" ", 1)[0]
            if not any(k in name for k in ("prefix_cache", "prompt_tokens_total",
                                           "generation_tokens_total", "num_preemptions")):
                continue
            try:
                out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
        return out


# ---------------------------------------------------------------- prompt building
def source_lines() -> list[str]:
    lines = []
    for p in sorted(SRC_ROOT.rglob("*.py")):
        try:
            body = p.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        lines.append(f"### FILE: {p.relative_to(SRC_ROOT.parent)}\n")
        lines.extend(l + "\n" for l in body.splitlines())
    return lines


def build_prompts(count, n_requests: int, preamble_tokens: int = PREAMBLE_TOKENS,
                  suffix_tokens: int = SUFFIX_TOKENS):
    """(preamble, [prompt per request], {stats}). ``count`` maps text -> token count."""
    lines = source_lines()
    head = "You are an expert reviewer. The repository files below are shared context.\n\n"
    # preamble: the largest whole-line prefix of the pool at or under the target
    lo, hi = 1, min(len(lines), preamble_tokens)   # >= 1 token per line, so hi suffices
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(head + "".join(lines[:mid])) <= preamble_tokens:
            lo = mid
        else:
            hi = mid - 1
    preamble = head + "".join(lines[:lo])
    pre_n = count(preamble)
    chars_per_tok = len(preamble) / pre_n
    # suffixes: disjoint slices of the rest of the pool, sized by characters
    rest = "".join(lines[lo:])
    per = int(suffix_tokens * chars_per_tok)
    need = (n_requests + 1) * per
    if len(rest) < need:
        rest = (rest + "".join(lines)) * (need // max(len(rest), 1) + 1)
    prompts = []
    for i in range(n_requests + 1):               # index 0 is the warmup
        body = rest[i * per:(i + 1) * per]
        prompts.append(preamble + f"\n### REQUEST {i}\n" + body
                       + f"\n\nQuestion: in one paragraph, what does the code in REQUEST {i} do?\nAnswer:")
    return preamble, prompts, {"preamble_tokens": pre_n, "chars_per_token": round(chars_per_tok, 3)}


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    f = int(k)
    return xs[f] + (xs[min(f + 1, len(xs) - 1)] - xs[f]) * (k - f)


# --------------------------------------------------------------------------- run
def run_level(server: Server, prompts: list[str], conc: int) -> dict:
    m0 = server.metrics()
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=conc) as ex:
        futs = [ex.submit(server.stream_completion, p, OUT_TOKENS) for p in prompts]
        results, errors = [], []
        for f in futs:
            try:
                results.append(f.result())
            except Exception as exc:       # keep going; report
                errors.append(f"{type(exc).__name__}: {exc}"[:300])
    wall = time.perf_counter() - t0
    m1 = server.metrics()
    d = {k: m1.get(k, 0.0) - m0.get(k, 0.0) for k in m1}
    hits = sum(v for k, v in d.items() if "prefix_cache_hits" in k)
    queries = sum(v for k, v in d.items() if "prefix_cache_queries" in k)
    n_in = sum(r["n_in"] or 0 for r in results)
    n_out = sum(r["n_out"] or 0 for r in results)
    tpots = [r["tpot"] for r in results]
    out = {
        "concurrency": conc, "requests": len(prompts), "ok": len(results), "errors": errors,
        "wall_s": wall,
        "ttft_p50_s": pct([r["ttft"] for r in results], 0.5),
        "ttft_p90_s": pct([r["ttft"] for r in results], 0.9),
        "tpot_p50_ms": (pct(tpots, 0.5) or 0) * 1e3 or None,
        "decode_tok_s_per_seq_p50": (1.0 / pct(tpots, 0.5)) if pct(tpots, 0.5) else None,
        "e2e_p50_s": pct([r["e2e"] for r in results], 0.5),
        "agg_output_tok_s": n_out / wall if wall else None,
        "agg_input_tok_s": n_in / wall if wall else None,
        "prompt_tokens_mean": n_in / len(results) if results else None,
        "output_tokens_mean": n_out / len(results) if results else None,
        "cached_tokens_mean": (statistics.mean(r["cached_tokens"] for r in results)
                               if results and all(r["cached_tokens"] is not None for r in results) else None),
        "prefix_cache_hits": hits, "prefix_cache_queries": queries,
        "prefix_cache_hit_rate": (hits / queries) if queries else None,
        "metrics_delta": d,
        "per_request": [{k: r[k] for k in ("ttft", "tpot", "e2e", "n_in", "n_out", "cached_tokens")}
                        for r in results],
    }
    return out


def fmt(v, nd=2):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def markdown(res: dict) -> str:
    rows = ["| conc | reqs | TTFT p50 (s) | TTFT p90 (s) | TPOT p50 (ms) | decode tok/s/seq | "
            "agg out tok/s | agg in tok/s | prompt tok | prefix hit rate |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for lv in res["levels"]:
        rows.append(f"| {lv['concurrency']} | {lv['ok']}/{lv['requests']} | {fmt(lv['ttft_p50_s'])} | "
                    f"{fmt(lv['ttft_p90_s'])} | {fmt(lv['tpot_p50_ms'], 1)} | "
                    f"{fmt(lv['decode_tok_s_per_seq_p50'], 1)} | {fmt(lv['agg_output_tok_s'], 1)} | "
                    f"{fmt(lv['agg_input_tok_s'], 0)} | {fmt(lv['prompt_tokens_mean'], 0)} | "
                    f"{fmt(lv['prefix_cache_hit_rate'], 3)} |")
    s = res["sanity"]
    head = (f"# DeepSeek-V4.1-Flash on trn2.48xlarge — {res['timestamp']}\n\n"
            f"Workload: {res['workload']}\n\n"
            f"Sanity (greedy, 16 tokens): `The capital of France is` -> `{s.get('text', '')!r}`\n\n"
            f"Warmup: TTFT {fmt(res['warmup'].get('ttft'))} s, e2e {fmt(res['warmup'].get('e2e'))} s, "
            f"prompt {res['warmup'].get('n_in')} tokens\n\n")
    return head + "\n".join(rows) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="dsv41")
    ap.add_argument("--levels", default=",".join(map(str, DEFAULT_LEVELS)))
    ap.add_argument("--per-level", type=int, default=0,
                    help="requests per level; default max(4, 2 x concurrency)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tokenizer", default="/data/dsv41-served")
    ap.add_argument("--out-dir", default="/data/logs")
    a = ap.parse_args()
    levels = [int(x) for x in a.levels.split(",") if x]
    per_level = {c: (a.per_level or max(4, 2 * c)) for c in levels}
    n_total = sum(per_level.values())
    server = Server(a.url, a.model)

    if a.dry_run:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(a.tokenizer)
            count = lambda t: len(tok.encode(t, add_special_tokens=False))  # noqa: E731
            how = f"transformers tokenizer from {a.tokenizer}"
        except Exception as exc:
            print(f"no tokenizer ({exc}); using the server's /tokenize")
            count, how = server.count_tokens, "server /tokenize"
    else:
        count, how = server.count_tokens, "server /tokenize"

    t0 = time.time()
    preamble, prompts, stats = build_prompts(count, n_total)
    lens = [count(p) for p in prompts]
    pre_ids_ok = None
    if a.dry_run and how.startswith("transformers"):
        pre = tok.encode(preamble, add_special_tokens=False)
        pre_ids_ok = all(tok.encode(p, add_special_tokens=False)[: len(pre) - 2] == pre[:-2] for p in prompts[:4])
    workload = (f"preamble {stats['preamble_tokens']} tokens shared; prompts {min(lens)}–{max(lens)} tokens "
                f"(mean {statistics.mean(lens):.0f}); sharing {stats['preamble_tokens'] / statistics.mean(lens):.1%}; "
                f"output {OUT_TOKENS} tokens (ignore_eos, greedy); token counts via {how}")
    print(workload, flush=True)
    print(f"prompt build {time.time() - t0:.1f}s; {len(prompts)} prompts (1 warmup + {n_total}); "
          f"levels {per_level}" + (f"; preamble tokens are a prefix of every prompt: {pre_ids_ok}"
                                   if pre_ids_ok is not None else ""), flush=True)
    if a.dry_run:
        return 0

    ts = time.strftime("%Y%m%d-%H%M%S")
    res = {"timestamp": ts, "url": a.url, "model": a.model, "workload": workload,
           "prompt_tokens": lens}
    status, models = server.get("/v1/models")
    res["models"] = json.loads(models) if status == 200 else models
    # sanity: coherent English?
    s = server.post("/v1/completions", {"model": a.model, "prompt": "The capital of France is",
                                        "max_tokens": 16, "temperature": 0.0})
    res["sanity"] = {"text": s["choices"][0]["text"], "usage": s.get("usage")}
    print(f"SANITY: The capital of France is{res['sanity']['text']!r}", flush=True)
    # warmup: populates the prefix cache with the preamble
    w = server.stream_completion(prompts[0], OUT_TOKENS)
    res["warmup"] = {k: w[k] for k in ("ttft", "tpot", "e2e", "n_in", "n_out", "cached_tokens", "text")}
    print(f"warmup: TTFT {fmt(w['ttft'])}s e2e {fmt(w['e2e'])}s in {w['n_in']} out {w['n_out']}", flush=True)
    res["levels"] = []
    i = 1
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in levels:
        lv = run_level(server, prompts[i:i + per_level[c]], c)
        i += per_level[c]
        res["levels"].append(lv)
        print(f"c={c}: TTFT p50 {fmt(lv['ttft_p50_s'])}s p90 {fmt(lv['ttft_p90_s'])}s | TPOT p50 "
              f"{fmt(lv['tpot_p50_ms'], 1)} ms ({fmt(lv['decode_tok_s_per_seq_p50'], 1)} tok/s/seq) | agg out "
              f"{fmt(lv['agg_output_tok_s'], 1)} tok/s, in {fmt(lv['agg_input_tok_s'], 0)} tok/s | prefix hit "
              f"{fmt(lv['prefix_cache_hit_rate'], 3)} | ok {lv['ok']}/{lv['requests']}"
              + (f" | errors {lv['errors'][:2]}" if lv["errors"] else ""), flush=True)
        # write as we go: the instance may not last
        (out_dir / f"dsv41-bench-{ts}.json").write_text(json.dumps(res, indent=1))
        (out_dir / f"dsv41-bench-{ts}.md").write_text(markdown(res))
    print(markdown(res), flush=True)
    print(f"wrote {out_dir}/dsv41-bench-{ts}.{{json,md}}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
