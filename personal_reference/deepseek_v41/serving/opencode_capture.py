"""Run opencode once against a local capture endpoint: record its real chat request, and
pre-fetch its @ai-sdk/openai-compatible provider into /data. Token counts use the V4.1
chat template.

    python opencode_capture.py /data/dsv41-served OUT_DIR

The capture server answers every /v1/chat/completions call with one short assistant
message, streamed or not, so opencode finishes after one turn. Nothing touches the vLLM
server.
"""
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SERVED, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
PORT = 31999
BODIES = []


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        b = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send(200, json.dumps({"object": "list", "data": [{"id": "dsv41", "object": "model",
                                                                 "max_model_len": 131072}]}))

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        BODIES.append({"path": self.path, "body": req})
        text = "Done."
        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def w(s):
                b = s.encode()
                self.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n")
                self.wfile.flush()
            base = {"id": "x", "object": "chat.completion.chunk", "created": int(time.time()), "model": "dsv41"}
            w("data: " + json.dumps({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]}) + "\n\n")
            w("data: " + json.dumps({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                                     "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}) + "\n\n")
            w("data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        else:
            self._send(200, json.dumps({"id": "x", "object": "chat.completion", "created": int(time.time()),
                                        "model": "dsv41", "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                                                       "finish_reason": "stop"}],
                                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}))


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    cfg = OUT / "opencode.json"
    cfg.write_text(json.dumps({"$schema": "https://opencode.ai/config.json", "autoupdate": False, "provider": {
        "capture": {"npm": "@ai-sdk/openai-compatible", "name": "capture",
                    "options": {"baseURL": f"http://127.0.0.1:{PORT}/v1", "apiKey": "unused"},
                    "models": {"dsv41": {"name": "dsv41", "reasoning": True, "tool_call": True,
                                         "limit": {"context": 131072, "output": 8192}}}}}}, indent=1))
    work = OUT / "work"
    work.mkdir(exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=work)
    xdg = Path("/data/opencode/xdg")
    env = {**os.environ, "OPENCODE_CONFIG": str(cfg), "XDG_DATA_HOME": str(OUT / "data"),
           "XDG_CACHE_HOME": str(xdg / "cache"), "XDG_CONFIG_HOME": str(xdg / "config"),
           "XDG_STATE_HOME": str(xdg / "state"), "PATH": f"/data/opencode/bin:{os.environ['PATH']}"}
    t0 = time.time()
    r = subprocess.run(["opencode", "run", "-m", "capture/dsv41", "List the files here and say hi."],
                       cwd=work, env=env, capture_output=True, text=True, timeout=180)
    print(f"opencode rc={r.returncode} in {time.time() - t0:.0f}s; requests captured: {len(BODIES)}")
    if r.returncode:
        print("stderr:", r.stderr[-1500:])
    chats = [b["body"] for b in BODIES if b["path"].endswith("/chat/completions")]
    (OUT / "requests.json").write_text(json.dumps(BODIES, indent=1))
    if not chats:
        return 1
    from transformers import AutoTokenizer
    sys.path.insert(0, "/data/dsv41-scripts")
    tok = AutoTokenizer.from_pretrained(str(SERVED))
    from vllm.entrypoints.chat_utils import _postprocess_messages
    for i, body in enumerate(chats):
        msgs = body["messages"]
        for m in msgs:          # opencode sends content parts; flatten text parts for the count
            if isinstance(m.get("content"), list):
                m["content"] = "".join(p.get("text", "") for p in m["content"] if isinstance(p, dict))
            if m.get("role") == "developer":
                m["role"] = "system"
        _postprocess_messages(msgs)
        for kw in ({"enable_thinking": False}, {"enable_thinking": True}):
            text = tok.apply_chat_template(msgs, tools=body.get("tools"), tokenize=False,
                                           add_generation_prompt=True, **kw)
            n = len(tok.encode(text, add_special_tokens=False))
            print(f"request {i}: {len(msgs)} messages, {len(body.get('tools') or [])} tools, "
                  f"stream={body.get('stream')}, max_tokens={body.get('max_tokens')}, "
                  f"keys={sorted(body)}; prompt {n} tokens ({kw})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
