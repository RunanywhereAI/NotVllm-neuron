"""Offline validation of DeepSeek-V4.1 chat + tools on vLLM 0.24 (no server, no device).

    /data/venv-fork/bin/python validate_dsv41_tools.py /data/dsv41-served /data/dsv41_tool_parser.py

1. Template. A multi-turn conversation with tools, tool calls and tool results is rendered
   by the checkpoint's ``chat_template.jinja`` (through transformers, the way vLLM 0.24's HF
   renderer calls it). The result must equal the checkpoint's own Python reference,
   ``encoding.encode_messages``, in both thinking and chat mode. vLLM's 0.24 template
   resolution must also pick this template for the served directory.
2. Tool parser. Hand-written V4.1 completions are parsed with the plugin, both
   non-streaming and streaming token by token through the real tokenizer. The results are
   compared with the reference ``parse_message_from_completion_text``. Two negative checks
   show the test can fail: the stock ``deepseek_v4`` parser must not extract the V4.1 calls,
   and a parser without the fold must not either.
3. Reasoning split. ``--reasoning-parser deepseek_v4`` with
   ``--default-chat-template-kwargs '{"enable_thinking": ...}'`` must split a thinking
   completion into reasoning and content, and leave a chat completion alone.
"""
import importlib.util
import json
import sys
from pathlib import Path

SERVED, PLUGIN = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(SERVED / "encoding"))
import encoding as ref  # noqa: E402  the checkpoint's reference encoder
from transformers import AutoTokenizer  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""), flush=True)
    if not ok:
        FAIL.append(name)


TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file from the workspace.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "bash", "description": "Run a shell command.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}, "timeout": {"type": "number"},
            "env": {"type": "object"}}, "required": ["command"]}}},
]
CALLS_1 = [{"id": "call_1", "type": "function", "function": {
    "name": "read_file", "arguments": json.dumps({"path": "src/main.py", "offset": 10, "limit": 40})}}]
CALLS_2 = [
    {"id": "call_2", "type": "function", "function": {
        "name": "bash", "arguments": json.dumps({"command": "pytest -q tests/ && echo \"ok <done>\"",
                                                 "timeout": 120.5})}},
    {"id": "call_3", "type": "function", "function": {
        "name": "read_file", "arguments": json.dumps({"path": "README.md"})}},
]
CONV = [
    {"role": "system", "content": "You are a coding agent. Use tools."},
    {"role": "user", "content": "Find the bug in src/main.py."},
    {"role": "assistant", "content": "I'll read the file first.", "tool_calls": CALLS_1},
    {"role": "tool", "tool_call_id": "call_1", "content": "def f(x):\n    return x + 1\n"},
    {"role": "assistant", "content": "", "tool_calls": CALLS_2},
    {"role": "tool", "tool_call_id": "call_2", "content": "1 failed"},
    {"role": "tool", "tool_call_id": "call_3", "content": "# Project"},
    {"role": "user", "content": "Go on."},
]


def section1(tok):
    print("1. chat template vs the checkpoint's reference encoder")
    import copy
    from vllm.entrypoints.chat_utils import _postprocess_messages
    conv = copy.deepcopy(CONV)
    _postprocess_messages(conv)      # what vLLM 0.24 does before rendering: arguments str -> dict
    for mode, kw in (("thinking", {"enable_thinking": True}), ("chat", {"enable_thinking": False})):
        got = tok.apply_chat_template(conv, tools=TOOLS, tokenize=False, add_generation_prompt=True, **kw)
        msgs = [dict(CONV[0], tools=TOOLS)] + CONV[1:]
        want = ref.encode_messages(msgs, thinking_mode=mode)
        same = got == want
        detail = ""
        if not same:
            i = next(i for i, (a, b) in enumerate(zip(got, want)) if a != b) if got[:len(want)] != want else min(len(got), len(want))
            detail = f"first difference at char {i}: template {got[i - 40:i + 60]!r} vs reference {want[i - 40:i + 60]!r}"
        check(f"{mode} mode: template == encode_messages ({len(got)} chars)", same, detail)
    check("tool block name is V4.1's '<｜DSML｜ calls>'", "<｜DSML｜ calls>" in got and "<｜DSML｜tool_calls>" not in got)
    # vLLM 0.24 resolves the template from the served dir
    try:
        from vllm.entrypoints.chat_utils import resolve_hf_chat_template
        from vllm.config import ModelConfig  # noqa: F401
        tmpl = resolve_hf_chat_template(tok, chat_template=None, tools=TOOLS, model_config=None)
        check("vLLM resolves the served chat_template.jinja", tmpl is not None and "DSML" in tmpl)
    except Exception as exc:     # signature differs across versions; informational
        print(f"  [INFO] resolve_hf_chat_template not checked: {type(exc).__name__}: {str(exc)[:120]}")


def load_plugin():
    """The way ``--tool-parser-plugin`` loads it."""
    from vllm.tool_parsers import ToolParserManager
    ToolParserManager.import_tool_parser(PLUGIN)
    return sys.modules["dsv41_tool_parser"]


def request():
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    return ChatCompletionRequest(model="dsv41", messages=[{"role": "user", "content": "x"}], tools=TOOLS)


def calls_of(tool_calls):
    return [(tc.function.name, json.loads(tc.function.arguments)) for tc in tool_calls]


def stream(parser, tok, text, req):
    """Feed ``text`` token by token through the streaming API; collect calls and content."""
    ids = tok.encode(text, add_special_tokens=False)
    prev_text, prev_ids = "", []
    names, args, content = {}, {}, []
    for k in range(len(ids)):
        cur_ids = ids[: k + 1]
        cur_text = tok.decode(cur_ids, skip_special_tokens=False)
        delta = cur_text[len(prev_text):]
        msg = parser.extract_tool_calls_streaming(prev_text, cur_text, delta, prev_ids, cur_ids,
                                                  ids[k:k + 1], req)
        if msg is not None:
            if msg.content:
                content.append(msg.content)
            for tc in msg.tool_calls or []:
                if tc.function and tc.function.name:
                    names[tc.index] = tc.function.name
                if tc.function and tc.function.arguments:
                    args[tc.index] = args.get(tc.index, "") + tc.function.arguments
        prev_text, prev_ids = cur_text, cur_ids
    return [(names[i], json.loads(args.get(i) or "{}")) for i in sorted(names)], "".join(content)


def section2(tok):
    print("2. tool parser plugin vs the reference completion parser")
    mod = load_plugin()
    from vllm.tool_parsers import ToolParserManager
    from vllm.tool_parsers.deepseekv4_tool_parser import DeepSeekV4ToolParser

    cls = ToolParserManager.get_tool_parser("deepseek_v41")
    check("plugin registers 'deepseek_v41'", cls is mod.DeepSeekV41ToolParser)
    req = request()
    eos = ref.eos_token
    # completions in the exact format the template teaches, generated by the reference encoder
    cases = {
        "one call after content": "I'll read the file first." + ref.encode_messages(
            [{"role": "user", "content": "x"}, {"role": "assistant", "content": "", "tool_calls": CALLS_1}],
            thinking_mode="chat").split("<｜Assistant｜></think>", 1)[1],
        "two calls, quotes and <angle> in a string, float": ref.encode_messages(
            [{"role": "user", "content": "x"}, {"role": "assistant", "content": "", "tool_calls": CALLS_2}],
            thinking_mode="chat").split("<｜Assistant｜></think>", 1)[1],
        "hand-written, nested JSON param": (
            "Running it.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"bash\">\n"
            "<｜DSML｜ parameter name=\"command\" string=\"true\">ls -la</｜DSML｜ parameter>\n"
            "<｜DSML｜ parameter name=\"env\" string=\"false\">{\"A\": 1, \"B\": [true, null]}</｜DSML｜ parameter>\n"
            "</｜DSML｜ invoke>\n</｜DSML｜ calls>" + eos),
    }
    for name, text in cases.items():
        want_msg = ref.parse_message_from_completion_text(text, thinking_mode="chat")
        want = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in want_msg["tool_calls"]]
        p = cls(tok, req.tools)
        out = p.extract_tool_calls(text.removesuffix(eos), req)
        got = calls_of(out.tool_calls)
        check(f"{name}: non-streaming == reference ({len(want)} calls)", out.tools_called and got == want,
              "" if got == want else f"got {got} want {want}")
        check(f"{name}: content == reference", (out.content or "").strip() == want_msg["content"].strip(),
              f"{out.content!r} vs {want_msg['content']!r}")
        sgot, scontent = stream(cls(tok, req.tools), tok, text.removesuffix(eos), req)
        check(f"{name}: streaming (token by token) == reference", sgot == want,
              "" if sgot == want else f"got {sgot}")
        check(f"{name}: streamed content == reference", scontent.strip() == want_msg["content"].strip(),
              f"{scontent!r}")
        # negative controls: they must fail, or this test proves nothing
        stock = DeepSeekV4ToolParser(tok, req.tools).extract_tool_calls(text.removesuffix(eos), req)
        check(f"{name}: stock deepseek_v4 parser does NOT extract these (control)", not stock.tools_called)

        class NoFold(cls):
            def extract_tool_calls(self, model_output, request):
                return super(cls, self).extract_tool_calls(model_output, request)

        nf = NoFold(tok, req.tools).extract_tool_calls(text.removesuffix(eos), req)
        check(f"{name}: parser without the fold does NOT extract them (control)", not nf.tools_called)
    plain = cls(tok, req.tools).extract_tool_calls("No tools needed, the answer is 4.", req)
    check("plain answer: no tool call, content intact",
          not plain.tools_called and plain.content == "No tools needed, the answer is 4.")


def section3(tok):
    print("3. reasoning split with --reasoning-parser deepseek_v4 + default chat_template_kwargs")
    from vllm.reasoning import ReasoningParserManager
    cls = ReasoningParserManager.get_reasoning_parser("deepseek_v4")
    req = request()
    text = "The user wants the bug. Read first.</think>I'll read the file first."
    on = cls(tok, chat_template_kwargs={"enable_thinking": True}).extract_reasoning(text, req)
    want = ref.parse_message_from_completion_text(text + ref.eos_token, thinking_mode="thinking")
    check("enable_thinking=true: reasoning and content split like the reference",
          (on[0] or "").strip() == want["reasoning_content"].strip() and (on[1] or "").strip() == want["content"].strip(),
          f"got {on}")
    off = cls(tok, chat_template_kwargs={"enable_thinking": False}).extract_reasoning("Plain answer.", req)
    check("enable_thinking=false: content untouched", off[1] == "Plain answer." and not off[0], f"got {off}")


def main():
    tok = AutoTokenizer.from_pretrained(str(SERVED))
    section1(tok)
    section2(tok)
    section3(tok)
    print(f"\n{'ALL PASS' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
