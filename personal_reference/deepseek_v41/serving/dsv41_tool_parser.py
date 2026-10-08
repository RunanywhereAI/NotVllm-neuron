# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 DSML tool-call parser for vLLM 0.24.0, loaded with ``--tool-parser-plugin``.

    vllm serve ... --enable-auto-tool-choice --tool-parser-plugin /data/dsv41_tool_parser.py \\
        --tool-call-parser deepseek_v41

V4.1 keeps the DSML grammar of V3.2 and V4 (invoke / parameter with ``string="true|false"``)
and changes only the tag names. The block is ``<｜DSML｜ calls>`` instead of
``<｜DSML｜tool_calls>``, and every tag name now carries a leading space:
``<｜DSML｜ invoke name=..>``, ``<｜DSML｜ parameter ..>``. See the checkpoint's
``encoding/README.md`` ("V4.1 changes relative to V4") and ``chat_template.jinja``. Upstream
vLLM serves this as ``deepseekv41_engine_tool_parser`` on its newer parser engine, which
0.24 does not have.

So this subclasses 0.24's ``DeepSeekV32ToolParser``, whose grammar matches, and folds the
V4.1 tags onto it. The fold, ``<｜DSML｜ x`` to ``<｜DSML｜x``, runs on the whole output when
not streaming. When streaming it runs on the accumulated buffer, never on a single delta,
because the space can arrive in the token after ``｜DSML｜``. The block tokens become
``<｜DSML｜calls>`` after the fold.

``structural_tag_model`` is None. vLLM 0.24's builtin structural tags know the V4 block
name, not V4.1's, and a wrong grammar is worse than none.

Validated offline by ``validate_dsv41_tools.py`` against the checkpoint's own encoder and
its ``parse_message_from_completion_text``.
"""
from __future__ import annotations

from collections.abc import Sequence

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.deepseekv32_tool_parser import DeepSeekV32ToolParser


def fold_v41_tags(text: str) -> str:
    """``<｜DSML｜ calls>`` -> ``<｜DSML｜calls>``, and the same for invoke / parameter and
    the closing tags. Text without the DSML marker is returned unchanged."""
    if "｜DSML｜ " not in text:
        return text
    return text.replace("<｜DSML｜ ", "<｜DSML｜").replace("</｜DSML｜ ", "</｜DSML｜")


@ToolParserManager.register_module("deepseek_v41")
class DeepSeekV41ToolParser(DeepSeekV32ToolParser):
    tool_call_start_token: str = "<｜DSML｜calls>"     # after fold_v41_tags
    tool_call_end_token: str = "</｜DSML｜calls>"
    structural_tag_model = None

    def extract_tool_calls(self, model_output, request):
        return super().extract_tool_calls(fold_v41_tags(model_output), request)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request,
    ) -> DeltaMessage | None:
        # The body of DeepSeekV32ToolParser.extract_tool_calls_streaming (vLLM 0.24.0),
        # with the fold applied to the buffer rather than to the delta.
        if not previous_text:
            self._reset_streaming_state()
        self._buffer = fold_v41_tags(self._buffer + delta_text)
        content_parts: list[str] = []
        tool_call_deltas: dict = {}
        self._process_streaming_buffer(content_parts, tool_call_deltas)
        if content_parts or tool_call_deltas:
            content = "".join(content_parts) or None
            return DeltaMessage(content=content, tool_calls=list(tool_call_deltas.values()))
        if not delta_text and delta_token_ids and self.prev_tool_call_arr:
            return DeltaMessage(content="")
        return None
