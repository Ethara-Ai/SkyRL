"""Tool-call parsers — Phase 7 / spec §15.3 / plan §15.

Three parser implementations cover every shipped policy adapter:

    * `OpenAIFunctionsParser` — OpenAI Chat Completions `tools` schema
      (also Bedrock `converse` toolUse blocks, also litellm normalised
      output, also vLLM-OpenAI-compat). The 99% case.
    * `QwenToolCallParser` — Qwen-Coder's `<tool_call>...</tool_call>`
      XML-tagged format. Used when the policy is Qwen and we configure
      it without a tool-call grammar.
    * `LlamaToolCallParser` — Llama-3's `<|python_tag|>...<|eom_id|>`
      format. Used when the policy is Llama-3 family.

The rollout driver auto-selects via `select_parser(tokenizer_name)`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class ToolCall:
    """Parsed tool call. ``raw`` keeps the original model text for replay."""

    tool_name: str
    tool_input: dict[str, Any]
    raw: str = ""
    parse_errors: list[str] = field(default_factory=list)


class ToolCallParser(Protocol):
    """Stateless parser. Returns 0 or more ToolCall objects."""

    def parse(self, model_output: str) -> list[ToolCall]: ...


# ----------------------------------------------------------------------- OpenAI


class OpenAIFunctionsParser:
    """Parses the OpenAI / Bedrock-converse / litellm-normalised tool-use shape.

    Accepts either:
      * a string containing one or more `{"tool_name": ..., "arguments": ...}`
        JSON blobs (the simple per-message case), or
      * the full response dict with `tool_calls` (the API-shape case).

    The rollout driver always calls `parse_response(...)` on the dict;
    `parse(text)` is a convenience for unit tests with raw strings.
    """

    def parse(self, model_output: str) -> list[ToolCall]:
        calls: list[ToolCall] = []
        # Brace-balanced scan: find every top-level JSON object that contains
        # a `"name"` or `"tool_name"` key. Handles nested braces in arguments
        # (e.g. `{"name": "submit", "arguments": "{}"}`) where a naive
        # `\{[^{}]*\}` regex would miss the closing brace.
        i = 0
        n = len(model_output)
        while i < n:
            if model_output[i] != "{":
                i += 1
                continue
            depth = 0
            start = i
            in_str = False
            escape = False
            end = None
            while i < n:
                ch = model_output[i]
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = not in_str
                elif not in_str:
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                i += 1
            if end is None:
                break  # unbalanced — stop
            blob = model_output[start:end]
            i = end
            # Quick reject: must mention a tool-name key.
            if '"name"' not in blob and '"tool_name"' not in blob:
                continue
            try:
                obj = json.loads(blob)
            except json.JSONDecodeError as exc:
                calls.append(
                    ToolCall(tool_name="", tool_input={}, raw=blob,
                             parse_errors=[f"json decode: {exc}"])
                )
                continue
            if not isinstance(obj, dict):
                continue
            name = obj.get("tool_name") or obj.get("name") or ""
            args = obj.get("arguments") or obj.get("tool_input") or obj.get("input") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            calls.append(ToolCall(tool_name=name, tool_input=args, raw=blob))
        return calls

    def parse_response(self, response: dict[str, Any]) -> list[ToolCall]:
        """OpenAI/litellm response dict → ToolCalls.

        Handles both `choices[0].message.tool_calls` (OpenAI/litellm) and the
        Bedrock `converse` `output.message.content[*].toolUse` block.
        """
        # OpenAI / litellm path
        choices = response.get("choices") or []
        if choices:
            msg = (choices[0] or {}).get("message") or {}
            tcs = msg.get("tool_calls") or []
            out: list[ToolCall] = []
            for tc in tcs:
                fn = tc.get("function") or tc
                name = fn.get("name", "")
                args_raw = fn.get("arguments", "{}")
                if isinstance(args_raw, str):
                    try:
                        args = json.loads(args_raw)
                    except json.JSONDecodeError:
                        args = {}
                else:
                    args = args_raw
                out.append(ToolCall(tool_name=name, tool_input=args, raw=json.dumps(tc)))
            if out:
                return out

        # Bedrock converse path
        out_msg = (response.get("output") or {}).get("message") or {}
        content = out_msg.get("content") or []
        out2: list[ToolCall] = []
        for block in content:
            tu = block.get("toolUse")
            if not tu:
                continue
            out2.append(
                ToolCall(
                    tool_name=tu.get("name", ""),
                    tool_input=tu.get("input", {}) or {},
                    raw=json.dumps(block),
                )
            )
        return out2


# ------------------------------------------------------------------------- Qwen


class QwenToolCallParser:
    """Parses Qwen-Coder `<tool_call>...</tool_call>` blocks.

    Each block contains a JSON object `{"name": "...", "arguments": {...}}`.
    """

    _BLOCK_RE = re.compile(
        r"<tool_call>\s*(?P<body>\{.*?\})\s*</tool_call>", re.DOTALL
    )

    def parse(self, model_output: str) -> list[ToolCall]:
        out: list[ToolCall] = []
        for match in self._BLOCK_RE.finditer(model_output):
            body = match.group("body")
            try:
                obj = json.loads(body)
            except json.JSONDecodeError as exc:
                out.append(
                    ToolCall(
                        tool_name="",
                        tool_input={},
                        raw=match.group(0),
                        parse_errors=[f"json decode: {exc}"],
                    )
                )
                continue
            out.append(
                ToolCall(
                    tool_name=obj.get("name", ""),
                    tool_input=obj.get("arguments", {}),
                    raw=match.group(0),
                )
            )
        return out


# ------------------------------------------------------------------------ Llama


class LlamaToolCallParser:
    """Parses Llama-3 `<|python_tag|>...<|eom_id|>` blocks.

    The body is either a JSON object (with `name`/`parameters`) or a
    Python-style `function_name(arg1=...)` call. We only handle the JSON form
    here; the Python-style form is rare in practice and will surface as a
    parse error (which the calling driver will treat as TIR -1 per spec
    §4.4.7).
    """

    _BLOCK_RE = re.compile(
        r"<\|python_tag\|>\s*(?P<body>.*?)\s*<\|eom_id\|>", re.DOTALL
    )

    def parse(self, model_output: str) -> list[ToolCall]:
        out: list[ToolCall] = []
        for match in self._BLOCK_RE.finditer(model_output):
            body = match.group("body").strip()
            try:
                obj = json.loads(body)
                out.append(
                    ToolCall(
                        tool_name=obj.get("name", ""),
                        tool_input=obj.get("parameters", obj.get("arguments", {})),
                        raw=match.group(0),
                    )
                )
            except json.JSONDecodeError as exc:
                out.append(
                    ToolCall(
                        tool_name="",
                        tool_input={},
                        raw=match.group(0),
                        parse_errors=[f"json decode: {exc}"],
                    )
                )
        return out


# ---------------------------------------------------------------------- registry


_PARSERS: dict[str, ToolCallParser] = {
    "openai": OpenAIFunctionsParser(),
    "qwen": QwenToolCallParser(),
    "llama": LlamaToolCallParser(),
}


def get_parser(name: str) -> ToolCallParser:
    name = name.lower()
    if name not in _PARSERS:
        raise KeyError(f"unknown tool-call parser {name!r}; have {sorted(_PARSERS)}")
    return _PARSERS[name]


def select_parser(tokenizer_name: str | None) -> ToolCallParser:
    """Auto-select a parser based on the tokenizer / model name.

    Per plan §15: Qwen-tokenizer → QwenToolCallParser; Llama-tokenizer →
    LlamaToolCallParser; everything else → OpenAIFunctionsParser (covers
    Anthropic, OpenAI, Gemini, Mistral, vLLM-served Qwen-with-tools-grammar).
    """
    if not tokenizer_name:
        return _PARSERS["openai"]
    name = tokenizer_name.lower()
    if "qwen" in name and "tool" not in name:
        # Qwen models served *without* the native tool-call grammar
        # use the <tool_call> tagged form. Qwen served *with* the grammar
        # uses the OpenAI shape; tokenizer_name typically contains "tool"
        # or "function" in that case.
        return _PARSERS["qwen"]
    if "llama" in name and "3" in name:
        return _PARSERS["llama"]
    return _PARSERS["openai"]
