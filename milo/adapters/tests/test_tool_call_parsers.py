"""Tests for the three tool-call parser implementations."""

from __future__ import annotations

import json

from milo.adapters.tool_call_parsers import (
    LlamaToolCallParser,
    OpenAIFunctionsParser,
    QwenToolCallParser,
    get_parser,
    select_parser,
)


# ---------------- OpenAI / litellm / Bedrock shape ----------------


def test_openai_parses_litellm_response() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "tc_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": json.dumps({"path": "src/a.py"}),
                            },
                        }
                    ]
                }
            }
        ]
    }
    calls = OpenAIFunctionsParser().parse_response(response)
    assert len(calls) == 1
    assert calls[0].tool_name == "read_file"
    assert calls[0].tool_input == {"path": "src/a.py"}


def test_openai_parses_bedrock_converse_response() -> None:
    response = {
        "output": {
            "message": {
                "content": [
                    {"toolUse": {"name": "run_command", "input": {"cmd": "ls"}}},
                ]
            }
        }
    }
    calls = OpenAIFunctionsParser().parse_response(response)
    assert len(calls) == 1
    assert calls[0].tool_name == "run_command"
    assert calls[0].tool_input == {"cmd": "ls"}


def test_openai_parses_raw_json_blob_in_string() -> None:
    text = 'I will call: {"name": "submit", "arguments": "{}"}'
    calls = OpenAIFunctionsParser().parse(text)
    assert len(calls) == 1
    assert calls[0].tool_name == "submit"


# ---------------- Qwen <tool_call> ----------------


def test_qwen_parses_single_block() -> None:
    text = (
        "Thinking...\n"
        '<tool_call>{"name": "apply_patch", "arguments": {"diff": "diff..."}}</tool_call>\n'
        "done."
    )
    calls = QwenToolCallParser().parse(text)
    assert len(calls) == 1
    assert calls[0].tool_name == "apply_patch"
    assert calls[0].tool_input == {"diff": "diff..."}


def test_qwen_parses_multiple_blocks() -> None:
    text = (
        '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
        '<tool_call>{"name": "b", "arguments": {"x": 1}}</tool_call>'
    )
    calls = QwenToolCallParser().parse(text)
    assert [c.tool_name for c in calls] == ["a", "b"]


def test_qwen_parse_error_does_not_crash() -> None:
    text = "<tool_call>{not json}</tool_call>"
    calls = QwenToolCallParser().parse(text)
    assert len(calls) == 1
    assert calls[0].parse_errors
    assert calls[0].tool_name == ""


# ---------------- Llama <|python_tag|> ----------------


def test_llama_parses_json_body() -> None:
    text = (
        '<|python_tag|>{"name": "search_grep", "parameters": {"pattern": "TODO"}}<|eom_id|>'
    )
    calls = LlamaToolCallParser().parse(text)
    assert len(calls) == 1
    assert calls[0].tool_name == "search_grep"
    assert calls[0].tool_input == {"pattern": "TODO"}


# ---------------- Registry / selection ----------------


def test_get_parser_known() -> None:
    assert isinstance(get_parser("openai"), OpenAIFunctionsParser)
    assert isinstance(get_parser("qwen"), QwenToolCallParser)
    assert isinstance(get_parser("llama"), LlamaToolCallParser)


def test_select_parser_by_tokenizer_name() -> None:
    assert isinstance(select_parser(None), OpenAIFunctionsParser)
    assert isinstance(select_parser("Qwen/Qwen2.5-Coder-32B-Instruct"), QwenToolCallParser)
    assert isinstance(select_parser("meta-llama/Llama-3.1-8B-Instruct"), LlamaToolCallParser)
    assert isinstance(select_parser("anthropic/claude-opus-4-6"), OpenAIFunctionsParser)
    # Qwen with native tool grammar → OpenAI parser
    assert isinstance(select_parser("Qwen/Qwen2.5-Coder-tool-grammar"), OpenAIFunctionsParser)
