"""Tool Integrity Reward tests — spec §4.4.7, plan §4.3 acceptance.

Plan §4.3 mandates at least four cases: valid call → 0; missing required
field → -1; wrong type → -1; `apply_patch` that fails to apply → 0 (the
scope-regression guard). Plus a few extras for full coverage of the 6 tools.
"""

from __future__ import annotations

import json

from milo.reward.tir import MILO_TOOL_SCHEMAS, compute_tir


# ---------------------------------------------------------------------------
# Mandatory cases per plan §4.3
# ---------------------------------------------------------------------------


def test_valid_read_file_returns_zero() -> None:
    call = json.dumps(
        {"name": "read_file", "parameters": {"path": "/workspace/repo/main.py"}}
    )
    assert compute_tir(call) == 0


def test_missing_required_field_returns_minus_one() -> None:
    """read_file missing `path` (required) → -1."""
    call = json.dumps({"name": "read_file", "parameters": {"start_line": 1}})
    assert compute_tir(call) == -1


def test_wrong_type_returns_minus_one() -> None:
    """read_file with non-string `path` → -1."""
    call = json.dumps({"name": "read_file", "parameters": {"path": 123}})
    assert compute_tir(call) == -1


def test_apply_patch_that_fails_to_apply_is_zero_not_minus_one() -> None:
    """SCOPE GUARD: schema-valid call whose execution will fail → 0.

    This is the most important regression test in the TIR module — spec
    D-17 and §4.4.7 are explicit that TIR does NOT penalize valid-but-
    failed tool calls (those are training signal).
    """
    call = json.dumps(
        {
            "name": "apply_patch",
            "parameters": {"diff": "this is not a valid unified diff"},
        }
    )
    assert compute_tir(call) == 0


# ---------------------------------------------------------------------------
# Extra coverage
# ---------------------------------------------------------------------------


def test_malformed_json_returns_minus_one() -> None:
    """A non-parseable JSON string for the call payload → -1."""
    assert compute_tir("{not json at all", tool_name="read_file") == -1


def test_unknown_tool_returns_zero_no_schema_to_fail_against() -> None:
    """Tool not in registry → 0 (the gym rejects upstream; not a TIR concern)."""
    call = json.dumps({"name": "totally_made_up", "parameters": {}})
    assert compute_tir(call) == 0


def test_unknown_field_returns_minus_one_additional_properties_false() -> None:
    """Unknown fields are rejected (additionalProperties: False)."""
    call = json.dumps(
        {"name": "read_file", "parameters": {"path": "/x", "noisy": True}}
    )
    assert compute_tir(call) == -1


def test_list_files_required_path_missing() -> None:
    call = json.dumps({"name": "list_files", "parameters": {"recursive": True}})
    assert compute_tir(call) == -1


def test_search_grep_pattern_required() -> None:
    call = json.dumps(
        {"name": "search_grep", "parameters": {"file_glob": "*.py"}}
    )
    assert compute_tir(call) == -1


def test_run_command_valid_minimal() -> None:
    call = json.dumps({"name": "run_command", "parameters": {"cmd": "ls"}})
    assert compute_tir(call) == 0


def test_run_command_timeout_out_of_range() -> None:
    """timeout=601 exceeds the max=600 in the schema."""
    call = json.dumps(
        {"name": "run_command", "parameters": {"cmd": "ls", "timeout": 601}}
    )
    assert compute_tir(call) == -1


def test_submit_with_optional_summary() -> None:
    call = json.dumps({"name": "submit", "parameters": {"summary": "fixed"}})
    assert compute_tir(call) == 0


def test_submit_with_no_parameters() -> None:
    call = json.dumps({"name": "submit", "parameters": {}})
    assert compute_tir(call) == 0


def test_dict_input_accepted_directly() -> None:
    """compute_tir tolerates both raw JSON string and pre-parsed dict."""
    call_dict = {
        "name": "read_file",
        "parameters": {"path": "/x"},
    }
    assert compute_tir(call_dict) == 0


def test_dict_input_with_invalid_params_returns_minus_one() -> None:
    call_dict = {
        "name": "read_file",
        "parameters": {"path": 12345},
    }
    assert compute_tir(call_dict) == -1


def test_tool_name_argument_overrides_payload() -> None:
    """If caller passes tool_name explicitly, use it even when payload is just params."""
    # Just the params, no wrapping {"name":..., "parameters":...}
    call = json.dumps({"path": "/x"})
    assert compute_tir(call, tool_name="read_file") == 0
    call = json.dumps({"start_line": 1})
    assert compute_tir(call, tool_name="read_file") == -1


def test_all_six_tool_names_are_registered() -> None:
    """Sanity-check the registry against the spec §4.2 tool list."""
    expected = {
        "read_file",
        "list_files",
        "search_grep",
        "apply_patch",
        "run_command",
        "submit",
    }
    assert set(MILO_TOOL_SCHEMAS.keys()) == expected


def test_apply_patch_missing_diff_field() -> None:
    call = json.dumps({"name": "apply_patch", "parameters": {}})
    assert compute_tir(call) == -1
