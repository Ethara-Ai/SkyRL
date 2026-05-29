"""Tool Integrity Reward (TIR) — spec §4.4.7.

Implements `compute_tir` and the default `MILO_TOOL_SCHEMAS` registry for
the six gym tools defined in spec §4.2. TIR scope is narrow and deliberate:
`-1` only when a tool call fails JSON-schema validation (malformed JSON,
missing required field, wrong type, unknown field); `0` for *every* other
case, including valid calls whose execution fails (those are training signal,
per spec D-17).
"""

from __future__ import annotations

import json
from typing import Any

import jsonschema

# Draft202012Validator is the strictest available draft and was added in
# jsonschema 4.0. Fall back to Draft7Validator on older installs (e.g.
# system jsonschema 3.x) — Draft 7 is sufficient for the simple schemas
# in MILO_TOOL_SCHEMAS (no draft-2020 features like prefixItems are used).
try:
    from jsonschema import Draft202012Validator as _DefaultValidator  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - environment-dependent
    from jsonschema import Draft7Validator as _DefaultValidator  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Default tool-schema registry — the 6 tools from spec §4.2.
#
# These mirror the normative JSON schemas in the spec (§4.2.1 .. §4.2.6) but
# are written for use with `jsonschema.Draft202012Validator` so we can run
# `additionalProperties: False` for unknown-field detection (the spec
# definition is informal; we apply `additionalProperties: False` everywhere
# to make the validation strict — this matches the spec language "unknown
# fields").
# ---------------------------------------------------------------------------

_READ_FILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["path"],
    "properties": {
        "path": {"type": "string"},
        "start_line": {"type": "integer", "minimum": 1, "default": 1},
        "end_line": {"type": "integer", "minimum": 1},
    },
}

_LIST_FILES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["path"],
    "properties": {
        "path": {"type": "string"},
        "recursive": {"type": "boolean", "default": False},
        "max_depth": {"type": "integer", "minimum": 1, "maximum": 8, "default": 3},
    },
}

_SEARCH_GREP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["pattern"],
    "properties": {
        "pattern": {"type": "string"},
        "scope": {"type": "string"},
        "file_glob": {"type": "string"},
        "case_sensitive": {"type": "boolean", "default": True},
        "max_results": {
            "type": "integer",
            "minimum": 1,
            "maximum": 1000,
            "default": 200,
        },
    },
}

_APPLY_PATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["diff"],
    "properties": {"diff": {"type": "string"}},
}

_RUN_COMMAND_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["cmd"],
    "properties": {
        "cmd": {"type": "string"},
        "timeout": {
            "type": "integer",
            "minimum": 1,
            "maximum": 600,
            "default": 300,
        },
        "cwd": {"type": "string"},
    },
}

_SUBMIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"summary": {"type": "string"}},
}


MILO_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "read_file": _READ_FILE_SCHEMA,
    "list_files": _LIST_FILES_SCHEMA,
    "search_grep": _SEARCH_GREP_SCHEMA,
    "apply_patch": _APPLY_PATCH_SCHEMA,
    "run_command": _RUN_COMMAND_SCHEMA,
    "submit": _SUBMIT_SCHEMA,
}

# Pre-compile validators once at import time. Draft202012Validator is the
# strictest available Draft in `jsonschema`; it enforces `required`, type
# coercion semantics, and (with `additionalProperties: False` per-schema)
# unknown-field rejection.
_VALIDATORS: dict[str, Any] = {
    name: _DefaultValidator(schema) for name, schema in MILO_TOOL_SCHEMAS.items()
}


def _parse_call(tool_call: str | dict[str, Any]) -> tuple[str | None, Any]:
    """Best-effort parse of a tool call into `(name, params)`.

    The trainer / gym may hand us either:
      * a JSON string of `{"name": "...", "parameters": {...}}` (preferred),
      * a JSON string of just the parameters (caller knows the name), or
      * an already-parsed dict.

    Returns `(name, params_obj)`. `name` is None if not embedded in the
    payload. `params_obj` is whatever the JSON parsed to (None on parse
    error → caller treats as schema-invalid).
    """
    if isinstance(tool_call, dict):
        if "parameters" in tool_call:
            return tool_call.get("name"), tool_call["parameters"]
        # Caller already handed us parameters directly.
        return tool_call.get("name"), tool_call
    if not isinstance(tool_call, str):
        return None, None
    try:
        parsed = json.loads(tool_call)
    except (json.JSONDecodeError, ValueError):
        return None, None
    if isinstance(parsed, dict) and "parameters" in parsed:
        return parsed.get("name"), parsed["parameters"]
    return None, parsed


def compute_tir(
    tool_call: str | dict[str, Any],
    tool_schemas: dict[str, dict[str, Any]] | None = None,
    *,
    tool_name: str | None = None,
) -> int:
    """Return -1 if `tool_call` fails JSON-schema validation; 0 otherwise.

    Implements spec §4.4.7. The narrow scope is enforced here: this function
    *only* judges schema validity. It returns 0 for valid-but-failed tool
    calls (e.g. `apply_patch` of a diff that doesn't apply, `read_file` of a
    missing path, `run_command` whose tests fail) — those are training signal
    per spec D-17.

    Parameters
    ----------
    tool_call:
        Either a JSON string from the LLM (preferred for honest schema
        checks: this catches malformed JSON, which a pre-parsed dict cannot),
        or an already-parsed dict. The function tolerates both because some
        integrators parse upstream of the reward path.
    tool_schemas:
        Registry mapping tool name → JSON Schema. Defaults to
        `MILO_TOOL_SCHEMAS` (the 6 spec-§4.2 tools).
    tool_name:
        If the caller knows which tool was being called, pass it here.
        Otherwise we try to read it from the payload (`tool_call["name"]`).
        When neither is available we cannot decide which schema to validate
        against, so we conservatively return 0 — there's no JSON-schema
        violation to detect when there's no schema to check.

    Returns
    -------
    int: -1 (schema-invalid) or 0 (everything else).
    """
    schemas = tool_schemas if tool_schemas is not None else MILO_TOOL_SCHEMAS

    embedded_name, params = _parse_call(tool_call)
    # Malformed JSON or non-string/non-dict input → schema-invalid by spec.
    if params is None and not isinstance(tool_call, dict):
        return -1

    name = tool_name or embedded_name
    if name is None:
        # No tool name and we can't infer it: nothing to validate against.
        # Per the narrow-scope rule, return 0.
        return 0

    schema = schemas.get(name)
    if schema is None:
        # Unknown tool name. The gym would reject this upstream; from a
        # *schema-validity* viewpoint there is no schema to fail against, so
        # we return 0 (consistent with the "valid-but-failed" rule).
        return 0

    # Use the pre-compiled validator for the default registry; fall back to
    # ad-hoc validation for caller-supplied schemas.
    validator = _VALIDATORS.get(name)
    try:
        if validator is not None and schemas is MILO_TOOL_SCHEMAS:
            validator.validate(params)
        else:
            jsonschema.validate(instance=params, schema=schema)
    except jsonschema.ValidationError:
        return -1
    except jsonschema.SchemaError:
        # A broken caller-supplied schema is not a TIR penalty; the call
        # itself wasn't malformed, the schema is. Return 0 and let the gym's
        # config validation catch this elsewhere.
        return 0
    return 0
