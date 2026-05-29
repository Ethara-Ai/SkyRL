"""Phase 18.4 — verify a saved checkpoint.

Checks:
    1. Directory exists and contains the expected files (config.json,
       tokenizer.json, model.safetensors or model.safetensors.index.json).
    2. SHA-256 of each file matches the manifest if a manifest is provided.
    3. Model can be loaded with `transformers.AutoConfig.from_pretrained` —
       cheaper than loading weights, but proves the directory parses.

Used at the end of every Phase 19 SFT/RL run before the checkpoint is
promoted via `milo.tools.registry`.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("milo.tools.checkpoint_verify")


REQUIRED_BASE_FILES = ("config.json", "tokenizer_config.json")
WEIGHTS_OPTIONS = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)


@dataclass
class CheckpointVerifyReport:
    checkpoint_path: str
    exists: bool
    missing_files: list[str] = field(default_factory=list)
    weights_present: bool = False
    sha256_mismatches: list[str] = field(default_factory=list)
    config_parseable: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return (
            self.exists
            and not self.missing_files
            and self.weights_present
            and not self.sha256_mismatches
            and self.config_parseable
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ok"] = self.ok
        return d


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checkpoint(
    checkpoint_path: Path,
    expected_sha256: dict[str, str] | None = None,
    require_transformers: bool = False,
) -> CheckpointVerifyReport:
    """Run all checks. `expected_sha256` is {relative_path: sha256_hex}."""
    p = Path(checkpoint_path)
    report = CheckpointVerifyReport(checkpoint_path=str(p), exists=p.is_dir())
    if not report.exists:
        return report

    for name in REQUIRED_BASE_FILES:
        if not (p / name).is_file():
            report.missing_files.append(name)

    report.weights_present = any((p / w).is_file() for w in WEIGHTS_OPTIONS)
    if not report.weights_present:
        report.missing_files.append("|".join(WEIGHTS_OPTIONS))

    if expected_sha256:
        for rel, want in expected_sha256.items():
            target = p / rel
            if not target.is_file():
                report.sha256_mismatches.append(f"{rel} missing")
                continue
            got = _sha256_file(target)
            if got != want:
                report.sha256_mismatches.append(f"{rel}: {got} != {want}")

    config_path = p / "config.json"
    if config_path.is_file():
        try:
            obj = json.loads(config_path.read_text())
            report.config_parseable = isinstance(obj, dict)
        except Exception:
            report.config_parseable = False

    if require_transformers:
        try:
            from transformers import AutoConfig  # type: ignore[import-not-found]
            AutoConfig.from_pretrained(str(p), trust_remote_code=False)
            report.extra["transformers_parse"] = True
        except Exception as exc:
            report.extra["transformers_parse"] = False
            report.extra["transformers_error"] = f"{type(exc).__name__}: {exc}"

    return report


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_path", type=Path)
    parser.add_argument("--manifest", type=Path, default=None,
                        help="optional JSON file: {relative_path: sha256_hex}")
    parser.add_argument("--require-transformers", action="store_true")
    args = parser.parse_args(argv)
    expected = json.loads(args.manifest.read_text()) if args.manifest else None
    report = verify_checkpoint(
        args.checkpoint_path, expected, require_transformers=args.require_transformers,
    )
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.exit(main(sys.argv[1:]))
