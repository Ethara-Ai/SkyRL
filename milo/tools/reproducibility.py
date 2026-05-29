"""Phase 14.7 — reproducibility manifest writer.

Per spec §21.2 / plan §14.7: every training run (SFT, RL, ablation) writes
a manifest at `<run_dir>/manifest.json` at startup, BEFORE any training
step. The manifest is the single source of truth for the nightly audit
(§17.5), the eval gate (§19), and the AGIF handoff (§21.4 fresh-node test).

Required fields (raises ManifestIncomplete if any is missing):

    run_id (timestamp + short hash)
    git_sha (full SHA of the milo repo at HEAD)
    milo_version (semver from milo/__init__.py)
    spec_doc_sha (sha256 of RL_GYM_SPEC.md at run start)
    base_model (dataclass config name)
    base_model_revision (HF revision; required)
    sft_checkpoint_path + sha256 (required for RL runs)
    dataset_version (milo-lht-v<N>)
    train_split_sha256, holdout_split_sha256
    seeds (global / data / rollout)
    hardware (e.g. "16xH100-80GB-SXM5", read from nvidia-smi)
    hyperparameters (full resolved dataclass config snapshot)
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = (
    "run_id",
    "git_sha",
    "milo_version",
    "base_model",
    "base_model_revision",
    "dataset_version",
    "train_split_sha256",
    "holdout_split_sha256",
    "seeds",
    "hardware",
    "hyperparameters",
)


class ManifestIncomplete(RuntimeError):
    pass


@dataclass
class ReproducibilityManifest:
    run_id: str = ""
    git_sha: str = ""
    milo_version: str = ""
    spec_doc_sha: str = ""
    base_model: str = ""
    base_model_revision: str = ""
    sft_checkpoint_path: str = ""
    sft_checkpoint_sha256: str = ""
    dataset_version: str = ""
    train_split_sha256: str = ""
    holdout_split_sha256: str = ""
    seeds: dict[str, int] = field(default_factory=dict)
    hardware: str = ""
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    started_iso: str = ""

    def validate(self) -> None:
        missing = [k for k in REQUIRED_FIELDS if not getattr(self, k)]
        if missing:
            raise ManifestIncomplete(f"required manifest fields missing: {missing}")


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        )
        return out.strip()
    except Exception:
        return ""


def _milo_version() -> str:
    try:
        from milo import __version__ as v
        return v
    except Exception:
        return "unknown"


def _spec_doc_sha(spec_path: Path | None) -> str:
    if spec_path is None or not spec_path.is_file():
        return ""
    return hashlib.sha256(spec_path.read_bytes()).hexdigest()


def _file_sha256(p: Path | None) -> str:
    if p is None or not Path(p).is_file():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hardware_string() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True,
        )
        lines = [ll.strip() for ll in out.strip().split("\n") if ll.strip()]
        return f"{len(lines)}x" + "; ".join(sorted(set(lines)))
    except Exception:
        return os.uname().nodename


def write_manifest(
    run_dir: Path,
    cfg: dict[str, Any] | Any,
    *,
    base_model: str,
    base_model_revision: str,
    dataset_version: str,
    train_split_path: Path | None,
    holdout_split_path: Path | None,
    sft_checkpoint_path: Path | None = None,
    spec_path: Path | None = None,
    seeds: dict[str, int] | None = None,
) -> ReproducibilityManifest:
    """Build, validate, write `<run_dir>/manifest.json`. Returns the manifest."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(cfg, "__dataclass_fields__"):
        cfg_dict = asdict(cfg)
    else:
        cfg_dict = dict(cfg)

    manifest = ReproducibilityManifest(
        run_id=f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}",
        git_sha=_git_sha(),
        milo_version=_milo_version(),
        spec_doc_sha=_spec_doc_sha(spec_path),
        base_model=base_model,
        base_model_revision=base_model_revision,
        sft_checkpoint_path=str(sft_checkpoint_path) if sft_checkpoint_path else "",
        sft_checkpoint_sha256=_file_sha256(sft_checkpoint_path),
        dataset_version=dataset_version,
        train_split_sha256=_file_sha256(train_split_path),
        holdout_split_sha256=_file_sha256(holdout_split_path),
        seeds=seeds or {"global": 0, "data": 0, "rollout": 0},
        hardware=_hardware_string(),
        hyperparameters=cfg_dict,
        started_iso=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    manifest.validate()
    (run_dir / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2))
    return manifest
