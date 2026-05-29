"""Phase 18.5 — JSON-backed model registry.

Per `IMPLEMENTATION_PLAN.md` v0.4 §18.5: tracks `(name, path, manifest)`
for every registered checkpoint. JSON file lives at
`${MILO_REGISTRY_PATH:-milo/data/registry.json}`. Atomic write via tmp+rename.

`register()` is intentionally idempotent: registering the same name twice
with the same path is a no-op; with a different path raises unless
`override=True`. This pattern matches `milo/adapters/registry.py`.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_REGISTRY_PATH_ENV = "MILO_REGISTRY_PATH"
DEFAULT_REGISTRY_PATH = "milo/data/registry.json"


@dataclass
class RegisteredModel:
    name: str
    path: str
    manifest: dict[str, Any] = field(default_factory=dict)
    registered_iso: str = ""


class ModelRegistry:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(
            path
            if path is not None
            else os.environ.get(DEFAULT_REGISTRY_PATH_ENV, DEFAULT_REGISTRY_PATH)
        )
        self._models: dict[str, RegisteredModel] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        obj = json.loads(self.path.read_text() or "{}")
        for name, fields in obj.get("models", {}).items():
            self._models[name] = RegisteredModel(**fields)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", delete=False, dir=str(self.path.parent), prefix=".registry-", suffix=".tmp"
        ) as f:
            json.dump(
                {"models": {n: asdict(m) for n, m in self._models.items()}},
                f, indent=2,
            )
            tmp = Path(f.name)
        os.replace(tmp, self.path)

    # ------------------------------------------------------------------ API

    def register(
        self,
        name: str,
        path: str | Path,
        manifest: dict[str, Any] | None = None,
        override: bool = False,
    ) -> RegisteredModel:
        from datetime import datetime, timezone

        path = str(path)
        if name in self._models and not override:
            existing = self._models[name]
            if existing.path != path:
                raise ValueError(
                    f"model {name!r} already registered at {existing.path}; "
                    f"pass override=True to replace"
                )
            return existing
        model = RegisteredModel(
            name=name,
            path=path,
            manifest=manifest or {},
            registered_iso=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self._models[name] = model
        self._save()
        return model

    def get(self, name: str) -> RegisteredModel:
        if name not in self._models:
            raise KeyError(name)
        return self._models[name]

    def list_names(self) -> list[str]:
        return sorted(self._models)

    def unregister(self, name: str) -> None:
        self._models.pop(name, None)
        self._save()
