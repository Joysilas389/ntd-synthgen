"""Discovers disease modules at runtime.

A module is any subdirectory under `/modules` that contains:

    schema.json       feature definitions + categorical hints
    config.py         (optional) constraints + post-processing hooks
    description.md    human description for the UI

This pattern means new diseases can be added by dropping a folder.
"""
from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .logging_config import get_logger

log = get_logger(__name__)


@dataclass
class DiseaseModule:
    name: str               # e.g. "water_borne"
    title: str              # human label, e.g. "Water-borne NTDs"
    description: str        # markdown description
    schema: dict[str, Any]  # full schema.json contents
    constraints: Callable[[Any], Any] | None = None  # optional post-processing

    @property
    def feature_columns(self) -> list[str]:
        return [f["name"] for f in self.schema.get("features", [])]

    @property
    def categorical_columns(self) -> list[str]:
        return [
            f["name"] for f in self.schema.get("features", [])
            if f.get("type") == "categorical"
        ]

    @property
    def target_column(self) -> str | None:
        return self.schema.get("target")


class ModuleRegistry:
    def __init__(self, modules_dir: Path | str) -> None:
        self.modules_dir = Path(modules_dir)
        self._cache: dict[str, DiseaseModule] | None = None

    def discover(self) -> dict[str, DiseaseModule]:
        if self._cache is not None:
            return self._cache

        modules: dict[str, DiseaseModule] = {}
        if not self.modules_dir.exists():
            log.warning("Modules dir %s does not exist", self.modules_dir)
            self._cache = modules
            return modules

        for entry in sorted(self.modules_dir.iterdir()):
            if not entry.is_dir():
                continue
            schema_path = entry / "schema.json"
            if not schema_path.exists():
                continue
            try:
                with open(schema_path) as f:
                    schema = json.load(f)
            except Exception as exc:
                log.warning("Skipping %s: bad schema (%s)", entry.name, exc)
                continue

            desc_path = entry / "description.md"
            description = desc_path.read_text() if desc_path.exists() else ""

            constraints: Callable | None = None
            cfg_path = entry / "config.py"
            if cfg_path.exists():
                try:
                    spec = importlib.util.spec_from_file_location(
                        f"modules.{entry.name}.config", cfg_path,
                    )
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                        constraints = getattr(mod, "apply_constraints", None)
                except Exception as exc:  # pragma: no cover
                    log.warning("Could not import config for %s: %s", entry.name, exc)

            modules[entry.name] = DiseaseModule(
                name=entry.name,
                title=schema.get("title", entry.name.replace("_", " ").title()),
                description=description,
                schema=schema,
                constraints=constraints,
            )
            log.info("Registered module: %s", entry.name)

        self._cache = modules
        return modules

    def get(self, name: str) -> DiseaseModule:
        modules = self.discover()
        if name not in modules:
            raise KeyError(
                f"Unknown module '{name}'. Available: {sorted(modules)}"
            )
        return modules[name]

    def list_summaries(self) -> list[dict[str, Any]]:
        return [
            {
                "name": m.name,
                "title": m.title,
                "description": m.description,
                "features": m.feature_columns,
                "target": m.target_column,
            }
            for m in self.discover().values()
        ]
