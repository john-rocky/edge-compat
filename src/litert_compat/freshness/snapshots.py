"""Versioned sweep snapshot discovery (`data/sweep/<version>/<date>/`).

A snapshot is one full sweep run: a directory of sweep_result JSON files
(one per model) under `<litertjs_version>/<YYYY-MM-DD>/`. Snapshots are
append-only — history is what makes deltas computable — and ordered by
(version, date), so "the two most recent snapshots" is well defined without
touching file mtimes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from litert_compat.freshness.releases import parse_version
from litert_compat.matrix.canonical import load_json

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class SnapshotError(ValueError):
    """A sweep snapshot directory or result file is malformed."""


@dataclass(frozen=True)
class SweepSnapshot:
    """One dated sweep of the catalog against one @litertjs/core version."""

    litertjs_version: str
    date: str
    path: Path
    #: model_id -> parsed sweep_result document, sorted by model_id.
    models: dict[str, dict[str, Any]] = field(compare=False)

    def sort_key(self) -> tuple[tuple[int, ...], str, str]:
        return (parse_version(self.litertjs_version) or (), self.litertjs_version, self.date)


def _load_snapshot_dir(version: str, date: str, path: Path) -> SweepSnapshot:
    models: dict[str, dict[str, Any]] = {}
    for file in sorted(path.glob("*.json")):
        try:
            doc = load_json(file)
        except (OSError, ValueError) as exc:
            raise SnapshotError(f"{file}: not valid JSON: {exc}") from exc
        model_id = doc.get("model_id") if isinstance(doc, dict) else None
        if not isinstance(model_id, str) or not model_id:
            raise SnapshotError(f"{file}: not a sweep result (missing model_id)")
        if file.stem != model_id:
            raise SnapshotError(
                f"{file}: file name must be <model_id>.json (model_id is {model_id!r})"
            )
        if model_id in models:
            raise SnapshotError(f"{path}: duplicate model_id {model_id!r}")
        models[model_id] = doc
    return SweepSnapshot(litertjs_version=version, date=date, path=path, models=models)


def discover_snapshots(sweep_root: Path) -> list[SweepSnapshot]:
    """All snapshots under `sweep_root`, ordered oldest -> newest by
    (numeric version, version string, date). Non-conforming directories are
    an error, not silently skipped — a typo'd snapshot dir must not make a
    delta quietly compare the wrong pair."""
    if not sweep_root.is_dir():
        return []
    snapshots: list[SweepSnapshot] = []
    for version_dir in sorted(p for p in sweep_root.iterdir() if p.is_dir()):
        date_dirs = sorted(p for p in version_dir.iterdir() if p.is_dir())
        if not date_dirs:
            raise SnapshotError(
                f"{version_dir}: snapshot version directory has no <YYYY-MM-DD> subdirectory"
            )
        for date_dir in date_dirs:
            if not _DATE_RE.match(date_dir.name):
                raise SnapshotError(
                    f"{date_dir}: snapshot date directory must be named YYYY-MM-DD"
                )
            snapshots.append(_load_snapshot_dir(version_dir.name, date_dir.name, date_dir))
    return sorted(snapshots, key=SweepSnapshot.sort_key)
