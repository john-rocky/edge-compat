"""meta.yaml — the human-authored card fields (model identity, conversion, pitfalls).

Forgiving on input (unknown keys warned and ignored), strict on the fields the
card schema requires. Pitfalls are surfaced verbatim — never generated.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

_MODEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_MODEL_KEYS = ("id", "family", "task", "source_url", "license")
_CONVERSION_KEYS = ("tool", "tool_version", "command", "quantization")
_TOP_KEYS = ("model", "conversion", "pitfalls")


class MetaError(ValueError):
    """meta.yaml is unusable; `errors` lists every problem found."""

    def __init__(self, path: Path, errors: list[str]) -> None:
        self.errors = errors
        lines = "\n".join(f"  {e}" for e in errors)
        super().__init__(f"invalid meta file {path}:\n{lines}")


def _require_str_fields(
    section: str, data: dict[str, Any], keys: tuple[str, ...], errors: list[str]
) -> None:
    for key in keys:
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{section}.{key}: required non-empty string")


def load_meta(path: Path) -> tuple[dict[str, Any], list[str]]:
    """Parse and validate meta.yaml -> (meta, warnings). Raises MetaError on defects."""
    errors: list[str] = []
    warnings: list[str] = []
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise MetaError(path, [f"not valid YAML: {exc}"]) from exc
    if not isinstance(raw, dict):
        raise MetaError(path, ["top level must be a mapping"])

    for key in sorted(set(raw) - set(_TOP_KEYS)):
        warnings.append(f"unknown key {key!r} ignored")

    model = raw.get("model")
    if not isinstance(model, dict):
        errors.append("model: required mapping")
        model = {}
    _require_str_fields("model", model, _MODEL_KEYS, errors)
    model_id = model.get("id")
    if isinstance(model_id, str) and not _MODEL_ID_RE.match(model_id):
        errors.append(f"model.id: {model_id!r} must match {_MODEL_ID_RE.pattern}")
    for key in sorted(set(model) - set(_MODEL_KEYS)):
        warnings.append(f"unknown key model.{key!r} ignored")

    conversion = raw.get("conversion")
    if not isinstance(conversion, dict):
        errors.append("conversion: required mapping")
        conversion = {}
    _require_str_fields("conversion", conversion, _CONVERSION_KEYS, errors)
    for key in sorted(set(conversion) - set(_CONVERSION_KEYS)):
        warnings.append(f"unknown key conversion.{key!r} ignored")

    pitfalls = raw.get("pitfalls", [])
    if not isinstance(pitfalls, list) or any(
        not isinstance(p, str) or not p.strip() for p in pitfalls
    ):
        errors.append("pitfalls: must be a list of non-empty strings")
        pitfalls = []

    if errors:
        raise MetaError(path, errors)
    return (
        {
            "model": {k: model[k].strip() for k in _MODEL_KEYS},
            "conversion": {k: conversion[k].strip() for k in _CONVERSION_KEYS},
            "pitfalls": [p.strip() for p in pitfalls],
        },
        warnings,
    )
