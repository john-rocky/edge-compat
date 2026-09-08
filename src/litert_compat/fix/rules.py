"""Transform rule files: load, validate, scaffold.

Rules are owner-authored JSON files (`<id>.json`) in a rules directory,
validating against `schemas/transform_rule.schema.json` plus the semantic
checks here. Loading is strict: an invalid rule file is a data defect that
fails the whole run (like an invalid matrix snapshot) — with one deliberate
exception: `op_sequence` rules are schema-valid but not executable by the v1
engine, so they load and are REFUSED at planning time with a precise reason
instead of bricking the rules directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from litert_compat.cards.schema_io import schema_errors
from litert_compat.fix.options_registry import UnknownOptionsOpError, default_options_for
from litert_compat.matrix.canonical import canonical_dumps, load_json
from litert_compat.parser.opcodes import BUILTIN_NAME_TO_CODE

_REQUIRED_MEASURED_EVIDENCE = ("source_model", "litert_version", "date")


class RuleError(ValueError):
    """One or more rule files are invalid. `errors` lists every finding."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("invalid transform rule(s):\n" + "\n".join(f"  {e}" for e in errors))


@dataclass(frozen=True)
class TransformRule:
    id: str
    title: str
    action: str
    match: dict[str, Any]
    params: dict[str, Any]
    expected_effect: str
    provenance: str
    evidence: dict[str, Any] | None
    notes: str | None
    fixture: dict[str, str]
    source: Path  # the rule file; fixture paths resolve relative to its parent
    #: Re-verification failure flag (schema 1.1), written by `rules reverify`.
    stale: dict[str, Any] | None = None

    @property
    def uses_op_sequence(self) -> bool:
        return "op_sequence" in self.match

    def fixture_model_path(self) -> Path:
        return (self.source.parent / self.fixture["model"]).resolve()

    def fixture_matrix_path(self) -> Path:
        return (self.source.parent / self.fixture["matrix"]).resolve()


def _emittable(op: str) -> str | None:
    """None when a rule may emit `op`; otherwise the precise reason it may not."""
    if op not in BUILTIN_NAME_TO_CODE:
        return f"unknown builtin operator name {op!r}"
    try:
        default_options_for(op)
    except UnknownOptionsOpError as exc:
        return str(exc)
    return None


def _semantic_errors(doc: dict[str, Any], path: Path) -> list[str]:
    errors: list[str] = []
    rule_id = doc.get("id", "")
    if path.stem != rule_id:
        errors.append(f"{path.name}: file must be named <id>.json (id is {rule_id!r})")
    if doc.get("provenance") == "measured":
        evidence = doc.get("evidence") or {}
        missing = [k for k in _REQUIRED_MEASURED_EVIDENCE if not evidence.get(k)]
        if missing:
            errors.append(
                f"{path.name}: measured rule missing evidence field(s): {', '.join(missing)}"
            )

    action = doc.get("action")
    params = doc.get("params", {})
    if action == "replace_op":
        reason = _emittable(params.get("new_op", ""))
        if reason:
            errors.append(f"{path.name}: params.new_op: {reason}")
    elif action == "decompose":
        errors.extend(f"{path.name}: {e}" for e in _decompose_errors(params))
    return errors


def _decompose_errors(params: dict[str, Any]) -> list[str]:
    """Static recipe checks. Bounds of in:K / out:K depend on the matched
    node's arity and are checked at planning time; everything checkable
    without a node is checked here."""
    errors: list[str] = []
    declared_tmp = set()
    for i, spec in enumerate(params.get("intermediates") or []):
        name = spec["name"]
        if name in declared_tmp:
            errors.append(f"intermediates[{i}]: duplicate name {name!r}")
        declared_tmp.add(name)

    produced_tmp: set[str] = set()
    produced_out: list[int] = []
    consumed_tmp: set[str] = set()
    for i, node in enumerate(params.get("nodes", [])):
        reason = _emittable(node.get("op", ""))
        if reason:
            errors.append(f"nodes[{i}].op: {reason}")
        for ref in node.get("inputs", []):
            if ref.startswith("tmp:"):
                name = ref.removeprefix("tmp:")
                if name not in declared_tmp:
                    errors.append(f"nodes[{i}]: input {ref} is not a declared intermediate")
                elif name not in produced_tmp:
                    errors.append(f"nodes[{i}]: input {ref} is consumed before it is produced")
                consumed_tmp.add(name)
            elif ref.startswith("out:"):
                errors.append(f"nodes[{i}]: input {ref} — out:K refs are write-only")
        for ref in node.get("outputs", []):
            if ref.startswith("in:"):
                errors.append(f"nodes[{i}]: output {ref} — in:K refs are read-only")
            elif ref.startswith("tmp:"):
                name = ref.removeprefix("tmp:")
                if name not in declared_tmp:
                    errors.append(f"nodes[{i}]: output {ref} is not a declared intermediate")
                elif name in produced_tmp:
                    errors.append(f"nodes[{i}]: output {ref} is produced twice")
                produced_tmp.add(name)
            else:
                produced_out.append(int(ref.removeprefix("out:")))

    for k in sorted(set(produced_out)):
        if produced_out.count(k) > 1:
            errors.append(f"recipe writes out:{k} more than once")
    if produced_out and sorted(set(produced_out)) != list(range(max(produced_out) + 1)):
        errors.append(
            "recipe output refs must cover out:0..out:N contiguously "
            f"(got {sorted(set(produced_out))})"
        )
    if not produced_out:
        errors.append("recipe produces no out:K tensor — the matched node's outputs "
                      "would be left unwritten")
    for name in sorted(declared_tmp - produced_tmp - consumed_tmp):
        errors.append(f"intermediate {name!r} is declared but never used")
    return errors


def load_rule_file(path: Path) -> TransformRule:
    """Load and validate one rule file; raises RuleError."""
    try:
        doc = load_json(path)
    except ValueError as exc:
        raise RuleError([f"{path.name}: not valid JSON: {exc}"]) from exc
    errors = [f"{path.name}: {e}" for e in schema_errors(doc, "transform_rule.schema.json")]
    if not errors:
        errors = _semantic_errors(doc, path)
    if errors:
        raise RuleError(errors)
    return TransformRule(
        id=doc["id"],
        title=doc["title"],
        action=doc["action"],
        match=doc["match"],
        params=doc["params"],
        expected_effect=doc["expected_effect"],
        provenance=doc["provenance"],
        evidence=doc.get("evidence"),
        notes=doc.get("notes"),
        fixture=doc["fixture"],
        source=path,
        stale=doc.get("stale"),
    )


def load_rules(rules_dir: Path) -> list[TransformRule]:
    """Load every `*.json` rule in a directory, sorted by id. Strict: any
    invalid file fails the load; duplicate ids are a defect."""
    if not rules_dir.is_dir():
        raise RuleError([f"rules directory not found: {rules_dir}"])
    rules: list[TransformRule] = []
    errors: list[str] = []
    for path in sorted(rules_dir.glob("*.json")):
        try:
            rules.append(load_rule_file(path))
        except RuleError as exc:
            errors.extend(exc.errors)
    seen: dict[str, Path] = {}
    for rule in rules:
        if rule.id in seen:
            errors.append(
                f"duplicate rule id {rule.id!r} ({seen[rule.id].name} and {rule.source.name})"
            )
        seen.setdefault(rule.id, rule.source)
    if errors:
        raise RuleError(errors)
    return sorted(rules, key=lambda r: r.id)


def set_stale(path: Path, stale: dict[str, Any]) -> bool:
    """Flag a rule file `stale` in place (schema 1.1) after a measured
    re-verification failure. The rule is never deleted or otherwise edited.
    Returns True when the file's bytes changed."""
    doc = load_json(path)
    updated = dict(doc)
    updated["schema_version"] = "1.1"
    updated["stale"] = stale
    return _rewrite_if_changed(path, updated)


def clear_stale(path: Path) -> bool:
    """Remove a rule file's `stale` flag after a passing re-verification —
    the same upgrade-on-re-measurement pattern as inferred -> measured."""
    doc = load_json(path)
    if "stale" not in doc:
        return False
    updated = {k: v for k, v in doc.items() if k != "stale"}
    return _rewrite_if_changed(path, updated)


def _rewrite_if_changed(path: Path, doc: dict[str, Any]) -> bool:
    text = canonical_dumps(doc)
    if path.read_text(encoding="utf-8") == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True


_SCAFFOLD_PARAMS: dict[str, dict[str, Any]] = {
    "replace_op": {"new_op": "TODO_NEW_OP"},
    "decompose": {
        "nodes": [{"op": "TODO_OP", "inputs": ["in:0"], "outputs": ["out:0"]}],
    },
    "insert_cast": {"to": "float32"},
    "io_cast": {"to": "int32", "io": "both"},
}


def scaffold_rule(rule_id: str, action: str, op: str | None, fixture_model: str,
                  fixture_matrix: str, generated_at: str) -> dict[str, Any]:
    """A template rule document for `edge-fix rules new`. TODO markers show
    what the author must fill in; `rules validate` will name each one left."""
    match: dict[str, Any] = {"dtypes": ["float32"]}
    if action != "io_cast":
        match = {"op": op or "TODO_OP", "dtypes": ["float32"]}
    return {
        "schema_version": "1.0",
        "id": rule_id,
        "title": "TODO: one-line description of the rewrite",
        "match": match,
        "action": action,
        "params": _SCAFFOLD_PARAMS[action],
        "expected_effect": "TODO: what the re-lint improvement gate should observe",
        "notes": "TODO: conditions under which this rewrite preserves semantics",
        "provenance": "example",
        "evidence": {
            "source_model": "TODO",
            "litert_version": "TODO",
            "date": generated_at,
        },
        "fixture": {"model": fixture_model, "matrix": fixture_matrix},
    }
