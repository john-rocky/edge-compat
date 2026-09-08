"""edge-fix: diagnosis → repair (Phase 10).

The matrix's `rewrite_hints` are human prose; this package promotes them to
executable, deterministic graph surgery. Pipeline: lint → match owner-authored
transform rules to findings → plan (ordered, conflict-checked) → apply on a
copy → re-lint (the metric must improve, else automatic rollback) → numerical
verification on the LiteRT CPU interpreter (the Phase 8 `runners` extra).

Data Integrity extension (spec §10.1): the agent builds the ENGINE; the owner
authors the RULES. Example-provenance rules exist only to exercise the engine
on synthetic fixtures.
"""

from litert_compat.fix.engine import FixResult, run_fix
from litert_compat.fix.loader import UnsupportedModelError, load_graph_spec
from litert_compat.fix.rules import RuleError, TransformRule, load_rules

__all__ = [
    "FixResult",
    "RuleError",
    "TransformRule",
    "UnsupportedModelError",
    "load_graph_spec",
    "load_rules",
    "run_fix",
]
