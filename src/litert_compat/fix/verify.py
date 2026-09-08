"""Numerical verification: original vs fixed on the LiteRT CPU interpreter.

Reuses the Phase 8 `edge-compat[runners]` extra and its CPU machinery — no
parallel verify dependency. Inputs are seeded deterministically from the
ORIGINAL model's sha256, generated against the FIXED model's input signature
and cast to the original's input dtypes (io_cast may have changed the
interface); original outputs are cast to the fixed model's output dtypes
before comparison. The pass bound is the shared tolerance policy of the sweep
and probe runners: |out - ref| <= max(absolute, relative * |ref|) per element.

Statuses: `passed` / `failed` are measurements; `skipped` means the runners
extra is unavailable; `error` means the interpreter could not execute one of
the models. A measured `failed` is never overridable — `--allow-unverified`
only covers `skipped` / `error`.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

from litert_compat.probe.runners import Tolerance


def availability() -> str | None:
    """None when verification can run; otherwise the human-readable reason."""
    from litert_compat.probe.runners import CpuRunner

    return CpuRunner().availability()


def _block(status: str, tolerance: Tolerance, reason: str | None = None,
           max_abs: float | None = None, max_rel: float | None = None) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "tolerance": {"absolute": tolerance.absolute, "relative": tolerance.relative},
    }


def skipped_block(reason: str, tolerance: Tolerance) -> dict[str, Any]:
    return _block("skipped", tolerance, reason=reason)


def run_verification(
    original_path: Path, original_bytes: bytes, fixed_bytes: bytes, tolerance: Tolerance
) -> dict[str, Any]:
    """The fix report's `verification` block. Assumes `availability()` is None."""
    import numpy as np

    from litert_compat.parser.reader import parse_tflite
    from litert_compat.probe.runners import interpreter_outputs, seeded_feeds_for

    seed_key = hashlib.sha256(original_bytes).hexdigest()
    original_inputs = _io_dtypes(parse_tflite(original_bytes), np)

    with tempfile.TemporaryDirectory(prefix="edge-fix-verify-") as tmp:
        fixed_path = Path(tmp) / "fixed.tflite"
        fixed_path.write_bytes(fixed_bytes)
        try:
            fixed_feeds = seeded_feeds_for(fixed_path, seed_key)
            if len(fixed_feeds) != len(original_inputs):
                return _block(
                    "error", tolerance,
                    reason=f"input count changed: {len(original_inputs)} → {len(fixed_feeds)}",
                )
            original_feeds = [
                (tensor, array.astype(dtype))
                for (tensor, array), dtype in zip(fixed_feeds, original_inputs, strict=True)
            ]
            original_outputs = interpreter_outputs(original_path, original_feeds)
            fixed_outputs = interpreter_outputs(fixed_path, fixed_feeds)
        except Exception as exc:  # interpreter failures are a verification error
            note = f"{type(exc).__name__}: {exc}".splitlines()[0][:200]
            return _block("error", tolerance, reason=f"interpreter failed: {note}")

    if len(original_outputs) != len(fixed_outputs):
        return _block(
            "error", tolerance,
            reason=f"output count changed: {len(original_outputs)} → {len(fixed_outputs)}",
        )

    max_abs = 0.0
    max_rel = 0.0
    within = True
    for ref, out in zip(original_outputs, fixed_outputs, strict=True):
        # The original is the reference; compare in the fixed model's output
        # dtype (io_cast contract), in float64.
        ref64 = np.asarray(ref).astype(np.asarray(out).dtype).astype(np.float64)
        out64 = np.asarray(out, dtype=np.float64)
        if out64.size != ref64.size:
            return _block(
                "error", tolerance,
                reason=f"output size changed: {ref64.size} → {out64.size}",
            )
        out64 = out64.reshape(ref64.shape)
        if not np.all(np.isfinite(out64)):
            return _block("failed", tolerance, reason="non-finite values in fixed output",
                          max_abs=float("inf"), max_rel=float("inf"))
        diff = np.abs(out64 - ref64)
        if diff.size:
            max_abs = max(max_abs, float(diff.max()))
            max_rel = max(max_rel, float((diff / np.maximum(np.abs(ref64), 1e-12)).max()))
        bound = np.maximum(tolerance.absolute, tolerance.relative * np.abs(ref64))
        within = within and bool(np.all(diff <= bound))

    if within:
        return _block("passed", tolerance, max_abs=max_abs, max_rel=max_rel)
    return _block(
        "failed", tolerance,
        reason="numerics outside tolerance vs the original model",
        max_abs=max_abs, max_rel=max_rel,
    )


def _io_dtypes(parsed: Any, np: Any) -> list[Any]:
    subgraph = parsed.subgraphs[0]
    return [np.dtype(subgraph.tensors[i].dtype) for i in subgraph.inputs]
