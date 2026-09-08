"""Latest-known-release registry (`data/releases.json`).

The registry records the newest UPSTREAM release of each runtime that this
repo knows about — maintained by `autobump.yml` (CI) or the owner, never by
measurement tools. It is what staleness is measured against: a matrix
snapshot, sweep record, or site row whose verified-against version lags the
registry's version is flagged stale (decay is shown, never hidden), while
verdicts and data stay unchanged.

No network access at runtime: consumers only read the committed file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from litert_compat.matrix.canonical import load_json

#: Browser backend IDs registered in matrix.schema.json 1.1. For these the
#: version axis is the @litertjs/core version; everything else is native
#: LiteRT (ai-edge-litert).
WEB_BACKENDS = frozenset({"wasm_xnnpack", "webgpu_mldrift", "webnn"})

#: Registry key per version axis.
LITERT_KEY = "litert"
LITERTJS_KEY = "litertjs_core"
LITERTLM_KEY = "litertlm"

#: Device-run env.runtime value -> registry axis (Phase 13). The .tflite-on-NPU
#: lane runs on native LiteRT; the .litertlm lane on LiteRT-LM.
RUNTIME_AXES = {"litert": LITERT_KEY, "litert-lm": LITERTLM_KEY}


class ReleasesError(ValueError):
    """The releases registry file is unreadable or malformed."""


@dataclass(frozen=True)
class KnownRelease:
    version: str
    checked_at: str
    source: str


def parse_version(version: str) -> tuple[int, ...] | None:
    """Numeric-prefix version tuple: '2.1.6' -> (2, 1, 6); '0.1.0-rc1' ->
    (0, 1, 0). None when no leading numeric component exists — comparisons
    against unparsable versions are skipped rather than guessed."""
    parts: list[int] = []
    for component in version.split("."):
        match = re.match(r"(\d+)", component)
        if match is None:
            break
        parts.append(int(match.group(1)))
        if match.group(1) != component:
            break
    return tuple(parts) if parts else None


def lags_behind(version: str, latest: str) -> bool:
    """True when `version` is strictly older than `latest` by numeric-prefix
    comparison. False on ties and on unparsable versions (no warning is
    better than a wrong one)."""
    a, b = parse_version(version), parse_version(latest)
    if a is None or b is None:
        return False
    width = max(len(a), len(b))
    return a + (0,) * (width - len(a)) < b + (0,) * (width - len(b))


def load_releases(path: Path) -> dict[str, KnownRelease]:
    """Load the registry: axis key -> KnownRelease. Strict: a malformed
    registry raises rather than silencing every staleness check."""
    try:
        doc = load_json(path)
    except (OSError, ValueError) as exc:
        raise ReleasesError(f"{path}: cannot read releases registry: {exc}") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("releases"), dict):
        raise ReleasesError(f"{path}: expected an object with a 'releases' object")
    releases: dict[str, KnownRelease] = {}
    for key, value in doc["releases"].items():
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("version"), str)
            or not isinstance(value.get("checked_at"), str)
        ):
            raise ReleasesError(
                f"{path}: releases[{key!r}] must carry string 'version' and 'checked_at'"
            )
        releases[key] = KnownRelease(
            version=value["version"],
            checked_at=value["checked_at"],
            source=str(value.get("source", "")),
        )
    return releases


def axis_for_backend(backend: str) -> str:
    return LITERTJS_KEY if backend in WEB_BACKENDS else LITERT_KEY


def axis_for_runtime(runtime: str) -> str:
    """Registry axis for a device-run record's env.runtime value."""
    return RUNTIME_AXES.get(runtime, LITERT_KEY)


def staleness_warning(
    backend: str, litert_version: str, releases: dict[str, KnownRelease]
) -> str | None:
    """The soft-warning text for a matrix snapshot that lags the latest known
    release of its version axis; None when current, unknown, or unparsable.
    Warning only — verdicts are never changed by staleness (spec 11.3).

    Only strictly numeric snapshot versions can lag: placeholder versions
    ('0.0.0-example') make no claim about a real runtime, so comparing them
    against real releases would warn about nothing."""
    if re.fullmatch(r"\d+(\.\d+)*", litert_version) is None:
        return None
    axis = axis_for_backend(backend)
    latest = releases.get(axis)
    if latest is None or not lags_behind(litert_version, latest.version):
        return None
    runtime = "@litertjs/core" if axis == LITERTJS_KEY else "litert"
    return (
        f"matrix snapshot is verified against {runtime} {litert_version}, but the "
        f"latest known release is {latest.version} (checked {latest.checked_at}). "
        f"Verdicts are unchanged; re-measure via `edge-compat release-check` "
        f"(native) or a re-sweep (web) to refresh."
    )


def load_releases_if_configured(
    releases_path: Path | None, default: Path = Path("data/releases.json")
) -> dict[str, KnownRelease]:
    """Resolve the registry for CLI use: an explicit path must load; the
    default path is used only when it exists (repos without a registry get no
    staleness checks, not errors)."""
    if releases_path is not None:
        return load_releases(releases_path)
    if default.is_file():
        return load_releases(default)
    return {}
