# Matrix commands

## Install

Python ≥ 3.11.

```sh
uv sync            # or: pip install -e .
uv run edge-compat --help
```

The Phase 8 backend runners (real probe measurement) and the `edge-fix`
numerical verification need the `runners` extra:

```sh
uv sync --extra runners    # or: pip install -e '.[runners]'
```

Without it every tool still works; runners just report "unavailable",
`release-check` keeps the affected entries honestly `inferred`, and
`edge-fix --apply` refuses unless `--allow-unverified` is passed.

## CLI

### `edge-compat matrix validate <file>`

JSON Schema validation (against `schemas/matrix.schema.json`) plus semantic lint:
contradictory duplicate entries for the same (op, dtypes, constraints) signature, missing
evidence on `measured` entries. Prints per-provenance entry counts.
Exit codes: `0` pass · `1` findings · `2` usage error.

### `edge-compat matrix import --csv <file> -o <matrix.json> --litert-version X --backend B [--device … --soc … --driver … --generated-at YYYY-MM-DD]`

Imports a curated CSV table ([contract](matrix-api.md#csv-column-contract)). Forgiving on input — whitespace trimmed, any
column order, unknown columns warned and ignored. Strict on output — the result must pass
`matrix validate` or nothing is written. Deterministic: same CSV + same options →
byte-identical JSON (`--generated-at` defaults to today; pass it explicitly for
reproducible output).
Exit codes: `0` ok · `1` CSV data errors or invalid result · `2` usage error.

⚠ **Import REPLACES the output file — it does not merge.** Writing `-o` onto an existing
snapshot silently drops every entry not in the CSV (cost a recovery on 2026-08-14: a
1-row CSV clobbered the 3-row `gpu_metal_mac__0.16.0.json`; restored from git, merged by
hand, re-validated). To add rows to an existing snapshot: import to a temp file, append
the new entries into the existing JSON, then `matrix validate` — or keep the staging CSV
cumulative. Field vocab that trips first-time rows: `status` ∈ crash/delegated/fallback/
incorrect/partial/unknown (no "conditional"), dtypes are schema names (`float32`, not
"fp32"), `constraints` must parse as key=value.

### `edge-compat matrix diff <old> <new> [--json]`

Deterministic semantic change summary between two snapshots of the **same backend**:
entries added/removed, ops added/removed, `status` transitions, `provenance` transitions.
Free-text edits (conditions, rewrite hints) do not count as semantic change.
Exit codes: `0` no semantic difference · `1` differences found · `2` usage error
(including invalid input files and backend mismatch) — so a release watcher can trigger
on exit `1` knowing it always means real change.

### `edge-compat matrix carry-forward <file> --to-litert-version X.Y.Z -o <new> [--generated-at YYYY-MM-DD]`

Produces the next-version snapshot; refuses to write in place. Every `measured` entry is
downgraded to `provenance: "inferred"` with `evidence.derived_from` recording the source
snapshot's `litert_version` and `generated_at`. Re-measurement later upgrades entries back
to `measured`.
Exit codes: `0` ok · `2` usage error.

### `edge-compat probe gen [--from-lint report.json]… [--from-matrix snapshot.json]… [-o data/probes] [--json]`

Builds minimal single-op `.tflite` probe fixtures from complete lookup signatures —
`needs_probe` records of `edge-lint --json` reports, and/or the `inferred` entries of a
matrix snapshot. Deterministic: signature → stable fixture id
(`<OP>__<dtypes>__<meta-hash>`) and byte-identical file. Signatures the template registry
cannot realize are listed as `unprobeable` with a reason — never approximated (see
[Release pipeline](release-pipeline.md)).
Exit codes: `0` all generated · `1` unprobeable signatures present · `2` usage error.

### `edge-compat release-check <snapshot.json>… --to-litert-version X.Y.Z -o <out-dir> [options]`

The one command per runtime release: carry-forward → probe → re-measure → upgrade → diff
→ report. See [Release pipeline](release-pipeline.md).
Exit codes: `0` nothing changed · `1` semantic changes found · `2` error.

### `edge-compat release-gate [--repo-root .] [--scope data/release_scope.json] [--site-dist site/dist]`

The pre-publication gate (Phase 12): run by the release runbook and
the `deploy-zoo` CI job before anything goes public. Deterministic; three guards:

1. **Benchmark scope** — no card presents a non-example `cross_runtime` comparison for a
   model outside the approved public scope in `data/release_scope.json` (owner-editable
   data; initially Gemma 4 only per the launch plan — widening it is a data change, not a
   code change).
2. **Disclosure** — the disclosure line (`src/litert_compat/branding.py`, mirrored in
   `site/src/shared/branding.ts` with test-enforced parity) renders in `README.md`,
   `cards/README.md`, `llms.txt`, and — with `--site-dist` — every generated HTML page.
3. **Example banners** — cards carrying `example`-provenance records must render their
   example banner; the site index must show its fixtures banner when example records are
   displayed.

Exit codes: `0` pass · `1` findings · `2` usage error.
