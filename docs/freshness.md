# Freshness system (Phase 11)

The base rules keep staleness *visible* (every record dated, versioned,
env-labeled) and Phase 8 automates native re-measurement. The freshness
system keeps the **browser lane** current where automation honestly can, and
turns what it cannot automate into scheduled reminders instead of silent
decay. Hard rule, enforced by test: freshness automation performs **zero
writes under `data/matrix/`** — matrix truth is written only by Phase 8's
`release-check` probe pipeline (native) and, for web backends, by the
still-deferred browser probe runner. The Phase 6 trap rule is never
overridden.

### `data/releases.json` — the latest-known-release registry

The reference point for every staleness check: the newest *known* upstream
release per version axis (`litert` = ai-edge-litert on PyPI, `litertjs_core`
= @litertjs/core on npm). Maintained by `autobump.yml` or the owner — never
by measurement tools. It changes no verdict; it only decides whether a
verified-against version is flagged stale.

### Staleness surfacing (spec 11.3)

- `edge-lint` emits a **soft warning on stderr** when the matrix snapshot's
  `litert_version` lags the registry (`--releases <file>`; defaults to
  `data/releases.json` when present). Warning only — report bytes, verdicts,
  and exit codes are unchanged, proven by test. Placeholder versions
  (`0.0.0-example`) never warn.
- The demo-zoo index, demo pages, and the playground's op cross-reference all
  display *verified-against version + date*, with a visible `stale` flag when
  that version lags the registry (site generator `--releases` flag, same
  default). Display only; the flag never changes a status or measurement.

### `edge-compat freshness` commands

```sh
# Diff the two most recent sweep snapshots under data/sweep/<version>/<date>/
# — and, per runtime, the two most recent device-run snapshots under
# data/device_runs/ (Phase 13) — and upsert the generated entries into
# DELTA_LOG.md (never hand-edited).
# Exit: 0 no observed changes · 1 changes recorded · 2 usage error.
edge-compat freshness delta [--sweep-dir data/sweep] [--log DELTA_LOG.md] \
  [--latency-threshold-pct 25] [--rules-dir data/transforms] [--json]

# List manual-verify register items whose re-verify interval has lapsed.
# Exit: 0 nothing overdue · 1 overdue items · 2 usage error.
edge-compat freshness overdue [--register MAINTENANCE.md] [--as-of YYYY-MM-DD] [--json]
```

Sweep snapshots are **append-only**: one full sweep = one
`data/sweep/<litertjs_version>/<YYYY-MM-DD>/` directory of
`<model_id>.json` result files; history is what makes deltas computable.
`DELTA_LOG.md` is the web-lane counterpart of Phase 8's release report:
per-model/per-backend regressions and improvements (index status
vocabulary), latency shifts beyond the threshold, catalog changes, and
transform rules currently flagged stale.

### `edge-fix rules reverify` (spec 11.4)

```sh
edge-fix rules reverify data/transforms [--runtime-version X] [--date YYYY-MM-DD] [--json]
# Exit: 0 all pass · 1 stale rules · 2 data error / verification unavailable
```

Re-runs every rule's declared fixture against the current runtime. A
**measured** failure flags the rule `stale` in its file (transform_rule
schema 1.1, additive) — never auto-deleted; a later passing run removes the
flag. Without the `runners` extra nothing can be measured, so nothing is
ever flagged (exit 2 instead). Stale rules are listed in the next
`DELTA_LOG.md` entry.

### The four workflows

| Workflow | Trigger | What it does |
|---|---|---|
| `resweep.yml` | weekly · `@litertjs/core` release (via autobump) · manual | quick tier (harness self-test + example catalog) every trigger; full catalog weekly/on release. Commits a new dated snapshot under `data/sweep-ci/` (a root separate from the owner's Mac snapshots in `data/sweep/`; the Mac stays the catalog's sweep axis and CI is a labelled second lane: no delta entry, no card re-enrichment from CI — DECISIONS #166(e), ruled 2026-08-29), re-verifies transform rules. Runs the harness in a fresh process per batch of 8 models with a transient per-model download (`--cache-max-bytes 0`: the 33 GB catalog fits neither the runner's disk nor the 10 GB Actions cache); a model that kills the harness even alone stays unmeasured — the measured snapshot is still committed and the job goes red naming it (DECISIONS #166). Site redeploy stays owner-dispatched while §C holds. |
| `autobump.yml` | daily · manual | custom checker for the two runtime axes: refreshes `data/releases.json`; on a new `@litertjs/core` release opens a bump PR and fires the full re-sweep. Native LiteRT release detection stays with the owner's watcher wired to Phase 8 `release-check`. Standard deps: Dependabot (`.github/dependabot.yml`). |
| `agentfix.yml` | red resweep/autobump run · manual | packages logs + delta + stale rules and invokes a headless coding agent (env-configured: `vars.AGENTFIX_COMMAND` + `secrets.AGENTFIX_API_KEY`) constrained to the failing component. Output is a **draft PR**, never auto-merged; unconfigured, it is a dispatch-testable stub that attaches the failure report only. |
| `reminders.yml` | monthly · manual | runs `freshness overdue` against `MAINTENANCE.md` and opens or bumps one reminder issue per overdue item. |

`MAINTENANCE.md` is the manual-verify register: what cannot be CI'd
(on-device card benchmarks, matrix entries needing an attached device,
unprobeable signatures), each with a re-verify interval and a
`last_verified` date the owner updates by hand.

### Sweep download cache

`npm run sweep -- … --cache-dir <dir>` downloads URL-sourced catalog models
once (content-addressed by source URL, atomic writes) and serves them from
the local server on later runs — a re-run of an unchanged catalog downloads
nothing. A failed download falls back to the direct URL so the page records
the honest `fetch_failed` result.
