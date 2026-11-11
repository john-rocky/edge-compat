# Device runs

`DECISIONS` and `PROGRESS` citations on this page refer to the logs of the private lab repository.

## Device-run results (Phase 13): the NPU list and the LLM table

Matrix schema `1.3` registers the NPU backend IDs — `npu_qnn` (Qualcomm Hexagon
via QNN), `npu_neuropilot` (MediaTek) — with **zero entries** (the Phase 6
pattern; final ID set pending the owner's device fleet). NPU behavior varies
per SoC and vendor SDK, so `target` (`device`, `soc`, `driver` — the driver
field carries the vendor SDK version) is **mandatory on NPU snapshots**,
enforced by `matrix validate`. `edge-lint --backend npu_qnn` runs fine and
honestly reports `unknown` until log-backed or probe-backed entries exist.

Model-level facts live in **device-run results**
(`schemas/device_run_result.schema.json`, deliberately parallel to the sweep
schema): per (model × device × accelerator) — loads / runs / delegation
signals (with delegated-op counts when the runtime log states them) /
output-match vs the CPU reference / CV latency / LLM metrics
(`prefill_tokens_per_s`, `decode_tokens_per_s`, `ttft_ms`, `peak_mem_mb`,
context length) — each with verbatim log evidence and a mandatory `env` block
(`runtime` is `litert` for the .tflite-on-NPU lane or `litert-lm` for the
.litertlm lane; that choice also selects the staleness axis in
`data/releases.json`, where Phase 13 added the `litertlm` axis watched by
autobump). Snapshots are append-only:
`data/device_runs/<runtime_version>/<date>/<model_id>__<device>.json`, one
runtime per snapshot directory, verified at discovery.

**The trap rule, device edition:** op-level matrix entries are **never derived
from model-level device-run results** — in either lane. Op-level entries
require a runtime log naming the specific op (captured as `evidence` — the
gpu_audit logs contain exactly such extractions) or an `npu_adb` probe result.
Enforced by the same zero-matrix-writes tests as the web lane.

Ingestion adapters wrap the owner's EXISTING measurement instruments — those
tools remain the instruments; this repo is the structured home of their
results. Mappings were built against real staged samples
(`data/examples/llm_samples/`, owner decisions recorded in its README):

```sh
# gpu_gate_mac.sh logs (LiteRT-LM Mac GPU gate). Verdict comes from the LOG
# BODY only — the CLI exits 0 even on failure, and a results block with
# 0.00 tokens/s is a zero_throughput FAILURE, not a measurement. Fields the
# log does not carry (runtime version, date, host) are owner-supplied options;
# --speeds-contaminated ingests a retracted batch's verdict while withholding
# every speed figure.
edge-compat device-run ingest-gpu-audit model.gpu.log \
  --model-id gemma3-1b --device mac-m4-max --device-name "M4 Max" \
  --runtime-version 0.15.0 --date 2026-07-23 --out data/device_runs

# devicemark leaderboard rows: only *__litertlm rows are this lane (coreai
# rows belong in a card's cross_runtime list and are skipped with a note);
# mem_measured:false memory is carried as metrics.peak_mem_est_mb, never as a
# measured peak_mem_mb.
edge-compat device-run ingest-devicemark measurements.jsonl \
  --runtime-version 0.15.0 --date 2026-07-22 --accelerator cpu

# compat_check reports (litert-lm version × artifact loadability; this axis is
# NOT absorbed into the matrix). Refuse-until-sampled: status branches with no
# staged real sample (BROKEN, SUSPECT) are refused by name, never invented.
edge-compat device-run ingest-compat-check compat_0.15.0.json \
  --date 2026-08-10 --device mac-m4-max --accelerator cpu

edge-compat device-run validate data/device_runs   # schema + layout, exit 0/1/2
edge-compat device-run validate data/device_runs --cards cards   # + every card is the
                                                     # newest measurement per cell (#165)
```

`edge-card enrich --device-runs-root data/device_runs` merges records into
cards as the additive schema-1.2 `device` block (accelerator records embedded
verbatim): every snapshot under the root is read and each (device x
accelerator x prompt-length) cell takes its newest measurement, so no snapshot
is hand-picked (DECISIONS #165). `--device-runs <snapshot-dir>` (repeatable)
enriches from exactly those snapshots instead — and because a card's device
block is rebuilt from what is passed, a run that would drop a cell a card
carries today is refused with nothing written unless `--allow-drop` states the
drop. Enrichment adds the Device column to `cards/index.json` /
`cards/README.md`, links snapshots from `llms.txt`, and the site index gains
env-labeled, stale-flagged device columns. `device-run validate --cards`
checks that the cards are exactly that selection; `tests/test_device_coverage.py`
runs the same check against the committed tree. `edge-compat freshness
delta` diffs the two most recent device-run snapshots **per runtime** into
`DELTA_LOG.md` alongside the sweep entries.

## Catalog regression (own `.litertlm` bundles × runtime wheel)

A ship gate runs once per file, on the day of the ship. `tools/regression/`
re-runs one functional suite over **every** own `.litertlm` chat bundle on one
litert-lm wheel, so a check learned later (a thinking-off digit loop, a chat
template that drops the user turn, a shared Engine that carries recurrent state
across conversations, a tool-call format the runtime does not parse) reaches
the files already shipped. Results are model-level and sit beside the device
runs; they never become matrix rows (`tests/test_regression.py` enforces it).

- `data/regression/bundles.json` — the own-bundle list, regenerated by
  `tools/regression/list_bundles.py` from the shipping manifests, the cards
  and the Hub API (file-level ownership by commit author; live LFS sha256).
  `--check` exits 1 when the Hub moved.
- `data/regression/<runtime_version>/<YYYY-MM-DD>/<bundle_id>.json` — one
  result per bundle (`schemas/regression_result.schema.json`: 13 checks, each
  a fresh process, `PASS | FAIL | NA` with a fixed reason vocabulary and the
  first 160 characters of the answer) plus the generated `SUMMARY.md` of the
  directory (`tools/regression/summarize.py`, `--prev` diffs against an
  earlier wheel). Append-only, dated, one runtime per directory.
- `tools/regression/README.md` — the suite table, the new-wheel procedure and
  the `--only … --checks …` re-run used when a hazard row is added.

Decided in DECISIONS #205 (2026-10-12); first full run on litert-lm 0.18.0.
