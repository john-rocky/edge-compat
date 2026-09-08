# LLM adapter-mapping samples

Real output samples from the owner's three LLM measurement lanes, staged so the
**Phase 13 (Device-Run Columns / LiteRT-LM lane)** session can build ingestion adapters
against actual data instead of guessed field mappings.

**These are samples, not data.** Nothing here is ingested. No card, matrix row, or
benchmark record may be derived from this directory until `schemas/device_run_result.schema.json`
exists and the owner has confirmed the mappings marked **OPEN** below. Per the spec's Data
Integrity Rules, a field that a source does not carry is an open question, never a default.

The full outputs stay in their home repos; only excerpts live here.

## What is staged

| File | Source (read-only) | Source mtime |
|---|---|---|
| `gpu_audit/lfm25-12b-instruct-int4.gpu.log` | `~/code/litertlm-convert/reports/gpu_audit/` | 2026-07-22 |
| `gpu_audit/gemma3-1b-official.gpu.log` | same | 2026-07-23 |
| `gpu_audit/lfm25_wi8_composite_conv040dev_0150.gpu.log` | same | 2026-08-08 |
| `gpu_audit/summary.excerpt.txt` | `…/gpu_audit/summary.txt` (6 of 38 lines) | 2026-08-11 |
| `gpu_audit/MATRIX.excerpt.md` | `…/gpu_audit/MATRIX.md` | 2026-07-23 |
| `devicemark/measurements.slice.jsonl` | `~/code/devicemark/data/leaderboard/measurements.jsonl` (5 of 17 lines) | 2026-07-22 |
| `yardstick/northmv-wi8.iphone17pro.{quality,vision-no-text}.json` | `~/code/litertlm-convert/northmv_work/iphone_results_keep/` (iOS BenchmarkApp `--yardstick-autorun` result JSONs, 2 of 3 tasks of one run) | 2026-08-19 |
| `compat_check/compat_0.15.0.json` | `~/code/litertlm-convert/qa/out/compat_0.15.0.json` (whole file, 1 record) | 2026-08-10 |

**Only modification to the copied bytes:** the absolute home prefix `/Users/USER`
was replaced with `~` in the logs and in the compat_check `runtime` field. Line content is
otherwise verbatim. `MATRIX.excerpt.md` additionally drops rows for models not staged here
and one row containing a private thread reference (noted in its header comment).

Sanitization check: no private-registry evidence IDs and no `~/Downloads/meeting/` prose
crossed into this directory (spec §C firewall). Everything here comes from `litertlm-convert`
and `devicemark` tool output.

---

## Lane 1 — gpu_audit (Mac GPU gate, `.litertlm`)

Harness: `~/code/litertlm-convert/scripts/gpu_gate_mac.sh <model.litertlm> [tag]`, which runs

```
litert-lm benchmark <model> --backend gpu --cache no -p 256 -d 256 --max-num-tokens 1024
```

(the three numbers are env-overridable defaults `PREFILL` / `DECODE` / `MAXTOK`), writes the
combined stdout+stderr to `<tag>.gpu.log`, and appends one line to `summary.txt`.

Three log shapes are staged because the adapter has to distinguish all three:

1. **PASS with a Results block** — `gemma3-1b-official.gpu.log`. Carries
   `Prefill speed`, `Decode speed`, `Init time`, `Time to first token`.
2. **FAIL before engine creation** — `lfm25-12b-instruct-int4.gpu.log`. No Results block;
   carries the GPU delegate's `ERROR: Following operations are not supported by GPU delegate:`
   block with one line per unsupported op, plus the partition count
   (`536 operations will run on the GPU, and the remaining 43 operations will run on the CPU.`).
   This is the op-level matrix evidence: `ADD`/`CAST`/`SUM` rejected for `Tensor type(INT64)`,
   `GATHER_ND` (`Operation is not supported.`), `GREATER_EQUAL`/`LESS_EQUAL`
   (`Can't parse inputs with const tensors.`).
3. **Ran but produced nothing** — `lfm25_wi8_composite_conv040dev_0150.gpu.log`. A Results
   block *is* printed, with `0.00 tokens/s` for both prefill and decode, after every generate
   call failed (`The given map is missing some output TensorBuffers`).

### Traps this lane carries (all verified in the staged files)

- **A Results block is not a pass.** Shape 3 prints `Prefill speed: 0.00 tokens/s`. The
  harness was later hardened for exactly this (it now requires non-zero decode), but
  `summary.txt` still holds the pre-hardening row
  `RESULT PASS lfm25_wi8_composite_conv040dev_0150 prefill=0.00 decode=0.00` — so
  **`summary.txt` mixes two historical line formats** and the adapter must treat zero
  throughput as a failure regardless of the recorded verdict.
- **Exit code is not a signal.** The FAIL rows in the excerpt record `rc=0`, while the LFM
  log ends in a Python `RuntimeError` — the CLI exits 0 even when engine creation fails.
  Verdict has to come from the log body.
- **`summary.txt` is append-only with duplicate tags.** `gemma3-1b-official` appears twice
  (FAIL, then PASS). The script overwrites `<tag>.gpu.log` each run, so summary rows and log
  files are **not** 1:1 — only the newest run survives on disk.
- **The 2026-07-22 batch speeds are contaminated** and were retracted by the owner in
  `MATRIX.md` (parallel GPU compile load; BitCPM re-measured quiet at 1521/56 → 2965/167.7).
  PASS/FAIL verdicts from that batch stand; the numbers do not. The excerpt keeps both
  BitCPM rows so the adapter can be tested against a pair that must not both be ingested.
- **Mac GPU pass ≠ Android ML Drift pass** (different delegates). Backend identity must not
  be collapsed across the two.

### Fields the log does not carry (**OPEN** — owner input required)

`date` · litert-lm version (`harness_version`) · host machine / SoC (`device`, `soc`) ·
quantization variant (only inferable from the file name, e.g. `int4`, `wi8`, `b128`) ·
backend id to use for "litert-lm Mac GPU" · whether `litert-lm benchmark` internally repeats
runs (i.e. what `runs` / `statistic` should say — the script itself invokes it once).

File mtime is currently the only date signal. A cheap Phase 13 recommendation for the owner:
have `gpu_gate_mac.sh` stamp date + `litert-lm --version` into the log header, which removes
this whole class of open question for future runs.

### Proposed mapping (**unconfirmed**)

Results block → `benchmark_result.schema.json`: `Decode speed` → `tokens_per_s`,
`Prefill speed` → `metrics.prefill_tok_s`, `Time to first token` → `metrics.ttft_s`,
`Init time` → `metrics.init_s`, `harness` → `gpu_gate_mac.sh`, `provenance` → `measured`
(only for clean, non-zero rows).
Unsupported-op block → matrix rows (op, dtype constraint, status, evidence = this log
excerpt + model + version + date). Per the skill's rule, matrix rows sourced this way go to
the owner for review before import.

---

## Lane 2 — devicemark leaderboard

Five rows of `measurements.jsonl`, chosen to cover both runtimes and both `mem_measured`
values. Row fields: `artifact_id`, `runtime`, `device`, `decode_tok_s`, `peak_mem_mb`,
`mem_measured`, `power_w`.

Definitions from the source dataset card (`~/code/devicemark/data/leaderboard/DATASET_CARD.md`):

- `artifact_id` = `<slug>__<quant>__<format>`, the join key — e.g. `gemma-4-e2b__int4__litertlm`.
- `decode_tok_s` = **warm-state** S=1 pipelined decode (engine loaded and warmed, cold load
  excluded), measured by PipelinedBench: 128-token prompt / 256-token decode, two trials,
  settled device, numerics-gated.
- `mem_measured: false` = `peak_mem_mb` is **estimated, not device-measured**.

### Traps

- **`mem_measured: false` must not become a measured `peak_mem_mb`.** Three of the five
  staged rows are estimates. Either drop the field or carry the estimate flag through
  `metrics` / `notes` — silently promoting it would put a fabricated number in a card.
- **Two runtimes are mixed.** Only `*__litertlm` rows are the LiteRT-LM lane; `coreai` rows
  belong in a card's `cross_runtime` list, which requires the `runtime` field to be set.
- The dataset is decode-only — no prefill, no TTFT — so a devicemark-sourced record can
  never fill the LLM prefill column. gpu_audit is the only staged source that has prefill.

### Fields the rows do not carry (**OPEN**)

`date` · `harness_version` · `backend` · runtime version. The harness identity
(`PipelinedBench`, with its prompt/decode lengths and two-trial protocol) is documented in
the dataset card, not in the row — the adapter must take it from there rather than invent it.
This is the same gap the existing stub flags in
`src/litert_compat/cards/adapters/owner_harness.py`.

---

## Lane 3 — compat_check (litert-lm version × artifact loadability)

`compat_0.15.0.json`, produced by `~/code/litertlm-convert/qa/compat_check.py`. Top level:
`runtime` (binary path), `runtime_version`, `checked`, `broken`, `skipped`, `results[]`.
Each result: `repo`, `file`, `lane` (`llm` | `vlm` | `tflite`), `source`, `status`, `error`,
`answer` (the LLM lane asks a fixed arithmetic question and records the reply — the
loadability + sanity signal).

### Traps

- **The sample only exercises `status: "ok"`.** Reading `compat_check.py` shows two more
  states: `BROKEN` (load/run failure) and `SUSPECT` (the bundle loads and generates but gets
  the fixed arithmetic prompt wrong — the shape a silent numeric corruption takes). The
  `SUSPECT` error string is fixed in the source (`answer not 42: '<reply>'`); the `BROKEN`
  error string is whatever the subprocess emitted and is **not sampled here**. Ask the owner
  for one `BROKEN` sample rather than inventing that branch, or refuse records with an
  unrecognized status.
- `skipped` entries in the wider run are targets with no local copy (`compat_run.log` shows
  35 of them) — absence is "not checked", not "fails".
- No `date` field; file mtime is the only signal.

### Scope question (**OPEN**, already on the books)

PROGRESS.md "Open integration decisions" #2 — whether this runtime-version axis is ever
absorbed into the matrix — is still undecided, with the spec recommending *not before matrix
v1*. Phase 13 should treat this lane as a `device_run_result` input (runtime × artifact
loadability), not as a matrix source, unless the owner rules otherwise.

---

## Summary of what Phase 13 must ask the owner

1. Backend ids for the LLM lane (litert-lm Mac GPU, Android ML Drift, CPU) — the matrix and
   benchmark records both key on these.
2. litert-lm version + date + host machine for each staged gpu_audit log (none are recorded
   in the artifacts) — and whether the harness should start stamping them.
3. One `BROKEN` and one `SUSPECT` compat_check sample, so those branches are built against
   real strings.
4. How estimated `peak_mem_mb` should be represented once ingested.
5. Confirmation that contaminated (2026-07-22 batch) speed rows are import-blocked while
   their PASS/FAIL verdicts remain usable.

---

## Owner decisions (2026-08-11 — "recommendations accepted; #2 unknown")

1. **Backend ids — recommended scheme adopted.** Follow the candidate registry already
   proposed in the repo: `gpu_mldrift` (Android ML Drift), `cpu_xnnpack` (CPU). For the
   litert-lm Mac GPU lane, the Phase 13 session fixes the id following the DECISIONS #79
   evidence (Mac wheels serve `HardwareAccelerator.GPU` via the METAL delegate — proposal:
   `gpu_metal_mac`); the schema-level enum tightening stays deferred as before.
2. **Version/date/host for the staged logs: the owner does not know.** Resolution path for
   Phase 13: attempt recovery from the `litertlm-convert` git history (commit dates for the
   logs; lockfiles/env records pinning the litert-lm version at those commits; host evidence
   — the logs very likely come from the owner's Mac, but confirm rather than assume).
   Whatever stays unrecovered: ingest device-run records with the recoverable bounds only,
   and hold back op-level matrix rows from those logs (measured entries require
   `evidence.litert_version`) until a re-run under a stamping harness supplies it.
   **Forward fix approved:** `gpu_gate_mac.sh` should start stamping date +
   `litert-lm --version` + host into the log header (owner-side one-line change).
3. **Refuse-until-sampled adopted.** The adapter refuses compat_check records whose status
   branch has no staged real sample (`BROKEN`, `SUSPECT`) instead of inventing the strings;
   the owner supplies real samples when they next occur, unblocking those branches.
4. **Carry-with-flag adopted.** Estimated memory (`mem_measured: false`) is never written to
   a measured `peak_mem_mb`; it is carried as an explicitly-estimated metric
   (e.g. `metrics.peak_mem_est_mb`) with the flag preserved, or omitted where the schema has
   no honest slot.
5. **Confirmed.** 2026-07-22 batch: speed numbers import-blocked; PASS/FAIL verdicts usable.

Also confirmed (lane 3 scope, per the spec's standing recommendation): compat_check stays a
`device_run_result` input — the runtime-version axis is NOT absorbed into the matrix before
matrix v1 (PROGRESS "Open integration decisions" #2 unchanged).

## yardstick (iOS BenchmarkApp) — added 2026-08-20, adapter `ingest-yardstick`

The JSON stamps its own UTC `timestamp`, `task`, `outputSample`, `parameters`, `runtime`
name, `device.{modelIdentifier,systemVersion,physicalMemoryMB}` and the measured
`metrics` (`promptTokensPerSecond`, `decodeTokensPerSecond`, `firstTokenLatencyMS`,
`memoryPeakDuringDecodeMB` — device-measured, `loadTimeSeconds`, token counts,
`stopReason`). It does **not** carry the runtime **version** (linked into the app build),
the **accelerator** (the app's `--litert-cpu` flag; absent = Metal GPU on iOS), the
**published artifact** (`model.primaryFile` is the staged copy's name and
`model.displayName/quantization` are the app catalog's text — in this very sample the
catalog text still says "fp16 vision" while the staged file was the int8-vision build),
or any **expected answer** for the vision probes. Adapter decisions: those four are
caller-supplied; the date defaults to the source's UTC day; `output_match` stays null
with the `outputSample` carried verbatim in evidence; several task files of one
(model, device, accelerator) merge into one record (throughput from the longest
generation, peak memory = max over tasks, every task's numbers under task-prefixed
`metrics` keys).

## npubench/ (added 2026-08-27)

Real excerpt of the classic-`.tflite` NPU/GPU sweep journal
(`litertlm-convert/community_accel_work/s2_npu_sweep/results.jsonl`, Galaxy S26
/ SM8850, LiteRT 2.2.0 APK, JIT + AOT lanes). Covers: accepted JIT cold/cached
+ GPU rows, an accepted AOT row, a multi-file repo (id gets the `__<stem>`
suffix), every staged failure shape (compile/load/invoke fail, process crash,
AOT host compile-fail, silent-XNNPACK-fallback), an annul pair, a soc-mismatch
skip and an unobtainable row. `ingest-npubench` is mapped against exactly
these shapes and refuses error strings not staged here.

## npubench/results_p8a_sample.jsonl + npubench/parity_report_sample.json (added 2026-09-02)

Two more real shapes from the same `com.litertzoo.npubench` APK (LiteRT 2.2.0):

- `results_p8a_sample.jsonl` — three verbatim rows of
  `litertlm-convert/community_accel_work/s6_p8a/results.jsonl` (Pixel 8a /
  Tensor G3, `run_p8a.py`, 2026-08-28): an accepted GPU row, an accepted CPU
  row (both carrying the driver's captured `evidence.replacing` residency
  line), and an unaccepted CPU attempt (thermal `LIGHT->LIGHT`). This driver
  writes **no `repo`/`slug`**, so `ingest-npubench` takes `--repo-for
  <file>=<repo>` from the sweep README and derives the slug; the `cpu`
  backend lands as `cpu_xnnpack`.
- `parity_report_sample.json` — two verbatim keys of
  `litertlm-convert/granite_speech_work/s26/s26_gate_report.json` (the
  `#parity` test, Galaxy S26, 2026-09-01): one `RAN` CPU leg with the parity
  fields (`ids_match_*`, `logit_maxdiff_mac`, transcript) and one `FAILED`
  GPU leg (`LiteRtException: Failed to compile model`). The report carries no
  file names, signature names, or timestamps — `ingest-npubench-parity` takes
  `--file <variant>=<file>`, `--signature <dur>=<name>` (from the logcat's
  `#transcribe_10s` asset suffix) and `--date`.

## pi5/ (added 2026-09-02)

Two verbatim rows of `litertlm-convert/community_accel_work/pi5/data/results.jsonl`
(Raspberry Pi 5 Model B Rev 1.1, `pi5_bench.py`, LiteRT `benchmark_model` from
litert-cli-nightly 0.2.0.dev20260805 / ai-edge-litert-nightly 2.2.0.dev20260804,
2026-08-31): a single-signature file and one signature row of a multi-signature
file (`talker_int4.tflite` / `decode`). Every row carries `versions` (the
runtime version is read from the row, never supplied), a UTC `ts`, and three
invocations with per-invocation temperature, `vcgencmd get_throttled` word,
XNNPACK flag, stats and the tool's peak footprint. `run_log_excerpt.txt` is the
head of one raw `benchmark_model` run log from `data/logs.tgz` — it prints no
`Replacing N out of M` line, so delegation counts stay null on this lane.
`ingest-pi5-benchmark` is mapped against these two rows; error shapes are not
staged (the 121-row sweep had none) and are refused.

## pi5/llm_results_sample.jsonl + pi5/llm_run_log_excerpt.txt (added 2026-09-08)

Wave 2 of the Raspberry Pi 5 card campaign: the owner's LLM `.litertlm` bundles through
`litert-lm benchmark` (litert-lm 0.16.1 in the Pi's `venv-lm`, `--backend cpu
--cpu-thread-count 4 -p 256 -d 256 --runs 1 --cache memory`, 3 invocations per file with
cool-down to <=52 C, `vcgencmd measure_temp` + `get_throttled` per invocation, peak RSS
polled from /proc, and a real-generation gate — `litert-lm run` with the degenerate-output
check — before a file's numbers are quotable). Source journal
`~/code/litertlm-convert/community_accel_work/pi5/data/llm_results.jsonl` (55 rows,
2026-09-01/02); protocol in `community_accel_work/pi5/PI5_LLM_WORKLIST.md`. Three rows are
staged verbatim: a multi-bundle repo (LFM2.5-1.2B-Instruct int4 + int8, so the id-suffix
rule has a real case) and a single-bundle repo (granite-4.0-h-1b). Every row carries
`versions` (`litert-lm` is the runtime version — read from the row, never supplied),
a UTC `ts`, the gate verdict, the three invocations with their own exit / throttle word /
temperatures / per-invocation figures, and the medians the driver quotes on cards.
`llm_run_log_excerpt.txt` is the head of one raw `litert-lm benchmark` run log from
`data/llm_logs.tgz` (the `Number of tokens in …` headers are present; no `Replacing N out
of M` line on the CPU path, so delegation counts stay null on this lane).
`ingest-pi5-llm` is mapped against these rows; `--model-id-for` files a bundle under the
cards' hand-chosen id where one exists (joins are still a rendering concern, this only
names the id the record is written under). Rows with `error`, a non-`cpu` backend, a failed
gate, or a throttled / non-zero invocation are skipped with a note, never guessed.
