# device_runs_staging — records that await an owner decision before they enter data/device_runs

Same layout and schema as `data/device_runs/` (`<runtime_version>/<YYYY-MM-DD>/<model_id>__<device>.json`,
`edge-compat device-run validate data/device_runs_staging` passes), but each section below names a decision the owner
has not made yet (a runtime version token, or whether and how a failure is recorded). Nothing here is read by
`edge-card enrich`, the tests or the site until it is moved.

## prebuilt-55bf01c8/2026-09-28 — Audio8-TTS-Preview-0.6b on the Galaxy S26 (7 files, 14 records)

Source: `~/code/litertlm-convert/audio8_tts_work/evidence/logs/s26_*.log` (LiteRT `benchmark_model` legs, 2026-09-28).
The binary that produced them is `/data/local/tmp/litert-cli/benchmark_model` on the phone, sha256
`55bf01c89beeb1175ab3889fd820a9178a5d995a13f4a4e5c54476136e125f2f` (8,284,184 B, pushed 2026-09-22 13:05 JST), byte-identical
to `~/.cache/litert-cli/binaries/arm64-v8a/benchmark_model` on the Mac (the litert-cli package's download cache, same mtime).
What is established (2026-09-28):

- It is not the `2.2.0` release binary (`gs://litert/binaries/2.2.0/android_arm64/benchmark_model`, sha256 f3729eaf…), not `2.1.6`
  (ed943c8c…) and not the `latest` object (cf61b4ff…, 8,053,552 B, unchanged since 2026-08-12).
- litert-cli-nightly 0.3.0.dev20260911 and later fetch `gs://litert/binaries/nightly/android_arm64/benchmark_model`, an object that is
  overwritten daily (today's: 8,301,368 B, written 2026-09-28 04:33 UTC); the 09-22 download therefore was the nightly of 2026-09-21 or
  2026-09-22 (13:05 JST = 04:05 UTC, before that day's 04:33 UTC upload) and its bytes are no longer in the bucket. The binary prints no
  version string and none appears in its strings.
- The device-run schema needs a non-empty `runtime_version` that also names the snapshot directory. The placeholder here is the binary's
  digest prefix. Owner decision: keep `prebuilt-55bf01c8`, or use a nightly token (`2.3.0.dev20260921`-style needs the ai-edge-litert-nightly
  version of that build, which is not established), or a form of the owner's choosing.

To land: pick the token, then
`sed -i '' 's/"runtime_version": "prebuilt-55bf01c8"/"runtime_version": "<token>"/' data/device_runs_staging/prebuilt-55bf01c8/2026-09-28/*.json`,
`git mv data/device_runs_staging/prebuilt-55bf01c8 data/device_runs/<token>`, `uv run edge-compat device-run validate data/device_runs`,
`uv run edge-card enrich --device-runs-root data/device_runs --device-runs-root data/examples/device_runs --sweep data/sweep/2.5.3/2026-09-28 cards`
(the 2026-09-28 sweep must be passed together with the device roots — `build` and `enrich` rebuild the whole block), `uv run edge-card index cards`,
`uv run pytest -q`. The `gpu_mldrift__NEEDS-VERSION-audio8-additions.csv` rows in `data/matrix_staging/` take the same token in their
`evidence_litert_version` column and file name.

Record conventions used (see DECISIONS #174): benchmark_model prints avg / min only, so `latency_p50_ms` stays null and the figures sit in
`metrics` (`inference_avg_ms`, `inference_min_ms`, `timed_runs`, `init_footprint_mb`, `overall_footprint_mb`); `artifact` is the file name
the bench driver used on the phone, with the mapping to the published name in `notes` and no `artifact_sha256` (the device copy was not
digested in the log); unpublished predecessor builds measured in the same session (three-signature slow AR, v1 codec attention, T256 codec,
`_g` encoder) are cited in `notes`/matrix conditions, never rows.

## 0.16.0/2026-09-29 — decider-2b-vision fp16-int8vocab on the Galaxy S26 GPU (1 file, 1 record): a failure the owner has to classify

Source: `~/code/litertlm-convert/decider2bv_work/logs/s26_r5/r5_L2_v7c_gpu_game_pong_atari_level.{err,out,probe.txt,poll}` and
`results/s26_r5_rows/r5_L2_v7c_gpu_game_pong_atari_level.json` (round 5, 2026-09-29 02:39-02:41 JST). The runtime version is not the open
question: the S4 kit's `litert_lm_advanced_main` is the same binary as the int8 rows that went straight into `data/device_runs/0.16.0/2026-09-29/`.
What happened: the file (published as `decider-2b-vision_fp16-int8vocab.litertlm`, sha256 72f68360…, device-side sha256 checked after the push)
was run on the GPU with the published row game_pong_atari_level; the lane's device-side guard (VmHWM cap 6,347,656 kB = 6.5e9 B, fixed before the
run because this phone has restarted its framework above ~6 GB and rebooted on other int8 bundles) killed the process during engine creation:
`KILLED_OVER_RAM VmHWM_KB=6396212 CAP_KB=6347656`, `EXIT=137`, MemAvailable down to 693,688 kB, lmkd reclaiming three cached apps, the only
printed delegation line `Replacing 25945 out of 25945 node(s) with delegate (LITERT_CL) … (prefill_1024)` followed by
`Initializing OpenCL-based API from serialized data.`; no reboot, no framework restart. The owner-written Hub card publishes it as "engine not
created: stopped at 6.55 GB VmHWM during OpenCL set-up of prefill_1024 (our 6.5 GB guard)".

The record is staged as `loads: false, runs: false`, `failure_class: "memory_cap_exceeded"`, the probe line as `error`, `peak_mem_mb` null (the
value at the kill is a lower bound, carried as `metrics.vmhwm_at_kill_mb`), no `prefill_tokens` (engine creation takes no prompt here:
`--disable_input_prompt_as_hint=true`), so its cell is `gpu`. Owner decisions:
1. the failure class — keep `memory_cap_exceeded` (a new name in the open set, the #168(b) way `degenerate_output` was added; it mirrors
   `timeout`, the watchdog form of a harness limit), use `null` with the cause only in `error` (#164(d)), or another name;
2. whether a guard-bounded negative belongs on the card at all. #164(e) voided every negative produced under an accidental wall-clock ceiling
   ("a harness with a ceiling cannot produce a negative result about a model, only about the pairing"); this ceiling was deliberate and the Hub
   card states the result, so the row would read "does not fit within a 6.5 GB VmHWM guard on this phone", which is what `error` says.

To land: optionally edit `failure_class`, then `git mv data/device_runs_staging/0.16.0/2026-09-29/decider-2b-vision-fp16-int8vocab__galaxy-s26.json data/device_runs/0.16.0/2026-09-29/`
(the target directory already exists), `uv run edge-compat device-run validate data/device_runs`,
`uv run edge-card enrich --device-runs-root data/device_runs --device-runs-root data/examples/device_runs cards` (the decider cards have no sweep
block, so no `--sweep` is needed), `uv run edge-compat freshness delta`, `uv run pytest -q`.
