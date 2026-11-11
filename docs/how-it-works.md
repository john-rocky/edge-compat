# How the project works

**Will this `.tflite` run on the GPU, the NPU, or in the browser — and where will it fall back?**
edge-compat answers that for [LiteRT](https://ai.google.dev/edge/litert) (Google's on-device
runtime, the new name for TensorFlow Lite) with measurements, not guesses: an op-level delegate
compatibility matrix (659 entries in 18 backend × runtime-version snapshots), `edge-lint` — a
static pre-flight that reads a `.tflite` and predicts delegate fallback before you run it — 203
model cards with 751 dated device runs (Galaxy S26, Pixel 8a, iPhone 17 Pro, Raspberry Pi 5,
Apple M4 Max), and a browser sweep of 92 models on `@litertjs/core` 2.5.3. Passing and failing
models are both listed. Counts as of 2026-09-08; every row carries its own date and environment.

**Questions this repository answers**

- Which TFLite ops does the LiteRT GPU delegate (ML Drift, the CompiledModel GPU accelerator)
  run, fall back on, or compute wrong, per runtime version? → [`data/matrix/`](../data/matrix/),
  `gpu_mldrift*__<version>.json`; every entry carries its evidence and provenance.
- Why does my model fall back to CPU, or return wrong numbers, on the GPU? → `edge-lint` names
  the op, the matched entry, and the rewrite the matrix recorded; `edge-fix` applies recorded
  transforms as a dry run.
- Which ops does the Qualcomm NPU path (QNN / HTP) accept or reject? →
  `data/matrix/npu_qnn_htp_android__*.json`.
- Does this model run in the browser with LiteRT.js on WebGPU, and how fast? → the
  [site table](https://john-rocky.github.io/edge-compat/) and [`data/sweep/`](../data/sweep/).
- How fast is model X on device Y with which runtime version? → its card under
  [`cards/<model>/CARD.md`](../cards/README.md), with the raw record under `data/device_runs/`.

Pick the snapshot for your backend and runtime version from [`data/matrix/`](../data/matrix/)
(one file per backend × version). An op with no measured entry reports `unknown` — never a
guess. To verify an install, run the shipped example after `uv sync` in a clone:
`uv run edge-lint data/examples/model_mixed_example.tflite --matrix data/examples/matrix_example.json`
(exit 0, "40.0% of 5 ops delegated · 3 blocking ops").

**Public mirror.** The public repository is a scrubbed export of the project's private lab
repository, produced by `tools/export_public.sh`; `EXPORT.md` in the mirror names the source
commit. Numbered `DECISIONS`/`PROGRESS` citations in these docs refer to the lab's logs.
Site: https://john-rocky.github.io/edge-compat/ — `llms.txt` at the site root.

Developer tooling built around one shared data asset: a **delegate compatibility matrix**
for LiteRT, derived from real conversions and real device measurements. LiteRT's GPU/NPU
delegates are closed-source and there is no public ground truth for op coverage; this repo's
matrix is that ground truth, exposed to humans, CI, and AI agents through stable JSON, exit
codes, and clean markdown.

**Machine consumption first.** All generated output is deterministic (sorted keys, stable
ordering, byte-identical reruns) so diffs are reviewable and pipelines can trigger on real
change.

*(Disclosure line and affiliation notice — single-sourced in `src/litert_compat/branding.py`,
mirrored in `site/src/shared/branding.ts`, rendered on every public surface and checked by
`edge-compat release-gate`.)*

## Not to be confused with

`litertlm-convert/qa/compat_check.py` (and `quant_scan.py`) check a **different axis**:
release-day *runtime-version × shipped-artifact loadability*. This repo's matrix is
**op-level delegate support** per (backend × LiteRT version). The two are complementary
and stay separate.

## Data integrity

- **No fabricated data, ever.** Every entry carries `provenance`:
  `measured | vendor_doc | inferred | example`. All real data comes from the owner's
  measurements; `data/examples/` is clearly marked `"provenance": "example"` and is what
  all tests run against.
- **One snapshot file per (backend × litert_version).** Delegate op support changes across
  runtime versions and differs per delegate, so backends and versions are never merged
  into one file, and every lookup verdict carries the snapshot's `backend` +
  `litert_version`.
- **Staleness is visible, never silent.** Snapshots are carried forward explicitly
  (`matrix carry-forward`), never edited in place: carried `measured` entries downgrade to
  `inferred` with `evidence.derived_from` pinning their source. The `inferred` count of a
  file *is* its staleness debt; `matrix validate` reports per-provenance counts.
- Op names are **exact TFLite builtin operator names** (`GATHER_ND`, `TRANSPOSE_CONV`, …).
  No aliases.
