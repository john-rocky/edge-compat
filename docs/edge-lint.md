# edge-lint: static delegate pre-flight

```sh
edge-lint MODEL.tflite \
  --matrix data/matrix/<backend>__<litert_version>.json \
  [--backend gpu_mldrift] \
  [--json | --md] \
  [--fail-on fallback] [--fail-on partitions:N] \
  [--rules data/transforms]
```

Statically lints a `.tflite` model against a matrix snapshot — **no LiteRT
runtime, no model execution**: the flatbuffer is parsed directly, so the linter
runs anywhere Python runs. LiteRT-LM `.litertlm` bundles are accepted too
(detected by magic bytes): the embedded TFLite model is extracted from the
container TOC and linted statically — the report's `model.path` is marked
`::TFLiteModel@section<i>` and `model.sha256` hashes the analyzed embedded
bytes, so container analysis is always visible as such. Exactly one
`TFLiteModel` section is required; ambiguous multi-model bundles are refused
with the section listing (E2E bundle audits stay in litertlm-convert).
Pipeline: parse → classify every node through
`Matrix.lookup` (signature from the node's output tensors) → simulate
partitions (maximal runs of delegate-claimed nodes in execution order; every
partition boundary implies a host↔device sync) → rank blocking ops by a
labeled heuristic → report.

- **Formats**: human text (default), `--md` markdown, `--json` validating
  against `schemas/lint_report.schema.json` (a public, versioned API).
- **Claimed statuses**: `delegated`, `partial`, and `incorrect` count as
  claimed by the delegate for partition simulation — `incorrect` ops *do* run
  on the delegate, just wrongly, and are additionally surfaced as correctness
  warnings. `unknown` never claims: no matrix entry, no guess.
- **Rewrite suggestions** are surfaced verbatim from matched entries'
  `rewrite_hints` — never generated.
- **Needs-probe list**: every unknown-verdict builtin op is emitted as a
  complete lookup signature (op, dtypes, shape_meta, backend) — sufficient to
  construct a probe without reopening the model (consumed by the Phase 8 probe
  suite). Custom ops are listed separately; they are outside the matrix
  vocabulary and not probeable.
- **Constraint keys the linter computes** (matched, never reported — pinned
  reports do not change): `rank` and `dynamic_shape` of the output; per input
  `operand_a..d` (`constant` | `activation`) and `operand_a_rank..d_rank`; for the
  PAD family `axis` (`innermost` | `non-innermost` | `none` | `unknown`, read from
  the constant paddings); for `FULLY_CONNECTED` / `CONV_2D` / `DEPTHWISE_CONV_2D`
  `weight_dtype` and `weight_source` (`constant` | `DEQUANTIZE` | `activation`);
  and `consumer_ops` (the distinct builtins consuming output 0, `+`-joined).
  These are what the rows of the LiteRT #10445 class name — a rank-2 runtime
  tensor plus a rank-3 constant (#10231), a rank-3 `PAD` on a non-innermost
  axis (#9272), an int8 `FULLY_CONNECTED` feeding an `ADD` (#9277), an
  explicit-dequantize `FULLY_CONNECTED` (#9523) — so those measured rows match
  nodes instead of staying documentation-only. Any other constraint key still
  matches by equality against this dict; a key the linter does not compute
  never matches (unknowable never resolves toward a verdict).
- **`--backend`** is optional and defaults to the matrix file's backend; a
  mismatch is a usage error — one snapshot per (backend × litert_version),
  and no output ever presents one backend's result as another's.

Exit codes: `0` clean · `1` a `--fail-on` rule triggered · `2` usage/parse
error. `--fail-on fallback` triggers on `fallback`, `incorrect`, **and**
`crash` verdicts; `--fail-on partitions:N` triggers when the partition count
exceeds N; the flag is repeatable and rules combine.

### Static pre-flight vs runtime confirmation

The upstream `litert_gpu_toolkit` checker
([`checker.py` in google-ai-edge/litert-samples](https://github.com/google-ai-edge/litert-samples),
`check_gpu_compatibility()`) is a **runtime compile-and-compare** check: it
needs TensorFlow and the LiteRT runtime, actually compiles the model for GPU,
and returns a binary verdict. `edge-lint` is its **static complement**:
flatbuffer-only, matrix-backed, partition simulation, stable JSON and exit
codes, and it names *which* op blocks and *why* (with the matrix entry's
provenance). Run both: `edge-lint` as the cheap pre-flight everywhere (CI,
agents, laptops), the runtime checker as confirmation where a runtime and
device are available.

### CI example (GitHub Actions)

```yaml
- name: LiteRT delegate pre-flight
  run: |
    pip install edge-compat
    edge-lint model.tflite \
      --matrix data/matrix/gpu_mldrift__2.2.0.json \
      --fail-on fallback --fail-on partitions:1 \
      --json > lint_report.json
  # exit 1 fails the job when any op falls back / crashes / is numerically
  # incorrect, or when the graph splits into more than one partition
```
