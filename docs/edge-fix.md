# edge-fix: diagnosis and repair (Phase 10)

`DECISIONS` and `PROGRESS` citations on this page refer to the logs of the private lab repository.

The matrix's `rewrite_hints` are human prose; `edge-fix` promotes them to
executable, deterministic graph surgery: lint finds why a model won't delegate, fix
makes it delegate — then proves it didn't change the math.

```sh
edge-fix MODEL.tflite --backend gpu_mldrift \
  --matrix data/matrix/<snapshot>.json --rules data/transforms/ \
  [--dry-run (default) | --apply --out FIXED.tflite] \
  [--json report.json] [--allow-unverified] \
  [--tolerance-abs 1e-5] [--tolerance-rel 1e-3]
```

Pipeline: **lint** the model → **match** transform rules to the non-delegated
findings → **plan** (ordered, conflict-checked: a node claimed by two rules is left
untouched and reported) → **apply** on a loaded copy via the flatbuffer object API →
**re-lint**: the metric must improve — delegated coverage first, then
incorrect+crash count, then partition count; `io_cast` plans may instead reduce the
interface-violation count without regressing the lint metrics — else automatic
rollback and a `no_improvement` report → **numerical verification**: original vs
fixed on the LiteRT CPU interpreter (the same `edge-compat[runners]` extra and
tolerance policy as the sweep and probe runners) on seeded deterministic inputs.

Safety defaults, all tested: dry-run is the default; the input file is never
modified in place; a `schemas/fix_report.schema.json`-valid report is always
emitted; a **failed** verification rolls back and is never overridable. Without the
runners extra, `--apply` refuses unless `--allow-unverified` is explicitly passed —
and then the report is marked `unverified`. Deterministic and idempotent: same
inputs → byte-identical fixed model and report; running fix on its own output
applies zero further changes.

Exit codes: `0` fixed (or nothing to fix) · `1` unfixed issues remain or automatic
rollback · `2` usage/data error (including the `--apply` verification refusal and
models the rewrite path cannot round-trip faithfully — multiple subgraphs, custom
ops, custom quantization `details`, unregistered builtin-options tables are refused
with precise reasons, never silently dropped). Real converter output — quantized
tensors, model metadata, signature defs, appended-weight buffers, optional `-1`
inputs — round-trips faithfully (DECISIONS #153–#155).

### Transform rules: machinery from the toolkit, knowledge from the owner

Rules are JSON files (`<id>.json`, `schemas/transform_rule.schema.json`) in a rules
directory — `data/transforms/` for the owner's real rules (none yet; the committed
example-provenance rules live in `data/examples/transforms/` and are exercised only
on synthetic fixtures). Every applied change in a fix report traces to a rule `id`
with its evidence; unmatched findings are reported, never guessed at. Matrix schema
`1.2` adds the link in the other direction: a `rewrite_hints` entry may carry a
`transform_id` naming the rule that executes it (surfaced verbatim by `edge-lint`,
report schema `1.1`). With `edge-lint --rules DIR`, hints whose `transform_id`
resolves to a rule in DIR are annotated **fixable by edge-fix** in the
text/markdown formats, with the exact `edge-fix` command; a dangling
`transform_id` is called out as unresolvable. The JSON report never changes —
it carries `transform_id` verbatim for machine consumers to resolve themselves.

A rule: `{id, title, match: {op | op_sequence, dtypes, constraints}, action,
params, expected_effect, provenance, evidence, notes, fixture}`. `match` uses the
exact `Matrix.lookup` semantics (dtypes subset, constraints vs shape_meta).
`fixture` declares the rule's own demonstration model + matrix; `edge-fix rules
validate` dry-runs the rule against it and requires the fix to succeed end-to-end.
(`op_sequence` is part of the schema API but not executed by the v1 engine — such
rules are refused with a precise reason, not errors.)

`action` is a **closed vocabulary** — anything not expressible in it is not a rule;
the vocabulary is extended by spec amendment, not freeform actions. One worked
example per action (each committed under `data/examples/transforms/`):

- **`replace_op`** — swap the operator in place, inputs/outputs verbatim.
  `"params": {"new_op": "FLOOR_DIV"}` — `example-replace-div-floor-div` rewrites
  integer `DIV` → `FLOOR_DIV` (exact for non-negative operands; the validity caveat
  lives in the rule's `notes` and is surfaced in every report).
- **`decompose`** — replace the node with a fixed recipe of nodes.
  `"params": {"nodes": [{"op": "MUL", "inputs": ["in:0", "in:0"], "outputs":
  ["out:0"]}]}` — `example-decompose-square-mul` rewrites `SQUARE(x)` → `MUL(x, x)`.
  Refs: `in:K`/`out:K` are the matched node's tensors, `tmp:NAME` a declared
  intermediate whose dtype/shape borrow from a matched tensor (`like:in:K`).
- **`insert_cast`** — compute the node in another dtype, CASTs at the edges.
  `"params": {"to": "float32"}` — `example-insert-cast-mul-int32` computes integer
  `MUL` in float32 (exact while products stay under 2^24).
- **`io_cast`** — wrap graph I/O to satisfy interface constraints (e.g. the
  LiteRT.js float32/int32 tensor I/O rule); the interior graph is unchanged.
  `"match": {"dtypes": ["int64"]}, "params": {"to": "int32", "io": "both"}` —
  `example-io-cast-int64`.

Rule-emitted ops are limited to the v1 options registry (option-free unary ops,
binary elementwise ops, `CAST`, `RESHAPE`) — a rule cannot emit an op whose
builtin-options contract is unknown. Extending the registry is additive, the same
pattern as the probe templates.

### Authoring path

```sh
edge-fix rules new --id my-rule --action replace_op --op DIV \
  --out-dir data/transforms/          # scaffold with TODO markers
# fill the TODOs, then:
edge-fix rules validate data/transforms/my-rule.json
# PASS requires: schema + semantic checks green, and the declared fixture
# dry-run reaches outcome `fixed` (metric improves; verification passes when
# the runners extra is installed).
```
