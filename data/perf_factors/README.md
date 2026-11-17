# perf_factors

**Status: promoted 2026-08-28**: the
schema is registered as `schemas/perf_factor.schema.json` at version 1.0, all
rows validate in CI (`tests/test_perf_factors.py`), and the dataset is reference
data like the matrix snapshots. It adds a third dimension the
matrix and the transforms do not carry: *both graph forms delegate, but one
is N× faster*. The transforms answer "how do I make this op run at all";
this dataset answers "which mathematically-equivalent form should I ship".

Origin: the 2026-08-24 S26 Hexagon factor experiments. Headline result:
swapping DINOv2-S's tanh-GELU decomposition for the builtin GELU op flipped it
from losing 0.63× to winning 1.32× against the Adreno GPU, at feature corr
0.999992 — one op choice decided the accelerator.

Format: one JSON per measured factor, `schemas/perf_factor.schema.json` (1.0).
Fields follow the repo's rules: every record dated,
device/runtime pinned, provenance `measured`, extent stated in `scope`.

Categories:
- `rewrite` — mathematically-equivalent op substitution (identity / bit-exact
  / approximation, with the equivalence evidence quoted)
- `shape` — same graph family, different tensor-shape choice (not equivalent;
  needs padding/masking or a resize to apply)
- `null-result` — substitutions measured to do nothing (these prevent wasted
  effort and are as load-bearing as the positive rows)

Rows to date: Galaxy S26 (SM8850, Hexagon v81), LiteRT 2.2.0 JIT path,
npubench `#sweep`, N=50 medians, thermal status NONE, 2026-08-24/28. GPU rows
where present are the same rig's Adreno via ML Drift (Mali re-checks noted
inline). Null factor marks a feasibility boundary (a form that does not
compile at all).
