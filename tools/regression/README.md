# Catalog regression

One litert-lm wheel, every own `.litertlm` chat bundle, the same functional
suite, one fresh process per check. It answers "does each bundle we ship still
do what a developer will ask of it on this wheel?" — a question a per-file ship
gate answers once, on the day of the ship, and never again.

## Suite v1

| check | backend | what is asked | PASS when |
|---|---|---|---|
| `describe` | – | `litert-lm describe`, `litert-lm-peek`, LlmMetadata, chat template, prefill signature lengths (no engine) | the file is readable |
| `plain_paris` | CPU | "What is the capital of France? Answer in one word." (cap 400) | the answer contains "paris" |
| `digit_default_1/2/3` | CPU | "What is 17 times 23?" (cap 400, no ThinkingConfig), three fresh processes | the final number is 391 and the answer is not degenerate |
| `digit_default_uncapped` | CPU | same, no `max_output_tokens` (the Engine's `max_num_tokens` bounds it) | same |
| `digit_think_off` | CPU | same, `ThinkingConfig(enable_thinking=False)` | same |
| `digit_think_on` | CPU | same, `ThinkingConfig(enable_thinking=True, thinking_token_budget=300)`, cap 700 (the thought budget + 400) | as the digit checks, on the content (not the thought channel) |
| `tool_call_auto` | CPU | "What is the weather in Paris right now?" with a `get_weather` tool, automatic tool calling | the runtime reports the call (a tool event, an execution or a tool call); an answer that names 18 and cloudy without one is not a call (`details.answer_has_tool_result` keeps that observation) |
| `engine_shared_2conv` | CPU | one Engine: a pirate conversation, closed, then a single-word conversation; control = the single-word conversation on a fresh Engine (cap 200 each) | both get paris / 391 / blue, the shared one carries no "Arr", and it obeys "Answer with a single word." where the control does (FAIL `state_leak` when the control keeps all three answers to 2 words and the shared one does not on 2 or more) |
| `two_turn_memory` | CPU | "My name is Kenji." then "What is my name?" (cap 100) | the second answer contains "kenji" |
| `digit_gpu_default` | GPU | the digit question on `GPU()` (cap 400) | as the digit checks |
| `cpu_gpu_match` | – | derived from `digit_default_1` and `digit_gpu_default` | both PASS (one PASS = FAIL; no answer or both FAIL = NA) |

The final number of a digit answer is its last integer of 3+ digits
(thousands commas dropped, LaTeX read as text): an answer that passes 391 on
the way to 791 is `wrong_answer`. Degenerate = one character 30+ times in a
row, a 12+ character span repeated back to back 4+ times, or the cap reached:
an output token count of at least the cap - 10 where one can be read (the API
reports no output token count unless benchmark mode is on, which this suite
never turns on; the probe reads it as the turn's token-count growth minus the
prompt's, measured by asking the same question again with a 1-token cap), else
1200+ characters. A readable count decides on its own: a long answer under
the cap is not `cap_hit`. A
word, for the single-word rule, is a run of `[A-Za-z0-9']`. A tool call that
comes back as text instead of a call (`<tool_call>`, `<function`,
`<|tool_call_start|>`, `<|tool_call_end|>`, `[get_weather(`,
`"name": "get_weather"`) is `raw_tag`; any other answer without a call is
`not_called`. An answer with no text (for `tool_call_auto`, no call either)
whose thought channel holds text is `cap_hit` (`details.thought_only`) when its
output token count is at least the cap - 10, or, where no token count can be
read (`plain_paris`, `tool_call_auto` and `two_turn_memory` measure no prompt),
when the thought runs 3 characters per token of the cap (1200 for a 400-token
cap, 300 for `two_turn_memory`'s 100); otherwise it stays `empty` /
`not_called`. The sampler is
never passed: the bundle's default is what an app gets. The Engine gets
`max_num_tokens = min(1024, Max Context Tokens)` and an engine `cache_dir`
under the work directory. Timeouts: 900 s per CPU process,
1800 s per GPU process, counted from the probe's start. Every model run goes
through `quiet_wait.py`, one process at a time.

## Roles

Suite v1 asks chat questions. `roles.py` names the repos whose model is not a
chat model (classifier, ocr, translate, transform, decision) or runs on an NPU
only (npu-only), quoting each model card; every `bundles.json` row carries its
repo's `role` (else `chat`). Those bundles still run every check; `SUMMARY.md`
counts them apart (chat-only totals in the header, `(<role>)` after the bundle
id, FAIL and NA subtotals by role). Records are not changed.

## A new wheel

```sh
/opt/homebrew/opt/python@3.14/bin/python3.14 -m venv ~/venvs/lt<ver>run && ~/venvs/lt<ver>run/bin/pip install litert-lm==<ver>
uv run python tools/regression/list_bundles.py --check data/regression/bundles.json   # exit 1: regenerate the list first
uv run python tools/regression/run_regression.py --wheel <ver> --python ~/venvs/lt<ver>run/bin/python \
    --bundles data/regression/bundles.json --out data/regression/<ver>/$(date +%F) --work <scratch dir> \
    --retry-fails [--order-first '<regex>']
uv run python tools/regression/summarize.py data/regression/<ver>/<date> --prev data/regression/<previous wheel>/<its latest date>
```

- `--retry-fails` runs every FAIL check (not `describe`, not `cpu_gpu_match`)
  once more right after the bundle's checks, under the same conditions; the
  new record replaces the old, which stays in `details.previous_attempts`
  (status, reason, raw_head, elapsed_s, run_at, details). The default sampler
  is not greedy, so one answer can differ from the next: `SUMMARY.md` marks a
  cell whose status changed with `*` and lists them under Flaky.
- `--order-first <regex>` runs the rows whose `bundle_id hf_repo file`
  matches (case-insensitive) first; each group runs smallest file first.
- The same command resumes a stopped run: a bundle whose file in `--out`
  holds all 13 checks with no NA "not run" is skipped (unless `--only` names
  it). `--out` keeps its date across midnight; `notes` carry the wall clock.
- Two timeouts in one bundle record its remaining checks as NA and the run
  moves on. Exit codes: 0 done; 3 = 10 bundles in a row created no engine;
  4 = the free disk would fall under the file's size + 30 GB; 5 = 5 bundles
  in a row failed to download; 6 = quiet_wait gave up on 3 checks. Without
  `--only` the code is also written to `<work>/logs/full_run.done`.
  `SUMMARY.md` is regenerated after every bundle.

`--check` re-reads the Hub for every listed row and exits 1 when a row's
`hub_commit`, `sha256` or size changed or an own-org repo gained a
`.litertlm`; a file added to a litert-community repo needs the per-file
ownership check, so only a regeneration finds it. `list_bundles.py` (no
`--check`) regenerates `data/regression/bundles.json` from the shipping
manifests, the lab cards and the Hub API; a file that left the Hub keeps its
row as `hub_missing`, and every row keeps the `bundle_id` it had. A Hub token
(`HF_TOKEN`, else `$HF_HOME/token`) opens the commit history of gated repos and
the downloads of their files; it is sent from a header file only (never on a
command line, in a log or in the list), and private repos it can see are
skipped.

When a hazard row is added (a new runtime or device hazard), re-run only what
it touches:

```sh
uv run python tools/regression/run_regression.py ... --only <bundle_id> ... --checks <check_id> ...
```

Selected checks replace their records in the day's `<bundle_id>.json`; the
others are kept, and `SUMMARY.md` is regenerated.

When the judge changes, re-judge a recorded run instead of running it again
(no model runs):

```sh
uv run python tools/regression/run_regression.py --rejudge data/regression/<ver>/<date> \
    --raw <work>/run [--previous-judge <label of the judge that wrote it>]
```

Every record is judged again from the probe JSON it was judged from (paired
by its `.started` marker and `details.run_at`). A record whose status or
reason changes keeps raw_head, elapsed_s and its other details, gets the new
pair and the new verdict's `thought_only` / `thought_head` / `cap_hit_by` (a
key the new verdict does not write goes: a PASS keeps no `cap_hit_by`), and
carries the old pair in `details.previous_judge`; `cpu_gpu_match` is
derived again, the summary recounted, `notes` gain `rejudged <date>: n
records relabelled`, and `SUMMARY.md` is regenerated. Unchanged records and
files stay byte for byte; records without a probe JSON (describe,
cpu_gpu_match, NA "not run", a crash) are left as they are. Each attempt under
`previous_attempts` is judged again from the JSON its re-run stashed
(`<check_id>.a<n>.json`; the stash keeps no start marker, so the pair is
checked by the attempt's log name and raw_head) and, when its verdict changes,
carries its own `details.previous_judge`; a flaky mark then compares two
verdicts of the same judge, and `notes` add `, m earlier attempts`.

## Data

- `data/regression/bundles.json` — the own-bundle list: live Hub `hub_commit` /
  LFS `sha256` / `size_bytes` per row, kind (chat / chat+vision → suite v1;
  chat+audio / other → v2), role (Roles above), the lab card that names the
  file, the evidence.
- `data/regression/<wheel>/<YYYY-MM-DD>/<bundle_id>.json` — one result per
  bundle (`schemas/regression_result.schema.json`), and the generated
  `SUMMARY.md` of that directory (never hand-edited).
- Logs stay in the work directory: `<work>/logs/<bundle_id>/<check_id>.log`,
  and so do the probe JSONs `--rejudge` reads:
  `<work>/run/<bundle_id>/<check_id>.json` (the first attempt of a re-run
  check: `<check_id>.a1.json`).

Downloads (`<work>/dl/`) and engine caches (`<work>/cache/<bundle_id>/`) are
deleted after each bundle unless `--keep`. HF cache blobs and `--extra-local`
copies are read, never written, moved or deleted.

These are model-level results: op-level matrix entries are never written from
them.
