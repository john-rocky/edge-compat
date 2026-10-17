"""Catalog regression: one litert-lm wheel x own `.litertlm` bundles x suite v1.

    uv run python tools/regression/run_regression.py --wheel 0.18.0 \\
        --python ~/venvs/lt0180run/bin/python --bundles data/regression/bundles.json \\
        [--only <bundle_id> ...] [--checks <check_id> ...] \\
        [--order-first <regex>] [--retry-fails] \\
        --out data/regression/0.18.0/<YYYY-MM-DD> --work <scratch dir> \\
        [--extra-local <dir> ...] [--keep]
    uv run python tools/regression/run_regression.py --rejudge <out dir> --raw <work>/run \\
        [--previous-judge r2]

Stdlib only. For each bundle: find the file (HF cache blob, then an
`--extra-local` copy, then a download from `resolve/<hub_commit>`), check its
sha256 against the list (a mismatch runs nothing and is recorded), then run
every check in its own process,

    <python> quiet_wait.py --max-wait 3600 -- <wheel python> -I probe_one.py ...

one process at a time, killed after 900 s (CPU) / 1800 s (GPU) counted from
the probe's start. The probe records observations; the judge is here. The
result is `<out>/<bundle_id>.json` (schemas/regression_result.schema.json),
canonical JSON dated by the `<out>` directory (a run that crosses midnight
keeps writing there; the wall-clock times are in `notes`); `SUMMARY.md` there
is regenerated after every bundle. Logs go to
`<work>/logs/<bundle_id>/<check_id>.log`.

`--retry-fails` runs every FAIL check (not describe, not cpu_gpu_match) once
more right after the bundle's checks, under the same conditions: the new
record replaces the old one, which is kept in `details.previous_attempts`.
`--order-first` runs the rows whose "bundle_id hf_repo file" matches the
regex (case-insensitive) first; each group runs smallest file first. A bundle
whose file in `<out>` already holds all 13 checks with no NA "not run" is
skipped unless `--only` names it, so the same command resumes a stopped run.

Two timeouts in a bundle record its remaining checks as NA and the run moves
on. The run stops with an exit code when 10 bundles in a row created no
engine (3), the free disk would fall under the file's size + 30 GB (4),
5 bundles in a row failed to download (5), or quiet_wait gave up on 3 checks
(6). Without `--only`, `<work>/logs/full_run.done` receives the exit code.

Downloads and engine caches are deleted after the bundle unless `--keep`;
HF cache blobs and `--extra-local` files are only ever read. A Hub token
(`HF_TOKEN`, else `$HF_HOME/token`) goes with Hub downloads (gated repos)
from a header file, never on a command line or in a log.

`--rejudge` runs no model: it re-judges every record of `<out dir>` from the
probe JSONs the run left in `<work>/run/<bundle_id>/` with the judge as it is
now. A record whose status or reason changes gets them replaced and keeps the
old pair in `details.previous_judge`; `cpu_gpu_match` is derived again, the
summary recounted, the file's notes say how many records changed, and
`SUMMARY.md` is regenerated. Unchanged records and files stay byte for byte.
Records with no probe JSON (describe, cpu_gpu_match, NA "not run", a crash)
are left as they are. Each attempt under `previous_attempts` is judged again
from the JSON its re-run stashed (`<check_id>.a<n>.json`) when there is one,
so a flaky mark compares verdicts of the same judge.

Model-level behavior only: nothing here reads or writes `data/matrix`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_one  # noqa: E402 - stdlib-only at import time; the suite table lives there
import summarize  # noqa: E402 - stdlib only; SUMMARY.md after every bundle

SUITE_V1 = probe_one.SUITE_V1
CHECKS = probe_one.CHECKS
PROBE = HERE / "probe_one.py"
HUB = "https://huggingface.co"
QUIET_WAIT = os.environ.get("QUIET_WAIT", "~/code/standup/tools/quiet/quiet_wait.py")
QUIET_MAX_WAIT_S = 3600
TIMEOUT_S = {"cpu": 900, "gpu": 1800}
MIN_FREE_BYTES = 30 * 10**9
SLOW_DOWNLOAD_BPS = 5 * 10**6
PARALLEL_PARTS = 8
SCHEMA_VERSION = "1.0"
NO_RETRY = ("describe", "cpu_gpu_match")
TIMEOUTS_PER_BUNDLE = 2
STOP_NO_ENGINE_STREAK = 10
STOP_DOWNLOAD_FAIL_STREAK = 5
STOP_WAITED_OUT = 3
DONE_FILE = "full_run.done"

# Degenerate output: one character 30+ times in a row; a 12+ character
# substring repeated back to back 4+ times; or the cap reached: an output
# token count within CAP_MARGIN of the cap when it can be read (the probe
# measures the prompt's own token count; see output_tokens()), else 1200+
# characters.
RUN_OF_ONE_CHAR = re.compile(r"(.)\1{29,}", re.S)
REPEATED_SPAN = re.compile(r"(.{12,200}?)\1{3,}", re.S)
CAP_CHARS = 1200
CAP_MARGIN = 10
# An answer with no text whose thought ran into a cap no token count can
# confirm: a thought of THOUGHT_CHARS_PER_TOKEN characters per token of the cap.
THOUGHT_CHARS_PER_TOKEN = 3
PIRATE = re.compile(r"\barr+\b", re.I)
KEYS = ("paris", "391", "blue")
# The digit answer is the last integer of 3+ digits (thousands commas dropped).
INTEGER = re.compile(r"\d+(?:,\d{3})*")
EXPECTED = "391"
# SYS_B ("Answer with a single word."): an answer of at most 2 words obeys it.
WORD = re.compile(r"[A-Za-z0-9']+")
SYS_B_MAX_WORDS = 2
# Tool-call text the runtime returned as plain text instead of parsing it into
# a call: Hermes / Qwen <tool_call>, <function...>, LFM2 <|tool_call_start|>
# [get_weather(location="Paris")]<|tool_call_end|>, a bare JSON call.
RAW_TOOL_TAGS = ("<tool_call>", "<function", "<|tool_call_start|>", "<|tool_call_end|>",
                 "[get_weather(", '"name": "get_weather"')


# --- judge (pure functions; tests call them directly) ------------------------------


def degenerate(text: str, out_tokens: int | None = None, cap: int | None = None) -> str | None:
    """'loop' / 'cap_hit' / None. A readable token count decides the cap on
    its own; the length (CAP_CHARS) only stands in when there is none."""
    if RUN_OF_ONE_CHAR.search(text) or REPEATED_SPAN.search(text):
        return "loop"
    if out_tokens is not None and cap is not None:
        return "cap_hit" if out_tokens >= cap - CAP_MARGIN else None
    return "cap_hit" if len(text) >= CAP_CHARS else None


def output_tokens(turn: dict[str, Any], probe_count: int | None) -> int | None:
    """Output tokens of a turn = its token-count growth minus the prompt's
    (probe_count = the count after the same prompt answered with 1 token)."""
    before, after = turn.get("token_count_before"), turn.get("token_count_after")
    if not all(isinstance(v, int) for v in (before, after, probe_count)):
        return None
    return after - before - (probe_count - 1)


def final_number(text: str) -> str | None:
    """The last integer of 3+ digits in the text, thousands commas dropped;
    LaTeX such as \\( 17 \\times 23 = 391 \\) is read as text."""
    found = [n.replace(",", "") for n in INTEGER.findall(text)]
    found = [n for n in found if len(n) >= 3]
    return found[-1] if found else None


def judge_digit(content: str, out_tokens: int | None = None,
                cap: int | None = None) -> tuple[str, str]:
    """PASS = not degenerate and the final number is 391 (an answer that
    passes 391 on the way to 791 is wrong_answer)."""
    if not content.strip():
        return "FAIL", "empty"
    bad = degenerate(content, out_tokens, cap)
    if bad:
        return "FAIL", bad
    if final_number(content) != EXPECTED:
        return "FAIL", "wrong_answer"
    return "PASS", ""


def judge_contains(content: str, key: str) -> tuple[str, str]:
    if not content.strip():
        return "FAIL", "empty"
    if key not in content.lower():
        return "FAIL", "wrong_answer"
    return "PASS", ""


def judge_tool_call(content: str, tool_calls: list[Any], events: list[Any],
                    executions: list[Any]) -> tuple[str, str]:
    """PASS only when the runtime reports the call (a tool event, an execution
    or a tool call): an answer can name 18 C and clouds without calling."""
    if events or executions or tool_calls:
        return "PASS", ""
    if any(tag in content for tag in RAW_TOOL_TAGS):
        return "FAIL", "raw_tag"
    return "FAIL", "not_called"


def thought_cap_hit(turn: dict[str, Any], out_tokens: int | None,
                    cap: int | None) -> str | None:
    """An answer with no text whose thought channel used up the cap: 'tokens'
    (the output token count within CAP_MARGIN of the cap), 'chars' (no token
    count and a thought of THOUGHT_CHARS_PER_TOKEN x cap characters, CAP_CHARS
    when the cap is unknown), or None (a plain empty)."""
    thought = (turn.get("channels") or {}).get("thought") or ""
    if (turn.get("content") or "").strip() or not thought:
        return None
    if out_tokens is not None and cap is not None:
        return "tokens" if out_tokens >= cap - CAP_MARGIN else None
    limit = THOUGHT_CHARS_PER_TOKEN * cap if cap is not None else CAP_CHARS
    return "chars" if len(thought) >= limit else None


def word_count(answer: str) -> int:
    return len(WORD.findall(answer))


def judge_shared(shared: list[str], control: list[str]) -> tuple[str, str, str]:
    """(status, reason, leak_kind). shared / control = the three conv2
    answers (SYS_B) on the shared Engine and on a fresh Engine."""
    if any(PIRATE.search(a) for a in shared):
        return "FAIL", "state_leak", "pirate_marker"
    for key, s_ans, c_ans in zip(KEYS, shared, control, strict=True):
        if key in c_ans.lower() and key not in s_ans.lower():
            return "FAIL", "state_leak", "key_missing"
    if all(word_count(a) <= SYS_B_MAX_WORDS for a in control) and sum(
            word_count(a) > SYS_B_MAX_WORDS for a in shared) >= 2:
        return "FAIL", "state_leak", "sysB_ignored"
    if not all(key in c_ans.lower() for key, c_ans in zip(KEYS, control, strict=True)):
        return "FAIL", "control_failed", ""
    return "PASS", "", ""


def judge_cpu_gpu(cpu: dict[str, Any], gpu: dict[str, Any]) -> tuple[str, str]:
    """digit_default_1 vs digit_gpu_default records."""
    def no_answer(rec: dict[str, Any]) -> bool:
        return rec["timeout"] or rec["reason"].startswith(
            ("error:", "engine_create_failed", "timeout", "not run"))

    for name, rec in (("cpu", cpu), ("gpu", gpu)):
        if rec["status"] == "NA" or no_answer(rec):
            return "NA", f"{name} gave no answer ({rec['reason'] or rec['status']})"
    if cpu["status"] == gpu["status"] == "PASS":
        return "PASS", ""
    if cpu["status"] == "PASS":
        return "FAIL", f"gpu failed ({gpu['reason']})"
    if gpu["status"] == "PASS":
        return "FAIL", f"cpu failed ({cpu['reason']})"
    return "NA", f"both failed (cpu {cpu['reason']}, gpu {gpu['reason']})"


def head(text: str, n: int = 160) -> str:
    return text[:n]


def synthetic_head(turn: dict[str, Any]) -> str:
    """raw_head for an answer with no text: say what came instead, in <>."""
    if turn.get("tool_calls"):
        return head("<no text; tool_calls " + json.dumps(turn["tool_calls"])[:120] + ">")
    thought = (turn.get("channels") or {}).get("thought")
    if thought:
        return head(f"<no text; thought channel {len(thought)} chars: {thought[:100]}>")
    return "<empty content>"


# --- retries, order, resume (pure; tests call them directly) -------------------------


def attempt_summary(rec: dict[str, Any]) -> dict[str, Any]:
    """One entry of details.previous_attempts: the replaced record."""
    details = {k: v for k, v in rec["details"].items()
               if k not in ("previous_attempts", "run_at")}
    return {"status": rec["status"], "reason": rec["reason"], "raw_head": rec["raw_head"],
            "elapsed_s": rec["elapsed_s"], "run_at": rec["details"].get("run_at"),
            "details": details}


def merge_retry(first: dict[str, Any], again: dict[str, Any]) -> dict[str, Any]:
    """The record to keep after a FAIL ran once more: the new one, carrying
    the first in details.previous_attempts; the first one when the retry
    never ran (quiet_wait gave up)."""
    if again["status"] == "NA" and again["reason"].startswith("not run"):
        first["details"]["retry_not_run"] = again["reason"]
        return first
    again["details"]["previous_attempts"] = [
        *first["details"].get("previous_attempts", []), attempt_summary(first)]
    return again


def order_rows(rows: list[dict[str, Any]], pattern: str | None) -> list[dict[str, Any]]:
    """Rows matching the pattern first, then the rest; each group smallest
    file first (ties by bundle_id). No pattern = the list's own order."""
    if not pattern:
        return list(rows)
    rx = re.compile(pattern, re.I)

    def size(r: dict[str, Any]) -> tuple[int, str]:
        return (r["size_bytes"] or 0, r["bundle_id"])

    first = {r["bundle_id"] for r in rows
             if rx.search(" ".join((r["bundle_id"], r["hf_repo"] or "", r["file"])))}
    return (sorted((r for r in rows if r["bundle_id"] in first), key=size)
            + sorted((r for r in rows if r["bundle_id"] not in first), key=size))


def bundle_complete(path: Path) -> bool:
    """The day's record holds all 13 checks and no NA "not run"."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    checks = {c.get("id"): c for c in doc.get("checks") or []}
    if set(checks) != set(SUITE_V1):
        return False
    return not any(c.get("status") == "NA" and "not run" in (c.get("reason") or "")
                   for c in checks.values())


def iso(epoch: float) -> str:
    text = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(epoch))
    return text[:-2] + ":" + text[-2:]


# --- process control ---------------------------------------------------------------


@dataclass
class ProbeRun:
    exit_code: int | None
    timed_out: bool
    elapsed_s: float
    data: dict[str, Any] | None
    log: Path
    waited_out: bool  # quiet_wait gave up: the probe never started
    run_at: str = ""  # wall clock of the probe's start (of the launch if it never started)


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.work = Path(args.work).expanduser().resolve()
        self.python = str(Path(args.python).expanduser())
        self.quiet_wait = str(Path(args.quiet_wait).expanduser())
        self.child: subprocess.Popen[bytes] | None = None
        self.waited_out_total = 0
        self.new_bundle()

    def new_bundle(self) -> None:
        self.timeouts_in_bundle = 0
        self.engine_attempts = 0  # probes that started and needed an engine
        self.engines_made = 0  # ... and got one

    def stop_child(self, signum: int, frame: Any) -> None:
        """SIGTERM / SIGINT: take the running probe's process group down too
        (it runs in its own session), then leave."""
        if self.child is not None and self.child.poll() is None:
            os.killpg(self.child.pid, signal.SIGKILL)
        raise SystemExit(128 + signum)

    def paths(self, bid: str, stem: str) -> tuple[Path, Path]:
        return self.work / "run" / bid / f"{stem}.json", self.work / "logs" / bid / f"{stem}.log"

    def stash(self, bid: str, check: str, n: int) -> None:
        """Keep attempt n's probe JSON and log (`<stem>.a<n>.json / .log`)
        before the same check runs again and overwrites them."""
        parts = CHECKS[check].get("parts") or (None,)
        for part in parts:
            stem = check + (f".{part}" if part else "")
            for path in self.paths(bid, stem):
                if path.exists():
                    path.replace(path.with_name(f"{stem}.a{n}{path.suffix}"))

    def probe(self, bid: str, model: Path, check: str, *, backend: str,
              ctx: int | None, part: str | None = None) -> ProbeRun:
        stem = check + (f".{part}" if part else "")
        out, log = self.paths(bid, stem)
        run_dir = out.parent
        run_dir.mkdir(parents=True, exist_ok=True)
        log.parent.mkdir(parents=True, exist_ok=True)
        started_marker = Path(str(out) + ".started")
        for stale in (out, started_marker):
            stale.unlink(missing_ok=True)
        cmd = [sys.executable, self.quiet_wait, "--max-wait", str(QUIET_MAX_WAIT_S), "--",
               self.python, "-I", str(PROBE), str(model), check, "--out", str(out),
               "--backend", backend, "--work", str(run_dir),
               "--cache-dir", str(self.work / "cache" / bid)]
        if ctx:
            cmd += ["--ctx", str(ctx)]
        if part:
            cmd += ["--part", part]
        limit = TIMEOUT_S[backend]
        launched = time.time()
        started: float | None = None  # wall clock the probe wrote at its start
        timed_out = False
        with open(log, "w", encoding="utf-8") as fh:
            fh.write("$ " + " ".join(cmd) + "\n")
            fh.flush()
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=run_dir,
                                    start_new_session=True)
            self.child = proc
            while proc.poll() is None:
                time.sleep(0.5)
                if started is None:
                    started = probe_start(started_marker)
                over = (started is not None and time.time() - started > limit) or (
                    started is None and time.time() - launched > QUIET_MAX_WAIT_S + 300)
                if over:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                    timed_out = started is not None
                    break
            self.child = None
        ended = time.time()
        if started is None:
            started = probe_start(started_marker)
        elapsed = round(ended - started, 1) if started is not None else 0.0
        data = None
        if out.exists():
            try:
                data = json.loads(out.read_text(encoding="utf-8"))
            except ValueError:
                data = None
        self.timeouts_in_bundle += timed_out
        if started is None:
            self.waited_out_total += 1
        elif check != "describe":
            self.engine_attempts += 1
            self.engines_made += bool(data and data.get("engine_s") is not None)
        return ProbeRun(proc.returncode, timed_out, elapsed, data, log,
                        waited_out=started is None, run_at=iso(started or launched))


def probe_start(marker: Path) -> float | None:
    try:
        return float(marker.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def log_tail(path: Path, n: int = 160) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    return " ".join(text[-n * 2:].split())[-n:]


# --- records -------------------------------------------------------------------------


def base_record(check: str) -> dict[str, Any]:
    spec = CHECKS[check]
    thinking = None
    if spec["thinking"] is not None:
        thinking = {"enable_thinking": True, "thinking_token_budget": -1, **spec["thinking"]}
    return {
        "id": check, "backend": spec["backend"], "status": "NA", "reason": "", "raw_head": "",
        "elapsed_s": 0.0, "exit_code": None, "timeout": False,
        "max_output_tokens": spec["cap"], "thinking_config": thinking, "details": {},
    }


def not_run(check: str, why: str) -> dict[str, Any]:
    rec = base_record(check)
    rec["reason"] = f"not run: {why}"
    return rec


def failed_run(rec: dict[str, Any], run: ProbeRun, backend: str | None) -> bool:
    """Fill rec for a run that produced no usable observation; True if so."""
    rec["exit_code"] = run.exit_code
    rec["elapsed_s"] = run.elapsed_s
    rec["timeout"] = run.timed_out
    rec["details"]["log"] = run.log.name
    rec["details"]["run_at"] = run.run_at
    if run.waited_out:
        rec.update(status="NA", reason="not run: measurement window held past quiet_wait "
                   f"--max-wait {QUIET_MAX_WAIT_S} s", raw_head="")
        return True
    if run.timed_out:
        limit = TIMEOUT_S[backend or "cpu"]
        rec.update(status="FAIL", reason=f"timeout {limit}",
                   raw_head=head(f"<killed after {limit} s> " + log_tail(run.log, 120)))
        return True
    if run.data is None:
        rec.update(status="FAIL", reason=f"error:crash(exit {run.exit_code})",
                   raw_head=head(log_tail(run.log) or f"<no output, exit {run.exit_code}>"))
        return True
    if run.data.get("error"):
        err = run.data["error"]
        stage = run.data.get("stage", "")
        rec["details"]["error_stage"] = stage
        if stage == "engine" and backend == "gpu":
            rec["reason"] = f"engine_create_failed:{err[:120]}"
        else:
            rec["reason"] = f"error:{run.data.get('error_type') or 'Exception'}"
        rec.update(status="FAIL", raw_head=head(err))
        return True
    return False


def turn_details(turn: dict[str, Any]) -> dict[str, Any]:
    content = turn.get("content") or ""
    before, after = turn.get("token_count_before"), turn.get("token_count_after")
    out = {
        "chars": len(content),
        "final_number": final_number(content),
        "turn_s": turn.get("elapsed_s"),
        "channels": sorted((turn.get("channels") or {}).keys()),
    }
    if isinstance(before, int) and isinstance(after, int):
        out["token_count_delta"] = after - before  # prompt + output of this turn
    return out


def judge_chat(check: str, run: ProbeRun) -> dict[str, Any]:
    rec = base_record(check)
    if failed_run(rec, run, CHECKS[check]["backend"]):
        return rec
    data = run.data or {}
    rec["details"]["engine_s"] = data.get("engine_s")
    rec["details"]["max_num_tokens"] = data.get("max_num_tokens")
    turns = data.get("turns") or []
    last = turns[-1] if turns else {}
    content = last.get("content") or ""
    probe_count = data.get("prompt_probe_token_count")  # digit checks only
    out_tokens = output_tokens(last, probe_count)
    cap = data.get("max_output_tokens")
    if check == "plain_paris":
        status, reason = judge_contains(content, "paris")
    elif check == "two_turn_memory":
        status, reason = judge_contains(content, "kenji")
        rec["details"]["first_answer_head"] = head((turns[0].get("content") or ""), 120)
    elif check == "tool_call_auto":
        events = data.get("tool_events") or []
        executions = data.get("tool_executions") or []
        status, reason = judge_tool_call(content, last.get("tool_calls") or [], events,
                                         executions)
        rec["details"].update(tool_events=len(events), tool_executions=executions,
                              tool_calls_in_reply=last.get("tool_calls") or [],
                              answer_has_tool_result="18" in content
                              and "cloudy" in content.lower())
    else:  # digit_*
        if cap is None and isinstance(data.get("max_num_tokens"), int) \
                and isinstance(probe_count, int):
            cap = data["max_num_tokens"] - (probe_count - 1)  # the Engine bounds it
        status, reason = judge_digit(content, out_tokens, cap)
        rec["details"].update(output_tokens=out_tokens, cap_tokens=cap,
                              prompt_tokens=None if probe_count is None else probe_count - 1,
                              contains_391=EXPECTED in content)
        if reason == "cap_hit":
            rec["details"]["cap_hit_by"] = (
                "tokens" if out_tokens is not None and cap is not None
                and out_tokens >= cap - CAP_MARGIN else "chars")
        if data.get("prompt_probe_error"):
            rec["details"]["prompt_probe_error"] = data["prompt_probe_error"]
        if check == "digit_think_on":
            thought = (last.get("channels") or {}).get("thought")
            rec["details"]["thought_channel_present"] = thought is not None
            rec["details"]["thought_head"] = head(thought or "")
    no_text = reason == "empty" or (check == "tool_call_auto" and reason == "not_called")
    by = thought_cap_hit(last, out_tokens, cap) if no_text else None
    if by:  # no text (and no call), only a thought that ran into the cap
        reason = "cap_hit"
        rec["details"].update(thought_only=True, cap_hit_by=by,
                              thought_head=head(last["channels"]["thought"]))
    rec["details"].update(turn_details(last))
    rec["max_output_tokens"] = data.get("max_output_tokens", rec["max_output_tokens"])
    rec["thinking_config"] = data.get("thinking_config", rec["thinking_config"])
    rec.update(status=status, reason=reason,
               raw_head=head(content) if content else synthetic_head(last),
               exit_code=run.exit_code, elapsed_s=run.elapsed_s, timeout=run.timed_out)
    return rec


def judge_shared_runs(shared: ProbeRun, control: ProbeRun) -> dict[str, Any]:
    rec = base_record("engine_shared_2conv")
    rec["details"]["parts"] = {}
    for name, run in (("shared", shared), ("control", control)):
        part = base_record("engine_shared_2conv")
        if failed_run(part, run, "cpu"):
            rec.update(status=part["status"], reason=part["reason"],
                       raw_head=head(f"{name}: {part['raw_head']}"),
                       exit_code=run.exit_code, timeout=run.timed_out,
                       elapsed_s=round(shared.elapsed_s + control.elapsed_s, 1))
            rec["details"].update(failed_part=name, run_at=shared.run_at)
            return rec
    answers: dict[str, list[str]] = {}
    for name, run in (("shared", shared), ("control", control)):
        for turn in (run.data or {}).get("turns") or []:
            label = f"{name}_{turn['conv']}"
            answers.setdefault(label, []).append(turn.get("content") or "")
    shared_b = answers.get("shared_conv2", [])
    control_b = answers.get("control_conv2", [])
    leak_kind = ""
    if len(shared_b) != 3 or len(control_b) != 3:
        status, reason = "FAIL", "error:incomplete_turns"
    else:
        status, reason, leak_kind = judge_shared(shared_b, control_b)
    rec["details"]["answers"] = {k: [head(a, 120) for a in v] for k, v in sorted(answers.items())}
    rec["details"]["word_counts"] = {k: [word_count(a) for a in v]
                                     for k, v in sorted(answers.items())}
    if leak_kind:
        rec["details"]["leak_kind"] = leak_kind
    rec["details"]["pirate_marker_in_shared_conv2"] = any(PIRATE.search(a) for a in shared_b)
    rec["details"]["shared_equals_control"] = [
        " ".join(a.lower().split()) == " ".join(b.lower().split())
        for a, b in zip(shared_b, control_b, strict=False)]
    rec.update(status=status, reason=reason,
               raw_head=head(" | ".join(" ".join(a.split()) for a in shared_b)) or
               "<empty content>",
               exit_code=max(shared.exit_code or 0, control.exit_code or 0),
               elapsed_s=round(shared.elapsed_s + control.elapsed_s, 1), timeout=False)
    rec["details"]["parts"] = {"shared_s": shared.elapsed_s, "control_s": control.elapsed_s}
    rec["details"]["run_at"] = shared.run_at
    return rec


def judge_describe(run: ProbeRun) -> tuple[dict[str, Any], dict[str, Any]]:
    rec = base_record("describe")
    if failed_run(rec, run, "cpu"):
        return rec, {}
    data = run.data or {}
    describe = data.get("describe") or {}
    readable = data.get("describe_exit") == 0 and isinstance(describe.get("sections"), list)
    stdout = describe.get("describe_stdout") or ""
    lines = [ln.strip() for ln in stdout.splitlines()
             if ln.strip() and not ln.startswith(("=", "File:")) and "Model Info Report" not in ln]
    rec.update(status="PASS" if readable else "FAIL",
               reason="" if readable else "unreadable",
               raw_head=head(" | ".join(lines)) or head(data.get("describe_stderr_tail") or
                                                         "<no describe output>"),
               exit_code=run.exit_code, elapsed_s=run.elapsed_s)
    rec["details"] = {"describe_exit": data.get("describe_exit"),
                      "peek_exit": data.get("peek_exit"),
                      "max_context_tokens": describe.get("max_context_tokens"),
                      "run_at": run.run_at}
    return rec, describe


def declared_backends(describe: dict[str, Any]) -> set[str] | None:
    """Backends `litert-lm describe` lists for text ("Text Backends: CPU GPU"),
    or None when it prints no such line."""
    m = re.search(r"^\s*Text Backends:\s*(.+)$", describe.get("describe_stdout") or "", re.M)
    return set(m.group(1).split()) if m else None


def derive_cpu_gpu(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rec = base_record("cpu_gpu_match")
    cpu, gpu = records.get("digit_default_1"), records.get("digit_gpu_default")
    if cpu is None or gpu is None:
        rec.update(status="NA", reason="not run: digit_default_1 or digit_gpu_default missing")
        return rec
    status, reason = judge_cpu_gpu(cpu, gpu)
    rec.update(status=status, reason=reason,
               raw_head=head(f"cpu {cpu['status']}: {cpu['raw_head'][:70]} | "
                             f"gpu {gpu['status']}: {gpu['raw_head'][:70]}"),
               elapsed_s=0.0)
    rec["details"] = {"cpu": f"{cpu['status']} {cpu['reason']}".strip(),
                      "gpu": f"{gpu['status']} {gpu['reason']}".strip()}
    return rec


# --- files ---------------------------------------------------------------------------


def _home(path: Path | str) -> str:
    """A path as written into data: the home directory as `~`."""
    text, home = str(path), str(Path.home())
    return "~" + text[len(home):] if text == home or text.startswith(home + os.sep) else text


def sha256_of(path: Path) -> str:
    out = subprocess.run(["shasum", "-a", "256", str(path)], capture_output=True, text=True,
                         check=True).stdout
    return out.split()[0]


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def hub_token() -> str | None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    try:
        return (home / "token").read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def download(row: dict[str, Any], dl_dir: Path, log) -> Path:
    """resolve/<hub_commit>/<file> into dl_dir/<sha256>.litertlm; one
    connection, switching to PARALLEL_PARTS byte ranges when the first 30 s
    run below SLOW_DOWNLOAD_BPS. The token, if any, rides in a header file
    (curl drops it on the redirect to the storage host)."""
    dl_dir.mkdir(parents=True, exist_ok=True)
    final = dl_dir / f"{row['sha256']}.litertlm"
    if final.exists() and final.stat().st_size == row["size_bytes"]:
        return final
    url = (f"{HUB}/{row['hf_repo']}/resolve/{row['hub_commit']}/"
           f"{urllib.parse.quote(row['file'])}")
    env = dict(os.environ, HF_HUB_DISABLE_XET="1")
    with tempfile.TemporaryDirectory() as tmp:
        auth: list[str] = []
        token = hub_token()
        if token:
            header = Path(tmp) / "auth"
            header.touch(mode=0o600)
            header.write_text(f"Authorization: Bearer {token}\n", encoding="utf-8")
            auth = ["-H", f"@{header}"]
        part = dl_dir / f"{row['sha256']}.litertlm.part"
        part.unlink(missing_ok=True)
        t0 = time.monotonic()
        proc = subprocess.Popen(["curl", "-fsSL", "--retry", "3", *auth, "-o", str(part), url],
                                env=env)
        checked = False
        while proc.poll() is None:
            time.sleep(2)
            if not checked and time.monotonic() - t0 >= 30:
                checked = True
                size = part.stat().st_size if part.exists() else 0
                rate = size / (time.monotonic() - t0)
                if rate < SLOW_DOWNLOAD_BPS:
                    log(f"  download {rate / 1e6:.2f} MB/s < {SLOW_DOWNLOAD_BPS / 1e6:.0f} MB/s: "
                        f"switching to {PARALLEL_PARTS} byte ranges")
                    proc.kill()
                    proc.wait()
                    part.unlink(missing_ok=True)
                    return download_ranges(url, row, dl_dir, env, final, auth)
        if proc.returncode != 0:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"curl exit {proc.returncode} for {url}")
        part.replace(final)
        return final


def download_ranges(url: str, row: dict[str, Any], dl_dir: Path, env: dict[str, str],
                    final: Path, auth: list[str]) -> Path:
    size = int(row["size_bytes"])
    chunk = -(-size // PARALLEL_PARTS)
    parts = []
    procs = []
    for i in range(PARALLEL_PARTS):
        start, end = i * chunk, min(size, (i + 1) * chunk) - 1
        if start > end:
            break
        path = dl_dir / f"{row['sha256']}.litertlm.part{i}"
        parts.append((path, end - start + 1))
        procs.append(subprocess.Popen(
            ["curl", "-fsSL", "--retry", "3", *auth, "-r", f"{start}-{end}", "-o", str(path),
             url], env=env))
    codes = [p.wait() for p in procs]
    bad = [i for i, ((path, n), code) in enumerate(zip(parts, codes, strict=True))
           if code != 0 or not path.exists() or path.stat().st_size != n]
    if bad:
        for path, _ in parts:
            path.unlink(missing_ok=True)
        raise RuntimeError(f"byte-range parts {bad} failed for {url}")
    tmp = Path(str(final) + ".part")
    with open(tmp, "wb") as out:
        for path, _ in parts:
            with open(path, "rb") as fh:
                shutil.copyfileobj(fh, out, 64 * 1024 * 1024)
            path.unlink()
    tmp.replace(final)
    return final


@dataclass
class Located:
    source: str
    path: Path | None
    downloaded: bool = False
    sha_note: str | None = None
    failure: str | None = None  # the download failed
    disk_stop: bool = False


def locate(runner: Runner, row: dict[str, Any], log) -> Located:
    want = row["sha256"]
    if row.get("local"):
        blob = Path(row["local"]).expanduser()
        if blob.exists():
            got = sha256_of(blob)
            if got == want:
                return Located("hf_cache", blob)
            log(f"  hf cache blob {blob.name[:12]}... hashes to {got[:12]}...; not used")
    for directory in runner.args.extra_local or []:
        candidate = Path(directory).expanduser() / row["file"]
        if candidate.is_file() and candidate.stat().st_size == row["size_bytes"]:
            if sha256_of(candidate) == want:
                return Located("extra_local", candidate)
            log(f"  {candidate} does not hash to the listed sha256; not used")
    need = int(row["size_bytes"]) + MIN_FREE_BYTES
    if free_bytes(runner.work) < need:
        log(f"STOP: {free_bytes(runner.work) / 1e9:.1f} GB free, need "
            f"{need / 1e9:.1f} GB to download {row['file']}")
        return Located("download", None, disk_stop=True)
    t0 = time.monotonic()
    try:
        path = download(row, runner.work / "dl", log)
    except (RuntimeError, OSError) as exc:
        log(f"  download failed after {time.monotonic() - t0:.0f} s: {exc}")
        return Located("download", None, failure=str(exc)[:160])
    secs = time.monotonic() - t0
    log(f"  downloaded {row['size_bytes'] / 1e9:.2f} GB in {secs:.0f} s "
        f"({row['size_bytes'] / 1e6 / max(secs, 1e-3):.1f} MB/s)")
    got = sha256_of(path)
    if got != want:
        return Located("download", path, True, (f"sha_mismatch: the downloaded file hashes to "
                                                f"{got}, the list says {want}"))
    return Located("download", path, True)


# --- the run -------------------------------------------------------------------------


def dumps(doc: Any) -> str:
    return json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def env_block(args: argparse.Namespace) -> dict[str, str]:
    def run_py(code: str) -> str:
        return subprocess.run([str(Path(args.python).expanduser()), "-I", "-c", code],
                              capture_output=True, text=True, check=True).stdout.strip()

    installed = run_py("import importlib.metadata as m; print(m.version('litert-lm'))")
    if installed != args.wheel:
        raise SystemExit(f"--wheel {args.wheel} but {args.python} has litert-lm {installed}")
    os_version = subprocess.run(["sw_vers", "-productVersion"], capture_output=True, text=True,
                                check=False).stdout.strip() or "unverified"
    return {
        "runtime": "litert-lm",
        "runtime_version": installed,
        "wheel_source": "PyPI",
        "python": run_py("import sys; print(sys.version.split()[0])"),
        "host": args.machine_label,
        "machine_label": args.machine_label,
        "os": os_version,
    }


def clock(fmt: str) -> str:
    return subprocess.run(["date", fmt], capture_output=True, text=True,
                          check=True).stdout.strip()


def summary_of(checks: list[dict[str, Any]]) -> dict[str, int]:
    return {"pass": sum(c["status"] == "PASS" for c in checks),
            "fail": sum(c["status"] == "FAIL" for c in checks),
            "na": sum(c["status"] == "NA" for c in checks)}


@dataclass
class BundleOutcome:
    code: int = 0  # 4 = disk stop: nothing written
    download_failed: bool = False
    engine_attempts: int = 0
    engines_made: int = 0


def run_check(runner: Runner, bid: str, link: Path, check: str, ctx: int | None,
              doc: dict[str, Any]) -> dict[str, Any]:
    backend = CHECKS[check]["backend"] or "cpu"
    if check == "describe":
        rec, describe = judge_describe(runner.probe(bid, link, check, backend="cpu", ctx=None))
        if describe:
            doc["describe"] = describe
        return rec
    if check == "engine_shared_2conv":
        shared = runner.probe(bid, link, check, backend=backend, ctx=ctx, part="shared")
        control = runner.probe(bid, link, check, backend=backend, ctx=ctx, part="control")
        return judge_shared_runs(shared, control)
    return judge_chat(check, runner.probe(bid, link, check, backend=backend, ctx=ctx))


def show(log, rec: dict[str, Any], tag: str = "") -> None:
    log(f"  {rec['id'] + tag:24s} {rec['status']:4s} {rec['reason'][:40]:40s} "
        f"{rec['elapsed_s']:7.1f}s  {' '.join(rec['raw_head'].split())[:60]}")


def run_bundle(runner: Runner, row: dict[str, Any], selected: list[str], out_dir: Path,
               env: dict[str, str], date: str, log) -> BundleOutcome:
    bid = row["bundle_id"]
    target = out_dir / f"{bid}.json"
    previous = json.loads(target.read_text(encoding="utf-8")) if target.exists() else None
    prev_records = {c["id"]: c for c in (previous or {}).get("checks", [])}
    records: dict[str, dict[str, Any]] = {}
    # The previous notes stay while some of its records do (a --checks re-run).
    keeps_some = any(c not in selected for c in SUITE_V1 if not CHECKS[c].get("derived"))
    notes: list[str] = ([previous["notes"]] if previous and previous.get("notes")
                        and keeps_some else [])
    t_start = clock("+%F %H:%M")
    runner.new_bundle()

    found = locate(runner, row, log)
    if found.disk_stop:
        return BundleOutcome(code=4)
    doc: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "bundle_id": bid, "hf_repo": row["hf_repo"],
        "hub_commit": row["hub_commit"], "artifact": row["file"],
        "artifact_sha256": row["sha256"], "size_bytes": row["size_bytes"],
        "source": found.source, "env": env, "date": date, "suite": "v1",
        "provenance": "measured", "describe": (previous or {}).get("describe", {}),
    }
    if found.failure or found.sha_note:
        if found.sha_note:
            notes.append(found.sha_note)
        else:
            notes.append(f"download failed: {found.failure}")
        for check in SUITE_V1:
            if found.sha_note:
                records[check] = base_record(check)
                records[check]["reason"] = "sha_mismatch"
            else:
                records[check] = not_run(check, f"download failed ({found.failure[:100]})")
        finish(doc, records, prev_records, notes, target, found, runner, [], t_start, [], log)
        return BundleOutcome(download_failed=found.failure is not None)

    model = found.path
    assert model is not None
    link_dir = runner.work / "links" / bid
    link_dir.mkdir(parents=True, exist_ok=True)
    link = link_dir / row["file"]
    link.unlink(missing_ok=True)
    link.symlink_to(model)

    ctx = None
    run_list = list(selected)
    if "describe" not in run_list:
        ctx = (doc["describe"] or {}).get("max_context_tokens")
        if not isinstance(ctx, int):
            run_list.insert(0, "describe")
    stopped = False
    gone = ""
    for check in SUITE_V1:
        if check not in run_list or CHECKS[check].get("derived"):
            continue
        if not gone and not link.exists():  # a read-only source deleted under the run
            gone = f"the model file went away ({_home(model)})"
            log(f"  {gone}: the rest is NA")
        if stopped or gone:
            records[check] = not_run(check, gone or f"{TIMEOUTS_PER_BUNDLE} timeouts in "
                                     "this bundle")
            show(log, records[check])
            continue
        declared = declared_backends(doc["describe"] or {})
        if check != "describe" and declared is not None and not declared & {"CPU", "GPU"}:
            # A bundle compiled for an NPU only: CPU and GPU checks would only
            # record that it is not a CPU / GPU bundle.
            records[check] = not_run(check, "the bundle declares Text Backends "
                                     + " ".join(sorted(declared)))
            show(log, records[check])
            continue
        records[check] = run_check(runner, bid, link, check, ctx, doc)
        if check == "describe":
            ctx = (doc["describe"] or {}).get("max_context_tokens")
            ctx = ctx if isinstance(ctx, int) else None
        show(log, records[check])
        if runner.timeouts_in_bundle >= TIMEOUTS_PER_BUNDLE:
            stopped = True
            log(f"  {TIMEOUTS_PER_BUNDLE} timeouts in bundle {bid}: the rest is NA")

    retried: list[str] = []
    if runner.args.retry_fails and not stopped and not gone:
        for check in SUITE_V1:
            first = records.get(check)
            if check in NO_RETRY or first is None or first["status"] != "FAIL":
                continue
            if runner.timeouts_in_bundle >= TIMEOUTS_PER_BUNDLE:
                first["details"]["retry_skipped"] = (f"{TIMEOUTS_PER_BUNDLE} timeouts in this "
                                                     "bundle")
                continue
            if not link.exists():
                first["details"]["retry_skipped"] = "the model file went away"
                continue
            runner.stash(bid, check, 1)
            if "log" in first["details"]:
                first["details"]["log"] = first["details"]["log"].replace(".log", ".a1.log")
            first_label = f"{first['status']} {first['reason']}"
            again = run_check(runner, bid, link, check, ctx, doc)
            show(log, again, " (re-run)")
            records[check] = merge_retry(first, again)
            retried.append(f"{check} ({first_label} -> {again['status']} "
                           f"{again['reason']})".replace(" )", ")"))
    finish(doc, records, prev_records, notes, target, found, runner, selected, t_start,
           retried, log)
    return BundleOutcome(engine_attempts=runner.engine_attempts,
                         engines_made=runner.engines_made)


def finish(doc: dict[str, Any], records: dict[str, dict[str, Any]],
           prev_records: dict[str, dict[str, Any]], notes: list[str], target: Path,
           found: Located, runner: Runner, selected: list[str], t_start: str,
           retried: list[str], log) -> None:
    for check in SUITE_V1:
        if check in records or CHECKS[check].get("derived"):
            continue
        if check in prev_records:
            records[check] = prev_records[check]
        else:
            records[check] = not_run(check, "not selected by --checks")
    if "cpu_gpu_match" not in records:
        records["cpu_gpu_match"] = derive_cpu_gpu(records)
    checks = [records[c] for c in SUITE_V1]
    if not selected:
        ran = "nothing run (" + records["describe"]["reason"] + ")"
    else:
        ran = ", ".join(c for c in SUITE_V1 if c in selected or c == "cpu_gpu_match")
    end = clock("+%F %H:%M")
    span = f"{t_start}-{end[11:] if end[:10] == t_start[:10] else end} {clock('+%Z')}"
    notes.append(f"run {span}: {ran} from {doc['source']}"
                 + (f"; re-run once: {', '.join(retried)}" if retried else ""))
    doc.update(checks=checks, summary=summary_of(checks), notes="; ".join(notes))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dumps(doc), encoding="utf-8")
    log(f"  -> {target}  {doc['summary']}")
    (target.parent / "SUMMARY.md").write_text(summarize.render(target.parent), encoding="utf-8")
    if not runner.args.keep:
        if found.downloaded and found.path is not None:
            found.path.unlink(missing_ok=True)
        shutil.rmtree(runner.work / "cache" / doc["bundle_id"], ignore_errors=True)


def run_all(args: argparse.Namespace, runner: Runner, log) -> int:
    out_dir = Path(args.out)
    date = out_dir.name
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) or out_dir.parent.name != args.wheel \
            or date > clock("+%F"):
        raise SystemExit(f"--out must end in <wheel>/<YYYY-MM-DD> (today or earlier) = "
                         f"{args.wheel}/<date>")
    env = env_block(args)
    rows = json.loads(Path(args.bundles).read_text(encoding="utf-8"))["bundles"]
    by_id = {r["bundle_id"]: r for r in rows}
    if args.only:
        unknown = [b for b in args.only if b not in by_id]
        if unknown:
            raise SystemExit(f"unknown bundle_id(s): {unknown}")
        rows = [by_id[b] for b in args.only]
    todo = [r for r in rows if r["status"] == "ok" and r["suite"] == "v1"]
    for r in rows:
        if r not in todo:
            print(f"skip {r['bundle_id']}: status {r['status']}, suite {r['suite']}")
    todo = order_rows(todo, args.order_first)
    selected = [c for c in SUITE_V1 if not CHECKS[c].get("derived")]
    if args.checks:
        selected = [c for c in args.checks if not CHECKS[c].get("derived")]
    for sub in ("run", "logs", "cache", "links", "dl"):
        (runner.work / sub).mkdir(parents=True, exist_ok=True)

    log(f"wheel {env['runtime_version']} python {env['python']} on {env['host']} "
        f"macOS {env['os']}; {len(todo)} bundle(s); checks {', '.join(selected)}"
        + ("; FAIL checks re-run once" if args.retry_fails else ""))
    no_engine = download_fails = 0
    for i, row in enumerate(todo, 1):
        bid = row["bundle_id"]
        if not args.only and bundle_complete(out_dir / f"{bid}.json"):
            log(f"[{i}/{len(todo)}] {bid}: complete in {out_dir}; skipped")
            continue
        log(f"[{i}/{len(todo)}] {bid}  ({row['hf_repo']}/{row['file']}, "
            f"{row['size_bytes'] / 1e9:.2f} GB)")
        outcome = run_bundle(runner, row, selected, out_dir, env, date, log)
        if outcome.code:
            return outcome.code
        download_fails = download_fails + 1 if outcome.download_failed else 0
        if outcome.engine_attempts:
            no_engine = 0 if outcome.engines_made else no_engine + 1
        if download_fails >= STOP_DOWNLOAD_FAIL_STREAK:
            log(f"STOP: {download_fails} bundles in a row failed to download")
            return 5
        if no_engine >= STOP_NO_ENGINE_STREAK:
            log(f"STOP: {no_engine} bundles in a row created no engine")
            return 3
        if runner.waited_out_total >= STOP_WAITED_OUT:
            log(f"STOP: quiet_wait gave up on {runner.waited_out_total} checks")
            return 6
    return 0


# --- re-judging a recorded run (no model runs) ---------------------------------------


def rejudged(rec: dict[str, Any], raw_dir: Path) -> dict[str, Any] | None:
    """The record judged again from the probe JSON(s) it was judged from, or
    None when there is none to read: describe and cpu_gpu_match, NA "not run",
    a crash or a kill (no JSON), a re-run that never started (the kept first
    attempt's .a1.json has no start marker to pair), or a JSON of another
    attempt (its .started marker disagrees with details.run_at)."""
    check = rec["id"]
    if check in ("describe", "cpu_gpu_match") or rec["reason"].startswith("not run") \
            or "retry_not_run" in rec["details"]:
        return None
    runs = []
    for part in CHECKS[check].get("parts") or (None,):
        path = raw_dir / (check + (f".{part}" if part else "") + ".json")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        started = probe_start(Path(str(path) + ".started"))
        runs.append(ProbeRun(rec["exit_code"], rec["timeout"], rec["elapsed_s"], data,
                             Path(rec["details"].get("log") or "-"), waited_out=False,
                             run_at=iso(started) if started is not None else ""))
    if runs[0].run_at != rec["details"].get("run_at"):
        return None
    if check == "engine_shared_2conv":
        return judge_shared_runs(*runs)
    return judge_chat(check, runs[0])


def rejudged_attempt(rec: dict[str, Any], n: int, attempt: dict[str, Any],
                     raw_dir: Path) -> dict[str, Any] | None:
    """Attempt n (1-based) of rec's previous_attempts judged again from the
    probe JSON(s) its re-run stashed (`<stem>.a<n>.json`), or None when there
    is none to read or it is not that attempt's. The stash keeps no start
    marker, so the pair is checked by the log name the attempt recorded
    (`<stem>.a<n>.log`) and by its raw_head, which no judge changes."""
    check = rec["id"]
    log = (attempt.get("details") or {}).get("log")
    if log and not log.endswith(f".a{n}.log"):
        return None
    runs = []
    for part in CHECKS[check].get("parts") or (None,):
        path = raw_dir / (check + (f".{part}" if part else "") + f".a{n}.json")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        runs.append(ProbeRun(None, attempt["reason"].startswith("timeout"),
                             attempt["elapsed_s"], data, Path(log or "-"), waited_out=False,
                             run_at=attempt.get("run_at") or ""))
    new = (judge_shared_runs(*runs) if check == "engine_shared_2conv"
           else judge_chat(check, runs[0]))
    return new if new["raw_head"] == attempt["raw_head"] else None


def take_verdict(target: dict[str, Any], details: dict[str, Any], new: dict[str, Any],
                 judge: str) -> None:
    """Give target (a record, or an attempt under previous_attempts) the new
    judge's status and reason, the old pair in details.previous_judge, and the
    details that belong to a verdict as the new judge writes them (a key it
    does not write goes: a PASS carries no cap_hit_by); raw_head, elapsed_s and
    the other details stay."""
    details["previous_judge"] = {"status": target["status"], "reason": target["reason"],
                                 "judge": judge}
    target.update(status=new["status"], reason=new["reason"])
    for key in ("thought_only", "thought_head", "cap_hit_by"):
        if key in new["details"]:
            details[key] = new["details"][key]
        else:
            details.pop(key, None)


def relabel(rec: dict[str, Any], new: dict[str, Any], judge: str) -> dict[str, Any]:
    """A copy of rec carrying the new verdict (take_verdict)."""
    out = json.loads(json.dumps(rec))
    take_verdict(out, out["details"], new, judge)
    return out


def rejudge_doc(doc: dict[str, Any], raw_dir: Path, judge: str,
                date: str) -> tuple[list[str], list[str]]:
    """Re-judge one result in place: one 'bundle check: OLD -> NEW' line per
    record whose status or reason changed, and one 'bundle check attempt n:
    OLD -> NEW' line per earlier attempt that did (none = doc untouched)."""
    changed: list[str] = []
    checks = doc["checks"]
    for i, rec in enumerate(checks):
        new = rejudged(rec, raw_dir)
        if new is not None and (new["status"], new["reason"]) != (rec["status"], rec["reason"]):
            checks[i] = relabel(rec, new, judge)
            changed.append(rec["id"])
    if {"digit_default_1", "digit_gpu_default"} & set(changed):
        i = SUITE_V1.index("cpu_gpu_match")
        old, new = checks[i], derive_cpu_gpu({c["id"]: c for c in checks})
        if (new["status"], new["reason"]) != (old["status"], old["reason"]):
            checks[i] = relabel(old, new, judge)
            checks[i]["raw_head"] = new["raw_head"]
            checks[i]["details"].update(new["details"])  # the cpu / gpu labels it reads
            changed.append("cpu_gpu_match")
    attempt_lines: list[str] = []
    for rec in checks:  # flaky marks compare the attempts with the record: same judge
        for n, attempt in enumerate(rec["details"].get("previous_attempts") or [], 1):
            new = rejudged_attempt(rec, n, attempt, raw_dir)
            if new is None or (new["status"], new["reason"]) == (attempt["status"],
                                                                 attempt["reason"]):
                continue
            old = summarize.label(attempt)
            take_verdict(attempt, attempt.setdefault("details", {}), new, judge)
            attempt_lines.append(f"{doc['bundle_id']} {rec['id']} attempt {n}: {old} -> "
                                 f"{summarize.label(attempt)}")
    if not changed and not attempt_lines:
        return [], []
    doc["summary"] = summary_of(checks)
    note = f"rejudged {date}: {len(changed)} records relabelled"
    if attempt_lines:
        note += f", {len(attempt_lines)} earlier attempts"
    doc["notes"] = "; ".join(filter(None, (doc.get("notes"), note)))
    by_id = {c["id"]: c for c in checks}
    return [f"{doc['bundle_id']} {c}: "
            f"{summarize.label(by_id[c]['details']['previous_judge'])} -> "
            f"{summarize.label(by_id[c])}" for c in SUITE_V1 if c in changed], attempt_lines


def rejudge(out_dir: Path, raw_root: Path, judge: str, log) -> int:
    files = sorted(out_dir.glob("*.json"))
    if not files or not raw_root.is_dir():
        raise SystemExit(f"--rejudge needs result files in {out_dir} and a --raw directory "
                         f"({raw_root})")
    date = clock("+%F")
    records = attempts = 0
    for path in files:
        doc = json.loads(path.read_text(encoding="utf-8"))
        record_lines, attempt_lines = rejudge_doc(doc, raw_root / doc["bundle_id"], judge, date)
        if record_lines or attempt_lines:
            path.write_text(dumps(doc), encoding="utf-8")
        for line in record_lines + attempt_lines:
            log(line)
        records += len(record_lines)
        attempts += len(attempt_lines)
    (out_dir / "SUMMARY.md").write_text(summarize.render(out_dir), encoding="utf-8")
    log(f"{records} records relabelled, {attempts} earlier attempts; "
        f"wrote {out_dir / 'SUMMARY.md'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wheel")
    ap.add_argument("--python", help="interpreter of the venv holding the wheel")
    ap.add_argument("--bundles")
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--checks", nargs="+", default=None, choices=list(SUITE_V1))
    ap.add_argument("--out")
    ap.add_argument("--work")
    ap.add_argument("--extra-local", nargs="+", default=None)
    ap.add_argument("--order-first", default=None, metavar="REGEX")
    ap.add_argument("--retry-fails", action="store_true")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--quiet-wait", default=QUIET_WAIT)
    ap.add_argument("--machine-label", default="mac-studio-m4-max")
    ap.add_argument("--rejudge", metavar="OUT_DIR", help="re-judge a run directory from "
                    "--raw with the current judge; runs no model")
    ap.add_argument("--raw", metavar="DIR", help="the <work>/run directory of that run")
    ap.add_argument("--previous-judge", default="r2",
                    help="label of the judge the records were written with")
    args = ap.parse_args(argv)
    if args.rejudge:
        if not args.raw:
            ap.error("--rejudge needs --raw")
        return rejudge(Path(args.rejudge), Path(args.raw).expanduser(), args.previous_judge,
                       print)
    missing = [f"--{name}" for name in ("wheel", "python", "bundles", "out", "work")
               if getattr(args, name) is None]
    if missing:
        ap.error("the following arguments are required: " + ", ".join(missing))

    runner = Runner(args)
    signal.signal(signal.SIGTERM, runner.stop_child)
    signal.signal(signal.SIGINT, runner.stop_child)

    def log(msg: str) -> None:
        print(f"[{clock('+%H:%M:%S')}] {msg}", flush=True)

    code: int | str = 1
    try:
        code = run_all(args, runner, log)
    except SystemExit as exc:
        code = exc.code if exc.code is not None else 0
        raise
    except BaseException:
        traceback.print_exc()
        code = 1
        raise
    finally:
        if not args.only:
            done = runner.work / "logs" / DONE_FILE
            done.parent.mkdir(parents=True, exist_ok=True)
            done.write_text(f"{code}\n", encoding="utf-8")
            log(f"exit {code} ({done})")
    return int(code)


if __name__ == "__main__":
    sys.exit(main())
