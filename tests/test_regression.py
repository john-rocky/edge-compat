"""Catalog regression lane (tools/regression/): the result schema and its
example fixture, the committed run directories (layout, provenance, the
generated SUMMARY.md), the own-bundle list, the mechanical judge, and the
rule that these tools never touch the op-level matrix.

data/regression/ is lab-only until the owner decides otherwise, so every
test over it holds for an absent directory too (the public mirror).
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from conftest import REPO_ROOT

TOOLS = REPO_ROOT / "tools" / "regression"
SCHEMA = json.loads((REPO_ROOT / "schemas" / "regression_result.schema.json").read_text(
    encoding="utf-8"))
EXAMPLE_ROOT = REPO_ROOT / "data" / "examples" / "regression"
DATA_ROOT = REPO_ROOT / "data" / "regression"
BUNDLES = DATA_ROOT / "bundles.json"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _tool(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"regression_{name}", TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through it
    spec.loader.exec_module(module)
    return module


run_regression = _tool("run_regression")
summarize = _tool("summarize")
probe_one = _tool("probe_one")
list_bundles = _tool("list_bundles")
roles = _tool("roles")


def _errors(doc: Any) -> list[str]:
    return [f"{'/'.join(map(str, e.absolute_path))}: {e.message}"
            for e in Draft202012Validator(SCHEMA).iter_errors(doc)]


def _result_files(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("*/*/*.json"))


def _run_dirs(root: Path) -> list[Path]:
    return sorted({p.parent for p in _result_files(root)})


def _example() -> dict[str, Any]:
    return json.loads(_result_files(EXAMPLE_ROOT)[0].read_text(encoding="utf-8"))


# --- schema + example fixture --------------------------------------------------


def test_schema_is_valid_and_lists_the_suite_in_order() -> None:
    Draft202012Validator.check_schema(SCHEMA)
    ids = [item["allOf"][1]["properties"]["id"]["const"]
           for item in SCHEMA["properties"]["checks"]["prefixItems"]]
    assert ids == list(probe_one.SUITE_V1) == list(summarize.SUITE_V1)
    assert len(ids) == 13


def test_example_fixture_validates_and_is_example_provenance() -> None:
    files = _result_files(EXAMPLE_ROOT)
    assert len(files) == 1
    doc = _example()
    assert _errors(doc) == []
    assert doc["provenance"] == "example"
    assert files[0].name == f"{doc['bundle_id']}.json"
    assert files[0].parent.name == doc["date"]
    assert files[0].parent.parent.name == doc["env"]["runtime_version"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["checks"].pop(),  # 12 checks
        lambda d: d["checks"].reverse(),  # wrong order
        lambda d: d["checks"][1].update(reason="wrong_answer"),  # PASS with a reason
        lambda d: d["checks"][3].update(raw_head=""),  # FAIL without the answer
        lambda d: d["checks"][3].update(reason="flaky"),  # FAIL outside the vocabulary
        lambda d: d["checks"][11].update(reason=""),  # NA without a reason
        lambda d: d.update(provenance="inferred"),
        lambda d: d["env"].pop("runtime_version"),
        lambda d: d["checks"][2].update(raw_head="x" * 161),
    ],
)
def test_schema_rejects_bad_shapes(mutate: Any) -> None:
    doc = _example()
    mutate(doc)
    assert _errors(doc)


# --- committed runs -----------------------------------------------------------------


def test_committed_runs_validate_with_layout_and_measured_provenance() -> None:
    for path in _result_files(DATA_ROOT):
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert _errors(doc) == [], path
        assert doc["provenance"] == "measured", path
        assert path.name == f"{doc['bundle_id']}.json", path
        assert DATE_RE.match(path.parent.name) and path.parent.name == doc["date"], path
        assert path.parent.parent.name == doc["env"]["runtime_version"], path
        statuses = [c["status"] for c in doc["checks"]]
        assert doc["summary"] == {"pass": statuses.count("PASS"),
                                  "fail": statuses.count("FAIL"),
                                  "na": statuses.count("NA")}, path
        assert path.read_text(encoding="utf-8") == run_regression.dumps(doc), path


@pytest.mark.parametrize("root", [EXAMPLE_ROOT, DATA_ROOT], ids=["examples", "data"])
def test_every_summary_is_the_generator_output(root: Path) -> None:
    for run_dir in _run_dirs(root):
        summary = run_dir / "SUMMARY.md"
        assert summary.is_file(), f"{run_dir}: SUMMARY.md missing (run summarize.py)"
        text = summary.read_text(encoding="utf-8")
        m = re.match(re.escape(summarize.MARKER)
                     + r" from (\S+)(?: --prev (\S+))?(?: --bundles (\S+))?; do not edit", text)
        assert m, f"{summary}: not a generated file"
        prev = REPO_ROOT / m.group(2) if m.group(2) else None
        bundles = REPO_ROOT / m.group(3) if m.group(3) else None
        assert text == summarize.render(run_dir, prev, bundles), (
            f"{summary} differs from summarize.py output: regenerate, never hand-edit")


def test_summary_numbers_every_fail_and_na_once() -> None:
    text = summarize.render(_run_dirs(EXAMPLE_ROOT)[0])
    notes = re.findall(r"^- \[(\d+)\] ", text, re.M)
    doc = _example()
    assert len(notes) == doc["summary"]["fail"] + doc["summary"]["na"]
    assert notes == [str(i) for i in range(1, len(notes) + 1)]


def test_summary_diff_section(tmp_path: Path) -> None:
    before = tmp_path / "0.0.0" / "2026-01-01"
    after = tmp_path / "0.0.1" / "2026-01-02"
    before.mkdir(parents=True)
    after.mkdir(parents=True)
    doc = _example()
    (before / "example-bundle.json").write_text(run_regression.dumps(doc), encoding="utf-8")
    fixed = json.loads(json.dumps(doc))
    fixed["checks"][3].update(status="PASS", reason="", raw_head="391")  # was FAIL loop
    fixed["checks"][1].update(status="FAIL", reason="empty", raw_head="<empty content>")
    (after / "example-bundle.json").write_text(run_regression.dumps(fixed), encoding="utf-8")
    text = summarize.render(after, before)
    assert "new FAIL: 1; fixed: 1;" in text
    assert "- example-bundle plain_paris: PASS -> FAIL empty" in text
    assert "- example-bundle digit_default_2: FAIL loop -> PASS" in text
    assert summarize.render(after, before) == text  # deterministic


def test_summary_counts_roles_apart(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = tmp_path / "0.0.0" / "2026-01-01"
    run.mkdir(parents=True)
    doc = _example()  # 7 PASS / 4 FAIL / 2 NA
    for bid in ("chat-bundle", "ocr-bundle"):
        (run / f"{bid}.json").write_text(run_regression.dumps(dict(doc, bundle_id=bid)),
                                         encoding="utf-8")
    bundles = tmp_path / "bundles.json"
    bundles.write_text(json.dumps({"bundles": [{"bundle_id": "chat-bundle", "role": "chat"},
                                               {"bundle_id": "ocr-bundle", "role": "ocr"}]}),
                       encoding="utf-8")
    text = summarize.render(run, bundles=bundles)
    assert text.splitlines()[0].endswith(f" --bundles {bundles}; do not edit -->")
    assert "- bundles: 2 (chat 1 / non-chat 1: ocr 1)\n" in text
    assert "- checks: 14 PASS / 8 FAIL / 4 NA\n" in text  # every bundle, as before
    assert "- checks (chat bundles only): 7 PASS / 4 FAIL / 2 NA\n" in text
    assert "| chat-bundle | P | P | P | F [1] |" in text
    assert "| ocr-bundle (ocr) | P | P | P | F [7] |" in text
    assert "- [7] ocr-bundle (ocr) digit_default_2: FAIL loop;" in text
    assert "By role: chat 4 FAIL / 2 NA; ocr 4 FAIL / 2 NA.\n" in text
    assert capsys.readouterr().err == ""
    # Without the list every bundle is chat, with one warning.
    text = summarize.render(run, bundles=tmp_path / "absent.json")
    assert "- bundles: 2 (chat 2 / non-chat 0)\n" in text
    assert "- checks (chat bundles only): 14 PASS / 8 FAIL / 4 NA\n" in text
    assert "(ocr)" not in text and "By role" not in text
    assert capsys.readouterr().err.count("warning: ") == 1


# --- the bundle list ------------------------------------------------------------------

ROW_FIELDS = {
    "bundle_id", "hf_repo", "file", "hub_commit", "sha256", "size_bytes", "kind", "suite",
    "role", "lab_card", "manifest", "evidence", "local", "notes", "status",
}


def test_bundle_list_shape() -> None:
    if not BUNDLES.exists():
        pytest.skip("data/regression/bundles.json is lab-only")
    text = BUNDLES.read_text(encoding="utf-8")
    doc = json.loads(text)
    assert text == list_bundles.dumps(doc)  # sorted keys, trailing newline
    rows = doc["bundles"]
    for row in rows:
        assert set(row) >= ROW_FIELDS, row.get("bundle_id")
        assert row["status"] in ("ok", "hub_missing")
        assert row["kind"] in ("chat", "chat+vision", "chat+audio", "other", "unverified")
        assert row["role"] == roles.role_of(row["hf_repo"]), (
            f"{row['bundle_id']}: role differs from roles.py: regenerate the list")
        if row["status"] == "ok":
            assert re.fullmatch(r"[0-9a-f]{64}", row["sha256"]), row["bundle_id"]
            assert re.fullmatch(r"[0-9a-f]{40}", row["hub_commit"]), row["bundle_id"]
            assert isinstance(row["size_bytes"], int) and row["size_bytes"] > 0
        else:
            assert row["sha256"] is None and "last_known" in row
    ids = [r["bundle_id"] for r in rows]
    assert len(ids) == len(set(ids))
    keys = [(r["hf_repo"], r["file"]) for r in rows]
    assert len(keys) == len(set(keys))
    assert keys == sorted(keys, key=lambda k: (k[0] is None, k[0] or "", k[1]))
    counts = doc["meta"]["counts"]
    assert counts == list_bundles.count(rows, counts["skipped_not_own"])
    assert counts["variants"] == (counts["lab_card_matched"] + counts["manifest_only"]
                                  + counts["lab_card_only"] + counts["hf_only"])


def test_roles_name_listed_repos() -> None:
    assert roles.DEFAULT == roles.VALUES[0] == "chat"
    assert set(roles.ROLES.values()) <= set(roles.VALUES) - {roles.DEFAULT}
    assert roles.role_of(None) == roles.role_of("org/not-in-the-map") == roles.DEFAULT
    if not BUNDLES.exists():
        pytest.skip("data/regression/bundles.json is lab-only")
    listed = {r["hf_repo"] for r in json.loads(BUNDLES.read_text(encoding="utf-8"))["bundles"]}
    assert set(roles.ROLES) <= listed, sorted(set(roles.ROLES) - listed)


def test_kind_and_slug() -> None:
    assert list_bundles.kind_of(["tf_lite_prefill_decode"]) == "chat"
    assert list_bundles.kind_of(["tf_lite_embedder", "tf_lite_prefill_decode"]) == "chat"
    assert list_bundles.kind_of(
        ["tf_lite_prefill_decode", "tf_lite_vision_encoder", "tf_lite_vision_adapter"]
    ) == "chat+vision"
    assert list_bundles.kind_of(
        ["tf_lite_audio_encoder_hw", "tf_lite_prefill_decode"]) == "chat+audio"
    assert list_bundles.kind_of(["tf_lite_embedder"]) == "other"
    assert list_bundles.slug("Qwen3.5-2B_int8") == "qwen3.5-2b-int8"


# --- the judge -----------------------------------------------------------------------


def test_digit_judge() -> None:
    judge = run_regression.judge_digit
    assert judge("17 times 23 is 391.") == ("PASS", "")
    assert judge("17 times 23 is 3" + "9" * 40) == ("FAIL", "loop")
    assert judge("$391$ " * 60) == ("FAIL", "loop")
    assert judge("17 x 23 = 391. " + "a b " * 400) == ("FAIL", "loop")
    long_answer = "".join(f"line {i}: 17 times 23 is 391 when worked out.\n" for i in range(40))
    assert judge(long_answer) == ("FAIL", "cap_hit")
    assert judge("17 times 23 is 399.") == ("FAIL", "wrong_answer")
    assert judge("  \n") == ("FAIL", "empty")
    # The 400-token cap, read from token counts: 422 tokens after the turn,
    # the prompt alone = 23 - 1 = 22 tokens -> 400 output tokens.
    turn = {"token_count_before": 0, "token_count_after": 422}
    assert run_regression.output_tokens(turn, 23) == 400
    assert run_regression.output_tokens(turn, None) is None
    assert judge("17 x 23 = 391, and then (17 \\times", 400, 400) == ("FAIL", "cap_hit")
    assert judge("17 times 23 is 391.", 250, 400) == ("PASS", "")


def test_digit_judge_reads_the_final_number() -> None:
    final, judge = run_regression.final_number, run_regression.judge_digit
    # The last integer of 3+ digits, thousands commas dropped, LaTeX read as text.
    assert final("is 1,234 and then 391.") == "391"
    assert final("17 x 23 = 391, then 1,000") == "1000"
    assert final("\\( 17 \\times 23 = 391 \\)") == "391"
    assert final("\\boxed{391} (17 and 23)") == "391"
    assert final("17 times 23") is None
    # 391 on the way, 791 at the end (MiniCPM5-2B int8, 2026-10-09 / 10-12): wrong.
    minicpm = "Adjustment = 400 - 9 = 391. So 400 + 391 = **791**."
    assert final(minicpm) == "791"
    assert judge(minicpm) == ("FAIL", "wrong_answer")
    assert judge("17 times 23 is 391 (check: 391 / 17 = 23).") == ("PASS", "")
    assert judge("The answer is 391. Note: 2026.") == ("FAIL", "wrong_answer")


def test_think_on_cap_is_thought_budget_plus_content() -> None:
    spec = probe_one.CHECKS["digit_think_on"]
    assert spec["cap"] == 700 and spec["thinking"]["thinking_token_budget"] == 300
    assert run_regression.base_record("digit_think_on")["max_output_tokens"] == 700
    judge = run_regression.judge_digit
    assert judge("17 x 23 = 391.", 689, 700) == ("PASS", "")
    assert judge("17 x 23 = 391.", 690, 700) == ("FAIL", "cap_hit")
    assert all(probe_one.CHECKS[c]["cap"] == 400 for c in (
        "digit_default_1", "digit_default_2", "digit_default_3", "digit_think_off",
        "digit_gpu_default"))


def test_tool_and_shared_judges() -> None:
    tool = run_regression.judge_tool_call
    # r2c (2026-10-13): the runtime has to report the call. Hy-MT2-1.8B int8 and
    # LFM2.5-VL-450M int8 answered "15-18°C ... partly cloudy" with no event.
    assert tool("It is 18°C and cloudy in Paris.", [], [], []) == ("FAIL", "not_called")
    assert tool("It is 18 C and cloudy.", [], [{"event": "a"}, {"event": "b"}], []) == (
        "PASS", "")
    assert tool("", [], [{"event": "approve_tool_call"}], []) == ("PASS", "")
    assert tool("<tool_call>{\"name\": \"get_weather\"}</tool_call>", [], [], []) == (
        "FAIL", "raw_tag")
    assert tool("It is sunny in Paris.", [], [], []) == ("FAIL", "not_called")

    shared = run_regression.judge_shared
    good = ["Paris", "391", "Blue"]
    assert shared(good, good) == ("PASS", "", "")
    assert shared(["Paris, arr!", "391", "Blue"], good) == ("FAIL", "state_leak", "pirate_marker")
    assert shared(["Paris", "399", "Blue"], good) == ("FAIL", "state_leak", "key_missing")
    assert shared(["Paris", "399", "Blue"], ["Paris", "399", "Blue"]) == (
        "FAIL", "control_failed", "")
    assert shared(["An array: Paris", "391", "Blue"], good) == ("PASS", "", "")  # not a pirate


def test_shared_judge_sys_b_compliance() -> None:
    shared, words = run_regression.judge_shared, run_regression.word_count
    assert words("Paris.") == 1 and words("It's light blue") == 3 and words("391") == 1
    good = ["Paris", "391", "Blue"]
    # Qwen3.5-2B int8, r1 first dry run: every key there, every answer a sentence.
    sentences = ["The capital of France is Paris.", "17 times 23 is 391.",
                 "The sky on a clear day is blue."]
    assert shared(sentences, good) == ("FAIL", "state_leak", "sysB_ignored")
    assert shared(["Paris", "17 times 23 is 391.", "Light blue"], good) == ("PASS", "", "")
    # Two of three ignored is enough; the control must obey all three.
    assert shared(["Paris", sentences[1], sentences[2]], good)[2] == "sysB_ignored"
    assert shared(sentences, ["Paris", "It is 391.", "Blue"]) == ("PASS", "", "")


def test_declared_backends() -> None:
    declared = run_regression.declared_backends
    assert declared({"describe_stdout": "  Text Backends:          CPU GPU\n"}) == {"CPU", "GPU"}
    assert declared({"describe_stdout": "  Text Backends:          NPU\n"}) == {"NPU"}
    assert declared({}) is None


def test_cpu_gpu_derivation() -> None:
    def rec(status: str, reason: str = "", timeout: bool = False) -> dict[str, Any]:
        return {"status": status, "reason": reason, "timeout": timeout, "raw_head": "x"}

    derive = run_regression.judge_cpu_gpu
    assert derive(rec("PASS"), rec("PASS")) == ("PASS", "")
    assert derive(rec("PASS"), rec("FAIL", "wrong_answer"))[0] == "FAIL"
    assert derive(rec("FAIL", "loop"), rec("PASS")) == ("FAIL", "cpu failed (loop)")
    assert derive(rec("PASS"), rec("FAIL", "engine_create_failed:OOM"))[0] == "NA"
    assert derive(rec("PASS"), rec("FAIL", "timeout 1800", timeout=True))[0] == "NA"
    assert derive(rec("FAIL", "loop"), rec("FAIL", "loop"))[0] == "NA"


def test_order_first_and_resume_rule(tmp_path: Path) -> None:
    rows = [{"bundle_id": b, "hf_repo": r, "file": f, "size_bytes": s} for b, r, f, s in (
        ("a", "o/Gemma", "a.litertlm", 5), ("b", "o/Qwen3.5-2B", "b.litertlm", 9),
        ("c", "o/X", "c.litertlm", 1), ("d", "o/LFM2-1B", "d.litertlm", 3),
        ("e", "o/Y", "zamba_e.litertlm", 7))]
    order = run_regression.order_rows
    pattern = r"qwen3\.5|lfm2|zamba"
    assert [r["bundle_id"] for r in order(rows, pattern)] == ["d", "e", "b", "c", "a"]
    assert [r["bundle_id"] for r in order(rows, None)] == ["a", "b", "c", "d", "e"]

    path = tmp_path / "x.json"
    complete = run_regression.bundle_complete
    assert not complete(path)
    doc = _example()
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert complete(path) == (not any(
        c["status"] == "NA" and "not run" in c["reason"] for c in doc["checks"]))
    doc["checks"][11].update(status="NA", reason="not run: 2 timeouts in this bundle")
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert not complete(path)
    doc["checks"][11].update(status="NA", reason="sha_mismatch")
    doc["checks"].pop()
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert not complete(path)  # 12 checks


def test_merge_retry() -> None:
    first = run_regression.base_record("digit_default_2")
    first.update(status="FAIL", reason="wrong_answer", raw_head="... = 791", elapsed_s=7.0)
    first["details"].update(final_number="791", run_at="2026-10-12T14:00:00+09:00")
    again = run_regression.base_record("digit_default_2")
    again.update(status="PASS", reason="", raw_head="... = 391", elapsed_s=6.0)
    merged = run_regression.merge_retry(json.loads(json.dumps(first)), again)
    assert merged["status"] == "PASS"
    assert merged["details"]["previous_attempts"] == [{
        "status": "FAIL", "reason": "wrong_answer", "raw_head": "... = 791", "elapsed_s": 7.0,
        "run_at": "2026-10-12T14:00:00+09:00", "details": {"final_number": "791"}}]
    never = run_regression.not_run("digit_default_2", "measurement window held")
    kept = run_regression.merge_retry(json.loads(json.dumps(first)), never)
    assert kept["status"] == "FAIL" and "previous_attempts" not in kept["details"]
    assert kept["details"]["retry_not_run"].startswith("not run:")


def _fake_suite(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                answers: dict[tuple[str, int], Any]) -> Any:
    """Runner.probe replaced by canned observations: answers[(stem, attempt)]
    = the last turn's content, or "TIMEOUT"."""
    rr = run_regression
    model = tmp_path / "m.litertlm"
    model.write_bytes(b"x")
    monkeypatch.setattr(rr, "locate", lambda runner, row, log: rr.Located("hf_cache", model))
    seen: dict[str, int] = {}

    def probe(self: Any, bid: str, link: Path, check: str, *, backend: str, ctx: int | None,
              part: str | None = None) -> Any:
        stem = check + (f".{part}" if part else "")
        seen[stem] = seen.get(stem, 0) + 1
        answer = answers.get((stem, seen[stem]), answers.get((stem, 1)))
        log = tmp_path / f"{stem}.log"
        run_at = f"2026-10-12T14:{seen[stem]:02d}:00+09:00"
        if answer == "TIMEOUT":
            self.timeouts_in_bundle += 1
            return rr.ProbeRun(None, True, 900.0, None, log, False, run_at)
        if check == "describe":
            data = {"describe_exit": 0, "peek_exit": 0, "describe": {
                "describe_stdout": "  Text Backends:  CPU GPU\n", "max_context_tokens": 4096,
                "sections": []}}
        elif check == "engine_shared_2conv":
            convs = ["conv1"] * 3 + ["conv2"] * 3 if part == "shared" else ["conv2"] * 3
            data = {"engine_s": 1.0, "turns": [{"conv": c, "content": a}
                                               for c, a in zip(convs, answer, strict=True)]}
        else:
            data = {"engine_s": 1.0, "max_num_tokens": 1024,
                    "max_output_tokens": rr.CHECKS[check]["cap"],
                    "prompt_probe_token_count": 21, "tool_events": [{"event": "x"}],
                    "turns": [{"content": answer, "token_count_before": 0,
                               "token_count_after": 120, "channels": {}, "tool_calls": []}]}
        self.engine_attempts += 1
        self.engines_made += 1
        return rr.ProbeRun(0, False, 5.0, data, log, False, run_at)

    monkeypatch.setattr(rr.Runner, "probe", probe)
    args = argparse.Namespace(work=str(tmp_path / "work"), python=sys.executable,
                              quiet_wait="quiet_wait.py", retry_fails=True, keep=False,
                              extra_local=None)
    return rr.Runner(args)


ROW = {"bundle_id": "fake-bundle", "hf_repo": "org/Fake", "hub_commit": "0" * 40,
       "file": "fake.litertlm", "sha256": "0" * 64, "size_bytes": 1}
ENV = {"runtime": "litert-lm", "runtime_version": "0.18.0", "wheel_source": "PyPI",
       "python": "3.14.6", "host": "test", "machine_label": "test", "os": "27.0"}
GOOD = {("plain_paris", 1): "Paris", ("two_turn_memory", 1): "Kenji",
        ("tool_call_auto", 1): "It is 18 C and cloudy.",
        ("engine_shared_2conv.shared", 1): ["Arr, Paris!", "Arr, 391!", "Arr, blue!",
                                            "Paris", "391", "Blue"],
        ("engine_shared_2conv.control", 1): ["Paris", "391", "Blue"],
        **{(c, 1): "17 x 23 = 391." for c in probe_one.SUITE_V1 if c.startswith("digit_")}}


def test_run_bundle_reruns_a_fail_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    answers = dict(GOOD)
    answers[("digit_default_2", 1)] = "400 - 9 = 391, so 400 + 391 = 791."
    answers[("digit_default_2", 2)] = "17 x 23 = 391."
    answers[("digit_default_3", 1)] = "17 x 23 = 399."
    answers[("digit_default_3", 2)] = "17 x 23 = 389."
    runner = _fake_suite(monkeypatch, tmp_path, answers)
    out = tmp_path / "out" / "0.18.0" / "2026-10-12"
    selected = [c for c in probe_one.SUITE_V1 if c != "cpu_gpu_match"]
    run_regression.run_bundle(runner, ROW, selected, out, ENV, "2026-10-12", lambda m: None)
    doc = json.loads((out / "fake-bundle.json").read_text(encoding="utf-8"))
    assert _errors(doc) == []
    recs = {c["id"]: c for c in doc["checks"]}
    assert recs["digit_default_2"]["status"] == "PASS"
    prev = recs["digit_default_2"]["details"]["previous_attempts"]
    assert [(a["status"], a["reason"]) for a in prev] == [("FAIL", "wrong_answer")]
    assert prev[0]["details"]["final_number"] == "791"
    assert prev[0]["run_at"] == "2026-10-12T14:01:00+09:00"
    assert recs["digit_default_2"]["details"]["run_at"] == "2026-10-12T14:02:00+09:00"
    assert recs["digit_default_3"]["status"] == "FAIL"  # failed twice: not flaky
    assert len(recs["digit_default_3"]["details"]["previous_attempts"]) == 1
    assert "previous_attempts" not in recs["plain_paris"]["details"]
    assert recs["cpu_gpu_match"]["status"] == "PASS"
    assert "re-run once: digit_default_2 (FAIL wrong_answer -> PASS), digit_default_3" \
        in doc["notes"]
    assert run_regression.bundle_complete(out / "fake-bundle.json")
    text = (out / "SUMMARY.md").read_text(encoding="utf-8")
    assert text == summarize.render(out)
    assert "- checks: 12 PASS / 1 FAIL / 0 NA; flaky 1" in text
    assert "| fake-bundle | P | P | P | P* | F [1] |" in text
    assert "- fake-bundle digit_default_2: FAIL wrong_answer -> PASS" in text
    assert "FAIL checks re-run once: 2; status changed: 1." in text


def test_run_bundle_two_timeouts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    answers = dict(GOOD)
    answers[("plain_paris", 1)] = "TIMEOUT"
    answers[("digit_default_1", 1)] = "TIMEOUT"
    runner = _fake_suite(monkeypatch, tmp_path, answers)
    out = tmp_path / "out" / "0.18.0" / "2026-10-12"
    selected = [c for c in probe_one.SUITE_V1 if c != "cpu_gpu_match"]
    outcome = run_regression.run_bundle(runner, ROW, selected, out, ENV, "2026-10-12",
                                        lambda m: None)
    assert outcome.code == 0
    doc = json.loads((out / "fake-bundle.json").read_text(encoding="utf-8"))
    assert _errors(doc) == []
    recs = {c["id"]: c for c in doc["checks"]}
    assert recs["plain_paris"]["reason"] == recs["digit_default_1"]["reason"] == "timeout 900"
    assert "previous_attempts" not in recs["plain_paris"]["details"]  # no re-run after 2
    rest = [c for c in probe_one.SUITE_V1[3:] if c != "cpu_gpu_match"]
    assert all(recs[c]["reason"] == "not run: 2 timeouts in this bundle" for c in rest)
    assert recs["cpu_gpu_match"]["status"] == "NA"
    assert not run_regression.bundle_complete(out / "fake-bundle.json")


def test_run_bundle_source_deleted_mid_run(monkeypatch: pytest.MonkeyPatch,
                                           tmp_path: Path) -> None:
    """A read-only source deleted under the run (2026-10-12: another lane's
    close removed the --extra-local copy mid-bundle): NA, not a wall of FAIL."""
    runner = _fake_suite(monkeypatch, tmp_path, dict(GOOD))
    fake_probe = run_regression.Runner.probe

    def probe(self: Any, bid: str, link: Path, check: str, **kw: Any) -> Any:
        run = fake_probe(self, bid, link, check, **kw)
        if check == "digit_default_1":
            (tmp_path / "m.litertlm").unlink()
        return run

    monkeypatch.setattr(run_regression.Runner, "probe", probe)
    out = tmp_path / "out" / "0.18.0" / "2026-10-12"
    selected = [c for c in probe_one.SUITE_V1 if c != "cpu_gpu_match"]
    run_regression.run_bundle(runner, ROW, selected, out, ENV, "2026-10-12", lambda m: None)
    doc = json.loads((out / "fake-bundle.json").read_text(encoding="utf-8"))
    assert _errors(doc) == []
    recs = {c["id"]: c for c in doc["checks"]}
    assert recs["digit_default_1"]["status"] == "PASS"
    rest = [c for c in probe_one.SUITE_V1[3:] if c != "cpu_gpu_match"]
    assert all(recs[c]["reason"].startswith("not run: the model file went away") for c in rest)
    assert not run_regression.bundle_complete(out / "fake-bundle.json")


def test_rejudge_relabels_from_raw(tmp_path: Path) -> None:
    """--rejudge (2026-10-13): LFM2.5's `<|tool_call_start|>` text, not_called
    under the r2 judge, is raw_tag; an empty answer whose thought channel used
    up the cap is cap_hit. Records the judge still agrees with, a probe JSON
    of another attempt and a file with nothing to change stay byte for byte."""
    rr = run_regression
    out, raw = tmp_path / "0.18.0" / "2026-10-12", tmp_path / "run"
    out.mkdir(parents=True)
    steady = _example()
    steady["bundle_id"] = "steady-bundle"  # no probe JSONs: nothing to re-judge
    (out / "steady-bundle.json").write_text(rr.dumps(steady), encoding="utf-8")
    doc = _example()
    recs = {c["id"]: c for c in doc["checks"]}

    def recorded(check: str, turn: dict[str, Any], *, part: str | None = None,
                 started: float = 1760245200.0, **data: Any) -> None:
        stem = check + (f".{part}" if part else "")
        (raw / doc["bundle_id"]).mkdir(parents=True, exist_ok=True)
        probe = {"engine_s": 1.0, "max_num_tokens": 1024,
                 "max_output_tokens": rr.CHECKS[check]["cap"], "turns": [turn], **data}
        (raw / doc["bundle_id"] / f"{stem}.json").write_text(json.dumps(probe), encoding="utf-8")
        (raw / doc["bundle_id"] / f"{stem}.json.started").write_text(f"{started}\n",
                                                                     encoding="utf-8")
        recs[check]["details"]["run_at"] = rr.iso(1760245200.0)

    def turn(content: str, after: int, thought: str | None = None) -> dict[str, Any]:
        return {"content": content, "channels": {"thought": thought} if thought else {},
                "tool_calls": [], "token_count_before": 0, "token_count_after": after}

    lfm = '<|tool_call_start|>[get_weather(location="Paris")]<|tool_call_end|>'
    recs["tool_call_auto"].update(raw_head=lfm)
    recs["tool_call_auto"]["details"]["previous_attempts"] = [{
        "status": "FAIL", "reason": "not_called", "raw_head": lfm, "elapsed_s": 1.0,
        "run_at": None, "details": {}}]  # no .a1.json: the attempt stays as it is
    recorded("tool_call_auto", turn(lfm, 114), tool_events=[], tool_executions=[])
    # Qwen3-0.6B digit_default_1 (2026-10-12): 418 tokens after the turn, the
    # prompt alone 19 - 1 = 18 -> 400 output tokens, all of them thought.
    thought = "Okay, so I need to find out what 17 multiplied by 23 is. " * 19
    recs["digit_default_1"].update(status="FAIL", reason="empty",
                                   raw_head=f"<no text; thought channel {len(thought)} chars>")
    recorded("digit_default_1", turn("", 418, thought), prompt_probe_token_count=19)
    recs["digit_gpu_default"].update(status="PASS", reason="", raw_head="17 x 23 = 391.")
    recorded("digit_gpu_default", turn("17 x 23 = 391.", 40), prompt_probe_token_count=19)
    recs["cpu_gpu_match"].update(status="FAIL", reason="cpu failed (empty)")
    recs["cpu_gpu_match"]["details"] = {"cpu": "FAIL empty", "gpu": "PASS"}
    recorded("plain_paris", turn("Paris (example answer)", 30))  # still PASS
    # A thought-only answer from another attempt than the record's: left alone.
    recorded("digit_think_off", turn("", 418, thought), prompt_probe_token_count=19,
             started=1760245999.0)
    doc["summary"] = rr.summary_of(doc["checks"])
    (out / "example-bundle.json").write_text(rr.dumps(doc), encoding="utf-8")
    before = {p.name: p.read_text(encoding="utf-8") for p in out.glob("*.json")}

    lines: list[str] = []
    assert rr.rejudge(out, raw, "r2", lines.append) == 0
    assert lines[:-1] == [
        "example-bundle digit_default_1: FAIL empty -> FAIL cap_hit",
        "example-bundle tool_call_auto: FAIL not_called -> FAIL raw_tag",
        "example-bundle cpu_gpu_match: FAIL cpu failed (empty) -> FAIL cpu failed (cap_hit)"]
    assert (out / "steady-bundle.json").read_text(encoding="utf-8") == before["steady-bundle.json"]
    new = json.loads((out / "example-bundle.json").read_text(encoding="utf-8"))
    assert _errors(new) == []
    old_by, new_by = recs, {c["id"]: c for c in new["checks"]}
    for check in probe_one.SUITE_V1:
        if check not in ("digit_default_1", "tool_call_auto", "cpu_gpu_match"):
            assert new_by[check] == old_by[check], check
    tool = new_by["tool_call_auto"]
    assert (tool["status"], tool["reason"], tool["raw_head"]) == ("FAIL", "raw_tag", lfm)
    assert tool["details"]["previous_judge"] == {"status": "FAIL", "reason": "not_called",
                                                 "judge": "r2"}
    assert tool["details"]["previous_attempts"][0]["reason"] == "not_called"
    digit = new_by["digit_default_1"]
    assert digit["reason"] == "cap_hit" and digit["raw_head"] == old_by["digit_default_1"][
        "raw_head"]
    assert digit["details"]["thought_only"] is True and digit["details"]["cap_hit_by"] == "tokens"
    assert digit["details"]["thought_head"] == thought[:160]
    assert new_by["cpu_gpu_match"]["details"] == {
        "cpu": "FAIL cap_hit", "gpu": "PASS",
        "previous_judge": {"status": "FAIL", "reason": "cpu failed (empty)", "judge": "r2"}}
    assert new["summary"] == doc["summary"]  # FAIL -> FAIL only
    assert re.search(r"; rejudged \d{4}-\d{2}-\d{2}: 3 records relabelled$", new["notes"])
    assert (out / "SUMMARY.md").read_text(encoding="utf-8") == summarize.render(out)
    # A second pass finds nothing: the files stay byte for byte.
    after = {p.name: p.read_text(encoding="utf-8") for p in out.glob("*.json")}
    assert rr.rejudge(out, raw, "r2", lines.append) == 0
    assert {p.name: p.read_text(encoding="utf-8") for p in out.glob("*.json")} == after
    # Without a token count, the thought rule falls back to its length.
    empty_answer = {"content": "", "channels": {"thought": "x" * 1200}}
    assert rr.thought_cap_hit(empty_answer, None, 400) == "chars"
    assert rr.thought_cap_hit({**empty_answer, "channels": {"thought": "x" * 1199}},
                              None, 400) is None
    assert rr.thought_cap_hit(empty_answer, 389, 400) is None  # tokens read: under the cap
    assert rr.judge_tool_call('{"name": "get_weather", "arguments": {}}', [], [], []) == (
        "FAIL", "raw_tag")


def _numbered_answer(length: int) -> str:
    """A non-repeating answer of exactly `length` characters ending in 391."""
    tail = "</think> So 17 x 23 = 391."
    body = "<think>" + " ".join(f"{i} x 23 = {i * 23}." for i in range(1, 400))
    text = body[:length - len(tail)] + tail
    assert len(text) == length and run_regression.final_number(text) == "391"
    return text


def test_cap_rules_tokens_first_and_thought_chars_per_cap() -> None:
    """r2c (2026-10-13): a readable token count decides the cap on its own
    (twil-lm3 digit_think_on, 2026-10-12: 368 tokens of a 700 cap, 1221
    characters with the raw <think> block, final 391); the length stands in
    only without one. An answer with no text whose thought ran into the cap,
    when no token count can be read: 3 characters per token of the cap
    (two_turn_memory's 100 -> 300), tool_call_auto included."""
    rr = run_regression
    judge = rr.judge_digit
    assert judge(_numbered_answer(1221), 368, 700) == ("PASS", "")
    assert judge(_numbered_answer(1300)) == ("FAIL", "cap_hit")  # no token count
    assert judge(_numbered_answer(1300), None, 700) == ("FAIL", "cap_hit")
    assert judge("17 x 23 = 391.", 395, 400) == ("FAIL", "cap_hit")

    def no_text(thought: str) -> dict[str, Any]:
        return {"content": "", "channels": {"thought": thought}, "tool_calls": [],
                "token_count_before": 0, "token_count_after": 420}

    assert rr.thought_cap_hit(no_text("x" * 320), None, 100) == "chars"
    assert rr.thought_cap_hit(no_text("x" * 299), None, 100) is None
    assert rr.thought_cap_hit(no_text("x" * 743), None, 400) is None  # plain_paris stays empty
    assert rr.thought_cap_hit(no_text("x" * 1200), None, None) == "chars"  # cap unknown

    def tool_run(turn: dict[str, Any]) -> Any:
        data = {"engine_s": 1.0, "max_num_tokens": 1024, "max_output_tokens": 400,
                "turns": [turn], "tool_events": [], "tool_executions": []}
        return rr.ProbeRun(0, False, 5.0, data, Path("tool_call_auto.log"), False,
                           rr.iso(1760245200.0))

    rec = rr.judge_chat("tool_call_auto", tool_run(no_text("Let me think. " * 108)))
    assert (rec["status"], rec["reason"]) == ("FAIL", "cap_hit")
    assert rec["details"]["thought_only"] is True and rec["details"]["cap_hit_by"] == "chars"
    assert rec["raw_head"].startswith("<no text; thought channel 1512 chars")
    short = rr.judge_chat("tool_call_auto", tool_run(no_text("x" * 1199)))
    assert (short["status"], short["reason"]) == ("FAIL", "not_called")
    bare = rr.judge_chat("tool_call_auto", tool_run({**no_text(""), "channels": {}}))
    assert (bare["status"], bare["reason"]) == ("FAIL", "not_called")
    broken = rr.judge_chat("tool_call_auto", tool_run({
        **no_text(""), "content": '{"arguments": {"location": "Paris"}}, "get_weather"}'}))
    assert (broken["status"], broken["reason"]) == ("FAIL", "not_called")  # text: no rule


def test_rejudge_reads_earlier_attempts(tmp_path: Path) -> None:
    """r2c: --rejudge judges each attempt under previous_attempts again from
    the JSON its re-run stashed (<check>.a1.json), paired by the attempt's log
    name and raw_head; an attempt whose stash holds another answer stays."""
    rr = run_regression
    out, raw = tmp_path / "out", tmp_path / "run"
    out.mkdir()
    doc = _example()
    recs = {c["id"]: c for c in doc["checks"]}
    bundle_raw = raw / doc["bundle_id"]
    bundle_raw.mkdir(parents=True)

    def stash(name: str, content: str, after: int, **data: Any) -> None:
        check = name.split(".")[0]
        probe = {"engine_s": 1.0, "max_num_tokens": 1024,
                 "max_output_tokens": rr.CHECKS[check]["cap"], **data,
                 "turns": [{"content": content, "channels": {}, "tool_calls": [],
                            "token_count_before": 0, "token_count_after": after}]}
        (bundle_raw / name).write_text(json.dumps(probe), encoding="utf-8")
        if not name.endswith(".a1.json"):
            (bundle_raw / f"{name}.started").write_text("1760245200.0\n", encoding="utf-8")

    def attempt(status: str, reason: str, raw_head: str, check: str) -> dict[str, Any]:
        return {"status": status, "reason": reason, "raw_head": raw_head, "elapsed_s": 9.0,
                "run_at": rr.iso(1760244000.0), "details": {"log": f"{check}.a1.log"}}

    lfm = '<|tool_call_start|>[get_weather(location="Paris")]<|tool_call_end|>'
    tool = recs["tool_call_auto"]
    tool.update(status="FAIL", reason="raw_tag", raw_head=lfm)
    tool["details"].update(run_at=rr.iso(1760245200.0), previous_attempts=[
        attempt("FAIL", "not_called", lfm, "tool_call_auto")])  # judged before raw_tag
    stash("tool_call_auto.json", lfm, 114, tool_events=[], tool_executions=[])
    stash("tool_call_auto.a1.json", lfm, 114, tool_events=[], tool_executions=[])
    # twil-lm3 digit_think_on: the record is 368 tokens of 700 (PASS now); the
    # attempt before it ran 700 tokens (still cap_hit) -> a flaky cell.
    think = _numbered_answer(1221)
    on = recs["digit_think_on"]
    on.update(status="FAIL", reason="cap_hit", raw_head=think[:160])
    on["details"].update(run_at=rr.iso(1760245200.0), cap_hit_by="chars", previous_attempts=[
        attempt("FAIL", "cap_hit", think[:160], "digit_think_on")])
    stash("digit_think_on.json", think, 387, prompt_probe_token_count=20)
    stash("digit_think_on.a1.json", think, 719, prompt_probe_token_count=20)
    # An attempt whose stash holds another answer: left as it is.
    memory = recs["two_turn_memory"]
    memory["details"]["previous_attempts"] = [
        attempt("FAIL", "empty", "<no text; thought channel 320 chars: x>", "two_turn_memory")]
    stash("two_turn_memory.a1.json", "Your name is Kenji.", 30)
    doc["summary"] = rr.summary_of(doc["checks"])
    (out / "example-bundle.json").write_text(rr.dumps(doc), encoding="utf-8")

    lines: list[str] = []
    assert rr.rejudge(out, raw, "r2b", lines.append) == 0
    assert lines[:-1] == [
        "example-bundle digit_think_on: FAIL cap_hit -> PASS",
        "example-bundle tool_call_auto attempt 1: FAIL not_called -> FAIL raw_tag"]
    assert lines[-1].startswith("1 records relabelled, 1 earlier attempts; wrote ")
    new = json.loads((out / "example-bundle.json").read_text(encoding="utf-8"))
    assert _errors(new) == []
    by = {c["id"]: c for c in new["checks"]}
    first = by["tool_call_auto"]["details"]["previous_attempts"][0]
    assert (first["status"], first["reason"], first["raw_head"]) == ("FAIL", "raw_tag", lfm)
    assert first["details"]["previous_judge"] == {"status": "FAIL", "reason": "not_called",
                                                  "judge": "r2b"}
    assert by["tool_call_auto"]["status"] == "FAIL" and "previous_judge" not in by[
        "tool_call_auto"]["details"]
    assert by["digit_think_on"]["status"] == "PASS"
    assert "cap_hit_by" not in by["digit_think_on"]["details"]  # the old verdict's
    assert by["digit_think_on"]["details"]["previous_attempts"][0]["reason"] == "cap_hit"
    assert summarize.flaky(by["digit_think_on"])  # FAIL at 700 tokens, then PASS
    assert by["two_turn_memory"] == recs["two_turn_memory"]
    assert re.search(r"rejudged \d{4}-\d{2}-\d{2}: 1 records relabelled, 1 earlier attempts$",
                     new["notes"])
    assert (out / "SUMMARY.md").read_text(encoding="utf-8") == summarize.render(out)
    after = (out / "example-bundle.json").read_text(encoding="utf-8")
    assert rr.rejudge(out, raw, "r2b", lines.append) == 0
    assert (out / "example-bundle.json").read_text(encoding="utf-8") == after


def test_bundle_ids_stay_put() -> None:
    def rows() -> list[dict[str, Any]]:
        return [{"hf_repo": r, "file": f, "lab_card": c, "bundle_id": None} for r, f, c in (
            ("o/A", "x.litertlm", "card-a"), ("o/A", "y.litertlm", None),
            ("o/B", "x.litertlm", None), ("p/B", "x.litertlm", None))]

    first = rows()
    list_bundles.assign_ids(first)
    assert [r["bundle_id"] for r in first] == ["card-a", "a__y", "b__x", "p-b__x"]
    previous = {(r["hf_repo"], r["file"]): r["bundle_id"] for r in first}
    # A new row that sorts first and slugs the same, and the card moving to a new row.
    later = [{"hf_repo": "a/B", "file": "x.litertlm", "lab_card": None, "bundle_id": None},
             *rows(), {"hf_repo": "o/C", "file": "z.litertlm", "lab_card": "card-a",
                       "bundle_id": None}]
    later[1]["lab_card"] = None
    list_bundles.assign_ids(later, previous)
    assert {(r["hf_repo"], r["file"]): r["bundle_id"] for r in later} == {
        **previous, ("a/B", "x.litertlm"): "a-b__x", ("o/C", "z.litertlm"): "c__z"}


def test_vanished_file_keeps_one_row_with_its_card() -> None:
    # The file left the Hub: the scan finds its lab card only (a repo-less row with the
    # card's id) while the previous list still has the Hub row — one row must survive.
    base = {"evidence": ["lab-card"], "suite": "v1", "role": "chat", "status": "ok"}
    old = dict(base, hf_repo="o/A", file="x.litertlm", lab_card="card-a", bundle_id="card-a",
               hub_commit="c1", sha256="ab", size_bytes=1)
    other = dict(base, hf_repo="o/B", file="y.litertlm", lab_card=None, bundle_id="b__y",
                 hub_commit="c2", sha256="cd", size_bytes=2)
    card_only = dict(base, hf_repo=None, file="x.litertlm", lab_card="card-a", bundle_id="card-a",
                     hub_commit=None, sha256=None, size_bytes=None, status="hub_missing",
                     suite=None)
    doc = {"meta": {"counts": {"skipped_not_own": 0}}, "bundles": [card_only, dict(other)]}
    list_bundles.keep_vanished(doc, {"bundles": [old, other]})
    ids = [r["bundle_id"] for r in doc["bundles"]]
    assert sorted(ids) == ["b__y", "card-a"]
    kept = next(r for r in doc["bundles"] if r["bundle_id"] == "card-a")
    assert (kept["hf_repo"], kept["status"], kept["sha256"], kept["lab_card"]) == (
        "o/A", "hub_missing", None, "card-a")
    assert kept["last_known"] == {"sha256": "ab", "size_bytes": 1, "hub_commit": "c1",
                                  "source": "previous list"}
    assert doc["meta"]["counts"]["variants"] == 2 and doc["meta"]["counts"]["hub_missing"] == 1


def test_check_ignores_head_moves_of_missing_rows(monkeypatch: pytest.MonkeyPatch,
                                                  tmp_path: Path,
                                                  capsys: pytest.CaptureFixture[str]) -> None:
    class FakeHub:
        calls = 0

        def __init__(self, token: str | None = None) -> None:
            pass

    monkeypatch.setattr(list_bundles, "Hub", FakeHub)
    monkeypatch.setattr(list_bundles, "hub_token", lambda: (None, "none"))
    monkeypatch.setattr(list_bundles, "repo_head", lambda hub, repo: (200, "head2", False))
    monkeypatch.setattr(list_bundles, "repo_litertlm", lambda hub, repo, commit: {})
    gone = {"bundle_id": "gone", "hf_repo": "o/A", "file": "x.litertlm", "hub_commit": "head1",
            "sha256": None, "size_bytes": None, "status": "hub_missing"}
    path = tmp_path / "bundles.json"
    path.write_text(json.dumps({"bundles": [gone]}), encoding="utf-8")
    assert list_bundles.check(path) == 0  # the repo's head moved on; the file is still gone
    here = dict(gone, bundle_id="here", file="y.litertlm", status="ok", sha256="ab", size_bytes=1)
    path.write_text(json.dumps({"bundles": [gone, here]}), encoding="utf-8")
    assert list_bundles.check(path) == 1
    out = capsys.readouterr().out
    assert "| here | hub_commit | head1 | head2 |" in out and "| gone |" not in out


def test_hub_token_sources(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert list_bundles.hub_token() == (None, "none")
    assert run_regression.hub_token() is None
    (tmp_path / "token").write_text("hf_file\n", encoding="utf-8")
    assert list_bundles.hub_token() == ("hf_file", "token file")
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    assert list_bundles.hub_token() == ("hf_env", "HF_TOKEN")
    assert run_regression.hub_token() == "hf_env"


def test_probe_parsers() -> None:
    peek = (
        "Section 0:\n  Items:\n    <None>\n  Begin Offset: 16384\n  End Offset:   20332\n"
        "  Data Type:    LlmMetadataProto\n\nSection 1:\n  Items:\n"
        "    Key: model_type, Value (String): tf_lite_prefill_decode\n"
        "  Begin Offset: 32768\n  End Offset:   65536\n  Data Type:    TFLiteModel\n"
    )
    assert probe_one.parse_peek_sections(peek) == [
        {"index": 0, "type": "LlmMetadataProto", "model_type": None, "size_bytes": 3948},
        {"index": 1, "type": "TFLiteModel", "model_type": "tf_lite_prefill_decode",
         "size_bytes": 32768},
    ]
    flags = probe_one.template_flags(
        "{% if content is string %}{{ content }}{% elif content is sequence %}"
        "{% for item in content %}{{ item['text'] }}{% endfor %}{% endif %}")
    assert flags["content_is_string"] and flags["reads_parts"]
    assert probe_one.template_flags(None)["has_template"] is False
    assert probe_one.engine_max_tokens(4096) == 1024
    assert probe_one.engine_max_tokens(512) == 512
    assert probe_one.engine_max_tokens(None) == 1024


# --- the hard rule: no op-level matrix from model-level results -----------------------


def test_regression_tools_never_touch_the_matrix() -> None:
    """Same intent as the zero-matrix-writes tests of the web and device
    lanes: nothing under tools/regression names data/matrix or imports the
    matrix package outside its docstrings."""
    for path in sorted(TOOLS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef)
            and node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and id(node) not in docstrings:
                assert "matrix" not in node.value.lower(), f"{path.name}: {node.value!r}"
            if isinstance(node, ast.ImportFrom):
                assert "matrix" not in (node.module or ""), f"{path.name}: {node.module}"
            if isinstance(node, ast.Import):
                assert not any("matrix" in a.name for a in node.names), path.name
