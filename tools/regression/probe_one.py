"""One catalog-regression check on one `.litertlm` bundle, in a fresh process.

    <venv>/bin/python -I tools/regression/probe_one.py <bundle> <check_id> --out <json>
        [--backend cpu|gpu] [--ctx N] [--part shared|control] [--work DIR] [--cache-dir DIR]

Run by `run_regression.py` through `quiet_wait.py`, under the interpreter of
the litert-lm wheel being tested. It records raw observations only — the
answer text, channels, tool events, timings, the exception and the stage it
happened in — and judges nothing: the judge lives in `run_regression.py`.

Module-level imports are stdlib only, so the runner can read `SUITE_V1` and
`CHECKS` from the lab interpreter; `litert_lm` is imported inside the checks.
`describe` also reads the container with the lab's own parser
(`src/litert_compat/parser`), put on `sys.path` explicitly because `-I`
leaves the script directory off it.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import mmap
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

SUITE_V1 = (
    "describe",
    "plain_paris",
    "digit_default_1",
    "digit_default_2",
    "digit_default_3",
    "digit_default_uncapped",
    "digit_think_off",
    "digit_think_on",
    "tool_call_auto",
    "engine_shared_2conv",
    "two_turn_memory",
    "digit_gpu_default",
    "cpu_gpu_match",
)

PROMPT_PARIS = "What is the capital of France? Answer in one word."
PROMPT_DIGIT = "What is 17 times 23?"
PROMPT_WEATHER = "What is the weather in Paris right now?"
SYS_A = (
    "You are a pirate. Answer every question in pirate speak and end every answer "
    "with 'Arr!'."
)
SYS_B = "Answer with a single word."
SHARED_QUESTIONS = (
    "What is the capital of France?",
    "What is 17 times 23?",
    "What color is the sky on a clear day?",
)
MEMORY_TURNS = ("My name is Kenji.", "What is my name? Answer with the name only.")

# backend: None = no engine (describe) or derived by the runner (cpu_gpu_match).
# cap: max_output_tokens per message; None = not passed (the Engine's
# max_num_tokens is the only bound). thinking: ThinkingConfig kwargs, or None
# = not passed. The sampler is never passed: the bundle's default is what an
# app gets.
CHECKS: dict[str, dict[str, Any]] = {
    "describe": {"backend": None, "cap": None, "thinking": None},
    "plain_paris": {"backend": "cpu", "cap": 400, "thinking": None},
    "digit_default_1": {"backend": "cpu", "cap": 400, "thinking": None},
    "digit_default_2": {"backend": "cpu", "cap": 400, "thinking": None},
    "digit_default_3": {"backend": "cpu", "cap": 400, "thinking": None},
    "digit_default_uncapped": {"backend": "cpu", "cap": None, "thinking": None},
    "digit_think_off": {"backend": "cpu", "cap": 400,
                        "thinking": {"enable_thinking": False}},
    # 700 = the thought budget (300) + the 400 every other digit check gets.
    "digit_think_on": {"backend": "cpu", "cap": 700,
                       "thinking": {"enable_thinking": True, "thinking_token_budget": 300}},
    "tool_call_auto": {"backend": "cpu", "cap": 400, "thinking": None},
    "engine_shared_2conv": {"backend": "cpu", "cap": 200, "thinking": None,
                            "parts": ("shared", "control")},
    "two_turn_memory": {"backend": "cpu", "cap": 100, "thinking": None},
    "digit_gpu_default": {"backend": "gpu", "cap": 400, "thinking": None},
    "cpu_gpu_match": {"backend": None, "cap": None, "thinking": None, "derived": True},
}
ENGINE_TOKEN_CEILING = 1024
PREFILL_READ_LIMIT_S = 60

# Executions and events of the weather tool, recorded for the judge.
TOOL_EXECUTIONS: list[dict[str, Any]] = []
TOOL_EVENTS: list[dict[str, Any]] = []


def get_weather(location: str) -> str:
    """Get the current weather for a city.

    Args:
      location: The city name, for example "Paris".
    """
    TOOL_EXECUTIONS.append({"location": location})
    return json.dumps({"location": location, "temperature_c": 18, "condition": "cloudy"})


def engine_max_tokens(ctx: int | None) -> int:
    return min(ENGINE_TOKEN_CEILING, ctx) if ctx and ctx > 0 else ENGINE_TOKEN_CEILING


# --- chat checks ---------------------------------------------------------------


def _text_of(message: Any) -> str:
    doc = message.to_json() if hasattr(message, "to_json") else dict(message)
    content = doc.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", "")) for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def run_chat(args: argparse.Namespace, out: dict[str, Any]) -> None:
    import litert_lm  # the wheel under test
    from litert_lm import interfaces

    spec = CHECKS[args.check]
    cap = spec["cap"]
    out["max_num_tokens"] = engine_max_tokens(args.ctx)
    out["max_output_tokens"] = cap
    thinking = None
    if spec["thinking"] is not None:
        thinking = interfaces.ThinkingConfig(**spec["thinking"])
        out["thinking_config"] = {"enable_thinking": thinking.enable_thinking,
                                  "thinking_token_budget": thinking.thinking_token_budget}
    else:
        out["thinking_config"] = None
    backend = interfaces.GPU() if args.backend == "gpu" else interfaces.CPU()
    out["backend"] = args.backend
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(args.work) / "engine_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out["cache_dir"] = str(cache_dir)
    out["turns"] = []

    out["stage"] = "engine"
    t0 = time.monotonic()
    engine = litert_lm.Engine(str(args.bundle), backend=backend,
                              max_num_tokens=out["max_num_tokens"], cache_dir=str(cache_dir))
    out["engine_s"] = round(time.monotonic() - t0, 2)

    def send(conv: Any, prompt: str, label: str | None, turn_cap: int | None) -> None:
        turn: dict[str, Any] = {"conv": label, "prompt": prompt}
        out["stage"] = f"send:{label or ''}:{prompt[:24]}"
        try:
            turn["token_count_before"] = conv.token_count
        except Exception:  # optional observation
            turn["token_count_before"] = None
        t = time.monotonic()
        kwargs = {} if turn_cap is None else {"max_output_tokens": turn_cap}
        message = conv.send_message(prompt, **kwargs)
        turn["elapsed_s"] = round(time.monotonic() - t, 2)
        turn["content"] = _text_of(message)
        turn["channels"] = {k: str(v) for k, v in (message.channels or {}).items()}
        turn["tool_calls"] = [_jsonable(tc.to_json() if hasattr(tc, "to_json") else repr(tc))
                              for tc in message.tool_calls]
        try:
            turn["token_count_after"] = conv.token_count
        except Exception:  # optional observation
            turn["token_count_after"] = None
        out["turns"].append(turn)

    def conversation(**kwargs: Any) -> Any:
        out["stage"] = "conversation"
        if thinking is not None:
            kwargs["thinking_config"] = thinking
        return engine.create_conversation(**kwargs)

    check = args.check
    if check == "plain_paris":
        send(conversation(), PROMPT_PARIS, None, cap)
    elif check.startswith("digit_"):
        send(conversation(), PROMPT_DIGIT, None, cap)
        # The API reports no output token count (that needs benchmark mode,
        # which this suite never turns on), only the conversation's token
        # count. Measure what the prompt alone adds — a second conversation,
        # opened after the judged turn, asked the same thing with a 1-token
        # cap — so the runner can tell an answer that ran into the cap.
        out["stage"] = "prompt-token-probe"
        try:
            probe = conversation()
            probe.send_message(PROMPT_DIGIT, max_output_tokens=1)
            out["prompt_probe_token_count"] = probe.token_count
            probe.close()
        except Exception as exc:  # recorded; the judged turn is already in
            out["prompt_probe_error"] = f"{type(exc).__name__}: {exc}"[:200]
    elif check == "tool_call_auto":

        class Recorder(interfaces.ToolEventHandler):
            def approve_tool_call(self, tool_call: dict[str, Any]) -> bool:
                TOOL_EVENTS.append({"event": "approve_tool_call", "data": _jsonable(tool_call)})
                return True

            def process_tool_response(self, tool_response: dict[str, Any]) -> dict[str, Any]:
                TOOL_EVENTS.append({"event": "process_tool_response",
                                    "data": _jsonable(tool_response)})
                return tool_response

        try:
            send(conversation(tools=[get_weather], tool_event_handler=Recorder()),
                 PROMPT_WEATHER, None, cap)
        finally:
            out["tool_events"] = TOOL_EVENTS
            out["tool_executions"] = TOOL_EXECUTIONS
    elif check == "engine_shared_2conv":
        if args.part == "shared":
            first = conversation(system_message=SYS_A)
            for q in SHARED_QUESTIONS:
                send(first, q, "conv1", cap)
            first.close()
        second = conversation(system_message=SYS_B)
        for q in SHARED_QUESTIONS:
            send(second, q, "conv2", cap)
        second.close()
    elif check == "two_turn_memory":
        conv = conversation()
        for q in MEMORY_TURNS:
            send(conv, q, None, cap)
    else:
        raise ValueError(f"no chat procedure for check {check!r}")
    out["stage"] = "close"
    with contextlib.suppress(Exception):  # closing is not part of the check
        engine.close()
    out["stage"] = "done"


# --- describe --------------------------------------------------------------------


def _run_cli(argv: list[str], timeout: int = 300) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                              check=False)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"timeout after {timeout} s"


def parse_peek_sections(text: str) -> list[dict[str, Any]]:
    """Section list (index / type / model_type / size) from litert-lm-peek."""
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in text.splitlines():
        m = re.match(r"^Section (\d+):", line)
        if m:
            current = {"index": int(m.group(1)), "type": None, "model_type": None,
                       "begin": None, "end": None}
            sections.append(current)
            continue
        if current is None:
            continue
        m = re.match(r"^\s+Key: model_type, Value \(String\): (\S+)", line)
        if m:
            current["model_type"] = m.group(1)
        m = re.match(r"^\s+Begin Offset:\s+(\d+)", line)
        if m:
            current["begin"] = int(m.group(1))
        m = re.match(r"^\s+End Offset:\s+(\d+)", line)
        if m:
            current["end"] = int(m.group(1))
        m = re.match(r"^\s+Data Type:\s+(\S+)", line)
        if m:
            current["type"] = m.group(1)
    out = []
    for s in sections:
        size = s["end"] - s["begin"] if s["begin"] is not None and s["end"] is not None else None
        out.append({"index": s["index"], "type": s["type"], "model_type": s["model_type"],
                    "size_bytes": size})
    return out


def template_flags(template: str | None) -> dict[str, Any]:
    if not template:
        return {"has_template": False, "content_is_string": False, "reads_parts": False}
    return {
        "has_template": True,
        "content_is_string": bool(re.search(r"content\s+is\s+string", template)),
        "reads_parts": bool(re.search(r"content\s+is\s+sequence", template)
                            or re.search(r"item\[\s*['\"]type['\"]\s*\]", template)),
        "length": len(template),
        "sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
    }


def llm_metadata_fields(md: Any) -> dict[str, Any]:
    from google.protobuf import json_format

    def has(name: str) -> bool:
        try:
            return md.HasField(name)
        except ValueError:  # repeated or not a presence field
            return bool(getattr(md, name, None))

    fields: dict[str, Any] = {}
    which = md.llm_model_type.WhichOneof("model_type") if has("llm_model_type") else None
    fields["llm_model_type"] = which or "unverified"
    fields["max_num_tokens"] = md.max_num_tokens if has("max_num_tokens") else "unverified"
    fields["channels"] = [json_format.MessageToDict(c) for c in md.channels]
    for name in ("supports_thinking", "supports_function_calling"):
        fields[name] = getattr(md, name) if has(name) else "unverified"
    fields["sampler_params"] = (json_format.MessageToDict(md.sampler_params)
                                if has("sampler_params") else "unverified")
    fields["min_runtime_version"] = (md.min_runtime_version if has("min_runtime_version")
                                     else "unverified")
    return fields


class _Deadline(Exception):
    pass


def prefill_lengths(model: Path, data: Any, section: Any) -> tuple[list[int], list[str]]:
    """Prefill lengths from the prefill_decode model's signature_defs, read
    with the lab reader's flatbuffer accessors over the mapped section (no
    copy, no dump). TFLite schema: Model.signature_defs = slot 7, SignatureDef
    {inputs 0, signature_key 2, subgraph_index 4}, TensorMap {name 0,
    tensor_index 1}, SubGraph.tensors = slot 0, Tensor {shape 0,
    shape_signature 7}."""
    import flatbuffers
    from flatbuffers import number_types as N
    from flatbuffers.table import Table

    from litert_compat.parser.reader import _int32_vector, _scalar, _string, _table_vector

    buf = memoryview(data)[section.begin : section.end]
    model_tab = Table(buf, flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, 0))
    subgraphs = _table_vector(model_tab, 8)
    keys: list[str] = []
    lengths: set[int] = set()
    for sig in _table_vector(model_tab, 18):
        key = _string(sig, 8) or ""
        keys.append(key)
        if not key.startswith("prefill"):
            continue
        sub_index = _scalar(sig, 12, N.Uint32Flags, 0)
        tensors = _table_vector(subgraphs[sub_index], 4) if sub_index < len(subgraphs) else []
        inputs = {(_string(tm, 4) or ""): _scalar(tm, 6, N.Uint32Flags, 0)
                  for tm in _table_vector(sig, 4)}
        found = None
        for name, axis in (("input_pos", -1), ("tokens", -1), ("embeddings", 1)):
            if name in inputs and inputs[name] < len(tensors):
                tensor = tensors[inputs[name]]
                shape = _int32_vector(tensor, 18) or _int32_vector(tensor, 4) or ()
                if shape and shape[axis] > 0:
                    found = shape[axis]
                    break
        if found is None:
            m = re.search(r"(\d+)$", key)
            found = int(m.group(1)) if m else None
        if found:
            lengths.add(int(found))
    return sorted(lengths), keys


def run_describe(args: argparse.Namespace, out: dict[str, Any]) -> None:
    bin_dir = Path(sys.executable).parent
    model = Path(args.bundle)
    describe: dict[str, Any] = {}
    out["describe"] = describe

    out["stage"] = "describe-cli"
    code, stdout, stderr = _run_cli([str(bin_dir / "litert-lm"), "describe", str(model)])
    describe["describe_stdout"] = stdout
    out["describe_exit"] = code
    if code != 0:
        out["describe_stderr_tail"] = stderr[-600:]
    m = re.search(r"Max Context Tokens:\s+(-?\d+)", stdout)
    describe["max_context_tokens"] = int(m.group(1)) if m else "unverified"

    out["stage"] = "peek-cli"
    code, stdout, stderr = _run_cli([str(bin_dir / "litert-lm-peek"), "--litertlm_file",
                                     str(model)])
    out["peek_exit"] = code
    describe["sections"] = parse_peek_sections(stdout) if code == 0 else "unverified"
    if code != 0:
        out["peek_stderr_tail"] = stderr[-600:]

    out["stage"] = "container"
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from litert_compat.parser.litertlm import parse_litertlm

    with open(model, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as data:
        bundle = parse_litertlm(data)  # type: ignore[arg-type]
        describe["container_version"] = ".".join(str(v) for v in bundle.version)

        # The same field `litert-lm unpack --chat-template` writes out, read
        # in place: unpack itself dumps every section (GBs) to its output dir.
        out["stage"] = "llm-metadata"
        meta_sections = [s for s in bundle.sections if s.data_type == 5]
        template = None
        if meta_sections:
            from litert_lm_builder.runtime.proto import llm_metadata_pb2

            md = llm_metadata_pb2.LlmMetadata()
            md.ParseFromString(bytes(data[meta_sections[0].begin : meta_sections[0].end]))
            describe["llm_metadata"] = llm_metadata_fields(md)
            template = md.jinja_prompt_template or None
        else:
            describe["llm_metadata"] = "unverified"
        describe["template"] = template_flags(template)
        if template:
            (Path(args.work) / "chat_template.jinja").write_text(template, encoding="utf-8")

        out["stage"] = "prefill-signatures"
        prefill = [s for s in bundle.sections
                   if s.data_type == 3 and (s.model_type or "").endswith("prefill_decode")]
        if not prefill:
            describe["prefill_lengths"] = "unverified"
            describe["prefill_lengths_reason"] = "no tf_lite_prefill_decode section"
        else:
            def on_alarm(signum: int, frame: Any) -> None:
                raise _Deadline()

            previous = signal.signal(signal.SIGALRM, on_alarm)
            signal.alarm(PREFILL_READ_LIMIT_S)
            try:
                lengths, keys = prefill_lengths(model, data, prefill[0])
                describe["prefill_lengths"] = lengths or "unverified"
                describe["signature_keys"] = keys
                if not lengths:
                    describe["prefill_lengths_reason"] = "no prefill signature found"
            except _Deadline:
                describe["prefill_lengths"] = "unverified"
                describe["prefill_lengths_reason"] = (
                    f"signature read exceeded {PREFILL_READ_LIMIT_S} s")
            except Exception as exc:  # recorded, never fatal
                describe["prefill_lengths"] = "unverified"
                describe["prefill_lengths_reason"] = f"{type(exc).__name__}: {exc}"[:200]
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous)
    out["stage"] = "done"


# --- main ------------------------------------------------------------------------


def runtime_version() -> str:
    from importlib import metadata

    try:
        return metadata.version("litert-lm")
    except metadata.PackageNotFoundError:
        return "unverified"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bundle")
    ap.add_argument("check", choices=[c for c in SUITE_V1 if not CHECKS[c].get("derived")])
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=("cpu", "gpu"), default=None)
    ap.add_argument("--ctx", type=int, default=None)
    ap.add_argument("--part", choices=("shared", "control"), default=None)
    ap.add_argument("--work", default=".")
    ap.add_argument("--cache-dir", default=None,
                    help="engine cache_dir (default <work>/engine_cache)")
    args = ap.parse_args(argv)
    args.backend = args.backend or CHECKS[args.check]["backend"] or "cpu"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.work).mkdir(parents=True, exist_ok=True)
    # The runner's timeout clock starts here (after any quiet_wait wait).
    Path(str(out_path) + ".started").write_text(f"{time.time():.3f}\n", encoding="utf-8")

    out: dict[str, Any] = {"check": args.check, "part": args.part, "pid": os.getpid(),
                           "python": sys.version.split()[0], "error": None,
                           "stage": "import"}
    t0 = time.monotonic()
    code = 0
    try:
        out["runtime_version"] = runtime_version()
        if args.check == "describe":
            run_describe(args, out)
        else:
            run_chat(args, out)
    except BaseException as exc:  # every failure is a result
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["error_type"] = type(exc).__name__
        out["traceback"] = traceback.format_exc()[-4000:]
        code = 1
    out["elapsed_s"] = round(time.monotonic() - t0, 2)
    tmp = Path(str(out_path) + ".tmp")
    tmp.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=repr) + "\n",
                   encoding="utf-8")
    tmp.replace(out_path)
    sys.stdout.flush()
    return code


if __name__ == "__main__":
    sys.exit(main())
