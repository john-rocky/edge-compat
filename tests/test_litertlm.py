"""LiteRT-LM container reader: TOC parsing, extraction identity, refusals,
and the edge-lint wiring (deferred-item graduation, 2026-08-12).

The committed example bundle is byte-pinned to the writer's output, mirroring
the .tflite fixture discipline. A skip-marked integration test lints a real
local bundle when one exists (no golden pinning on private local files).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from typer.testing import CliRunner

from litert_compat.lint.cli import app as lint_app
from litert_compat.lint.report import load_lint_report_schema
from litert_compat.parser.litertlm import (
    TFLITE_MODEL,
    LitertlmParseError,
    build_bundle,
    build_example_bundle,
    extract_tflite_model,
    parse_litertlm,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "data" / "examples"
BUNDLE = EXAMPLES / "model_bundle_example.litertlm"
REAL_BUNDLE = Path.home() / "code/litertlm-convert/out/qwen3_1_7b_int8/model.litertlm"

runner = CliRunner()


def test_committed_bundle_fixture_is_writer_output() -> None:
    assert BUNDLE.read_bytes() == build_example_bundle(EXAMPLES), (
        "fixture out of date: python -m litert_compat.parser.litertlm"
    )


def test_toc_parses_sections_and_items() -> None:
    bundle = parse_litertlm(BUNDLE.read_bytes())
    assert bundle.version == (1, 5, 0)
    assert [s.data_type_name for s in bundle.sections] == [
        "GenericBinaryData",
        "TFLiteModel",
    ]
    assert bundle.sections[1].items == {"model_type": "tf_lite_prefill_decode"}
    assert bundle.sections[1].model_type == "tf_lite_prefill_decode"


def test_extraction_is_byte_identical_to_the_wrapped_model() -> None:
    extracted, section = extract_tflite_model(BUNDLE.read_bytes())
    assert extracted == (EXAMPLES / "model_clean_example.tflite").read_bytes()
    assert section.index == 1


def test_refuses_non_bundle_bytes() -> None:
    with pytest.raises(LitertlmParseError, match="not a LiteRT-LM container"):
        parse_litertlm((EXAMPLES / "model_clean_example.tflite").read_bytes())


def test_refuses_truncated_container() -> None:
    data = BUNDLE.read_bytes()
    with pytest.raises(LitertlmParseError, match="outside the file"):
        parse_litertlm(data[:40])  # header end offset points past EOF
    with pytest.raises(LitertlmParseError):
        parse_litertlm(data[:16])


def test_refuses_section_offsets_outside_file() -> None:
    # Keep the header region, drop the payloads: TOC offsets now dangle.
    data = BUNDLE.read_bytes()[:4100]
    with pytest.raises(LitertlmParseError, match="outside the file"):
        parse_litertlm(data)


def test_refuses_zero_and_ambiguous_model_sections() -> None:
    model = (EXAMPLES / "model_clean_example.tflite").read_bytes()
    no_model = build_bundle([(1, {}, b"payload")])
    with pytest.raises(LitertlmParseError, match="no TFLiteModel section"):
        extract_tflite_model(no_model)

    two = build_bundle(
        [
            (TFLITE_MODEL, {"model_type": "tf_lite_prefill_decode"}, model),
            (TFLITE_MODEL, {"model_type": "tf_lite_vision_encoder"}, b"other"),
        ]
    )
    with pytest.raises(LitertlmParseError, match="ambiguous: 2 TFLiteModel"):
        extract_tflite_model(two)
    # A model_type selector that matches exactly one resolves the ambiguity.
    chosen, section = extract_tflite_model(two, model_type="tf_lite_prefill_decode")
    assert chosen == model and section.index == 0
    with pytest.raises(LitertlmParseError, match="no TFLiteModel section with"):
        extract_tflite_model(two, model_type="tf_lite_audio")


def test_lint_cli_lints_the_embedded_model() -> None:
    result = runner.invoke(
        lint_app,
        [str(BUNDLE), "--matrix", str(EXAMPLES / "matrix_example.json"), "--json"],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert list(Draft202012Validator(load_lint_report_schema()).iter_errors(report)) == []
    assert report["model"]["path"].endswith(
        "model_bundle_example.litertlm::TFLiteModel@section1(tf_lite_prefill_decode)"
    )
    embedded = (EXAMPLES / "model_clean_example.tflite").read_bytes()
    assert report["model"]["sha256"] == hashlib.sha256(embedded).hexdigest()


def test_lint_cli_refuses_broken_bundle(tmp_path: Path) -> None:
    broken = tmp_path / "broken.litertlm"
    broken.write_bytes(BUNDLE.read_bytes()[:40])
    result = runner.invoke(
        lint_app, [str(broken), "--matrix", str(EXAMPLES / "matrix_example.json")]
    )
    assert result.exit_code == 2
    assert "outside the file" in result.output


@pytest.mark.skipif(not REAL_BUNDLE.is_file(), reason="real local bundle not present")
def test_real_bundle_lints_end_to_end() -> None:
    snapshot = REPO_ROOT / "data" / "matrix" / "gpu_metal_mac__0.14.0.json"
    result = runner.invoke(
        lint_app, [str(REAL_BUNDLE), "--matrix", str(snapshot), "--json"]
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert list(Draft202012Validator(load_lint_report_schema()).iter_errors(report)) == []
    assert "::TFLiteModel@section" in report["model"]["path"]
    assert report["summary"]["total_ops"] > 0
