"""Card assembly: meta parsing, input identity checks, delegation mapping, determinism."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from conftest import REPO_ROOT
from litert_compat.cards.build import CardBuildError, build_card
from litert_compat.cards.meta import MetaError, load_meta
from litert_compat.cards.schema_io import schema_errors
from litert_compat.matrix.canonical import canonical_dumps, load_json

EXAMPLES = REPO_ROOT / "data" / "examples"


@pytest.fixture()
def meta_clean() -> dict[str, Any]:
    return load_meta(EXAMPLES / "meta_clean_example.yaml")[0]


@pytest.fixture()
def bench_clean() -> dict[str, Any]:
    return load_json(EXAMPLES / "bench_clean_example.json")


@pytest.fixture()
def lint_clean() -> dict[str, Any]:
    return load_json(EXAMPLES / "lint_clean_example.json")


@pytest.fixture()
def cross_clean() -> dict[str, Any]:
    return load_json(EXAMPLES / "cross_runtime_clean_example.json")


MODEL_CLEAN = EXAMPLES / "model_clean_example.tflite"


class TestMeta:
    def test_loads_example_meta(self, meta_clean: dict[str, Any]) -> None:
        assert meta_clean["model"]["id"] == "example-tiny-clean"
        assert meta_clean["conversion"]["tool"] == "example-converter"
        assert len(meta_clean["pitfalls"]) == 1

    def test_missing_pitfalls_defaults_to_empty(self) -> None:
        meta, _ = load_meta(EXAMPLES / "meta_mixed_example.yaml")
        assert meta["pitfalls"] == []

    def test_unknown_keys_warn_not_fail(self, tmp_path: Path) -> None:
        path = tmp_path / "meta.yaml"
        source = (EXAMPLES / "meta_mixed_example.yaml").read_text(encoding="utf-8")
        path.write_text(source + "\nextra_key: 1\n", encoding="utf-8")
        _, warnings = load_meta(path)
        assert any("extra_key" in w for w in warnings)

    def test_all_defects_collected(self, tmp_path: Path) -> None:
        path = tmp_path / "meta.yaml"
        path.write_text("model:\n  id: Bad_ID\nconversion: {}\n", encoding="utf-8")
        with pytest.raises(MetaError) as exc:
            load_meta(path)
        messages = "\n".join(exc.value.errors)
        assert "model.id" in messages  # pattern violation
        assert "model.family" in messages  # missing
        assert "conversion.tool" in messages  # missing

    def test_non_mapping_is_error(self, tmp_path: Path) -> None:
        path = tmp_path / "meta.yaml"
        path.write_text("- just\n- a list\n", encoding="utf-8")
        with pytest.raises(MetaError):
            load_meta(path)


class TestBuildCard:
    def test_full_card_validates(
        self,
        meta_clean: dict[str, Any],
        bench_clean: dict[str, Any],
        lint_clean: dict[str, Any],
        cross_clean: dict[str, Any],
    ) -> None:
        card = build_card(MODEL_CLEAN, meta_clean, bench_clean, lint_clean, cross_clean)
        assert schema_errors(card, "card.schema.json") == []
        assert card["benchmarks"] == bench_clean["results"]  # verbatim, never reformatted
        assert card["cross_runtime"] == cross_clean["results"]

    def test_delegation_mapped_from_lint_report(
        self, meta_clean: dict[str, Any], lint_clean: dict[str, Any]
    ) -> None:
        card = build_card(MODEL_CLEAN, meta_clean, lint_report=lint_clean)
        delegation = card["delegation"]
        assert delegation["backend"] == lint_clean["backend"]
        assert delegation["litert_version"] == lint_clean["litert_version"]
        assert delegation["coverage_ops_pct"] == lint_clean["summary"]["coverage_ops_pct"]
        assert delegation["partitions"] == lint_clean["summary"]["partition_count"]
        assert (
            delegation["matched_provenance_counts"]
            == lint_clean["summary"]["matched_provenance_counts"]
        )
        assert delegation["lint_report_version"] == lint_clean["schema_version"]

    def test_blocking_ops_ranked_and_deduplicated(self, meta_clean: dict[str, Any]) -> None:
        lint_mixed = load_json(EXAMPLES / "lint_mixed_example.json")
        meta_mixed, _ = load_meta(EXAMPLES / "meta_mixed_example.yaml")
        card = build_card(EXAMPLES / "model_mixed_example.tflite", meta_mixed,
                          lint_report=lint_mixed)
        ranked_ops = [b["op"] for b in lint_mixed["blocking_ops"]]
        expected: list[str] = []
        for op in ranked_ops:
            if op not in expected:
                expected.append(op)
        assert card["delegation"]["blocking_ops"] == expected

    def test_without_optional_inputs(self, meta_clean: dict[str, Any]) -> None:
        card = build_card(MODEL_CLEAN, meta_clean)
        assert card["benchmarks"] == []
        assert card["delegation"] is None
        assert card["cross_runtime"] == []
        assert schema_errors(card, "card.schema.json") == []

    def test_bench_model_id_mismatch_refused(
        self, meta_clean: dict[str, Any], bench_clean: dict[str, Any]
    ) -> None:
        bench_clean["model_id"] = "some-other-model"
        with pytest.raises(CardBuildError, match="some-other-model"):
            build_card(MODEL_CLEAN, meta_clean, bench_clean)

    def test_lint_sha256_mismatch_refused(
        self, meta_clean: dict[str, Any], lint_clean: dict[str, Any]
    ) -> None:
        lint_clean["model"]["sha256"] = "0" * 64
        with pytest.raises(CardBuildError, match="different model"):
            build_card(MODEL_CLEAN, meta_clean, lint_report=lint_clean)

    def test_cross_record_without_runtime_refused(
        self, meta_clean: dict[str, Any], cross_clean: dict[str, Any]
    ) -> None:
        del cross_clean["results"][0]["runtime"]
        with pytest.raises(CardBuildError, match="runtime"):
            build_card(MODEL_CLEAN, meta_clean, cross_bench=cross_clean)

    def test_deterministic(
        self,
        meta_clean: dict[str, Any],
        bench_clean: dict[str, Any],
        lint_clean: dict[str, Any],
    ) -> None:
        dumps = [
            canonical_dumps(
                build_card(
                    MODEL_CLEAN,
                    json.loads(json.dumps(meta_clean)),
                    json.loads(json.dumps(bench_clean)),
                    json.loads(json.dumps(lint_clean)),
                )
            )
            for _ in range(2)
        ]
        assert dumps[0] == dumps[1]


class TestRemoteArtifact:
    REMOTE = {"file": "granite-x_int4.litertlm", "sha256": "ab" * 32, "size_bytes": 2191445936}

    def test_remote_artifact_builds_without_local_bytes(
        self, meta_clean: dict[str, Any]
    ) -> None:
        card = build_card(None, meta_clean, remote_artifact=self.REMOTE)
        assert schema_errors(card, "card.schema.json") == []
        artifact = card["artifacts"][0]
        assert artifact["file"] == self.REMOTE["file"]
        assert artifact["sha256"] == self.REMOTE["sha256"]
        assert artifact["size_mb"] == round(self.REMOTE["size_bytes"] / (1024 * 1024), 3)

    def test_exactly_one_identity_source(self, meta_clean: dict[str, Any]) -> None:
        with pytest.raises(CardBuildError):
            build_card(None, meta_clean)
        with pytest.raises(CardBuildError):
            build_card(MODEL_CLEAN, meta_clean, remote_artifact=self.REMOTE)

    def test_remote_artifact_missing_keys_collected(self, meta_clean: dict[str, Any]) -> None:
        with pytest.raises(CardBuildError) as excinfo:
            build_card(None, meta_clean, remote_artifact={"file": "x.litertlm"})
        assert "sha256" in excinfo.value.errors[0]

    def test_lint_identity_checked_against_remote_sha(
        self, meta_clean: dict[str, Any], lint_clean: dict[str, Any]
    ) -> None:
        with pytest.raises(CardBuildError) as excinfo:
            build_card(None, meta_clean, lint_report=lint_clean, remote_artifact=dict(
                self.REMOTE))
        # remote sha aa.. never matches the example lint report's sha
        assert any("different model" in e or "exactly one" in e for e in excinfo.value.errors)


class TestBenchmarkSchema:
    def test_example_files_validate(
        self, bench_clean: dict[str, Any], cross_clean: dict[str, Any]
    ) -> None:
        assert schema_errors(bench_clean, "benchmark_result.schema.json") == []
        assert schema_errors(cross_clean, "benchmark_result.schema.json") == []

    def test_missing_provenance_rejected(self, bench_clean: dict[str, Any]) -> None:
        del bench_clean["results"][0]["provenance"]
        assert schema_errors(bench_clean, "benchmark_result.schema.json") != []

    def test_unknown_record_field_rejected(self, bench_clean: dict[str, Any]) -> None:
        bench_clean["results"][0]["surprise"] = 1
        assert schema_errors(bench_clean, "benchmark_result.schema.json") != []

    def test_devicemark_shaped_record_fits(self) -> None:
        """The devicemark JSONL columns map onto the record without schema changes."""
        doc = {
            "schema_version": "1.0",
            "model_id": "example-tiny-clean",
            "results": [
                {
                    "device": "Example Device",
                    "backend": "gpu_mldrift",
                    "runtime": "example-runtime",
                    "tokens_per_s": 210,
                    "peak_mem_mb": 443,
                    "metrics": {"power_w": None, "mem_measured": True},
                    "harness": "example-harness",
                    "date": "2026-08-10",
                    "provenance": "example",
                }
            ],
        }
        assert schema_errors(doc, "benchmark_result.schema.json") == []
