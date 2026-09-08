"""edge-card CLI: exit-code contract, collected failures, index generation."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import REPO_ROOT
from litert_compat.cards.cli import app

runner = CliRunner()

EXAMPLES = REPO_ROOT / "data" / "examples"
BUILD_CLEAN = [
    "build",
    "--model", str(EXAMPLES / "model_clean_example.tflite"),
    "--meta", str(EXAMPLES / "meta_clean_example.yaml"),
    "--bench", str(EXAMPLES / "bench_clean_example.json"),
    "--lint", str(EXAMPLES / "lint_clean_example.json"),
    "--cross-bench", str(EXAMPLES / "cross_runtime_clean_example.json"),
]


def invoke(*args: str):
    return runner.invoke(app, list(args))


class TestBuild:
    def test_build_ok(self, tmp_path: Path) -> None:
        result = invoke(*BUILD_CLEAN, "-o", str(tmp_path / "card"))
        assert result.exit_code == 0, result.output
        assert (tmp_path / "card" / "card.json").is_file()
        assert (tmp_path / "card" / "CARD.md").is_file()

    def test_invalid_bench_is_exit_1_nothing_written(self, tmp_path: Path) -> None:
        bad_bench = tmp_path / "bench.json"
        bad_bench.write_text('{"schema_version": "1.0"}', encoding="utf-8")
        out = tmp_path / "card"
        result = invoke(
            "build",
            "--model", str(EXAMPLES / "model_clean_example.tflite"),
            "--meta", str(EXAMPLES / "meta_clean_example.yaml"),
            "--bench", str(bad_bench),
            "-o", str(out),
        )
        assert result.exit_code == 1
        assert not out.exists()

    def test_missing_model_file_is_usage_error(self, tmp_path: Path) -> None:
        result = invoke(
            "build",
            "--model", str(tmp_path / "nope.tflite"),
            "--meta", str(EXAMPLES / "meta_clean_example.yaml"),
            "-o", str(tmp_path / "card"),
        )
        assert result.exit_code == 2

    def test_deterministic_across_runs(self, tmp_path: Path) -> None:
        for name in ("a", "b"):
            assert invoke(*BUILD_CLEAN, "-o", str(tmp_path / name)).exit_code == 0
        for filename in ("card.json", "CARD.md"):
            assert (tmp_path / "a" / filename).read_bytes() == (
                tmp_path / "b" / filename
            ).read_bytes()


class TestBuildAll:
    def test_collected_failures(self, tmp_path: Path) -> None:
        """The example manifest's third row is deliberately broken: 2 built, exit 1."""
        out = tmp_path / "cards"
        result = invoke(
            "build-all",
            "--manifest", str(EXAMPLES / "cards_manifest_example.csv"),
            "--out", str(out),
        )
        assert result.exit_code == 1
        assert "built 2 of 3" in result.output
        assert "model_missing_example.tflite" in result.output
        assert sorted(p.name for p in out.iterdir()) == [
            "example-tiny-clean",
            "example-tiny-mixed",
        ]

    def test_all_good_manifest_exits_0(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.csv"
        good_rows = (
            (EXAMPLES / "cards_manifest_example.csv")
            .read_text(encoding="utf-8")
            .splitlines()[:3]
        )
        manifest.write_text("\n".join(good_rows) + "\n", encoding="utf-8")
        for name in ("model_clean_example.tflite", "meta_clean_example.yaml",
                     "bench_clean_example.json", "lint_clean_example.json",
                     "cross_runtime_clean_example.json", "model_mixed_example.tflite",
                     "meta_mixed_example.yaml", "lint_mixed_example.json"):
            shutil.copy(EXAMPLES / name, tmp_path / name)
        result = invoke("build-all", "--manifest", str(manifest), "--out", str(tmp_path / "out"))
        assert result.exit_code == 0, result.output
        assert "built 2 of 2" in result.output

    def test_headerless_manifest_is_usage_error(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.csv"
        manifest.write_text("a,b\n1,2\n", encoding="utf-8")
        result = invoke("build-all", "--manifest", str(manifest), "--out", str(tmp_path / "out"))
        assert result.exit_code == 2


@pytest.fixture()
def built_tree(tmp_path: Path) -> Path:
    """A tmp repo root with cards/ built from the two example models."""
    cards = tmp_path / "cards"
    result = invoke(
        "build-all",
        "--manifest", str(EXAMPLES / "cards_manifest_example.csv"),
        "--out", str(cards),
    )
    assert result.exit_code == 1  # broken row collected; two cards built
    return tmp_path


class TestIndex:
    def test_outputs_written_and_deterministic(self, built_tree: Path) -> None:
        outputs = ("cards/index.json", "cards/README.md", "llms.txt")
        contents = []
        for _ in range(2):
            assert invoke("index", str(built_tree / "cards")).exit_code == 0
            contents.append([(built_tree / f).read_bytes() for f in outputs])
        assert contents[0] == contents[1]
        index_doc = (built_tree / "cards" / "index.json").read_text(encoding="utf-8")
        assert '"example-tiny-clean"' in index_doc
        llms = (built_tree / "llms.txt").read_text(encoding="utf-8")
        assert "cards/example-tiny-clean/CARD.md" in llms
        assert "none yet" in llms  # no matrix snapshots in the tmp tree

    def test_matrix_snapshots_listed(self, built_tree: Path) -> None:
        matrix_dir = built_tree / "data" / "matrix"
        matrix_dir.mkdir(parents=True)
        shutil.copy(EXAMPLES / "matrix_example.json", matrix_dir / "gpu_mldrift__0.0.0.json")
        assert invoke("index", str(built_tree / "cards")).exit_code == 0
        llms = (built_tree / "llms.txt").read_text(encoding="utf-8")
        assert "data/matrix/gpu_mldrift__0.0.0.json" in llms

    def test_invalid_card_is_exit_1_nothing_written(self, built_tree: Path) -> None:
        card_json = built_tree / "cards" / "example-tiny-clean" / "card.json"
        card_json.write_text('{"schema_version": "1.0"}', encoding="utf-8")
        result = invoke("index", str(built_tree / "cards"))
        assert result.exit_code == 1
        assert not (built_tree / "cards" / "index.json").exists()

    def test_id_directory_mismatch_is_exit_1(self, built_tree: Path) -> None:
        src = built_tree / "cards" / "example-tiny-clean"
        src.rename(built_tree / "cards" / "wrong-name")
        result = invoke("index", str(built_tree / "cards"))
        assert result.exit_code == 1
