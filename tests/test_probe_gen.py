"""Probe generator: templates, determinism, round-trip verification, CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from helpers import entry, make_doc
from litert_compat.cli import app
from litert_compat.lint.classify import node_signature
from litert_compat.matrix.canonical import write_canonical
from litert_compat.parser.builder import build_tflite
from litert_compat.parser.reader import parse_tflite
from litert_compat.probe.gen import (
    BINARY_ELEMENTWISE,
    UNARY_ELEMENTWISE,
    build_probe_graph,
    generate_fixture,
)
from litert_compat.probe.signatures import (
    DEFAULT_PROBE_RANK,
    ProbeSignature,
    UnprobeableSignatureError,
    make_signature,
    shape_meta_from_constraints,
    signature_from_entry,
    signatures_from_lint_report,
)

GOLDEN_SPLIT = Path(__file__).resolve().parent / "golden" / "split.json"

cli = CliRunner()


def sig(
    op: str = "ABS",
    dtypes: tuple[str, ...] = ("float32",),
    meta: dict | None = None,
    backend: str = "gpu_mldrift",
) -> ProbeSignature:
    return make_signature(
        backend, op, list(dtypes), meta or {"dynamic_shape": False, "rank": 4}
    )


ALL_TEMPLATE_OPS = sorted(UNARY_ELEMENTWISE) + sorted(BINARY_ELEMENTWISE) + [
    "SOFTMAX",
    "RESHAPE",
]


@pytest.mark.parametrize("op", ALL_TEMPLATE_OPS)
@pytest.mark.parametrize("rank", [1, 2, 4])
def test_every_template_round_trips_to_its_signature(op: str, rank: int) -> None:
    s = sig(op, meta={"dynamic_shape": False, "rank": rank})
    data = build_tflite(build_probe_graph(s))
    subgraph = parse_tflite(data).subgraphs[0]
    assert len(subgraph.nodes) == 1
    assert subgraph.nodes[0].op == op
    dtypes, shape_meta = node_signature(subgraph, subgraph.nodes[0])
    assert dtypes == ["float32"]
    assert shape_meta == {"dynamic_shape": False, "rank": rank}


def test_dynamic_unary_round_trips() -> None:
    s = sig("NEG", meta={"dynamic_shape": True, "rank": 2})
    subgraph = parse_tflite(build_tflite(build_probe_graph(s))).subgraphs[0]
    dtypes, shape_meta = node_signature(subgraph, subgraph.nodes[0])
    assert (dtypes, shape_meta) == (["float32"], {"dynamic_shape": True, "rank": 2})


def test_rank5_and_rank0_unary() -> None:
    for rank in (0, 5):
        s = sig("ABS", meta={"dynamic_shape": False, "rank": rank})
        subgraph = parse_tflite(build_tflite(build_probe_graph(s))).subgraphs[0]
        _, shape_meta = node_signature(subgraph, subgraph.nodes[0])
        assert shape_meta["rank" if rank else "dynamic_shape"] in (rank, False)


def test_generation_is_deterministic(tmp_path: Path) -> None:
    s = sig("RESHAPE", meta={"dynamic_shape": False, "rank": 2})
    first = generate_fixture(s, tmp_path / "a")
    second = generate_fixture(s, tmp_path / "b")
    assert first.fixture_id == second.fixture_id == s.fixture_id
    assert first.path.read_bytes() == second.path.read_bytes()
    # Regeneration into the same directory is byte-stable.
    again = generate_fixture(s, tmp_path / "a")
    assert again.path == first.path
    assert again.path.read_bytes() == first.path.read_bytes()


def test_fixture_id_stable_and_meta_sensitive() -> None:
    a = sig("SOFTMAX", meta={"dynamic_shape": False, "rank": 2})
    b = sig("SOFTMAX", meta={"dynamic_shape": False, "rank": 4})
    assert a.fixture_id.startswith("SOFTMAX__float32__")
    assert a.fixture_id != b.fixture_id
    assert a.fixture_id == sig("SOFTMAX", meta={"rank": 2, "dynamic_shape": False}).fixture_id


@pytest.mark.parametrize(
    ("probe_sig", "reason_part"),
    [
        (sig("LSTM"), "no probe template"),
        (sig("ABS", dtypes=("float32", "int8")), "multi-dtype"),
        (sig("ABS", dtypes=()), "no dtypes"),
        (sig("ABS", meta={"dynamic_shape": False}), "unknown output rank"),
        (sig("ABS", meta={"dynamic_shape": True, "rank": 0}), "rank >= 1"),
        (sig("SOFTMAX", meta={"dynamic_shape": False, "rank": 0}), "SOFTMAX requires rank"),
        (sig("RESHAPE", meta={"dynamic_shape": True, "rank": 2}), "dynamic RESHAPE"),
        (sig("ABS", dtypes=("float99",)), "unknown dtype"),
        # Weighted-template honesty limits (the graduated families).
        (sig("CONV_2D", meta={"dynamic_shape": True, "rank": 4}), "dynamic-shape CONV_2D"),
        (sig("CONV_2D", meta={"dynamic_shape": False, "rank": 3}), "requires rank 4"),
        (sig("FULLY_CONNECTED", meta={"dynamic_shape": False, "rank": 1}), "rank >= 2"),
        (sig("GREATER", dtypes=("float32",)), "output dtype is bool"),
        (sig("SELECT", dtypes=("int64",)), "float32/int32"),
        (sig("EMBEDDING_LOOKUP", meta={"dynamic_shape": False, "rank": 1}), "rank >= 2"),
        # Quantized-template honesty limits (the graduated deferral).
        (sig("DEQUANTIZE", dtypes=("int8",), meta={"dynamic_shape": False, "rank": 2}),
         "dequantizes float16 to float32"),
        (sig("POW", dtypes=("int8",), meta={"dynamic_shape": False, "rank": 2}),
         "no int8 quantized kernel"),
        (sig("TRANSPOSE_CONV", dtypes=("int8",)), "fp32 activations"),
        (sig("FULLY_CONNECTED", dtypes=("int8",)), "requires rank 2"),
        (sig("TRANSPOSE", dtypes=("int8",), meta={"dynamic_shape": False, "rank": 5}),
         "rank <= 4"),
        (sig("TANH", dtypes=("int8",), meta={"dynamic_shape": False, "rank": 2}),
         "fixed output scale"),
        (sig("SOFTMAX", dtypes=("int8",), meta={"dynamic_shape": False, "rank": 2}),
         "fixed output scale"),
    ],
)
def test_unprobeable_signatures_are_refused(
    probe_sig: ProbeSignature, reason_part: str
) -> None:
    with pytest.raises(UnprobeableSignatureError, match=reason_part):
        build_probe_graph(probe_sig)


# --- quantized templates (the graduated quantized-weight deferral) -----------


def test_dequantize_template_is_float16_to_float32() -> None:
    s = sig("DEQUANTIZE", meta={"dynamic_shape": False, "rank": 2})
    graph = build_probe_graph(s)
    assert [t.dtype for t in graph.tensors] == ["float16", "float32"]
    assert graph.ops[0].version == 3
    subgraph = parse_tflite(build_tflite(graph)).subgraphs[0]
    dtypes, shape_meta = node_signature(subgraph, subgraph.nodes[0])
    assert (dtypes, shape_meta) == (["float32"], {"dynamic_shape": False, "rank": 2})


@pytest.mark.parametrize(
    ("op", "rank"),
    [
        ("CONV_2D", 4),
        ("DEPTHWISE_CONV_2D", 4),
        ("FULLY_CONNECTED", 2),
        ("AVERAGE_POOL_2D", 4),
        ("MAX_POOL_2D", 4),
        ("ADD", 4),
        ("MUL", 2),
        ("SUB", 3),
        ("SQUARED_DIFFERENCE", 2),
        ("PAD", 4),
        ("TRANSPOSE", 4),
        ("RESHAPE", 2),
        ("HARD_SWISH", 2),
        ("RELU", 4),
    ],
)
def test_int8_templates_carry_quantization_and_round_trip(op: str, rank: int) -> None:
    s = sig(op, dtypes=("int8",), meta={"dynamic_shape": False, "rank": rank})
    graph = build_probe_graph(s)
    int8_tensors = [t for t in graph.tensors if t.dtype == "int8"]
    assert int8_tensors and all(t.quantization is not None for t in int8_tensors)
    # int32 bias (weighted families) carries qparams too; index vectors don't.
    for tensor in graph.tensors:
        if tensor.name == "bias":
            assert tensor.dtype == "int32" and tensor.quantization is not None
    subgraph = parse_tflite(build_tflite(graph)).subgraphs[0]
    dtypes, shape_meta = node_signature(subgraph, subgraph.nodes[0])
    assert (dtypes, shape_meta) == (["int8"], {"dynamic_shape": False, "rank": rank})


def test_int8_conv_filter_is_per_channel_symmetric() -> None:
    s = sig("CONV_2D", dtypes=("int8",), meta={"dynamic_shape": False, "rank": 4})
    graph = build_probe_graph(s)
    filter_tensor = next(t for t in graph.tensors if t.name == "filter")
    out_ch = filter_tensor.shape[0]
    quant = filter_tensor.quantization
    assert quant is not None
    assert len(quant.scale) == out_ch
    assert quant.zero_point == (0,) * out_ch
    # bias_scale == input_scale * filter_scale, exactly (power-of-two scales).
    bias = next(t for t in graph.tensors if t.name == "bias")
    input_tensor = next(t for t in graph.tensors if t.name == "input")
    assert bias.quantization is not None and input_tensor.quantization is not None
    assert bias.quantization.scale[0] == input_tensor.quantization.scale[0] * quant.scale[0]


# One case per weighted-template family, at the ranks the real needs_probe
# lists carry (the 2026-08-11 release reports); dtypes follow the signature
# contract (outputs): bool for comparisons, int64 for the widening CAST.
WEIGHTED_CASES = [
    ("CONV_2D", ("float32",), 4),
    ("DEPTHWISE_CONV_2D", ("float32",), 4),
    ("TRANSPOSE_CONV", ("float32",), 4),
    ("AVERAGE_POOL_2D", ("float32",), 4),
    ("MAX_POOL_2D", ("float32",), 4),
    ("RESIZE_BILINEAR", ("float32",), 4),
    ("RESIZE_NEAREST_NEIGHBOR", ("float32",), 4),
    ("FULLY_CONNECTED", ("float32",), 2),
    ("FULLY_CONNECTED", ("float32",), 3),
    ("FULLY_CONNECTED", ("float32",), 4),
    ("BATCH_MATMUL", ("float32",), 3),
    ("CONCATENATION", ("float32",), 2),
    ("TRANSPOSE", ("float32",), 5),
    ("SLICE", ("float32",), 3),
    ("SUM", ("float32",), 1),
    ("MEAN", ("float32",), 1),
    ("REDUCE_MAX", ("float32",), 2),
    ("REDUCE_MIN", ("float32",), 3),
    ("PAD", ("float32",), 3),
    ("LEAKY_RELU", ("float32",), 3),
    ("CAST", ("float32",), 2),
    ("CAST", ("int32",), 2),
    ("CAST", ("int64",), 2),
    ("GREATER", ("bool",), 4),
    ("EQUAL", ("bool",), 2),
    ("SELECT", ("float32",), 4),
    ("SELECT_V2", ("float32",), 2),
    ("EMBEDDING_LOOKUP", ("float32",), 2),
]


@pytest.mark.parametrize(("op", "dtypes", "rank"), WEIGHTED_CASES)
def test_weighted_templates_round_trip_to_their_signatures(
    op: str, dtypes: tuple[str, ...], rank: int
) -> None:
    s = sig(op, dtypes=dtypes, meta={"dynamic_shape": False, "rank": rank})
    data = build_tflite(build_probe_graph(s))
    subgraph = parse_tflite(data).subgraphs[0]
    assert len(subgraph.nodes) == 1
    assert subgraph.nodes[0].op == op
    node_dtypes, shape_meta = node_signature(subgraph, subgraph.nodes[0])
    assert tuple(node_dtypes) == dtypes
    assert shape_meta == {"dynamic_shape": False, "rank": rank}


def test_weighted_generation_is_deterministic(tmp_path: Path) -> None:
    """Constant weights are seeded from the signature: same signature, same
    bytes — including regeneration into a fresh directory."""
    s = sig("CONV_2D", meta={"dynamic_shape": False, "rank": 4})
    first = generate_fixture(s, tmp_path / "a")
    second = generate_fixture(s, tmp_path / "b")
    assert first.path.read_bytes() == second.path.read_bytes()
    other = sig("DEPTHWISE_CONV_2D", meta={"dynamic_shape": False, "rank": 4})
    assert (
        generate_fixture(other, tmp_path / "c").path.read_bytes()
        != first.path.read_bytes()
    )


def test_shape_meta_from_constraints_mapping() -> None:
    assert shape_meta_from_constraints(None) == {
        "dynamic_shape": False,
        "rank": DEFAULT_PROBE_RANK,
    }
    assert shape_meta_from_constraints({"rank": 2, "max_rank": 5}) == {
        "dynamic_shape": False,
        "rank": 2,
    }
    assert shape_meta_from_constraints({"max_rank": 3, "min_rank": 1}) == {
        "dynamic_shape": False,
        "rank": 3,
    }
    assert shape_meta_from_constraints({"min_rank": 2, "dynamic_shape": True}) == {
        "dynamic_shape": True,
        "rank": 2,
    }
    with pytest.raises(UnprobeableSignatureError, match="keep_dims"):
        shape_meta_from_constraints({"keep_dims": True})
    with pytest.raises(UnprobeableSignatureError, match="not a valid rank"):
        shape_meta_from_constraints({"rank": True})


def test_signature_from_entry_keeps_entry_identity_out() -> None:
    matrix_entry = entry(
        op="TANH",
        provenance="inferred",
        dtypes=["float16"],
        constraints={"dynamic_shape": True, "max_rank": 4},
    )
    s = signature_from_entry("gpu_mldrift", matrix_entry)
    assert s.op == "TANH"
    assert s.dtypes == ("float16",)
    assert s.shape_meta == {"dynamic_shape": True, "rank": 4}


def test_signatures_from_golden_lint_report() -> None:
    doc = json.loads(GOLDEN_SPLIT.read_text(encoding="utf-8"))
    sigs = signatures_from_lint_report(doc)
    assert [(s.op, s.backend) for s in sigs] == [
        ("RESHAPE", "gpu_mldrift"),
        ("SOFTMAX", "gpu_mldrift"),
    ]
    assert all(s.shape_meta == {"dynamic_shape": False, "rank": 2} for s in sigs)


def test_cli_gen_from_lint(tmp_path: Path) -> None:
    result = cli.invoke(
        app, ["probe", "gen", "--from-lint", str(GOLDEN_SPLIT), "-o", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert result.output.count("generated") == 2
    assert (tmp_path / f"{sig('RESHAPE', meta={'dynamic_shape': False, 'rank': 2}).fixture_id}"
            ".tflite").exists()


def test_cli_gen_json_output_and_unprobeable_exit_1(tmp_path: Path) -> None:
    doc = make_doc(
        [
            entry(op="NEG", provenance="inferred", dtypes=["float32"]),
            entry(
                op="MEAN",
                provenance="inferred",
                dtypes=["float32"],
                constraints={"keep_dims": True},
            ),
            entry(op="LSTM", provenance="inferred", dtypes=["float32"]),
            entry(op="CONV_2D", provenance="example", dtypes=["float32"]),  # not inferred
        ]
    )
    matrix_path = tmp_path / "snapshot.json"
    write_canonical(doc, matrix_path)
    result = cli.invoke(
        app,
        ["probe", "gen", "--from-matrix", str(matrix_path), "-o", str(tmp_path / "out"),
         "--json"],
    )
    assert result.exit_code == 1, result.output
    rows = json.loads(result.output)
    by_status = {row["op"]: row["status"] for row in rows}
    assert by_status == {"LSTM": "unprobeable", "MEAN": "unprobeable", "NEG": "generated"}


def test_cli_gen_no_inputs_exit_2() -> None:
    assert cli.invoke(app, ["probe", "gen"]).exit_code == 2


def test_cli_gen_bad_lint_report_exit_2(tmp_path: Path) -> None:
    bad = tmp_path / "not_a_report.json"
    bad.write_text("{}", encoding="utf-8")
    assert cli.invoke(app, ["probe", "gen", "--from-lint", str(bad)]).exit_code == 2
