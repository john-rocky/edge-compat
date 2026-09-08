"""Partition simulation and impact ranking on hand-built classified nodes."""

from __future__ import annotations

from litert_compat.lint.classify import ClassifiedNode
from litert_compat.lint.partition import rank_blocking_ops, simulate_partitions
from litert_compat.matrix.query import Verdict
from litert_compat.parser.reader import Node


def make_node(
    index: int, status: str, elements: int = 10, subgraph: int = 0, op: str = "CONV_2D"
) -> ClassifiedNode:
    return ClassifiedNode(
        subgraph=subgraph,
        node=Node(index=index, op=op, custom_code=None, op_version=1, inputs=(), outputs=()),
        dtypes=["float32"],
        shape_meta={"dynamic_shape": False, "rank": 4},
        verdict=Verdict(
            status=status,
            matched_entry=None,
            reason="test",
            backend="gpu_mldrift",
            litert_version="0.0.0-example",
        ),
        output_elements=elements,
    )


def statuses_to_nodes(statuses: list[str]) -> list[ClassifiedNode]:
    return [make_node(i, s) for i, s in enumerate(statuses)]


def test_all_claimed_single_partition() -> None:
    partitions, boundaries = simulate_partitions(statuses_to_nodes(["delegated"] * 4))
    assert len(partitions) == 1
    assert boundaries == 0
    assert (partitions[0].first_node, partitions[0].last_node) == (0, 3)


def test_none_claimed_no_partitions() -> None:
    partitions, boundaries = simulate_partitions(statuses_to_nodes(["fallback", "unknown"]))
    assert partitions == []
    assert boundaries == 0
    assert rank_blocking_ops(statuses_to_nodes(["fallback", "unknown"]))[0].impact_score == 0.0


def test_partial_and_incorrect_are_claimed_fallback_crash_unknown_are_not() -> None:
    partitions, boundaries = simulate_partitions(
        statuses_to_nodes(["partial", "incorrect", "crash", "delegated", "unknown", "fallback"])
    )
    assert [(p.first_node, p.last_node) for p in partitions] == [(0, 1), (3, 3)]
    assert boundaries == 3  # TT|F|T|FF -> 4 runs, 3 transitions


def test_partitions_do_not_span_subgraphs() -> None:
    nodes = [
        make_node(0, "delegated", subgraph=0),
        make_node(0, "delegated", subgraph=1),
    ]
    partitions, boundaries = simulate_partitions(nodes)
    assert len(partitions) == 2
    assert boundaries == 0


def test_impact_splits_adjacent_cost_across_island() -> None:
    nodes = [
        make_node(0, "delegated", elements=2048),
        make_node(1, "fallback", elements=1, op="GATHER_ND"),
        make_node(2, "unknown", elements=1, op="RESHAPE"),
        make_node(3, "delegated", elements=10),
        make_node(4, "unknown", elements=1, op="SOFTMAX"),
    ]
    ranked = rank_blocking_ops(nodes)
    assert [(r.classified.node.op, r.impact_score) for r in ranked] == [
        ("GATHER_ND", 1029.0),  # (2048 + 10) / island of 2
        ("RESHAPE", 1029.0),  # tie keeps execution order
        ("SOFTMAX", 10.0),  # tail island: one adjacent run
    ]


def test_ranking_deterministic() -> None:
    nodes = statuses_to_nodes(["delegated", "fallback", "delegated", "crash", "delegated"])
    assert rank_blocking_ops(nodes) == rank_blocking_ops(nodes)
