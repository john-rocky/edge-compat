"""Partition simulation and blocking-op impact ranking.

Partitions are maximal runs of claimed nodes in flatbuffer execution
(topological) order, computed per subgraph — a linear approximation of the
delegate's graph partitioning. Every boundary between a claimed and an
unclaimed region implies a host<->device sync.

The impact ranking is a HEURISTIC, labeled as such in every output format:
node cost is its output element count (a FLOPs proxy); a blocking node's
impact is the summed cost of the claimed runs adjacent to its CPU island,
split across the island's members — "how much delegated work this op (and its
island-mates) keeps split".
"""

from __future__ import annotations

from dataclasses import dataclass

from litert_compat.lint.classify import ClassifiedNode

IMPACT_NOTE = (
    "impact_score is a heuristic (output element count as a FLOPs proxy, "
    "adjacent-partition cost split across each CPU island) — a ranking aid, "
    "not a measurement"
)


@dataclass(frozen=True)
class Partition:
    subgraph: int
    index: int  # 0-based per model, in execution order
    first_node: int
    last_node: int
    ops: tuple[str, ...]

    @property
    def op_count(self) -> int:
        return len(self.ops)


@dataclass(frozen=True)
class RankedBlockingOp:
    classified: ClassifiedNode
    impact_score: float


def _runs(nodes: list[ClassifiedNode]) -> list[tuple[bool, list[ClassifiedNode]]]:
    """Maximal runs of equal claimed-ness, in execution order."""
    runs: list[tuple[bool, list[ClassifiedNode]]] = []
    for classified in nodes:
        if runs and runs[-1][0] == classified.claimed:
            runs[-1][1].append(classified)
        else:
            runs.append((classified.claimed, [classified]))
    return runs


def simulate_partitions(
    classified: list[ClassifiedNode],
) -> tuple[list[Partition], int]:
    """-> (claimed partitions across all subgraphs, sync boundary count)."""
    partitions: list[Partition] = []
    boundaries = 0
    for subgraph_index in sorted({c.subgraph for c in classified}):
        nodes = [c for c in classified if c.subgraph == subgraph_index]
        runs = _runs(nodes)
        boundaries += max(len(runs) - 1, 0)
        for claimed, members in runs:
            if claimed:
                partitions.append(
                    Partition(
                        subgraph=subgraph_index,
                        index=len(partitions),
                        first_node=members[0].node.index,
                        last_node=members[-1].node.index,
                        ops=tuple(c.node.op for c in members),
                    )
                )
    return partitions, boundaries


def rank_blocking_ops(classified: list[ClassifiedNode]) -> list[RankedBlockingOp]:
    """Unclaimed nodes ranked by heuristic impact, descending; ties keep
    execution order. Deterministic."""
    ranked: list[RankedBlockingOp] = []
    for subgraph_index in sorted({c.subgraph for c in classified}):
        nodes = [c for c in classified if c.subgraph == subgraph_index]
        runs = _runs(nodes)
        run_costs = [sum(c.output_elements for c in members) for _, members in runs]
        for i, (claimed, members) in enumerate(runs):
            if claimed:
                continue
            adjacent = 0
            if i > 0 and runs[i - 1][0]:
                adjacent += run_costs[i - 1]
            if i + 1 < len(runs) and runs[i + 1][0]:
                adjacent += run_costs[i + 1]
            impact = round(adjacent / len(members), 2)
            ranked.extend(RankedBlockingOp(c, impact) for c in members)
    ranked.sort(
        key=lambda r: (
            -r.impact_score,
            r.classified.subgraph,
            r.classified.node.index,
        )
    )
    return ranked
