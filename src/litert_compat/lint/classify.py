"""Per-node classification: lookup signature extraction + `Matrix.lookup`.

The lookup signature is derived from the node's **output** tensors: input
operands are heterogeneous (weights, biases, index/shape tensors — e.g.
TRANSPOSE_CONV's first input is an int32 shape tensor) and would defeat
compute-dtype matching, while outputs carry the compute dtype. Builtin-option
attributes (keep_dims, half_pixel_centers, ...) are not extracted in this
phase; matrix entries constrained on them never match, so affected nodes
surface honestly as `unknown` and land on the needs-probe list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from litert_compat.matrix.query import Matrix, Verdict
from litert_compat.parser.reader import Node, ParsedModel, SubgraphGraph

# Statuses the delegate would claim. `incorrect` is claimed — the delegate
# compiles and runs the op, just wrongly — which is exactly why it is the worst
# failure mode; it still triggers `--fail-on fallback`. `unknown` never claims:
# no default to delegated, ever.
CLAIMED_STATUSES = frozenset({"delegated", "partial", "incorrect"})


@dataclass(frozen=True)
class ClassifiedNode:
    subgraph: int
    node: Node
    dtypes: list[str]
    shape_meta: dict[str, Any]
    verdict: Verdict
    output_elements: int  # element count of output tensors (impact heuristic input)
    # Input-side facts that matrix/rule constraints may name but that are NOT
    # part of the reported shape_meta (reports are pinned; these keys only
    # widen what `constraints_match` can see): operand_a..d = constant|activation.
    operand_meta: dict[str, Any] = field(default_factory=dict)

    @property
    def match_meta(self) -> dict[str, Any]:
        """shape_meta + operand_meta — the dict constraints are matched against."""
        return {**self.shape_meta, **self.operand_meta}

    @property
    def claimed(self) -> bool:
        return self.verdict.status in CLAIMED_STATUSES

    @property
    def is_custom(self) -> bool:
        return self.node.custom_code is not None


_OPERAND_KEYS = ("operand_a", "operand_b", "operand_c", "operand_d")


def operand_meta(subgraph: SubgraphGraph, node: Node) -> dict[str, Any]:
    """operand_a..d = "constant" (the input tensor's buffer carries bytes) or
    "activation", for the node's first four inputs. Lets a matrix row such as
    `ADD rank=2, operand_b=constant` (a rank-2 add against a rank-2 constant,
    measured silently wrong on one GPU delegate) actually match a node instead
    of staying documentation-only. Kept out of shape_meta so pinned lint
    reports do not change."""
    meta: dict[str, Any] = {}
    for key, tensor_index in zip(_OPERAND_KEYS, node.inputs):
        tensor = subgraph.tensors[tensor_index]
        meta[key] = "constant" if tensor.is_constant else "activation"
    return meta


def node_signature(
    subgraph: SubgraphGraph, node: Node
) -> tuple[list[str], dict[str, Any]]:
    """-> (dtypes, shape_meta) for `Matrix.lookup`, from the node's outputs."""
    outputs = [subgraph.tensors[i] for i in node.outputs]
    dtypes = sorted({t.dtype for t in outputs})
    ranks = [t.rank for t in outputs if t.rank is not None]
    shape_meta: dict[str, Any] = {
        "dynamic_shape": any(t.is_dynamic for t in outputs),
    }
    if ranks:
        shape_meta["rank"] = max(ranks)
    return dtypes, shape_meta


def _output_elements(subgraph: SubgraphGraph, node: Node) -> int:
    total = 0
    for i in node.outputs:
        shape = subgraph.tensors[i].shape
        if shape is None:
            continue
        elements = 1
        for dim in shape:
            elements *= max(dim, 1)  # dynamic (-1) dims count as 1
        total += elements
    return total


def classify_model(model: ParsedModel, matrix: Matrix) -> list[ClassifiedNode]:
    """Classify every node in every subgraph, in execution order.

    May raise `MatrixLookupTieError` — a matrix data defect, reported by
    `matrix validate`; the CLI surfaces it as a usage/data error (exit 2).
    """
    out: list[ClassifiedNode] = []
    for subgraph in model.subgraphs:
        for node in subgraph.nodes:
            dtypes, shape_meta = node_signature(subgraph, node)
            operands = operand_meta(subgraph, node)
            if node.custom_code is not None:
                verdict = Verdict(
                    status="unknown",
                    matched_entry=None,
                    reason=f"custom op {node.custom_code!r} — not a builtin, "
                    "outside the matrix vocabulary",
                    backend=matrix.backend,
                    litert_version=matrix.litert_version,
                )
            else:
                verdict = matrix.lookup(
                    node.op, dtypes=dtypes, shape_meta=shape_meta, operand_meta=operands
                )
            out.append(
                ClassifiedNode(
                    subgraph=subgraph.index,
                    node=node,
                    dtypes=dtypes,
                    shape_meta=shape_meta,
                    verdict=verdict,
                    output_elements=_output_elements(subgraph, node),
                    operand_meta=operands,
                )
            )
    return out
