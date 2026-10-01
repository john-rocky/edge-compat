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

import struct
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
_PAD_OPS = frozenset({"PAD", "PADV2", "MIRROR_PAD"})
_WEIGHTED_OPS = frozenset({"FULLY_CONNECTED", "CONV_2D", "DEPTHWISE_CONV_2D"})


@dataclass(frozen=True)
class GraphIndex:
    """Producer / consumer maps of one subgraph (tensor index -> nodes), built
    once per subgraph so operand_meta stays O(1) per node."""

    producer: dict[int, Node]
    consumers: dict[int, list[Node]]

    @classmethod
    def of(cls, subgraph: SubgraphGraph) -> GraphIndex:
        producer: dict[int, Node] = {}
        consumers: dict[int, list[Node]] = {}
        for node in subgraph.nodes:
            for t in node.outputs:
                producer[t] = node
            for t in node.inputs:
                consumers.setdefault(t, []).append(node)
        return cls(producer=producer, consumers=consumers)


def _paddings(tensor: Any) -> list[tuple[int, int]] | None:
    """Decode a constant [rank, 2] int32/int64 paddings tensor, or None."""
    data = tensor.const_data
    if data is None or tensor.shape is None or len(tensor.shape) != 2 or tensor.shape[1] != 2:
        return None
    n = tensor.shape[0] * 2
    if tensor.dtype == "int32" and len(data) >= 4 * n:
        values = struct.unpack_from(f"<{n}i", data)
    elif tensor.dtype == "int64" and len(data) >= 8 * n:
        values = struct.unpack_from(f"<{n}q", data)
    else:
        return None
    return [(values[2 * i], values[2 * i + 1]) for i in range(tensor.shape[0])]


def pad_axis(subgraph: SubgraphGraph, node: Node) -> str:
    """`axis` for the PAD family: "innermost" when every padded axis is the
    last one, "non-innermost" when any other axis is padded, "none" when the
    paddings are all zero, "unknown" when they are not a readable constant.
    The LiteRT #9272 class (MLDrift rank-3 PAD on a non-innermost axis) is a
    matrix row with `constraints: {rank: 3, axis: "non-innermost"}`."""
    if len(node.inputs) < 2:
        return "unknown"
    data_tensor = subgraph.tensors[node.inputs[0]]
    pads = _paddings(subgraph.tensors[node.inputs[1]])
    if pads is None or data_tensor.rank is None or len(pads) != data_tensor.rank:
        return "unknown"
    axes = [i for i, (lo, hi) in enumerate(pads) if lo or hi]
    if not axes:
        return "none"
    return "innermost" if axes == [data_tensor.rank - 1] else "non-innermost"


def operand_meta(
    subgraph: SubgraphGraph, node: Node, index: GraphIndex | None = None
) -> dict[str, Any]:
    """Input-side facts a matrix or rule constraint may name, kept out of the
    reported shape_meta so pinned lint reports do not change:

    - `operand_a..d` = "constant" (the input tensor's buffer carries bytes) or
      "activation", for the node's first four inputs — lets a row such as
      `ADD rank=2, operand_b=constant` match a node instead of staying
      documentation-only;
    - `operand_a_rank..d_rank` = the input tensor's rank when known — the
      mixed-rank rows of the LiteRT #10445 class (`ADD` of a rank-2 runtime
      tensor and a rank-3 constant, #10231) name both ranks;
    - PAD family: `axis` (see `pad_axis`);
    - FULLY_CONNECTED / CONV_2D / DEPTHWISE_CONV_2D: `weight_dtype` (the
      constant weight's dtype, e.g. "int8") and `weight_source` ("constant",
      "DEQUANTIZE" when the weight input is produced by a DEQUANTIZE of a
      constant — the explicit-dequantize recipe of #9523 — or "activation");
    - `consumer_ops` = the distinct builtin names consuming output 0, sorted
      and "+"-joined ("ADD", "ADD+RESHAPE", "none") — the DRQ int8
      FULLY_CONNECTED + ADD form of #9277 names its consumer.
    """
    index = index or GraphIndex.of(subgraph)
    meta: dict[str, Any] = {}
    for key, tensor_index in zip(_OPERAND_KEYS, node.inputs, strict=False):
        tensor = subgraph.tensors[tensor_index]
        meta[key] = "constant" if tensor.is_constant else "activation"
        if tensor.rank is not None:
            meta[f"{key}_rank"] = tensor.rank
    if node.op in _PAD_OPS:
        meta["axis"] = pad_axis(subgraph, node)
    if node.op in _WEIGHTED_OPS and len(node.inputs) >= 2:
        weight = subgraph.tensors[node.inputs[1]]
        source = index.producer.get(weight.index)
        if weight.is_constant:
            meta["weight_dtype"] = weight.dtype
            meta["weight_source"] = "constant"
        elif source is not None and source.op == "DEQUANTIZE" and source.inputs:
            quantized = subgraph.tensors[source.inputs[0]]
            meta["weight_dtype"] = quantized.dtype
            meta["weight_source"] = "DEQUANTIZE" if quantized.is_constant else "activation"
        else:
            meta["weight_source"] = "activation"
    if node.outputs:
        names = sorted({c.op for c in index.consumers.get(node.outputs[0], [])})
        meta["consumer_ops"] = "+".join(names) if names else "none"
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
        index = GraphIndex.of(subgraph)
        for node in subgraph.nodes:
            dtypes, shape_meta = node_signature(subgraph, node)
            operands = operand_meta(subgraph, node, index)
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
