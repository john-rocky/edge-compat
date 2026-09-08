"""Static `.tflite` flatbuffer reading and synthetic fixture building.

No dependency on the LiteRT runtime — only the `flatbuffers` package and the
public TFLite schema layout, so the linter runs anywhere Python runs.
"""

from litert_compat.parser.builder import GraphSpec, OpSpec, TensorSpec, build_tflite
from litert_compat.parser.reader import (
    Node,
    ParsedModel,
    SubgraphGraph,
    TensorInfo,
    TfliteParseError,
    parse_tflite,
    parse_tflite_file,
)

__all__ = [
    "GraphSpec",
    "Node",
    "OpSpec",
    "ParsedModel",
    "SubgraphGraph",
    "TensorInfo",
    "TensorSpec",
    "TfliteParseError",
    "build_tflite",
    "parse_tflite",
    "parse_tflite_file",
]
