"""Compatibility matrix: load, validate, query, import, diff, carry-forward."""

from litert_compat.matrix.carry import carry_forward
from litert_compat.matrix.diff import BackendMismatchError, diff_documents, render_text
from litert_compat.matrix.importer import CsvImportError, ImportResult, import_csv
from litert_compat.matrix.query import Matrix, MatrixLookupTieError, Verdict
from litert_compat.matrix.resolve import SnapshotResolutionError, resolve_snapshot
from litert_compat.matrix.validate import (
    Finding,
    MatrixValidationError,
    load_matrix_schema,
    provenance_counts,
    validate_document,
)

__all__ = [
    "BackendMismatchError",
    "CsvImportError",
    "Finding",
    "ImportResult",
    "Matrix",
    "MatrixLookupTieError",
    "MatrixValidationError",
    "SnapshotResolutionError",
    "Verdict",
    "carry_forward",
    "diff_documents",
    "import_csv",
    "load_matrix_schema",
    "provenance_counts",
    "render_text",
    "resolve_snapshot",
    "validate_document",
]
