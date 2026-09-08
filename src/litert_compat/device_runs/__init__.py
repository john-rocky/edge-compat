"""Device-run results (Phase 13): the NPU 対応可否 and LLM 対応表 lanes.

Model-level facts per (model x device x accelerator), schema
device_run_result.schema.json, stored as append-only snapshots under
data/device_runs/<runtime_version>/<date>/. Op-level matrix entries are NEVER
derived from these records (the trap rule, device edition).
"""
