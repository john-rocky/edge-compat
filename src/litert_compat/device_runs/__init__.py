"""Device-run results (Phase 13): the NPU support list and the LLM table.

Model-level facts per (model x device x accelerator), schema
device_run_result.schema.json, stored as append-only snapshots under
data/device_runs/<runtime_version>/<date>/. Op-level matrix entries are NEVER
derived from these records (the trap rule, device edition).
"""
