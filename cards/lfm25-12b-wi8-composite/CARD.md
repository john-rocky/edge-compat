---
family: lfm2.5
license: lfm-open-license-v1.0
model_id: lfm25-12b-wi8-composite
source_url: https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct
task: text-generation
---

# lfm25-12b-wi8-composite

| | |
|---|---|
| **Task** | text-generation |
| **Family** | lfm2.5 |
| **Source** | https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct |
| **License** | lfm-open-license-v1.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (via hf-to-litertlm lfm_work/convert_lfm25.py + litertlm-convert scripts/add_executor_metadata.py) 0.9.3 (litert-converter 0.3.1; venv ~/venvs/ltconv040dev per RESULTS.md 2026-08-11) |
| **Command** | `python hf-to-litertlm/lfm_work/convert_lfm25.py LiquidAI/LFM2.5-1.2B-Instruct out_wi8_composite_conv031 --keep-softmax-composite` |
| **Quantization** | int8 dynamic weights / fp32 activations (litert-torch recipe dynamic_wi8_afp32) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `model_execmeta.litertlm` | `7ed2d8f9cc41eff47a0f2e21a5908b51c859bfbdf0a86c6403997d9caa49329d` | 1189.481 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-12/lfm25-12b-wi8-composite__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-studio-m4-max | gpu | pass | - | - | 256 | - | 4643.93 | 262.55 | 58.9 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.16.0 | 2026-08-12 | measured |

## Pitfalls

- This variant deliberately KEEPS the odml.softmax composites (--keep-softmax-composite) to verify litert-converter 0.3.1 lowers them: 0.3.0 left 216 markers, 0.3.1 leaves 0, and the op histogram is IDENTICAL to the 0.4.0.dev20260806 baseline. Marker counts alone are not evidence — dead metadata can remain; compare op histograms (RESULTS.md 手順 2-3).
- litert-torch 0.9.3 does not write the ExecutorMetadata section (unlike main); litert-lm >=0.15 requires it to bind the hybrid ShortConv/attention state buffers (22 here) — retrofitted with scripts/add_executor_metadata.py (RESULTS.md 手順 4, add_execmeta.log).
- litert-torch 0.9.3 pins litert-converter==0.3.*, so a fresh install today pulls the lowering 0.3.1 — but the same install on 2026-08-08 pulled 0.3.0 (no lowering). Check the converter version, not just litert-torch (RESULTS.md 環境).
- Mac GPU prefill numbers swing about +/-11% between sessions on op-identical graphs (4830 vs 4297 tok/s here); treat prefill deltas inside that band as measurement noise — decode and init are stable, and PASS/FAIL judgments are load-independent (RESULTS.md 数字の読み方).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
