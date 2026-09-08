---
family: ram-plus
license: apache-2.0
model_id: ram-plus__ram_stage3_tail_fp16
source_url: https://huggingface.co/litert-community/RAM-Plus-LiteRT
task: image-classification
---

# ram-plus__ram_stage3_tail_fp16

| | |
|---|---|
| **Task** | image-classification |
| **Family** | ram-plus |
| **Source** | https://huggingface.co/litert-community/RAM-Plus-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (ramplus-work build scripts, litert_torch.convert; fp16 via ai_edge_quantizer float_casting) 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `python build_hybrid.py` |
| **Quantization** | fp16 |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `ram_stage3_tail_fp16.tflite` | `c46e40cab69f070a4b7ba86b9de29f15148f9c67804c5e0888290d243f24ed4e` | 117.326 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/ram-plus__ram_stage3_tail_fp16__raspberry-pi-5.json`, `data/device_runs/2.2.0/2026-08-26/ram-plus__ram_stage3_tail_fp16__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu_mldrift | pass | - | - | - | 8.005 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 | 2026-08-26 | measured |
| galaxy-s26 | npu_qnn | pass | - | - | - | 14.568 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · QAIRT (Hexagon, JIT on-device) · Android 16 | 2026-08-26 | measured |
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 235.099 | - | - | - | 390.73 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

## Pitfalls

- Swin-L stage 3 must run on CPU: it fp16-miscomputes on the Mali GPU delegate (fp16 matmul accumulation in the deep, high-magnitude blocks — not overflow, not head_dim) (ram/README.md L33-41; HF card).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
