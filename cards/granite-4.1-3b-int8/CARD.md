---
family: granite
license: apache-2.0
model_id: granite-4.1-3b-int8
source_url: https://huggingface.co/litert-community/granite-4.1-3b
task: text-generation
---

# granite-4.1-3b-int8

| | |
|---|---|
| **Task** | text-generation |
| **Family** | granite |
| **Source** | https://huggingface.co/litert-community/granite-4.1-3b |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (pristine released stack; repro = hf-to-litertlm granite41_work/) 0.9.3 (litert-converter 0.3.1, ai-edge-quantizer 0.8.0, litert-lm-builder 0.16.0 — FINDINGS.md .venv-vl093) |
| **Command** | `hf-to-litertlm granite41_work/ reproduction script (pip-only stack, no PYTHONPATH patches)` |
| **Quantization** | int8 dynamic per-channel on linears + embedding |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `granite-4.1-3b_int8.litertlm` | `97bd368d56b29740d52532abcc8c8d1fb0bf8e0828c226c39ed52d1ef4551193` | 3654.16 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-17/granite-4.1-3b-int8__mac-m4-max.json`, `data/device_runs/0.16.0/2026-08-17/granite-4.1-3b-int8__pixel-8a.json`, `data/device_runs/0.16.0/2026-08-25/granite-4.1-3b-int8__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-06/granite-4.1-3b-int8__galaxy-s26.json`, `data/device_runs/0.16.1/2026-09-01/granite-4.1-3b-int8__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 202 | - | 67.58 | 8.61 | 3110.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-06 | measured |
| galaxy-s26 | gpu | pass | yes | - | 202 | - | 162.16 | 9.52 | 1350.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-25 | measured |
| mac-m4-max | gpu | pass | - | - | 256 | - | 743.3 | 32.74 | 398.9 | - | Mac Studio M4 Max · litert-lm 0.16.0 | 2026-08-17 | measured |
| pixel-8a | gpu | pass | yes | - | 17 | - | 15.48 | 3.86 | 1360.0 | - | Pixel 8a · Tensor G3 · litert-lm 0.16.0 | 2026-08-17 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 12.64 | 1.73 | 20908.9 | 4984.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-01 | measured |

## Pitfalls

- Requires litert-lm >= 0.16 (HF card header).
- No start_token in the metadata, on purpose: Granite sets add_bos_token False and its BOS is its EOS (<|end_of_text|>); with the converter-written start_token the runtime prepends end-of-text and the model echoes the question back — 5/8 vs 8/8 on the sanity gate. Reproduces on bf16 PyTorch with the token prepended by hand (HF card conversion notes).
- Prefill ladder trimmed to six signatures (1024/256/64/16/4/1): every exported signature is charged engine memory whether or not called; the eleven-signature build was killed by iOS during Metal init unless context was capped at 1024 (HF card conversion notes).
- Embedder externalised so the tied 100352x2560 vocab table sits in its own section, clear of the ~2 GiB single-section mmap ceiling on iOS (HF card).
- GSM8K 0-shot CoT n=100: bf16 88.0 / int8 87.0 / int4 84.0 — int4 costs 4 points for 1.7x smaller bytes (HF card quality table).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
