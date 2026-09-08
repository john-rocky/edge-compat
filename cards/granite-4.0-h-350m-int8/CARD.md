---
family: granite
license: apache-2.0
model_id: granite-4.0-h-350m-int8
source_url: https://huggingface.co/litert-community/granite-4.0-h-350m
task: text-generation
---

# granite-4.0-h-350m-int8

| | |
|---|---|
| **Task** | text-generation |
| **Family** | granite |
| **Source** | https://huggingface.co/litert-community/granite-4.0-h-350m |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (README Conversion section; sibling card granite-4.0-h-350m-int8-gpu carries the lane record) TODO (the README names the converter but not its version) |
| **Command** | `TODO (not stated in the README)` |
| **Quantization** | int8 dynamic weight-only on linears + embedding; convs/SSM float (README file table) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `granite-4.0-h-350m_int8.litertlm` | `feae9e8c2aa8832c254770e88d81e80c424ab9b5c0e9a3f3d2689b3ace4810f1` | 415.775 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-24/granite-4.0-h-350m-int8__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-05/granite-4.0-h-350m-int8__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-05/granite-4.0-h-350m-int8__pixel-8a.json`, `data/device_runs/0.16.1/2026-09-02/granite-4.0-h-350m-int8__raspberry-pi-5.json`, `data/device_runs/0.17.0/2026-09-06/granite-4.0-h-350m-int8__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 224 | - | 10.52 | 11.48 | 21390.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| galaxy-s26 | gpu | load_failed | no | - | - | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-24 | measured |
| mac-studio-m4-max | cpu | pass | - | - | - | - | - | - | - | - | Mac Studio (M4 Max) · litert-lm 0.17.0 | 2026-09-06 | measured |
| pixel-8a | cpu | fallback | no | - | 224 | - | 8.03 | 8.86 | 28010.0 | - | Pixel 8a · Tensor G3 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 8.85 | 5.57 | 29121.7 | 2682.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-02 | measured |

## Pitfalls

- Requires litert-lm >= 0.15 — the Mamba2 conv/SSM state buffers are bound through 0.15's generalized state binding (README intro).
- Correctness (README): a float export matches the HF model exactly at every decode position (8-step teacher-forced, per-position max |logit diff| <= 3.9e-4, corr 1.000000, top-1 identical), both the chunked prefill path and the single-token path.
- The bundle does NOT prepend a BOS token — matches Granite's official chat template; at 350M scale a prepended <|end_of_text|> measurably degrades greedy answers (README Correctness).
- Known limitation — engine state reuse, not quantization: if one Engine object is reused across many conversations whose prompts share a growing common prefix, a reply can stop after a few tokens (on the int8 file at chat-templated lengths 33-37, on _int8_gpu at 37-41); fresh engine per conversation avoids it (README Correctness).
- This file is the CPU build: on the Galaxy S26 GPU the README records 'does not run' (109 of 3673 / 3807 ops in the first subgraph reached) for the int8 and fp16 files — the _int8_gpu sibling (rank-<=4 selective scan, fp32 activations declared) is the GPU file (README S26 table).
- LiteRT-LM .litertlm bundle: LiteRT.js cannot run it, so delegation stays null and there is no browser block; device rows come from data/device_runs/.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
