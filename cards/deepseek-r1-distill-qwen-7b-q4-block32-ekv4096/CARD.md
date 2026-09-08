---
family: deepseek-r1-distill-qwen
license: mit
model_id: deepseek-r1-distill-qwen-7b-q4-block32-ekv4096
source_url: https://huggingface.co/litert-community/DeepSeek-R1-Distill-Qwen-7B
task: text-generation
---

# deepseek-r1-distill-qwen-7b-q4-block32-ekv4096

| | |
|---|---|
| **Task** | text-generation |
| **Family** | deepseek-r1-distill-qwen |
| **Source** | https://huggingface.co/litert-community/DeepSeek-R1-Distill-Qwen-7B |
| **License** | mit |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch export_hf — the official upstream converter run from a clean git worktree at upstream/main, dev-fork patches excluded; Qwen2ForCausalLM rides the stock converter with no custom code (HF card Conversion) TODO (the HF card names the converter but not its version; no export log in the sources) |
| **Command** | `TODO (the HF card states the recipe, not the invocation)` |
| **Quantization** | int4 blockwise (block 32) + OCTAV on linears, INT8 embedding externalized into its own bundle section; KV cache 4096 (HF card Conversion) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `DeepSeek-R1-Distill-Qwen-7B_q4_block32_ekv4096.litertlm` | `511d59c11704f7ab39b9cb3a0eef1a88f8c05673d0b74bcb4b19c15845ab4a7f` | 4322.031 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-24/deepseek-r1-distill-qwen-7b-q4-block32-ekv4096__galaxy-s26.json`, `data/device_runs/0.16.1/2026-09-01/deepseek-r1-distill-qwen-7b-q4-block32-ekv4096__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu | pass | yes | - | 197 | - | 107.53 | 8.43 | 1950.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-24 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 10.99 | 1.76 | 28253.3 | 5511.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-01 | measured |

## Pitfalls

- License: MIT for the model weights, inherited from deepseek-ai/DeepSeek-R1-Distill-Qwen-7B; the Qwen2.5 base is Apache-2.0; commercial use and derivatives permitted (HF card License).
- A ~4.2 GB single bundle (4,531,978,224 bytes): desktop / high-memory devices; the Raspberry Pi 5 CPU row (litert-lm 0.16.1, --cache memory) is the only phone-class-or-below measurement in this repo (device runs).
- LiteRT-LM .litertlm bundle: LiteRT.js cannot run it, so delegation stays null and there is no browser block; device rows come from data/device_runs/ (Pi 5 / S26 / Pixel 8a / Mac / iPhone as measured).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
