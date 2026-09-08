---
family: qwen3
license: apache-2.0
model_id: qwen3-4b-thinking
source_url: https://huggingface.co/litert-community/Qwen3-4B-Thinking-2507
task: text-generation
---

# qwen3-4b-thinking

| | |
|---|---|
| **Task** | text-generation |
| **Family** | qwen3 |
| **Source** | https://huggingface.co/litert-community/Qwen3-4B-Thinking-2507 |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch export_hf (official converter; standard Qwen3ForCausalLM on the existing Qwen3 path, no custom graph code) (HF card Conversion) TODO (the HF card names the converter but not its version; no export log in the sources) |
| **Command** | `from litert_torch.generative.export_hf.export import export; export(model='Qwen/Qwen3-4B-Thinking-2507', output_dir='out', quantization_recipe='qwen3_int4_block128_octav.json', cache_length=4096, externalize_embedder=True) (HF card Conversion snippet)` |
| **Quantization** | int4 blockwise-128 + OCTAV, symmetric; embeddings INT8 (externalized); KV cache 4096 (HF card Conversion) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `model.litertlm` | `356b2d7778d27d55fbb0d2e0c5e8801e9de136db59849c41c882bd97333840d5` | 2359.731 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.14.0/2026-07-23/qwen3-4b-thinking__mac-studio-m4-max.json`, `data/device_runs/0.16.0/2026-09-05/qwen3-4b-thinking__galaxy-s26.json`, `data/device_runs/0.16.1/2026-09-01/qwen3-4b-thinking__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 204 | - | 63.02 | 5.99 | 3400.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 256 | - | 970.01 | 68.29 | 278.6 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.14.0 | 2026-07-23 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 10.61 | 1.49 | 25486.3 | 3974.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-01 | measured |

## Pitfalls

- Reasoning model (thinking on by default in the 2507 release); this repo's rows are the Mac compat/gpu_audit runs, the S26 CPU S7 row and the Pi 5 CPU row (device runs).
- License Apache-2.0, inherited from Qwen/Qwen3-4B-Thinking-2507 (HF card License).
- LiteRT-LM .litertlm bundle: LiteRT.js cannot run it, so delegation stays null and there is no browser block; device rows come from data/device_runs/ (Pi 5 / S26 / Pixel 8a / Mac / iPhone as measured).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
