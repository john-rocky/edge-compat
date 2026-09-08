---
family: qwen3
license: apache-2.0
model_id: polaris-4b
source_url: https://huggingface.co/litert-community/Polaris-4B-Preview
task: text-generation
---

# polaris-4b

| | |
|---|---|
| **Task** | text-generation |
| **Family** | qwen3 |
| **Source** | https://huggingface.co/litert-community/Polaris-4B-Preview |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch via litertlm-convert scripts/export_simple_template.py (reproduce_llm.sh key polaris-4b), then the 2026-08-22 metadata-only thought-channel re-ship (qa/add_thought_channel.py through the litert-lm 0.16.0 CLI; every tflite section byte-identical) TODO (owner to fill — searched 2026-08-25 four ways: no convert log, reproduce_llm.sh pins no version, the bundle header carries no converter version, the tflite sections carry only 'MLIR Converted.'; the RE-SHIP step is pinned to the litert-lm 0.16.0 CLI) |
| **Command** | `FORCE_SPM=1 EXTERNALIZE_EMBEDDER=1 CACHE=4096 python scripts/export_simple_template.py POLARIS-Project/Polaris-4B-Preview out/polaris-4b templates/qwen3_think.jinja BOCTAV4_128 ; python qa/add_thought_channel.py in.litertlm out.litertlm --start '<think>' --end '</think>' (2026-08-25 staged meta; FORCE_SPM because the published bundle carries an SP_Tokenizer section)` |
| **Quantization** | int4 blockwise (block 128) + OCTAV, symmetric; embedding INT8 (externalized section); KV cache 4096; ChatML template (HF card Conversion) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `model.litertlm` | `e1edad5042c8354bec1d6187c22cad9e460a38428da517fc7ef22b3aa828eb3b` | 2359.731 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.14.0/2026-07-23/polaris-4b__mac-studio-m4-max.json`, `data/device_runs/0.16.1/2026-09-02/polaris-4b__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-studio-m4-max | gpu | pass | - | - | 256 | - | 971.19 | 68.87 | 278.1 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.14.0 | 2026-07-23 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 10.54 | 1.49 | 25656.6 | 3959.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-02 | measured |

## Pitfalls

- Standard dense Qwen3ForCausalLM with rope_scaling: yarn, exported with a cache within original_max_position_embeddings so base RoPE is exact; externalize_embedder keeps every section under the iOS ~2 GiB single-section mmap limit so it loads on iPhone (HF card Conversion).
- License Apache-2.0, inherited from POLARIS-Project/Polaris-4B-Preview (HF card License).
- LiteRT-LM .litertlm bundle: LiteRT.js cannot run it, so delegation stays null and there is no browser block; device rows come from data/device_runs/ (Pi 5 / S26 / Pixel 8a / Mac / iPhone as measured).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
