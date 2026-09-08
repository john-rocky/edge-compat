---
family: qwen3
license: apache-2.0
model_id: jan-nano-model-block32
source_url: https://huggingface.co/litert-community/Jan-nano
task: text-generation
---

# jan-nano-model-block32

| | |
|---|---|
| **Task** | text-generation |
| **Family** | qwen3 |
| **Source** | https://huggingface.co/litert-community/Jan-nano |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch via litertlm-convert scripts/export_simple_template.py (reproduce_llm.sh key jan-nano), then the 2026-08-22 metadata-only thought-channel re-ship (qa/add_thought_channel.py through the litert-lm 0.16.0 CLI; every tflite section byte-identical) TODO (owner to fill — searched 2026-08-25 four ways: no convert log, reproduce_llm.sh pins no version, the bundle header carries no converter version, the tflite sections carry only 'MLIR Converted.'; the RE-SHIP step is pinned to the litert-lm 0.16.0 CLI) |
| **Command** | `EXTERNALIZE_EMBEDDER=1 CACHE=4096 python scripts/export_simple_template.py Menlo/Jan-nano out/jan-nano templates/qwen3_think.jinja BOCTAV4 (block 32 recipe; the block-128 line is the model.litertlm build) ; python qa/add_thought_channel.py … (2026-08-25 staged meta)` |
| **Quantization** | int4 (symmetric) + OCTAV, blockwise block 32; embeddings INT8 (externalized section) (HF card Conversion) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `model_block32.litertlm` | `74712688d25de50a1b3ad5ed95e96be05d2445cbca8e684fb6490a4e446b7d40` | 2539.528 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-24/jan-nano-model-block32__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-05/jan-nano-model-block32__galaxy-s26.json`, `data/device_runs/0.16.1/2026-09-01/jan-nano-model-block32__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 202 | - | 28.97 | 5.94 | 7140.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| galaxy-s26 | gpu | pass | yes | - | 202 | - | 102.41 | 9.3 | 2080.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-24 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 10.38 | 1.45 | 28000.1 | 4443.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-01 | measured |

## Pitfalls

- Which file: block 128 (model.litertlm) is the recommended on-device build — GSM8K 88.0% (max_tokens 2048), loads on the iPhone 17 Pro at ~14 tok/s; block 32 (model_block32.litertlm) 85.0%, its 2.11 GiB section sits at the iOS memory ceiling and may fail to load — desktop/Android (HF card 'Which file?').
- Standard Qwen3ForCausalLM on the existing Qwen3 path, no custom graph code; embeddings INT8 externalized; KV cache 4096 (HF card Conversion).
- License Apache-2.0, inherited from Menlo/Jan-nano (itself fine-tuned from Qwen/Qwen3-4B) (HF card License).
- LiteRT-LM .litertlm bundle: LiteRT.js cannot run it, so delegation stays null and there is no browser block; device rows come from data/device_runs/ (Pi 5 / S26 / Pixel 8a / Mac / iPhone as measured).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
