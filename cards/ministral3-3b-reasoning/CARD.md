---
family: ministral
license: apache-2.0
model_id: ministral3-3b-reasoning
source_url: https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512
task: text-generation
---

# ministral3-3b-reasoning

| | |
|---|---|
| **Task** | text-generation |
| **Family** | ministral |
| **Source** | https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512 |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (via litertlm-convert scripts/export_simple_template.py) 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `python scripts/extract_text_backbone.py mistralai/Ministral-3-3B-Reasoning-2512 src_models/ministral-3-3b-reasoning-text && EXTERNALIZE_EMBEDDER=1 FORCE_SPM=1 CACHE=4096 python scripts/export_simple_template.py src_models/ministral-3-3b-reasoning-text out/ministral-3-3b-reasoning-ext templates/mistral_simple.jinja BOCTAV4` |
| **Quantization** | int4 blockwise-32 symmetric + OCTAV clipping; tied embedding/lm_head int8 (BOCTAV4) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `model.litertlm` | `f7170ef16d28dd4723385dfb1afca666b4f902e93dc89dc5c1a2a36f04e05fba` | 2232.098 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-12/ministral3-3b-reasoning__mac-studio-m4-max.json`, `data/device_runs/0.16.1/2026-09-01/ministral3-3b-reasoning__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-studio-m4-max | gpu | pass | - | - | 256 | - | 1234.34 | 94.65 | 222.0 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.16.0 | 2026-08-12 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 14.04 | 1.9 | 20988.2 | 3794.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-01 | measured |

## Pitfalls

- iOS: a single TFLite weight section over 2 GiB fails to mmap (engine creation error 'Failed to map section') — the tied embedding is externalized (externalize_embedder=True) so every section stays under 2 GiB and the model loads on iPhone.
- Must be exported with the native Mistral [INST] template and real EOS </s>, not ChatML — Mistral's tokenizer has no <|im_end|> token, so a ChatML export never hits a registered stop and runs away after the answer.
- Android GPU needs roughly 2x the model size in RAM (weights plus the ML Drift GPU weight cache); GPU is only offered on ~12 GB+ devices — on an 8 GB phone only CPU is selectable.
- Reasoning model: give it a generous max-tokens budget (GSM8K was scored at max-tokens 2048; scoring at 512 falsely penalises it).
- Text-only conversion: the Pixtral vision tower is dropped before export (strict missing=0 / unexpected=0 weight check).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
