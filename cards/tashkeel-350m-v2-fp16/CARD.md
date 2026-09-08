---
family: granite
license: apache-2.0
model_id: tashkeel-350m-v2-fp16
source_url: https://huggingface.co/mlboydaisuke/Tashkeel-350M-v2-LiteRT
task: text-generation
---

# tashkeel-350m-v2-fp16

| | |
|---|---|
| **Task** | text-generation |
| **Family** | granite |
| **Source** | https://huggingface.co/mlboydaisuke/Tashkeel-350M-v2-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | hf-to-litertlm, family recipe granite_work/convert_granite4h.py (2026-08-25) (README Conversion & verification) TODO (the README names the converter but not its version) |
| **Command** | `granite_work/convert_granite4h.py (hf-to-litertlm, 2026-08-25; invocation not stated in the README)` |
| **Quantization** | fp16 weights (convs/SSM fp32) (README file table) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Tashkeel-350M-v2_fp16.litertlm` | `02034a77bd178608eb9da106c95cb7e2dead2507424969bef8e8bc8788d5af67` | 732.901 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-25/tashkeel-350m-v2-fp16__mac-studio-m4-max.json`, `data/device_runs/0.16.0/2026-09-05/tashkeel-350m-v2-fp16__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-05/tashkeel-350m-v2-fp16__pixel-8a.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 224 | - | 77.17 | 23.69 | 2940.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 256 | - | 661.63 | 58.18 | 404.1 | - | mac-studio-m4-max · Apple M4 Max · litert-lm 0.16.0 | 2026-08-25 | measured |
| pixel-8a | cpu | fallback | no | - | 224 | - | 41.57 | 14.42 | 5460.0 | - | Pixel 8a · Tensor G3 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |

## Pitfalls

- Arabic diacritization (tashkeel) fine-tune of ibm-granite/granite-4.0-h-350m trained on Misraj/Sadeed_Tashkeela; the checkpoint's chat template (byte-equal to the granite base's, 6418/6418) is embedded and applied at runtime (README intro + Conversion).
- The metadata start token is dropped — granite's template has no leading BOS, and at 350M scale a prepended <|end_of_text|> flips correct diacritization into garbage (measured on this checkpoint) (README Conversion).
- Task gate: the model card's worked example plus nine undiacritized MSA probes, bundle vs HF fp32 greedy on identical rendered strings (granite_work/gate_tashkeel.py): fp16 10/10 byte-identical, int8 8/10 (the README carries a note on the two int8 misses) (README file table).
- License Apache-2.0, inherited from the source model and its granite base (README License).
- LiteRT-LM .litertlm bundle: LiteRT.js cannot run it, so delegation stays null and there is no browser block; device rows come from data/device_runs/.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
