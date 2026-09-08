---
family: granite
license: apache-2.0
model_id: mordant-3b-think-int8
source_url: https://huggingface.co/mlboydaisuke/Mordant-3B-Think-LiteRT
task: text-generation
---

# mordant-3b-think-int8

| | |
|---|---|
| **Task** | text-generation |
| **Family** | granite |
| **Source** | https://huggingface.co/mlboydaisuke/Mordant-3B-Think-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | hf-to-litertlm, one command (python scripts/convert.py Kezmark/Mordant-3B-Think, 2026-08-25) (README Conversion & verification) TODO (the README names the converter but not its version) |
| **Command** | `python scripts/convert.py Kezmark/Mordant-3B-Think (hf-to-litertlm, 2026-08-25)` |
| **Quantization** | dynamic int8 on linears + embedding (README file table) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Mordant-3B-Think_int8.litertlm` | `85205709015e412915ca86ca1e790177f204d438fbee288317b2c7cc2deb484e` | 3587.91 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-25/mordant-3b-think-int8__mac-studio-m4-max.json`, `data/device_runs/0.16.0/2026-09-05/mordant-3b-think-int8__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 206 | - | 62.91 | 7.7 | 3400.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| galaxy-s26 | gpu | pass | yes | - | 206 | - | 232.68 | 9.59 | 990.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 256 | - | 97.25 | 20.18 | 2682.0 | - | mac-studio-m4-max · Apple M4 Max · litert-lm 0.16.0 | 2026-08-25 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 256 | - | 1128.98 | 71.85 | 240.7 | - | mac-studio-m4-max · Apple M4 Max · litert-lm 0.16.0 | 2026-08-25 | measured |

## Pitfalls

- A full fine-tune of ibm-granite/granite-4.1-3b for AI image-generation prompt composition with chain-of-thought reasoning; the finetune's thinking-form chat template (opens the assistant turn with <think>) is embedded verbatim (byte-equal 1474/1474) (README intro + Conversion).
- The spurious metadata start token is dropped: this family declares bos == eos == <|end_of_text|> and its template never renders a leading BOS, so an engine-prepended start token reads as 'this document already ended' — measured on this checkpoint it flips HF bf16 greedy output into a code-fence loop (README Conversion).
- Reduced 7-signature prefill ladder + externalized embedder (the >=3B ship shape; the full 11-signature ladder is killed by iOS at Metal init on this family); quality gate 8/8 (think-aware budget), non-degenerate (README Conversion).
- License Apache-2.0, inherited from the source model and its granite-4.1 base (README License).
- LiteRT-LM .litertlm bundle: LiteRT.js cannot run it, so delegation stays null and there is no browser block; device rows come from data/device_runs/.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
