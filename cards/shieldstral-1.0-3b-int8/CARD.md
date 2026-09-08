---
family: shieldstral
license: apache-2.0
model_id: shieldstral-1.0-3b-int8
source_url: https://huggingface.co/litert-community/Shieldstral-1.0-3B
task: text-classification
---

# shieldstral-1.0-3b-int8

| | |
|---|---|
| **Task** | text-classification |
| **Family** | shieldstral |
| **Source** | https://huggingface.co/litert-community/Shieldstral-1.0-3B |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch via hf-to-litertlm (reproduction script there); the bundle embeds no Jinja — plain prefix/suffix turn markers only; minimum runtime litert-lm 0.15.0 (HF card Conversion) litert-torch 0.9.2 · litert-converter 0.3.0 · ai-edge-quantizer 0.8.0 · litert-lm-builder 0.15.0 · transformers 5.14.1 (HF card Conversion) |
| **Command** | `TODO (reproduction script in hf-to-litertlm; invocation not stated on the card)` |
| **Quantization** | export-time dynamic int8, externalised embedder (text-only bundle, vision tower dropped — verified output-neutral for text input: bit-identical logits on the floor set) (HF card Variants + Limitations) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Shieldstral-1.0-3B_int8.litertlm` | `f950d2d172af44dddc200cdcc23af6646b6ab4e285f406f273f2dee902f7db3a` | 3794.41 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-24/shieldstral-1.0-3b-int8__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-05/shieldstral-1.0-3b-int8__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 231 | - | 129.19 | 8.3 | 1910.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| galaxy-s26 | gpu | pass | yes | - | 231 | - | 289.34 | 8.63 | 910.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-24 | measured |

## Pitfalls

- Desktop / high-memory only: the 3.33 GiB single section exceeds the practical iOS mmap budget (~2.1 GiB with default entitlements; entitlement-relaxed apps have topped out below 3 GiB); pick int4 unless you have a reason not to — it matches int8 on every gate metric (HF card Variants).
- Policy-adaptive safety classifier: one policy per call (ask a single yes/no question per call rather than combining policies); a continuous text score costs two prefills — if you only need a thresholded decision at 0.5, generate one token; the logit margin is faithful in ordering but not scale (engine ≈ 1.16 x reference − 0.43 on CPU int8, residual sd 1.33), so thresholds tuned on the source at values other than 0.5 need re-tuning (HF card Limitations).
- No safety guarantee: a moderation aid, not a moderation system — on the gate set precision ≈ 0.78 at recall ≈ 0.96 with a broad 'is this unsafe?' query; both the query wording and the threshold change that trade-off substantially (HF card Limitations).
- Text quality gates: every variant sits inside a 1.2-point F1 band with the unquantized reference and its bf16 control (label flips only near |margin| < 0.7); 8-item floor set 8/8 on int4 CPU, int4 GPU and int8 CPU; prefill-length sweep over 82 lengths (95-1067 tokens) with zero failures; upstream reports 81.4 F1 on the full 1,680-item benchmark (HF card Quality gates).
- Context exported at 4096 tokens (the source supports far more; longer documents need a re-export) (HF card Limitations).
- LiteRT-LM .litertlm bundle: LiteRT.js cannot run it, so delegation stays null and there is no browser block; device rows come from data/device_runs/ (Pi 5 / S26 / Pixel 8a / Mac / iPhone as measured).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
