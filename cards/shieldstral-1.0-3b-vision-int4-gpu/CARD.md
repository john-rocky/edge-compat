---
family: shieldstral
license: apache-2.0
model_id: shieldstral-1.0-3b-vision-int4-gpu
source_url: https://huggingface.co/mistralai/Shieldstral-1.0-3B
task: image-text-to-text
---

# shieldstral-1.0-3b-vision-int4-gpu

| | |
|---|---|
| **Task** | image-text-to-text |
| **Family** | shieldstral |
| **Source** | https://huggingface.co/mistralai/Shieldstral-1.0-3B |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch export (decoder re-export, composite-free) + bundle rebuild (build_shieldstral_bundle.py) decoder re-export on ~/venvs/ltconv040dev, py3.10.13: litert-torch 0.9.3 / litert-converter 0.3.1 / ai-edge-quantizer 0.8.0 — the converter is the lever, not the torch bump (0.3.1 lowers odml.softmax natively where 0.3.0 cannot). Bundle repack: TODO(owner) — build_shieldstral_bundle.py imports only litert_lm_builder, but the interpreter that ran it is unrecorded, so no builder version is establishable. |
| **Command** | `shieldstral_work/vision_gpu_20260827/build_decoder_gpu.sh -> build_shieldstral_bundle.py` |
| **Quantization** | int4 decoder + int8 vision adapter/tower (per bundle inputs in s5_gate/ARTIFACTS.md) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Shieldstral-1.0-3B-vision_int4_gpu.litertlm` | `d4b1a34690ed991892bfa8c8dcbafc78a681cbcb9b18e6c8b4544fccdc748339` | 2654.158 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-28/shieldstral-1.0-3b-vision-int4-gpu__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-05/shieldstral-1.0-3b-vision-int4-gpu__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 231 | - | 23.2 | 1.33 | 10710.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| galaxy-s26 | gpu | pass | yes | - | 231 | - | 314.18 | 11.73 | 820.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-28 | measured |

## Pitfalls

- The pre-rebundle published file refused GPU engine creation outright: STABLEHLO_COMPOSITE odml.softmax at 52/1187 ops on the first subgraph (S4 idx 034). The rebundled decoder is composite-free and delegates 1187/1187 x13 subgraphs on the Galaxy S26 — the text sibling's numbers exactly (s5_gate results, device_runs 2026-08-28).
- The card's claim 'the vision bundle can replace the text one' was false on GPU for the old file and true again only after this rebundle — the sentence is true only of the _int4_gpu file; the old _int4 stays CPU-only (CARD_ADDITIONS.md §1).
- Safety classifier usage: the model scores via a single-token output contract — use the scoring API pattern, not free generation, and letterboxing beats naive resize for image inputs (memory shieldstral-3b-shipped; published HF card).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
