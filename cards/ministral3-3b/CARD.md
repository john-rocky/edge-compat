---
family: mistral
license: apache-2.0
model_id: ministral3-3b
source_url: https://huggingface.co/litert-community/Ministral-3-3B-Instruct-2512
task: text-generation
---

# ministral3-3b

| | |
|---|---|
| **Task** | text-generation |
| **Family** | mistral |
| **Source** | https://huggingface.co/litert-community/Ministral-3-3B-Instruct-2512 |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (hf-to-litertlm reproduce_llm.sh key `ministral3-3b`) TODO |
| **Command** | `extract_ministral3_text.py (drops the Pixtral vision tower) -> CACHE=4096 EXTERNALIZE_EMBEDDER=1 export_simple_template.py <text-only src> <out> templates/mistral_simple.jinja BOCTAV4` |
| **Quantization** | int4 weights, BLOCKWISE_32 symmetric + OCTAV; tied embedding/lm_head INT8 |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Ministral-3-3B-Instruct-2512_q4_block32_ekv4096.litertlm` | `e68fcf0d52e5e9d86663c819b8ab53998694a633190b9102b456e12de4f16270` | 2232.535 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-17/ministral3-3b__pixel-8a.json`, `data/device_runs/0.16.0/2026-08-24/ministral3-3b__galaxy-s26.json`, `data/device_runs/0.16.1/2026-09-01/ministral3-3b__raspberry-pi-5.json`, `data/device_runs/0.17.0/2026-10-05/ministral3-3b__iphone-18-pro.json`, `data/device_runs/0.17.1/2026-10-05/ministral3-3b__mac-studio-m4-max.json`, `data/device_runs/gallery-1.0.15/2026-07-23/ministral3-3b__pixel-8a.json`, `data/device_runs/prebuilt-adac974b/2026-10-05/ministral3-3b__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 16 | - | 69.6 | 10.2 | 330.0 | 3654.3 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm prebuilt-adac974b · Android 16 | 2026-10-05 | measured |
| galaxy-s26 | cpu | fallback | no | - | 1024 | - | 52.34 | 8.48 | 20230.0 | 3683.5 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm prebuilt-adac974b · Android 16 | 2026-10-05 | measured |
| galaxy-s26 | gpu | pass | yes | - | 16 | - | 230.9 | 22.09 | 110.0 | 1388.9 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm prebuilt-adac974b · Android 16 | 2026-10-05 | measured |
| galaxy-s26 | gpu | pass | yes | - | 198 | - | 223.58 | 11.26 | 970.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-24 | measured |
| galaxy-s26 | gpu | pass | yes | - | 1024 | - | 338.25 | 22.13 | 3070.0 | 1401.8 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm prebuilt-adac974b · Android 16 | 2026-10-05 | measured |
| iphone-18-pro | cpu | fallback | no | - | 16 | - | 81.53 | 15.52 | 599.4 | - | iPhone 18 Pro (iPhone19,2) · litert-lm 0.17.0 · iOS 27.0 | 2026-10-05 | measured |
| iphone-18-pro | cpu | fallback | no | - | 128 | - | 99.0 | 18.08 | 2438.4 | - | iPhone 18 Pro (iPhone19,2) · litert-lm 0.17.0 · iOS 27.0 | 2026-10-05 | measured |
| iphone-18-pro | gpu | pass | yes | - | 16 | - | 337.11 | 43.68 | 73.5 | - | iPhone 18 Pro (iPhone19,2) · litert-lm 0.17.0 · iOS 27.0 | 2026-10-05 | measured |
| iphone-18-pro | gpu | pass | yes | - | 128 | - | 277.65 | 43.99 | 480.6 | - | iPhone 18 Pro (iPhone19,2) · litert-lm 0.17.0 · iOS 27.0 | 2026-10-05 | measured |
| mac-studio-m4-max | cpu | fallback | no | - | 16 | - | 112.34 | 17.77 | 588.6 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.17.1 | 2026-10-05 | measured |
| mac-studio-m4-max | cpu | fallback | no | - | 256 | - | 124.62 | 22.45 | 2506.6 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.17.1 | 2026-10-05 | measured |
| mac-studio-m4-max | gpu | pass | yes | - | 16 | - | 918.37 | 122.96 | 37.6 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.17.1 | 2026-10-05 | measured |
| mac-studio-m4-max | gpu | pass | yes | - | 256 | - | 1340.49 | 129.49 | 201.2 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.17.1 | 2026-10-05 | measured |
| pixel-8a | gpu | run_failed | - | - | - | - | - | - | - | - | Pixel 8a · Google Tensor G3 · litert-lm gallery-1.0.15 | 2026-07-23 | measured |
| pixel-8a | gpu | pass | yes | - | 13 | - | 8.49 | 6.8 | 1680.0 | - | Pixel 8a · Tensor G3 · litert-lm 0.16.0 | 2026-08-17 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 13.74 | 1.91 | 21261.7 | 3828.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-01 | measured |

## Pitfalls

- Text-only conversion: the Ministral-3 text decoder, with the Pixtral vision tower dropped (HF card).
- Embedding externalized so every section stays under 2 GiB, which is what lets it load on iOS (HF card).
- The bundle carries a start_token (<s>), and here that is CORRECT rather than the Granite defect: upstream bos is <s> and eos is </s>, different tokens, so prepending BOS is the Mistral convention. The Granite trap needs bos == eos (or add_bos_token False) to bite (CATALOG_REGATE.md).
- Two true statements that must not be collapsed: the Gallery only OFFERS GPU on ~12 GB+ devices, because its accelerator choice is fixed at import time from device RAM; the MODEL itself runs fully delegated on an 8 GB Pixel 8a GPU when litert_lm_main drives it directly — 1187/1187 prefill, 1087/1087 decode, zero rejections, correct output, measured 2026-08-17. The first is a property of the app, the second of the runtime (HF card + CATALOG_REGATE.md).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
