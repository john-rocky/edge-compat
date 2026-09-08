---
family: falcon-h1
license: other (falcon-llm-license)
model_id: falcon-h1-0.5b-instruct-int8
source_url: https://huggingface.co/tiiuae/Falcon-H1-0.5B-Instruct
task: text-generation
---

# falcon-h1-0.5b-instruct-int8

| | |
|---|---|
| **Task** | text-generation |
| **Family** | falcon-h1 |
| **Source** | https://huggingface.co/tiiuae/Falcon-H1-0.5B-Instruct |
| **License** | other (falcon-llm-license) |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch + hybrid-cache patch (repro: john-rocky/hf-to-litertlm falcon_h1_work/) TODO |
| **Command** | `TODO` |
| **Quantization** | post-hoc dynamic int8 over linears + embedding only; convs and the selective scan stay float; fp32 activations declared for GPU |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Falcon-H1-0.5B-Instruct_int8.litertlm` | `b8d1e5298bc3c8ed69b1f1f33a344fb9f992b033043f5438778b14477902fad8` | 620.045 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-24/falcon-h1-0.5b-instruct-int8__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-05/falcon-h1-0.5b-instruct-int8__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-05/falcon-h1-0.5b-instruct-int8__pixel-8a.json`, `data/device_runs/0.16.1/2026-09-02/falcon-h1-0.5b-instruct-int8__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 212 | - | 316.81 | 40.48 | 690.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| galaxy-s26 | gpu | pass | yes | - | 212 | - | 388.05 | 28.14 | 580.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-24 | measured |
| pixel-8a | cpu | fallback | no | - | 212 | - | 98.73 | 25.86 | 2190.0 | - | Pixel 8a · Tensor G3 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 53.85 | 7.0 | 4897.3 | 1358.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-02 | measured |

## Pitfalls

- GPU execution is verified on macOS (Metal), iPhone 17 Pro (Metal), and Pixel 8a (Arm Mali, OpenCL) only — not on Qualcomm Adreno; on a Snapdragon phone use the CPU backend unless you have confirmed the GPU path yourself (card Honest notes).
- On low-end Android the GPU buys prefill and time-to-first-token, not decode (decode is memory-bandwidth-bound; the CPU path reads int8 weights while the fp32-activation GPU path reads expanded ones) — long prompts favour the GPU, long answers favour the CPU; on Apple hardware the GPU wins across the board (card Honest notes).
- GPU runs with fp32 activations declared in the bundle — expect the memory multiple (iPhone 17 Pro: 2.50 GB GPU vs 0.66 GB CPU, card device table).
- At 0.5B the model itself is weak at arithmetic and non-English trivia; int8 adds borderline greedy flips on exactly those items (the float graph misses the same 8Q items) — no degeneracy (card Honest notes).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
