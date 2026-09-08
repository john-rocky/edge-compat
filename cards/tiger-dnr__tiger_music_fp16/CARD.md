---
family: tiger
license: apache-2.0
model_id: tiger-dnr__tiger_music_fp16
source_url: https://huggingface.co/litert-community/TIGER-DnR-LiteRT
task: audio-to-audio
---

# tiger-dnr__tiger_music_fp16

| | |
|---|---|
| **Task** | audio-to-audio |
| **Family** | tiger |
| **Source** | https://huggingface.co/litert-community/TIGER-DnR-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `python tiger/scripts/build_tiger.py` |
| **Quantization** | fp16 |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `baf273503d26aea297f0475cd4ace6da843e7917330f61b7cbc4b2340ce8cb7c.tflite` | `7a97fe583d5bea5fec57635fdcf8498aad97b8b97006c6e7492d305711d2aaa5` | 15.37 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-11/tiger-dnr__tiger_music_fp16.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 28192.605 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |
| webgpu_mldrift | output_mismatch | yes | MISMATCH | 645078.9137990962 | 497.43 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/tiger-dnr__tiger_music_fp16__raspberry-pi-5.json`, `data/device_runs/2.2.0/2026-08-26/tiger-dnr__tiger_music_fp16__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | npu_qnn | load_failed | - | - | - | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · QAIRT (Hexagon, JIT on-device) · Android 16 | 2026-08-26 | measured |
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 42479.118 | - | - | - | 1134.8 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

## Pitfalls

- The caller must reflect-pad the 12.06 s chunk by 1024 samples on both sides (torch.stft(center=True) equivalent); iSTFT and overlap-add are host-side (HF card; zoo README L1254-1255).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
