---
family: qwen3_5
license: apache-2.0
model_id: qwen35-2b-vl-int8
source_url: https://huggingface.co/litert-community/Qwen3.5-2B
task: image-text-to-text
---

# qwen35-2b-vl-int8

| | |
|---|---|
| **Task** | image-text-to-text |
| **Family** | qwen3_5 |
| **Source** | https://huggingface.co/litert-community/Qwen3.5-2B |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch plus a hybrid-cache patch (reproduction script + patch: hf-to-litertlm qwen35_work/) — rank-4 chunk kernel for the gated delta rule, export cache layers for the linear-attention layers, re-authored static 512x512 vision tower, pad guard on the externalised-embedder graph (HF card Conversion notes) TODO (the HF card names the converter but not its version; no export log in the sources) |
| **Command** | `TODO (qwen35_work/ in hf-to-litertlm; invocation not stated on the card)` |
| **Quantization** | text decoder: int8 dynamic on linears + embedding (convs and the delta rule stay float), fp32 activations declared; vision: the checkpoint's own ViT shipped fp16 (int8 drops the tower's correlation to 0.97-0.98 on real photographs), int8 adapter; static 512x512; six-signature prefill ladder (HF card file table + Honest notes) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Qwen3.5-2B-VL_int8.litertlm` | `f8821e403dcfc939a033f6b2c9633a5d6b78c29a380ca825e0b90a62df61afe5` | 3003.03 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-09-05/qwen35-2b-vl-int8__galaxy-s26.json`, `data/device_runs/0.16.1/2026-09-01/qwen35-2b-vl-int8__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 207 | - | 180.67 | 18.98 | 1200.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| galaxy-s26 | gpu | pass | yes | - | 207 | - | 450.06 | 17.47 | 520.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 55.91 | 4.24 | 5037.9 | 3268.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-01 | measured |

## Pitfalls

- Six prefill signatures, not eleven: every exported signature is charged engine memory whether or not it is called; the text build's full 1-1024 ladder peaks ~5.3 GB on iPhone GPU and fits, but adding the vision tower pushes the same ladder past a 12 GB phone's jetsam ceiling (HF card Honest notes).
- Positions are 1-D: the fast_vlm contract feeds sequential positions, so the checkpoint's M-RoPE collapses to plain RoPE; against the full M-RoPE reference on nine image/prompt pairs one generation is identical and the other eight are fluent same-content paraphrases (HF card Honest notes).
- Android is not gated for this build: the text build's notes apply (Pixel-class 8 GB GPU does not fit), and on Mali the fp16 vision encoder is known to crash the device on other models of this shape (HF card Honest notes); this repo's S26 GPU/CPU and Pi 5 CPU rows come from the S7 backfill / Pi 5 wave 2 (device runs).
- Composite 8-question probe: on Mac this file answers 7 of 8, missing only 17 + 25 identically on GPU and CPU (the model, not the conversion; the text build records the same slip) (HF card Honest notes).
- The vision tower is re-authored for one static 512x512 image (Conv3d patch-embed folded to Conv2d over the duplicated temporal frame, learned position embeddings resampled) because the upstream dynamic-resolution grid_thw preprocessing aborts torch.export (HF card Conversion notes).
- LiteRT-LM .litertlm bundle: LiteRT.js cannot run it, so delegation stays null and there is no browser block; device rows come from data/device_runs/ (Pi 5 / S26 / Pixel 8a / Mac / iPhone as measured).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
