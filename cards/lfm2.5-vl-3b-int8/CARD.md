---
family: lfm2.5
license: other (lfm-open-license-v1.0)
model_id: lfm2.5-vl-3b-int8
source_url: https://huggingface.co/litert-community/LFM2.5-VL-3B
task: image-text-to-text
---

# lfm2.5-vl-3b-int8

| | |
|---|---|
| **Task** | image-text-to-text |
| **Family** | lfm2.5 |
| **Source** | https://huggingface.co/litert-community/LFM2.5-VL-3B |
| **License** | other (lfm-open-license-v1.0) |

## Conversion

| | |
|---|---|
| **Tool** | hf-to-litertlm lfm_work/convert_lfm25_vl.py (litert-torch 0.9.3, --task image_text_to_text) — exact recipe, int4 post-processing and executor metadata all in the open pipeline (HF card Conversion) 0.9.3 (HF card Conversion) |
| **Command** | `lfm_work/convert_lfm25_vl.py --task image_text_to_text (hf-to-litertlm; HF card Conversion)` |
| **Quantization** | int8 dynamic on text linears + convs + embedding, vision tower int8 (HF card file table) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `LFM2.5-VL-3B_int8.litertlm` | `392eeae18ddbe7a851d5b37feae8ee13819def03abcd5a64cf56e5a2f302b298` | 3387.193 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-24/lfm2.5-vl-3b-int8__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-05/lfm2.5-vl-3b-int8__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 205 | - | 277.36 | 19.87 | 790.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| galaxy-s26 | gpu | pass | yes | - | 205 | - | 743.88 | 20.15 | 330.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-24 | measured |

## Pitfalls

- Android: Pixel 8a (Tensor G3), litert_lm_main built from the v0.16.0 release tag, 292-token prompt, decode run to EOS (≈1.1k tokens sustained), --disable_cache worst-case load — measured on the int4 file (HF card Speed); this repo's S26 CPU row for the int8 file is from the S7 backfill (device runs).
- License: LFM Open License v1.0 (HF tag license:other; card license_name lfm-open-license-v1.0, LICENSE file in the repo) (HF card front matter).
- LiteRT-LM .litertlm bundle: LiteRT.js cannot run it, so delegation stays null and there is no browser block; device rows come from data/device_runs/ (Pi 5 / S26 / Pixel 8a / Mac / iPhone as measured).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
