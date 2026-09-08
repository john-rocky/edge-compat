---
family: qwen3_5
license: apache-2.0
model_id: qwen35-0.8b-vl-int8
source_url: https://huggingface.co/litert-community/Qwen3.5-0.8B
task: image-text-to-text
---

# qwen35-0.8b-vl-int8

| | |
|---|---|
| **Task** | image-text-to-text |
| **Family** | qwen3_5 |
| **Source** | https://huggingface.co/litert-community/Qwen3.5-0.8B |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch + qwen35 hybrid-cache patch (decoder), convert_qwen35_vision.py (vision), build_qwen35vl_bundle.py + scripts/add_executor_metadata.py (bundle); public repro = hf-to-litertlm qwen35_work/ script + patch (HF card; FINDINGS.md) litert-torch 0.9.3 (writes no ExecutorMetadataProto; 0.9.4 does) + litert-converter 0.4.0 / torch 2.13.0 (~/venvs/lt093ctl, the STRIDED_SLICE build); decoder from the qwen35_work patched worktree 115a136 + qwen35_hybrid_litert_torch.patch; transformers 5.14.1 per the 2B-section .venv-vl093 line (FINDINGS.md) |
| **Command** | `convert_qwen35_vision.py (static 512) -> export_qwen35vl_decoder.py on the patched worktree (six-signature ladder 1024,256,64,16,4,1) -> build_qwen35vl_bundle.py -> scripts/add_executor_metadata.py <in> <out> ('48 state buffers: 36 linear-attn, 12 kv') (FINDINGS.md); public repro = hf-to-litertlm qwen35_work/ (HF card)` |
| **Quantization** | decoder int8 dynamic on linears + embedding (convs and the delta rule stay float), prefer_activation_type = fp32 declared in-bundle; vision encoder fp16, vision adapter int8; static 512x512; six-signature prefill ladder |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Qwen3.5-0.8B-VL_int8.litertlm` | `e3360b658c929ff35ab740a21f5e4b688096a72e351b8d347e26ccda314121b7` | 1241.934 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-28/qwen35-0.8b-vl-int8__iphone-17-pro.json`, `data/device_runs/0.16.0/2026-08-28/qwen35-0.8b-vl-int8__mac-studio-m4-max.json`, `data/device_runs/0.16.0/2026-08-31/qwen35-0.8b-vl-int8__iphone-17-pro.json`, `data/device_runs/0.16.0/2026-09-05/qwen35-0.8b-vl-int8__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-05/qwen35-0.8b-vl-int8__pixel-8a.json`, `data/device_runs/0.16.1/2026-09-01/qwen35-0.8b-vl-int8__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 207 | - | 213.9 | 32.41 | 1000.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| galaxy-s26 | gpu | pass | yes | - | 207 | - | 551.13 | 35.07 | 400.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| iphone-17-pro | cpu | pass | - | - | - | - | 100.09 | 26.01 | 1988.0 | 643.316 | iPhone 17 Pro · A19-Pro · litert-lm 0.16.0 · iOS 27.0 | 2026-08-31 | measured |
| iphone-17-pro | gpu | pass | - | - | - | - | 587.81 | 45.71 | 1684.0 | 3930.396 | iPhone 17 Pro · A19-Pro · litert-lm 0.16.0 · iOS 27.0 | 2026-08-28 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 256 | - | 649.12 | 46.63 | 436.4 | - | Mac Studio (M4 Max) · litert-lm 0.16.0 | 2026-08-28 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 256 | - | 1890.53 | 126.17 | 154.4 | - | Mac Studio (M4 Max) · litert-lm 0.16.0 | 2026-08-28 | measured |
| pixel-8a | cpu | fallback | no | - | 207 | - | 148.09 | 15.77 | 1460.0 | - | Pixel 8a · Tensor G3 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 71.56 | 6.91 | 3805.0 | 2117.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-01 | measured |

## Pitfalls

- Requires litert-lm >= 0.15: the hybrid's per-layer states bind through an ExecutorMetadata section that only 0.15+ reads (HF card header + conversion notes; manifest min_runtime litert-lm 0.15.0). The card's GPU examples run with --backend gpu --vision-backend gpu --cache no.
- Composite 8-question probe: Mac answers 8/8 on GPU and 8/8 on CPU, asked one at a time and as one combined prompt; iPhone 17 Pro scores 6/8 on Metal and 3/8 on CPU. Not one of the device misses reproduces on Mac, so they are device-side rather than the conversion — FINDINGS: 'a difference was measured, a cause was not established'; the mechanism is not identified and the card does not guess at one. The 3/8 is deterministic (byte-identical cold re-run three days later after an app reinstall — not thermal); the same eight questions asked one at a time answer 8/8 on the same phone CPU; the degradation needs an information-dense prompt of roughly 114 prefill tokens or more. The published text-only file scores the same 3/8 with four identical wrong answers on the same phone (control 2026-08-31), so it is a property of the phone's CPU path on this checkpoint family, not the vision build. On one probe arm (4 questions + natural filler, 123 tokens) Mac CPU also answers wrong — matching the fp32 reference's own wrong answer — so 'Mac is correct' is prompt-dependent; the device asymmetry is on the canonical prompt. On iPhone prefer the GPU backend (HF card Correctness + Honest notes; FINDINGS.md 2026-08-28..09-01).
- Toolchain trap with no accuracy signature: vision_adapter.tflite built under litert-converter 0.3.1 (~/venvs/ltconv040dev — the venv name is backwards) contains 6x GATHER_ND + 6x TRANSPOSE, a mobile-GPU hard wall; the same source under litert-converter 0.4.0 (~/venvs/lt093ctl) gives 6x STRIDED_SLICE and zero gather. Parity is identical across both builds (corr 0.99999999985), so only the op histogram catches it — require enc/adp ops gather/flex/custom all empty before anything consumes a vision tflite (FINDINGS.md 0.8B section).
- Android is not gated for this build. On Mali the fp16 vision encoder is known to crash the device on other models of this shape; an int8-vision build would be the Android path and has not been measured on a phone. The int8 encoder holds 0.9916-0.9951 correlation on real photographs here (better than the 2B tower's 0.974-0.984), and a local _int8vis_final variant (1217.7 MB) exists but is not published (HF card Honest notes; FINDINGS.md; manifest known_issues).
- litert-torch 0.9.3 writes no ExecutorMetadataProto section, so a fresh bundle fails at engine creation with 'No KV cache inputs found' although the decoder tflite is fine (7 signatures, 48 kv_cache inputs). Standing fix: scripts/add_executor_metadata.py <in> <out> -> 48 state buffers (36 linear-attn + 12 kv); that is what the _final suffix (+3201 bytes) means (FINDINGS.md; HF card conversion notes 'Runtime state binding').
- Vision is static 512x512 under the fast_vlm contract: one image per turn, 1024 patches merged 2x2 into 256 soft tokens injected at the image position, no DeepStack (deepstack_visual_indexes is empty upstream). Positions are 1-D, so the checkpoint's M-RoPE collapses to plain RoPE — expect fluent same-content paraphrase rather than token-exact agreement with a full M-RoPE reference; counting and dense-layout questions are where it shows (HF card Vision build + Honest notes; manifest platform_notes).
- The fp16-safe LayerNorm scale profile is this checkpoint's own: scales reach 16 from block 6 on, with 512 at the final norm (the 2B reaches 32 from block 11). Scales are calibrated per export; copying the sibling's table would silently mis-scale the tower (HF card Honest notes; FINDINGS.md 'rides unchanged was wrong').
- GPU runs with fp32 activations (declared in-bundle) — that is where the GPU memory multiple comes from (iPhone Metal peak ~3.8 GB against ~0.6 GB on CPU). The six-signature prefill ladder (1024,256,64,16,4,1) fit iPhone Metal on the first export — no jetsam, no maxNumTokens override — where the 2B's 11-signature ladder was jetsam-killed at 11 of 12 signatures; every exported signature is charged memory, and the long ladder also throttled CPU (XNNPACK repack tax) (HF card Honest notes; FINDINGS.md 2B iPhone leg; manifest platform_notes).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
