---
family: nemotron-3-diarization
license: openmdw-1.1
model_id: nemotron-3-diarization__nemotron3_diar_encoder_offline_fp16
source_url: https://huggingface.co/litert-community/Nemotron-3-Diarization-LiteRT
task: voice-activity-detection
---

# nemotron-3-diarization__nemotron3_diar_encoder_offline_fp16

| | |
|---|---|
| **Task** | voice-activity-detection |
| **Family** | nemotron-3-diarization |
| **Source** | https://huggingface.co/litert-community/Nemotron-3-Diarization-LiteRT |
| **License** | openmdw-1.1 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.9.4 (litert-converter 0.4.0; torch 2.11.0; ai-edge-quantizer 0.9.0 for the float16 cast) |
| **Command** | `python build_nemotron3diar.py --run-dir $RUN --model-dir $RUN/hf_model --mode offline --ln safe  (exports/n3d_encoder_off_safe_fp16.tflite, renamed on publish)` |
| **Quantization** | float16 weight storage (ai-edge-quantizer FLOAT_CASTING; compute stays floating point; bfloat16-exact checkpoint — card 'Contents', NOTICE) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `nemotron3_diar_encoder_offline_fp16.tflite` | `b312fd74b1931af83185d84f017dd330d0478887db7ccf6ff0e4b65ff0dda28b` | 189.49 |

## Performance

No benchmark data yet.

## Delegation (static pre-flight)

Static `edge-lint` verdicts against the delegate compatibility matrix — no runtime execution. Verdicts are only valid for this backend and LiteRT version.

| | |
|---|---|
| **Backend** | gpu_mldrift |
| **LiteRT version** | 2.2.0 |
| **Op coverage** | 93.5% |
| **Partitions** | 191 |
| **Blocking ops** | DEQUANTIZE |
| **Verdict provenance** | measured=2919 |
| **Lint report schema** | 1.1 |

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-09-25/nemotron-3-diarization__nemotron3_diar_encoder_offline_fp16.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 7200.502 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-09-25 | measured |
| webgpu_mldrift | pass | yes | pass | 1.164627734622992e-06 | 58.1 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-09-25 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0/2026-09-24/nemotron-3-diarization__nemotron3_diar_encoder_offline_fp16__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu_mldrift | pass | yes | pass | sig:GpuOptions precision=FP32 | 172.6847135 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (SDK 36) | 2026-09-24 | measured |
| galaxy-s26 | gpu_mldrift | output_mismatch | yes | MISMATCH | sig:GpuOptions precision=default (fp16 compute) | 87.179922 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (SDK 36) | 2026-09-24 | measured |

## Pitfalls

- GPU precision decides correctness of the closed loop: at the S26 GPU's default precision (fp16 compute) one step is off by max |Δlogit| 0.34 and the speaker cache keeps 3-17 different frames per compression (37 -> 40 segments on the 97.6 s clip); GpuOptions(precision = FP32) matches the transformers reference on every frame at 171 ms per 0.72 s step (card: On-device results). FP32 is the recommended setting.
- All 64 LayerNorms are a scaled form with eps / S^2: the plain LayerNorm export runs fully delegated on the S26 GPU but returns wrong values without NaN at default precision (LayerNorm inputs reach |x| ~ 956, (x-mu)^2 exceeds float16 range) (card: Conversion notes, 'LayerNorm in float16').
- attn_bias is 0 on the L real rows and -30000 on the zero rows (offline graph: 0 / -16384 masked key / -32768 pad); the graph derives its row mask before the output convolution from these values, so other bias values change the output (card: Graph interfaces).
- rope_cos / rope_sin are host-computed tables for positions 0..T-1 (the same every step); they are inputs, not baked constants (card: Graph interfaces).
- Continuous back-to-back streaming heats the S26 GPU until its clock is capped (1300 -> 500 MHz after ~8 s; graph B 135 -> 288 ms); real-time pacing and the offline graph stayed at full speed (card: On-device results).
- The streaming state (Arrival-Order Speaker Cache + FIFO) is host code: sigmoid, mean over 8 rows, FIFO push and cache compression follow transformers' Nemotron3DiarizationSpeakerCache (Python conversion/nemotron3_diar_litert.py, Kotlin android/SpeakerCache.kt) — the graph alone does not diarize (card: Streaming loop).
- Log-mel must match torch.stft rounding: a float32 radix-2 FFT was 2.7e-4 off in the log domain; the Kotlin host ports pocketfft's real FFT and the Python host uses numpy's float32 path rfft(norm='forward') * 512 (card: Conversion notes, 'The FFT of the host mel').

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
