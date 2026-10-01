---
family: nemotron-3-diarization
license: openmdw-1.1
model_id: nemotron-3-diarization__nemotron3_diar_frontend
source_url: https://huggingface.co/litert-community/Nemotron-3-Diarization-LiteRT
task: voice-activity-detection
---

# nemotron-3-diarization__nemotron3_diar_frontend

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
| **Command** | `python build_nemotron3diar.py --run-dir $RUN --model-dir $RUN/hf_model --mode low_latency --ln safe  (graph A = exports/n3d_frontend.tflite, renamed on publish)` |
| **Quantization** | none — float32 (graph A is published as the float32 export; conversion/README.md step 6) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `nemotron3_diar_frontend.tflite` | `c01df58c31b425183c5a008552b517e53007fee39748baa2a4c49c1a55072810` | 2.001 |

## Performance

No benchmark data yet.

## Delegation (static pre-flight)

Static `edge-lint` verdicts against the delegate compatibility matrix — no runtime execution. Verdicts are only valid for this backend and LiteRT version.

| | |
|---|---|
| **Backend** | gpu_mldrift |
| **LiteRT version** | 2.2.0 |
| **Op coverage** | 100.0% |
| **Partitions** | 1 |
| **Blocking ops** | none |
| **Verdict provenance** | measured=2 |
| **Lint report schema** | 1.1 |

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-09-25/nemotron-3-diarization__nemotron3_diar_frontend.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 0.425 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-09-25 | measured |
| webgpu_mldrift | pass | yes | pass | 0.0013167669386284582 | 0.532 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-09-25 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0/2026-09-24/nemotron-3-diarization__nemotron3_diar_frontend__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu_mldrift | pass | yes | pass | sig:GpuOptions precision=FP32 | 0.2066925 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (SDK 36) | 2026-09-24 | measured |
| galaxy-s26 | gpu_mldrift | output_mismatch | yes | MISMATCH | sig:GpuOptions precision=default (fp16 compute) | 0.2169005 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (SDK 36) | 2026-09-24 | measured |

## Pitfalls

- Run graph A at GPU precision FP32: its rows stay in the speaker cache and FIFO for the whole session; at default precision they differ from the reference by up to 0.38 (|x| <= 141), at FP32 by 2.0e-4, for 0.2 ms (card: Conversion notes, 'Graph A in FP32').
- Input is log-mel of one chunk: 104 frames x 128 slaney mel bins, pre-emphasis 0.97, 400-sample Hann in a 512-point FFT, hop 160, log(x + 2^-24), no normalization; zero rows after the last frame (card: Graph interfaces, Streaming loop).
- Log-mel must match torch.stft rounding: a float32 radix-2 FFT was 2.7e-4 off in the log domain; the Kotlin host ports pocketfft's real FFT and the Python host uses numpy's float32 path rfft(norm='forward') * 512 (card: Conversion notes, 'The FFT of the host mel').

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
