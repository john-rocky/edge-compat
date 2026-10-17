---
family: ema-lightning
license: apache-2.0
model_id: ema-lightning__ema_decoder_multi_fp16
source_url: https://huggingface.co/litert-community/EMA-Lightning-LiteRT
task: text-to-speech
---

# ema-lightning__ema_decoder_multi_fp16

| | |
|---|---|
| **Task** | text-to-speech |
| **Family** | ema-lightning |
| **Source** | https://huggingface.co/litert-community/EMA-Lightning-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.9.4 (LiteRT 2.2.0 CompiledModel for the checks) |
| **Command** | `TODO — the Hub card states the tool and the runtime but publishes no conversion command or script (the repo ships the graphs, config.json and the host loop ema_host.py)` |
| **Quantization** | fp16 weights (card 'Files' / 'Weights': decoder fp16 — lossless-grade on the CPU, native on the GPU; config.json forms.decoder = fp16) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `ema_decoder_multi_fp16.tflite` | `d94569d7b3eebf12cbaebdf4248d3bb8dd10c604ac75e59fdcde56dd6cd902b3` | 6.059 |

## Performance

No benchmark data yet.

## Delegation (static pre-flight)

Static `edge-lint` verdicts against the delegate compatibility matrix — no runtime execution. Verdicts are only valid for this backend and LiteRT version.

| | |
|---|---|
| **Backend** | gpu_mldrift |
| **LiteRT version** | 2.2.0 |
| **Op coverage** | 76.7% |
| **Partitions** | 442 |
| **Blocking ops** | DEQUANTIZE, LEAKY_RELU, TRANSPOSE_CONV, TANH |
| **Verdict provenance** | measured=1720, unmatched=220 |
| **Lint report schema** | 1.1 |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0/2026-10-11/ema-lightning__ema_decoder_multi_fp16__galaxy-s26.json`, `data/device_runs/2.2.0/2026-10-11/ema-lightning__ema_decoder_multi_fp16__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu_xnnpack | pass | yes | - | sig:ema_decoder_P120 (the shipped file); CPU XNNPACK 4 threads; gate app | 99.552291 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (build BP4A.251205.006.S942QOPS1AZH9) | 2026-10-11 | measured |
| galaxy-s26 | gpu_mldrift | output_mismatch | yes | MISMATCH | sig:ema_decoder_P120 (the shipped file); GPU default precision (fp16 activations); gate app | 30.506614 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (build BP4A.251205.006.S942QOPS1AZH9) | 2026-10-11 | measured |
| galaxy-s26 | gpu_mldrift | output_mismatch | yes | MISMATCH | sig:ema_decoder_P48 (the shipped file); GPU default precision (fp16 activations); gate app | 14.624141000000002 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (build BP4A.251205.006.S942QOPS1AZH9) | 2026-10-11 | measured |
| mac-studio-m4-max | cpu_xnnpack | pass | - | - | sig:ema_decoder_P{48,120} (the shipped file); host loop, one call per fixture | - | - | - | - | - | Mac Studio (M4 Max) · Apple M4 Max · litert 2.2.0 · macOS 27.0 (26A428) | 2026-10-11 | measured |

## Pitfalls

- Decoder contract (card 'Files'): z [1,64,P] + 7 masks [1,1,len_i] -> audio float32 [1, P·1920] (HiFi-GAN, 48 kHz); signatures ema_decoder_P48 and ema_decoder_P120, shapes fixed per signature; config.json decoder_stage_lengths gives the seven mask lengths per bucket (P48: 48, 384, 2304, 11521, 23042, 46084, 92168; P120: 120, 960, 5760, 28801, 57602, 115204, 230408).
- Windows: the host decodes 25 frames first (the first second, so audio starts early: the P48 signature) then 100 frames at a time (P120), with 8 frames of context on each side cut away after decoding; the pieces join sample-exactly (card 'How the host loop works' step 5).
- The decoder is where the GPU pays off on the Galaxy S26 (P120: 98.8 ms CPU 4 threads, 54.9 ms GPU fp32, 31.0 ms GPU fp16 at 40.8 dB; P48: 39.1 / 22.2 / 14.6 ms at 46.4 dB) and it is the wrong workload for the HTP: P48 took 518 s to compile and runs 127.7 ms (31 dB), P120 did not compile in 600 s (card 'Performance').
- Dynamic int8 weights on the decoder fail the parity bar (broadband int8 conv noise, 24 dB) and do not ship; the fp16 weights are lossless-grade (56.4 dB vs the PyTorch package over six sentences) (card 'Parity').
- The base card's RTF figures are the authors' own (an RTX 4090 and a cloud CPU) and the published Hub clips assets/sample-*.wav are not the reference (a different noise stream) (card 'Performance', 'Parity').
- Turkish only, one voice; fixed buckets — a piece longer than 256 letters / 512 frames is cut by the chunker first (card 'Limitations').

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
