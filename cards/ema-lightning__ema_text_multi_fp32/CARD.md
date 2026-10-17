---
family: ema-lightning
license: apache-2.0
model_id: ema-lightning__ema_text_multi_fp32
source_url: https://huggingface.co/litert-community/EMA-Lightning-LiteRT
task: text-to-speech
---

# ema-lightning__ema_text_multi_fp32

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
| **Quantization** | none: fp32 weights (card 'Files' / 'Weights': text fp32 — its fp16-weight file runs on the CPU but does not compile on the GPU delegate) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `ema_text_multi_fp32.tflite` | `fbef33f9a96b4cd9f2bd68f4615421f0a0fcd99b75e1d7fcf7956665dc2ab8e6` | 4.68 |

## Performance

No benchmark data yet.

## Delegation (static pre-flight)

Static `edge-lint` verdicts against the delegate compatibility matrix — no runtime execution. Verdicts are only valid for this backend and LiteRT version.

| | |
|---|---|
| **Backend** | gpu_mldrift |
| **LiteRT version** | 2.2.0 |
| **Op coverage** | 89.9% |
| **Partitions** | 60 |
| **Blocking ops** | SQUARED_DIFFERENCE, LOGISTIC, SQRT, DEPTHWISE_CONV_2D, RESHAPE, EMBEDDING_LOOKUP, MINIMUM, EXP |
| **Verdict provenance** | measured=608, unmatched=68 |
| **Lint report schema** | 1.1 |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0/2026-10-11/ema-lightning__ema_text_multi_fp32__galaxy-s26.json`, `data/device_runs/2.2.0/2026-10-11/ema-lightning__ema_text_multi_fp32__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu_xnnpack | fallback | no | pass | sig:ema_text_L64 (single-bucket fp32 export ema_text_L64_fp32.tflite); CPU XNNPACK 4 threads; gate app | 0.655286 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (build BP4A.251205.006.S942QOPS1AZH9) | 2026-10-11 | measured |
| galaxy-s26 | cpu_xnnpack | fallback | no | pass | sig:ema_text_L64 (the shipped file); CPU XNNPACK 4 threads; gate app | 0.566771 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (build BP4A.251205.006.S942QOPS1AZH9) | 2026-10-11 | measured |
| galaxy-s26 | gpu_mldrift | output_mismatch | yes | MISMATCH | sig:ema_text_L64 (single-bucket fp32 export ema_text_L64_fp32.tflite); GPU default precision (fp16 activations); gate app | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (build BP4A.251205.006.S942QOPS1AZH9) | 2026-10-11 | measured |
| galaxy-s26 | gpu_mldrift | pass | yes | pass | sig:ema_text_L64 (single-bucket fp32 export ema_text_L64_fp32.tflite); GpuOptions precision=FP32; gate app | 1.3771875 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (build BP4A.251205.006.S942QOPS1AZH9) | 2026-10-11 | measured |
| galaxy-s26 | gpu_mldrift | pass | yes | pass | sig:ema_text_L64 (the shipped file); GpuOptions precision=FP32; gate app | 1.850495 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (build BP4A.251205.006.S942QOPS1AZH9) | 2026-10-11 | measured |
| galaxy-s26 | npu_qnn | output_mismatch | yes | MISMATCH | sig:ema_text_L64 (single-bucket fp32 export ema_text_L64_fp32.tflite); HTP JIT via the Qualcomm compiler plugin (BURST, cold cache); C-API runner | 0.7004999999999999 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Qualcomm QNN HTP runtime libraries staged beside the runner (libQnnHtp.so, libQnnHtpPrepare.so, libQnnHtpV81Skel.so / Stub.so, libQnnSystem.so; the Kev lane's set kev_work/npu/libs, sha256 MATCH in device/n1/s0_setup.txt) with LiteRT's libLiteRtCompilerPlugin_Qualcomm.so / libLiteRtDispatch_Qualcomm.so; no version string is printed by the runner · Android 16 (build BP4A.251205.006.S942QOPS1AZH9) | 2026-10-11 | measured |
| mac-studio-m4-max | cpu_xnnpack | pass | - | - | sig:ema_text_L128 (single-bucket fp32 export ema_text_L128_fp32.tflite); bench_mac.py median of 20 | 0.84 | - | - | - | - | Mac Studio (M4 Max) · Apple M4 Max · litert 2.2.0 · macOS 27.0 (26A428) | 2026-10-11 | measured |
| mac-studio-m4-max | cpu_xnnpack | pass | - | - | sig:ema_text_L256 (single-bucket fp32 export ema_text_L256_fp32.tflite); bench_mac.py median of 20 | 1.27 | - | - | - | - | Mac Studio (M4 Max) · Apple M4 Max · litert 2.2.0 · macOS 27.0 (26A428) | 2026-10-11 | measured |
| mac-studio-m4-max | cpu_xnnpack | pass | - | - | sig:ema_text_L32 (single-bucket fp32 export ema_text_L32_fp32.tflite); bench_mac.py median of 20 | 0.38 | - | - | - | - | Mac Studio (M4 Max) · Apple M4 Max · litert 2.2.0 · macOS 27.0 (26A428) | 2026-10-11 | measured |
| mac-studio-m4-max | cpu_xnnpack | pass | - | - | sig:ema_text_L64 (single-bucket fp32 export ema_text_L64_fp32.tflite); bench_mac.py median of 20 | 0.56 | - | - | - | - | Mac Studio (M4 Max) · Apple M4 Max · litert 2.2.0 · macOS 27.0 (26A428) | 2026-10-11 | measured |
| mac-studio-m4-max | cpu_xnnpack | pass | - | - | sig:ema_text_L{32,64,128,256} (the shipped file); host loop, one call per fixture | - | - | - | - | - | Mac Studio (M4 Max) · Apple M4 Max · litert 2.2.0 · macOS 27.0 (26A428) | 2026-10-11 | measured |

## Pitfalls

- Text graph contract (card 'Files'): ids int32 [1,L], mask float32 [1,L] -> h [1,L,224], dur [1,L] (letters -> text features + frames per letter); one signature per bucket, ema_text_L{32,64,128,256}, shapes fixed per signature; the host picks the signature by the piece's length.
- Turkish only, one voice. The graphs read spoken-form, lowercased Turkish in the model's 49-symbol alphabet (a–z ç ö ü ğ ı ş, space and ! " % & ' ( ) , - . / : ; ?); numbers, dates, amounts and abbreviations are turned into words by normalizer-tr outside the graphs — an app either builds that crate or takes pre-normalized text (card intro, 'Limitations').
- GPU at its default precision (fp16 activations) is wrong on this graph: the card's parity table reads 3 dB vs the fp32 graph on the Galaxy S26 while GPU fp32 reads >= 98 dB and the CPU is bit-exact — run the text graph at GPU fp32 or on the CPU (card 'Performance', 'Parity').
- The fp16-weight variant of this file runs on the CPU but does not compile on the GPU delegate, which is why the shipped text file is fp32; dynamic int8 weights anywhere fail the parity bar (6–24 dB) and do not ship (card 'Files' / 'Weights', 'Parity').
- The small acoustic graphs run faster on the CPU than on the GPU on the Galaxy S26 (text L64: 0.7 ms CPU 4 threads vs 1.4 ms GPU fp32, per-launch overhead); the card's ship configuration keeps text + align on the CPU (card 'Performance').
- The shipped checkpoint has no self-attention in its text encoder (text_attn: 0 in ema.pt); the base card's architecture text describes two attention blocks (card 'Limitations').
- Fixed buckets: a piece longer than 256 letters / 512 frames is cut by the chunker first; speeds far from 1.0 may move a piece to a larger bucket (card 'Limitations').

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
