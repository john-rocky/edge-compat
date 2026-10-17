---
family: ema-lightning
license: apache-2.0
model_id: ema-lightning__ema_align_multi_fp32
source_url: https://huggingface.co/litert-community/EMA-Lightning-LiteRT
task: text-to-speech
---

# ema-lightning__ema_align_multi_fp32

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
| **Quantization** | none: fp32 weights (card 'Files' / 'Weights': aligner fp32) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `ema_align_multi_fp32.tflite` | `c80117ce7fcf38e7f872b80cc35237adaa290342187239e1fee0ca2433f3a6b4` | 0.629 |

## Performance

No benchmark data yet.

## Delegation (static pre-flight)

Static `edge-lint` verdicts against the delegate compatibility matrix — no runtime execution. Verdicts are only valid for this backend and LiteRT version.

| | |
|---|---|
| **Backend** | gpu_mldrift |
| **LiteRT version** | 2.2.0 |
| **Op coverage** | 92.5% |
| **Partitions** | 15 |
| **Blocking ops** | COS, SIN |
| **Verdict provenance** | measured=245, unmatched=20 |
| **Lint report schema** | 1.1 |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0/2026-10-11/ema-lightning__ema_align_multi_fp32__galaxy-s26.json`, `data/device_runs/2.2.0/2026-10-11/ema-lightning__ema_align_multi_fp32__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu_xnnpack | pass | yes | pass | sig:ema_align_L64_T96 (single-bucket fp32 export ema_align_L64_T96_fp32.tflite); CPU XNNPACK 4 threads; gate app | 0.2490625 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (build BP4A.251205.006.S942QOPS1AZH9) | 2026-10-11 | measured |
| galaxy-s26 | cpu_xnnpack | pass | yes | pass | sig:ema_align_L64_T96 (the shipped file); CPU XNNPACK 4 threads; gate app | 0.22031250000000002 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (build BP4A.251205.006.S942QOPS1AZH9) | 2026-10-11 | measured |
| galaxy-s26 | gpu_mldrift | output_mismatch | yes | MISMATCH | sig:ema_align_L64_T96 (single-bucket fp32 export ema_align_L64_T96_fp32.tflite); GPU default precision (fp16 activations); gate app | 0.6097915 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (build BP4A.251205.006.S942QOPS1AZH9) | 2026-10-11 | measured |
| galaxy-s26 | gpu_mldrift | pass | yes | pass | sig:ema_align_L64_T96 (single-bucket fp32 export ema_align_L64_T96_fp32.tflite); GpuOptions precision=FP32; gate app | 0.6510415 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (build BP4A.251205.006.S942QOPS1AZH9) | 2026-10-11 | measured |
| galaxy-s26 | npu_qnn | output_mismatch | yes | MISMATCH | sig:ema_align_L64_T96 (single-bucket fp32 export ema_align_L64_T96_fp32.tflite); HTP JIT via the Qualcomm compiler plugin (BURST, cold cache); C-API runner | 0.487 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Qualcomm QNN HTP runtime libraries staged beside the runner (libQnnHtp.so, libQnnHtpPrepare.so, libQnnHtpV81Skel.so / Stub.so, libQnnSystem.so; the Kev lane's set kev_work/npu/libs, sha256 MATCH in device/n1/s0_setup.txt) with LiteRT's libLiteRtCompilerPlugin_Qualcomm.so / libLiteRtDispatch_Qualcomm.so; no version string is printed by the runner · Android 16 (build BP4A.251205.006.S942QOPS1AZH9) | 2026-10-11 | measured |
| mac-studio-m4-max | cpu_xnnpack | pass | - | - | sig:ema_align_L128_T192 (single-bucket fp32 export ema_align_L128_T192_fp32.tflite); bench_mac.py median of 20 | 0.55 | - | - | - | - | Mac Studio (M4 Max) · Apple M4 Max · litert 2.2.0 · macOS 27.0 (26A428) | 2026-10-11 | measured |
| mac-studio-m4-max | cpu_xnnpack | pass | - | - | sig:ema_align_L256_T384 (single-bucket fp32 export ema_align_L256_T384_fp32.tflite); bench_mac.py median of 20 | 1.01 | - | - | - | - | Mac Studio (M4 Max) · Apple M4 Max · litert 2.2.0 · macOS 27.0 (26A428) | 2026-10-11 | measured |
| mac-studio-m4-max | cpu_xnnpack | pass | - | - | sig:ema_align_L256_T512 (single-bucket fp32 export ema_align_L256_T512_fp32.tflite); bench_mac.py median of 20 | 1.17 | - | - | - | - | Mac Studio (M4 Max) · Apple M4 Max · litert 2.2.0 · macOS 27.0 (26A428) | 2026-10-11 | measured |
| mac-studio-m4-max | cpu_xnnpack | pass | - | - | sig:ema_align_L32_T48 (single-bucket fp32 export ema_align_L32_T48_fp32.tflite); bench_mac.py median of 20 | 0.18 | - | - | - | - | Mac Studio (M4 Max) · Apple M4 Max · litert 2.2.0 · macOS 27.0 (26A428) | 2026-10-11 | measured |
| mac-studio-m4-max | cpu_xnnpack | pass | - | - | sig:ema_align_L64_T96 (single-bucket fp32 export ema_align_L64_T96_fp32.tflite); bench_mac.py median of 20 | 0.33 | - | - | - | - | Mac Studio (M4 Max) · Apple M4 Max · litert 2.2.0 · macOS 27.0 (26A428) | 2026-10-11 | measured |
| mac-studio-m4-max | cpu_xnnpack | pass | - | - | sig:ema_align_L{L}_T{T} (the shipped file); host loop, one call per fixture | - | - | - | - | - | Mac Studio (M4 Max) · Apple M4 Max · litert 2.2.0 · macOS 27.0 (26A428) | 2026-10-11 | measured |

## Pitfalls

- Aligner contract (card 'Files'): h [1,L,224], cg [1,L], fg [1,T], allow [1,1,T,L] -> cond [1,T,224] — the windowed Gaussian aligner, once per piece; signatures ema_align_L{L}_T{T} for (32,48) (64,96) (128,192) (256,384) (256,512), shapes fixed per signature. cg / fg are the letter and frame positions inside their word and allow is the window mask (each frame reads only its own and the adjacent words) that the host builds from the text graph's per-letter frame counts (card 'How the host loop works' steps 2–3).
- The host groups the text graph's frame counts per word, rounds them to whole frames (1–250 per word, <= 3000 per piece) and gives every frame a word and a position inside it; this bookkeeping is outside the graphs (card 'How the host loop works').
- On the Galaxy S26 this graph takes GPU fp16 (48 dB vs the fp32 graph) but runs faster on the CPU (0.2 ms CPU 4 threads vs 0.7 ms GPU fp32, per-launch overhead); the card's ship configuration keeps text + align on the CPU (card 'Performance', 'Parity').
- Turkish only, one voice; the text frontend (normalizer-tr) is a separate component; fixed buckets — a piece longer than 256 letters / 512 frames is cut by the chunker first, speeds far from 1.0 may move a piece to a larger bucket (card 'Limitations').
- Dynamic int8 weights anywhere fail the parity bar (6–24 dB) and do not ship (card 'Parity').

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
