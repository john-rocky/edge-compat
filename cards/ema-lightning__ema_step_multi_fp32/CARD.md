---
family: ema-lightning
license: apache-2.0
model_id: ema-lightning__ema_step_multi_fp32
source_url: https://huggingface.co/litert-community/EMA-Lightning-LiteRT
task: text-to-speech
---

# ema-lightning__ema_step_multi_fp32

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
| **Quantization** | none: fp32 weights (card 'Files' / 'Weights': DiT steps fp32 — fp16 weights on the DiT failed the parity bar on one of six sentences) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `ema_step_multi_fp32.tflite` | `71b98f74248127f25f6bd0f236f5c99cac729595a044be7a14701be76c091a5b` | 14.608 |

## Performance

No benchmark data yet.

## Delegation (static pre-flight)

Static `edge-lint` verdicts against the delegate compatibility matrix — no runtime execution. Verdicts are only valid for this backend and LiteRT version.

| | |
|---|---|
| **Backend** | gpu_mldrift |
| **LiteRT version** | 2.2.0 |
| **Op coverage** | 96.4% |
| **Partitions** | 100 |
| **Blocking ops** | SQUARED_DIFFERENCE, LOGISTIC |
| **Verdict provenance** | measured=2555, unmatched=95 |
| **Lint report schema** | 1.1 |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0/2026-10-11/ema-lightning__ema_step_multi_fp32__galaxy-s26.json`, `data/device_runs/2.2.0/2026-10-11/ema-lightning__ema_step_multi_fp32__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu_xnnpack | pass | yes | pass | sig:ema_step_T384 (the shipped file); CPU XNNPACK 4 threads; gate app | 14.2302605 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (build BP4A.251205.006.S942QOPS1AZH9) | 2026-10-11 | measured |
| galaxy-s26 | cpu_xnnpack | pass | yes | pass | sig:ema_step_T96 (single-bucket fp32 export ema_step_T96_fp32.tflite); CPU XNNPACK 4 threads; gate app | 2.472969 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (build BP4A.251205.006.S942QOPS1AZH9) | 2026-10-11 | measured |
| galaxy-s26 | cpu_xnnpack | pass | yes | pass | sig:ema_step_T96 (the shipped file); CPU XNNPACK 4 threads; gate app | 2.5629429999999997 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (build BP4A.251205.006.S942QOPS1AZH9) | 2026-10-11 | measured |
| galaxy-s26 | gpu_mldrift | output_mismatch | yes | MISMATCH | sig:ema_step_T96 (single-bucket fp32 export ema_step_T96_fp32.tflite); GPU default precision (fp16 activations); gate app | 2.4051825 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (build BP4A.251205.006.S942QOPS1AZH9) | 2026-10-11 | measured |
| galaxy-s26 | gpu_mldrift | pass | yes | - | sig:ema_step_T96 (single-bucket fp32 export ema_step_T96_fp32.tflite); GpuOptions precision=FP32 through the C-API runner (the A/B control of the HTP legs); C-API runner | 3.6925 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (build BP4A.251205.006.S942QOPS1AZH9) | 2026-10-11 | measured |
| galaxy-s26 | gpu_mldrift | pass | yes | - | sig:ema_step_T96 (single-bucket fp32 export ema_step_T96_fp32.tflite); GpuOptions precision=FP32; gate app | 3.2139325000000003 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (build BP4A.251205.006.S942QOPS1AZH9) | 2026-10-11 | measured |
| galaxy-s26 | npu_qnn | output_mismatch | yes | MISMATCH | sig:ema_step_T96 (single-bucket fp32 export ema_step_T96_fp32.tflite); HTP JIT via the Qualcomm compiler plugin (BURST, cold cache); C-API runner | 1.557 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Qualcomm QNN HTP runtime libraries staged beside the runner (libQnnHtp.so, libQnnHtpPrepare.so, libQnnHtpV81Skel.so / Stub.so, libQnnSystem.so; the Kev lane's set kev_work/npu/libs, sha256 MATCH in device/n1/s0_setup.txt) with LiteRT's libLiteRtCompilerPlugin_Qualcomm.so / libLiteRtDispatch_Qualcomm.so; no version string is printed by the runner · Android 16 (build BP4A.251205.006.S942QOPS1AZH9) | 2026-10-11 | measured |
| mac-studio-m4-max | cpu_xnnpack | pass | - | - | sig:ema_step_T192 (single-bucket fp32 export ema_step_T192_fp32.tflite); bench_mac.py median of 20 | 3.97 | - | - | - | - | Mac Studio (M4 Max) · Apple M4 Max · litert 2.2.0 · macOS 27.0 (26A428) | 2026-10-11 | measured |
| mac-studio-m4-max | cpu_xnnpack | pass | - | - | sig:ema_step_T384 (single-bucket fp32 export ema_step_T384_fp32.tflite); bench_mac.py median of 20 | 6.98 | - | - | - | - | Mac Studio (M4 Max) · Apple M4 Max · litert 2.2.0 · macOS 27.0 (26A428) | 2026-10-11 | measured |
| mac-studio-m4-max | cpu_xnnpack | pass | - | - | sig:ema_step_T48 (single-bucket fp32 export ema_step_T48_fp32.tflite); bench_mac.py median of 20 | 1.97 | - | - | - | - | Mac Studio (M4 Max) · Apple M4 Max · litert 2.2.0 · macOS 27.0 (26A428) | 2026-10-11 | measured |
| mac-studio-m4-max | cpu_xnnpack | pass | - | - | sig:ema_step_T512 (single-bucket fp32 export ema_step_T512_fp32.tflite); bench_mac.py median of 20 | 9.57 | - | - | - | - | Mac Studio (M4 Max) · Apple M4 Max · litert 2.2.0 · macOS 27.0 (26A428) | 2026-10-11 | measured |
| mac-studio-m4-max | cpu_xnnpack | pass | - | - | sig:ema_step_T96 (single-bucket fp32 export ema_step_T96_fp32.tflite); bench_mac.py median of 20 | 2.76 | - | - | - | - | Mac Studio (M4 Max) · Apple M4 Max · litert 2.2.0 · macOS 27.0 (26A428) | 2026-10-11 | measured |
| mac-studio-m4-max | cpu_xnnpack | pass | - | - | sig:ema_step_T{48,96,192,384,512} (the shipped file); host loop, one call per fixture | - | - | - | - | - | Mac Studio (M4 Max) · Apple M4 Max · litert 2.2.0 · macOS 27.0 (26A428) | 2026-10-11 | measured |

## Pitfalls

- Step contract (card 'Files'): x [1,T,64], cond [1,T,224], bc [1,6,224], sg [1,2,224], fbias [1,1,1,T] -> v [1,T,64] — one DiT velocity pass, four calls per piece; signatures ema_step_T{48,96,192,384,512}, shapes fixed per signature. bc / sg come from ema_step_table.npz (the four steps' modulation vectors at t = 0, 1/4, 1/2, 3/4) and the sampler is the host's: x1 = x + (1 - t)·v, then x = (1 - t')·noise' + t'·x1 with seeded Gaussian noise (the package's DMD2-distilled sampler; seed 0 on CPU reproduces the PyTorch package) (card 'How the host loop works' step 4).
- Run the steps at fp32: fp16 weights on the DiT failed the bar on one of six sentences (26–28 dB) and GPU fp16 activations sit on the bar for one step (31 dB) and fail for the four unrolled steps (21 dB); dynamic int8 weights fail everywhere (card 'Files' / 'Weights', 'Parity').
- The HTP (LiteRT's NPU accelerator with the Qualcomm compiler plugin and runtime libraries, JIT) is this graph's fastest home on the Galaxy S26: 1.56 ms per step at 41 dB, compile 1 s per process — the card's fastest configuration runs the four steps there; without the vendor NPU libraries the steps run on the CPU (2.5 ms each) (card 'Performance').
- Turkish only, one voice; fixed buckets — a piece longer than 256 letters / 512 frames is cut by the chunker first, speeds far from 1.0 may move a piece to a larger bucket (card 'Limitations').

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
