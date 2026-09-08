---
family: nanbeige
license: apache-2.0
model_id: nanbeige4.2-3b-fp32act
source_url: https://huggingface.co/Nanbeige/Nanbeige4.2-3B
task: text-generation
---

# nanbeige4.2-3b-fp32act

| | |
|---|---|
| **Task** | text-generation |
| **Family** | nanbeige |
| **Source** | https://huggingface.co/Nanbeige/Nanbeige4.2-3B |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | same export as the shipped model.litertlm (looped-arch unroll, num_loops=2) with one model.toml repack same lineage as the shipped Nanbeige4.2-3B model.litertlm (built 2026-08-20, gated 2026-08-27) |
| **Command** | `nanbeige_work/gpu_quest/ repack — prefer_activation_type=fp32 (FINDINGS.md)` |
| **Quantization** | unchanged from the shipped bundle (repack does not touch weights) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `model_fp32act.litertlm` | `c384366d9c75c5d61ebd7befd7b4d60ca7f0cf9f84b60204981d48ad508b132b` | 2459.791 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.15.0/2026-08-28/nanbeige4.2-3b-fp32act__iphone-17-pro.json`, `data/device_runs/0.16.0/2026-08-27/nanbeige4.2-3b-fp32act__galaxy-s26.json`, `data/device_runs/0.16.1/2026-09-01/nanbeige4.2-3b-fp32act__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | pass | - | - | 205 | - | 29.64 | 3.45 | 7210.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-27 | measured |
| galaxy-s26 | gpu | pass | yes | - | 205 | - | 27.0 | 3.31 | 7890.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-27 | measured |
| iphone-17-pro | cpu | pass | - | - | - | - | - | - | - | - | iPhone 17 Pro · A19-Pro · litert-lm 0.15.0 · iOS 27.0 | 2026-08-28 | measured |
| iphone-17-pro | gpu | pass | - | - | - | - | - | - | - | - | iPhone 17 Pro · A19-Pro · litert-lm 0.15.0 · iOS 27.0 | 2026-08-28 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 5.73 | 1.07 | 45880.1 | 4311.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-01 | measured |

## Pitfalls

- The shipped model.litertlm FAILs on GPU while fully delegated: <unk> flood on Adreno (S4 idx 031) and on Mac Metal, with CPU correct on the same prompt — fp16 activations mis-read the loop boundary of the unrolled num_loops=2 graph. The fix is prefer_activation_type=fp32; the controlled A/B (same day, machine, runtime, prompt; weights byte-identical) flips Metal GPU from <unk> flood to a correct answer (ship_gpu_20260827/RESULTS.md).
- The fix crosses GPU backends: verified on S26 Adreno (2492/2492, real generation; device_runs 2026-08-27) and Mac Metal, plus an Adreno 840 leg (1689685). An earlier Apple-silicon-specific claim was wrong and retracted.
- iPhone 17 Pro: NEITHER bundle starts — both fail identically at engine init on memory, so fp32 activations are not the cause and the card carries the wall rather than an iOS row (882e48a; ship record). The output-head corruption investigated separately is family-wide and not caused by this repack (34f9bb1: four causes ruled out).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
