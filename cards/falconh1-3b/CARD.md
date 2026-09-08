---
family: falcon-h1
license: falcon-llm-license
model_id: falconh1-3b
source_url: https://huggingface.co/litert-community/Falcon-H1-3B-Instruct
task: text-generation
---

# falconh1-3b

| | |
|---|---|
| **Task** | text-generation |
| **Family** | falcon-h1 |
| **Source** | https://huggingface.co/litert-community/Falcon-H1-3B-Instruct |
| **License** | falcon-llm-license |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch + hybrid-cache patch (repro = hf-to-litertlm falcon_h1_work/) TODO(owner): patched-checkout version not pinned in the HF card |
| **Command** | `hf-to-litertlm falcon_h1_work/ reproduction script + patch` |
| **Quantization** | post-hoc dynamic int8 over linears + embedding only; convs and the scan stay float; fp32 activations declared for GPU |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Falcon-H1-3B-Instruct_int8.litertlm` | `add075d24b309a68ba7312bb89410345c06ff1a6628e42785f45f57045795d2b` | 3228.476 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-14/falconh1-3b__iphone-17-pro.json`, `data/device_runs/0.16.0/2026-08-14/falconh1-3b__mac-m4-max.json`, `data/device_runs/0.16.0/2026-08-24/falconh1-3b__galaxy-s26.json`, `data/device_runs/0.16.1/2026-09-02/falconh1-3b__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu | pass | yes | - | 212 | - | 109.22 | 11.49 | 2030.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-24 | measured |
| iphone-17-pro | cpu | pass | - | - | 146 | - | 48.71 | 7.85 | 3141.0 | 1455.5 | iPhone 17 Pro · litert-lm 0.16.0 · iOS 27.0 | 2026-08-14 | measured |
| iphone-17-pro | gpu | pass | - | - | 146 | - | 111.48 | 14.04 | 1494.0 | 3033.8 | iPhone 17 Pro · litert-lm 0.16.0 · iOS 27.0 | 2026-08-14 | measured |
| mac-m4-max | cpu | pass | - | - | 256 | - | 120.7 | 20.88 | 2168.9 | - | mac-m4-max · litert-lm 0.16.0 | 2026-08-14 | measured |
| mac-m4-max | gpu | pass | - | - | 256 | - | 979.31 | 65.32 | 276.7 | - | mac-m4-max · litert-lm 0.16.0 | 2026-08-14 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 23.32 | 1.9 | 11509.2 | 4413.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-02 | measured |

## Pitfalls

- Requires litert-lm >= 0.15 (HF card header).
- Composite hybrid cache layer: each layer holds KV + conv + recurrent state at one layer index — full-attention and Mamba2 in the same cache class; the runtime binds states by tensor name (HF card conversion notes).
- Folded selective scan: the Mamba2 scan is re-expressed as batched matmuls with chunk/head axes folded into batch (rank <= 4, no BROADCAST_TO, no int64 index math) — this is what makes the graph fully GPU-delegable (HF card conversion notes).
- Prefill-pad guard: partially-filled prefill chunks are made exact identity steps for the SSM; the stored conv window is gathered at the last valid column (HF card conversion notes).
- Falcon-specific wiring preserved: mup_vector and ssm_in_multiplier in the traced scan; exporter timestamp-index kwargs re-injected at the attention layer (HF card conversion notes).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
