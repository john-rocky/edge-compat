---
family: qwen3.5
license: apache-2.0
model_id: qwen35-08b-v4gpufix
source_url: https://huggingface.co/Qwen/Qwen3.5-0.8B
task: text-generation
---

# qwen35-08b-v4gpufix

| | |
|---|---|
| **Task** | text-generation |
| **Family** | qwen3.5 |
| **Source** | https://huggingface.co/Qwen/Qwen3.5-0.8B |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch, pinned base + Qwen3.5 hybrid patch; v4 revision re-expresses the gated-delta-rule chunk kernel in an all-rank-<=4, pad-free form (the 5 chunk-kernel rank-3 PADs replaced with concat-zeros) editable checkout 115a136 + qwen35_hybrid_litert_torch.patch (convert_qwen35_hybrid.py docstring; python 3.12 homebrew per export_v4_gpu.log) |
| **Command** | `python convert_qwen35_hybrid.py Qwen/Qwen3.5-0.8B out_v4_gpu  # wi8fc + fp32-activation declaration; export log qwen35_work/export_v4_gpu.log` |
| **Quantization** | int8 dynamic on linears (wi8fc); recurrent/conv state paths float; fp32 activations declared (GPU requirement) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `model_wi8fc_fp32act.litertlm` | `07e1746c90234d60a90babe8ad2251f0b426025381e6feb747c077a2d06c03c0` | 765.92 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-13/qwen35-08b-v4gpufix__pixel-8a.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pixel-8a | gpu | pass | yes | - | 21 | - | 22.67 | 7.45 | 1060.0 | - | pixel-8a · Google Tensor G3 · litert-lm 0.16.0 | 2026-08-13 | measured |

## Pitfalls

- The v4 rewrite exists because ML Drift mis-executes rank-3 PAD (bit-exact head+1 shift on [H,T,1], row scramble on [H,T,D]); only rank-3 non-innermost PAD corrupts — last-axis and rank-4 are exact. Reported upstream with a 1-op repro as LiteRT#9272; the kernel now avoids the shape entirely (commit 045714f; HF card update).
- fp32 activations are mandatory on GPU for THIS checkpoint: fp16act scores 0/8 from a real-weight fp16 range overflow in layer-0 head 4 — a model property, not a converter defect (commit 045714f).
- Gates on this lineage: FP parity corr 1.0000 (48 positions), 8Q 8/8 on CPU and GPU on litert-lm 0.15.0 AND 0.16.0, BANANA hermetic CPU 40/40 + GPU 20/20 (commit 8ff812f). Mac bench (0.16.0, cache no, p256/d256): GPU 2015.9/158.4 tok/s TTFT 0.13 s vs CPU 641.4/48.2 (commit df0f5eb).
- GSM8K context for quality readers: bf16 12% vs int8 11% — the low absolute score is the model, not the conversion (commit 1b9c9ce).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
