---
family: qwen3_5
license: apache-2.0
model_id: qwen35-0.8b
source_url: https://huggingface.co/Qwen/Qwen3.5-0.8B
task: text-generation
---

# qwen35-0.8b

| | |
|---|---|
| **Task** | text-generation |
| **Family** | qwen3_5 |
| **Source** | https://huggingface.co/Qwen/Qwen3.5-0.8B |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch plus a hybrid-cache patch (export cache layers for linear_attention, state-continuation tracing, ExecutorMetadata appended at package time) TODO (owner) — card names no version number for litert-torch or the patch |
| **Command** | `TODO (owner) — card points to the reproduction script + patch in hf-to-litertlm qwen35_work/ but does not record the command` |
| **Quantization** | post-hoc dynamic int8 on linears + embedding; convs and the delta rule stay float. File Qwen3.5-0.8B_int8.litertlm, 963 MB. Text decoder only — vision tower and MTP heads dropped |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Qwen3.5-0.8B_int8.litertlm` | `684d4d34adf7176eb47f6026ff65c33d42584737254e5524a8d1ad62edc21b98` | 918.565 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.1/2026-09-01/qwen35-0.8b__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 70.52 | 7.06 | 3771.9 | 2510.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-01 | measured |

## Pitfalls

- Requires litert-lm >= 0.15 (per-layer states bind through an ExecutorMetadata section; both backends gated on 0.15.0 and 0.16.0).
- GPU inference requires fp32 activations — the bundle's TOML declares prefer_activation_type = "fp32", required for correct GPU numerics on this family today. That is where the GPU memory multiple comes from: iPhone 17 Pro peak 5.48 GB (GPU) vs 1.21 GB (CPU). The fp16-activation formulation is unfinished: the residual issue is a real-weight fp16 range overflow in one layer-0 head — a property of the checkpoint, not the conversion.
- Pixel 8a cannot compile this file on its GPU: fp32-expanded weights plus the full prefill-ladder of compiled programs exceed the phone's ~3.8 GB available memory (a reduced dev build of the same graph runs correctly, fully delegated — the limit is memory, not ops). CPU works.
- GPU execution is verified on macOS (Metal), iPhone 17 Pro (Metal), and Pixel 8a Mali (OpenCL, dev build) only — NOT verified on Qualcomm Adreno; on Snapdragon use the CPU backend unless confirmed on-device.
- GPU delegate miscomputes rank-3 non-final-axis PAD (reported upstream as LiteRT#9272) — the vendored chunk kernel avoids the shape by writing every tail-pad as a concat with a zeros constant, and keeps all tensors rank <= 4 with no BROADCAST_TO and no int64 index math.
- GPU trap: a rank-0 scalar entering broadcast arithmetic is silently miscomputed by the GPU delegate — reductions in the prefill-pad guard keep their batch dimension (keepdim=True).
- torch.eye inside the traced function lowers to STABLEHLO_IOTA, which no released TFLite kernel set registers — the identity matrix is lifted as a graph constant.
- Stop-token trap: Qwen3.5 uses different tokens for chat turn-end and config.json's eos_token_id. <|im_end|> must be declared as a stop token alongside <|endoftext|>; with only the latter, the literal <|im_end|> text leaks into output and multi-turn history records the marker as text (fixed metadata-only 2026-08-07).
- Chat template is a simplified ChatML template, not the stock Qwen3.5 template. Thinking is disabled via an empty <think>\n\n</think> block opening each assistant turn, and that block is deliberately kept in history renders — the stock template strips it from past turns, which breaks LiteRT-LM's incremental conversation rendering (each turn's render must be a string-extension of the previous one) and kills multi-turn on turn 2. Tool-calling and vision sections are not included.
- Quality at 0.8B int8: GSM8K 11% vs 12% for the PyTorch bf16 reference (greedy, 0-shot CoT, max-tokens 512, n=100, non-thinking). The 8-question sanity gate is 8/8, but on a harder composite probe int8 measurably costs answers at this scale — ask for a float/fp16 variant for maximum fidelity.
- On low-end Android GPUs decode is memory-bandwidth-bound and does not beat the CPU; the GPU win is on Apple hardware (and, generally, prefill/TTFT). Measured (litert-lm benchmark 0.16.0, M4 Max, -p 256 -d 256 --runs 3 --cache no): GPU 1972/161.8 tok/s prefill/decode, CPU 666/46.7.
- Multi-length prefill signatures (1-1024) are exported so the runtime picks tight chunks; pad positions are made identity steps for the delta rule and the stored conv window is gathered at the last valid column.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
