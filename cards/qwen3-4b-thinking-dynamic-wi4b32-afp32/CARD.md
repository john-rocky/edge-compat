---
family: qwen3
license: apache-2.0
model_id: qwen3-4b-thinking-dynamic-wi4b32-afp32
source_url: https://huggingface.co/litert-community/Qwen3-4B-Thinking-2507
task: text-generation
---

# qwen3-4b-thinking-dynamic-wi4b32-afp32

| | |
|---|---|
| **Task** | text-generation |
| **Family** | qwen3 |
| **Source** | https://huggingface.co/litert-community/Qwen3-4B-Thinking-2507 |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | LiteRT Torch (litert-torch) path, quantized with AI Edge Quantizer; the artifact incorporates LiteRT-LM GPU graph optimizations (composite ops for RoPE, fused QKV, …) (README Conversion Notes) TODO (the README names the converter but not its version) |
| **Command** | `TODO (not stated in the README)` |
| **Quantization** | dynamic INT4 block-32 weights, FP32 activations (dynamic_wi4b32_afp32), KV 4096 (README Artifact (Block 32) table) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Qwen3_4b_thinking_dynamic_wi4b32_afp32.litertlm` | `a412d77da3a15047232ce525d126a8cdb803691d29df791f13a9ef4cffab309a` | 2168.84 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-24/qwen3-4b-thinking-dynamic-wi4b32-afp32__galaxy-s26.json`, `data/device_runs/0.16.1/2026-09-01/qwen3-4b-thinking-dynamic-wi4b32-afp32__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu | pass | yes | - | 202 | - | 129.44 | 11.2 | 1650.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-24 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 8.92 | 1.43 | 29403.6 | 4507.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-01 | measured |

## Pitfalls

- The block-32 build is the CPU file: it degraded more on GSM8K (−9 pt vs bf16 90.0%, against −4 pt for the block-128 build) and produced corrupted output under the iPhone GPU delegate; the corruption is not iPhone-specific — on macOS Metal it degenerates too (question-echo loops, truncated reasoning, measured 2026-08-31 before and after the metadata repair). Use the block-32 file on CPU; block 128 is the recommended build on every backend (README Accuracy note).
- Both bundles run fully delegated on the Galaxy S26 GPU per the README (3764/3764 ops across 3 subgraphs on LiteRT GPU, 1627 MB); this repo carries the Pi 5 CPU row (device runs).
- 2026-08-31 metadata-only repairs (weights byte-identical, verified section by section): the thought channel (<think>\n / \n</think>) declared in the bundle metadata and the think pre-fill repaired — without the channel the runtime streams raw reasoning into the answer (README Conversion Notes update).
- Evaluating a reasoning model at a short token budget understates int4 — benchmark with max_tokens >= 2048 (README Accuracy note).
- License Apache-2.0, inherited from Qwen/Qwen3-4B-Thinking-2507 (README License).
- LiteRT-LM .litertlm bundle: LiteRT.js cannot run it, so delegation stays null and there is no browser block; device rows come from data/device_runs/.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
