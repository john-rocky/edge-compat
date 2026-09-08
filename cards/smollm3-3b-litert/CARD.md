---
family: smollm3
license: apache-2.0
model_id: smollm3-3b-litert
source_url: https://huggingface.co/mlboydaisuke/SmolLM3-3B-LiteRT
task: text-generation
---

# smollm3-3b-litert

| | |
|---|---|
| **Task** | text-generation |
| **Family** | smollm3 |
| **Source** | https://huggingface.co/mlboydaisuke/SmolLM3-3B-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch export_hf (generic path; SmolLM3ForCausalLM on the existing converter, the NoPE attention schedule — rotary disabled on every 4th layer — lowers to generic ops with no custom kernel) (README Conversion) TODO (the README names the converter but not its version) |
| **Command** | `TODO (not stated in the README)` |
| **Quantization** | int4 blockwise (block 32) + OCTAV optimal-clipping, embedding INT8 externalized into its own section so the main weights section stays under the iOS ~2 GiB single-mmap limit; KV cache 4096 (README Conversion) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `model.litertlm` | `8eaf512bc608151ad0e4ccdf968443cb842a41d087ec7741dab8a95faff32b1c` | 1909.502 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.17.0/2026-09-06/smollm3-3b-litert__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-studio-m4-max | cpu | pass | - | - | - | - | - | - | - | - | Mac Studio (M4 Max) · litert-lm 0.17.0 | 2026-09-06 | measured |

## Pitfalls

- GSM8K (n=100, greedy, 0-shot CoT asking for '#### <n>', identical prompt and extraction): bf16 81.0% / LiteRT int4 (BOCTAV4) 81.0% — fully at parity; visible step-by-step chain-of-thought, clean stop at <|im_end|> (README Quality).
- Blockwise (not channelwise) int4 plus OCTAV is what holds reasoning accuracy at parity (README Conversion).
- License Apache-2.0, inherited from HuggingFaceTB/SmolLM3-3B (README License).
- LiteRT-LM .litertlm bundle: LiteRT.js cannot run it, so delegation stays null and there is no browser block; device rows come from data/device_runs/.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
