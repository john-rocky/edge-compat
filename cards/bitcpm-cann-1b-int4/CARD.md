---
family: bitcpm
license: apache-2.0
model_id: bitcpm-cann-1b-int4
source_url: https://huggingface.co/openbmb/BitCPM-CANN-1B
task: text-generation
---

# bitcpm-cann-1b-int4

| | |
|---|---|
| **Task** | text-generation |
| **Family** | bitcpm |
| **Source** | https://huggingface.co/openbmb/BitCPM-CANN-1B |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (via litertlm-convert bitcpm_work/convert_bitcpm.sh — export_static_longrope.py + quantize_minicpm5.py + litert-lm-builder) 0.9.1 (lane venv ~/venvs/minicpm5 per SESSION_STATE.md 2026-07-21) |
| **Command** | `zsh bitcpm_work/convert_bitcpm.sh` |
| **Quantization** | int4 blockwise-32 symmetric min-max linears + int8 channelwise embedding/lm_head (wi4b32_wi8, algo minmax) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `bitcpm-cann-1b_wi4b32_wi8.litertlm` | `3d1d4a8c6f68a5dea91637b4400fb9cd90c6d992d2d6c60b64f36e3adfe7bf7e` | 1001.868 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-12/bitcpm-cann-1b-int4__mac-studio-m4-max.json`, `data/device_runs/gallery-1.0.15/2026-07-23/bitcpm-cann-1b-int4__pixel-8a.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-studio-m4-max | gpu | pass | - | - | 256 | - | 2946.29 | 188.85 | 92.2 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.16.0 | 2026-08-12 | measured |
| pixel-8a | gpu | pass | - | - | - | - | 136.6 | 7.84 | 2000.0 | - | Pixel 8a · Google Tensor G3 · litert-lm gallery-1.0.15 | 2026-07-23 | measured |

## Pitfalls

- int4-b32 min-max is a lossless container for BitCPM's ternary QAT weights: per 128-input-channel group the values are {-a, 0, +a} and map onto {-7, 0, +7} with zero rounding decisions (only fp16 per-block-scale rounding, <=4.04e-4 relative). OCTAV is unnecessary here and could deviate (REPORT.md core finding, verified on the checkpoint by verify_ternary.py).
- Do not generalize this recipe: the same data-free wi4b32_wi8 minmax on the non-ternary MiniCPM5-1B (same family) loses 13 GSM8K points (48 vs 61) — the ternary structure is what makes data-free int4 lossless (REPORT.md results).
- Tokenizer trap (MiniCPM4 family): the raw SentencePiece model lacks <|im_end|> (id 73440), so generation never hits a registered stop — the bundle must carry an SP model extended with the HF added tokens (fix_sp_added_tokens.py, +8 USER_DEFINED pieces -> 73448; convert_bitcpm.sh step 5).
- The 1B HF repo ships no added_tokens.json (synthesized from tokenizer_config), and longrope must be made static for torch.export: long==short factors with factor 1, so stripping @dynamic_rope_update is exact (prep_bitcpm_as_llama.py + export_static_longrope.py docstring).
- Mac GPU bench numbers for this file swing with thermal/system load (alternating runs 1015/59.9 -> 2016/105.5 tok/s): only matched back-to-back pairs or median-of-N are comparable; never compare against a bench recorded on another day (REPORT.md results).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
