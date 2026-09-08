---
family: nanbeige
license: apache-2.0
model_id: nanbeige4.1-3b
source_url: https://huggingface.co/litert-community/Nanbeige4.1-3B
task: text-generation
---

# nanbeige4.1-3b

| | |
|---|---|
| **Task** | text-generation |
| **Family** | nanbeige |
| **Source** | https://huggingface.co/litert-community/Nanbeige4.1-3B |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (standard dense LlamaForCausalLM on the existing converter, no custom graph code; ChatML prompt template) (HF card Conversion) TODO (the HF card names the converter but not its version; no export log in the sources) |
| **Command** | `TODO (the HF card states the recipe; export_simple_template.py with FORCE_SPM per the polaris-4b meta's header-control check, not stated on the card)` |
| **Quantization** | int4 blockwise (block 32) + OCTAV, symmetric; embedding INT8, externalized (required for iPhone); KV cache 4096 (HF card Conversion) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `model.litertlm` | `1a2b741bd39edb9de665d42e11cf137a586eb517e7f4e10eb1e802e80a68369e` | 2297.713 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.14.0/2026-07-22/nanbeige4.1-3b__mac-studio-m4-max.json`, `data/device_runs/0.16.0/2026-09-05/nanbeige4.1-3b__galaxy-s26.json`, `data/device_runs/0.16.1/2026-09-02/nanbeige4.1-3b__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 223 | - | 39.32 | 10.58 | 5770.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 256 | - | - | - | - | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.14.0 | 2026-07-22 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 16.5 | 2.49 | 18139.6 | 3184.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-02 | measured |

## Pitfalls

- externalize_embedder=True is required for iPhone: the 166k-token vocab makes the weights a >2 GiB single TFLite section, over the ~2 GiB single-section mmap limit on iOS; externalizing drops the main section under 2 GiB so the model loads on iPhone (Metal GPU) as well as Android/desktop; same weights, GSM8K unchanged (HF card Conversion).
- Added-tokens tokenizer fix: Nanbeige's 10 special tokens (<|im_start|>, <|im_end|>, <think>, </think>, <tool_call>, …) live at ids 166100-166109 above the base SentencePiece vocab; the base SP conversion drops them and the runtime would crash with 'Token id out of range' — the converted tokenizer appends them as USER_DEFINED pieces at their exact ids (HF card Conversion).
- GSM8K (n=50, greedy, 0-shot CoT, max-tokens 2048): 84% — non-degenerate; passes the local 8-question gate 8/8 with a clean stop at <|im_end|> (HF card Quality).
- License Apache-2.0, inherited from Nanbeige/Nanbeige4.1-3B (HF card License).
- LiteRT-LM .litertlm bundle: LiteRT.js cannot run it, so delegation stays null and there is no browser block; device rows come from data/device_runs/ (Pi 5 / S26 / Pixel 8a / Mac / iPhone as measured).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
