---
family: granite
license: apache-2.0
model_id: granite-4.2-3b-int4
source_url: https://huggingface.co/litert-community/granite-4.2-3b
task: text-generation
---

# granite-4.2-3b-int4

| | |
|---|---|
| **Task** | text-generation |
| **Family** | granite |
| **Source** | https://huggingface.co/litert-community/granite-4.2-3b |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (pristine released stack, no patched checkout; repro = hf-to-litertlm granite42_work/) 0.9.3 (litert-converter 0.4.0, ai-edge-quantizer 0.9.0, litert-lm-builder 0.16.1, transformers 5.14.1 — FINDINGS.md ~/venvs/lt093ctl) |
| **Command** | `granite42_work/convert_granite42_3b.sh (RECIPES=int4 → BOCTAV4), then qa/add_thought_channel.py <out> --start '<think>' --end '</think>' (mirror: tools/add_thought_channel.py)` |
| **Quantization** | int4 blockwise-32 + OCTAV on linears, int8 embedding |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `granite-4.2-3b_int4.litertlm` | `2abaaca65ecd4cf13cd0acd594cec53d48cae1ef975e30bfa898a7d2f6e7dbc0` | 2089.223 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.15.0/2026-08-31/granite-4.2-3b-int4__iphone-17-pro.json`, `data/device_runs/0.16.0/2026-08-31/granite-4.2-3b-int4__galaxy-s26.json`, `data/device_runs/0.16.0/2026-08-31/granite-4.2-3b-int4__mac-studio-m4-max.json`, `data/device_runs/0.16.1/2026-09-02/granite-4.2-3b-int4__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 206 | - | 22.44 | 9.83 | 9280.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-31 | measured |
| galaxy-s26 | gpu | pass | yes | - | 206 | - | 235.12 | 15.61 | 940.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-31 | measured |
| iphone-17-pro | cpu | pass | - | - | - | - | - | - | - | - | iPhone 17 Pro · A19-Pro · litert-lm 0.15.0 · iOS 27.0 | 2026-08-31 | measured |
| iphone-17-pro | gpu | pass | - | - | - | - | - | - | - | - | iPhone 17 Pro · A19-Pro · litert-lm 0.15.0 · iOS 27.0 | 2026-08-31 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 256 | - | 106.37 | 21.47 | 2944.0 | - | Mac Studio (M4 Max) · litert-lm 0.16.0 | 2026-08-31 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 256 | - | 1239.98 | 85.46 | 236.7 | - | Mac Studio (M4 Max) · litert-lm 0.16.0 | 2026-08-31 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 11.69 | 2.15 | 24620.6 | 3445.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-02 | measured |

## Pitfalls

- Requires litert-lm >= 0.16 (HF card header; curated manifest min_runtime_version 0.16.0).
- Reasoning model: it works problems inside <think>…</think> before answering, so give it an output budget of >= 2048 tokens — truncated mid-thought it produces no final answer at all. A thought channel (<think> / </think>) is declared in the bundle metadata so the runtime separates reasoning from the answer and honors a thinking budget; without that declaration the runtime streams raw reasoning into the answer and silently ignores any thinking budget. The exporter does not emit the channel on this path — it was added post-export (qa/add_thought_channel.py) with the tflite sections verified byte-identical (HF card usage notes + conversion notes; FINDINGS.md int8 build).
- The think opener is pre-filled by the bundle's assistant prefix (<|im_start|>assistant\n<think>\n, as IBM's own chat template does). Structured-template trap: the converter derives model.prefix from an assistant HISTORY message rendered with add_generation_prompt=False, so a jinja that opens <think> only in its generation branch silently ships a bare <|im_start|>assistant\n prefix — the first int8 export came out that way; templates/granite42_think.jinja puts the opener in the history branch too (chatml_think shape) and the re-export carries the prefill. Read prompt_templates.model.prefix out of the bundle before trusting it (HF card conversion notes; FINDINGS.md structured-template section).
- No start_token in the metadata, on purpose: the tokenizer declares <s> as BOS but never prepends it (post-processor adds nothing; bos <s>=100283 != eos/pad <|im_end|>=100257), so the converter's unconditional start_token write was suppressed with NO_START_TOKEN=1. Different mechanism from 4.1's BOS==EOS echo bug, same knob (HF card conversion notes; FINDINGS.md premises table).
- Memory shape: 4096-token KV budget with a six-signature prefill ladder (1024/256/64/16/4/1), not eleven — every exported signature is charged engine memory whether or not it is called, and the eleven-signature build of the same-shape granite-4.1-3b was killed by iOS during Metal engine init. Input-embedding table (100352x2560) externalised into its own section to stay clear of the ~2 GiB single-section mmap ceiling on iOS; embeddings are untied, so the lm_head stays in the main graph (HF card usage notes + conversion notes).
- Block size: int4 is BOCTAV4 block-32, reused from the granite-4.1-3b ship recipe. FINDINGS notes the reasoning-ship preference for block128 (block-32 carries an iPhone-GPU corruption risk — the same day's Qwen3-4B-Thinking block-32 file reads 1/8 degenerate on Mac Metal) and that a block128 A/B (RECIPES=int4b128) was NOT part of this ship; the block-32 file itself passed Mac Metal 8/8, Galaxy S26 Adreno GPU (full OpenCL delegation 1783/1783, zero rejected ops) and iPhone 17 Pro Metal 7/8 (FINDINGS.md int4 recipe note, S26 and iPhone tables; HF card Correctness).
- GSM8K parity (n=100, greedy, 0-shot CoT, max_tokens 2048, identical prompt/extraction; engine rows litert-lm-api 0.16.1 CPU backend, scored after </think>; harness = minicpm5_work/eval_gsm8k_api.py protocol, bf16 via granite42_work/gsm8k_bf16.py on MPS): bf16 91.0 / int8 90.0 / int4 block-32 80.0 — int4 costs 11 points, ~3x the -4 the identical recipe cost the non-thinking granite-4.1-3b of identical shape. Recorded hypothesis only: the thinking chain lengthens generation and multiplies exposure to int4 noise. Prefer int8 where 3.76 GB fits (HF card Accuracy; FINDINGS.md GSM8K section; curated manifest known_issues).
- iPhone 17 Pro (int4 only; composite 8-question prompt, --max-tokens 2048, engine at the bundle's full 4096 budget): Metal GPU 7/8, CPU 8/8, threshold 6/8, no jetsam, available memory >= ~4.2 GB throughout. The one GPU miss is the rhyme line at the END of the composite prompt (answered 'red'); the same question standalone scores correctly on every Mac/S26 leg, so it is attributed to the composite-prompt format, not conversion damage (4.1 had the same miss shape on its CPU leg) (HF card Correctness; FINDINGS.md iPhone table).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
