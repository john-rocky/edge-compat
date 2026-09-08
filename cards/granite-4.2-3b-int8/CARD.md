---
family: granite
license: apache-2.0
model_id: granite-4.2-3b-int8
source_url: https://huggingface.co/litert-community/granite-4.2-3b
task: text-generation
---

# granite-4.2-3b-int8

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
| **Command** | `granite42_work/convert_granite42_3b.sh (RECIPES=int8 → dynamic_wi8_afp32), then qa/add_thought_channel.py <out> --start '<think>' --end '</think>' (mirror: tools/add_thought_channel.py)` |
| **Quantization** | int8 dynamic per-channel on linears + embedding |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `granite-4.2-3b_int8.litertlm` | `3e5a0dd0c4eff06a50997e9966db9fee7d20e1c34cd04c7b5755c755339402a6` | 3587.363 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-31/granite-4.2-3b-int8__galaxy-s26.json`, `data/device_runs/0.16.0/2026-08-31/granite-4.2-3b-int8__mac-studio-m4-max.json`, `data/device_runs/0.16.1/2026-09-02/granite-4.2-3b-int8__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 206 | - | 50.81 | 7.03 | 4200.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-31 | measured |
| galaxy-s26 | gpu | pass | yes | - | 206 | - | 240.92 | 9.22 | 960.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-31 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 256 | - | 264.49 | 21.12 | 2611.6 | - | Mac Studio (M4 Max) · litert-lm 0.16.0 | 2026-08-31 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 256 | - | 1207.62 | 70.65 | 244.4 | - | Mac Studio (M4 Max) · litert-lm 0.16.0 | 2026-08-31 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 12.83 | 1.73 | 20599.4 | 4843.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-02 | measured |

## Pitfalls

- Requires litert-lm >= 0.16 (HF card header; curated manifest min_runtime_version 0.16.0).
- Reasoning model: it works problems inside <think>…</think> before answering, so give it an output budget of >= 2048 tokens — truncated mid-thought it produces no final answer at all. A thought channel (<think> / </think>) is declared in the bundle metadata so the runtime separates reasoning from the answer and honors a thinking budget; without that declaration the runtime streams raw reasoning into the answer and silently ignores any thinking budget. The exporter does not emit the channel on this path — it was added post-export (qa/add_thought_channel.py) with the tflite sections verified byte-identical, size unchanged (HF card usage notes + conversion notes; FINDINGS.md int8 build).
- The think opener is pre-filled by the bundle's assistant prefix (<|im_start|>assistant\n<think>\n, as IBM's own chat template does). Structured-template trap: the converter derives model.prefix from an assistant HISTORY message rendered with add_generation_prompt=False, so a jinja that opens <think> only in its generation branch silently ships a bare <|im_start|>assistant\n prefix — the FIRST int8 export of this model came out exactly that way and was re-exported with templates/granite42_think.jinja (opener in the history branch too, chatml_think shape); the shipped bundle's model.prefix carries the prefill. Read prompt_templates.model.prefix out of the bundle before trusting it (HF card conversion notes; FINDINGS.md structured-template section).
- No start_token in the metadata, on purpose: the tokenizer declares <s> as BOS but never prepends it (post-processor adds nothing; bos <s>=100283 != eos/pad <|im_end|>=100257), so the converter's unconditional start_token write was suppressed with NO_START_TOKEN=1. Different mechanism from 4.1's BOS==EOS echo bug, same knob (HF card conversion notes; FINDINGS.md premises table).
- Memory shape: 4096-token KV budget with a six-signature prefill ladder (1024/256/64/16/4/1), not eleven — every exported signature is charged engine memory whether or not it is called, and the eleven-signature build of the same-shape granite-4.1-3b was killed by iOS during Metal engine init. Input-embedding table (100352x2560) externalised into its own section (259 MB) to stay clear of the ~2 GiB single-section mmap ceiling on iOS; embeddings are untied, so the lm_head stays in the main graph — main TFLite section 3.50 GB (HF card usage notes + conversion notes; FINDINGS.md int8 build).
- Bundle template is a simple ChatML think template, not IBM's verbatim: it drops the empty system block IBM's template always emits (bf16 oracle 8/8 on both templates, so the gate shape shows no cost), and it renders assistant history VERBATIM instead of applying upstream's truncate_history_thinking, because truncation violates the runtime's render-prefix contract (FINDINGS.md premises table + bf16 oracle section).
- GSM8K parity (n=100, greedy, 0-shot CoT, max_tokens 2048, identical prompt/extraction; engine rows litert-lm-api 0.16.1 CPU backend, scored after </think>; harness = minicpm5_work/eval_gsm8k_api.py protocol, bf16 via granite42_work/gsm8k_bf16.py on MPS): bf16 91.0 / int8 90.0 / int4 block-32 80.0 — int8 is at parity (-1, same as 4.1); the card steers quality-sensitive deployments to this 3.76 GB file (HF card Accuracy; FINDINGS.md GSM8K section).
- Device coverage: the iPhone 17 Pro gate was run on the int4 file only — no iPhone result for int8 is recorded in the card, FINDINGS, or manifest. int8 did pass the Galaxy S26 (SM8850, Adreno) GPU and CPU gates with full OpenCL delegation (1783/1783, zero rejected ops) and the thought channel separating on-device, peak process RSS 4.3 GB on the CPU leg, no OOM on the 12 GB device; the curated manifest still labels int8 the desktop / quality-reference build (HF card Correctness + Performance; FINDINGS.md S26 table; curated manifest int8 platform_notes).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
