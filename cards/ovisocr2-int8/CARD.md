---
family: qwen3_5
license: apache-2.0
model_id: ovisocr2-int8
source_url: https://huggingface.co/mlboydaisuke/OvisOCR2-LiteRT
task: image-text-to-text
---

# ovisocr2-int8

| | |
|---|---|
| **Task** | image-text-to-text |
| **Family** | qwen3_5 |
| **Source** | https://huggingface.co/mlboydaisuke/OvisOCR2-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch + qwen35 hybrid-cache patch (decoder, patched worktree), convert_qwen35_vision.py (vision), bundle + ExecutorMetadata retrofit; public repro = hf-to-litertlm qwen35vl_work/ REPRODUCE section with the env-override scripts (HF card; FINDINGS.md) litert-torch 0.9.3 + litert-converter 0.4.0 (~/venvs/lt093ctl — FINDINGS.md); decoder from the qwen35_work patched worktree per the inherited 0.8B rail (qwen35vl_work/FINDINGS.md: 115a136 + qwen35_hybrid_litert_torch.patch) |
| **Command** | `MODEL=ATH-MaaS/OvisOCR2 IMG=512 convert_qwen35_vision.py out/ovisocr2-vision; decoder export on the patched worktree with PREFILL=1024,256,64,16,4,1 CACHE=4096 (export_qwen35vl_decoder.py per the 0.8B rail); bundle -> out/ovisocr2-bundle/OvisOCR2_int8_final.litertlm with ExecutorMetadata retrofitted (FINDINGS.md); step-by-step = hf-to-litertlm qwen35vl_work/ (HF card)` |
| **Quantization** | decoder int8 (dynamic on linears + embedding; convs and the delta rule stay float, fp32 activations declared) + fp16 vision encoder / int8 vision adapter, static 512x512, six-signature prefill ladder |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `OvisOCR2_int8.litertlm` | `6482ff193826b70093a9b86f559669b1c12c13ab170f6dfc0f4eb6037a88887d` | 1241.934 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-09-01/ovisocr2-int8__iphone-17-pro.json`, `data/device_runs/0.16.0/2026-09-01/ovisocr2-int8__mac-studio-m4-max.json`, `data/device_runs/0.16.0/2026-09-05/ovisocr2-int8__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-05/ovisocr2-int8__pixel-8a.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | run_failed | no | - | - | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| galaxy-s26 | gpu | pass | yes | - | 207 | - | 537.1 | 35.18 | 410.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| iphone-17-pro | cpu | pass | - | - | - | - | 197.18 | 18.05 | 1022.0 | 914.988 | iPhone 17 Pro · A19-Pro · litert-lm 0.16.0 · iOS 27.0 | 2026-09-01 | measured |
| iphone-17-pro | gpu | pass | - | - | - | - | 602.98 | 48.5 | 1513.0 | 3982.693 | iPhone 17 Pro · A19-Pro · litert-lm 0.16.0 · iOS 27.0 | 2026-09-01 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 256 | - | 659.46 | 48.93 | 429.0 | - | Mac Studio (M4 Max) · litert-lm 0.16.0 | 2026-09-01 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 256 | - | 2174.87 | 140.9 | 133.7 | - | Mac Studio (M4 Max) · litert-lm 0.16.0 | 2026-09-01 | measured |
| pixel-8a | cpu | run_failed | no | - | - | - | - | - | - | - | Pixel 8a · Tensor G3 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |

## Pitfalls

- The OCR post-training erodes general instruction following, so the chat-style gates are not valid gates for this checkpoint: the 8-question text gate reads Mac GPU 4/8 (1 degenerate) / CPU 3/8, but every miss is the model transcribing or echoing the prompt ('17 + 25? / Answer briefly.' echoed back, a 'quick brown fox' non sequitur, numbered-list loops) — not the wrong-token face of quant collapse. The fp32 original on the stock HF stack fails the same questions the same way, question by question (echoes '17 + 25', loops 'Tokyo', loops a numbered list on the French question, and leaks literal think scaffolds into its own output), so the conversion is exonerated per-question, not just by score (FINDINGS.md Mac gates + HF fp32 anchors; HF card 'Scope').
- BANANA hermetic pad sweep 12-51 reads 1/40 and is INVALID for this checkpoint: every 'failure' is the same uniform echo of the filler at every fill count (coherent, not garbage) because the instrument assumes instruction-following ('reply BANANA'), and the fp32 original echoes the filler too. Pad-guard assurance rests instead on the decoder being structurally identical to the Qwen3.5-0.8B VL decoder that passed 40/40 (pad handling is architecture, not weights) and on the OCR E2E runs exercising ~110-token multi-chunk prefill with byte-identical CPU/GPU output (FINDINGS.md).
- Transcription-mode echo on the quality task: the on-device 'quality' legs (iPhone 17 Pro Metal and CPU) produce transcription-mode echo of the prompt — checkpoint behavior, same as Mac — so those rows measure echo output, not answers. This is a document parser, not a chat model: feed it document pages with the upstream reference transcription prompt (verbatim on the card); on-device image legs are grounded (a poster's title transcribed; a photo without text gets the picture-region img tag, exactly its document convention) (FINDINGS.md iPhone gate; HF card 'Scope' + Usage).
- No start token; stop tokens are <|endoftext|> (248044) and <|im_end|> (248046). The upstream repo has no generation_config.json, so both ids were taken from config.json + the tokenizer and verified against the tokenizer at bundle build. Consequence for anyone comparing against stock transformers: HF's eos is only <|endoftext|>, so fp32 generation runs through the turn end and rambles (literal </think>, hallucinated bbox tags, formula repeats) — add 248046 to eos_token_id on the HF side before comparing; the bundle stops cleanly (HF card conversion notes + 'What it does'; FINDINGS.md HF fp32 anchors).
- Requires litert-lm >= 0.15 (HF card header; manifest: hybrid state binds via ExecutorMetadata). litert-torch 0.9.3 writes no ExecutorMetadata section, so it is retrofitted at bundle time (the standing repair) — 48 state buffers (36 linear-attn + 12 kv), prefer_activation_type = fp32 declared (FINDINGS.md Conversion).
- Android/Mali is pending: no Android gate has been run for this build. The northmv law says fp16 vision reboots Mali, so the Android path would be the un-shipped int8-vision variant, whose parts are kept locally (int8/int8 document-fixture corr 0.994-0.998) — not measured on a phone (FINDINGS.md post-ship housekeeping).
- Vision input is static 512x512 and the runtime resizes the attachment to it (pre-render pages at 512x512 to control the aspect ratio). Dense academic pages (9 pt text) exceed what 512x512 can resolve: the opening is transcribed correctly, then the model invents and eventually repeats. The HF fp32 original does the same on the same input — every variant loops on that page, and the bundle matches the 1-D-position fp32 reference byte-for-byte through the entire legible region (first 522 chars), diverging only inside the invented continuation. Upstream's own reference code ships a repeat-cleanup post-processor for exactly this. Use pages legible at 512x512, or tile (HF card 'What it does'; FINDINGS.md HF fp32 anchors).
- Rail inheritance has two per-checkpoint costs. (a) The fp16-safe LayerNorm scales are calibrated per checkpoint — OvisOCR2's have the same shape as base 0.8B (block 6+ at 16, final 512) but absmax moved 2390.9 -> 2803.9, so recalibration is not optional. (b) Positions are 1-D under fast_vlm: the formula page matches the 1-D-position fp32 reference at similarity 1.0000 but the full M-RoPE reference at 0.9968 — a one-token $B$ vs B difference, the position-contract cost (HF card conversion notes + 'What it does'; FINDINGS.md).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
