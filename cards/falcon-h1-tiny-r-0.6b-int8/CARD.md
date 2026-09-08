---
family: falcon-h1
license: other (falcon-llm-license)
model_id: falcon-h1-tiny-r-0.6b-int8
source_url: https://huggingface.co/litert-community/Falcon-H1-Tiny-R-0.6B
task: text-generation
---

# falcon-h1-tiny-r-0.6b-int8

| | |
|---|---|
| **Task** | text-generation |
| **Family** | falcon-h1 |
| **Source** | https://huggingface.co/litert-community/Falcon-H1-Tiny-R-0.6B |
| **License** | other (falcon-llm-license) |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch + falcon_h1 hybrid-cache patch (repro = hf-to-litertlm falcon_h1_work/convert_falcon_tinyr.py + falcon_h1_litert_torch.patch) TODO (owner) — sources name only: litert-torch (no version) + the falcon_h1 patch regenerated against tree 115a136 + working-tree exts (FINDINGS); litert-lm 0.16.0 for the CLI gates, the Mac bench and the litert-lm-builder repack (FINDINGS 'lt0160run'); no litert-torch / litert-converter / ai-edge-quantizer / litert-lm-builder pins are stated in the card, FINDINGS or manifest |
| **Command** | `hf-to-litertlm falcon_h1_work/convert_falcon_tinyr.py (private lane: convert_tinyr.py -> repack_tinyr_ext.py = quantize decoder-only wi8f -> litert-lm-builder tflite_model --model_type prefill_decode --prefer_activation_type fp32 + tflite_model --model_type embedder -> add_executor_metadata -> add_thought_channel; FINDINGS 2026-09-01)` |
| **Quantization** | post-hoc dynamic int8 on linears (FC only); embedding table externalized to its own CPU-side section and kept float; convs and the selective scan stay float; fp32 activations declared for GPU |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Falcon-H1-Tiny-R-0.6B_int8.litertlm` | `66b9e6a5fa630d453d59516cc362e44fff3943eb7c63d07b116cf9a525a1bb94` | 832.801 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-09-01/falcon-h1-tiny-r-0.6b-int8__iphone-17-pro.json`, `data/device_runs/0.16.0/2026-09-01/falcon-h1-tiny-r-0.6b-int8__mac-studio-m4-max.json`, `data/device_runs/0.16.0/2026-09-05/falcon-h1-tiny-r-0.6b-int8__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-05/falcon-h1-tiny-r-0.6b-int8__pixel-8a.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 215 | - | 123.96 | 23.9 | 1780.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| galaxy-s26 | gpu | pass | yes | - | 215 | - | 309.77 | 16.13 | 760.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| iphone-17-pro | cpu | pass | - | - | - | - | 223.93 | 29.38 | 8767.0 | 819.957 | iPhone 17 Pro · A19-Pro · litert-lm 0.16.0 · iOS 27.0 | 2026-09-01 | measured |
| iphone-17-pro | gpu | pass | - | - | - | - | 259.84 | 22.84 | 12883.0 | 3005.848 | iPhone 17 Pro · A19-Pro · litert-lm 0.16.0 · iOS 27.0 | 2026-09-01 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 256 | - | 381.32 | 47.75 | 692.5 | - | Mac Studio (M4 Max) · litert-lm 0.16.0 | 2026-09-01 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 256 | - | 2215.8 | 97.76 | 125.8 | - | Mac Studio (M4 Max) · litert-lm 0.16.0 | 2026-09-01 | measured |
| pixel-8a | cpu | fallback | no | - | 215 | - | 71.16 | 14.02 | 3090.0 | - | Pixel 8a · Tensor G3 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| pixel-8a | gpu | pass | yes | - | 215 | - | 108.77 | 9.98 | 2080.0 | - | Pixel 8a · Tensor G3 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |

## Pitfalls

- Requires litert-lm >= 0.16 (HF card header; manifest platform_notes).
- Reasoning model that self-emits <think>…</think> with no think prefill in the upstream template: give the session an output budget >= 2048 or set --thinking-budget, because truncated mid-thought it produces no final answer (manifest session_defaults). The thought channel (<think>/</think>) is declared post-hoc in LlmMetadata.channels so the runtime separates reasoning from the answer and --thinking-budget works (verified: budget 16 cuts at exactly 16 tokens on Mac CPU/GPU and iPhone). Reasoning length is prompt-sensitive — on nonsense/filler prompts it can think past any budget without closing, reproduced at bf16 (HF card Correctness, Honest notes, Conversion notes; FINDINGS qa/add_thought_channel.py).
- Embedding table externalized and kept float on purpose: an int8 table — even per-row — measurably destabilizes this checkpoint's reasoning (runaway thinking, a repetition loop to budget, greedy flips) that the FC-only recipe does not show, and a float lm_head does not recover it, so the input embedding is the confirmed cause (FINDINGS quant_ab_tinyr.py). The Mac GPU delegate accepts EMBEDDING_LOOKUP only as int8 (float table = hard CHECK crash, fp16 cast = DEQUANTIZE unsupported, partial delegation refused), so quality and GPU delegation are unsatisfiable in one graph; the embedder runs as its own CPU-side section and the decoder graph stays fully GPU-delegable (HF card; FINDINGS trap 3; manifest platform_notes).
- No start_token in the metadata, on purpose: the template carries the literal <|begin_of_text|> so the bundle reproduces the upstream apply_chat_template token stream byte-for-byte from BOS on. The start_token variant drops the pre-BOS space token and that alone costs a correct answer at this size — bf16 A/B on the two streams 8/8 vs 7/8, 'capital of Japan' flips to Hiroshima (HF card Prompt-stream fidelity; FINDINGS).
- Prefill-pad guard from position monotonicity: the externalized decoder graph carries no token ids, so the family's input_ids != 0 pad guard silently disabled and partially-filled prefill chunks corrupted the conv/SSM state on CPU (17+25 -> 19.5; float export identical, so not quantization; GPU unaffected because its pad values happen to be benign). Pad positions are now identified by non-increasing position ids and made exact identity steps for the SSM (HF card Conversion notes; FINDINGS trap 4, patch regenerated with the fallback).
- Quality gates: GSM8K (100 questions, greedy, 2048-token budget, scored after </think>) int8 76/100 vs bf16 PyTorch 71/100 on the identical prompt stream — 64 both-correct, 7 bf16-only, 12 engine-only, i.e. flips in both directions = quantization noise, not degradation (HF card Correctness; manifest quality). 8-question sanity gate (litert-lm 0.16.0 CLI, 2048 budget): Mac CPU 6/8, Mac GPU 7/8, iPhone 17 Pro CPU 6/8, iPhone Metal 6/8, zero unclosed thinks — every reasoning/logic/math item passes on every backend; the misses are two fact-recall items ('capital of Japan', 'thank you in French') that the bf16 model also flips under one-token prompt perturbations (HF card Correctness; FINDINGS re-gate table). Gate through the released CLI: the litert-mac-verify Aug-13 framework build produces greedy runaway thinking on this bundle that the CLI does not (FINDINGS).
- Multi-turn fact recall is weak at this size: the 3-turn gate's turn-3 'what did I tell you earlier' question fails, and the bf16 model fails it identically — treat it as a single-turn reasoner (HF card Honest notes; manifest session_defaults; FINDINGS multiturn row).
- GPU runs with fp32 activations (declared in the bundle) — expect a corresponding memory multiple over CPU. On iPhone 17 Pro the CPU backend decodes faster than Metal at this size (GPU setup/dispatch overhead dominates a 0.6B model), so the manifest recommends CPU on iOS and GPU on macOS; the GPU row exists for completeness and for devices where the CPU is busy (HF card Honest notes; manifest recommended).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
