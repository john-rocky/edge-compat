---
family: minicpm5
license: apache-2.0
model_id: minicpm5-2b-int8
source_url: https://huggingface.co/mlboydaisuke/MiniCPM5-2B-LiteRT
task: text-generation
---

# minicpm5-2b-int8

| | |
|---|---|
| **Task** | text-generation |
| **Family** | minicpm5 |
| **Source** | https://huggingface.co/mlboydaisuke/MiniCPM5-2B-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch export on the jinja path with the checkpoint's chat_template.jinja byte-identical (released stack only; repro = hf-to-litertlm `bash scripts/reproduce_llm.sh minicpm5-2b`; lane script minicpm52b_work/convert_minicpm5_2b.sh -> scripts/export_simple_template.py) litert-torch 0.9.3 / litert-converter 0.4.0 / ai-edge-quantizer 0.9.0 / litert-lm-builder 0.16.1 / transformers 5.14.1 (HF card Conversion notes; FINDINGS toolchain ~/venvs/lt093ctl, the granite-4.2-3b rail); card-grade bench and toggle probes on litert-lm 0.17.0 |
| **Command** | `convert_minicpm5_2b.sh (USE_JINJA=1 with the verbatim chat_template.jinja, CACHE=4096, PREFILL=1024,256,64,16,4,1, EXTERNALIZE_EMBEDDER=1, HF_HUB_OFFLINE=1, recipe BOCTAV4 | dynamic_wi8_afp32) -> int4: minicpm52b_work/fix_zero_scales_inplace.py (in-place scale-word patch, keeps the embedder section) ; int8: scripts/set_activation_type.py <int8> <out> --type fp32 (litert-lm 0.17.0 unpack/pack, only model.toml gains prefer_activation_type) (FINDINGS Rail; card Conversion notes)` |
| **Quantization** | dynamic int8 on linears + embedding (dynamic_wi8_afp32, externalized embedder section); prefer_activation_type = fp32 declared in the bundle for the GPU executor (metadata-only repack of the fp16-default export: every section byte-identical, only model.toml differs); 0 zero scales (per-channel quantization of the 13 dead rows is fine) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `MiniCPM5-2B_int8.litertlm` | `58da116a6eb563185eb5e8d0b1f1e1116b632a516cadc0285b6da35b9f886a86` | 2482.68 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-09-08/minicpm5-2b-int8__galaxy-s26.json`, `data/device_runs/0.17.0/2026-09-08/minicpm5-2b-int8__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 203 | - | 102.76 | 11.67 | 2060.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-08 | measured |
| galaxy-s26 | gpu | pass | yes | - | 203 | - | 150.24 | 10.86 | 1440.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-08 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 256 | - | 160.87 | 30.02 | 1624.7 | - | Mac Studio (M4 Max) · litert-lm 0.17.0 | 2026-09-08 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 256 | - | 1405.16 | 74.65 | 195.6 | - | Mac Studio (M4 Max) · litert-lm 0.17.0 | 2026-09-08 | measured |

## Pitfalls

- int8 declares fp32 activations in-bundle: with the runtime's default fp16 GPU activations the int8 model's reasoning on the rhyme gate question ran 2021 tokens without closing </think> (7/8 degenerate, fails the gate); with fp32 declared it closes in ~450 tokens and passes 8/8, at ~14% GPU decode (and ~18% prefill) cost on identical weights. On a ten-question thinking-on subset the two dtypes were mixed (fp16 finished 10/10, fp32act 7/10 with two cap hits) — a measured trade for the ship gate, not a cure (HF card Conversion notes; FINDINGS int8 gates + attribution item 3).
- int8 is the file for reasoning that has to complete: its thinking chains are ~3-4x shorter than int4's on the same questions and terminate where int4 runs into the budget; GSM8K thinking-off 91 vs bf16 92 (HF card).
- int8's main weight section is 2,330,645,824 bytes — above the single-section mmap ceiling of default-entitlement iOS apps — so it is a desktop / Android build (no iPhone row; the Spark-X2.5-4B int8 lane later showed a 4.24 GB section loading under the Increased Memory Limit entitlement, so the ceiling is entitlement-dependent) (HF card; FINDINGS int8 build).
- Galaxy S26 int8 fp32act: GPU gate PASS with full delegation, GPU bench prefill 150-160 / decode 10.9-12.8 tok/s at 0.94-1.10 GB peak VmHWM; CPU 103-157 / 11.7 tok/s at 2.90 GB — the card must say 'fp32 activations declared in-bundle' next to these rows (FINDINGS S26 int8 fp32act).
- Requires litert-lm >= 0.16 (thought channel + ThinkingConfig); measured on 0.17.0. The bundle carries the model's own chat_template.jinja verbatim (9060 chars, byte-compared) on the runtime's jinja path — the litert-community/MiniCPM5-1B packaging — so enable_thinking and the tool-calling format stay reachable; the thought channel (<think>/</think>) is auto-declared by the jinja path and was read out of the built bundle, never assumed (HF card; FINDINGS Rail + int4 header).
- Start token <s> is correct for this family: the template's own {{ bos_token }} renders empty at runtime and the engine prepends the metadata start token, so the model sees exactly one <s> as upstream (verified on the official 1B artifact 2026-08-01); stops are </s> (1) and <|im_end|> (130073). The 14 'X<|im_end|>\n' string stops in the metadata are emitted by litert-torch 0.9.3's own export_hf builder (the tokenizer has no such pieces) — a converter artifact, harmless (FINDINGS premises + int4 header).
- Thinking-OFF baseline: the bf16 PyTorch model scores 8/8 on the 8-question gate with thinking on or unset but 6/8 with thinking forced off ('opposite of hot' -> 'Cool.', rhyme -> 'Green.') — read any thinking-off result against 6/8, not 8/8; both toggle paths (ThinkingConfig and the 0.14-era extra_context) reach the verbatim template on 0.17.0 (HF card Correctness; FINDINGS bf16 oracle + toggle probe).
- GSM8K thinking OFF (the official 1B card protocol; first 100 test questions, greedy, 0-shot CoT prompt, max 2048): bf16 92 / int8 CPU 91 / int4 CPU 86 / int4 GPU fp16 87 / int4 GPU fp32act 88 — int8 at parity (7 of its 9 losses are bf16's own), int4 ~5 points down, and the GPU's fp16 activations cost nothing measurable in no-think mode (HF card; FINDINGS).
- In thinking mode int4's chains are ~4.5x longer than bf16's (median ~13.7k vs ~3k chars on a 10-question subset) and close 0/10 within 3584 tokens on the fp32 CPU reference, while int8 on CPU reproduces bf16 question for question (9/10 finished, same cap hit) — the thinking-length inflation is int4 quantization damage; int4 for direct answers / short thinking, int8 when the reasoning has to complete (HF card; FINDINGS Thinking-ON attribution).
- Galaxy S26 (litert_lm_advanced_main v0.16.0 kit): both files generate correctly on GPU and CPU with full OpenCL delegation (1873/1873 nodes on every prefill signature and decode, zero rejected ops) and the runtime separates the reasoning on-device ([thought] … [/thought], then the answer); on Adreno the int4 GPU decode edge over the same-device CPU is ~1.1x (prefill 5-10x), and the int8 file's fp32 activations bring its GPU decode level with its CPU (11-13 vs 11.7 tok/s) at ~2.6x lower prefill than the int4 file's (HF card Performance; FINDINGS S26 sections).
- Card-grade Mac bench (litert-lm 0.17.0 CLI, -p 256 -d 256 --runs 3 --cache no --max-num-tokens 1024, quiet machine, CPU cells first, >= 300 s rest before every GPU cell, gate_backend.py confirmed real generation on every cell); the earlier gate-time decode figures were taken under load and are not speed claims (FINDINGS Card-grade Mac bench + int8 gates note).
- Block-128 int4 was built and rejected: 7/8 on the CPU gate (the 'merci' question never closes <think>), GSM8K thinking-off 79 vs 87, and the thinking-length problem does not move (12.9k vs 13.7k chars); the 'reasoning ships prefer block128' heuristic does not apply here, as on granite-4.2 (HF card Conversion notes; FINDINGS block-128 A/B).
- Embedder externalised (130560x2048 table in its own section); tokenizer embedded as the upstream tokenizer.json (byte-level BPE, 130,072 base + 510 added tokens; <think>=8 / </think>=9 plain added tokens, /think and /no_think special) (HF card; FINDINGS premises).
- Local bundles, raw exports, compile caches and the HF source cache were deleted at close-out on 2026-09-08 after re-reading the remote sizes; regeneration = the reproduce script (int4 comes out zero-scale-fixed; the int8 ship additionally needs the fp32 activation repack) or a Hub download; two ~4 GB xnnpack caches remain on the S26 (FINDINGS Close-out).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
