---
family: smollm3
license: apache-2.0
model_id: smollm3-3b-int8
source_url: https://huggingface.co/litert-community/SmolLM3-3B
task: text-generation
---

# smollm3-3b-int8

| | |
|---|---|
| **Task** | text-generation |
| **Family** | smollm3 |
| **Source** | https://huggingface.co/litert-community/SmolLM3-3B |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch export_hf (card L183-L190, written for the repo's int4 file; SmolLM3ForCausalLM on the existing converter, NoPE attention lowers to generic ops); rebuilt on 2026-10-06 with hf-to-litertlm tools/gpu_graph/ (card L192-L199) TODO (the card names the converter but not its version) |
| **Command** | `TODO for the export (not stated in the card); the 2026-10-06 rewrite of the previous file: tools/gpu_graph/gpu_graph_retrofit.py previous.litertlm retrofit.litertlm --decomp exporter --mask add_bcast, then tools/gpu_graph/prefill_bucket_clone.py build retrofit.litertlm new.litertlm --lengths 16 --source prefill_256 --report bucket.json (card L195-L199)` |
| **Quantization** | int8 (card L38 names the second file int8 and gives no recipe beyond that) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `SmolLM3-3B.litertlm` | `c1c6c635194745456d03ba3a851b9ebf3a2714218ce35d930d9eb93f7cf97e20` | 2980.427 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-24/smollm3-3b-int8__galaxy-s26.json`, `data/device_runs/0.16.1/2026-09-01/smollm3-3b-int8__raspberry-pi-5.json`, `data/device_runs/0.17.1/2026-10-05/smollm3-3b-int8__mac-studio-m4-max.json`, `data/device_runs/prebuilt-adac974b/2026-10-05/smollm3-3b-int8__galaxy-s26.json`, `data/device_runs/selfbuilt-3dbb23e1e/2026-10-05/smollm3-3b-int8__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 1024 | - | 165.95 | 9.7 | 6290.0 | 4355.8 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm prebuilt-adac974b · Android 16 | 2026-10-05 | measured |
| galaxy-s26 | gpu | pass | yes | - | 16 | - | 241.56 | 16.73 | 130.0 | 999.0 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm prebuilt-adac974b · Android 16 | 2026-10-05 | measured |
| galaxy-s26 | gpu | pass | yes | - | 211 | - | 312.27 | 14.41 | 750.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-24 | measured |
| galaxy-s26 | gpu | pass | yes | - | 1024 | - | 452.19 | 16.02 | 2330.0 | 999.0 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm prebuilt-adac974b · Android 16 | 2026-10-05 | measured |
| galaxy-s26 | npu | load_failed | - | - | - | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm selfbuilt-3dbb23e1e · QAIRT 2.47.0.260601114230 (QNN JIT libraries) · Android 16 | 2026-10-05 | measured |
| mac-studio-m4-max | cpu | fallback | no | - | 16 | - | 230.15 | 11.07 | 1546.2 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.17.1 | 2026-10-05 | measured |
| mac-studio-m4-max | cpu | fallback | no | - | 256 | - | 459.79 | 24.18 | 1995.7 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.17.1 | 2026-10-05 | measured |
| mac-studio-m4-max | gpu | pass | yes | - | 16 | - | 819.21 | 100.22 | 37.6 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.17.1 | 2026-10-05 | measured |
| mac-studio-m4-max | gpu | pass | yes | - | 256 | - | 1640.66 | 105.96 | 168.3 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.17.1 | 2026-10-05 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 19.1 | 1.97 | 14036.7 | 4247.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-01 | measured |

## Pitfalls

- GPU graph rewrite of 2026-10-06 (weights unchanged): the two KV-cache writes per layer became one odml.cache_update composite, the attention products odml.runtime_bmm composites, a param_tensor input (INT32[1,1,1,7]) was added and a 16-token prefill signature copied from prefill_256 (it shares every weight buffer); the file is now 3,125,204,208 bytes (previous 3,109,328,112, f9aa844c…); on the Mac with litert-lm 0.17.1 its GPU decode went 77.0 → 106.0 tok/s and its CPU init 6.7 → 9.4 s (the CLI's printed Init time, which counts engine creation twice: LiteRT-LM #4057, card L127); with the default fp16 activations on the Mac GPU it gets 7 of the 8 questions right and passes the long prompt, with and without the new signature (card L41-L50).
- Galaxy S26 GPU: every transformer signature of this file is fully delegated, one partition each on LITERT_CL (prefill_256 1512/1512 nodes, prefill_16 1512/1512, decode 1346/1346), and it answered 5 of 5 test prompts (litert_lm_advanced_main from a LiteRT-LM main build of 2026-09-18, bundle default fp16 activations) (card L168-L177).
- CPU memory with the extra signature: --cache no runs the CPU without the XNNPACK weight cache, and then this file repacks its weights for the 16-token signature: init 6.62 → 9.62 s and peak footprint 6.86 → 9.48 GB on a 16-token prompt (without → with the signature); with the disk weight cache (the app default) init goes 0.41 → 0.56 s, footprint 1.01 → 0.87 GB and the 16-token TTFT 0.7 → 0.17 s (card L129).
- Raspberry Pi 5 CPU: 19.1 tok/s prefill / 2.0 decode, measured on the previous file (f9aa844c…) before the 2026-10-06 rewrite and not re-measured (card L236-L244); this repo's Pi 5 row is the wave-2 journal (device runs).
- The tokenizer in SmolLM3-3B.litertlm was a SentencePiece conversion of the model's BPE tokenizer that lost byte-level semantics for standalone accented letters/symbols, and both bundles later received a metadata-only repair declaring the reasoning channel (thought, <think>…</think>) and a default-system-turn change (every section except LlmMetadata byte-identical, sha256 per section) — read the card's 'Metadata-only change' notes for the exact state of the file you download (card L212-L234).
- Template: on 2026-09-21 the chat template was updated to accept the 0.18 content-parts form (the string form unchanged); weights, tokenizer and executor metadata byte-identical (card L52).
- Conversation preface with tools and no messages: LiteRT-LM renders the chat template with an empty message list, and the template reads messages[0] without checking that the list is non-empty, so the render fails with 'Failed to apply template: undefined value' when the preface is prefilled at creation (prefillPrefaceOnInit, default false; creating the conversation fails) and in the preface render (renderPrefaceIntoString and its C++ and C API forms); the same with extra context (such as enable_thinking) and no messages, and in the preface render with an empty preface (run in the runtime on one Qwen3 template; the other templates rendered with minijinja 2.14 and jinja2); read in the source of LiteRT-LM 0.17.0, 0.18.1 and main, measured with the 0.18.1 PyPI package through its C API on these files' template texts; the default path, where the first message is rendered with the preface, is not affected (the 0.18.1 Python API and CLI expose neither the flag nor the preface render); a system message or any initial message in the conversation config avoids it (card L249).
- License Apache-2.0, inherited from HuggingFaceTB/SmolLM3-3B (card L253-L254).
- LiteRT-LM .litertlm bundle: LiteRT.js cannot run it, so delegation stays null and there is no browser block; device rows come from data/device_runs/.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
