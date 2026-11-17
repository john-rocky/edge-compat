---
family: falcon-h1
license: other (falcon-llm-license)
model_id: falcon-h1-1.5b-deep-instruct-int8
source_url: https://huggingface.co/litert-community/Falcon-H1-1.5B-Deep-Instruct
task: text-generation
---

# falcon-h1-1.5b-deep-instruct-int8

| | |
|---|---|
| **Task** | text-generation |
| **Family** | falcon-h1 |
| **Source** | https://huggingface.co/litert-community/Falcon-H1-1.5B-Deep-Instruct |
| **License** | other (falcon-llm-license) |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch plus a hybrid-cache patch (reproduction script + patch: hf-to-litertlm falcon_h1_work/) (card L98) TODO (the card names the converter but not its version) |
| **Command** | `TODO (hf-to-litertlm falcon_h1_work/; invocation not stated in the card)` |
| **Quantization** | post-hoc dynamic int8 over linears + embedding only; convs and the scan stay float; fp32 activations declared for GPU (card L30, L104) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Falcon-H1-1.5B-Deep-Instruct_int8.litertlm` | `e101fc217a39f29518de78ca06a07cac871fadaeb347071270ef721c2ff9bfee` | 1748.166 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-24/falcon-h1-1.5b-deep-instruct-int8__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-05/falcon-h1-1.5b-deep-instruct-int8__galaxy-s26.json`, `data/device_runs/0.16.1/2026-09-02/falcon-h1-1.5b-deep-instruct-int8__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 212 | - | 103.14 | 11.97 | 2140.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| galaxy-s26 | gpu | pass | yes | - | 212 | - | 120.58 | 6.0 | 1920.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-24 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 24.59 | 2.05 | 10897.7 | 3367.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-02 | measured |

## Pitfalls

- Requires litert-lm >= 0.15 (card L24).
- Template: on 2026-09-21 the file's chat template was updated to accept the 0.18 content-parts form (the string form unchanged); weights, tokenizer and executor metadata byte-identical (card L32).
- Correctness: float export matches the HF model teacher-forced across 48 decode positions (top-1/top-5 identical, corr 1.0000, KL ≈ 0); 8-question check 8/8 on every path (GPU and CPU, litert-lm 0.15.0 and 0.16.0); hermetic prefill-chunk sweep clean (CPU fills 12-51, GPU 12-31); iPhone 17 Pro composite test 8/8 on GPU and CPU (card L36-L39).
- One Engine per conversation: on litert-lm 0.18.0 a new Conversation on an Engine that already ran another conversation starts from the earlier conversation's recurrent state (it reuses the prompt tokens the two share and does not reset the state; LiteRT-LM issue #3165); in the card's probe (three questions in a pirate-persona conversation, then in one told to answer in one word, the same Engine against a fresh Engine in a new process; CPU, litert-lm 0.18.0 on a Mac) this file answered all three in full sentences on the same Engine against one-word answers on the fresh one, in both runs; create a new Engine for each conversation, the first conversation on an Engine is unaffected (card L40).
- Hybrid packaging: composite hybrid cache layer (KV + conv + recurrent state at one layer index), folded selective scan as batched matmuls with chunk/head axes in the batch axis (all tensors rank <= 4, no BROADCAST_TO, no int64 index math — what makes the graph fully delegable on GPU), Falcon µP multiplier vector and ssm_in_multiplier preserved, prefill-pad guard (card L100-L103).
- Galaxy S26 GPU per the card: runs, 115393/115393 ops across 12 subgraphs on LiteRT GPU (card L81); this repo's S26 CPU row is from the S7 backfill (device runs).
- Conversation preface with tools and no messages: LiteRT-LM renders the chat template with an empty message list, and the template reads messages[0] without checking that the list is non-empty, so the render fails with 'Failed to apply template: undefined value' when the preface is prefilled at creation (prefillPrefaceOnInit, default false; creating the conversation fails) and in the preface render (renderPrefaceIntoString and its C++ and C API forms); the same with extra context (such as enable_thinking) and no messages, and in the preface render with an empty preface (run in the runtime on one Qwen3 template; the other templates rendered with minijinja 2.14 and jinja2); read in the source of LiteRT-LM 0.17.0, 0.18.1 and main, measured with the 0.18.1 PyPI package through its C API on this file's template text; the default path, where the first message is rendered with the preface, is not affected (the 0.18.1 Python API and CLI expose neither the flag nor the preface render); a system message or any initial message in the conversation config avoids it (card L116).
- License: Falcon LLM License inherited from the base model; changes = weights converted and quantized as stated, tokenizer and chat template repackaged unmodified; community conversion (card L120).
- LiteRT-LM .litertlm bundle: LiteRT.js cannot run it, so delegation stays null and there is no browser block; device rows come from data/device_runs/.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
