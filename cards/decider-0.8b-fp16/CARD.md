---
family: decider
license: apache-2.0
model_id: decider-0.8b-fp16
source_url: https://huggingface.co/litert-community/decider-0.8b-LiteRT
task: text-classification
---

# decider-0.8b-fp16

| | |
|---|---|
| **Task** | text-classification |
| **Family** | decider |
| **Source** | https://huggingface.co/litert-community/decider-0.8b-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch fork john-rocky/litert-torch @115a13607c730c81018bb9789138a3e5e5119e3d with the owner's hybrid patch (card 'License and changes') see the Hub card's Reproduction link (hf-to-litertlm decider_work/); runtime checks on LiteRT-LM 0.17.1 (describe + engine) and ai-edge-litert 2.2.0 (graph reader) |
| **Command** | `hf-to-litertlm decider_work/ (converter, gates, fixtures); source checkpoint Mapika/decider-0.8b rev 1ea54127d3bd52f6d753d9257b32a6380b873907` |
| **Quantization** | FP16 casting of fully connected and embedding weights; float compute; fp32 activation preference (Hub card 'Files', 'License and changes'). |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `decider-0.8b_fp16.litertlm` | `477c1dcadbb229833cbcb9613f3436ad7057dcdbc4de2038ca3e8f4ec0485f25` | 1549.774 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-09-23/decider-0.8b-fp16__galaxy-s26.json`, `data/device_runs/0.17.1/2026-09-23/decider-0.8b-fp16__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 262 | - | 44.94 | - | 5920.0 | 6389.82 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-23 | measured |
| galaxy-s26 | gpu | load_failed | no | - | 262 | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-23 | measured |
| galaxy-s26 | gpu | load_failed | no | - | 503 | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-23 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 256 | - | 533.23 | - | 533.5 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.17.1 · macOS 27.0 (26A428) | 2026-09-23 | measured |

## Pitfalls

- Probability fidelity: on Mac CPU (exact-fit chunked prefill, ai-edge-litert 2.2.0 graph reader) all 120 fixture rows keep the upstream fp32 argmax with max |dp| 6.6e-6; the Hub card recommends this file for desktop/CPU (card 'Correctness and probability preservation').
- Not recommended on phones (card 'Performance'): on the Galaxy S26 (LiteRT-LM v0.16.0 S4 kit) the 262-token CPU row ran at 44.94 tok/s prefill, 5.92 s TTFT and 6,094 MiB peak; the 503-token CPU row lost the device (framework restart); GPU engine creation failed because partial delegation was refused (the device record quotes the log: 25,656 of 25,885 prefill_1024 nodes on LITERT_CL, 'Hint fully delegated to single delegate is set, but the graph is not fully delegated').
- The model returns option probabilities, not text: build one independent row per decision (score questions expand to one yes/no row per level), prefill ids[:-1], decode ids[-1] once, read only the label logits at the last token and softmax at T = 1.03; generated text from runtime benchmarks is not a model capability claim (card 'Readout contract', 'Limitations').
- GPU feeding rule: zero all state per row and feed ids[:-1] in ONE padded prefill chunk (the smallest exported prefill signature that fits); chaining several prefill chunks on the GPU corrupts the carried state of this hybrid graph (Mac Metal max |dp| 0.177 multi-chunk vs 9.8e-6 one padded chunk). Rows longer than 1024 tokens run on CPU (card 'Readout contract').
- Bundle metadata: identity template, no start token, stop token 248044, ExecutorMetadata for 48 state buffers, fp32 activation preference; at most 4096 tokens per rendered row; requires LiteRT-LM >= 0.15 for generalized hybrid-state binding (card intro, 'Files').
- Engine scoring (RunTextScoring) advances session state on this hybrid architecture: use a fresh process/engine per scoring call; shared-engine prefix reuse can retain stale recurrent/conv state (card 'Readout contract').
- Validation is conversion agreement on 40 synthetic English fixtures (120 rows) against pinned upstream fp32 on Apple M4 Max CPU, not real-traffic calibration or task accuracy (card 'Correctness', 'Limitations').

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
