---
family: mnn-pocket
license: apache-2.0
model_id: mnn-pocket-0.3b-qualcomm-sm8850-ekv2048
source_url: https://huggingface.co/litert-community/MNN-Pocket-0.3B
task: text-generation
---

# mnn-pocket-0.3b-qualcomm-sm8850-ekv2048

| | |
|---|---|
| **Task** | text-generation |
| **Family** | mnn-pocket |
| **Source** | https://huggingface.co/litert-community/MNN-Pocket-0.3B |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | the NPU file compiled with litert-torch's NPU compiler at commit 6d4c622 (compile_litertlm, backend qualcomm, SoC SM8850, the small graphs left to the phone's CPU); repro = hf-to-litertlm REPRODUCE.md, entry "2026-11-07 — the SM8850 file" (card L197, L201) litert-torch 6d4c622 in a container with ai-edge-litert 2.2.0 and the QNN libraries of QAIRT 2.47.0.260601 (ai-edge-litert-sdk-qualcomm 2.2.0); on the phone LiteRT-LM 0.19.1 built from the v0.19.1 tag with the Qualcomm dispatch library and the QNN libraries, without the compiler plugin (card L194, L197) |
| **Command** | `MNN-Pocket-0.3B_qualcomm_npu_ekv2048.litertlm -> compile_litertlm (backend qualcomm, SoC SM8850), optimization level O3, one compiled context of 319,000,576 bytes for both signatures with their weights shared; every section of the NPU file byte for byte except the main graph, held compiled (card L41, L197)` |
| **Quantization** | as the NPU file: int8 weights (one scale per output channel) with 16-bit integer activations; KV cache of 2,048 tokens; one prefill signature of 128 tokens; the main graph compiled for the SM8850 (card L33-L34, L41) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `MNN-Pocket-0.3B_qualcomm_sm8850_ekv2048.litertlm` | `ab3aa3aa48a27f2170caf025deb59dc5f2e6c392ff6968d73100f3a02c178138` | 362.753 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.19.1/2026-11-07/mnn-pocket-0.3b-qualcomm-sm8850-ekv2048__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | npu | fallback | no | pass | sig:the NPU file's replies, through litert_lm_advanced_main built from the v0.19.1 tag without the compiler plugin: 8 short questions, two three-turn conversations and GSM8K questions 1-40, each full reply compared character for character with the NPU file's (blank lines at the end of a reply not compared) | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.19.1 · QAIRT 2.47.0.260601114230 (QNN libraries on the phone; no compiler plugin) · Android 16 (BP4A.251205.006) | 2026-11-07 | measured |
| galaxy-s26 | npu | fallback | no | - | 1024 | - | 11198.8125 | 157.7375 | 100.0 | 166.3672 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.19.1 · QAIRT 2.47.0.260601114230 (QNN libraries on the phone; no compiler plugin) · Android 16 (BP4A.251205.006) | 2026-11-07 | measured |

## Pitfalls

- SM8850 only: the file is compiled for the Snapdragon SM8850 with QAIRT 2.47.0.260601 and was run only on the Galaxy S26 with that version's QNN libraries; a phone with another Snapdragon chip should take the NPU file (card L39, L189).
- Needs Android binaries built from source, like the NPU file: the v0.19.1 tag's command-line tool with the Qualcomm dispatch library and the QAIRT 2.47.0.260601 QNN libraries (card L104, L189).
- Compiler plugin: removing it from the library folder is optional; with it left in, the file gave the same reply but the engine took 0.37 s instead of 0.27 s while the plugin tried to compile the small graphs for the NPU, and QNN printed errors (one run each). Push it back before running the NPU file again (card L117).
- One prefill signature of 128 tokens: a longer prompt is prefilled 128 tokens at a time (card L188).
- Greedy on the NPU: LiteRT-LM 0.19.1 picks the most likely token at every step on the NPU, so the file's sampling settings do not apply there (card L102).
- Speed against the NPU file: in the same session its processes ran with the CPU clocks capped less than the NPU file's, so part of the decode difference may come from the CPU (which runs the embedding lookup and the small graphs) and part from the two compiles, which differ; neither was measured (card L143).
- Accuracy scope: its replies equal the NPU file's on the 8 short questions, the two three-turn conversations and the first 40 GSM8K questions; the other 60 GSM8K questions were not run on it (card L179).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
