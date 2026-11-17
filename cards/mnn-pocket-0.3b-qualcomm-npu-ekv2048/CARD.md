---
family: mnn-pocket
license: apache-2.0
model_id: mnn-pocket-0.3b-qualcomm-npu-ekv2048
source_url: https://huggingface.co/litert-community/MNN-Pocket-0.3B
task: text-generation
---

# mnn-pocket-0.3b-qualcomm-npu-ekv2048

| | |
|---|---|
| **Task** | text-generation |
| **Family** | mnn-pocket |
| **Source** | https://huggingface.co/litert-community/MNN-Pocket-0.3B |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch at commit 6d4c622, npu_export in three steps (the int8 weights of dynamic_wi8_afp32, activation ranges from 48 prompts, static quantization of the activations to 16-bit integers), packed with litert-lm-builder 0.17.0; repro = hf-to-litertlm REPRODUCE.md, entry "2026-11-07 — the Qualcomm NPU file" (card L196, L201) litert-torch 6d4c622, litert-lm-builder 0.17.0; on the phone LiteRT-LM 0.19.1 built from the v0.19.1 tag with Qualcomm's dispatch and compiler-plugin libraries and the QNN libraries of QAIRT 2.47.0.260601 (card L67, L194, L196) |
| **Command** | `taobao-mnn/MNN-Pocket-0.3B rev 087a52fd -> npu_export, prefill signature 128, KV cache 2,048; calibration on 32 questions from GSM8K's training split (in the 4-shot form of the GSM8K check) and the first 16 of litert-torch's sample calibration prompts; two changes to npu_export's output: the prefill attention mask broadcast over the query heads (557,056 bytes instead of 3,899,392) and the 11 ranges of the two rotary-position graphs raised to 2,047.9375 (card L39, L193, L196)` |
| **Quantization** | int8 weights (one scale per output channel) with 16-bit integer activations (static ranges); KV cache of 2,048 tokens; one prefill signature of 128 tokens; no compiled NPU code (the phone compiles it at engine creation); model type qwen2p5; the int8 embedding table held twice with the same values, for the input embedding and the output layer (card L33, L39, L41, L187) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `MNN-Pocket-0.3B_qualcomm_npu_ekv2048.litertlm` | `e8af457e3d2840a1c5b35f66e7e8bf17d1c88212608f2ec0c374ef01966060a0` | 356.878 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.19.1/2026-11-07/mnn-pocket-0.3b-qualcomm-npu-ekv2048__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | npu | fallback | no | pass | sig:GSM8K-100 on the NPU through the litert-lm-api 0.19.1 Android wheel (the test split's first 100, 4-shot chain of thought, greedy, cap 1,024 new tokens, one pass); bar 53 = the source model in bfloat16 (58) minus 5 | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.19.1 · QAIRT 2.47.0.260601114230 (QNN JIT libraries) · Android 16 (BP4A.251205.006) | 2026-11-07 | measured |
| galaxy-s26 | npu | fallback | no | pass | sig:the same replies through litert_lm_advanced_main built from the v0.19.1 tag (the speed record's binary and libraries): 8 short questions, two three-turn conversations and GSM8K questions 1-40, each full reply compared character for character with the Android wheel's reply | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.19.1 · QAIRT 2.47.0.260601114230 (QNN JIT libraries) · Android 16 (BP4A.251205.006) | 2026-11-07 | measured |
| galaxy-s26 | npu | fallback | no | - | 1024 | - | 10801.9775 | 147.2525 | 100.0 | 3427.3116 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.19.1 · QAIRT 2.47.0.260601114230 (QNN JIT libraries) · Android 16 (BP4A.251205.006) | 2026-11-07 | measured |

## Pitfalls

- Compile at every engine creation: each new engine compiles the file for the NPU before the first reply, 27 to 29 s on a Galaxy S26 whose CPU was not capped when the engine started, longer once earlier compiles have capped it; LiteRT-LM 0.19.1 keeps no compiled copy between processes (card L102). The process's peak memory reached 3.43 GB and the phone's MemAvailable fell to 4.12 GB during the compile (card L140, L143).
- Needs Android binaries built from source: LiteRT-LM 0.19.1 releases no Android build of the tool or of its NPU libraries, so build them from the v0.19.1 tag with Qualcomm's compiler plugin added and take the QNN libraries from QAIRT 2.47.0.260601 (card L67, L187).
- LiteRT-LM 0.19.1's CPU backend does not open the file, and no Mac backend computes its 16-bit integer graph the way the phone's NPU does, so there is no Mac reply to match word for word (card L39, L177).
- Greedy on the NPU: LiteRT-LM 0.19.1 picks the most likely token at every step on the NPU, so the file's sampling settings do not apply there (card L102).
- One prefill signature of 128 tokens: a longer prompt is prefilled 128 tokens at a time (card L188).
- Attention mask size: on the SM8850 NPU with QAIRT 2.47 an int16 mask concatenated to more than 1 MiB was added to the wrong rows (litert-torch#1184; QAIRT 2.50 adds the issue's two-operation case correctly), so the mask is broadcast over the query heads and stays under 1 MiB (card L196).
- Rotary-position ranges: the calibrated ranges ended at the largest positions the calibration reached (511 in prefill, 527 in decode); with them the cosines and sines were off by up to 1.99 past those positions and 31 of the first 40 GSM8K replies on the Galaxy S26 NPU ran to the 1,024-token limit, against 10 with the ranges raised to 2,047.9375 (card L196).
- CPU work around the NPU: the main graph runs whole on the NPU (1,477 operations of prefill, 1,376 of decode); the embedding lookup and the small graphs for the rotary position, the attention mask and the cache update run on the CPU, and the phone capped the CPU clocks during the compile and lowered them further during the runs (card L136, L145).
- One phone: the file was run on a Galaxy S26 (Snapdragon SM8850) only; another Snapdragon phone compiles it for its own NPU at engine start, untested (card L187).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
