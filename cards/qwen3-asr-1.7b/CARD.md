---
family: qwen3-asr
license: apache-2.0 (the license of Qwen/Qwen3-ASR-1.7B-hf)
model_id: qwen3-asr-1.7b
source_url: https://huggingface.co/litert-community/Qwen3-ASR-1.7B
task: automatic-speech-recognition
---

# qwen3-asr-1.7b

| | |
|---|---|
| **Task** | automatic-speech-recognition |
| **Family** | qwen3-asr |
| **Source** | https://huggingface.co/litert-community/Qwen3-ASR-1.7B |
| **License** | apache-2.0 (the license of Qwen/Qwen3-ASR-1.7B-hf) |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch export_hf (task automatic_speech_recognition) for the fp32 graphs, with the encoder attention patched to the original 104-token windows; per-section quantization with ai-edge-quantizer; repack with litert-lm pack as a generic_model bundle whose encoder starts at raw PCM frames and computes the Whisper-style log-mel in-graph litert-torch 0.10.0 (731ef0a), transformers 5.14.1, torch 2.13.0, ai-edge-quantizer 0.9.0, ai-edge-litert 2.2.0, litert-lm-builder 0.17.0 (PYTHONPATH overlay), litert-lm (pack) 0.17.1 (litert-lm-builder 0.17.1) |
| **Command** | `export_fp32.py --input_sec 30 --cache_length 1024 (block-diagonal 104-token attention mask as a numpy constant; litert-torch PR #1295 carries the same fix) -> quant_sections.py --lm dynamic_wi8_afp32 --emb dynamic_wi8_afp32 --enc fp16 -> export_bundle.py --window 30 --prompt litert (encoder audio f32 [1,3000,160] -> features f32 [1,390,2048] + mask uint8 [1,390], fp16 weights, in-graph log-mel; generic_model metadata; litert-torch Qwen3-ASR prompt as the jinja). One-command reproduction: https://github.com/john-rocky/hf-to-litertlm/tree/main/qwen3_asr17_work` |
| **Quantization** | LM: int8 dynamic weights with an int8 embedding table (separate embedder section); audio encoder: fp16 weights, fp32 compute; 30 s window (3000 frames x 160 samples at 16 kHz -> 390 audio tokens); KV cache 1024 |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Qwen3-ASR-1.7B.litertlm` | `6155d5d5f8d57e7a3e50f669fd487ae0d8aed0845bb4091f39a894eea7067b95` | 2568.306 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.17.1/2026-10-03/qwen3-asr-1.7b__mac-studio-m4-max.json`, `data/device_runs/0.17.1/2026-10-09/qwen3-asr-1.7b__galaxy-s26.json`, `data/device_runs/0.18.0/2026-10-13/qwen3-asr-1.7b__galaxy-s26.json`, `data/device_runs/0.18.0/2026-10-13/qwen3-asr-1.7b__mac-studio-m4-max.json`, `data/device_runs/0.18.0/2026-10-13/qwen3-asr-1.7b__pixel-8a.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | pass | - | - | sig:Android sample app (LiteRT-Models asr_litertlm), AsrDeviceCheck; Engine on the CPU with 4 threads and the audio encoder on the CPU, one Conversation per clip | 2490.0 | - | - | - | 4327.0 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.17.1 · Android 16 (build BP4A.251205.006.S942QOPS1AZH9) | 2026-10-09 | measured |
| galaxy-s26 | cpu | fallback | no | - | 398 | 2178.7 | 570.73 | 25.6 | 740.0 | 4229.54 | Galaxy S26 (SM-S942Q) · SM8850 · litert-lm 0.18.0 · Android 16 (BP4A.251205.006) | 2026-10-13 | measured |
| galaxy-s26 | gpu | pass | yes | - | 398 | 2770.9 | 587.28 | 24.79 | 720.0 | 2505.65 | Galaxy S26 (SM-S942Q) · SM8850 · litert-lm 0.18.0 · Android 16 (BP4A.251205.006) | 2026-10-13 | measured |
| mac-studio-m4-max | cpu | pass | - | - | - | 1700.9 | - | - | - | 4333.0 | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.17.1 · macOS 27.0 | 2026-10-03 | measured |
| mac-studio-m4-max | cpu | fallback | no | - | 398 | 1243.06 | - | - | - | 4518.756 | Mac Studio (M4 Max) · litert-lm 0.18.0 | 2026-10-13 | measured |
| mac-studio-m4-max | gpu | pass | yes | - | - | 663.5 | - | - | - | 4572.7 | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.17.1 · macOS 27.0 | 2026-10-03 | measured |
| mac-studio-m4-max | gpu | pass | yes | - | 398 | 555.04 | - | - | - | 2492.465 | Mac Studio (M4 Max) · litert-lm 0.18.0 | 2026-10-13 | measured |
| pixel-8a | cpu | fallback | no | - | 398 | 9484.6 | 122.68 | 11.04 | 3330.0 | 4148.64 | Pixel 8a · Tensor G3 · litert-lm 0.18.0 · Android 16 (CP1A.260505.005) | 2026-10-13 | measured |
| pixel-8a | gpu | pass | yes | - | 398 | 10009.0 | 151.32 | 10.4 | 2730.0 | 2702.2 | Pixel 8a · Tensor G3 · litert-lm 0.18.0 · Android 16 (CP1A.260505.005) | 2026-10-13 | measured |

## Pitfalls

- One message = one clip of up to 30 s: the runtime zero-fills the window after the clip and the encoder emits 390 audio tokens per window, so encoding and prefill cost the same for a 4 s and a 29 s clip; clips longer than 30 s were not tested (Hub card 'How to send audio').
- The answer is 'language <name><asr_text><transcript>'; split at <asr_text>. To force the language, send the text item 'language Chinese<asr_text>' in the same message; the model then writes the transcript only (11 of 11 Mandarin / Japanese clips) (Hub card).
- Prompt: the bundle renders the litert-torch Qwen3-ASR prompt (no system turn). With the checkpoint's own chat template (empty system turn) the same weights scored Mandarin CER 6.65 % / Japanese CER 6.06 % vs 6.19 / 5.73 and tagged 8 of 50 Mandarin clips English; the PyTorch model shows the same pattern (6.86 / 5.50 vs 6.04 / 5.24, 9 of 50). System messages are ignored by the template (Hub card 'Prompt').
- The audio encoder must run on the CPU (audio_backend=CPU()): on the Mac the WebGPU/Metal delegate refuses 'BATCH_MATMUL: Not supported batched mat mul case: non-constant tensor' in the encoder graph and create_conversation fails ('Hint fully delegated to single delegate is set, but the graph is not fully delegated'); the LM runs on the GPU (prefill_128 1298/1298 and decode 1244/1244 nodes delegated on Metal) (FINDINGS; gates/rt/q17_30s_C_off_zh_gpu_gpuaudio.log, same encoder section).
- Correctness (FLEURS test, 50 clips per language, greedy, one conversation per clip; scorer: en WER after uppercasing/punctuation removal, zh/ja CER after NFKC+lower with punctuation/whitespace removed): Mac CPU 4.43 / 6.19 / 5.73 %, Mac LM-on-GPU 4.52 / 6.19 / 5.32 %, PyTorch fp32 with the same prompt 4.35 / 6.04 / 5.24 %; no empty transcript (Hub card 'Correctness').
- Stock litert-torch export deviation: the exporter's audio attention attends within 13-token (1 s) chunks while the original attends within 104-token windows (encoder output cosine 0.81-0.82 on 5 s, 0.76 on 30 s); the constant block-diagonal mask makes the fp32 encoder match the original exactly — litert-torch Draft PR #1295 / issue #1296 (Hub card 'Conversion' 1).
- The released runtime's built-in mel front end is not the Qwen3-ASR log-mel; the bundle takes raw PCM frames and computes the original log-mel in the graph (max |Δ| 0.028 vs the original on 10 clips, from the fp16 weights) (Hub card 'Conversion' 2).
- The omni/asr runner in the LiteRT-LM repository was not run with this file; the generic_model bundle is the shipped route (Hub card 'Conversion'; the omni/asr start-token defect is LiteRT-LM issue #3827, measured on confucius4-r2t2).
- Mac speed rows were measured while other jobs ran on the machine (load average 4.5-8.5); the GPU run was the file's first GPU load and wrote the 1.72 GB GPU weight cache during the run (Hub card 'Performance').
- Galaxy S26 CPU (v0.16.1 tag CLI, --benchmark): the first process writes the XNNPACK weight caches (prefill 329 tok/s, TTFT 1.25 s), later processes load them (prefill 538-547 tok/s, TTFT 0.77-0.78 s). After adb push the cache key's mtime second can alternate between processes so every process rebuilds the caches (LiteRT-LM #3772; seen on the first build's S26 leg) — touch the bundle on the phone before timing warm runs (FINDINGS 'Shipped file').

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
