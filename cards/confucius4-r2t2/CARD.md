---
family: qwen3-asr
license: other (NetEase Youdao Model Use License Agreement; MODEL_LICENSE + MODEL_LICENSE_zh
  in the repo, the Chinese text prevails)
model_id: confucius4-r2t2
source_url: https://huggingface.co/mlboydaisuke/Confucius4-R2T2-LiteRT
task: automatic-speech-recognition
---

# confucius4-r2t2

| | |
|---|---|
| **Task** | automatic-speech-recognition |
| **Family** | qwen3-asr |
| **Source** | https://huggingface.co/mlboydaisuke/Confucius4-R2T2-LiteRT |
| **License** | other (NetEase Youdao Model Use License Agreement; MODEL_LICENSE + MODEL_LICENSE_zh in the repo, the Chinese text prevails) |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch export_hf (task automatic_speech_recognition) for the fp32 graphs, with the encoder attention patched to the original 104-token windows; per-section quantization with ai-edge-quantizer; repack with litert-lm pack as a generic_model bundle whose encoder starts at raw PCM frames and computes the Whisper-style log-mel in-graph litert-torch 0.10.0 (main 731ef0a / 5a31633, same qwen3_asr.py), transformers 5.14.1, torch 2.13.0, ai-edge-quantizer 0.9.0, ai-edge-litert 2.2.0, litert-lm-builder 0.17.0 (PYTHONPATH overlay), litert-lm (pack) 0.16.0 |
| **Command** | `convert_layout.py (original thinker.* checkpoint -> transformers -hf layout, rename only) -> export_bundle_r2.py --input_sec 30 --cache_length 1024 --quantization_recipe none (block-diagonal 104-token attention mask as a numpy constant; litert-torch PR #1295 carries the same fix) -> quant_sections.py --lm dynamic_wi8_afp32 --emb dynamic_wi8_afp32 --enc fp16 -> export_bundle_r3.py --window 30 --prompt litert (encoder audio f32 [1,3000,160] -> features f32 [1,390,2048] + mask uint8 [1,390], fp16 weights, in-graph log-mel; generic_model metadata; litert-torch Qwen3-ASR prompt as the jinja). One-command reproduction: https://github.com/john-rocky/hf-to-litertlm/tree/main/confucius4_r2t2_work` |
| **Quantization** | LM: int8 dynamic weights with an int8 embedding table (separate embedder section); audio encoder: fp16 weights, fp32 compute; 30 s window (3000 frames x 160 samples at 16 kHz -> 390 audio tokens); KV cache 1024 |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Confucius4-R2T2.litertlm` | `6938f236ec25b0a3c2c8d8a2a43bf6c5891b857331e3246f28b9e95140312c5e` | 2568.306 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.1/2026-10-02/confucius4-r2t2__galaxy-s26.json`, `data/device_runs/0.17.1/2026-10-02/confucius4-r2t2__mac-studio-m4-max.json`, `data/device_runs/0.17.1/2026-10-10/confucius4-r2t2__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | - | - | - | - | - | 7470.0 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.1 · Android 16 | 2026-10-02 | measured |
| galaxy-s26 | cpu | pass | - | - | sig:Android sample app (LiteRT-Models asr_litertlm), AsrDeviceCheck; Engine on the CPU with 4 threads and the audio encoder on the CPU, one Conversation per clip | 3480.0 | - | - | - | 4767.0 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.17.1 · Android 16 | 2026-10-10 | measured |
| galaxy-s26 | gpu | pass | yes | - | - | - | - | - | - | 1932.0 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.1 · Android 16 | 2026-10-02 | measured |
| mac-studio-m4-max | cpu | pass | - | - | - | 1420.0 | - | - | - | 4546.0 | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.17.1 · macOS 27.0 | 2026-10-02 | measured |
| mac-studio-m4-max | gpu | pass | yes | - | - | 610.0 | - | - | - | 2683.0 | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.17.1 · macOS 27.0 | 2026-10-02 | measured |

## Pitfalls

- One message = one clip of up to 30 s: the runtime zero-fills the window after the clip and the encoder emits 390 audio tokens per window, so encoding and prefill cost the same for a 4 s and a 29 s clip; clips longer than 30 s were not tested (Hub card 'How to send audio').
- The answer is 'language <name><asr_text><transcript>'; split at <asr_text>. To force the language, send the text item 'language Chinese<asr_text>' in the same message (the template puts the user's text at the start of the answer) (Hub card).
- Prompt: the bundle renders the litert-torch Qwen3-ASR prompt (no system turn). With the original repository's prompt (empty system turn) the same file scored Mandarin CER 10.75 % / Japanese CER 9.33 % vs 6.40 / 6.21 — Chinese clips tagged English/None, Chinese numerals, two clips translated into English; the PyTorch model shows the same shift (7.78 / 7.47 vs 6.24 / 6.21). System messages are ignored by the template (Hub card 'Prompt').
- The audio encoder must run on the CPU (audio_backend=CPU()): on the Mac the WebGPU/Metal delegate refuses 'BATCH_MATMUL: Not supported batched mat mul case: non-constant tensor' in the encoder graph and create_conversation fails ('Hint fully delegated to single delegate is set, but the graph is not fully delegated'); the LM runs on the GPU (prefill_128 1298/1298 and decode 1244/1244 nodes delegated on Metal and on OpenCL) (FINDINGS round 3; gates/r3/r3_rt_ex5_vendor_gpu_gpuaudio.log).
- Correctness (FLEURS test, 50 clips per language, greedy, one conversation per clip; scorer: en WER after uppercasing/punctuation removal, zh/ja CER after NFKC+lower with punctuation/whitespace removed): Mac CPU 4.43 / 6.40 / 6.21 %, Mac LM-on-GPU 4.52 / 6.45 / 6.06 %, PyTorch fp32 with the same prompt 4.43 / 6.24 / 6.21 %; no empty transcript (Hub card 'Correctness').
- Quantization chosen on the 155 full clips: int8 LM + int8 embedder + fp16 encoder. All-int8 translated one Mandarin clip into English (zh 9.67 %), an fp16 LM stopped after the language tag on one English clip (en 6.26 %) (Hub card 'Conversion' 3; FINDINGS round 2).
- Stock litert-torch export deviation: the exporter's audio attention attends within 13-token (1 s) chunks while the original attends within 104-token windows, which on 5 s clips moved the transcripts by 1.6 / 5.0 / 9.0 points (en / zh / ja); the constant block-diagonal mask makes the fp32 export match the original (60/60 crops) — litert-torch Draft PR #1295 / issue #1296 (Hub card 'Conversion' 1).
- The released runtime's built-in mel front end (HTK-style filterbank on the magnitude spectrum, natural log, frames not centred) is not the Qwen3-ASR log-mel; the bundle takes raw PCM frames and computes the original log-mel in the graph (max |Δ| 0.028 vs the original on 10 clips) (Hub card 'Conversion' 2; GENERIC_CONTRACT.md).
- The omni/asr engine path (litert_lm_main / asr_runner with the qwen3-asr-0.6b metadata entry) is not a valid route for this file: the metadata is compiled in and LmDecoder feeds token 0 when decodeStartTokenId is unset, which drops the prompt's final newline (LiteRT-LM issue #3827; 44/60 crops empty) — the generic_model bundle is the shipped route (FINDINGS rounds 1-3).
- Streaming: the original r2t2 streaming_transcribe loop (2 s chunks; the accumulated audio plus the previous answer minus its last 5 tokens as a text item, in a new conversation per call) runs through the same Conversation API; about 1 s per call on the Mac CPU, committed text revised 2 times in 211 calls (Hub card 'Streaming').
- Galaxy S26 rows are reference values: the CPU frequency limit fell from 3.63 / 4.74 GHz to 1.79 / 1.75 GHz during the CPU leg and the GPU leg started under it; the CPU leg ran without a weight cache (Hub card 'Performance').
- License: NetEase Youdao Model Use License Agreement — a quantized file is a Derivative Work (1.5); clause 4.1(a) requires the disclaimer statement on the distribution page (present on the card); 2.2 separate license above 100 M MAU or the revenue threshold (EN 'RMB 1 billion', ZH '1亿人民币', Chinese prevails per clause 9); 3.4(c) no use to improve other AI models; 4.2 no high-risk deployment (Hub card 'License'; MODEL_LICENSE / MODEL_LICENSE_zh).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
