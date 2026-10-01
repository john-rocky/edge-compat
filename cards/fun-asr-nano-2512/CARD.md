---
family: fun-asr
license: apache-2.0
model_id: fun-asr-nano-2512
source_url: https://huggingface.co/litert-community/Fun-ASR-Nano-2512
task: automatic-speech-recognition
---

# fun-asr-nano-2512

| | |
|---|---|
| **Task** | automatic-speech-recognition |
| **Family** | fun-asr |
| **Source** | https://huggingface.co/litert-community/Fun-ASR-Nano-2512 |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (audio-encoder export from a self-contained torch port of funasr's fbank + LFR + SAN-M + adaptor; LM export through scripts/export_simple_template.py) + ai-edge-quantizer (fp16 casting of the encoder) + litert-lm-builder bundle assembly (repro = hf-to-litertlm funasr_nano_work/: make_fixtures.py -> oracle_funasr.py -> build_lm_native.py -> port_eval.py -> export_audio_encoder.py -> quant_encoder.py --variants fp16 -> export_simple_template.py -> litert-lm unpack -> build_bundle.py --prefer-act fp32 -> mac_gate.py) litert-torch 0.9.4 (audio encoder, ~/venvs/lt094dev: torch 2.13.0, transformers 5.14.1, ai-edge-quantizer 0.9.0, ai-edge-litert 2.2.0, litert-lm-builder 0.16.1) / 0.9.3 (LM export, ~/venvs/ltconv040dev, ai-edge-quantizer 0.8.0); reference = funasr 1.4.16 fp32 in its own venv; runtime gates litert-lm-api 0.17.1 (Mac) and litert_lm_advanced_main v0.16.1 tag build (Galaxy S26) (Hub card Conversion notes; FINDINGS Env) |
| **Command** | `export_audio_encoder.py (one signature: audio f32 [1,504,960] raw 16 kHz PCM in 960-sample frames -> features f32 [1,63,1024] + mask uint8 [1,63]; Kaldi fbank as constant DFT/mel matmuls, LFR 7/6, SAN-M x70, adaptor; the clip length inferred from the last non-zero sample; fp32 export) -> quant_encoder.py --variants fp16 (FLOAT_CASTING, frontend matmuls kept fp32) -> EXTERNALIZE_EMBEDDER=1 CACHE=2048 PREFILL=512,128,32 export_simple_template.py out/lm_native … chatml_simple.jinja dynamic_wi8_afp32 -> litert-lm unpack -> build_bundle.py --enc audio_encoder_504f_fp16.tflite --lm lm_int8/unpack --prefer-act fp32 (GenericModel audio metadata: skip_mel_spectrogram_extraction, 16 kHz, frame=hop=960, delimiter/audio_token_regex on <|AUDIO|>, no start/end audio tokens; ChatML jinja with the default instruction 语音转写：; stops <|im_end|>/<|endoftext|>; sections EMBEDDER, PREFILL_DECODE (prefer_activation_type fp32), AUDIO_ENCODER_HW, HF tokenizer) (FINDINGS rounds 1-3; REPRODUCE entry)` |
| **Quantization** | LM: int8 dynamic weights with an int8 embedding table in a separate embedder section, fp32 activations declared on the prefill/decode section (with the GPU's default fp16 activations the LM emits only '!' tokens, also text-only); audio encoder: fp16 weights, fp32 compute, 30.24 s window (Hub card File table + Conversion notes) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Fun-ASR-Nano-2512.litertlm` | `508c3c7e2ea0df9a756d4b42f9751d5aa08db3d60f959ade6bb2c2a2dafebcd4` | 1197.715 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.1/2026-09-27/fun-asr-nano-2512__galaxy-s26.json`, `data/device_runs/0.17.1/2026-09-27/fun-asr-nano-2512__galaxy-s26.json`, `data/device_runs/0.17.1/2026-09-27/fun-asr-nano-2512__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | pass | - | - | - | 1237.1 | - | - | - | 2780.1 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.17.1 · Android 16 | 2026-09-27 | measured |
| galaxy-s26 | cpu | fallback | no | - | 256 | - | 331.43 | 31.55 | 800.0 | 2411.11 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.1 · Android 16 | 2026-09-27 | measured |
| galaxy-s26 | gpu | pass | yes | - | 256 | - | 386.45 | 26.26 | 700.0 | 2233.34 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.1 · Android 16 | 2026-09-27 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 256 | - | 1027.03 | 48.94 | 480.3 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.17.1 · macOS 27.0 | 2026-09-27 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 512 | - | 1349.72 | 50.21 | 620.9 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.17.1 · macOS 27.0 | 2026-09-27 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 256 | - | 3090.74 | 143.03 | 96.7 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.17.1 · macOS 27.0 | 2026-09-27 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 512 | - | 6169.85 | 137.18 | 97.4 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.17.1 · macOS 27.0 | 2026-09-27 | measured |

## Pitfalls

- One message = one utterance of up to 30.24 s (504 frames x 960 samples): a longer clip in one message is cut by the runtime into windows and the model transcribes only the first window's speech (60.2 s test clip: WER 83/143). Split longer audio into pieces of at most 30 s and send each piece in a NEW conversation, then join the texts (same clip cut at 30.24 s: WER 10/143 = the original model on the whole clip; the two pieces as two turns of one conversation: 22/143, the second turn repeats the start of the first) (Hub card How to send audio; FINDINGS round 3 B).
- The audio encoder must run on the CPU (audio_backend): on the Mac the GPU delegate does not take DEQUANTIZE / PAD / SELECT_V2 of the fp16-weight encoder and conversation creation fails ('Hint fully delegated to single delegate is set, but the graph is not fully delegated'); the LM runs on CPU or GPU (Hub card Known issues; FINDINGS round 2 row 4).
- The prefill/decode section declares prefer_activation_type fp32: with the runtime's default fp16 GPU activations the LM emits token 0 ('!') at every step with 'Invalid decode and sample result … casted to 0', also for a text-only prompt, on Mac WebGPU and on the S26 OpenCL path; leave the engine activation type unset (Hub card Conversion notes; FINDINGS round 3 A).
- Correctness (25 clips = 5 official samples + 20 LibriSpeech dev-clean, 448 words; reference = funasr 1.4.16 fp32, greedy): with the encoder and LM left in fp32 the runtime reproduces the reference text 25/25, so every difference is quantization — int8 LM: Mac CPU 21/25 (WER 19/448), Mac GPU 24/25 (20/448), S26 CPU 21/25 (20/448), S26 GPU 24/25 (20/448) vs the reference 20/448; the differences are one surname's spelling, punctuation on two clips and kana on the Korean sample. FLEURS test 50 clips per language (Mac CPU): en WER 5.48 % (reference 5.13 %), zh CER 6.86 % (6.86 %), ja CER 6.81 % (6.95 %) (Hub card Correctness; FINDINGS rounds 2-3).
- Encoder quantization: fp16 weights are transcript-lossless (25/25 with an fp32 LM); int8 dynamic weights on the encoder's linear layers changed 6 of 25 transcripts and are not used; the fbank DFT/mel constant matmuls must stay fp32 in any FC recipe; ai-edge-quantizer's fp16 needs algorithm_key FLOAT_CASTING or the file stays fp32 unchanged. An int4 (blockwise) LM matched the reference on 14/25 and is not shipped (Hub card Conversion notes; FINDINGS round 2).
- Prompt contract: audio only -> the bundle template supplies 语音转写：; the model's other instructions (no inverse text normalization 语音转写，不进行文本规整：, a language 语音转写成英文：, the hotword prompt) go as a text item in the same message and are placed before the audio; the model has no marker tokens around the audio and the <|AUDIO|> placeholder is consumed by the runtime regex (Hub card How to send audio; FINDINGS round 1 premises).
- Send 16 kHz mono PCM WAV: the runtime's own MP3 decoding (miniaudio) changed the text on 3 of 5 official MP3 samples vs the same audio decoded to WAV by ffmpeg; exact digital silence at the very end of a clip is treated as padding because the runtime passes no clip length (transcripts unchanged on the test clips); no timestamps or CTC output (the checkpoint carries no CTC decoder weights) (Hub card Notes).
- Speed: Mac (litert-lm 0.17.1 benchmark, contended machine, load 5.8-16.0) GPU 3,091 tok/s prefill at p=256 (6,170 at p=512, the unpadded 512 signature) / 143.0 decode / TTFT 0.10 s, CPU 1,027 (1,350) / 48.9 / 0.48 s; end to end 0.37 s per clip with the LM on the GPU (RTF 0.046), 0.68 s on the CPU (0.091). Galaxy S26 (v0.16.1 CLI, uncapped, SKIN < 40 C) GPU (OpenCL) 386 / 26.3 tok/s / 0.70 s, CPU 331 / 31.6 / 0.80 s, peak 2,233 / 2,411 MB; per clip with a loaded engine (Kotlin app, LiteRT-LM Android 0.17.1, CPU) 1.03 / 1.24 / 1.42 s for the 5.6 / 7.2 / 7.2 s samples, engine load 0.64 s with the runtime cache. The earlier derived per-clip figure (process wall minus a load-only run) was retracted the same day (Hub card Performance; FINDINGS round 4 + CORRECTION).
- Reference-side traps (FINDINGS round 1): funasr's dither is only settable at construction (frontend_conf={'dither': 0.0}; an attribute assignment is reverted per generate), ncpu must be passed (torch.set_num_threads is overridden per generate), the first generate call embeds the prompt with the bf16 LLM (one discarded warm-up call), the default decode is greedy because from_config never reads generation_config.json; LayerNorm eps is 1e-5 in the encoder but 1e-12 in the adaptor (vLLM's copy uses 1e-12 for both).
- License Apache-2.0 declared by the base model's card metadata (the official repo ships no LICENSE file; the LICENSE here is the Apache-2.0 text from FunAudioLLM/Fun-ASR-Nano-2512-vllm); changes from the original: weights converted to LiteRT flatbuffers, LM int8 and encoder fp16, front end + encoder + adaptor exported as one graph, prompt format stored as metadata, no CTC decoder (Hub card License and changes).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
