---
family: vibevoice
license: mit
model_id: vibevoice-asr-bitnet
source_url: https://huggingface.co/litert-community/VibeVoice-ASR-BitNet
task: automatic-speech-recognition
---

# vibevoice-asr-bitnet

| | |
|---|---|
| **Task** | automatic-speech-recognition |
| **Family** | vibevoice |
| **Source** | https://huggingface.co/litert-community/VibeVoice-ASR-BitNet |
| **License** | mit |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (LM export through scripts/export_simple_template.py) + litert-torch audio-encoder export + litert-lm-builder bundle assembly (repro = hf-to-litertlm vibevoice_asr_work/: load_bitnet.py -> export_audio_encoder.py -> export_simple_template.py -> patch_tokenizer.py -> build_bundle.py) litert-torch 0.9.3 (LM export, ~/venvs/ltconv040dev) / 0.9.4 (load, encoder, bundle, eager; ~/venvs/lt094dev with torch 2.13.0, transformers 5.14.1 native vibevoice_asr, ai-edge-quantizer 0.9.0, litert-lm-builder 0.16.1); runtime gate litert-lm-api 0.16.1 (HF card Conversion notes; FINDINGS Env) |
| **Command** | `load_bitnet.py (legacy -> native rename, drop the VAE decoder, ternarize the 7 LM projections per tensor exactly as VibeASR.cpp's convert_lm_to_gguf.py, strict load, save hf_native/ + lm_native/) -> export_audio_encoder.py (one signature: audio f32 [1,T,3200] -> features f32 [1,T,1536], T=225 = 30 s, acoustic + semantic encoders + projector + the -25 dBFS normaliser in-graph, wi8fc) -> EXTERNALIZE_EMBEDDER=1 CACHE=2048 PREFILL=512,128,32 export_simple_template.py lm_native … chatml_simple.jinja BMIX4_128 -> patch_tokenizer.py (speech markers as specials on 151646/7/8) -> build_bundle.py (GenericModel audio metadata, skip_mel_spectrogram_extraction, 24 kHz, frame=hop=3200; sections EMBEDDER, PREFILL_DECODE, AUDIO_ENCODER_HW, HF tokenizer) (FINDINGS Recipe)` |
| **Quantization** | LM: per-tensor ternary weights (VibeASR.cpp absmean, s = 1/mean|w|, applied at conversion; alpha 0.0297-0.0695, 36.7% zeros) stored as int4 blockwise-128 min-max — exact for a per-tensor ternary since every block is {-a,0,+a} — with int8 embedding; audio encoder int8 dynamic on FULLY_CONNECTED (wi8fc, WER-lossless), convs fp32 (HF card; FINDINGS Recipe) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `VibeVoice-ASR-BitNet.litertlm` | `5ca907b0343d3e6bd9ec3dbf8aecbcc99b733633ef007a78a7e0f3502010af1b` | 1891.155 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-09-04/vibevoice-asr-bitnet__mac-studio-m4-max.json`, `data/device_runs/0.16.1/2026-09-04/vibevoice-asr-bitnet__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 264 | - | 163.01 | 35.43 | 1650.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.1 · Android 16 | 2026-09-04 | measured |
| galaxy-s26 | gpu | pass | yes | - | 264 | - | 652.54 | 36.12 | 430.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.1 · Android 16 | 2026-09-04 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 256 | - | 357.33 | 59.29 | 807.9 | - | Mac Studio (M4 Max) · litert-lm 0.16.0 | 2026-09-04 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 256 | - | 2044.72 | 138.57 | 133.8 | - | Mac Studio (M4 Max) · litert-lm 0.16.0 | 2026-09-04 | measured |

## Pitfalls

- The audio encoder must run on the CPU (audio_backend); the LM may run on CPU or GPU. On Metal/WebGPU the 30 s window exceeds the delegate's 65,535-workgroup dispatch limit ('Dispatch workgroup count X (90001) exceeds max compute workgroups per dimension'; the 20 s window still dispatches 240000, only <= ~5.5 s would fit), and on Adreno the fp16 delegate path returns an empty transcript with all 1857 nodes delegated and no error — the v3 fp16-safe RMS block did not cure it, so the remaining fp16 hazard is inside the conv stack (HF card Usage + Conversion notes; FINDINGS Gates + encoder history).
- WER (20 LibriSpeech dev-clean clips, 448 words, greedy, vendor prompt): LiteRT-LM 0.16.1 Mac CPU 2.68% = the fp32 eager reference with identical transcripts; LM on Metal 2.46%; Galaxy S26 CPU 3.12% (one clip flips 'On the' -> 'Under', an int8 dynamic-range numerics flip on Arm — a Pixel 8a produced the identical transcript); S26 GPU-LM 2.46% measured on the preceding build (identical LM sections; the re-run on the shipped build agreed on its first two clips before the phone dropped off USB). Microsoft reports 2.41% on the full test-clean set with VibeASR.cpp (HF card Correctness; FINDINGS Gates).
- Ship the 30 s window: runtime-style 10 s chunking (context reset at chunk boundaries) costs 7 words on 6 clips (4.24% vs 2.68%); sending the clip without the duration sentence works at 3.57% (the bundle's template then sends 'Please transcribe it.') — include the duration for best results (HF card; FINDINGS Gates).
- Traps that each cost real time (FINDINGS): a random lm_head shipped silently (VibeVoiceAsrForConditionalGeneration.tie_weights keys off the top-level config, so popping the tied head left it at init and every transcript was token soup — load the head explicitly and assert equal(lm_head, embed_tokens)); raw latent weights are garbage without the per-tensor ternarization (the quantization IS the model); the HF class adds vae_std*randn to the acoustic latents unconditionally (set vae_std=0, export the mean, as VibeASR.cpp does); the tokenizer has no string for the speech-marker ids (151646-151648) so they are added as specials or the generic path spells them out as BPE pieces; the runtime's AudioStaticEncoder requires an output literally named 'features'; rank-0 tensors break the GPU delegate (keep every reduction keepdim=True).
- Runtime contract verified in LiteRT-LM source (v0.16.1 tag): without an audio adapter the encoder is treated as a streaming encoder with window = T frames, audio_buffering_enabled defaults false (short clips zero-padded, valid_tokens = ceil(valid_frames / shrink)), longer clips chunked per window with context reset at boundaries; input name must be one of audio|src_inputs|segment_values; the miniaudio preprocessor resamples to audio_sample_rate_hz and, with skip-mel, frames raw PCM without input_scale (FINDINGS Traps).
- Prompt: the vendor system prompt is baked into the bundle template; the user turn renders as <|object_ref_start|> + audio embeddings + <|object_ref_end|> + newline + text; the vendor layout has no generation prompt and appending <|im_start|>assistant is measured harmless (identical transcripts) (HF card Conversion notes; FINDINGS Gates).
- Speed (LM only; the audio encoder is a fixed 1.3-1.7 s per 30 s window on the M4 Max CPU at 8 threads): Mac litert-lm benchmark 0.16.0 CPU 357 / 59.3 tok/s, GPU (Metal) 2045 / 138.6 tok/s; end-to-end RTF 0.24 CPU / 0.19 LM-on-Metal; Galaxy S26 (litert_lm_advanced_main built from the v0.16.1 tag, 264-token text prompt, one reading per cell) CPU 163 / 35.4, GPU (OpenCL) 652 / 36.1 tok/s; on the phone a 5-20 s clip transcribes in ~7-10 s wall from a cold process incl. engine load, peak private footprint 3.2 GB CPU / 2.3 GB LM-on-GPU. The Mac bench ran on out/bundle30; the LM sections are byte-identical across the encoder builds v1-v3 (FINDINGS), so the device-run records carry no artifact digest for the shipped bytes (HF card Performance; FINDINGS Speed).
- Not done at ship: the Pixel 8a re-gate on the final build and the S26 GPU-LM re-gate (both phones dropped off USB); audio encoder on GPU is parked (CPU-only) (HF card Correctness footnote; FINDINGS SHIPPED).
- License MIT inherited from microsoft/VibeVoice (LICENSE file included); changes from the original: LM ternarized and converted from fp32 safetensors, the VAE decoder (synthesis half) not included, three special tokens added, prompt template and audio-preprocessing parameters embedded as LiteRT-LM metadata (HF card License and changes).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
