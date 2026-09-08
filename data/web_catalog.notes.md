# web_catalog notes (exclusions and scope)

Catalog scope: owner's own uploads (zoo README ids x litert-community org),
.tflite artifacts only — enumerated via the HF API on 2026-08-11,
refreshed 2026-08-12 (+32 rows, 81 -> 113) and 2026-08-13
(+2 rows, 113 -> 115: LFM2.5-ColBERT-350M, published after the 08-12 refresh).
LLM .litertlm bundles are NOT browser-sweepable (LiteRT.js runs .tflite only);
the LLM lane's data lives in data/device_runs/ (対応表) instead.

On the 2026-08-12 refresh, "own upload" was settled from a primary source
rather than from the zoo README (which was not resolvable locally): every
candidate repo's full commit history on the Hub is authored solely by
mlboydaisuke. The 17 repos touched since the first enumeration were checked
that way; Matcha-TTS was already catalogued, and the two .litertlm repos
(LFM2.5-2.6B, granite-4.0-h-350m) belong to the LLM lane.

2026-08-13 additions, same split: three new .litertlm repos published this day
(LFM2.5-VL-3B, LFM2.5-VL-1.6B, LFM2.5-VL-450M — the LiteRT-LM VLM family) are
LLM-lane only, not browser-sweepable; their device-run records are in
`data/device_runs/0.16.0/2026-08-13/` (Pixel 8a console logs, int4/int8 ×
cpu/gpu each). No new .tflite artifacts were published, so the sweep catalog
is unchanged at 115 rows + header.

input_spec is parsed from subgraph 0, which is what the harness's positional
`model.run()` call reaches. The LFM2.5 encoder/embedding artifacts are
multi-signature files (4 subgraphs, embed_512 first); they are catalogued
rather than pre-excluded, because whether LiteRT.js can drive them is a
question the sweep answers with a real verdict instead of an assumption.

That question is now answered — probe sweep, @litertjs/core 2.5.3,
mac-studio-m4-max, 2026-08-13 (scratch run, not a committed snapshot):

- `LFM2.5-Embedding-350M_wi8fc` (371 MB, 4 subgraphs) **loads and runs on
  wasm_xnnpack**, p50 4570 ms. LiteRT.js does drive a multi-signature file
  through the positional `model.run()` path. Cataloguing them was right.
- The same file **fails to compile on webgpu_mldrift**. The error is generic
  (`litert_compiled_model_next.h:70`) and names no op, so it yields no
  op-level matrix evidence — do not mine it for a matrix row.
- `LFM2.5-Embedding-350M_fp16` (712 MB) fails on both: `wasm_memory_ceiling`
  on wasm, `compile_error` on webgpu.

Browser size ceiling, bracketed by measurements already in this repo plus the
above: 431 MB (`gfpgan_fp16`) loads and runs; 712 MB and 882 MB
(`qwen3-embedding-0.6b`, `qwen3-reranker-0.6b`) hit `wasm_memory_ceiling`. The
oversized rows are kept on purpose — a recorded ceiling is the compat data this
repo exists to produce, not a catalog defect to prune.

CI note: the full catalog is now ~19.5 GB of artifacts, against GitHub Actions'
10 GB per-repo cache limit, so `restore model download cache` in resweep.yml
cannot hold the whole set and the weekly full tier will re-fetch a large part of
it. Left as-is rather than trimming the catalog to fit a CI limit; if the job
starts failing on disk or duration, bound the sweep rather than delete rows.

Excluded rows (honest exclusions, not failures):
Kokoro-82M__kokoro_82m_fixedlen_fp32 | https://huggingface.co/litert-community/Kokoro-82M/resolve/main/kokoro_82m_fixedlen_fp32.tflite | apache-2.0 | unsweepable inputs: input dtype int64 on 'serving_default_args_0' (LiteRT.js tensor I/O is float32/int32)
Kokoro-82M__kokoro_predictor | https://huggingface.co/litert-community/Kokoro-82M/resolve/main/kokoro_predictor.tflite | apache-2.0 | unsweepable inputs: input dtype int64 on 'serving_default_args_0' (LiteRT.js tensor I/O is float32/int32)
NIMA-LiteRT__nima_aesthetic_fp16 | https://huggingface.co/litert-community/NIMA-LiteRT/resolve/main/nima_aesthetic_fp16.tflite | apache-2.0 | unsweepable inputs: dynamic/unknown dims [1, 224, 224, 3] (spec 1x224x224x3:float32)
NIMA-LiteRT__nima_technical_fp16 | https://huggingface.co/litert-community/NIMA-LiteRT/resolve/main/nima_technical_fp16.tflite | apache-2.0 | unsweepable inputs: dynamic/unknown dims [1, 224, 224, 3] (spec 1x224x224x3:float32)
Parakeet-tdt-ctc-110m-LiteRT, wav2vec2-base-960h-CTC-LiteRT | zoo README ids no longer in the litert-community org

2026-08-20 refresh (+9 rows, 115 -> 124): five new own-upload .tflite repos
published since the 08-13 refresh, all commit-authored solely by mlboydaisuke
on the Hub and all with fixed-shape float32 inputs (parsed clean, no
exclusions): Zipformer-medium-CR-CTC-LiteRT (3 variants: medium/small/large),
japanese-zipformer-base-LiteRT, wav2vec2-base-960h-LiteRT (frontend + head),
TIPSv2-B14-DPT-LiteRT, RF-DETR-Seg-Nano-LiteRT (graph A + B).

Same window, NOT catalogued: 16 new .litertlm repos (Falcon-H1 x4, LFM2.5-VL
x3, Zamba2 x2, Qwen3.5 x2, Nemotron-H-4B, codegemma-7b, Qwen2.5-Coder-1.5B,
granite-4.1-3b, North-Micro-Vision-Instruct) — LLM lane only, not
browser-sweepable (their device-run data is in data/device_runs/); and
Moboil (commit author amir1334r, not an own upload; carries no .tflite or
.litertlm artifact either).

2026-08-22 refresh (+13 rows, 129 -> 142). Scope unchanged (own uploads,
.tflite only). Enumerated via the HF API: the org holds 324 models, of which
235 carry a .tflite; the owner's shipped-conversions collection
(mlboydaisuke/litert-conversions-shipped-to-litert-community) lists 142 repos,
95 of them with .tflite artifacts. Sixteen of those 95 had no catalog row.
Own-upload status was re-confirmed the 08-12 way — every one of the sixteen
has a Hub commit history authored solely by mlboydaisuke.

Twelve of the sixteen were small enough to fetch and parse in this run
(~580 MB total); they produced 13 sweepable rows and 12 honest exclusions:

- catalogued: SAM2.1-Hiera-Tiny-Image-Encoder (2 variants),
  kitten-tts-nano-0.8 (vocoder_static80 only), Inflect-Nano-v2
  (decoder_static228 only), yolox-nano / -tiny / -s / -m, MiDaS-small,
  real-esrgan-x4v3, U-2-Net, lightweight-openpose (2 variants).
- excluded, dynamic input dims (LiteRT.js needs a static shape): the dynamic
  siblings of kitten-tts (predictor, prosody, vocoder x fp32/fp16),
  Inflect-Nano-v2 (decoder, text_encoder x fp32/fp16) and both NIMA graphs
  ([-1,224,224,3]). Every TTS/inflect repo here ships BOTH a dynamic graph and
  a static one; only the static sibling is sweepable, which is why those two
  repos contribute one row each.

NOT fetched this run — four own-upload .tflite repos totalling 34.6 GB, held
for an owner call because the size is a different order from anything in the
catalog and CI's model cache is already over GitHub's 10 GB limit:
FLUX.2-klein-4B-LiteRT (21 files, 10.6 GB), Z-Image-Turbo-LiteRT (13, 9.8 GB),
Bonsai-Image-ternary-4B (5, 9.7 GB), Nemotron-3-Embed-1B (3, 4.6 GB). The
standing rule in this file is that oversized rows are kept on purpose (a
recorded ceiling is data), so the default answer is probably "catalogue them",
but 34.6 GB of one-time download is worth asking about first.

⚠ Count drift in the older entries above: the running totals in the 08-13 and
08-20 paragraphs (115 -> 124) do not match the committed file, which held 129
data rows before this refresh (119 -> 120 -> 129 across 08caa64, d2b5cad,
d58aaf3). The per-refresh deltas are right; only the cumulative figures drifted.
Counts in this paragraph are measured from the file.

⚠ Correction to the 2026-08-22 commit message, same day. It said the four
webgpu_mldrift output-match failures were "all small-magnitude with relative
error blown up by near-zero reference elements ... tolerance-boundary behaviour
rather than broken delegation". That was inference, not measurement, and it is
only partly right. Two things were wrong with the reasoning: `max_abs_diff` and
`max_rel_diff` are maxima over INDEPENDENT elements (results.ts computes each
separately), so dividing one by the other to recover |ref| is invalid; and the
sweep schema stores no output tensors, so nothing in the committed record could
have supported the claim.

Measured instead by instrumenting compareOutputs locally (reverted after) and
re-running the four models, @litertjs/core 2.5.3, mac-studio-m4-max:

- yolox-nano       54 / 301,665 elements fail. max_abs 4.765e-4 at |ref| 1.398e-1;
                   max_rel 1.676e-1 at |ref| 2.109e-4 (abs there 3.535e-5).
- real-esrgan-x4v3 31 / 786,432 fail. max_abs 1.848e-5 at |ref| 2.252e-1;
                   max_rel 4.768e+2 at |ref| EXACTLY 0 (abs there 4.768e-7).
- yolox-tiny       2 / 301,665 fail. max_abs 6.354e-5 at |ref| 2.659e-1;
                   max_rel 7.542e-3 at |ref| 7.824e-4 (abs there 5.901e-6).
- yolox-m          NOT reproduced: the wasm_xnnpack reference timed out on the
                   re-run, so no comparison was made. Its original FAIL stands
                   as recorded but is unexplained.

So the near-zero story is confirmed for exactly one number — real-esrgan's
476x relative error is a reference element of exactly 0.0 with an absolute
deviation of 4.8e-7. It is NOT the general explanation: yolox-nano's largest
absolute deviation sits at a reference magnitude of 0.14 and genuinely exceeds
`max(1e-5, 1e-3*|ref|)`. What is measured is that the affected fraction is tiny
(0.0007%-0.018% of elements) and the absolute deviations are <= 4.8e-4. The
CAUSE is not established: no mechanism was measured, and no per-op attribution
is possible from a model-level sweep (Phase 6 trap rule). Other models on the
same backend and the same run pass output-match, which is a control against
"webgpu is broken" but says nothing about these four.

2026-08-23 refresh (+15 rows, 142 -> 157). The four large own-upload repos held
back on 08-22 were resolved by splitting them on size rather than fetching or
skipping them whole. Measured ceiling from this file: 431 MB loads and runs,
712 MB and 882 MB hit wasm_memory_ceiling. Files under ~500 MB therefore sit in
the band where the verdict is genuinely unknown; files above it mostly re-confirm
a ceiling already recorded, at a download cost CI cannot absorb (its model cache
is already over GitHub's 10 GB limit).

Catalogued: the 15 files under 500 MB, 2.42 GB total, all parsing clean with
static float32 inputs and zero exclusions —
  Z-Image-Turbo-LiteRT      6 (z_embx, zc_final, z_embc, zvae, z_refc, z_refx)
  FLUX.2-klein-4B-LiteRT    8 (kc_final, kce_final, kv_vae_enc, kv_vae, kc_prep,
                               kce_prep, kc_double1, kce_double1)
  Bonsai-Image-ternary-4B   1 (vae_dec_fp32)

NOT catalogued: 27 files, 32.21 GB, excluded on size with the reason recorded
here rather than dropped silently —
  Z-Image-Turbo-LiteRT      7  9.00 GB  qwen_enc 3.5 GB + zc_main0..5 at 908 MB
  FLUX.2-klein-4B-LiteRT   13  9.14 GB  ke_enc0..2 at 912 MB, kc/kce_double0 at
                                        739 MB, kc/kce_single0..3 at 615 MB
  Bonsai-Image-ternary-4B   4  9.46 GB  textenc_int8_weightonly 3.1 GB,
                                        dit(_gpu)_int4b32 2.27 GB, textenc_int4 1.8 GB
  Nemotron-3-Embed-1B       3  4.61 GB  fp16 2.29 GB, wi8fc 1.17 GB,
                                        wi8fc_128_512 1.15 GB — no file under 500 MB,
                                        so this repo contributes no sweep row at all
Every excluded file is >= 615 MB, i.e. at or above the 712 MB datapoint that
already fails. Revisit if LiteRT.js raises the wasm memory ceiling.

⚠ Transfer note: urllib and single-stream curl both stalled against the Hub on
these files (~1 MB/min at 0% CPU, repeatedly). huggingface_hub.hf_hub_download
moved the same files at ~80 MB/min; the fetch was redone through it and hardlinked
into ~/.cache/litert-models under sha256(url).tflite, the harness's cache key.

2026-08-26: granite-docling-258M (published 2026-08-25, sole commit author
mlboydaisuke, .litertlm only) added to the litertlm comment block — LLM lane,
not browser-sweepable. No new own-upload .tflite repos since the 08-24 refresh
(TIPSv2/DINOv2 were already catalogued).
