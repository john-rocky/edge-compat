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

2026-09-25 refresh (own uploads since 2026-08-26, HF API + commit authors = mlboydaisuke only): 9 rows added —
Nemotron-3-Diarization-LiteRT (frontend 2 MB, encoder low_latency 199 MB, encoder offline 199 MB; graph B's
attn_bias / rope_cos / rope_sin are host-computed tables, so the sweep's random inputs measure the graph, not a
diarization), DAC-16kHz-LiteRT (encoder 43 MB, deconly_zs 105 MB), PP-OCRv6-Small-LiteRT (det 640 + rec 320/640/960,
12-21 MB). Not added: Depth-Anything-3-Small-LiteRT (its da3_small_gpu_fp16.tflite is byte-identical to the
catalogued depth-anything-3-small row — same LFS sha256 e170369a…, a renamed repo); the kitten-tts-nano-0.8 and
Inflect-Nano-v2 non-static siblings (dynamic input dims, same exclusion as 2026-08-2x); oversized own uploads over
the ~500 MB ceiling: Bonsai-Image-ternary-4B (1.8-3.1 GB each), Nemotron-3-Embed-1B fp16 (2.3 GB; its wi8fc
variants were already catalogued/excluded earlier), FLUX.2-klein kc_double0 / kc_single0..1 (0.6-0.74 GB) and
Z-Image qwen_enc (3.5 GB) stay in the comment block above. Downloads via `hf download` (huggingface_hub); the
first `hf download` of a 105 MB file died with IncompleteRead and succeeded on retry.

2026-09-25 (later the same run): the catalog diff above had missed six own repos on a parsing slip (long
artifact lists); added the sweepable ones — GLiNER2.5-Small-LiteRT (s128/s256/s512 x fp32/wfp16, 54-128 MB),
Laya-Multilingual-LiteRT (act head + s256/s512 embeds wfp16, 250 MB each), Laya-English-LiteRT and laya-LiteRT
act heads (1 MB). Over the ceiling, not added: the Laya encoder fp32/wfp16 graphs (0.6-1.7 GB) and all four
parakeet-tdt_ctc-0.6b-ja files (0.6-2.4 GB). sopro-v2-turbo: all 45 graphs (2-223 MB each, 3.4 GB) downloaded on retry (7 of 45 needed a second `hf download`)
and catalogued (all static; int32 inputs are token ids / position tables, so the random sweep inputs measure the graph,
not speech).
sopro sweep note (2026-09-25): the single-process run lost its browser after the fourth model and mislabeled the rest
as compile_error — those files were discarded and the 45 graphs re-run through `npm run sweep:batches --batch 2`
(fresh browser per two models). Four graphs crash the page even alone on both backends and are kept as crash
results (the harness contract): acoustic_condition_t4096_r6 fp32 / wfp16 (a 1x1024x4375 input + a 4096x1024 table)
and ar_step fp32 / wfp16 (two 1x96x1024x64 KV-cache inputs) — input volume, same class as the wasm memory ceiling.
Two more report a LiteRT.js compile error on WebGPU (recorded verbatim). 27 of the 39 WebGPU runs report
output_match=false: random int32 token / position inputs and the r6 variants' large tables make the wasm reference
comparison meaningless for these graphs, so treat the sopro rows as load/run/latency evidence only.

2026-09-26: GLiNER2.5-Decide-LiteRT (own upload, all three Hub commits by mlboydaisuke; license apache-2.0 from the HF API
tag) — the three default wfp16 graphs added as rows (s128 660 MB, s256 711 MB, s512 811 MB; input_spec parsed with
litert_compat.parser.reader from sha256-verified downloads, static shapes, float32 only; resolve URLs verified by HEAD,
LFS sha256 = x-linked-etag). They sit above the ~500 MB catalogue rule on purpose: s128 is the first datapoint inside the
431-712 MB band where the wasm memory ceiling was unmeasured, and the row notes say where each file stands against the
712 MB datapoint. The fp32/ reference graphs (1.27-1.42 GB) are not catalogued (over the 2 GiB-class in-page fetch
limit's neighbourhood and duplicates of the wfp16 graphs' arithmetic). Downloads via huggingface_hub.hf_hub_download
(XET off; 91-100 s per file this time) hardlinked into ~/.cache/litert-models under sha256(url).tflite.
Sweep 2.5.3/2026-09-26 of those three rows (mac-studio-m4-max, default flags, one harness process per model): s128 and
s256 load on wasm_xnnpack but their 3 + 10 runs do not finish inside the default 60 s step budget (`timeout`, i.e. at
least 4.6 s per run on average; deliberately not re-run with a larger timeout — the record cannot state a non-default
timeout and the weekly CI would flip it back, DECISIONS #173); s512 loads and every run logs `Failed to allocate
tensors` (`backend_error` — after the harness fix that stops counting such 0.35 ms non-runs as passes, README "Traps");
webgpu_mldrift aborts at load (`RuntimeError: Aborted()`) for all three, a generic error naming no op, so no matrix
evidence. With the older 431 MB (runs) and 712 MB (wasm_memory_ceiling) datapoints the wasm picture is: 660-711 MB
loads and computes but slower than the default budget; 811 MB fails tensor allocation.

2026-09-26 (later the same day): GLiFormer-Large-NER-LiteRT (own upload, Hub revision dee6ced; license apache-2.0 from the HF API
tag) — the five default wfp16 graphs added as rows (s128 full 707 MB, s256 encoder 707 MB, s256 head 53 MB, s512 encoder 807 MB,
s512 head 56 MB; input_spec parsed with litert_compat.parser.reader from the byte-identical local copy of the published tree,
sha256 = manifest.json = the resolve URLs' x-linked-etag (HEAD-verified), static shapes, float32 only, subgraph buffer order which
is not the flatbuffer signature-map order). The two encoders and the full graph sit above the ~500 MB catalogue rule on purpose
(same reasoning as the GLiNER2.5-Decide rows: they bracket the wasm ceiling). The fp32/ reference graphs (1.31-1.41 GB full/encoder,
104-107 MB heads) are not catalogued (over the in-page fetch limit's neighbourhood, or duplicates of the wfp16 graphs' arithmetic).
The local files were hardlinked into ~/.cache/litert-models under sha256(url).tflite, so nothing was downloaded.
Sweep 2.5.3/2026-09-26 of those five rows (mac-studio-m4-max, default flags, one harness process per model via `--only`, four
minutes in all): the 707 MB s128 full graph and s256 encoder load on wasm_xnnpack but their 3 + 10 runs do not finish inside the
default 60 s step budget (`timeout`, i.e. at least 4.6 s per run on average — the GLiNER2.5-Decide s128/s256 outcome again, deliberately
not re-run with a larger timeout, DECISIONS #173); the 807 MB s512 encoder loads and every run logs `Failed to allocate tensors`
(`backend_error`, as the 811 MB GLiNER2.5-Decide s512 did); the 53 MB s256 head (12,568 operators, an unrolled T=256 BiLSTM) runs on
wasm at 2,425 ms p50 (output_match null: wasm is the reference backend); the 56 MB s512 head (25,112 operators) loads but its runs do
not finish inside the 60 s budget (`timeout`) — the first sub-100 MB timeout in this catalog, operator count rather than bytes.
webgpu_mldrift aborts at load (`RuntimeError: Aborted()`) for all five, including the two small heads, a generic error naming no
op, so no matrix evidence; the WebGPU failure is therefore not a size effect for this family.

2026-09-28: Audio8-TTS-Preview-0.6b (own upload, both Hub commits by mlboydaisuke, rev e8620a9b; license apache-2.0 from the HF API tag; 7 classic .tflite
graphs of a DualAR TTS + a Python host loop) — five graphs added as rows (fast_ar_int8 68 MB, codec_decoder_fp16_T128 / T192 262 MB, codec_decoder_int8_T128
132 MB, codec_encoder_fp16_10s 419 MB; input_spec parsed with litert_compat.parser.reader from the local out/ship copies whose sha256 equals the HF LFS sha256
(API ?blobs=true and the resolve URLs' x-linked-etag, HEAD 200), static shapes, float32/int32 only, subgraph input order). The int32 inputs are codebook indices
(0..15 from the dimension spec are valid codes; every gather is clamped in-graph) or token/position ids, so the random inputs measure the graphs, not speech.
Excluded, recorded in the catalog comment: slow_ar_int8 (552 MB) and slow_ar_int4 (386 MB) carry two signatures (prefill_256, decode; 51 inputs each incl. 48
KV-cache tensors [1,2,2048,64]) and the harness has no signature selection, so a run would measure an unstated signature; slow_ar_int8 is also over the ~500 MB
rule. The local files were hardlinked into ~/.cache/litert-models under sha256(url).tflite (no download).
Sweep 2.5.3/2026-09-28 of the five rows (mac-studio-m4-max, default flags, one harness process per model via `--only`, load avg 3.2-4.3, 1 min 11 s in all):
fast_ar_int8 runs on both backends (wasm 2.48 ms p50; WebGPU 3.42 ms partially delegated, 58 of 283 ops — ADD / RESHAPE rank 5, GATHER_ND, SLICE rank > 4
refused — with output_match false, max_abs_diff 0.889: a DRQ-int8 graph against the float GPU path, README caveat 2, not a numerics verdict). The three codec
decoders and the encoder produce no latency: codec_decoder_fp16_T128 / T192 and the encoder log `Failed to allocate tensors` (T128/T192) or `XNNPack delegate
failed to reshape runtime` + `Node number 1600 (TfLiteXNNPackDelegate) failed to prepare` (encoder) on wasm_xnnpack (backend_error; 262-419 MB fp16 files whose
weights unpack to fp32 — below the 431 MB "runs" datapoint by file size, so bytes alone do not explain it); codec_decoder_int8_T128 loads on wasm but its 3 + 10
runs exceed the 60 s budget (timeout, >= 4.6 s per run; not re-run with a larger timeout, DECISIONS #173). WebGPU: the fp16 T128 decoder fails compile after the
DEQUANTIZE / EMBEDDING_LOOKUP refusals (compile_error, litert_compiled_model_next.h:70), the int8 decoder fails ML Drift kernel init ("Unable to parse bc coord
for BATCH axis" on convolution1x1, compile_error), T192 and the encoder abort with `RuntimeError: memory access out of bounds` (wasm_memory_ceiling). The op-naming
lines are staged in data/matrix_staging/webgpu_mldrift__2.5.3-audio8-additions.csv; the rest stays model-level in the sweep records.
