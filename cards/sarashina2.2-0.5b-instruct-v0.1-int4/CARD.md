---
family: sarashina2.2
license: mit
model_id: sarashina2.2-0.5b-instruct-v0.1-int4
source_url: https://huggingface.co/litert-community/sarashina2.2-0.5b-instruct-v0.1
task: text-generation
---

# sarashina2.2-0.5b-instruct-v0.1-int4

| | |
|---|---|
| **Task** | text-generation |
| **Family** | sarashina2.2 |
| **Source** | https://huggingface.co/litert-community/sarashina2.2-0.5b-instruct-v0.1 |
| **License** | mit |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (released wheels only; repro = hf-to-litertlm sarashina_work/convert_sarashina.py wrapping scripts/export_simple_template.py, family driver with structured prompt_templates) 0.9.3 (transformers 5.14.1; ai-edge-quantizer named but unversioned in the sources — HF card Conversion notes; FINDINGS env ~/venvs/ltconv040dev; Mac gates on the litert-lm 0.16 lineage) |
| **Command** | `convert_sarashina.py <export_dir> <out> templates/sarashina_simple.jinja <recipe> — NO_START_TOKEN=1, HF tokenizer.json embedded (vocab_file cleared so export falls through to save_pretrained legacy_format=False -> HF_Tokenizer_Zlib), CACHE 4096, prefill ladder 1024..1 = 11 signatures; no post-processing (FINDINGS Recipe)` |
| **Quantization** | export-time int4 blockwise-32 + OCTAV on linears, int8 embedding (BOCTAV4); no post-processing |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `sarashina2.2-0.5b-instruct-v0.1_int4.litertlm` | `e5e873fdc7d31e5040e218ecf1af71bf9d8a27779715af30436edb3b4cbc3b5f` | 509.728 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-09-02/sarashina2.2-0.5b-instruct-v0.1-int4__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-02/sarashina2.2-0.5b-instruct-v0.1-int4__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 321 | - | 132.63 | 19.53 | 2470.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-02 | measured |
| galaxy-s26 | gpu | pass | yes | - | 321 | - | 903.16 | 40.3 | 380.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-02 | measured |
| mac-studio-m4-max | cpu | pass | - | - | - | - | 546.31 | 42.73 | 564.9 | - | Mac Studio (M4 Max) · litert-lm 0.16.0 | 2026-09-02 | measured |
| mac-studio-m4-max | gpu | pass | - | - | - | - | 5753.57 | 197.85 | 57.7 | - | Mac Studio (M4 Max) · litert-lm 0.16.0 | 2026-09-02 | measured |

## Pitfalls

- Quality: 8-question gate (Apple M4 Max, litert-lm 0.16 lineage, one process per question, greedy) EN/JA — see HF card Correctness table; JCommonsenseQA (own harness, n=100): bf16 65 / int8 62 / int4 65 — all within paired noise (bf16-only 4 vs int8-only 1 / int4-only 4) (HF card Accuracy; FINDINGS).
- The bundle embeds the HF tokenizer.json, not the vendor tokenizer.model: sarashina's SentencePiece model types every chat special (<|user|> <|assistant|> <|system|> </s>) as a CONTROL piece, which a bare sentencepiece encoder never matches from text (<|user|> becomes 5 pieces, no id 9/8/2); the HF fast tokenizer matches them as added tokens. Engine-vs-HF tokenize parity 7/7 probes on every bundle (HF card Conversion notes; FINDINGS Recipe item 1).
- No start_token is written: add_bos_token is false and the official template emits no <s>, but the exporter would still write start_token from tokenizer.bos_token; measured in bf16 with the same rendered prompt +/- BOS, the 0.5b drops EN 6/8->5/8 and JA 8/8->7/8 with a BOS (the 1b is unchanged), so NO_START_TOKEN=1 (HF card; FINDINGS Recipe item 2).
- Chat template = structured prefix/suffix pairs (<|system|>…</s>, <|user|>…</s>, <|assistant|>…</s>, generation prompt <|assistant|>) byte-identical to what the official jinja renders for plain chat shapes; the tool-calling branch is not carried (HF card; FINDINGS Template).
- int8 is the recommended file: closest to bf16 and, on both the Galaxy S26 and the Mac, not slower than int4 on either backend — the 102,400-entry untied vocab makes embedding + lm_head the dominant per-token cost at this size (262M of ~500M params on the 0.5b); int4 CPU prefill is 2-2.6x slower; int4 is the size option (HF card; FINDINGS S26 + Mac cardbench readings).
- Decode is far below LFM2.5-230M's 122 tok/s on the same S26 GPU for the same reason (untied 102,400 vocab; the 230M has a 65k tied vocab) (HF card Performance; FINDINGS).
- JCommonsenseQA numbers (JGLUE v1.3 valid, first 100, greedy, own harness scored by gold-choice-text match) are comparable only within that table, not to published JGLUE scores (HF card Accuracy; FINDINGS).
- The English 8-question misses are the checkpoint's own: the bf16 PyTorch reference scores 6/8 EN (misses 'merci' and the rhyme) and 8/8 JA; all legs non-degenerate; emoji / rare-kanji streaming probe (😀 𠮷 ☔) shows no U+FFFD on any leg (ByteFallback streams 4 tokens per emoji, so the streaming check matters) (HF card Correctness; FINDINGS Mac gates).
- Galaxy S26 (litert_lm_advanced_main v0.16.0 kit): every GPU leg fully delegated to LITERT_CL (1066/1066 on prefill_2-1024, 947/947 prefill_1, 975/975 decode, zero XNNPack fallback); all gate legs answered the Japanese capital question (HF card; FINDINGS Device gates).
- iPhone 17 Pro rows are not yet taken: the phone was 'unavailable' in devicectl and the BenchmarkApp install was refused with CoreDeviceError 4016 (screen lock) on 2026-09-02 — the card ships with an append-later note (FINDINGS Status).
- Mac cardbench (litert-lm 0.16.0 CLI, -p 256 -d 256 --runs 3 --cache no, quiet machine, 300 s rest before each GPU cell); the per-cell log carries no token-count header, so the device-run cells for the Mac carry no prompt-length condition (mac_cardbench.log; DECISIONS #163(e)).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
