---
family: minicpm5
license: apache-2.0
model_id: minicpm5-2b-int4
source_url: https://huggingface.co/mlboydaisuke/MiniCPM5-2B-LiteRT
task: text-generation
---

# minicpm5-2b-int4

| | |
|---|---|
| **Task** | text-generation |
| **Family** | minicpm5 |
| **Source** | https://huggingface.co/mlboydaisuke/MiniCPM5-2B-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch export on the jinja path with the checkpoint's chat_template.jinja byte-identical (released stack only; repro = hf-to-litertlm `bash scripts/reproduce_llm.sh minicpm5-2b`; lane script minicpm52b_work/convert_minicpm5_2b.sh -> scripts/export_simple_template.py) litert-torch 0.9.3 / litert-converter 0.4.0 / ai-edge-quantizer 0.9.0 / litert-lm-builder 0.16.1 / transformers 5.14.1 (HF card Conversion notes; FINDINGS toolchain ~/venvs/lt093ctl, the granite-4.2-3b rail); card-grade bench and toggle probes on litert-lm 0.17.0 |
| **Command** | `convert_minicpm5_2b.sh (USE_JINJA=1 with the verbatim chat_template.jinja, CACHE=4096, PREFILL=1024,256,64,16,4,1, EXTERNALIZE_EMBEDDER=1, HF_HUB_OFFLINE=1, recipe BOCTAV4 | dynamic_wi8_afp32) -> int4: minicpm52b_work/fix_zero_scales_inplace.py (in-place scale-word patch, keeps the embedder section) ; int8: scripts/set_activation_type.py <int8> <out> --type fp32 (litert-lm 0.17.0 unpack/pack, only model.toml gains prefer_activation_type) (FINDINGS Rail; card Conversion notes)` |
| **Quantization** | blockwise-32 int4 + OCTAV on linears, int8 embedding (externalized embedder section); 1,664 all-zero-block scales (13 dead layer-0 MLP rows, gate_proj + up_proj, 832 blocks each) set to the tensor's smallest nonzero scale in place — 3,328 bytes of the 1.55 GB file, dequantized weights unchanged; fp16-default GPU activations |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `MiniCPM5-2B_int4.litertlm` | `9858563beafbc6d5e0d25fcee3827541515296a9302ed3d088b16a58d4fbe7b8` | 1481.695 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.15.0/2026-09-08/minicpm5-2b-int4__iphone-17-pro.json`, `data/device_runs/0.16.0/2026-09-08/minicpm5-2b-int4__galaxy-s26.json`, `data/device_runs/0.17.0/2026-09-08/minicpm5-2b-int4__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 203 | - | 38.99 | 15.56 | 5270.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-08 | measured |
| galaxy-s26 | gpu | pass | yes | - | 203 | - | 400.74 | 18.56 | 560.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-08 | measured |
| iphone-17-pro | cpu | pass | - | - | - | - | - | - | - | - | iPhone 17 Pro · A19-Pro · litert-lm 0.15.0 · iOS 27.0 | 2026-09-08 | measured |
| iphone-17-pro | gpu | pass | - | - | - | - | - | - | - | - | iPhone 17 Pro · A19-Pro · litert-lm 0.15.0 · iOS 27.0 | 2026-09-08 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 256 | - | 148.58 | 31.12 | 1755.1 | - | Mac Studio (M4 Max) · litert-lm 0.17.0 | 2026-09-08 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 256 | - | 1699.15 | 92.79 | 161.6 | - | Mac Studio (M4 Max) · litert-lm 0.17.0 | 2026-09-08 | measured |

## Pitfalls

- int4 needed a zero-scale fix: the raw BOCTAV4 export refuses to load on CPU (XNNPACK: 'unsupported scale value (0.000000) in channel 15616 for INT4 tensor 372' -> Failed to allocate tensors) while the GPU delegate accepts it — decoder layer 0's MLP has 13 entire dead neurons (both gate_proj and up_proj zero on the same rows), nothing anywhere else; the epsilon patch is applied in place at the scale words' absolute offsets because the bonsai repack tool would have dropped the externalized embedder section. Pipeline order for this family is export -> zero-scale fix -> gate, as for LFM2.5-2.6B (HF card Conversion notes; FINDINGS zero-scale wall).
- int4 keeps the runtime's fp16 GPU default: it passes the 8Q gate there (8/8 CPU and GPU), and declaring fp32 only reproduces the CPU fp32 reference's non-terminating thinking chains (n=30 thinking-on: fp16 finished 22/30, fp32act 0/30) — 'declare fp32' is not a blanket fix (HF card; FINDINGS thinking-ON A/B + attribution).
- int4 is the phone file (smaller, fastest GPU decode on every platform measured: Mac Metal 92.8 tok/s, S26 OpenCL 16-19 tok/s); iPhone 17 Pro G41DeviceTest 7/8 on Metal and CPU (the rhyme miss is the composite-prompt artifact bf16 thinking-off shares) (HF card; FINDINGS iPhone).
- The one thinking-OFF data point to carry: int4 in no-think mode answered 17+25 = '52' on the toggle probe where bf16-off says 42 — the no-think mode has no reasoning to absorb int4 noise; the GSM8K thinking-off row (86-88 vs 92) is the quantified version (FINDINGS Mac gates on the fixed int4).
- Requires litert-lm >= 0.16 (thought channel + ThinkingConfig); measured on 0.17.0. The bundle carries the model's own chat_template.jinja verbatim (9060 chars, byte-compared) on the runtime's jinja path — the litert-community/MiniCPM5-1B packaging — so enable_thinking and the tool-calling format stay reachable; the thought channel (<think>/</think>) is auto-declared by the jinja path and was read out of the built bundle, never assumed (HF card; FINDINGS Rail + int4 header).
- Start token <s> is correct for this family: the template's own {{ bos_token }} renders empty at runtime and the engine prepends the metadata start token, so the model sees exactly one <s> as upstream (verified on the official 1B artifact 2026-08-01); stops are </s> (1) and <|im_end|> (130073). The 14 'X<|im_end|>\n' string stops in the metadata are emitted by litert-torch 0.9.3's own export_hf builder (the tokenizer has no such pieces) — a converter artifact, harmless (FINDINGS premises + int4 header).
- Thinking-OFF baseline: the bf16 PyTorch model scores 8/8 on the 8-question gate with thinking on or unset but 6/8 with thinking forced off ('opposite of hot' -> 'Cool.', rhyme -> 'Green.') — read any thinking-off result against 6/8, not 8/8; both toggle paths (ThinkingConfig and the 0.14-era extra_context) reach the verbatim template on 0.17.0 (HF card Correctness; FINDINGS bf16 oracle + toggle probe).
- GSM8K thinking OFF (the official 1B card protocol; first 100 test questions, greedy, 0-shot CoT prompt, max 2048): bf16 92 / int8 CPU 91 / int4 CPU 86 / int4 GPU fp16 87 / int4 GPU fp32act 88 — int8 at parity (7 of its 9 losses are bf16's own), int4 ~5 points down, and the GPU's fp16 activations cost nothing measurable in no-think mode (HF card; FINDINGS).
- In thinking mode int4's chains are ~4.5x longer than bf16's (median ~13.7k vs ~3k chars on a 10-question subset) and close 0/10 within 3584 tokens on the fp32 CPU reference, while int8 on CPU reproduces bf16 question for question (9/10 finished, same cap hit) — the thinking-length inflation is int4 quantization damage; int4 for direct answers / short thinking, int8 when the reasoning has to complete (HF card; FINDINGS Thinking-ON attribution).
- Galaxy S26 (litert_lm_advanced_main v0.16.0 kit): both files generate correctly on GPU and CPU with full OpenCL delegation (1873/1873 nodes on every prefill signature and decode, zero rejected ops) and the runtime separates the reasoning on-device ([thought] … [/thought], then the answer); on Adreno the int4 GPU decode edge over the same-device CPU is ~1.1x (prefill 5-10x), and the int8 file's fp32 activations bring its GPU decode level with its CPU (11-13 vs 11.7 tok/s) at ~2.6x lower prefill than the int4 file's (HF card Performance; FINDINGS S26 sections).
- Card-grade Mac bench (litert-lm 0.17.0 CLI, -p 256 -d 256 --runs 3 --cache no --max-num-tokens 1024, quiet machine, CPU cells first, >= 300 s rest before every GPU cell, gate_backend.py confirmed real generation on every cell); the earlier gate-time decode figures were taken under load and are not speed claims (FINDINGS Card-grade Mac bench + int8 gates note).
- Block-128 int4 was built and rejected: 7/8 on the CPU gate (the 'merci' question never closes <think>), GSM8K thinking-off 79 vs 87, and the thinking-length problem does not move (12.9k vs 13.7k chars); the 'reasoning ships prefer block128' heuristic does not apply here, as on granite-4.2 (HF card Conversion notes; FINDINGS block-128 A/B).
- Embedder externalised (130560x2048 table in its own section); tokenizer embedded as the upstream tokenizer.json (byte-level BPE, 130,072 base + 510 added tokens; <think>=8 / </think>=9 plain added tokens, /think and /no_think special) (HF card; FINDINGS premises).
- Local bundles, raw exports, compile caches and the HF source cache were deleted at close-out on 2026-09-08 after re-reading the remote sizes; regeneration = the reproduce script (int4 comes out zero-scale-fixed; the int8 ship additionally needs the fp32 activation repack) or a Hub download; two ~4 GB xnnpack caches remain on the S26 (FINDINGS Close-out).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
