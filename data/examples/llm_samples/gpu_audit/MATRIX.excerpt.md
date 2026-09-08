<!--
Excerpt of ~/code/litertlm-convert/reports/gpu_audit/MATRIX.md (file mtime 2026-07-23).
Staged as an adapter-mapping sample only. Verbatim below the rule, except that rows for
models not staged here were dropped and one row carrying a private thread reference was
omitted. The source file is Japanese; this file keeps the original wording so the excerpt
stays usable as evidence. English gloss lives in ../README.md.
-->

---

# GPU 実行監査マトリクス — Mac 速判+Pixel 8a 実機フェーズ(2026-07-22〜23)

道具: `scripts/gpu_gate_mac.sh <model.litertlm> [tag]` — `litert-lm benchmark --backend gpu --cache no -p 256 -d 256`、FAIL 時は非対応 op 名を自動抽出。ログは本ディレクトリ `<tag>.gpu.log`。

**注意1**: Mac WebGPU 合格 ≠ Android ML Drift 合格(別 delegate)。本表は「速い篩」。最終判定は Pixel 8a 実機(Gallery インポート時に Compatible accelerators で GPU をトグル、下記参照)。

**⚠注意2(7/23 訂正 — 測定衛生)**: 7/22 バッチの速度値は並行セッションの GPU コンパイル/連続実行の負荷で**大幅に低く汚染されていた**。7/23 の静音環境の再測で BitCPM は 1521/56 → **2965/167.7**(3倍!)。7/22 に立てた仮説「`--cache no` 半速」「decode は sync 律速でフラット」は**いずれも汚染データ由来で撤回**。教訓 = **Mac GPU ベンチは静音ウィンドウ+直列実行でのみ有効**(Bonsai セッションの matched-pairs 教訓と同一)。PASS/FAIL 判定は負荷に影響されないので全行有効。

## 判定結果(カタログ全数)

7/22 バッチ行の速度は参考値(汚染)。7/23 行はクリーン。

| モデル(出荷物) | 判定 | prefill / decode (tok/s) | 測定日 |
|---|---|---|---|
| **BitCPM-CANN-1B int4(ternary)** | ✅ PASS | **2965 / 167.7**(クリーン)| **7/23** |
| **Gemma3-1B int4(公式)** | ✅ PASS | 4344 / 185.8 | 7/23 |
| LFM2.5-1.2B-Instruct int4 | ❌ FAIL | — | 7/22 |

**LFM FAIL の切り分け(新情報)**: GATHER_ND(ローカルパッチの index_select、既知)に加え、**ShortConv conv 経路に INT64 の ADD / CAST / SUM**(パッチの `valid_len = mask.sum()` 由来、prefill シグネチャ)。→ litert-torch 次リリース(acc9eef の one-hot 化)で GATHER_ND が消えても **int64 演算が残れば GPU は落ちる**。再挑戦時のチェックは2点:(1) GATHER_ND 消滅 (2) int64 → int32/f32 化。ログ: `lfm25-12b-instruct-int4.gpu.log`。
