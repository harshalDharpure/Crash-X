# CrashX — Complete Research Results Documentation

**Paper:** *CrashX: Benchmarking Spatiotemporal Reasoning and Mitigating Temporal Hallucinations in Traffic Accident Video-LLMs*

**Status:** All experiments complete (September 2026)

---

## Document Index

| # | File | Contents |
|---|------|----------|
| 0 | [00_INDEX.md](00_INDEX.md) | This navigation page |
| 1 | [01_RESEARCH_OVERVIEW.md](01_RESEARCH_OVERVIEW.md) | Research goals, contributions, pipeline summary |
| 2 | [02_METHODOLOGY.md](02_METHODOLOGY.md) | Model, training, SEASON decoding, evaluation metrics |
| 3 | [03_DATASET_AND_SPLITS.md](03_DATASET_AND_SPLITS.md) | CCD dataset, partition audit, stratification |
| 4 | [04_TABLE_I_CAPTIONING.md](04_TABLE_I_CAPTIONING.md) | Table I — dense caption quality |
| 5 | [05_TABLE_II_TEMPORAL_FORENSIC.md](05_TABLE_II_TEMPORAL_FORENSIC.md) | Table II — temporal & forensic reasoning |
| 6 | [06_TABLE_III_STRUCTURED_FIELDS.md](06_TABLE_III_STRUCTURED_FIELDS.md) | Table III — structured field accuracy |
| 7 | [07_TABLE_IV_SEVERITY.md](07_TABLE_IV_SEVERITY.md) | Table IV & IVb — severity-stratified results |
| 8 | [08_TABLE_V_RELATIVE_GAINS.md](08_TABLE_V_RELATIVE_GAINS.md) | Table V — relative gains over zero-shot |
| 9 | [09_TABLE_VI_ABLATION.md](09_TABLE_VI_ABLATION.md) | Table VI — SEASON ablation study |
| 10 | [10_TABLE_VII_NLI.md](10_TABLE_VII_NLI.md) | Table VII — NLI faithfulness & hallucination |
| 11 | [11_FULL_METRICS_MASTER.md](11_FULL_METRICS_MASTER.md) | All conditions × all metrics (detailed) |
| 12 | [12_KEY_FINDINGS.md](12_KEY_FINDINGS.md) | Best results, paper narrative, reviewer talking points |

---

## Quick Summary (150 test videos)

| Method | BLEU-4 | tIoU | ArgusCost-O ↓ | NLI-Score ↑ |
|--------|--------|------|---------------|-------------|
| Zero-shot Qwen2.5-VL-7B | 0.016 | 0.012 | 0.462 | 0.047 |
| **CrashLogic-7B (Greedy)** | **0.142** | 0.373 | **0.107** | -0.005 |
| CrashLogic-7B + SEASON (α=1.0) | 0.140 | 0.375 | 0.112 | **0.090** |
| **Best ablation: SEASON shuffle** | 0.143 | **0.395** | 0.114 | — |
| **Best ablation: SEASON α=0.5** | 0.142 | 0.394 | **0.102** | — |

---

## Raw Output Files

```
results/
├── ZeroShot-Qwen2.5-VL-7B_predictions.json
├── CrashLogic-7B-Greedy_predictions.json
├── CrashLogic-7B-SEASON_predictions.json
├── CrashLogic-7B-SEASON-a0.5_predictions.json
├── CrashLogic-7B-SEASON-a1.5_predictions.json
├── CrashLogic-7B-SEASON-a2.0_predictions.json
├── CrashLogic-7B-SEASON-shuffle_predictions.json
├── CrashLogic-7B-SEASON-Full_predictions.json
├── comparison.md / comparison.tex
└── tables/all_tables.md / all_tables.tex
```

---

## Experimental Conditions (8 total)

1. **Zero-shot** — Qwen2.5-VL-7B-Instruct, no fine-tuning, greedy decode
2. **CrashLogic-7B Greedy** — QLoRA fine-tuned, greedy decode
3. **CrashLogic-7B + SEASON** — fine-tuned, SEASON α=1.0, reverse temporal negative
4. **SEASON α=0.5** — ablation
5. **SEASON α=1.5** — ablation
6. **SEASON α=2.0** — ablation
7. **SEASON shuffle** — shuffle temporal negative
8. **SEASON Full** — paper self-diagnostic SEASON (homogenization + spatial + JSD weights)
