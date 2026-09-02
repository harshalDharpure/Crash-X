# 11 — Full Metrics Master Table

**All 8 experimental conditions × all metrics.** Test set n=150.

---

## Complete Results

| Condition | BLEU-4 | ROUGE-L | METEOR | CIDEr | BERTScore | tIoU | THR@0.25 | THR@0.50 | Argus-H | Argus-O | Sev-Acc | TS-Parse |
|-----------|--------|---------|--------|-------|-----------|------|----------|----------|---------|---------|---------|----------|
| Zero-shot | 0.016 | 0.160 | 0.203 | 0.338 | 0.486 | 0.012 | 0.020 | 0.013 | 0.227 | 0.462 | 0.080 | 0.280 |
| Greedy | 0.142 | 0.336 | 0.371 | 0.508 | 0.686 | 0.373 | 0.573 | 0.440 | 0.223 | 0.107 | 0.460 | 1.000 |
| SEASON α=1.0 | 0.140 | 0.338 | 0.365 | 0.507 | 0.684 | 0.375 | 0.573 | 0.427 | 0.227 | 0.112 | 0.467 | 1.000 |
| **SEASON α=0.5** | 0.142 | 0.339 | 0.369 | **0.515** | 0.686 | **0.394** | — | — | 0.212 | **0.102** | — | — |
| SEASON α=1.5 | 0.133 | 0.327 | 0.357 | 0.499 | 0.681 | 0.354 | — | — | 0.249 | 0.124 | — | — |
| SEASON α=2.0 | 0.133 | 0.319 | 0.344 | 0.491 | 0.676 | 0.356 | — | — | 0.253 | 0.128 | — | — |
| **SEASON shuffle** | **0.143** | **0.340** | 0.366 | 0.511 | **0.688** | **0.395** | — | — | 0.233 | 0.114 | — | — |
| SEASON Full | 0.147 | 0.332 | 0.364 | 0.513 | 0.685 | 0.357 | — | — | 0.240 | 0.119 | — | — |

---

## NLI Metrics (Main 3 Conditions)

| Condition | NLI-Entail | NLI-Contradict | NLI-Score | NLI-Loss | Full-Entail | Full-Contradict | Full-Loss |
|-----------|------------|----------------|-----------|----------|-------------|-----------------|-----------|
| Zero-shot | 0.139 | 0.092 | 0.047 | 3.811 | 0.128 | 0.052 | 3.608 |
| Greedy | 0.224 | 0.229 | -0.005 | 3.124 | **0.402** | 0.155 | **2.006** |
| SEASON α=1.0 | **0.276** | 0.186 | **0.090** | **2.954** | 0.386 | 0.149 | 2.075 |

---

## Structured Field Metrics (Main 3 Conditions)

| Condition | Severity-Acc | Impact-Jaccard | Vehicle-Jaccard | Weather-Jaccard | Timestamp-ParseRate |
|-----------|-------------|----------------|-----------------|-----------------|---------------------|
| Zero-shot | 0.080 | 0.223 | 0.349 | 0.157 | 0.280 |
| Greedy | 0.460 | **0.516** | **0.570** | 0.880 | 1.000 |
| SEASON α=1.0 | **0.467** | 0.511 | 0.530 | **0.887** | 1.000 |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | Qwen2.5-VL-7B-Instruct |
| Fine-tuning | 4-bit QLoRA, r=16, α=32 |
| Epochs / steps | 5 / 750 |
| Train videos | 1,198 |
| Keyframes | 8 @ 224px |
| Final train loss | ~3.09 |
| LoRA checkpoint | `outputs/crashlogic_7b_lora/` (~190 MB) |

---

## File Locations

| File | Path |
|------|------|
| Predictions (per condition) | `results/{Condition}_predictions.json` |
| Metrics (per condition) | `results/{Condition}_metrics.json` |
| Detailed metrics | `results/{Condition}_detailed.json` |
| All tables combined | `results/tables/all_tables.md` |
| LaTeX tables | `results/tables/all_tables.tex` |
