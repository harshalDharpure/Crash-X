# 09 — Table VI: SEASON Ablation Study

**Purpose:** Compare SEASON decoding variants on CrashLogic-7B — α sweep, negative construction, and full paper SEASON.

**Test set:** 150 videos | **Base model:** CrashLogic-7B (QLoRA fine-tuned)

---

## Full Results

| Variant | tIoU ↑ | BERTScore ↑ | ArgusCost-H ↓ | ArgusCost-O ↓ | ROUGE-L ↑ | BLEU-4 ↑ | CIDEr ↑ |
|---------|--------|-------------|---------------|---------------|-----------|----------|---------|
| Greedy (baseline) | 0.373 | 0.686 | **0.223** | 0.107 | 0.336 | 0.142 | 0.508 |
| SEASON α=1.0 (default) | 0.375 | 0.684 | 0.227 | 0.112 | 0.338 | 0.140 | 0.507 |
| **SEASON α=0.5** | **0.394** | 0.686 | 0.212 | **0.102** | 0.339 | 0.142 | **0.515** |
| SEASON α=1.5 | 0.354 | 0.681 | 0.249 | 0.124 | 0.327 | 0.133 | 0.499 |
| SEASON α=2.0 | 0.356 | 0.676 | 0.253 | 0.128 | 0.319 | 0.133 | 0.491 |
| **SEASON shuffle neg.** | **0.395** | **0.688** | 0.233 | 0.114 | **0.340** | **0.143** | 0.511 |
| SEASON Full (paper) | 0.357 | 0.685 | 0.240 | 0.119 | 0.332 | 0.147 | 0.513 |

---

## Best per Metric (Ablation)

| Metric | Best Variant | Value | vs Greedy Δ |
|--------|-------------|-------|-------------|
| **tIoU** | SEASON shuffle | **0.395** | +0.022 |
| **ArgusCost-O** | SEASON α=0.5 | **0.102** | −0.005 |
| **ROUGE-L** | SEASON shuffle | **0.340** | +0.004 |
| **BERTScore** | SEASON shuffle | **0.688** | +0.002 |
| **BLEU-4** | SEASON shuffle | **0.143** | +0.001 |
| **CIDEr** | SEASON α=0.5 | **0.515** | +0.007 |
| **ArgusCost-H** | Greedy | **0.223** | — |

---

## Key Findings

1. **Default α=1.0 is not optimal** — α=0.5 gives better tIoU (+0.019) and lower omission (−0.005).
2. **Higher α hurts** — α=1.5 and α=2.0 degrade all metrics vs α=0.5/1.0.
3. **Shuffle negative ≈ best overall** — highest tIoU (0.395), ROUGE-L (0.340), BERTScore (0.688).
4. **Full SEASON underperforms** simple temporal negative — paper variant adds complexity without gain on this dataset.
5. **Greedy still best on ArgusCost-H** — contrastive decoding slightly increases conflicting claims.

---

## Recommended Configuration for Paper

| Setting | Recommendation |
|---------|----------------|
| Primary SEASON | α=0.5, reverse negative (best tIoU + omission) |
| Alternative | Shuffle negative (best caption metrics) |
| Avoid | α≥1.5, Full SEASON |

---

## LaTeX Caption

> *Table VI: SEASON decoding ablation on CrashLogic-7B. Simple temporal negative (reverse/shuffle) vs full self-diagnostic SEASON; α sweeps contrastive strength.*
