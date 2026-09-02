# 07 — Table IV & IVb: Severity-Stratified Results

## Table IV — CrashLogic-7B + SEASON by Severity

**Purpose:** Performance breakdown by ground-truth crash severity — identifies where the model struggles.

| Severity | n | tIoU ↑ | BERTScore ↑ | ArgusCost-H ↓ | ArgusCost-O ↓ |
|----------|---|--------|-------------|---------------|---------------|
| Minor | 38 | 0.390 | 0.683 | 0.226 | 0.101 |
| Moderate | 69 | 0.397 | **0.688** | **0.201** | 0.101 |
| Severe | 30 | **0.422** | 0.687 | 0.213 | **0.072** |
| Fatal | 4 | 0.333 | 0.679 | 0.350 | 0.125 |
| N/a | 9 | 0.000 | 0.647 | 0.422 | 0.370 |

### Observations
- **Severe crashes:** Best tIoU (0.422) and lowest omission (ArgusCost-O 0.072) — model localizes dramatic events well.
- **Moderate:** Largest group (n=69), stable across metrics.
- **Fatal (n=4):** High hallucination cost (0.350) — small sample, unreliable.
- **N/a (n=9):** No defined crash window → tIoU=0 by definition; high omission expected.

---

## Table IVb — Cross-Model tIoU by Severity

| Severity | Greedy | SEASON | Zero-shot |
|----------|--------|--------|-----------|
| Minor | **0.425** | 0.390 | 0.023 |
| Moderate | 0.351 | **0.397** | 0.009 |
| Severe | **0.472** | 0.422 | 0.013 |
| Fatal | 0.333 | 0.333 | 0.000 |

### Observations
- Fine-tuning helps **all severity levels** vs zero-shot (30–50× tIoU improvement).
- Greedy vs SEASON trade off by severity — no single decoder wins everywhere.
- Zero-shot near-zero across all severities.

---

## LaTeX Caption

> *Table IV: SEASON performance stratified by GT crash severity. Table IVb: Cross-model tIoU by severity stratum.*
