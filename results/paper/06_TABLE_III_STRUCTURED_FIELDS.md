# 06 — Table III: Structured Field Accuracy

**Purpose:** Per-field structured claim extraction — measures whether the model correctly parses and fills forensic fields from video.

---

## Results

| Method | Severity-Acc ↑ | Impact-Jaccard ↑ | Vehicle-Jaccard ↑ | Weather-Jaccard ↑ | Timestamp-ParseRate ↑ |
|--------|----------------|------------------|-------------------|-------------------|----------------------|
| Zero-shot Qwen2.5-VL-7B | 0.080 | 0.223 | 0.349 | 0.157 | 0.280 |
| CrashLogic-7B (Greedy) | 0.460 | **0.516** | **0.570** | 0.880 | **1.000** |
| CrashLogic-7B + SEASON | **0.467** | 0.511 | 0.530 | **0.887** | **1.000** |

---

## Best per Field

| Field | Best Method | Value |
|-------|-------------|-------|
| Severity exact match | SEASON | **46.7%** |
| Impact location overlap | Greedy | **51.6%** Jaccard |
| Vehicle identification | Greedy | **57.0%** Jaccard |
| Weather condition | SEASON | **88.7%** Jaccard |
| Timestamp parsing | Greedy / SEASON | **100%** |

---

## Analysis

- Fine-tuning teaches the **structured output schema** — timestamp parse rate goes from 28% → 100%.
- **Severity accuracy ~46%** — room for improvement; confusion between minor/moderate/severe is common.
- **Weather** is easiest field (88% Jaccard) — limited vocabulary (normal, rainy, snowy, etc.).
- **Impact and vehicles** are hardest structured fields (~51–57% Jaccard) — high vocabulary diversity in GT.
- SEASON vs Greedy: negligible difference on structured fields — fine-tuning is the dominant factor.

---

## LaTeX Caption

> *Table III: Structured claim extraction accuracy. Severity-Acc = exact match; Jaccard = token overlap; Timestamp-ParseRate = fraction with parseable Start/End.*
