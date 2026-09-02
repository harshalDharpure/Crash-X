# 08 — Table V: Relative Gains over Zero-shot

**Purpose:** Quantify improvement of fine-tuned methods vs zero-shot baseline as percentage change.

**Note:** Positive % on caption/temporal metrics = improvement. Positive % on ArgusCost = **reduction** in hallucination/omission (lower is better).

---

## Results

| Method | BLEU-4 | ROUGE-L | METEOR | CIDEr | BERTScore | tIoU | THR@0.50 | ArgusCost-H | ArgusCost-O |
|--------|--------|---------|--------|-------|-----------|------|----------|-------------|-------------|
| CrashLogic Greedy | **+812.6%** | +109.8% | **+83.0%** | **+50.5%** | **+41.1%** | +2890.8% | **+3200%** | +2.1% | **+76.9%** |
| CrashLogic + SEASON | +800.1% | **+111.4%** | +80.0% | +50.3% | +40.6% | **+2908.6%** | +3100% | 0.0% | +75.7% |

---

## Headline Numbers for Paper

| Claim | Value |
|-------|-------|
| BLEU-4 improvement | **+813%** (0.016 → 0.142) |
| BERTScore improvement | **+41%** (0.486 → 0.686) |
| tIoU improvement | **+29×** (0.012 → 0.373) |
| Omission reduction (ArgusCost-O) | **−77%** (0.462 → 0.107) |
| Timestamp parse rate | **28% → 100%** |

---

## Analysis

- **Fine-tuning is the dominant gain** — both Greedy and SEASON show massive improvements over zero-shot.
- SEASON adds marginal extra gains on ROUGE-L (+111% vs +110%) and tIoU (+2909% vs +2891%) vs Greedy.
- ArgusCost-H barely changes — hallucination rate is similar; the main win is **reducing omissions** (ArgusCost-O).

---

## LaTeX Caption

> *Table V: Relative change vs zero-shot Qwen2.5-VL-7B baseline. Positive % on ArgusCost indicates reduction in error rate.*
