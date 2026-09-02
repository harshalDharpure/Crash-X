# 04 — Table I: Main Captioning Quality

**Purpose:** Corpus-level dense caption quality on the 150-video CCD test split. Metrics computed on the **Explanation** field extracted from structured output.

**Direction:** ↑ higher is better for all metrics.

---

## Results

| Method | BLEU-4 | ROUGE-L | METEOR | CIDEr | BERTScore | Expl-BERTScore |
|--------|--------|---------|--------|-------|-----------|----------------|
| Zero-shot Qwen2.5-VL-7B | 0.016 | 0.160 | 0.203 | 0.338 | 0.486 | 0.486 |
| **CrashLogic-7B (Greedy)** | **0.142** | 0.336 | **0.371** | **0.508** | **0.686** | **0.686** |
| CrashLogic-7B + SEASON | 0.140 | **0.338** | 0.365 | 0.507 | 0.684 | 0.684 |

---

## Best per Metric

| Metric | Best Method | Value |
|--------|-------------|-------|
| BLEU-4 | CrashLogic Greedy | **0.142** |
| ROUGE-L | CrashLogic + SEASON | **0.338** |
| METEOR | CrashLogic Greedy | **0.371** |
| CIDEr | CrashLogic Greedy | **0.508** |
| BERTScore | CrashLogic Greedy | **0.686** |

---

## Analysis

- **Fine-tuning dominates zero-shot** across all caption metrics (BLEU-4 +788%, BERTScore +41%).
- Zero-shot produces free-form prose (no structured format) → very low BLEU/n-gram overlap despite sometimes reasonable semantic content.
- **SEASON vs Greedy:** Nearly identical on caption metrics; SEASON slightly better on ROUGE-L (+0.002), Greedy slightly better on BLEU-4 and METEOR.
- Caption quality is **not** where SEASON adds value — temporal/forensic metrics (Tables II, VII) show the SEASON advantage.

---

## LaTeX Caption

> *Table I: Corpus-level caption quality on CCD test split (n=150). Metrics on Explanation field. ↑ higher is better.*
