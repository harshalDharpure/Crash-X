## Table V — Relative Gains over Zero-shot

| Method | BLEU-4 | ROUGE-L | METEOR | CIDEr | BERTScore | tIoU | THR@0.50 | ArgusCost-H | ArgusCost-O |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CrashLogic-7B (Greedy) | +812.6% | +109.8% | +83.0% | +50.5% | +41.1% | +2890.8% | +3200.0% | +2.1% | +76.9% |
| CrashLogic-7B + SEASON | +800.1% | +111.4% | +80.0% | +50.3% | +40.6% | +2908.6% | +3100.0% | +0.0% | +75.7% |

_Table V: Relative change vs. zero-shot baseline. Positive % on caption/temporal metrics = improvement; positive % on ArgusCost = reduction in hallucination/omission._
