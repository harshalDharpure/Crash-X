## Table III — Structured Field Accuracy

| Method | Severity Acc | Impact Jaccard | Vehicle Jaccard | Weather Jaccard | Timestamp ParseRate |
|:---|:---:|:---:|:---:|:---:|:---:|
| CrashLogic-7B (Greedy) | 0.460 | 0.516 | 0.570 | 0.880 | 1.000 |
| CrashLogic-7B + SEASON | 0.467 | 0.511 | 0.530 | 0.887 | 1.000 |
| Zero-shot Qwen2.5-VL-7B | 0.080 | 0.223 | 0.349 | 0.157 | 0.280 |

_Table III: Structured claim extraction accuracy. Severity-Acc = exact match rate; Jaccard scores for token-overlap fields; Timestamp-ParseRate = fraction with parseable Start/End._
