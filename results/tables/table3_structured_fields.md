## Table III — Structured Field Accuracy

| Method | Severity Acc | Impact Jaccard | Vehicle Jaccard | Weather Jaccard | Timestamp ParseRate |
|:---|:---:|:---:|:---:|:---:|:---:|
| Zero-shot Qwen2.5-VL-7B | 0.080 | 0.223 | 0.349 | 0.157 | 0.280 |
| Zero-shot Qwen2.5-VL-3B | 0.187 | 0.257 | 0.376 | 0.187 | 0.413 |
| Zero-shot Qwen2-VL-2B | 0.087 | 0.144 | 0.232 | 0.140 | 0.087 |
| Zero-shot LLaVA-NeXT-Video-7B | 0.347 | 0.264 | 0.418 | 0.293 | 0.513 |
| CrashLogic-7B (Greedy) | 0.460 | 0.516 | 0.570 | 0.880 | 1.000 |
| CrashLogic-7B + TCD ($\alpha$=0.5) | 0.487 | 0.521 | 0.543 | 0.873 | 1.000 |

_Table III: Structured claim extraction accuracy. Severity-Acc = exact match rate; Jaccard scores for token-overlap fields; Timestamp-ParseRate = fraction with parseable Start/End._
