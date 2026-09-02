## Table E3 — Temporal Consistency (Support)

| Method | tIoU | THR@0.25 | THR@0.50 |
|:---|:---:|:---:|:---:|
| Zero-shot Qwen2.5-VL-7B | 0.012 | 0.000 | 0.000 |
| Zero-shot Qwen2.5-VL-3B | 0.010 | 0.000 | 0.000 |
| Zero-shot Qwen2-VL-2B | 0.000 | 0.000 | 0.000 |
| Zero-shot LLaVA-NeXT-Video-7B | 0.024 | 0.000 | 0.000 |
| CrashLogic-7B (Greedy) | 0.373 | 0.573 | 0.440 |
| CrashLogic-7B + TCD ($\alpha$=0.5) | 0.394 | 0.600 | 0.473 |

_Table E3: Temporal consistency support (crash window localization). Reported as evidence that better explanations align with correct event timing; not a severity / forensic-field evaluation._
