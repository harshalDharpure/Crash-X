## Table II — Temporal & Forensic Reasoning

| Method | tIoU | THR@0.25 | THR@0.50 | ArgusCost-H | ArgusCost-O |
|:---|:---:|:---:|:---:|:---:|:---:|
| CrashLogic-7B (Greedy) | 0.373 | 0.573 | 0.440 | 0.223 | 0.107 |
| CrashLogic-7B + SEASON | 0.375 | 0.573 | 0.427 | 0.227 | 0.112 |
| Zero-shot Qwen2.5-VL-7B | 0.012 | 0.020 | 0.013 | 0.227 | 0.462 |

_Table II: Spatiotemporal and forensic metrics. tIoU = mean temporal IoU of crash windows; THR@$\tau$ = fraction with tIoU $\geq \tau$. ArgusCost-H/O = structured hallucination / omission rates ($\downarrow$ better)._
