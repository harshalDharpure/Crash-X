## Table II — Temporal & Forensic Reasoning

| Method | tIoU | THR@0.25 | THR@0.50 | ArgusCost-H | ArgusCost-O |
|:---|:---:|:---:|:---:|:---:|:---:|
| Zero-shot Qwen2.5-VL-7B | 0.012 | 0.020 | 0.013 | 0.227 | 0.462 |
| Zero-shot Qwen2.5-VL-3B | 0.010 | 0.007 | 0.000 | 0.285 | 0.413 |
| Zero-shot Qwen2-VL-2B | 0.000 | 0.000 | 0.000 | 0.564 | 0.607 |
| Zero-shot LLaVA-NeXT-Video-7B | 0.024 | 0.033 | 0.013 | 0.401 | 0.334 |
| CrashLogic-7B (Greedy) | 0.373 | 0.573 | 0.440 | 0.223 | 0.107 |
| CrashLogic-7B + TCD ($\alpha$=0.5) | 0.394 | 0.600 | 0.473 | 0.212 | 0.102 |

_Table II: Spatiotemporal and forensic metrics. tIoU = mean temporal IoU of crash windows; THR@$\tau$ = fraction with tIoU $\geq \tau$. ArgusCost-H/O = structured hallucination / omission rates ($\downarrow$ better)._
