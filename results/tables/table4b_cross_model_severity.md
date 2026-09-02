## Table IVb — Cross-Model tIoU by Severity

| Severity | Zero-shot Qwen2.5-VL-7B | Zero-shot Qwen2.5-VL-3B | Zero-shot Qwen2-VL-2B | Zero-shot LLaVA-NeXT-Video-7B | CrashLogic-7B (Greedy) | CrashLogic-7B + TCD ($\alpha$=0.5) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Minor | 0.023 | 0.002 | 0.000 | 0.023 | 0.425 | 0.390 |
| Moderate | 0.009 | 0.013 | 0.000 | 0.004 | 0.351 | 0.431 |
| Severe | 0.013 | 0.011 | 0.000 | 0.015 | 0.472 | 0.439 |
| Fatal | 0.000 | 0.037 | 0.000 | 0.000 | 0.333 | 0.333 |

_Table IVb: Mean temporal IoU (tIoU) stratified by crash severity across methods. Higher is better._
