## Table IX — Paired Significance Tests

| Comparison | Metric | Wilcoxon $p$ | Significant ($p<0.05$) |
|:---|:---:|:---:|:---:|
| CrashLogic-7B (Greedy) vs CrashLogic-7B + TCD ($\alpha$=0.5) | tIoU | 0.289 | No |
| CrashLogic-7B (Greedy) vs CrashLogic-7B + TCD ($\alpha$=0.5) | ArgusCost-O | 0.240 | No |
| CrashLogic-7B (Greedy) vs CrashLogic-7B + TCD ($\alpha$=0.5) | BLEU-4 | 0.222 | No |
| CrashLogic-7B (Greedy) vs CrashLogic-7B + TCD ($\alpha$=0.5) | Severity-Acc | 0.103 | No |
| Zero-shot Qwen2.5-VL-7B vs CrashLogic-7B + TCD ($\alpha$=0.5) | tIoU | <0.001 | Yes |
| Zero-shot Qwen2.5-VL-7B vs CrashLogic-7B + TCD ($\alpha$=0.5) | ArgusCost-O | <0.001 | Yes |
| Zero-shot Qwen2.5-VL-7B vs CrashLogic-7B + TCD ($\alpha$=0.5) | BLEU-4 | <0.001 | Yes |
| Zero-shot Qwen2.5-VL-7B vs CrashLogic-7B + TCD ($\alpha$=0.5) | Severity-Acc | <0.001 | Yes |

_Table IX: Paired Wilcoxon signed-rank tests on aligned per-video scores. One-sided tests: higher tIoU/BLEU/Severity-Acc and lower ArgusCost-O are better._
