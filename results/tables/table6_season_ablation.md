## Table VI — SEASON Ablation

| Variant | tIoU | BERTScore | ArgusCost-H | ArgusCost-O | ROUGE-L |
|:---|:---:|:---:|:---:|:---:|:---:|
| CrashLogic-7B (Greedy) | 0.373 | 0.686 | 0.223 | 0.107 | 0.336 |
| CrashLogic-7B + SEASON | 0.375 | 0.684 | 0.227 | 0.112 | 0.338 |
| CrashLogic-7B + SEASON (Full) | 0.357 | 0.685 | 0.240 | 0.119 | 0.332 |
| SEASON ($\alpha$=0.5) | 0.394 | 0.686 | 0.212 | 0.102 | 0.339 |
| SEASON ($\alpha$=1.5) | 0.354 | 0.681 | 0.249 | 0.124 | 0.327 |
| SEASON ($\alpha$=2.0) | 0.356 | 0.676 | 0.253 | 0.128 | 0.319 |
| SEASON (shuffle neg.) | 0.395 | 0.688 | 0.233 | 0.114 | 0.340 |

_Table VI: SEASON decoding ablation on CrashLogic-7B. Simple temporal negative (reverse/shuffle) vs. full self-diagnostic SEASON; $\alpha$ sweeps contrastive strength._
