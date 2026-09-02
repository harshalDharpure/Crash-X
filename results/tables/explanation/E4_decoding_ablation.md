## Table E4 — Decoding Ablation (Explanation Focus)

| Variant | BLEU-4 | ROUGE-L | BERTScore | NLI-Score | tIoU |
|:---|:---:|:---:|:---:|:---:|:---:|
| CrashLogic-7B (Greedy) | 0.142 | 0.336 | 0.686 | -0.005 | 0.373 |
| CrashLogic-7B + TCD ($\alpha$=0.5) | 0.142 | 0.339 | 0.686 | 0.057 | 0.394 |
| CrashLogic-7B + TCD ($\alpha$=1.0) | 0.140 | 0.338 | 0.684 | 0.090 | 0.375 |
| SEASON ($\alpha$=1.5) | 0.133 | 0.327 | 0.681 | -0.022 | 0.354 |
| SEASON ($\alpha$=2.0) | 0.133 | 0.319 | 0.676 | 0.016 | 0.356 |
| SEASON (shuffle neg.) | 0.143 | 0.340 | 0.688 | -0.025 | 0.395 |
| CrashLogic-7B + SEASON (Full) | 0.147 | 0.332 | 0.685 | 0.064 | 0.357 |

_Table E4: Temporal contrastive decoding (TCD / SEASON) ablation on CrashLogic-7B. Primary recommended setting: $\alpha$=0.5 with reverse temporal negative._
