## Table VII — NLI Faithfulness & Hallucination

| Method | NLI Entailment | NLI Contradiction | NLI Score | NLI Loss | Full NLI Entailment | Full NLI Contradiction |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| CrashLogic-7B (Greedy) | 0.224 | 0.229 | -0.005 | 3.124 | 0.402 | 0.155 |
| CrashLogic-7B + SEASON | 0.276 | 0.186 | 0.090 | 2.954 | 0.386 | 0.149 |
| Zero-shot Qwen2.5-VL-7B | 0.139 | 0.092 | 0.047 | 3.811 | 0.128 | 0.052 |

_Table VII: NLI faithfulness via cross-encoder/nli-deberta-v3-small. GT forensic text as premise; model explanation (or full output) as hypothesis. NLI-Loss = $-\log P(\text{entail})$ (lower better); NLI-Score = $P_e - P_c$ (higher better)._
