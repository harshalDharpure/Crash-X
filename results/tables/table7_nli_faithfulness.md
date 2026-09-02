## Table VII — NLI Faithfulness & Hallucination

| Method | NLI Entailment | NLI Contradiction | NLI Score | NLI Loss | Full NLI Entailment | Full NLI Contradiction |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Zero-shot Qwen2.5-VL-7B | 0.139 | 0.092 | 0.047 | 3.811 | 0.128 | 0.052 |
| Zero-shot Qwen2.5-VL-3B | 0.145 | 0.056 | 0.089 | 3.562 | 0.152 | 0.041 |
| Zero-shot Qwen2-VL-2B | 0.092 | 0.099 | -0.007 | 4.462 | 0.099 | 0.094 |
| Zero-shot LLaVA-NeXT-Video-7B | 0.157 | 0.061 | 0.097 | 3.464 | 0.153 | 0.061 |
| CrashLogic-7B (Greedy) | 0.224 | 0.229 | -0.005 | 3.124 | 0.402 | 0.155 |
| CrashLogic-7B + TCD ($\alpha$=0.5) | 0.249 | 0.193 | 0.057 | 2.944 | 0.395 | 0.160 |

_Table VII: NLI faithfulness via cross-encoder/nli-deberta-v3-small. GT forensic text as premise; model explanation (or full output) as hypothesis. NLI-Loss = $-\log P(\text{entail})$ (lower better); NLI-Score = $P_e - P_c$ (higher better)._
