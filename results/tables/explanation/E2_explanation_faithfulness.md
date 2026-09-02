## Table E2 — Explanation Faithfulness (NLI)

| Method | NLI-Entail | NLI-Contradict | NLI-Score | NLI-Loss | Expl.-BERTScore |
|:---|:---:|:---:|:---:|:---:|:---:|
| Zero-shot Qwen2.5-VL-7B | 0.139 | 0.092 | 0.047 | 3.811 | 0.486 |
| Zero-shot Qwen2.5-VL-3B | 0.145 | 0.056 | 0.089 | 3.562 | 0.518 |
| Zero-shot Qwen2-VL-2B | 0.092 | 0.099 | -0.007 | 4.462 | 0.418 |
| Zero-shot LLaVA-NeXT-Video-7B | 0.157 | 0.061 | 0.097 | 3.464 | 0.547 |
| CrashLogic-7B (Greedy) | 0.224 | 0.229 | -0.005 | 3.124 | 0.686 |
| CrashLogic-7B + TCD ($\alpha$=0.5) | 0.249 | 0.193 | 0.057 | 2.944 | 0.686 |

_Table E2: Explanation faithfulness. GT forensic text as premise; model explanation as hypothesis (cross-encoder NLI). NLI-Score $= P_e - P_c$ ($\uparrow$); NLI-Loss $= -\log P_e$ ($\downarrow$)._
