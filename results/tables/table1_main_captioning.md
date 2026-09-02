## Table I — Main Captioning Quality

| Method | BLEU-4 | ROUGE-L | METEOR | CIDEr | BERTScore | Explanation-BERTScore |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Zero-shot Qwen2.5-VL-7B | 0.016 | 0.160 | 0.203 | 0.338 | 0.486 | 0.486 |
| Zero-shot Qwen2.5-VL-3B | 0.019 | 0.180 | 0.234 | 0.394 | 0.518 | 0.518 |
| Zero-shot Qwen2-VL-2B | 0.018 | 0.120 | 0.132 | 0.221 | 0.418 | 0.418 |
| Zero-shot LLaVA-NeXT-Video-7B | 0.026 | 0.197 | 0.275 | 0.449 | 0.547 | 0.547 |
| CrashLogic-7B (Greedy) | 0.142 | 0.336 | 0.371 | 0.508 | 0.686 | 0.686 |
| CrashLogic-7B + TCD ($\alpha$=0.5) | 0.142 | 0.339 | 0.369 | 0.515 | 0.686 | 0.686 |

_Table I: Corpus-level caption quality on the 150-video CCD test split. Metrics computed on the Explanation field unless noted. $\uparrow$ higher is better._
