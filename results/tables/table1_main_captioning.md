## Table I — Main Captioning Quality

| Method | BLEU-4 | ROUGE-L | METEOR | CIDEr | BERTScore | Explanation-BERTScore |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| CrashLogic-7B (Greedy) | 0.142 | 0.336 | 0.371 | 0.508 | 0.686 | 0.686 |
| CrashLogic-7B + SEASON | 0.140 | 0.338 | 0.365 | 0.507 | 0.684 | 0.684 |
| Zero-shot Qwen2.5-VL-7B | 0.016 | 0.160 | 0.203 | 0.338 | 0.486 | 0.486 |

_Table I: Corpus-level caption quality on the 150-video CCD test split. Metrics computed on the Explanation field unless noted. $\uparrow$ higher is better._
