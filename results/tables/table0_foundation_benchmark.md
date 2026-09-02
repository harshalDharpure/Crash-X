## Table 0 — Foundation Model Benchmark (Option A)

| Method | tIoU | ArgusCost-O | ArgusCost-H | Severity-Acc | Timestamp-ParseRate | BLEU-4 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Zero-shot Qwen2.5-VL-7B | 0.012 | 0.462 | 0.227 | 0.080 | 0.280 | 0.016 |
| Zero-shot Qwen2.5-VL-3B | 0.010 | 0.413 | 0.285 | 0.187 | 0.413 | 0.019 |
| Zero-shot Qwen2-VL-2B | 0.000 | 0.607 | 0.564 | 0.087 | 0.087 | 0.018 |
| Zero-shot LLaVA-NeXT-Video-7B | 0.024 | 0.334 | 0.401 | 0.347 | 0.513 | 0.026 |
| CrashLogic-7B (Greedy) | 0.373 | 0.107 | 0.223 | 0.460 | 1.000 | 0.142 |
| CrashLogic-7B + TCD ($\alpha$=0.5) | 0.394 | 0.102 | 0.212 | 0.487 | 1.000 | 0.142 |

_Table 0: CrashX foundation-model benchmark on 150-video CCD test split. Foundation VLMs (zero-shot) vs domain-adapted CrashLogic reference baselines. TCD = Task-Adapted Temporal Contrastive Decoding ($\alpha$=0.5)._
