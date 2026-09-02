# CrashX Journal Results

## Table I — Main Captioning Quality

| Method | BLEU-4 | ROUGE-L | METEOR | CIDEr | BERTScore | Explanation-BERTScore |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| CrashLogic-7B (Greedy) | 0.142 | 0.336 | 0.371 | 0.508 | 0.686 | 0.686 |
| CrashLogic-7B + SEASON | 0.140 | 0.338 | 0.365 | 0.507 | 0.684 | 0.684 |
| Zero-shot Qwen2.5-VL-7B | 0.016 | 0.160 | 0.203 | 0.338 | 0.486 | 0.486 |

_Table I: Corpus-level caption quality on the 150-video CCD test split. Metrics computed on the Explanation field unless noted. $\uparrow$ higher is better._

## Table II — Temporal & Forensic Reasoning

| Method | tIoU | THR@0.25 | THR@0.50 | ArgusCost-H | ArgusCost-O |
|:---|:---:|:---:|:---:|:---:|:---:|
| CrashLogic-7B (Greedy) | 0.373 | 0.573 | 0.440 | 0.223 | 0.107 |
| CrashLogic-7B + SEASON | 0.375 | 0.573 | 0.427 | 0.227 | 0.112 |
| Zero-shot Qwen2.5-VL-7B | 0.012 | 0.020 | 0.013 | 0.227 | 0.462 |

_Table II: Spatiotemporal and forensic metrics. tIoU = mean temporal IoU of crash windows; THR@$\tau$ = fraction with tIoU $\geq \tau$. ArgusCost-H/O = structured hallucination / omission rates ($\downarrow$ better)._

## Table III — Structured Field Accuracy

| Method | Severity Acc | Impact Jaccard | Vehicle Jaccard | Weather Jaccard | Timestamp ParseRate |
|:---|:---:|:---:|:---:|:---:|:---:|
| CrashLogic-7B (Greedy) | 0.460 | 0.516 | 0.570 | 0.880 | 1.000 |
| CrashLogic-7B + SEASON | 0.467 | 0.511 | 0.530 | 0.887 | 1.000 |
| Zero-shot Qwen2.5-VL-7B | 0.080 | 0.223 | 0.349 | 0.157 | 0.280 |

_Table III: Structured claim extraction accuracy. Severity-Acc = exact match rate; Jaccard scores for token-overlap fields; Timestamp-ParseRate = fraction with parseable Start/End._

## Table IV — Severity-Stratified (CrashLogic-7B + SEASON)

| Severity | tIoU | BERTScore | ArgusCost-H | ArgusCost-O | n |
|:---|:---:|:---:|:---:|:---:|:---:|
| Minor | 0.390 | 0.683 | 0.226 | 0.101 | 38 |
| Moderate | 0.397 | 0.688 | 0.201 | 0.101 | 69 |
| Severe | 0.422 | 0.687 | 0.213 | 0.072 | 30 |
| Fatal | 0.333 | 0.679 | 0.350 | 0.125 | 4 |
| N/a | 0.000 | 0.647 | 0.422 | 0.370 | 9 |

_Table IV: CrashLogic-7B + SEASON performance stratified by GT crash severity on the CCD test split._

## Table IVb — Cross-Model tIoU by Severity

| Severity | CrashLogic-7B (Greedy) | CrashLogic-7B + SEASON | Zero-shot Qwen2.5-VL-7B |
|:---|:---:|:---:|:---:|
| Minor | 0.425 | 0.390 | 0.023 |
| Moderate | 0.351 | 0.397 | 0.009 |
| Severe | 0.472 | 0.422 | 0.013 |
| Fatal | 0.333 | 0.333 | 0.000 |

_Table IVb: Mean temporal IoU (tIoU) stratified by crash severity across methods. Higher is better._

## Table V — Relative Gains over Zero-shot

| Method | BLEU-4 | ROUGE-L | METEOR | CIDEr | BERTScore | tIoU | THR@0.50 | ArgusCost-H | ArgusCost-O |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CrashLogic-7B (Greedy) | +812.6% | +109.8% | +83.0% | +50.5% | +41.1% | +2890.8% | +3200.0% | +2.1% | +76.9% |
| CrashLogic-7B + SEASON | +800.1% | +111.4% | +80.0% | +50.3% | +40.6% | +2908.6% | +3100.0% | +0.0% | +75.7% |

_Table V: Relative change vs. zero-shot baseline. Positive % on caption/temporal metrics = improvement; positive % on ArgusCost = reduction in hallucination/omission._

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

