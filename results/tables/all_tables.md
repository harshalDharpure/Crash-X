# CrashX Journal Results

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

## Table II — Temporal & Forensic Reasoning

| Method | tIoU | THR@0.25 | THR@0.50 | ArgusCost-H | ArgusCost-O |
|:---|:---:|:---:|:---:|:---:|:---:|
| Zero-shot Qwen2.5-VL-7B | 0.012 | 0.020 | 0.013 | 0.227 | 0.462 |
| Zero-shot Qwen2.5-VL-3B | 0.010 | 0.007 | 0.000 | 0.285 | 0.413 |
| Zero-shot Qwen2-VL-2B | 0.000 | 0.000 | 0.000 | 0.564 | 0.607 |
| Zero-shot LLaVA-NeXT-Video-7B | 0.024 | 0.033 | 0.013 | 0.401 | 0.334 |
| CrashLogic-7B (Greedy) | 0.373 | 0.573 | 0.440 | 0.223 | 0.107 |
| CrashLogic-7B + TCD ($\alpha$=0.5) | 0.394 | 0.600 | 0.473 | 0.212 | 0.102 |

_Table II: Spatiotemporal and forensic metrics. tIoU = mean temporal IoU of crash windows; THR@$\tau$ = fraction with tIoU $\geq \tau$. ArgusCost-H/O = structured hallucination / omission rates ($\downarrow$ better)._

## Table III — Structured Field Accuracy

| Method | Severity Acc | Impact Jaccard | Vehicle Jaccard | Weather Jaccard | Timestamp ParseRate |
|:---|:---:|:---:|:---:|:---:|:---:|
| Zero-shot Qwen2.5-VL-7B | 0.080 | 0.223 | 0.349 | 0.157 | 0.280 |
| Zero-shot Qwen2.5-VL-3B | 0.187 | 0.257 | 0.376 | 0.187 | 0.413 |
| Zero-shot Qwen2-VL-2B | 0.087 | 0.144 | 0.232 | 0.140 | 0.087 |
| Zero-shot LLaVA-NeXT-Video-7B | 0.347 | 0.264 | 0.418 | 0.293 | 0.513 |
| CrashLogic-7B (Greedy) | 0.460 | 0.516 | 0.570 | 0.880 | 1.000 |
| CrashLogic-7B + TCD ($\alpha$=0.5) | 0.487 | 0.521 | 0.543 | 0.873 | 1.000 |

_Table III: Structured claim extraction accuracy. Severity-Acc = exact match rate; Jaccard scores for token-overlap fields; Timestamp-ParseRate = fraction with parseable Start/End._

## Table IV — Severity-Stratified (CrashLogic-7B + TCD ($\alpha$=0.5))

| Severity | tIoU | BERTScore | ArgusCost-H | ArgusCost-O | n |
|:---|:---:|:---:|:---:|:---:|:---:|
| Minor | 0.390 | 0.681 | 0.216 | 0.092 | 38 |
| Moderate | 0.431 | 0.692 | 0.177 | 0.087 | 69 |
| Severe | 0.439 | 0.686 | 0.213 | 0.072 | 30 |
| Fatal | 0.333 | 0.675 | 0.300 | 0.083 | 4 |
| N/a | 0.000 | 0.660 | 0.422 | 0.370 | 9 |

_Table IV: CrashLogic-7B + TCD ($\alpha$=0.5) performance stratified by GT crash severity on the CCD test split._

## Table IVb — Cross-Model tIoU by Severity

| Severity | Zero-shot Qwen2.5-VL-7B | Zero-shot Qwen2.5-VL-3B | Zero-shot Qwen2-VL-2B | Zero-shot LLaVA-NeXT-Video-7B | CrashLogic-7B (Greedy) | CrashLogic-7B + TCD ($\alpha$=0.5) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Minor | 0.023 | 0.002 | 0.000 | 0.023 | 0.425 | 0.390 |
| Moderate | 0.009 | 0.013 | 0.000 | 0.004 | 0.351 | 0.431 |
| Severe | 0.013 | 0.011 | 0.000 | 0.015 | 0.472 | 0.439 |
| Fatal | 0.000 | 0.037 | 0.000 | 0.000 | 0.333 | 0.333 |

_Table IVb: Mean temporal IoU (tIoU) stratified by crash severity across methods. Higher is better._

## Table V — Relative Gains over Zero-shot

| Method | BLEU-4 | ROUGE-L | METEOR | CIDEr | BERTScore | tIoU | THR@0.50 | ArgusCost-H | ArgusCost-O |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Zero-shot Qwen2.5-VL-3B | +22.9% | +12.2% | +15.5% | +16.8% | +6.6% | -22.8% | -100.0% | -25.2% | +10.6% |
| Zero-shot Qwen2-VL-2B | +16.3% | -24.9% | -35.0% | -34.4% | -14.0% | -100.0% | -100.0% | -147.9% | -31.3% |
| Zero-shot LLaVA-NeXT-Video-7B | +64.4% | +23.1% | +35.5% | +33.1% | +12.4% | +91.7% | +0.0% | -76.5% | +27.6% |
| CrashLogic-7B (Greedy) | +812.6% | +109.8% | +83.0% | +50.5% | +41.1% | +2890.8% | +3200.0% | +2.1% | +76.9% |
| CrashLogic-7B + TCD ($\alpha$=0.5) | +808.0% | +111.8% | +82.0% | +52.7% | +41.0% | +3060.1% | +3450.0% | +6.8% | +77.9% |

_Table V: Relative change vs. zero-shot baseline. Positive % on caption/temporal metrics = improvement; positive % on ArgusCost = reduction in hallucination/omission._

## Table VI — SEASON Ablation

| Variant | tIoU | BERTScore | ArgusCost-H | ArgusCost-O | ROUGE-L |
|:---|:---:|:---:|:---:|:---:|:---:|
| CrashLogic-7B (Greedy) | 0.373 | 0.686 | 0.223 | 0.107 | 0.336 |
| CrashLogic-7B + SEASON ($\alpha$=1.0) | 0.375 | 0.684 | 0.227 | 0.112 | 0.338 |
| CrashLogic-7B + SEASON (Full) | 0.357 | 0.685 | 0.240 | 0.119 | 0.332 |
| CrashLogic-7B + TCD ($\alpha$=0.5) | 0.394 | 0.686 | 0.212 | 0.102 | 0.339 |
| SEASON ($\alpha$=1.5) | 0.354 | 0.681 | 0.249 | 0.124 | 0.327 |
| SEASON ($\alpha$=2.0) | 0.356 | 0.676 | 0.253 | 0.128 | 0.319 |
| SEASON (shuffle neg.) | 0.395 | 0.688 | 0.233 | 0.114 | 0.340 |

_Table VI: SEASON decoding ablation on CrashLogic-7B. Simple temporal negative (reverse/shuffle) vs. full self-diagnostic SEASON; $\alpha$ sweeps contrastive strength._

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

## Table VIII — Bootstrap 95% Confidence Intervals

| Method | Metric | Mean | 95% CI |
|:---|:---:|:---:|:---:|
| Zero-shot Qwen2.5-VL-7B | tIoU | 0.012 | [0.000, 0.028] |
| Zero-shot Qwen2.5-VL-7B | ArgusCost-O | 0.462 | [0.438, 0.486] |
| Zero-shot Qwen2.5-VL-7B | ArgusCost-H | 0.227 | [0.189, 0.267] |
| Zero-shot Qwen2.5-VL-7B | BLEU-4 | 0.015 | [0.013, 0.017] |
| Zero-shot Qwen2.5-VL-7B | ROUGE-L | 0.160 | [0.151, 0.168] |
| Zero-shot Qwen2.5-VL-7B | Severity-Acc | 0.080 | [0.040, 0.127] |
| Zero-shot Qwen2.5-VL-3B | tIoU | 0.010 | [0.003, 0.017] |
| Zero-shot Qwen2.5-VL-3B | ArgusCost-O | 0.413 | [0.393, 0.434] |
| Zero-shot Qwen2.5-VL-3B | ArgusCost-H | 0.285 | [0.246, 0.325] |
| Zero-shot Qwen2.5-VL-3B | BLEU-4 | 0.019 | [0.017, 0.021] |
| Zero-shot Qwen2.5-VL-3B | ROUGE-L | 0.180 | [0.173, 0.186] |
| Zero-shot Qwen2.5-VL-3B | Severity-Acc | 0.187 | [0.120, 0.253] |
| Zero-shot Qwen2-VL-2B | tIoU | 0.000 | [0.000, 0.000] |
| Zero-shot Qwen2-VL-2B | ArgusCost-O | 0.607 | [0.570, 0.643] |
| Zero-shot Qwen2-VL-2B | ArgusCost-H | 0.564 | [0.500, 0.633] |
| Zero-shot Qwen2-VL-2B | BLEU-4 | 0.014 | [0.011, 0.016] |
| Zero-shot Qwen2-VL-2B | ROUGE-L | 0.120 | [0.106, 0.134] |
| Zero-shot Qwen2-VL-2B | Severity-Acc | 0.087 | [0.047, 0.133] |
| Zero-shot LLaVA-NeXT-Video-7B | tIoU | 0.024 | [0.007, 0.047] |
| Zero-shot LLaVA-NeXT-Video-7B | ArgusCost-O | 0.334 | [0.314, 0.354] |
| Zero-shot LLaVA-NeXT-Video-7B | ArgusCost-H | 0.401 | [0.372, 0.432] |
| Zero-shot LLaVA-NeXT-Video-7B | BLEU-4 | 0.026 | [0.024, 0.028] |
| Zero-shot LLaVA-NeXT-Video-7B | ROUGE-L | 0.197 | [0.191, 0.202] |
| Zero-shot LLaVA-NeXT-Video-7B | Severity-Acc | 0.347 | [0.273, 0.427] |
| CrashLogic-7B (Greedy) | tIoU | 0.373 | [0.314, 0.434] |
| CrashLogic-7B (Greedy) | ArgusCost-O | 0.107 | [0.088, 0.126] |
| CrashLogic-7B (Greedy) | ArgusCost-H | 0.223 | [0.200, 0.248] |
| CrashLogic-7B (Greedy) | BLEU-4 | 0.123 | [0.113, 0.133] |
| CrashLogic-7B (Greedy) | ROUGE-L | 0.336 | [0.325, 0.347] |
| CrashLogic-7B (Greedy) | Severity-Acc | 0.460 | [0.387, 0.540] |
| CrashLogic-7B + TCD ($\alpha$=0.5) | tIoU | 0.394 | [0.333, 0.457] |
| CrashLogic-7B + TCD ($\alpha$=0.5) | ArgusCost-O | 0.102 | [0.083, 0.123] |
| CrashLogic-7B + TCD ($\alpha$=0.5) | ArgusCost-H | 0.212 | [0.188, 0.239] |
| CrashLogic-7B + TCD ($\alpha$=0.5) | BLEU-4 | 0.123 | [0.114, 0.132] |
| CrashLogic-7B + TCD ($\alpha$=0.5) | ROUGE-L | 0.339 | [0.329, 0.347] |
| CrashLogic-7B + TCD ($\alpha$=0.5) | Severity-Acc | 0.487 | [0.400, 0.567] |

_Table VIII: Bootstrap 95% confidence intervals ($N=1000$ resamples) on per-video scores. Primary forensic metrics for Option A benchmark paper._

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

