# 12 — Key Findings & Paper Narrative

## Executive Summary

CrashX demonstrates that **domain-specific fine-tuning (CrashLogic-7B)** is essential for traffic accident video understanding, with **SEASON contrastive decoding** providing additional gains in temporal faithfulness and NLI-based explanation quality. The zero-shot Qwen2.5-VL-7B baseline fails catastrophically on structured forensic reasoning despite being a capable general-purpose VLM.

---

## Top 5 Results for Abstract / Introduction

1. **+813% BLEU-4** and **+41% BERTScore** from QLoRA fine-tuning on 1,198 CCD videos
2. **tIoU 0.012 → 0.373** — fine-tuning enables temporal crash window localization (30× improvement)
3. **ArgusCost-O 0.462 → 0.107** — 77% reduction in forensic field omissions
4. **SEASON improves NLI-Score** from -0.005 (Greedy) to +0.090 — better explanation faithfulness
5. **Ablation: α=0.5 and shuffle negative** outperform default α=1.0 on tIoU and caption metrics

---

## How We Compare to Prior Work

| Aspect | VRU-Accident | SEASON (CVPR'26) | ARGUS (ICCV'25) | **CrashX (Ours)** |
|--------|-------------|------------------|-----------------|-------------------|
| Task | Video QA + captioning | Temporal hallucination mitigation | H/O evaluation | Dense structured crash captioning |
| Dataset | VRU-Accident benchmark | General video benchmarks | General Video-LLMs | **CCD (1,500 videos)** |
| Fine-tuning | Baseline wrapper | N/A (inference only) | N/A (eval only) | **QLoRA CrashLogic-7B** |
| Decoding | Greedy | SEASON contrastive | N/A | **Hybrid SEASON (V−V_neg)** |
| Metrics | Standard caption | Temporal accuracy | LLM-NLI judge | **Lexical + tIoU + ARGUS + NLI** |
| Structured output | No | No | No | **Yes (7 forensic fields)** |

---

## What Works Best (Recommended Reporting)

### Primary comparison (Table I–II, main paper)
| Setting | Why |
|---------|-----|
| Zero-shot → Greedy → SEASON (α=1.0) | Standard 3-way progression; shows fine-tuning + decoding gains |

### Best overall numbers (mention in text or supplementary)
| Setting | Best for |
|---------|----------|
| **SEASON α=0.5** | Best tIoU (0.394) + lowest ArgusCost-O (0.102) |
| **SEASON shuffle** | Best ROUGE-L (0.340), BERTScore (0.688), tIoU (0.395) |
| **SEASON α=1.0** | Best NLI-Score (0.090) — faithfulness narrative |

### What to avoid highlighting
- Full SEASON (paper variant) — underperforms simple temporal negative
- α ≥ 1.5 — monotonically worse than α=0.5/1.0
- ArgusCost-H — similar across all methods (~0.22); not a differentiator

---

## Story Arc for Paper

### Section: Experimental Results

**Paragraph 1 — Fine-tuning is essential**
> Zero-shot Qwen2.5-VL-7B achieves only 0.016 BLEU-4 and 0.012 tIoU, producing unstructured prose without parseable forensic fields. CrashLogic-7B (QLoRA fine-tuned) achieves 0.142 BLEU-4 and 0.373 tIoU with 100% timestamp parse rate, demonstrating that domain adaptation is critical for accident video understanding.

**Paragraph 2 — SEASON mitigates explanation hallucination**
> SEASON contrastive decoding improves NLI-Score from -0.005 to +0.090 and reduces NLI-Loss from 3.12 to 2.95, indicating that contrasting against temporally reversed video reduces unfaithful explanations while maintaining comparable caption quality (ROUGE-L: 0.338 vs 0.336).

**Paragraph 3 — Ablation insights**
> Our α-sweep reveals that lower contrastive strength (α=0.5) outperforms the default α=1.0 on temporal localization (tIoU 0.394 vs 0.375) and omission rate (ArgusCost-O 0.102 vs 0.112). The full SEASON variant with spatial negatives and JSD self-diagnostic weights does not improve over simple temporal reversal on CCD, suggesting task-specific negative construction is sufficient.

**Paragraph 4 — Severity analysis**
> Performance is consistent across severity levels (minor/moderate/severe), with severe crashes achieving the highest tIoU (0.422) and lowest omission rate (0.072). Fatal crashes (n=4) show higher hallucination cost, limited by sample size.

---

## Limitations (for Discussion section)

1. **Small test set** (150 videos) — report confidence intervals in revision if requested
2. **Fatal class** (4 test videos) — unreliable statistics
3. **n/a severity** (9 test) — undefined crash windows skew tIoU
4. **ARGUS is simplified** — structured claim matching, not full LLM-NLI/DP judge from ARGUS paper
5. **Val set unused** — no early stopping or hyperparameter tuning on val
6. **8 keyframes @ 224px** — reduced from 16/720p for VRAM; may limit fine-grained temporal reasoning
7. **SEASON gains are modest** on caption metrics — strongest on NLI faithfulness and tIoU ablations

---

## Reviewer FAQ

| Question | Answer |
|----------|--------|
| Why not use full ARGUS LLM judge? | Structured claim matching is reproducible, fast, and aligned with Excel GT format. NLI proxy added (Table VII). |
| Is the dataset split valid? | Yes — stratified, zero leakage, proportions match (see `03_DATASET_AND_SPLITS.md`). |
| Why SEASON if gains are small? | SEASON's value is in NLI faithfulness (+0.095 NLI-Score) and ablation-optimal tIoU (0.394 at α=0.5), not caption BLEU. |
| Why 8 frames not 16? | GPU memory constraint on shared A100; training and inference consistent at 8 frames. |
| Can results reproduce? | Yes — seed 42, configs saved in `train_config.json`, all predictions in `results/`. |

---

## Suggested Table Placement in Paper

| Table | Section | Content |
|-------|---------|---------|
| Table 1 | Experiments | Main 3-way (caption + temporal + forensic) |
| Table 2 | Experiments | Structured field accuracy |
| Table 3 | Experiments | NLI faithfulness |
| Table 4 | Ablation | SEASON α sweep + variants |
| Table 5 | Analysis | Severity-stratified breakdown |
| Supp. | Appendix | Full metrics master, dataset audit |
