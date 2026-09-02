# 01 — Research Overview

## Problem Statement

Traffic accident video understanding requires **spatiotemporal reasoning**: models must localize crash events in time, identify involved vehicles and impact locations, and produce forensic explanations aligned with ground truth. General-purpose Video-LLMs suffer from:

1. **Temporal hallucination** — incorrect or missing crash timestamps
2. **Unstructured outputs** — free-form prose instead of forensic fields
3. **High omission rates** — missing severity, vehicles, weather, impact

**CrashX** addresses these by combining domain fine-tuning (CrashLogic-7B) with **SEASON** contrastive decoding at inference.

---

## Research Contributions

| # | Contribution | Evidence |
|---|-------------|----------|
| 1 | **CrashLogic-7B** — QLoRA fine-tuned Qwen2.5-VL-7B for structured crash captioning on CCD | BLEU-4: 0.016 → 0.142; tIoU: 0.012 → 0.373 |
| 2 | **CrashX benchmark protocol** — 3-way comparison (zero-shot → FT → FT+SEASON) on 150-video test split | Tables I–VII |
| 3 | **Structured ARGUS metrics** — ArgusCost-H (hallucination) and ArgusCost-O (omission) vs Excel GT | ArgusCost-O: 0.462 → 0.107 |
| 4 | **SEASON integration** — hybrid temporal negative decoding for hallucination mitigation | NLI-Score: -0.005 → 0.090 |
| 5 | **Comprehensive ablation** — α sweep, shuffle vs reverse, full SEASON | Table VI |

---

## Pipeline Architecture

```
Car Crash Dataset (CCD)
  ├── 1,498 videos + Excel GT
  ├── Stratified splits: 1198 train / 150 val / 150 test
  │
  ▼
QLoRA Fine-tuning (CrashLogic-7B)
  ├── Base: Qwen2.5-VL-7B-Instruct (4-bit)
  ├── LoRA r=16, α=32, 5 epochs
  └── Output: outputs/crashlogic_7b_lora/
  │
  ▼
Inference (3 conditions + 5 ablations)
  ├── Greedy decoding
  ├── SEASON: logits = (1+α)·logits(V) − α·logits(V_neg)
  └── 8 keyframes @ 224px, max 256 tokens
  │
  ▼
Evaluation
  ├── Lexical: BLEU-4, ROUGE-L, METEOR, CIDEr, BERTScore
  ├── Temporal: tIoU, THR@0.25, THR@0.50
  ├── Forensic: ArgusCost-H, ArgusCost-O
  ├── NLI: Entailment, Contradiction, NLI-Score, NLI-Loss
  └── Structured fields: Severity-Acc, Impact/Vehicle/Weather Jaccard
```

---

## Models Compared

| Model | Description |
|-------|-------------|
| **Zero-shot Qwen2.5-VL-7B** | Off-the-shelf baseline, no domain adaptation |
| **CrashLogic-7B (Greedy)** | Our fine-tuned model, standard greedy generation |
| **CrashLogic-7B + SEASON** | Fine-tuned + SEASON contrastive decoding (α=1.0) |

---

## Key References

- **VRU-Accident** (Kim et al., ICCV 2025) — video QA / dense captioning baseline pattern
- **SEASON** (Wu et al., CVPR 2026) — self-diagnostic contrastive decoding for temporal hallucination
- **ARGUS** (ICCV 2025) — hallucination and omission evaluation for Video-LLMs
- **CCD** — Car Crash Dataset, 1,500 videos with Excel forensic ground truth

---

## Hardware & Runtime

| Stage | GPU | Time |
|-------|-----|------|
| QLoRA training (5 epochs) | A100 40GB | ~3.5 hours |
| Greedy inference (150 videos) | A100 | ~32 min |
| SEASON inference (150 videos) | A100 | ~2 hours |
| Full ablation suite (5 × 150) | A100 GPU 4 | ~15 hours |
