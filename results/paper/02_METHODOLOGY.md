# 02 — Methodology

## 2.1 Base Model

| Parameter | Value |
|-----------|-------|
| Backbone | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Quantization | 4-bit (bitsandbytes) |
| Adapter | LoRA (PEFT) |

## 2.2 Fine-tuning (CrashLogic-7B)

| Hyperparameter | Value |
|----------------|-------|
| Method | QLoRA SFT |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Learning rate | 2×10⁻⁴ |
| Epochs | 5 |
| Batch size | 1 (effective 8 via grad accumulation) |
| Train videos | 1,198 |
| Keyframes | 8 per video |
| Max frame side | 224 px |
| Max sequence length | 768 tokens |
| Final train loss | ~3.09 |

**Output format (structured):**
```
Severity: {minor|moderate|severe|fatal} | Impact: {locations} | Start: {t}s | End: {t}s |
Vehicles: {list} | NumVehicles: {n} | Weather: {condition} | Explanation: {text}
```

## 2.3 SEASON Contrastive Decoding

**Default (hybrid PoC):** temporal negative via frame reversal:
```
logits_final = (1 + α) · logits(V_original) − α · logits(V_reversed)
```

| Parameter | Default | Ablation range |
|-----------|---------|----------------|
| α (contrastive strength) | 1.0 | 0.5, 1.0, 1.5, 2.0 |
| Negative mode | reverse | reverse, shuffle |
| Full SEASON | False | True (homogenization + spatial noise + JSD weights) |

**α=0** reduces to standard greedy decoding.

## 2.4 Inference Settings

| Setting | Value |
|---------|-------|
| Test videos | 150 |
| Keyframes | 8 |
| Max new tokens | 256 |
| Max frame side | 224 px |
| Decode | Greedy or SEASON (no sampling) |

## 2.5 Evaluation Metrics

### Lexical (Explanation field)
| Metric | Direction | Description |
|--------|-----------|-------------|
| BLEU-4 | ↑ | 4-gram overlap (sacrebleu) |
| ROUGE-L | ↑ | Longest common subsequence F1 |
| METEOR | ↑ | Synonym-aware alignment |
| CIDEr | ↑ | TF-IDF weighted n-gram consensus |
| BERTScore | ↑ | Contextual embedding similarity (bert-base-uncased) |

### Temporal
| Metric | Direction | Description |
|--------|-----------|-------------|
| tIoU | ↑ | Mean temporal IoU of predicted vs GT crash windows |
| THR@0.25 | ↑ | Fraction of videos with tIoU ≥ 0.25 |
| THR@0.50 | ↑ | Fraction of videos with tIoU ≥ 0.50 |

### Forensic (CrashX-adapted ARGUS)
| Metric | Direction | Description |
|--------|-----------|-------------|
| ArgusCost-H | ↓ | Hallucination rate — predicted claims conflicting with GT |
| ArgusCost-O | ↓ | Omission rate — critical GT fields missing in prediction |

Structured claim matching against Excel GT fields (severity, impact, vehicles, weather, timestamps, explanation). Not full LLM-NLI/DP judge.

### NLI Faithfulness
| Metric | Direction | Description |
|--------|-----------|-------------|
| NLI-Entailment | ↑ | P(entailment) via cross-encoder/nli-deberta-v3-small |
| NLI-Contradiction | ↓ | P(contradiction) — hallucination proxy |
| NLI-Score | ↑ | P(entail) − P(contradict) |
| NLI-Loss | ↓ | −log P(entail) |

GT forensic text = premise; model explanation = hypothesis.

### Structured Field Accuracy
| Metric | Description |
|--------|-------------|
| Severity-Acc | Exact match rate for severity label |
| Impact-Jaccard | Token overlap for impact locations |
| Vehicle-Jaccard | Token overlap for vehicle types/colors |
| Weather-Jaccard | Token overlap for weather conditions |
| Timestamp-ParseRate | Fraction with parseable Start/End seconds |
