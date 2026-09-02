# 10 — Table VII: NLI Faithfulness & Hallucination

**Purpose:** NLI-based faithfulness scoring — measures whether model explanations are entailed by ground-truth forensic text.

**Model:** `cross-encoder/nli-deberta-v3-small`  
**Setup:** GT forensic text = premise; model output = hypothesis

---

## Results (Explanation-level)

| Method | NLI-Entail ↑ | NLI-Contradict ↓ | NLI-Score ↑ | NLI-Loss ↓ |
|--------|-------------|------------------|-------------|------------|
| Zero-shot Qwen2.5-VL-7B | 0.139 | **0.092** | 0.047 | 3.811 |
| CrashLogic-7B (Greedy) | 0.224 | 0.229 | -0.005 | 3.124 |
| **CrashLogic-7B + SEASON** | **0.276** | 0.186 | **0.090** | **2.954** |

## Results (Full structured output)

| Method | Full-NLI-Entail ↑ | Full-NLI-Contradict ↓ | Full-NLI-Score ↑ | Full-NLI-Loss ↓ |
|--------|-------------------|----------------------|------------------|-----------------|
| Zero-shot | 0.128 | **0.052** | 0.076 | 3.608 |
| **CrashLogic Greedy** | **0.402** | 0.155 | **0.247** | **2.006** |
| CrashLogic + SEASON | 0.386 | 0.149 | 0.237 | 2.075 |

---

## Best per Metric

| Metric | Best Method | Value |
|--------|-------------|-------|
| NLI-Entailment (explanation) | **SEASON** | **0.276** |
| NLI-Contradiction (explanation) | Zero-shot | 0.092 |
| NLI-Score (explanation) | **SEASON** | **0.090** |
| NLI-Loss (explanation) | **SEASON** | **2.954** |
| Full-NLI-Entailment | **Greedy** | **0.402** |
| Full-NLI-Loss | **Greedy** | **2.006** |

---

## Analysis

### Where SEASON wins
- **NLI-Score +0.090** (SEASON) vs -0.005 (Greedy) — SEASON produces explanations more faithful to GT forensic facts.
- **NLI-Entailment +23%** over Greedy (0.276 vs 0.224).
- **NLI-Loss −5%** lower than Greedy (2.954 vs 3.124) — stronger entailment confidence.

### Where Greedy wins
- **Full-output NLI** — structured fields (severity, impact, etc.) are more entailed by GT when using greedy decode.
- SEASON's contrastive perturbation can introduce minor field variations that reduce full-output entailment.

### Interpretation for paper
> SEASON improves **explanation-level faithfulness** (NLI-Score +0.095 over Greedy) while maintaining comparable structured field accuracy. This supports the claim that SEASON mitigates temporal/explanatory hallucination without degrading forensic structure.

---

## LaTeX Caption

> *Table VII: NLI faithfulness via DeBERTa-v3 NLI cross-encoder. NLI-Score = P(entail) − P(contradict); NLI-Loss = −log P(entail).*
