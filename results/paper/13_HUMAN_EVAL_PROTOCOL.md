# 13 — Human Evaluation Protocol (Phase 4)

**Purpose:** Validate forensic metric alignment with human judgment for A* submission (CVPR / ICCV / NeurIPS D&B).

---

## Study Design

| Parameter | Value |
|-----------|-------|
| Videos | 50 (stratified: 15 minor, 15 moderate, 12 severe, 4 fatal, 4 n/a) |
| Annotators | 3 (independent) |
| Conditions shown | Blind — annotators see video + model output only |
| Models compared | 4 foundation zero-shot + CrashLogic Greedy + CrashLogic TCD |

---

## Annotation Tasks

### Task 1 — Temporal Correctness (Likert 1–5)

> *Does the predicted crash time window (Start–End) align with when the collision occurs in the video?*

| Score | Definition |
|-------|------------|
| 1 | Completely wrong window or no timestamp |
| 2 | Partial overlap, mostly incorrect |
| 3 | Rough overlap (~50%) |
| 4 | Good overlap, minor offset |
| 5 | Accurate window |

### Task 2 — Forensic Faithfulness (Likert 1–5)

> *Are the stated severity, vehicles, impact points, and weather faithful to the video (no hallucinated claims)?*

| Score | Definition |
|-------|------------|
| 1 | Mostly hallucinated / fabricated |
| 2 | Several major errors |
| 3 | Mixed correct and incorrect |
| 4 | Mostly correct, minor errors |
| 5 | Fully faithful |

### Task 3 — Explanation Quality (Likert 1–5)

> *Does the causal explanation accurately describe what happened without inventing events?*

---

## Inter-Annotator Agreement

Report **Fleiss' κ** per task (target κ ≥ 0.65).

```python
# Example with statsmodels or simple implementation
from sklearn.metrics import cohen_kappa_score
# Pairwise κ averaged across 3 annotators
```

---

## Success Criteria (A* Defense)

| Criterion | Target |
|-----------|--------|
| Human temporal score correlates with tIoU | Spearman ρ ≥ 0.55 |
| Human faithfulness correlates with NLI-Score | Spearman ρ ≥ 0.45 |
| CrashLogic TCD preferred over Greedy (faithfulness) | Significant (p < 0.05, Wilcoxon) |
| Foundation models score < 2.5 on average | Clear failure narrative |

---

## Sampling Script

```bash
PYTHONPATH=. python -m crashx.eval.human_eval_sampler \
  --test-jsonl crashx/data/splits/test.jsonl \
  --out results/human_eval/sample_50.jsonl \
  --seed 42
```

---

## Annotation Template Fields

```json
{
  "video_id": "000542",
  "model_condition": "blind_A",
  "temporal_likert": 4,
  "faithfulness_likert": 3,
  "explanation_likert": 4,
  "annotator_id": "A1",
  "notes": ""
}
```

---

## Timeline

| Week | Activity |
|------|----------|
| Week 3 | Sample 50 videos, prepare blinded outputs |
| Week 4 | 3 annotators complete ratings (≈2 hours each) |
| Week 4 | Compute κ, correlations, add to paper Table X |
