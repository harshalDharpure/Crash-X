#!/usr/bin/env python3
"""NLI-based faithfulness scoring for CrashX (ARGUS-style entailment proxy).

Scores whether model explanations/claims are entailed by ground-truth forensic
text using a pretrained NLI cross-encoder (DeBERTa-v3-small).

Metrics (higher entail / lower contradiction is better):
  - NLI-Entailment: mean P(entailment)
  - NLI-Contradiction: mean P(contradiction)  [hallucination proxy]
  - NLI-Score: mean P(entailment) - P(contradiction)
  - NLI-Loss: mean -log(P(entailment) + eps)   [lower is better]
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import torch

from crashx.eval.metrics import extract_explanation

logger = logging.getLogger(__name__)

DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
# DeBERTa NLI label order
LABEL2ID = {"contradiction": 0, "entailment": 1, "neutral": 2}


class NLIScorer:
  """Batched NLI scorer with lazy model load."""

  def __init__(
      self,
      model_name: str = DEFAULT_NLI_MODEL,
      device: str | None = None,
      batch_size: int = 16,
  ) -> None:
      self.model_name = model_name
      self.batch_size = batch_size
      if device is None:
          device = "cuda" if torch.cuda.is_available() else "cpu"
      self.device = device
      self._tokenizer = None
      self._model = None

  def _load(self) -> None:
      if self._model is not None:
          return
      from transformers import AutoModelForSequenceClassification, AutoTokenizer

      logger.info("Loading NLI model %s on %s", self.model_name, self.device)
      self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
      self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
      self._model.to(self.device)
      self._model.eval()

  @torch.inference_mode()
  def score_pairs(
      self,
      premises: Sequence[str],
      hypotheses: Sequence[str],
  ) -> list[dict[str, float]]:
      """Return per-pair NLI probabilities."""
      assert len(premises) == len(hypotheses)
      self._load()
      assert self._tokenizer is not None and self._model is not None

      out: list[dict[str, float]] = []
      for i in range(0, len(premises), self.batch_size):
          batch_p = list(premises[i : i + self.batch_size])
          batch_h = list(hypotheses[i : i + self.batch_size])
          enc = self._tokenizer(
              batch_p,
              batch_h,
              padding=True,
              truncation=True,
              max_length=512,
              return_tensors="pt",
          )
          enc = {k: v.to(self.device) for k, v in enc.items()}
          logits = self._model(**enc).logits
          probs = torch.softmax(logits, dim=-1).cpu()
          for row in probs:
              p_con = float(row[LABEL2ID["contradiction"]])
              p_ent = float(row[LABEL2ID["entailment"]])
              p_neu = float(row[LABEL2ID["neutral"]])
              out.append(
                  {
                      "NLI-Contradiction": p_con,
                      "NLI-Entailment": p_ent,
                      "NLI-Neutral": p_neu,
                      "NLI-Score": p_ent - p_con,
                      "NLI-Loss": float(-torch.log(torch.tensor(p_ent + 1e-8))),
                  }
              )
      return out

  def unload(self) -> None:
      self._model = None
      self._tokenizer = None
      if torch.cuda.is_available():
          torch.cuda.empty_cache()


def _gt_premise(gt: dict[str, Any]) -> str:
    """Build GT forensic premise for NLI."""
    parts = [
        f"Crash severity is {gt.get('severity', 'unknown')}.",
        f"Impact location: {gt.get('impact', 'unknown')}.",
        f"Vehicles involved: {gt.get('vehicles', 'unknown')}.",
        f"Weather: {gt.get('weather', 'unknown')}.",
        f"Crash window from {gt.get('start_sec', '?')}s to {gt.get('end_sec', '?')}s.",
        extract_explanation(str(gt.get("explanation", ""))),
    ]
    return " ".join(p for p in parts if p.strip())


def evaluate_nli(
    predictions: Sequence[str],
    gt_rows: Sequence[dict[str, Any]],
    scorer: NLIScorer | None = None,
) -> dict[str, Any]:
    """Corpus-level NLI metrics: explanation + full structured prediction."""
    assert len(predictions) == len(gt_rows)
    own_scorer = scorer is None
    if scorer is None:
        scorer = NLIScorer()

    premises_exp, hyps_exp = [], []
    premises_full, hyps_full = [], []
    for pred, gt in zip(predictions, gt_rows):
        gt_p = _gt_premise(gt)
        premises_exp.append(gt_p)
        hyps_exp.append(extract_explanation(pred))
        premises_full.append(gt_p)
        hyps_full.append(pred.strip())

    exp_scores = scorer.score_pairs(premises_exp, hyps_exp)
    full_scores = scorer.score_pairs(premises_full, hyps_full)

    def _aggregate(scores: list[dict[str, float]], prefix: str) -> dict[str, float]:
        if not scores:
            return {}
        keys = ["NLI-Entailment", "NLI-Contradiction", "NLI-Score", "NLI-Loss"]
        return {f"{prefix}{k}" if prefix else k: sum(s[k] for s in scores) / len(scores) for k in keys}

    metrics = _aggregate(exp_scores, "")
    metrics.update(_aggregate(full_scores, "Full-"))

    if own_scorer:
        scorer.unload()

    return {
        **metrics,
        "per_sample_explanation": exp_scores,
        "per_sample_full": full_scores,
    }
