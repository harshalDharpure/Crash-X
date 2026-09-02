#!/usr/bin/env python3
"""Standard caption metrics + temporal IoU for CrashX evaluations.

Metrics: BLEU-4, ROUGE-L, METEOR, CIDEr, BERTScore (bert-base-uncased), tIoU.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MetricBundle:
    bleu4: float
    rouge_l: float
    meteor: float
    cider: float
    bertscore_f1: float
    tiou: float

    def as_dict(self) -> dict[str, float]:
        return {
            "BLEU-4": self.bleu4,
            "ROUGE-L": self.rouge_l,
            "METEOR": self.meteor,
            "CIDEr": self.cider,
            "BERTScore": self.bertscore_f1,
            "tIoU": self.tiou,
        }


def parse_timestamp_window(text: str) -> tuple[float | None, float | None]:
    """Extract Start/End seconds from structured or free-form model output."""
    start = end = None
    m_s = re.search(
        r"Start\s*[:=]\s*([-+]?\d*\.?\d+)\s*s?",
        text,
        flags=re.IGNORECASE,
    )
    m_e = re.search(
        r"End\s*[:=]\s*([-+]?\d*\.?\d+)\s*s?",
        text,
        flags=re.IGNORECASE,
    )
    if m_s:
        start = float(m_s.group(1))
    if m_e:
        end = float(m_e.group(1))
    # Alternate patterns: "from Xs to Ys", "[X, Y]"
    if start is None or end is None:
        m = re.search(
            r"(?:from|window)?\s*([-+]?\d*\.?\d+)\s*s?\s*(?:to|-|,)\s*([-+]?\d*\.?\d+)\s*s?",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            start = start if start is not None else float(m.group(1))
            end = end if end is not None else float(m.group(2))
    return start, end


def temporal_iou(
    pred_start: float | None,
    pred_end: float | None,
    gt_start: float,
    gt_end: float,
) -> float:
    """Intersection-over-union of predicted vs GT crash time windows."""
    if pred_start is None or pred_end is None:
        return 0.0
    if pred_end < pred_start:
        pred_start, pred_end = pred_end, pred_start
    if gt_end < gt_start:
        gt_start, gt_end = gt_end, gt_start
    inter = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    if union <= 0:
        return 1.0 if inter == 0 and abs(pred_start - gt_start) < 1e-6 else 0.0
    return float(inter / union)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def bleu4_score(hyps: Sequence[str], refs: Sequence[str]) -> float:
    try:
        import sacrebleu

        return float(sacrebleu.corpus_bleu(list(hyps), [list(refs)]).score) / 100.0
    except Exception as exc:  # noqa: BLE001
        logger.warning("sacrebleu failed (%s); using simple 4-gram precision", exc)
        scores = []
        for h, r in zip(hyps, refs):
            ht, rt = _tokenize(h), _tokenize(r)
            if len(ht) < 4:
                scores.append(0.0)
                continue
            from collections import Counter

            def ngrams(toks: list[str], n: int) -> Counter:
                return Counter(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))

            precisions = []
            for n in range(1, 5):
                hc, rc = ngrams(ht, n), ngrams(rt, n)
                overlap = sum((hc & rc).values())
                total = max(1, sum(hc.values()))
                precisions.append(overlap / total)
            if min(precisions) == 0:
                scores.append(0.0)
            else:
                geo = float(np.exp(np.mean(np.log(np.clip(precisions, 1e-12, 1)))))
                bp = 1.0 if len(ht) >= len(rt) else np.exp(1 - len(rt) / max(1, len(ht)))
                scores.append(bp * geo)
        return float(np.mean(scores)) if scores else 0.0


def rouge_l_score(hyps: Sequence[str], refs: Sequence[str]) -> float:
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        vals = [scorer.score(r, h)["rougeL"].fmeasure for h, r in zip(hyps, refs)]
        return float(np.mean(vals)) if vals else 0.0
    except Exception as exc:  # noqa: BLE001
        logger.warning("rouge-score failed (%s)", exc)
        return 0.0


def meteor_score(hyps: Sequence[str], refs: Sequence[str]) -> float:
    try:
        import evaluate

        metric = evaluate.load("meteor")
        result = metric.compute(predictions=list(hyps), references=list(refs))
        return float(result["meteor"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("METEOR unavailable (%s); returning 0.0", exc)
        return 0.0


def cider_score(hyps: Sequence[str], refs: Sequence[str]) -> float:
    """CIDEr via pycocoevalcap if present; otherwise TF-IDF n-gram cosine proxy."""
    try:
        from pycocoevalcap.cider.cider import Cider

        gts = {i: [r] for i, r in enumerate(refs)}
        res = {i: [h] for i, h in enumerate(hyps)}
        score, _ = Cider().compute_score(gts, res)
        return float(score)
    except Exception:
        pass
    # Lightweight CIDEr-ish proxy: average 1-4gram TF cosine
    from collections import Counter

    def tf(toks: list[str]) -> Counter:
        c: Counter = Counter()
        for n in range(1, 5):
            for i in range(len(toks) - n + 1):
                c[" ".join(toks[i : i + n])] += 1
        return c

    def cos(a: Counter, b: Counter) -> float:
        keys = set(a) | set(b)
        if not keys:
            return 0.0
        va = np.array([a[k] for k in keys], dtype=np.float64)
        vb = np.array([b[k] for k in keys], dtype=np.float64)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(va.dot(vb) / denom) if denom > 0 else 0.0

    scores = [cos(tf(_tokenize(h)), tf(_tokenize(r))) for h, r in zip(hyps, refs)]
    return float(np.mean(scores)) if scores else 0.0


def bertscore_f1(
    hyps: Sequence[str],
    refs: Sequence[str],
    model_type: str = "bert-base-uncased",
    batch_size: int = 16,
) -> float:
    try:
        from bert_score import score as bert_score_fn

        _, _, f1 = bert_score_fn(
            list(hyps),
            list(refs),
            model_type=model_type,
            verbose=False,
            batch_size=batch_size,
            lang="en",
        )
        return float(f1.mean().item())
    except Exception as exc:  # noqa: BLE001
        logger.warning("BERTScore failed (%s); returning 0.0", exc)
        return 0.0


def extract_explanation(text: str) -> str:
    """Pull Explanation field from structured output, else return full text."""
    m = re.search(r"Explanation\s*[:=]\s*(.+)$", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def compute_corpus_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
    gt_starts: Sequence[float],
    gt_ends: Sequence[float],
    use_explanation_only: bool = True,
) -> MetricBundle:
    """Aggregate corpus-level metrics for a list of pred/ref pairs."""
    assert len(predictions) == len(references) == len(gt_starts) == len(gt_ends)

    if use_explanation_only:
        hyps = [extract_explanation(p) for p in predictions]
        refs = [extract_explanation(r) for r in references]
    else:
        hyps = list(predictions)
        refs = list(references)

    tiou_vals = []
    for pred, gs, ge in zip(predictions, gt_starts, gt_ends):
        ps, pe = parse_timestamp_window(pred)
        tiou_vals.append(temporal_iou(ps, pe, gs, ge))

    return MetricBundle(
        bleu4=bleu4_score(hyps, refs),
        rouge_l=rouge_l_score(hyps, refs),
        meteor=meteor_score(hyps, refs),
        cider=cider_score(hyps, refs),
        bertscore_f1=bertscore_f1(hyps, refs),
        tiou=float(np.mean(tiou_vals)) if tiou_vals else 0.0,
    )
