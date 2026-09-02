#!/usr/bin/env python3
"""CrashX-adapted ARGUS dual-cost metrics (structured claim matching).

ArgusCost-H: fraction of predicted forensic claims that conflict with Excel GT.
ArgusCost-O: fraction of critical GT fields omitted / unmatched in the prediction.

Also reports BERTScore F1 between predicted and GT Explanation as a free-form
proxy (no LLM-NLI / DP judge).

Lower ArgusCost-* is better; higher explanation BERTScore is better.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from crashx.eval.metrics import (
    bertscore_f1,
    extract_explanation,
    parse_timestamp_window,
    temporal_iou,
)

SEVERITY_SET = {"minor", "moderate", "severe", "fatal", "n/a"}
WEATHER_SET = {
    "normal",
    "sunny",
    "rainy",
    "snowy",
    "cloudy",
    "mist",
    "foggy",
    "night",
    "unknown",
}


@dataclass
class ArgusSampleResult:
    argus_cost_h: float
    argus_cost_o: float
    explanation_bertscore: float
    details: dict[str, Any]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _field(text: str, name: str) -> str | None:
    """Extract `Name: value` from pipe-delimited structured text."""
    pat = rf"{name}\s*[:=]\s*([^|]+)"
    m = re.search(pat, text, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip()


def extract_predicted_claims(pred_text: str) -> dict[str, Any]:
    """Parse forensic claims from model output."""
    severity = _field(pred_text, "Severity")
    impact = _field(pred_text, "Impact")
    vehicles = _field(pred_text, "Vehicles")
    weather = _field(pred_text, "Weather")
    start, end = parse_timestamp_window(pred_text)
    explanation = extract_explanation(pred_text)

    # Fallback keyword severity if unstructured
    if severity is None:
        for s in ("fatal", "severe", "moderate", "minor"):
            if re.search(rf"\b{s}\b", pred_text, flags=re.IGNORECASE):
                severity = s
                break

    if weather is None:
        for w in ("snowy", "rainy", "cloudy", "mist", "foggy", "night", "normal", "sunny"):
            if re.search(rf"\b{w}\b", pred_text, flags=re.IGNORECASE):
                weather = w
                break

    return {
        "severity": _norm(severity) if severity else None,
        "impact": _impact_tokens(impact) if impact else _impact_tokens(pred_text),
        "vehicles": _vehicle_tokens(vehicles) if vehicles else _vehicle_tokens(pred_text),
        "weather": _weather_tokens(weather) if weather else _weather_tokens(pred_text),
        "start": start,
        "end": end,
        "explanation": explanation,
        "raw_impact": _norm(impact) if impact else None,
    }


def _impact_tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    t = _norm(text)
    keys = set()
    for k in (
        "front",
        "rear",
        "left",
        "right",
        "side",
        "corner",
        "front-left",
        "front-right",
        "rear-left",
        "rear-right",
        "front-end",
        "rear-end",
    ):
        if k in t:
            keys.add(k)
    # also capture hyphenated compounds present in GT
    for m in re.findall(r"[a-z]+-[a-z]+(?:-[a-z]+)?", t):
        keys.add(m)
    return keys


def _vehicle_tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    t = _norm(text)
    colors = {
        "black",
        "white",
        "silver",
        "gray",
        "grey",
        "red",
        "blue",
        "yellow",
        "orange",
        "green",
        "brown",
    }
    types = {"car", "truck", "bus", "van", "suv", "motorcycle", "bike", "scooter", "pedestrian"}
    toks = set(re.findall(r"[a-z]+", t))
    return (toks & colors) | (toks & types) | {w for w in toks if w.endswith("car") or w.endswith("truck")}


def _weather_tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    t = _norm(text)
    found = set()
    for w in WEATHER_SET:
        if w in t:
            found.add(w)
    return found


def _claim_conflicts(pred: dict[str, Any], gt: dict[str, Any]) -> list[str]:
    """Return list of hallucinated / conflicting claim names present in prediction."""
    conflicts: list[str] = []

    if pred["severity"]:
        gt_sev = _norm(gt.get("severity", ""))
        if pred["severity"] in SEVERITY_SET and gt_sev and pred["severity"] != gt_sev:
            conflicts.append("severity")

    if pred["weather"]:
        gt_w = _weather_tokens(gt.get("weather", ""))
        # conflict if pred asserts weather disjoint from GT
        if gt_w and pred["weather"].isdisjoint(gt_w):
            conflicts.append("weather")

    if pred["impact"]:
        gt_i = _impact_tokens(gt.get("impact", ""))
        if gt_i and pred["impact"].isdisjoint(gt_i):
            conflicts.append("impact")

    if pred["vehicles"]:
        gt_v = _vehicle_tokens(gt.get("vehicles", ""))
        if gt_v and pred["vehicles"].isdisjoint(gt_v):
            conflicts.append("vehicles")

    # Timestamp hallucination: predicted window with near-zero overlap
    if pred["start"] is not None and pred["end"] is not None:
        iou = temporal_iou(
            pred["start"],
            pred["end"],
            float(gt["start_sec"]),
            float(gt["end_sec"]),
        )
        if iou < 0.1:
            conflicts.append("timestamp")

    return conflicts


def _omissions(pred: dict[str, Any], gt: dict[str, Any]) -> list[str]:
    """Return list of critical GT fields missing / unmatched in prediction."""
    omitted: list[str] = []

    gt_sev = _norm(gt.get("severity", ""))
    if gt_sev and gt_sev not in ("n/a",):
        if not pred["severity"] or pred["severity"] != gt_sev:
            # if severity present but wrong, counted in H; still an omission of correct claim
            if not pred["severity"]:
                omitted.append("severity")

    gt_i = _impact_tokens(gt.get("impact", ""))
    if gt_i and (not pred["impact"] or pred["impact"].isdisjoint(gt_i)):
        omitted.append("impact")

    gt_v = _vehicle_tokens(gt.get("vehicles", ""))
    if gt_v and (not pred["vehicles"] or len(pred["vehicles"] & gt_v) == 0):
        omitted.append("vehicles")

    gt_w = _weather_tokens(gt.get("weather", ""))
    if gt_w and (not pred["weather"] or pred["weather"].isdisjoint(gt_w)):
        omitted.append("weather")

    if pred["start"] is None or pred["end"] is None:
        omitted.append("timestamp")
    else:
        iou = temporal_iou(
            pred["start"],
            pred["end"],
            float(gt["start_sec"]),
            float(gt["end_sec"]),
        )
        if iou < 0.25:
            omitted.append("timestamp_accuracy")

    # Explanation omission: empty / trivially short
    if len((pred.get("explanation") or "").split()) < 8:
        omitted.append("explanation")

    return omitted


def score_sample(
    pred_text: str,
    gt: dict[str, Any],
    explanation_bert: float | None = None,
) -> ArgusSampleResult:
    """Compute ArgusCost-H / ArgusCost-O for one prediction vs Excel GT row."""
    pred = extract_predicted_claims(pred_text)

    # Predicted claim inventory (only fields the model actually asserted)
    asserted = []
    if pred["severity"]:
        asserted.append("severity")
    if pred["weather"]:
        asserted.append("weather")
    if pred["impact"]:
        asserted.append("impact")
    if pred["vehicles"]:
        asserted.append("vehicles")
    if pred["start"] is not None and pred["end"] is not None:
        asserted.append("timestamp")

    conflicts = _claim_conflicts(pred, gt)
    # H = conflicting asserted claims / asserted claims (or 0 if nothing asserted)
    if asserted:
        # conflicts that are among asserted
        h = len([c for c in conflicts if c in asserted or c == "timestamp"]) / len(asserted)
    else:
        # asserting nothing forensically is treated as full hallucination cost of 1.0
        h = 1.0

    critical_gt = []
    if _norm(gt.get("severity", "")) not in ("", "n/a"):
        critical_gt.append("severity")
    if _impact_tokens(gt.get("impact", "")):
        critical_gt.append("impact")
    if _vehicle_tokens(gt.get("vehicles", "")):
        critical_gt.append("vehicles")
    if _weather_tokens(gt.get("weather", "")):
        critical_gt.append("weather")
    critical_gt.append("timestamp")
    if (gt.get("explanation") or "").strip():
        critical_gt.append("explanation")

    omitted = _omissions(pred, gt)
    # Map timestamp_accuracy → timestamp for omission accounting
    om_norm = []
    for o in omitted:
        om_norm.append("timestamp" if o.startswith("timestamp") else o)
    om_norm = list(dict.fromkeys(om_norm))
    o = len([x for x in om_norm if x in critical_gt]) / max(1, len(critical_gt))

    if explanation_bert is None:
        explanation_bert = 0.0

    return ArgusSampleResult(
        argus_cost_h=float(min(1.0, max(0.0, h))),
        argus_cost_o=float(min(1.0, max(0.0, o))),
        explanation_bertscore=float(explanation_bert),
        details={
            "asserted": asserted,
            "conflicts": conflicts,
            "omitted": omitted,
            "critical_gt": critical_gt,
            "pred_claims": {k: (list(v) if isinstance(v, set) else v) for k, v in pred.items()},
        },
    )


def evaluate_argus(
    predictions: Sequence[str],
    gt_rows: Sequence[dict[str, Any]],
    compute_explanation_bertscore: bool = True,
) -> dict[str, Any]:
    """Corpus-level ArgusCost-H / ArgusCost-O + explanation BERTScore proxy."""
    assert len(predictions) == len(gt_rows)

    if compute_explanation_bertscore:
        hyp_e = [extract_explanation(p) for p in predictions]
        ref_e = [extract_explanation(g.get("explanation", "")) for g in gt_rows]
        # per-sample approximate: use corpus mean then assign; also compute per-pair if small
        try:
            from bert_score import score as bert_score_fn

            _, _, f1 = bert_score_fn(
                hyp_e,
                ref_e,
                model_type="bert-base-uncased",
                verbose=False,
                lang="en",
            )
            per_f1 = [float(x) for x in f1.tolist()]
        except Exception:
            mean_f1 = bertscore_f1(hyp_e, ref_e)
            per_f1 = [mean_f1] * len(predictions)
    else:
        per_f1 = [0.0] * len(predictions)

    samples = []
    for pred, gt, f1 in zip(predictions, gt_rows, per_f1):
        samples.append(score_sample(pred, gt, explanation_bert=f1))

    n = max(1, len(samples))
    return {
        "ArgusCost-H": sum(s.argus_cost_h for s in samples) / n,
        "ArgusCost-O": sum(s.argus_cost_o for s in samples) / n,
        "Explanation-BERTScore": sum(s.explanation_bertscore for s in samples) / n,
        "n": len(samples),
        "samples": samples,
    }
