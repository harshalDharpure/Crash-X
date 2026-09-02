#!/usr/bin/env python3
"""Per-video metric vectors for bootstrap CIs and paired tests."""

from __future__ import annotations

from typing import Any

import numpy as np

from crashx.eval.argus_eval import _norm, extract_predicted_claims, score_sample
from crashx.eval.metrics import (
    bleu4_score,
    extract_explanation,
    rouge_l_score,
    temporal_iou,
)


def per_video_tiou(output: dict[str, Any]) -> float:
    pred = extract_predicted_claims(output["prediction"])
    if pred["start"] is None or pred["end"] is None:
        return 0.0
    return temporal_iou(
        pred["start"],
        pred["end"],
        float(output["start_sec"]),
        float(output["end_sec"]),
    )


def per_video_argus(output: dict[str, Any]) -> tuple[float, float]:
    gt = {
        "severity": output["severity"],
        "impact": output["impact"],
        "start_sec": output["start_sec"],
        "end_sec": output["end_sec"],
        "vehicles": output["vehicles"],
        "weather": output["weather"],
        "explanation": output["explanation"],
    }
    s = score_sample(output["prediction"], gt)
    return s.argus_cost_h, s.argus_cost_o


def per_video_severity_acc(output: dict[str, Any]) -> float:
    pred = extract_predicted_claims(output["prediction"])
    gt_sev = _norm(output["severity"])
    if pred["severity"] and gt_sev and pred["severity"] == gt_sev:
        return 1.0
    return 0.0


def per_video_bleu4(output: dict[str, Any]) -> float:
    hyp = extract_explanation(output["prediction"])
    ref = extract_explanation(output["reference"])
    return bleu4_score([hyp], [ref])


def per_video_rouge_l(output: dict[str, Any]) -> float:
    hyp = extract_explanation(output["prediction"])
    ref = extract_explanation(output["reference"])
    return rouge_l_score([hyp], [ref])


def extract_per_sample_vectors(
    outputs: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Build aligned per-video score vectors keyed by metric name."""
    tiou = []
    argus_h = []
    argus_o = []
    sev = []
    bleu = []
    rouge = []
    video_ids = []

    for o in outputs:
        video_ids.append(o["video_id"])
        tiou.append(per_video_tiou(o))
        h, om = per_video_argus(o)
        argus_h.append(h)
        argus_o.append(om)
        sev.append(per_video_severity_acc(o))
        bleu.append(per_video_bleu4(o))
        rouge.append(per_video_rouge_l(o))

    return {
        "video_id": np.array(video_ids),
        "tIoU": np.asarray(tiou, dtype=np.float64),
        "ArgusCost-H": np.asarray(argus_h, dtype=np.float64),
        "ArgusCost-O": np.asarray(argus_o, dtype=np.float64),
        "Severity-Acc": np.asarray(sev, dtype=np.float64),
        "BLEU-4": np.asarray(bleu, dtype=np.float64),
        "ROUGE-L": np.asarray(rouge, dtype=np.float64),
    }


def align_paired_vectors(
    vec_a: dict[str, np.ndarray],
    vec_b: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Align two per-sample dicts by video_id."""
    ids_a = {vid: i for i, vid in enumerate(vec_a["video_id"])}
    ids_b = {vid: i for i, vid in enumerate(vec_b["video_id"])}
    common = sorted(set(ids_a) & set(ids_b))
    if not common:
        raise ValueError("No overlapping video_ids between conditions")

    out_a: dict[str, np.ndarray] = {}
    out_b: dict[str, np.ndarray] = {}
    for key in vec_a:
        if key == "video_id":
            out_a[key] = np.array(common)
            out_b[key] = np.array(common)
            continue
        out_a[key] = np.array([vec_a[key][ids_a[v]] for v in common], dtype=np.float64)
        out_b[key] = np.array([vec_b[key][ids_b[v]] for v in common], dtype=np.float64)
    return out_a, out_b
