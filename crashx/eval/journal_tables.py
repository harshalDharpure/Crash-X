#!/usr/bin/env python3
"""Publication-style multi-table analysis for CrashX journal experiments."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from crashx.eval.argus_eval import (
    _impact_tokens,
    _norm,
    _vehicle_tokens,
    _weather_tokens,
    evaluate_argus,
    extract_predicted_claims,
)
from crashx.eval.nli_eval import NLIScorer, evaluate_nli
from crashx.eval.bootstrap_stats import bootstrap_ci, paired_wilcoxon
from crashx.eval.metrics import (
    bertscore_f1,
    bleu4_score,
    cider_score,
    compute_corpus_metrics,
    extract_explanation,
    meteor_score,
    parse_timestamp_window,
    rouge_l_score,
    temporal_iou,
)
from crashx.eval.per_sample_scores import align_paired_vectors, extract_per_sample_vectors

# Display names for paper tables (Option A: Benchmark + Diagnostic Framework)
MODEL_LABELS = {
    "ZeroShot-Qwen2.5-VL-7B": "Zero-shot Qwen2.5-VL-7B",
    "ZeroShot-Qwen2.5-VL-3B": "Zero-shot Qwen2.5-VL-3B",
    "ZeroShot-Qwen2-VL-2B": "Zero-shot Qwen2-VL-2B",
    "ZeroShot-InternVL2.5-8B": "Zero-shot InternVL2-8B (disabled)",
    "ZeroShot-LLaVA-Video-7B": "Zero-shot LLaVA-NeXT-Video-7B",
    "ZeroShot-GPT-4o": "Zero-shot GPT-4o",
    "ZeroShot-Gemini-1.5-Pro": "Zero-shot Gemini-1.5-Pro",
    "CrashLogic-7B-Greedy": "CrashLogic-7B (Greedy)",
    "CrashLogic-7B-Greedy-f16": "CrashLogic-7B (Greedy, 16f)",
    "CrashLogic-7B-SEASON": "CrashLogic-7B + SEASON ($\\alpha$=1.0)",
    "CrashLogic-7B-SEASON-a0.5": "CrashLogic-7B + TCD ($\\alpha$=0.5)",
    "CrashLogic-7B-SEASON-a0.5-f16": "CrashLogic-7B + TCD ($\\alpha$=0.5, 16f)",
    "CrashLogic-7B-SEASON-Full": "CrashLogic-7B + SEASON (Full)",
    "CrashLogic-7B-SEASON-a1.0": "SEASON ($\\alpha$=1.0)",
    "CrashLogic-7B-SEASON-a1.5": "SEASON ($\\alpha$=1.5)",
    "CrashLogic-7B-SEASON-a2.0": "SEASON ($\\alpha$=2.0)",
    "CrashLogic-7B-SEASON-shuffle": "SEASON (shuffle neg.)",
}

# Option A primary table order: foundation failures → adapted reference baselines
FOUNDATION_CONDITIONS = [
    "ZeroShot-Qwen2.5-VL-7B",
    "ZeroShot-Qwen2.5-VL-3B",
    "ZeroShot-Qwen2-VL-2B",
    "ZeroShot-LLaVA-Video-7B",
    "ZeroShot-GPT-4o",
    "ZeroShot-Gemini-1.5-Pro",
]

ADAPTED_CONDITIONS = [
    "CrashLogic-7B-Greedy",
    "CrashLogic-7B-SEASON-a0.5",
    "CrashLogic-7B-Greedy-f16",
    "CrashLogic-7B-SEASON-a0.5-f16",
]

PRIMARY_CONDITIONS = FOUNDATION_CONDITIONS + ADAPTED_CONDITIONS

BOOTSTRAP_METRICS = ["tIoU", "ArgusCost-O", "ArgusCost-H", "BLEU-4", "ROUGE-L", "Severity-Acc"]
SIGNIFICANCE_PAIRS = [
    ("CrashLogic-7B-Greedy", "CrashLogic-7B-SEASON-a0.5"),
    ("ZeroShot-Qwen2.5-VL-7B", "CrashLogic-7B-SEASON-a0.5"),
]


def _label(name: str) -> str:
    return MODEL_LABELS.get(name, name)


def load_predictions(results_dir: Path, condition: str) -> list[dict[str, Any]]:
    path = results_dir / f"{condition}_predictions.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_conditions(results_dir: Path) -> list[str]:
    return sorted(p.name.replace("_predictions.json", "") for p in results_dir.glob("*_predictions.json"))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def per_sample_field_metrics(outputs: list[dict[str, Any]]) -> dict[str, float]:
    """Structured field-level accuracy / overlap on test set."""
    sev_ok = imp_j = veh_j = wth_j = ts_parse = 0
    tiou_vals: list[float] = []
    thr_25 = thr_50 = 0
    n = len(outputs)

    for o in outputs:
        pred = extract_predicted_claims(o["prediction"])
        gt = {
            "severity": o["severity"],
            "impact": o["impact"],
            "vehicles": o["vehicles"],
            "weather": o["weather"],
            "start_sec": o["start_sec"],
            "end_sec": o["end_sec"],
            "explanation": o["explanation"],
        }

        gt_sev = _norm(gt["severity"])
        if pred["severity"] and gt_sev and pred["severity"] == gt_sev:
            sev_ok += 1

        imp_j += _jaccard(pred["impact"], _impact_tokens(gt["impact"]))
        veh_j += _jaccard(pred["vehicles"], _vehicle_tokens(gt["vehicles"]))
        wth_j += _jaccard(pred["weather"], _weather_tokens(gt["weather"]))

        if pred["start"] is not None and pred["end"] is not None:
            ts_parse += 1
            iou = temporal_iou(pred["start"], pred["end"], float(gt["start_sec"]), float(gt["end_sec"]))
            tiou_vals.append(iou)
            if iou >= 0.25:
                thr_25 += 1
            if iou >= 0.5:
                thr_50 += 1
        else:
            tiou_vals.append(0.0)

    return {
        "Severity-Acc": sev_ok / max(1, n),
        "Impact-Jaccard": imp_j / max(1, n),
        "Vehicle-Jaccard": veh_j / max(1, n),
        "Weather-Jaccard": wth_j / max(1, n),
        "Timestamp-ParseRate": ts_parse / max(1, n),
        "tIoU": float(np.mean(tiou_vals)) if tiou_vals else 0.0,
        "THR@0.25": thr_25 / max(1, n),
        "THR@0.50": thr_50 / max(1, n),
    }


def evaluate_outputs(
    outputs: list[dict[str, Any]],
    nli_scorer: NLIScorer | None = None,
) -> dict[str, float]:
    preds = [o["prediction"] for o in outputs]
    refs = [o["reference"] for o in outputs]
    gt_starts = [float(o["start_sec"]) for o in outputs]
    gt_ends = [float(o["end_sec"]) for o in outputs]
    gt_rows = [
        {
            "severity": o["severity"],
            "impact": o["impact"],
            "start_sec": o["start_sec"],
            "end_sec": o["end_sec"],
            "vehicles": o["vehicles"],
            "weather": o["weather"],
            "explanation": o["explanation"],
        }
        for o in outputs
    ]
    lexical = compute_corpus_metrics(preds, refs, gt_starts, gt_ends, use_explanation_only=True)
    argus = evaluate_argus(preds, gt_rows, compute_explanation_bertscore=True)
    nli = evaluate_nli(preds, gt_rows, scorer=nli_scorer)
    fields = per_sample_field_metrics(outputs)
    metrics = lexical.as_dict()
    metrics["ArgusCost-H"] = argus["ArgusCost-H"]
    metrics["ArgusCost-O"] = argus["ArgusCost-O"]
    metrics["Explanation-BERTScore"] = argus["Explanation-BERTScore"]
    for k, v in nli.items():
        if not k.startswith("per_sample"):
            metrics[k] = v
    metrics.update(fields)
    return metrics


def evaluate_by_severity(outputs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for o in outputs:
        buckets[_norm(o["severity"]) or "n/a"].append(o)
    out: dict[str, dict[str, float]] = {}
    for sev, rows in sorted(buckets.items()):
        m = evaluate_outputs(rows)
        m["n"] = len(rows)
        out[sev] = m
    return out


def _md_table(
    headers: list[str],
    rows: list[list[str]],
    align: list[str] | None = None,
) -> str:
    if align is None:
        align = ["l"] + ["c"] * (len(headers) - 1)
    sep_parts = []
    for a in align:
        sep_parts.append(":---" if a == "l" else "---:" if a == "r" else ":---:")
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(sep_parts) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _latex_table(
    headers: list[str],
    rows: list[list[str]],
    col_spec: str | None = None,
) -> str:
    ncol = len(headers)
    spec = col_spec or ("l" + "c" * (ncol - 1))
    lines = [
        r"\begin{tabular}{" + spec + "}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def table1_main_captioning(all_metrics: dict[str, dict[str, float]]) -> dict[str, str]:
    """Table I: Dense caption quality (higher is better)."""
    cols = ["BLEU-4", "ROUGE-L", "METEOR", "CIDEr", "BERTScore", "Explanation-BERTScore"]
    headers = ["Method"] + cols
    rows = []
    for name, m in all_metrics.items():
        rows.append([_label(name)] + [f"{m.get(c, 0):.3f}" for c in cols])
    note = (
        "_Table I: Corpus-level caption quality on the 150-video CCD test split. "
        "Metrics computed on the Explanation field unless noted. $\\uparrow$ higher is better._"
    )
    return {
        "title": "Table I — Main Captioning Quality",
        "markdown": _md_table(headers, rows) + "\n\n" + note,
        "latex": _latex_table(headers, rows),
    }


def table2_temporal_forensic(all_metrics: dict[str, dict[str, float]]) -> dict[str, str]:
    """Table II: Temporal reasoning & forensic fidelity."""
    cols = ["tIoU", "THR@0.25", "THR@0.50", "ArgusCost-H", "ArgusCost-O"]
    headers = ["Method"] + cols
    rows = []
    for name, m in all_metrics.items():
        rows.append([_label(name)] + [f"{m.get(c, 0):.3f}" for c in cols])
    note = (
        "_Table II: Spatiotemporal and forensic metrics. "
        "tIoU = mean temporal IoU of crash windows; THR@$\\tau$ = fraction with tIoU $\\geq \\tau$. "
        "ArgusCost-H/O = structured hallucination / omission rates ($\\downarrow$ better)._"
    )
    return {
        "title": "Table II — Temporal & Forensic Reasoning",
        "markdown": _md_table(headers, rows) + "\n\n" + note,
        "latex": _latex_table(headers, rows),
    }


def table3_structured_fields(all_metrics: dict[str, dict[str, float]]) -> dict[str, str]:
    """Table III: Per-field structured parsing accuracy."""
    cols = [
        "Severity-Acc",
        "Impact-Jaccard",
        "Vehicle-Jaccard",
        "Weather-Jaccard",
        "Timestamp-ParseRate",
    ]
    headers = ["Method"] + [c.replace("-", " ") for c in cols]
    rows = []
    for name, m in all_metrics.items():
        rows.append([_label(name)] + [f"{m.get(c, 0):.3f}" for c in cols])
    note = (
        "_Table III: Structured claim extraction accuracy. "
        "Severity-Acc = exact match rate; Jaccard scores for token-overlap fields; "
        "Timestamp-ParseRate = fraction with parseable Start/End._"
    )
    return {
        "title": "Table III — Structured Field Accuracy",
        "markdown": _md_table(headers, rows) + "\n\n" + note,
        "latex": _latex_table(headers, rows),
    }


def table4_severity_stratified(
    condition: str,
    by_severity: dict[str, dict[str, float]],
) -> dict[str, str]:
    """Table IV: Results stratified by crash severity (primary model comparison)."""
    sevs = ["minor", "moderate", "severe", "fatal", "n/a"]
    cols = ["tIoU", "BERTScore", "ArgusCost-H", "ArgusCost-O", "n"]
    headers = ["Severity"] + cols
    rows = []
    for sev in sevs:
        if sev not in by_severity:
            continue
        m = by_severity[sev]
        rows.append(
            [sev.capitalize()]
            + [f"{m.get(c, 0):.3f}" if c != "n" else str(int(m.get("n", 0))) for c in cols]
        )
    note = (
        f"_Table IV: { _label(condition) } performance stratified by GT crash severity "
        f"on the CCD test split._"
    )
    return {
        "title": f"Table IV — Severity-Stratified ({_label(condition)})",
        "markdown": _md_table(headers, rows) + "\n\n" + note,
        "latex": _latex_table(headers, rows),
    }


def table5_relative_gains(
    all_metrics: dict[str, dict[str, float]],
    baseline: str = "ZeroShot-Qwen2.5-VL-7B",
) -> dict[str, str]:
    """Table V: Relative improvement over zero-shot baseline (%)."""
    if baseline not in all_metrics:
        return {"title": "Table V", "markdown": "_Baseline missing_", "latex": ""}

    higher_better = ["BLEU-4", "ROUGE-L", "METEOR", "CIDEr", "BERTScore", "tIoU", "THR@0.50"]
    lower_better = ["ArgusCost-H", "ArgusCost-O"]
    cols = higher_better + lower_better
    headers = ["Method"] + cols
    base = all_metrics[baseline]
    rows = []
    for name, m in all_metrics.items():
        if name == baseline:
            continue
        cells = [_label(name)]
        for c in cols:
            b, v = base.get(c, 0), m.get(c, 0)
            if c in lower_better:
                gain = ((b - v) / max(b, 1e-9)) * 100 if b > 0 else 0.0
            else:
                gain = ((v - b) / max(b, 1e-9)) * 100 if b > 0 else 0.0
            cells.append(f"{gain:+.1f}%")
        rows.append(cells)
    note = (
        "_Table V: Relative change vs. zero-shot baseline. "
        "Positive % on caption/temporal metrics = improvement; "
        "positive % on ArgusCost = reduction in hallucination/omission._"
    )
    return {
        "title": "Table V — Relative Gains over Zero-shot",
        "markdown": _md_table(headers, rows) + "\n\n" + note,
        "latex": _latex_table(headers, rows),
    }


def table6_season_ablation(ablation_metrics: dict[str, dict[str, float]]) -> dict[str, str]:
    """Table VI: SEASON hyperparameter / variant ablation."""
    cols = ["tIoU", "BERTScore", "ArgusCost-H", "ArgusCost-O", "ROUGE-L"]
    headers = ["Variant"] + cols
    rows = []
    for name, m in ablation_metrics.items():
        rows.append([_label(name)] + [f"{m.get(c, 0):.3f}" for c in cols])
    note = (
        "_Table VI: SEASON decoding ablation on CrashLogic-7B. "
        "Simple temporal negative (reverse/shuffle) vs. full self-diagnostic SEASON; "
        "$\\alpha$ sweeps contrastive strength._"
    )
    return {
        "title": "Table VI — SEASON Ablation",
        "markdown": _md_table(headers, rows) + "\n\n" + note,
        "latex": _latex_table(headers, rows),
    }


def table4b_cross_model_severity(
    all_by_severity: dict[str, dict[str, dict[str, float]]],
) -> dict[str, str]:
    """Table IVb: Cross-model tIoU by severity stratum."""
    sevs = ["minor", "moderate", "severe", "fatal"]
    models = list(all_by_severity.keys())
    headers = ["Severity"] + [_label(m) for m in models]
    rows = []
    for sev in sevs:
        row = [sev.capitalize()]
        for m in models:
            if sev in all_by_severity.get(m, {}):
                v = all_by_severity[m][sev].get("tIoU", float("nan"))
                row.append(f"{v:.3f}")
            else:
                row.append("—")
        rows.append(row)
    note = (
        "_Table IVb: Mean temporal IoU (tIoU) stratified by crash severity across methods. "
        "Higher is better._"
    )
    return {
        "title": "Table IVb — Cross-Model tIoU by Severity",
        "markdown": _md_table(headers, rows) + "\n\n" + note,
        "latex": _latex_table(headers, rows),
    }


def table7_nli_faithfulness(all_metrics: dict[str, dict[str, float]]) -> dict[str, str]:
    """Table VII: NLI faithfulness / hallucination scores."""
    cols = [
        "NLI-Entailment",
        "NLI-Contradiction",
        "NLI-Score",
        "NLI-Loss",
        "Full-NLI-Entailment",
        "Full-NLI-Contradiction",
    ]
    headers = ["Method"] + [c.replace("-", " ") for c in cols]
    rows = []
    nli_order = [
        c
        for c in PRIMARY_CONDITIONS
        if c in all_metrics and "NLI-Entailment" in all_metrics[c]
    ]
    for name in nli_order:
        m = all_metrics[name]
        rows.append([_label(name)] + [f"{m.get(c, 0):.3f}" for c in cols])
    note = (
        "_Table VII: NLI faithfulness via cross-encoder/nli-deberta-v3-small. "
        "GT forensic text as premise; model explanation (or full output) as hypothesis. "
        "NLI-Loss = $-\\log P(\\text{entail})$ (lower better); NLI-Score = $P_e - P_c$ (higher better)._"
    )
    return {
        "title": "Table VII — NLI Faithfulness & Hallucination",
        "markdown": _md_table(headers, rows) + "\n\n" + note,
        "latex": _latex_table(headers, rows),
    }


def table8_bootstrap_ci(
    outputs_cache: dict[str, list[dict[str, Any]]],
    conditions: Sequence[str],
    n_bootstrap: int = 1000,
) -> dict[str, str]:
    """Table VIII: Bootstrap 95% CIs on key forensic metrics."""
    headers = ["Method", "Metric", "Mean", "95% CI"]
    rows: list[list[str]] = []
    ci_report: dict[str, dict[str, dict[str, float]]] = {}

    for cond in conditions:
        if cond not in outputs_cache:
            continue
        vectors = extract_per_sample_vectors(outputs_cache[cond])
        ci_report[cond] = {}
        for metric in BOOTSTRAP_METRICS:
            if metric not in vectors:
                continue
            res = bootstrap_ci(vectors[metric], n_bootstrap=n_bootstrap)
            ci_report[cond][metric] = {
                "mean": res.mean,
                "ci_low": res.ci_low,
                "ci_high": res.ci_high,
            }
            rows.append(
                [
                    _label(cond),
                    metric,
                    f"{res.mean:.3f}",
                    f"[{res.ci_low:.3f}, {res.ci_high:.3f}]",
                ]
            )

    note = (
        f"_Table VIII: Bootstrap 95% confidence intervals ($N={n_bootstrap}$ resamples) "
        "on per-video scores. Primary forensic metrics for Option A benchmark paper._"
    )
    return {
        "title": "Table VIII — Bootstrap 95% Confidence Intervals",
        "markdown": _md_table(headers, rows) + "\n\n" + note,
        "latex": _latex_table(headers, rows, col_spec="l l c c"),
        "ci_data": ci_report,
    }


def table9_significance_tests(
    outputs_cache: dict[str, list[dict[str, Any]]],
    pairs: Sequence[tuple[str, str]] | None = None,
) -> dict[str, str]:
    """Table IX: Paired Wilcoxon tests between key conditions."""
    pairs = list(pairs or SIGNIFICANCE_PAIRS)
    headers = ["Comparison", "Metric", "Wilcoxon $p$", "Significant ($p<0.05$)"]
    rows: list[list[str]] = []
    sig_data: list[dict[str, Any]] = []

    for cond_a, cond_b in pairs:
        if cond_a not in outputs_cache or cond_b not in outputs_cache:
            continue
        va = extract_per_sample_vectors(outputs_cache[cond_a])
        vb = extract_per_sample_vectors(outputs_cache[cond_b])
        va, vb = align_paired_vectors(va, vb)

        for metric in ["tIoU", "ArgusCost-O", "BLEU-4", "Severity-Acc"]:
            if metric in ("tIoU", "BLEU-4", "Severity-Acc"):
                alt = "less"  # test cond_a < cond_b → cond_b better
            else:
                alt = "greater"  # test cond_a > cond_b → cond_b lower/better
            test = paired_wilcoxon(va[metric], vb[metric], alternative=alt)
            rows.append(
                [
                    f"{_label(cond_a)} vs {_label(cond_b)}",
                    metric,
                    test.format_p(),
                    "Yes" if test.significant else "No",
                ]
            )
            sig_data.append(
                {
                    "a": cond_a,
                    "b": cond_b,
                    "metric": metric,
                    "p_value": test.p_value,
                    "significant": test.significant,
                }
            )

    note = (
        "_Table IX: Paired Wilcoxon signed-rank tests on aligned per-video scores. "
        "One-sided tests: higher tIoU/BLEU/Severity-Acc and lower ArgusCost-O are better._"
    )
    return {
        "title": "Table IX — Paired Significance Tests",
        "markdown": _md_table(headers, rows) + "\n\n" + note,
        "latex": _latex_table(headers, rows, col_spec="l l c c"),
        "significance_data": sig_data,
    }


def table0_foundation_benchmark(
    all_metrics: dict[str, dict[str, float]],
    conditions: Sequence[str],
) -> dict[str, str]:
    """Table 0: Main foundation-model benchmark (Option A headline table)."""
    cols = ["tIoU", "ArgusCost-O", "ArgusCost-H", "Severity-Acc", "Timestamp-ParseRate", "BLEU-4"]
    headers = ["Method"] + cols
    rows = []
    for name in conditions:
        if name not in all_metrics:
            continue
        m = all_metrics[name]
        rows.append([_label(name)] + [f"{m.get(c, 0):.3f}" for c in cols])
    note = (
        "_Table 0: CrashX foundation-model benchmark on 150-video CCD test split. "
        "Foundation VLMs (zero-shot) vs domain-adapted CrashLogic reference baselines. "
        "TCD = Task-Adapted Temporal Contrastive Decoding ($\\alpha$=0.5)._"
    )
    return {
        "title": "Table 0 — Foundation Model Benchmark (Option A)",
        "markdown": _md_table(headers, rows) + "\n\n" + note,
        "latex": _latex_table(headers, rows),
    }


def select_primary_conditions(
    all_metrics: dict[str, dict[str, float]],
    conditions: Sequence[str] | None = None,
) -> list[str]:
    """Return available primary conditions in canonical order."""
    order = list(conditions or PRIMARY_CONDITIONS)
    return [c for c in order if c in all_metrics]


TABLE_ORDER = [
    "table0_foundation_benchmark",
    "table1_main_captioning",
    "table2_temporal_forensic",
    "table3_structured_fields",
    "table4_severity_stratified",
    "table4b_cross_model_severity",
    "table5_relative_gains",
    "table6_season_ablation",
    "table7_nli_faithfulness",
    "table8_bootstrap_ci",
    "table9_significance_tests",
]


def generate_all_tables(
    results_dir: Path,
    conditions: Sequence[str] | None = None,
    severity_condition: str = "CrashLogic-7B-SEASON-a0.5",
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    """Build all journal tables from prediction JSON files."""
    results_dir = Path(results_dir)
    if conditions is None:
        conditions = discover_conditions(results_dir)

    all_metrics: dict[str, dict[str, float]] = {}
    outputs_cache: dict[str, list[dict[str, Any]]] = {}
    nli_scorer = NLIScorer()
    for cond in conditions:
        outputs = load_predictions(results_dir, cond)
        outputs_cache[cond] = outputs
        m = evaluate_outputs(outputs, nli_scorer=nli_scorer)
        m["n"] = len(outputs)
        all_metrics[cond] = m
    nli_scorer.unload()

    # Option A primary: foundation baselines + adapted reference (TCD α=0.5)
    primary = select_primary_conditions(all_metrics, PRIMARY_CONDITIONS)
    if len(primary) < 2:
        primary = select_primary_conditions(
            all_metrics,
            [
                "ZeroShot-Qwen2.5-VL-7B",
                "CrashLogic-7B-Greedy",
                "CrashLogic-7B-SEASON-a0.5",
            ],
        )
    primary_metrics = {c: all_metrics[c] for c in primary}

    tables: dict[str, dict[str, str]] = {}
    if primary_metrics:
        tables["table0_foundation_benchmark"] = table0_foundation_benchmark(
            all_metrics, primary
        )
    tables["table1_main_captioning"] = table1_main_captioning(primary_metrics)
    tables["table2_temporal_forensic"] = table2_temporal_forensic(primary_metrics)
    tables["table3_structured_fields"] = table3_structured_fields(primary_metrics)
    tables["table5_relative_gains"] = table5_relative_gains(primary_metrics)
    if any("NLI-Entailment" in all_metrics.get(c, {}) for c in primary):
        tables["table7_nli_faithfulness"] = table7_nli_faithfulness(primary_metrics)

    if severity_condition in outputs_cache:
        by_sev = evaluate_by_severity(outputs_cache[severity_condition])
        tables["table4_severity_stratified"] = table4_severity_stratified(
            severity_condition, by_sev
        )

    # Cross-model severity comparison for primary methods
    cross_sev: dict[str, dict[str, dict[str, float]]] = {}
    for cond in primary:
        cross_sev[cond] = evaluate_by_severity(outputs_cache[cond])
    if cross_sev:
        tables["table4b_cross_model_severity"] = table4b_cross_model_severity(cross_sev)

    # Ablation table: greedy + season variants
    ablation_keys = [
        c
        for c in conditions
        if "SEASON" in c or c == "CrashLogic-7B-Greedy"
    ]
    if len(ablation_keys) >= 2:
        tables["table6_season_ablation"] = table6_season_ablation(
            {c: all_metrics[c] for c in ablation_keys}
        )

    if primary:
        tables["table8_bootstrap_ci"] = table8_bootstrap_ci(
            outputs_cache, primary, n_bootstrap=n_bootstrap
        )
        tables["table9_significance_tests"] = table9_significance_tests(outputs_cache)

    return {
        "metrics": all_metrics,
        "tables": tables,
        "primary_conditions": primary,
    }


def write_tables(report: dict[str, Any], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_md = ["# CrashX Journal Results\n"]
    combined_tex: list[str] = []

    ordered_keys = [k for k in TABLE_ORDER if k in report["tables"]]
    ordered_keys += [k for k in report["tables"] if k not in ordered_keys]

    for key in ordered_keys:
        tbl = report["tables"][key]
        (out_dir / f"{key}.md").write_text(
            f"## {tbl['title']}\n\n{tbl['markdown']}\n", encoding="utf-8"
        )
        (out_dir / f"{key}.tex").write_text(tbl["latex"] + "\n", encoding="utf-8")
        combined_md.append(f"## {tbl['title']}\n\n{tbl['markdown']}\n")
        combined_tex.append(f"% {tbl['title']}\n{tbl['latex']}\n")

    (out_dir / "all_tables.md").write_text("\n".join(combined_md) + "\n", encoding="utf-8")
    (out_dir / "all_tables.tex").write_text("\n\n".join(combined_tex) + "\n", encoding="utf-8")

    with (out_dir / "full_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(report["metrics"], f, indent=2)

    stats_payload = {
        "primary_conditions": report.get("primary_conditions", []),
        "bootstrap_ci": report["tables"].get("table8_bootstrap_ci", {}).get("ci_data", {}),
        "significance": report["tables"].get("table9_significance_tests", {}).get(
            "significance_data", []
        ),
    }
    with (out_dir / "statistical_analysis.json").open("w", encoding="utf-8") as f:
        json.dump(stats_payload, f, indent=2)
