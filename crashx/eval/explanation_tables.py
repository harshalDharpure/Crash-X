#!/usr/bin/env python3
"""Explanation-only paper tables (guide-aligned; no severity).

Main claim: crash *explanation* quality + faithfulness.
Temporal IoU is included only as supporting evidence for temporal hallucination.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from crashx.eval.journal_tables import (
    MODEL_LABELS,
    _label,
    _latex_table,
    _md_table,
    discover_conditions,
    evaluate_outputs,
    load_predictions,
)
from crashx.eval.nli_eval import NLIScorer

# Canonical order for explanation paper
MAIN_METHODS = [
    "ZeroShot-Qwen2.5-VL-7B",
    "ZeroShot-Qwen2.5-VL-3B",
    "ZeroShot-Qwen2-VL-2B",
    "ZeroShot-LLaVA-Video-7B",
    "CrashLogic-7B-Greedy",
    "CrashLogic-7B-SEASON-a0.5",
]

ABLATION_METHODS = [
    "CrashLogic-7B-Greedy",
    "CrashLogic-7B-SEASON-a0.5",
    "CrashLogic-7B-SEASON",
    "CrashLogic-7B-SEASON-a1.5",
    "CrashLogic-7B-SEASON-a2.0",
    "CrashLogic-7B-SEASON-shuffle",
    "CrashLogic-7B-SEASON-Full",
]

EXPLAIN_LABELS = {
    **MODEL_LABELS,
    "CrashLogic-7B-SEASON-a0.5": "CrashLogic-7B + TCD ($\\alpha$=0.5)",
    "CrashLogic-7B-SEASON": "CrashLogic-7B + TCD ($\\alpha$=1.0)",
}


def elabel(name: str) -> str:
    return EXPLAIN_LABELS.get(name, _label(name))


def load_or_compute_metrics(
    results_dir: Path,
    conditions: list[str],
) -> dict[str, dict[str, float]]:
    """Prefer existing metrics; recompute if NLI missing for a condition."""
    need_nli = []
    metrics: dict[str, dict[str, float]] = {}
    for cond in conditions:
        mpath = results_dir / f"{cond}_metrics.json"
        if mpath.is_file():
            m = json.loads(mpath.read_text(encoding="utf-8"))
            if "NLI-Score" in m:
                metrics[cond] = m
                continue
        need_nli.append(cond)

    if need_nli:
        scorer = NLIScorer()
        for cond in need_nli:
            outs = load_predictions(results_dir, cond)
            m = evaluate_outputs(outs, nli_scorer=scorer)
            metrics[cond] = m
            (results_dir / f"{cond}_metrics.json").write_text(
                json.dumps(m, indent=2), encoding="utf-8"
            )
        scorer.unload()
    return metrics


def table_e1_quality(metrics: dict[str, dict[str, float]]) -> dict[str, str]:
    cols = ["BLEU-4", "ROUGE-L", "METEOR", "CIDEr", "BERTScore"]
    headers = ["Method"] + cols
    rows = []
    for name in MAIN_METHODS:
        if name not in metrics:
            continue
        m = metrics[name]
        rows.append([elabel(name)] + [f"{m.get(c, 0):.3f}" for c in cols])
    note = (
        "_Table E1: Crash **explanation** caption quality on the 150-video CCD test split. "
        "Metrics computed on the Explanation field. $\\uparrow$ higher is better._"
    )
    return {
        "title": "Table E1 — Explanation Quality",
        "markdown": _md_table(headers, rows) + "\n\n" + note,
        "latex": _latex_table(headers, rows),
    }


def table_e2_faithfulness(metrics: dict[str, dict[str, float]]) -> dict[str, str]:
    cols = [
        "NLI-Entailment",
        "NLI-Contradiction",
        "NLI-Score",
        "NLI-Loss",
        "Explanation-BERTScore",
    ]
    headers = ["Method", "NLI-Entail", "NLI-Contradict", "NLI-Score", "NLI-Loss", "Expl.-BERTScore"]
    rows = []
    for name in MAIN_METHODS:
        if name not in metrics:
            continue
        m = metrics[name]
        rows.append([elabel(name)] + [f"{m.get(c, 0):.3f}" for c in cols])
    note = (
        "_Table E2: Explanation faithfulness. GT forensic text as premise; model explanation as "
        "hypothesis (cross-encoder NLI). NLI-Score $= P_e - P_c$ ($\\uparrow$); "
        "NLI-Loss $= -\\log P_e$ ($\\downarrow$)._"
    )
    return {
        "title": "Table E2 — Explanation Faithfulness (NLI)",
        "markdown": _md_table(headers, rows) + "\n\n" + note,
        "latex": _latex_table(headers, rows),
    }


def table_e3_temporal_support(metrics: dict[str, dict[str, float]]) -> dict[str, str]:
    cols = ["tIoU", "THR@0.25", "THR@0.50"]
    headers = ["Method"] + cols
    rows = []
    for name in MAIN_METHODS:
        if name not in metrics:
            continue
        m = metrics[name]
        rows.append([elabel(name)] + [f"{m.get(c, 0):.3f}" for c in cols])
    note = (
        "_Table E3: Temporal consistency support (crash window localization). "
        "Reported as evidence that better explanations align with correct event timing; "
        "not a severity / forensic-field evaluation._"
    )
    return {
        "title": "Table E3 — Temporal Consistency (Support)",
        "markdown": _md_table(headers, rows) + "\n\n" + note,
        "latex": _latex_table(headers, rows),
    }


def table_e4_ablation(metrics: dict[str, dict[str, float]]) -> dict[str, str]:
    cols = ["BLEU-4", "ROUGE-L", "BERTScore", "NLI-Score", "tIoU"]
    headers = ["Variant"] + cols
    rows = []
    for name in ABLATION_METHODS:
        if name not in metrics:
            continue
        m = metrics[name]
        rows.append([elabel(name)] + [f"{m.get(c, float('nan')):.3f}" if c in m else "—" for c in cols])
    note = (
        "_Table E4: Temporal contrastive decoding (TCD / SEASON) ablation on CrashLogic-7B. "
        "Primary recommended setting: $\\alpha$=0.5 with reverse temporal negative._"
    )
    return {
        "title": "Table E4 — Decoding Ablation (Explanation Focus)",
        "markdown": _md_table(headers, rows) + "\n\n" + note,
        "latex": _latex_table(headers, rows),
    }


def table_e5_relative_gains(metrics: dict[str, dict[str, float]]) -> dict[str, str]:
    baseline = "ZeroShot-Qwen2.5-VL-7B"
    if baseline not in metrics:
        return {"title": "Table E5", "markdown": "_baseline missing_", "latex": ""}
    higher = ["BLEU-4", "ROUGE-L", "BERTScore", "NLI-Score", "tIoU"]
    headers = ["Method"] + [f"$\\Delta$ {c}" for c in higher]
    base = metrics[baseline]
    rows = []
    for name in MAIN_METHODS:
        if name == baseline or name not in metrics:
            continue
        m = metrics[name]
        cells = [elabel(name)]
        for c in higher:
            b, v = base.get(c, 0.0), m.get(c, 0.0)
            if abs(b) < 1e-9:
                cells.append("—")
            else:
                cells.append(f"{((v - b) / abs(b)) * 100:+.1f}%")
        rows.append(cells)
    note = (
        "_Table E5: Relative change vs zero-shot Qwen2.5-VL-7B on explanation / faithfulness metrics._"
    )
    return {
        "title": "Table E5 — Gains over Zero-shot (Explanation)",
        "markdown": _md_table(headers, rows) + "\n\n" + note,
        "latex": _latex_table(headers, rows),
    }


def generate_explanation_tables(results_dir: Path, out_dir: Path) -> dict[str, Any]:
    results_dir = Path(results_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    available = set(discover_conditions(results_dir))
    wanted = [c for c in dict.fromkeys(MAIN_METHODS + ABLATION_METHODS) if c in available]
    metrics = load_or_compute_metrics(results_dir, wanted)

    tables = {
        "E1_explanation_quality": table_e1_quality(metrics),
        "E2_explanation_faithfulness": table_e2_faithfulness(metrics),
        "E3_temporal_support": table_e3_temporal_support(metrics),
        "E4_decoding_ablation": table_e4_ablation(metrics),
        "E5_relative_gains": table_e5_relative_gains(metrics),
    }

    combined = [
        "# CrashX — Explanation-Only Results (Main Paper)\n",
        "> Scope: **explanation quality & faithfulness only**. "
        "Severity / structured forensic field tables intentionally omitted per advisor guidance.\n",
    ]
    tex_blocks = []
    for key, tbl in tables.items():
        (out_dir / f"{key}.md").write_text(f"## {tbl['title']}\n\n{tbl['markdown']}\n", encoding="utf-8")
        (out_dir / f"{key}.tex").write_text(tbl["latex"] + "\n", encoding="utf-8")
        combined.append(f"## {tbl['title']}\n\n{tbl['markdown']}\n")
        tex_blocks.append(f"% {tbl['title']}\n{tbl['latex']}\n")

    (out_dir / "all_explanation_tables.md").write_text("\n".join(combined) + "\n", encoding="utf-8")
    (out_dir / "all_explanation_tables.tex").write_text("\n\n".join(tex_blocks) + "\n", encoding="utf-8")

    slim = {
        name: {
            k: metrics[name][k]
            for k in [
                "BLEU-4",
                "ROUGE-L",
                "METEOR",
                "CIDEr",
                "BERTScore",
                "Explanation-BERTScore",
                "NLI-Entailment",
                "NLI-Contradiction",
                "NLI-Score",
                "NLI-Loss",
                "tIoU",
                "THR@0.25",
                "THR@0.50",
            ]
            if k in metrics[name]
        }
        for name in wanted
        if name in metrics
    }
    (out_dir / "explanation_metrics.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
    return {"metrics": slim, "tables": tables, "out_dir": str(out_dir)}


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Build explanation-only CrashX tables")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--out-dir", type=Path, default=Path("results/tables/explanation"))
    args = p.parse_args()
    report = generate_explanation_tables(args.results_dir, args.out_dir)
    print(f"Wrote explanation tables → {report['out_dir']}")
    for key, tbl in report["tables"].items():
        print(f"\n{'=' * 60}\n{tbl['title']}\n{'=' * 60}\n")
        print(tbl["markdown"])


if __name__ == "__main__":
    main()
