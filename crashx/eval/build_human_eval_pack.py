#!/usr/bin/env python3
"""Build blinded human-eval annotation packs + Fleiss' κ scorer (Phase 4)."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from crashx.data.dataset import load_jsonl
from crashx.eval.human_eval_sampler import stratified_sample


DEFAULT_MODELS = [
    "ZeroShot-Qwen2.5-VL-7B",
    "ZeroShot-LLaVA-Video-7B",
    "CrashLogic-7B-Greedy",
    "CrashLogic-7B-SEASON-a0.5",
]


def load_preds(results_dir: Path, name: str) -> dict[str, dict[str, Any]]:
    path = results_dir / f"{name}_predictions.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {r["video_id"]: r for r in rows}


def build_annotation_pack(
    sample: list[dict[str, Any]],
    model_preds: dict[str, dict[str, dict[str, Any]]],
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Create blinded items. Returns (rows, blind_code→model)."""
    rng = random.Random(seed)
    models = list(model_preds.keys())
    code_map = {f"M{i + 1}": m for i, m in enumerate(models)}
    # shuffle codes so M1 is not always the same model across papers if desired
    codes = list(code_map.keys())
    rng.shuffle(codes)
    code_map = {c: models[i] for i, c in enumerate(codes)}
    inv = {m: c for c, m in code_map.items()}

    rows: list[dict[str, Any]] = []
    for rec in sample:
        vid = rec["video_id"]
        for model, preds in model_preds.items():
            if vid not in preds:
                continue
            pred = preds[vid]["prediction"]
            rows.append(
                {
                    "item_id": f"{vid}_{inv[model]}",
                    "video_id": vid,
                    "blind_code": inv[model],
                    "video_path": rec.get("video_path", ""),
                    "gt_severity": rec.get("severity", ""),
                    "gt_start": rec.get("start_sec", ""),
                    "gt_end": rec.get("end_sec", ""),
                    "model_output": pred,
                    "temporal_likert": "",
                    "faithfulness_likert": "",
                    "explanation_likert": "",
                    "annotator_id": "",
                    "notes": "",
                }
            )
    rng.shuffle(rows)
    return rows, code_map


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No annotation rows")
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def fleiss_kappa(ratings: np.ndarray) -> float:
    """ratings: [n_items, n_categories] count matrix."""
    n_items, n_cats = ratings.shape
    n_raters = ratings.sum(axis=1)
    if np.any(n_raters == 0):
        raise ValueError("Some items have zero ratings")
    p = ratings / n_raters[:, None]
    P_i = (p * p).sum(axis=1)
    # correction for discrete counts
    P_i = (ratings * (ratings - 1)).sum(axis=1) / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()
    p_j = ratings.sum(axis=0) / ratings.sum()
    P_e = float((p_j ** 2).sum())
    if abs(1 - P_e) < 1e-12:
        return 1.0
    return float((P_bar - P_e) / (1 - P_e))


def compute_kappa_from_filled(
    csv_paths: list[Path],
    score_col: str = "temporal_likert",
    n_categories: int = 5,
) -> dict[str, Any]:
    """Merge multiple annotator CSVs and compute Fleiss' κ on a Likert column."""
    by_item: dict[str, list[int]] = defaultdict(list)
    for path in csv_paths:
        with path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                val = (row.get(score_col) or "").strip()
                if not val:
                    continue
                by_item[row["item_id"]].append(int(float(val)))

    items = sorted(by_item)
    # Keep items rated by ≥2 annotators
    items = [i for i in items if len(by_item[i]) >= 2]
    if not items:
        return {"n_items": 0, "kappa": None, "error": "Need ≥2 ratings per item"}

    mat = np.zeros((len(items), n_categories), dtype=np.int64)
    for r, item in enumerate(items):
        for score in by_item[item]:
            if 1 <= score <= n_categories:
                mat[r, score - 1] += 1
    return {
        "n_items": len(items),
        "n_annotator_files": len(csv_paths),
        "score_col": score_col,
        "kappa": fleiss_kappa(mat),
        "mean_rating": float(np.average(np.arange(1, n_categories + 1), weights=mat.sum(axis=0))),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CrashX human-eval pack + kappa")
    p.add_argument("--test-jsonl", type=Path, default=Path("crashx/data/splits/test.jsonl"))
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--out-dir", type=Path, default=Path("results/human_eval"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--n-annotators", type=int, default=3)
    p.add_argument(
        "--score-kappa",
        nargs="+",
        type=Path,
        default=None,
        help="Filled annotator CSV paths → print Fleiss' κ",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.score_kappa:
        for col in ("temporal_likert", "faithfulness_likert", "explanation_likert"):
            report = compute_kappa_from_filled(args.score_kappa, score_col=col)
            print(json.dumps({"column": col, **report}, indent=2))
        return

    records = load_jsonl(args.test_jsonl)
    targets = {"minor": 15, "moderate": 15, "severe": 12, "fatal": 4, "n/a": 4}
    sample = stratified_sample(records, targets, seed=args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.out_dir / "sample_50.jsonl"
    with sample_path.open("w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    model_preds = {}
    for m in args.models:
        path = args.results_dir / f"{m}_predictions.json"
        if not path.is_file():
            print(f"[skip missing] {m}")
            continue
        model_preds[m] = load_preds(args.results_dir, m)

    rows, code_map = build_annotation_pack(sample, model_preds, seed=args.seed)
    master = args.out_dir / "annotation_master.csv"
    write_csv(master, rows)
    (args.out_dir / "blind_code_map.json").write_text(
        json.dumps(code_map, indent=2), encoding="utf-8"
    )

    # Per-annotator empty copies
    for i in range(1, args.n_annotators + 1):
        copy_rows = []
        for r in rows:
            rr = dict(r)
            rr["annotator_id"] = f"A{i}"
            copy_rows.append(rr)
        write_csv(args.out_dir / f"annotator_A{i}.csv", copy_rows)

    readme = args.out_dir / "README.md"
    readme.write_text(
        f"""# CrashX Human Evaluation Pack

- Sample: `{sample_path.name}` ({len(sample)} videos, stratified)
- Master: `{master.name}` ({len(rows)} items = videos × models)
- Blind map: `blind_code_map.json` (**do not show to annotators**)
- Annotator sheets: `annotator_A1.csv` … `annotator_A{args.n_annotators}.csv`

## Instructions
1. Open the video at `video_path`.
2. Read `model_output` only (ignore blind_code meaning).
3. Score Likert 1–5 for temporal / faithfulness / explanation.
4. Save filled CSVs and run:

```bash
PYTHONPATH=. python -m crashx.eval.build_human_eval_pack \\
  --score-kappa results/human_eval/annotator_A1_filled.csv \\
               results/human_eval/annotator_A2_filled.csv \\
               results/human_eval/annotator_A3_filled.csv
```

Target: Fleiss' κ ≥ 0.65.
""",
        encoding="utf-8",
    )
    print(f"Wrote human-eval pack → {args.out_dir} ({len(sample)} videos, {len(rows)} items)")
    print(f"Blind map: {code_map}")


if __name__ == "__main__":
    main()
