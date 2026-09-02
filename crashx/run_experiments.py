#!/usr/bin/env python3
"""CrashX master experiment runner.

Compares three model states on the 150-video test split:
  1. Zero-shot Qwen2.5-VL-7B (greedy)
  2. Fine-tuned CrashLogic-7B (greedy)
  3. CrashLogic-7B + SEASON contrastive decoding (alpha=1.0, use_full_season=False)

Optional --full-season-ablation adds condition 3 with use_full_season=True.

Writes JSON predictions, a markdown comparison table, and LaTeX tabular code.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_experiments")

from crashx.data.dataset import load_jsonl, load_video_frames_cv2
from crashx.eval.argus_eval import evaluate_argus
from crashx.eval.metrics import compute_corpus_metrics
from crashx.eval.nli_eval import evaluate_nli
from crashx.inference.season_decoder import SeasonDecoder
from crashx.models.vru_baseline import DEFAULT_MODEL_ID, VRUBaselineModel
from crashx.prompts import USER_PROMPT


def _condition_specs(
    lora_path: Path | None,
    alpha: float,
    full_season_ablation: bool,
) -> list[dict[str, Any]]:
    specs = [
        {
            "name": "ZeroShot-Qwen2.5-VL-7B",
            "lora": None,
            "decode": "greedy",
            "alpha": 0.0,
            "use_full_season": False,
        },
        {
            "name": "CrashLogic-7B-Greedy",
            "lora": str(lora_path) if lora_path else None,
            "decode": "greedy",
            "alpha": 0.0,
            "use_full_season": False,
        },
        {
            "name": "CrashLogic-7B-SEASON",
            "lora": str(lora_path) if lora_path else None,
            "decode": "season",
            "alpha": alpha,
            "use_full_season": False,
        },
    ]
    if full_season_ablation:
        specs.append(
            {
                "name": "CrashLogic-7B-SEASON-Full",
                "lora": str(lora_path) if lora_path else None,
                "decode": "season",
                "alpha": alpha,
                "use_full_season": True,
            }
        )
    return specs


def run_condition(
    spec: dict[str, Any],
    records: list[dict[str, Any]],
    model_id: str,
    num_frames: int,
    max_new_tokens: int,
    load_in_4bit: bool,
    neg_mode: str,
    beta: float,
    limit: int | None,
) -> list[dict[str, Any]]:
    """Run inference for one experimental condition."""
    logger.info("=== Condition: %s ===", spec["name"])
    if spec["lora"] is None and spec["name"].startswith("CrashLogic"):
        logger.warning(
            "LoRA path missing for %s — falling back to base weights.",
            spec["name"],
        )

    backbone = VRUBaselineModel(
        model_id=model_id,
        lora_path=spec["lora"],
        load_in_4bit=load_in_4bit,
    ).load()
    decoder = SeasonDecoder(backbone)

    rows = records[:limit] if limit else records
    outputs: list[dict[str, Any]] = []
    for rec in tqdm(rows, desc=spec["name"]):
        frames = load_video_frames_cv2(rec["video_path"], num_frames=num_frames)
        if spec["decode"] == "greedy":
            text = decoder.decode_greedy(
                frames,
                USER_PROMPT,
                max_new_tokens=max_new_tokens,
            )
        else:
            text = decoder.decode_season(
                frames,
                USER_PROMPT,
                alpha=float(spec["alpha"]),
                max_new_tokens=max_new_tokens,
                neg_mode=neg_mode,
                use_full_season=bool(spec["use_full_season"]),
                beta=beta,
            )
        outputs.append(
            {
                "video_id": rec["video_id"],
                "prediction": text,
                "reference": rec["messages"][-1]["content"],
                "severity": rec["severity"],
                "impact": rec["impact"],
                "start_sec": rec["start_sec"],
                "end_sec": rec["end_sec"],
                "vehicles": rec["vehicles"],
                "weather": rec["weather"],
                "explanation": rec["explanation"],
            }
        )
    # Free GPU memory between conditions
    del decoder
    del backbone
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return outputs


def evaluate_condition(outputs: list[dict[str, Any]]) -> dict[str, float]:
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
    nli = evaluate_nli(preds, gt_rows)
    metrics = lexical.as_dict()
    metrics["ArgusCost-H"] = argus["ArgusCost-H"]
    metrics["ArgusCost-O"] = argus["ArgusCost-O"]
    metrics["Explanation-BERTScore"] = argus["Explanation-BERTScore"]
    for k, v in nli.items():
        if not k.startswith("per_sample"):
            metrics[k] = v
    return metrics


METRIC_ORDER = [
    "BLEU-4",
    "ROUGE-L",
    "METEOR",
    "CIDEr",
    "BERTScore",
    "tIoU",
    "NLI-Entailment",
    "NLI-Contradiction",
    "NLI-Score",
    "NLI-Loss",
    "ArgusCost-H",
    "ArgusCost-O",
    "Explanation-BERTScore",
]


def format_markdown_table(results: dict[str, dict[str, float]]) -> str:
    header = "| Model | " + " | ".join(METRIC_ORDER) + " |"
    sep = "|-------|" + "|".join(["------:" for _ in METRIC_ORDER]) + "|"
    lines = [header, sep]
    for name, metrics in results.items():
        cells = [name]
        for m in METRIC_ORDER:
            v = metrics.get(m, float("nan"))
            cells.append(f"{v:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def format_latex_table(results: dict[str, dict[str, float]]) -> str:
    cols = "l" + "c" * len(METRIC_ORDER)
    lines = [
        r"\begin{tabular}{" + cols + "}",
        r"\toprule",
        "Model & " + " & ".join(METRIC_ORDER) + r" \\",
        r"\midrule",
    ]
    for name, metrics in results.items():
        cells = [name.replace("_", r"\_")]
        for m in METRIC_ORDER:
            v = metrics.get(m, float("nan"))
            cells.append(f"{v:.3f}")
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CrashX experiment runner")
    p.add_argument("--test-jsonl", type=Path, default=Path("crashx/data/splits/test.jsonl"))
    p.add_argument("--lora-path", type=Path, default=Path("outputs/crashlogic_7b_lora"))
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--neg-mode", choices=["reverse", "shuffle"], default="reverse")
    p.add_argument("--beta", type=float, default=0.5, help="Temporal homogenization strength")
    p.add_argument("--full-season-ablation", action="store_true")
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="Optional cap on test videos")
    p.add_argument(
        "--only-condition",
        type=str,
        default=None,
        help="Run/score only this condition name (e.g. CrashLogic-7B-SEASON)",
    )
    p.add_argument(
        "--skip-infer",
        action="store_true",
        help="Only re-score existing predictions under results/",
    )
    p.add_argument(
        "--dry-run-metrics",
        action="store_true",
        help="Skip model load; score dummy preds=refs (sanity check for metrics)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.test_jsonl.is_file():
        raise FileNotFoundError(
            f"Missing {args.test_jsonl}. Run: python -m crashx.data.process_ccd"
        )

    records = load_jsonl(args.test_jsonl)
    logger.info("Loaded %d test records from %s", len(records), args.test_jsonl)

    lora_path = args.lora_path if args.lora_path.exists() else None
    if lora_path is None:
        logger.warning("LoRA checkpoint not found at %s", args.lora_path)

    specs = _condition_specs(lora_path, args.alpha, args.full_season_ablation)
    if args.only_condition:
        specs = [s for s in specs if s["name"] == args.only_condition]
        if not specs:
            raise ValueError(f"Unknown condition: {args.only_condition}")
    args.results_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: dict[str, dict[str, float]] = {}
    # Load metrics from existing runs when re-scoring a subset.
    for existing in args.results_dir.glob("*_metrics.json"):
        name = existing.name.replace("_metrics.json", "")
        if args.only_condition and name != args.only_condition:
            try:
                with existing.open("r", encoding="utf-8") as f:
                    all_metrics[name] = json.load(f)
            except Exception:  # noqa: BLE001
                pass

    for spec in specs:
        pred_path = args.results_dir / f"{spec['name']}_predictions.json"

        if args.dry_run_metrics:
            outputs = [
                {
                    "video_id": r["video_id"],
                    "prediction": r["messages"][-1]["content"],
                    "reference": r["messages"][-1]["content"],
                    "severity": r["severity"],
                    "impact": r["impact"],
                    "start_sec": r["start_sec"],
                    "end_sec": r["end_sec"],
                    "vehicles": r["vehicles"],
                    "weather": r["weather"],
                    "explanation": r["explanation"],
                }
                for r in (records[: args.limit] if args.limit else records)
            ]
        elif args.skip_infer and pred_path.is_file():
            with pred_path.open("r", encoding="utf-8") as f:
                outputs = json.load(f)
        else:
            outputs = run_condition(
                spec=spec,
                records=records,
                model_id=args.model_id,
                num_frames=args.num_frames,
                max_new_tokens=args.max_new_tokens,
                load_in_4bit=not args.no_4bit,
                neg_mode=args.neg_mode,
                beta=args.beta,
                limit=args.limit,
            )
            with pred_path.open("w", encoding="utf-8") as f:
                json.dump(outputs, f, indent=2, ensure_ascii=False)
            logger.info("Wrote predictions → %s", pred_path)

        metrics = evaluate_condition(outputs)
        all_metrics[spec["name"]] = metrics
        metrics_path = args.results_dir / f"{spec['name']}_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("%s metrics: %s", spec["name"], metrics)

    md = format_markdown_table(all_metrics)
    latex = format_latex_table(all_metrics)
    print("\n=== CrashX Comparison (Markdown) ===\n")
    print(md)
    print("\n=== CrashX Comparison (LaTeX) ===\n")
    print(latex)

    summary = {
        "metrics": all_metrics,
        "markdown": md,
        "latex": latex,
    }
    summary_path = args.results_dir / "comparison_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    (args.results_dir / "comparison.md").write_text(md + "\n", encoding="utf-8")
    (args.results_dir / "comparison.tex").write_text(latex + "\n", encoding="utf-8")
    logger.info("Saved comparison tables to %s", args.results_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
