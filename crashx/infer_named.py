#!/usr/bin/env python3
"""Run CrashLogic inference under a distinct condition name (e.g. *-f16)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("infer_named")

from crashx.data.dataset import load_jsonl
from crashx.run_experiments import evaluate_condition, run_condition
from crashx.run_journal_experiments import run_ablation_condition
from crashx.models.vru_baseline import DEFAULT_MODEL_ID


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--test-jsonl", type=Path, default=Path("crashx/data/splits/test.jsonl"))
    p.add_argument("--lora-path", type=Path, required=True)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--name", type=str, required=True)
    p.add_argument("--decode", choices=["greedy", "season"], default="greedy")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.test_jsonl)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    pred_path = args.results_dir / f"{args.name}_predictions.json"
    if pred_path.is_file():
        logger.info("Skip existing %s", pred_path)
        return

    if args.decode == "greedy":
        spec = {
            "name": args.name,
            "lora": str(args.lora_path),
            "decode": "greedy",
            "alpha": 0.0,
            "use_full_season": False,
        }
        outputs = run_condition(
            spec=spec,
            records=records,
            model_id=DEFAULT_MODEL_ID,
            num_frames=args.num_frames,
            max_new_tokens=args.max_new_tokens,
            load_in_4bit=True,
            neg_mode="reverse",
            beta=0.5,
            limit=args.limit,
        )
    else:
        spec = {
            "name": args.name,
            "lora": str(args.lora_path),
            "decode": "season",
            "alpha": args.alpha,
            "use_full_season": False,
            "neg_mode": "reverse",
        }
        outputs = run_ablation_condition(
            spec=spec,
            records=records,
            model_id=DEFAULT_MODEL_ID,
            num_frames=args.num_frames,
            max_new_tokens=args.max_new_tokens,
            load_in_4bit=True,
            beta=0.5,
            limit=args.limit,
        )

    with pred_path.open("w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)
    metrics = evaluate_condition(outputs)
    with (args.results_dir / f"{args.name}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("%s → tIoU=%.3f ArgusO=%.3f", args.name, metrics.get("tIoU", 0), metrics.get("ArgusCost-O", 0))


if __name__ == "__main__":
    main()
