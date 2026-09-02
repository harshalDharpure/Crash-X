#!/usr/bin/env python3
"""Run zero-shot foundation-model baselines for CrashX Option A benchmark."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("foundation_baselines")

from crashx.data.dataset import load_jsonl, load_video_frames_cv2
from crashx.models.foundation_backends import (
    FOUNDATION_SPECS,
    build_backend,
    spec_by_name,
)
from crashx.prompts import USER_PROMPT
from crashx.run_experiments import evaluate_condition


def run_foundation_condition(
    condition_name: str,
    backend_key: str,
    records: list[dict[str, Any]],
    num_frames: int,
    max_new_tokens: int,
    load_in_4bit: bool,
    max_side: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    backend = build_backend(
        backend_key,
        load_in_4bit=load_in_4bit,
        max_side=max_side,
    ).load()

    rows = records[:limit] if limit else records
    outputs: list[dict[str, Any]] = []
    for rec in tqdm(rows, desc=condition_name):
        frames = load_video_frames_cv2(rec["video_path"], num_frames=num_frames)
        text = backend.generate_greedy(frames, USER_PROMPT, max_new_tokens=max_new_tokens)
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

    backend.unload()
    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CrashX foundation-model zero-shot baselines")
    p.add_argument("--test-jsonl", type=Path, default=Path("crashx/data/splits/test.jsonl"))
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument(
        "--models",
        nargs="+",
        default=["internvl", "llava-video"],
        choices=["qwen", "qwen2-vl-2b", "qwen2.5-vl-3b", "internvl", "llava-video", "all"],
        help="Which foundation backends to run",
    )
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-side", type=int, default=224)
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force-rerun", action="store_true")
    p.add_argument(
        "--only-condition",
        type=str,
        default=None,
        help="Run a single condition by name (e.g. ZeroShot-InternVL2.5-8B)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.test_jsonl.is_file():
        raise FileNotFoundError(f"Missing test split: {args.test_jsonl}")

    records = load_jsonl(args.test_jsonl)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    model_keys = list(FOUNDATION_SPECS) if "all" in args.models else []
    if not model_keys:
        key_map = {s["backend"]: s for s in FOUNDATION_SPECS}
        for mk in args.models:
            if mk in key_map:
                model_keys.append(key_map[mk])

    if args.only_condition:
        model_keys = [s for s in FOUNDATION_SPECS if s["name"] == args.only_condition]
        if not model_keys:
            raise ValueError(f"Unknown condition: {args.only_condition}")

    for spec in model_keys:
        name = spec["name"]
        pred_path = args.results_dir / f"{name}_predictions.json"
        if pred_path.is_file() and not args.force_rerun:
            logger.info("Skipping %s (predictions exist)", name)
            continue

        logger.info("=== Foundation baseline: %s ===", name)
        outputs = run_foundation_condition(
            condition_name=name,
            backend_key=spec["backend"],
            records=records,
            num_frames=args.num_frames,
            max_new_tokens=args.max_new_tokens,
            load_in_4bit=not args.no_4bit,
            max_side=args.max_side,
            limit=args.limit,
        )

        with pred_path.open("w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
        logger.info("Wrote %s", pred_path)

        metrics = evaluate_condition(outputs)
        with (args.results_dir / f"{name}_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("%s metrics: %s", name, metrics)


if __name__ == "__main__":
    main(sys.argv[1:])
