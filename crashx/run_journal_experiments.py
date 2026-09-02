#!/usr/bin/env python3
"""CrashX journal experiment suite: multi-table results for publication.

Generates 5–6 LaTeX/Markdown tables:
  I.   Main captioning quality
  II.  Temporal & forensic reasoning
  III. Structured field accuracy
  IV.  Severity-stratified breakdown
  V.   Relative gains over zero-shot
  VI.  SEASON ablation (after extra inference runs)

Usage:
  # Tables only from existing predictions (fast)
  python -m crashx.run_journal_experiments --tables-only

  # Run missing ablations + regenerate all tables
  python -m crashx.run_journal_experiments --run-ablations
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
logger = logging.getLogger("journal_experiments")

from crashx.data.dataset import load_jsonl, load_video_frames_cv2
from crashx.eval.journal_tables import generate_all_tables, write_tables
from crashx.inference.season_decoder import SeasonDecoder
from crashx.models.vru_baseline import DEFAULT_MODEL_ID, VRUBaselineModel
from crashx.prompts import USER_PROMPT
from crashx.run_experiments import evaluate_condition


def _ablation_specs(lora_path: str) -> list[dict[str, Any]]:
    """Extra SEASON variants for Table VI."""
    return [
        {
            "name": "CrashLogic-7B-SEASON-a0.5",
            "lora": lora_path,
            "decode": "season",
            "alpha": 0.5,
            "use_full_season": False,
            "neg_mode": "reverse",
        },
        {
            "name": "CrashLogic-7B-SEASON-a1.5",
            "lora": lora_path,
            "decode": "season",
            "alpha": 1.5,
            "use_full_season": False,
            "neg_mode": "reverse",
        },
        {
            "name": "CrashLogic-7B-SEASON-a2.0",
            "lora": lora_path,
            "decode": "season",
            "alpha": 2.0,
            "use_full_season": False,
            "neg_mode": "reverse",
        },
        {
            "name": "CrashLogic-7B-SEASON-shuffle",
            "lora": lora_path,
            "decode": "season",
            "alpha": 1.0,
            "use_full_season": False,
            "neg_mode": "shuffle",
        },
        {
            "name": "CrashLogic-7B-SEASON-Full",
            "lora": lora_path,
            "decode": "season",
            "alpha": 1.0,
            "use_full_season": True,
            "neg_mode": "reverse",
        },
    ]


def run_ablation_condition(
    spec: dict[str, Any],
    records: list[dict[str, Any]],
    model_id: str,
    num_frames: int,
    max_new_tokens: int,
    load_in_4bit: bool,
    beta: float,
    limit: int | None,
) -> list[dict[str, Any]]:
    logger.info("=== Ablation: %s ===", spec["name"])
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
        text = decoder.decode_season(
            frames,
            USER_PROMPT,
            alpha=float(spec["alpha"]),
            max_new_tokens=max_new_tokens,
            neg_mode=spec.get("neg_mode", "reverse"),
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

    del decoder
    del backbone
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CrashX journal multi-table experiments")
    p.add_argument("--test-jsonl", type=Path, default=Path("crashx/data/splits/test.jsonl"))
    p.add_argument("--lora-path", type=Path, default=Path("outputs/crashlogic_7b_lora"))
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--tables-only",
        action="store_true",
        help="Only generate tables from existing prediction JSONs",
    )
    p.add_argument(
        "--run-ablations",
        action="store_true",
        help="Run missing SEASON ablation conditions then generate tables",
    )
    p.add_argument(
        "--force-rerun",
        action="store_true",
        help="Re-run ablations even if predictions already exist",
    )
    p.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Bootstrap resamples for Table VIII CIs",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    if args.run_ablations:
        if not args.lora_path.exists():
            raise FileNotFoundError(f"LoRA not found: {args.lora_path}")
        records = load_jsonl(args.test_jsonl)
        lora = str(args.lora_path)

        for spec in _ablation_specs(lora):
            pred_path = args.results_dir / f"{spec['name']}_predictions.json"
            if not args.force_rerun and pred_path.is_file():
                logger.info("Skipping %s (predictions exist)", spec["name"])
                continue

            outputs = run_ablation_condition(
                spec=spec,
                records=records,
                model_id=args.model_id,
                num_frames=args.num_frames,
                max_new_tokens=args.max_new_tokens,
                load_in_4bit=not args.no_4bit,
                beta=args.beta,
                limit=args.limit,
            )
            with pred_path.open("w", encoding="utf-8") as f:
                json.dump(outputs, f, indent=2, ensure_ascii=False)
            logger.info("Wrote %s", pred_path)

            metrics = evaluate_condition(outputs)
            with (args.results_dir / f"{spec['name']}_metrics.json").open("w") as f:
                json.dump(metrics, f, indent=2)

    if not args.tables_only and not args.run_ablations:
        logger.info("No action specified; defaulting to --tables-only")
        args.tables_only = True

    logger.info("Generating journal tables → %s", args.tables_dir)
    report = generate_all_tables(args.results_dir, n_bootstrap=args.n_bootstrap)
    write_tables(report, args.tables_dir)

    for key, tbl in report["tables"].items():
        print(f"\n{'=' * 60}\n{tbl['title']}\n{'=' * 60}\n")
        print(tbl["markdown"])

    logger.info("Saved %d tables to %s", len(report["tables"]), args.tables_dir)
    logger.info("Combined: %s", args.tables_dir / "all_tables.md")


if __name__ == "__main__":
    main(sys.argv[1:])
