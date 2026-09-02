#!/usr/bin/env python3
"""Re-score all existing prediction JSONs (NLI + full metrics) without VLM inference."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rescore_metrics")

from crashx.eval.journal_tables import discover_conditions, evaluate_outputs, write_tables, generate_all_tables
from crashx.eval.nli_eval import NLIScorer
from crashx.run_experiments import evaluate_condition


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    p.add_argument("--regen-tables", action="store_true", default=True)
    args = p.parse_args(argv)

    conditions = discover_conditions(args.results_dir)
    logger.info("Re-scoring %d conditions: %s", len(conditions), conditions)

    nli = NLIScorer()
    for cond in conditions:
        pred_path = args.results_dir / f"{cond}_predictions.json"
        with pred_path.open("r", encoding="utf-8") as f:
            outputs = json.load(f)
        metrics = evaluate_outputs(outputs, nli_scorer=nli)
        # Also keep run_experiments-compatible full eval
        metrics_path = args.results_dir / f"{cond}_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("%s → NLI-Entail=%.3f NLI-Loss=%.3f", cond, metrics.get("NLI-Entailment", 0), metrics.get("NLI-Loss", 0))

        # Per-sample detailed NLI
        detail_path = args.results_dir / f"{cond}_detailed.json"
        nli_detail = evaluate_condition(outputs)
        detail = {
            "metrics": nli_detail,
            "n_samples": len(outputs),
        }
        with detail_path.open("w", encoding="utf-8") as f:
            json.dump(detail, f, indent=2)

    nli.unload()

    if args.regen_tables:
        report = generate_all_tables(args.results_dir)
        write_tables(report, args.tables_dir)
        logger.info("Regenerated tables → %s", args.tables_dir / "all_tables.md")


if __name__ == "__main__":
    main(sys.argv[1:])
