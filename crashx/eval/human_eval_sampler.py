#!/usr/bin/env python3
"""Sample stratified videos for human evaluation (Phase 4)."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from crashx.data.dataset import load_jsonl


def stratified_sample(
    records: list[dict],
    targets: dict[str, int],
    seed: int = 42,
) -> list[dict]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        sev = (r.get("severity") or "n/a").strip().lower()
        buckets[sev].append(r)

    out: list[dict] = []
    for sev, n in targets.items():
        pool = buckets.get(sev, [])
        if len(pool) < n:
            raise ValueError(f"Not enough {sev} videos: need {n}, have {len(pool)}")
        out.extend(rng.sample(pool, n))
    rng.shuffle(out)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Sample videos for human evaluation")
    p.add_argument("--test-jsonl", type=Path, default=Path("crashx/data/splits/test.jsonl"))
    p.add_argument("--out", type=Path, default=Path("results/human_eval/sample_50.jsonl"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    records = load_jsonl(args.test_jsonl)
    targets = {"minor": 15, "moderate": 15, "severe": 12, "fatal": 4, "n/a": 4}
    sample = stratified_sample(records, targets, seed=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(sample)} videos → {args.out}")


if __name__ == "__main__":
    main()
