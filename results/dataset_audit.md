# CrashX Dataset Partition Audit

## Summary: **GOOD — no changes required for publication**

The current split is valid for a Q1/A* submission. Minor notes are documented below.

## Split configuration

| Split | Videos | Purpose |
|-------|--------|---------|
| Train | 1,198 | QLoRA fine-tuning |
| Val   | 150   | Held-out (reserved; not used in final 5-epoch run due to VRAM) |
| Test  | 150   | All reported metrics |

- **Total clean:** 1,498 / 1,500 Excel rows (2 skipped: unparseable timestamps)
- **Seed:** 42 (reproducible)
- **Strategy:** Stratified 80/10/10 by crash severity

## Leakage check

| Check | Result |
|-------|--------|
| train ∩ test video IDs | **0** |
| train ∩ val video IDs | **0** |
| val ∩ test video IDs | **0** |

## Severity proportions (stratification quality)

| Severity | Train | Val | Test |
|----------|-------|-----|------|
| moderate | 46.2% | 46.0% | 46.0% |
| minor    | 25.3% | 25.3% | 25.3% |
| severe   | 20.1% | 20.0% | 20.0% |
| fatal    | 2.8%  | 3.3%  | 2.7%  |
| n/a      | 5.5%  | 5.3%  | 6.0%  |

Proportions match across splits — stratification is working correctly.

## Minor notes (not blockers)

1. **Train size 1198 vs 1200:** 2 Excel rows dropped for bad timestamps; split auto-adjusted.
2. **Fatal class is small** (4 test videos): report confidence intervals or merge fatal+severe in ablation if reviewers ask.
3. **Val set unused in training:** final model used all train epochs without val early-stop; acceptable but val could be used for hyperparameter tuning in revision.
4. **n/a severity (9 test):** temporal metrics are near-zero for these; consider reporting metrics with/without n/a subset.

## Recommendation

**Keep current partition.** Optional sensitivity analysis: re-run metrics excluding `n/a` severity (9 videos) — can add as supplementary table if needed.
