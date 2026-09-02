# 03 — Dataset & Splits

## Car Crash Dataset (CCD)

| Property | Value |
|----------|-------|
| Source | Car Crash Text Dataset ground truth Excel |
| Total videos | 1,500 (`video1500/000001.mp4` … `0001500.mp4`) |
| Clean records | **1,498** (2 skipped: unparseable timestamps) |
| GT fields | Severity, Impact, Vehicles, Start/End time, Weather, Explanation |
| Task | Dense structured crash captioning |

## Split Configuration

| Split | Videos | % | Purpose |
|-------|--------|---|---------|
| **Train** | 1,198 | 80.0% | QLoRA fine-tuning |
| **Val** | 150 | 10.0% | Held-out (reserved; not used for early-stop in final run) |
| **Test** | 150 | 10.0% | **All reported metrics** |

- **Seed:** 42 (reproducible)
- **Strategy:** Stratified 80/10/10 by crash severity (`sklearn.StratifiedShuffleSplit`)

## Leakage Verification

| Check | Result |
|-------|--------|
| train ∩ test video IDs | **0** ✅ |
| train ∩ val video IDs | **0** ✅ |
| val ∩ test video IDs | **0** ✅ |

**Verdict: No data leakage.**

## Severity Distribution (Stratification Quality)

| Severity | Train | Val | Test | Total |
|----------|-------|-----|------|-------|
| moderate | 554 (46.2%) | 69 (46.0%) | 69 (46.0%) | 692 |
| minor | 303 (25.3%) | 38 (25.3%) | 38 (25.3%) | 379 |
| severe | 241 (20.1%) | 30 (20.0%) | 30 (20.0%) | 301 |
| fatal | 34 (2.8%) | 5 (3.3%) | 4 (2.7%) | 43 |
| n/a | 66 (5.5%) | 8 (5.3%) | 9 (6.0%) | 83 |

Proportions match across splits within ±0.3% — stratification is correct.

## Test Set Breakdown (n=150)

Used for all Tables I–VII.

## Publication Notes

1. **Train 1198 vs 1200:** 2 Excel rows dropped for bad timestamps; split auto-adjusted.
2. **Fatal class (4 test videos):** Small sample — report with caution or merge fatal+severe in sensitivity analysis.
3. **n/a severity (9 test):** No defined crash window — tIoU near zero expected; optional supplementary table excluding n/a.
4. **Val unused:** Final 5-epoch run did not use val for early stopping (VRAM constraints). Acceptable; mention in limitations.

## Recommendation

**Keep current partition** — valid for Q1/A* submission. No changes required.
