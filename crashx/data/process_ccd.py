#!/usr/bin/env python3
"""Parse CCD Excel ground truth and emit stratified train/val/test JSONL splits.

Default split sizes: 1200 train / 150 val / 150 test (80/10/10), stratified by
crash severity. Timestamps are converted from Excel day-fractions to seconds.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from crashx.prompts import USER_PROMPT, format_target_text

# Canonical column names after rename
COLUMN_MAP = {
    "Video Number": "video_number",
    "Severity of the Crash": "severity",
    "Type of Vehicles involved": "vehicles",
    "No. of Vehicles involved": "n_vehicles",
    "Location of impact": "impact",
    "Start of Crash": "start_raw",
    "End of Crash": "end_raw",
    "Explanation": "explanation",
    "Ambiguity": "ambiguity",
    "Camera View": "camera_view",
    "Weather Conditions": "weather",
}

SEVERITY_ALIASES = {
    "moderate": "moderate",
    "minor": "minor",
    "severe": "severe",
    "fatal": "fatal",
    "n/a": "n/a",
    "na": "n/a",
    "nan": "n/a",
}


def parse_timestamp_to_seconds(value: Any) -> float | None:
    """Convert Excel day-fraction, numeric seconds, or time-like strings to seconds."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        x = float(value)
        # Excel serial day fraction for sub-minute times is typically << 1
        if 0 < x < 1.0:
            return x * 86400.0
        return x
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "na", "n/a"}:
        return None
    # Dirty forms like "0;00:02" or "0:00:02"
    s_norm = s.replace(";", ":")
    m = re.match(r"^(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)$", s_norm)
    if m:
        hours = int(m.group(1) or 0)
        minutes = int(m.group(2))
        seconds = float(m.group(3))
        return hours * 3600 + minutes * 60 + seconds
    try:
        x = float(s)
        if 0 < x < 1.0:
            return x * 86400.0
        return x
    except ValueError:
        return None


def normalize_severity(value: Any) -> str:
    s = str(value).strip().lower() if value is not None else "n/a"
    return SEVERITY_ALIASES.get(s, s if s else "n/a")


def video_id_to_filename(video_number: Any) -> str:
    """Map Excel video number to zero-padded mp4 name (e.g. 1 -> 000001.mp4)."""
    raw = str(video_number).strip()
    # Handle floats like "1.0" and already-padded ids like "000897"
    if re.fullmatch(r"\d+\.0+", raw):
        raw = raw.split(".")[0]
    if re.fullmatch(r"\d+", raw):
        return f"{int(raw):06d}.mp4"
    digits = re.sub(r"\D", "", raw)
    if digits:
        return f"{int(digits):06d}.mp4"
    raise ValueError(f"Cannot map video number to filename: {video_number!r}")


def load_and_clean(
    excel_path: Path,
    video_dir: Path,
) -> pd.DataFrame:
    df = pd.read_excel(excel_path, engine="openpyxl")
    # Drop fully empty unnamed columns
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    missing = [c for c in COLUMN_MAP if c not in df.columns]
    if missing:
        raise KeyError(f"Excel missing expected columns: {missing}")

    df = df.rename(columns=COLUMN_MAP)
    records = []
    skipped = []
    for _, row in df.iterrows():
        try:
            fname = video_id_to_filename(row["video_number"])
        except ValueError as exc:
            skipped.append((row.get("video_number"), str(exc)))
            continue
        video_path = video_dir / fname
        start_sec = parse_timestamp_to_seconds(row["start_raw"])
        end_sec = parse_timestamp_to_seconds(row["end_raw"])
        if start_sec is None or end_sec is None:
            skipped.append((fname, "unparseable timestamps"))
            continue
        if end_sec < start_sec:
            start_sec, end_sec = end_sec, start_sec
        severity = normalize_severity(row["severity"])
        vehicles = str(row.get("vehicles") or "").strip()
        impact = str(row.get("impact") or "").strip()
        weather = str(row.get("weather") or "unknown").strip().lower()
        explanation = str(row.get("explanation") or "").strip()
        n_vehicles = row.get("n_vehicles")
        if pd.isna(n_vehicles):
            n_vehicles = ""
        else:
            try:
                n_vehicles = str(int(float(n_vehicles)))
            except (TypeError, ValueError):
                n_vehicles = str(n_vehicles).strip()

        target = format_target_text(
            severity=severity,
            impact=impact,
            start_sec=start_sec,
            end_sec=end_sec,
            vehicles=vehicles,
            weather=weather,
            explanation=explanation,
            n_vehicles=n_vehicles,
        )
        records.append(
            {
                "video_id": Path(fname).stem,
                "video_path": str(video_path.resolve()),
                "video_exists": video_path.is_file(),
                "severity": severity,
                "vehicles": vehicles,
                "n_vehicles": n_vehicles,
                "impact": impact,
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "weather": weather,
                "explanation": explanation,
                "ambiguity": str(row.get("ambiguity") or "").strip(),
                "camera_view": str(row.get("camera_view") or "").strip(),
                "messages": [
                    {"role": "user", "content": USER_PROMPT},
                    {"role": "assistant", "content": target},
                ],
            }
        )

    out = pd.DataFrame(records)
    if skipped:
        print(f"[process_ccd] Skipped {len(skipped)} rows (showing up to 5): {skipped[:5]}")
    missing_vids = (~out["video_exists"]).sum()
    if missing_vids:
        print(f"[process_ccd] Warning: {missing_vids} videos missing under {video_dir}")
    return out


def stratified_split(
    df: pd.DataFrame,
    train_n: int = 1200,
    val_n: int = 150,
    test_n: int = 150,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Stratified 80/10/10 split by severity with exact target counts when possible."""
    n = len(df)
    target_total = train_n + val_n + test_n
    if n < target_total:
        # Scale proportionally if rows were dropped
        ratio = n / target_total
        train_n = max(1, int(round(train_n * ratio)))
        val_n = max(1, int(round(val_n * ratio)))
        test_n = n - train_n - val_n
        print(
            f"[process_ccd] Adjusted split to {train_n}/{val_n}/{test_n} "
            f"(only {n} clean rows)"
        )

    y = df["severity"].astype(str).values
    indices = np.arange(n)

    # First peel off test (~10%)
    test_frac = test_n / n
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    train_val_idx, test_idx = next(sss1.split(indices, y))

    # Then peel val from remaining so val ≈ val_n
    remain = n - len(test_idx)
    val_frac = val_n / remain
    y_tv = y[train_val_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_rel, va_rel = next(sss2.split(train_val_idx, y_tv))
    train_idx = train_val_idx[tr_rel]
    val_idx = train_val_idx[va_rel]

    # Trim/pad to exact sizes if off-by-one from stratification
    def _fit(idx: np.ndarray, k: int, pool: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx)
        if len(idx) == k:
            return idx
        if len(idx) > k:
            rng = np.random.RandomState(seed)
            return rng.choice(idx, size=k, replace=False)
        need = k - len(idx)
        leftover = np.setdiff1d(pool, idx, assume_unique=False)
        rng = np.random.RandomState(seed)
        extra = rng.choice(leftover, size=min(need, len(leftover)), replace=False)
        return np.concatenate([idx, extra])

    all_idx = indices
    test_idx = _fit(test_idx, min(test_n, n), all_idx)
    used = set(test_idx.tolist())
    pool_tv = np.array([i for i in all_idx if i not in used])
    val_idx = _fit(val_idx, min(val_n, len(pool_tv)), pool_tv)
    used |= set(val_idx.tolist())
    pool_tr = np.array([i for i in all_idx if i not in used])
    train_idx = _fit(train_idx, min(train_n, len(pool_tr)), pool_tr)

    return {
        "train": df.iloc[train_idx].reset_index(drop=True),
        "val": df.iloc[val_idx].reset_index(drop=True),
        "test": df.iloc[test_idx].reset_index(drop=True),
    }


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            rec = row.to_dict()
            # numpy / pandas scalars → python
            for k, v in list(rec.items()):
                if isinstance(v, (np.floating,)):
                    rec[k] = float(v)
                elif isinstance(v, (np.integer,)):
                    rec[k] = int(v)
                elif isinstance(v, (np.bool_,)):
                    rec[k] = bool(v)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Process CCD Excel into CrashX JSONL splits")
    parser.add_argument(
        "--excel",
        type=Path,
        default=Path("Car_Crash_Text_Dataset_ground_truth.xlsx"),
        help="Path to ground-truth Excel file",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("video1500"),
        help="Directory containing 000001.mp4 … videos",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("crashx/data/splits"),
        help="Output directory for JSONL + manifest",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-n", type=int, default=1200)
    parser.add_argument("--val-n", type=int, default=150)
    parser.add_argument("--test-n", type=int, default=150)
    args = parser.parse_args()

    df = load_and_clean(args.excel, args.video_dir)
    print(f"[process_ccd] Clean rows: {len(df)}")
    print(f"[process_ccd] Severity counts:\n{df['severity'].value_counts()}")

    splits = stratified_split(
        df,
        train_n=args.train_n,
        val_n=args.val_n,
        test_n=args.test_n,
        seed=args.seed,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"seed": args.seed, "splits": {}}
    for name, sdf in splits.items():
        out_path = args.out_dir / f"{name}.jsonl"
        write_jsonl(sdf, out_path)
        manifest["splits"][name] = {
            "path": str(out_path),
            "n": len(sdf),
            "video_ids": sdf["video_id"].tolist(),
            "severity_counts": sdf["severity"].value_counts().to_dict(),
        }
        print(f"[process_ccd] Wrote {len(sdf)} → {out_path}")

    man_path = args.out_dir / "manifest.json"
    with man_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[process_ccd] Manifest → {man_path}")


if __name__ == "__main__":
    main()
