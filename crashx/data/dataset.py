#!/usr/bin/env python3
"""PyTorch dataset: uniform 16-keyframe spatiotemporal sampler for CCD videos."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def sample_uniform_indices(num_frames: int, num_samples: int = 16) -> np.ndarray:
    """Uniformly spaced frame indices in [0, num_frames-1]."""
    if num_frames <= 0:
        raise ValueError("Video has no frames")
    if num_frames <= num_samples:
        # Repeat last frame if clip is shorter than requested keyframes
        idx = np.linspace(0, num_frames - 1, num=num_samples)
        return np.round(idx).astype(np.int64).clip(0, num_frames - 1)
    idx = np.linspace(0, num_frames - 1, num=num_samples)
    return np.round(idx).astype(np.int64)


def load_video_frames_cv2(
    video_path: str | Path,
    num_frames: int = 16,
) -> np.ndarray:
    """Load a video and return RGB keyframes as uint8 array [T, H, W, 3]."""
    if cv2 is None:
        raise ImportError("opencv-python is required for video loading")

    path = str(video_path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    # Fallback: count by reading if metadata is missing
    if total <= 0:
        frames_buf = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames_buf.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames_buf:
            raise RuntimeError(f"Empty video: {path}")
        arr = np.stack(frames_buf, axis=0)
        indices = sample_uniform_indices(arr.shape[0], num_frames)
        return arr[indices]

    indices = sample_uniform_indices(total, num_frames)
    out = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok or frame is None:
            # Retry sequential read from start as fallback
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(i) - 1))
            ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {i} from {path}")
        out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(out, axis=0)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class CrashDataset(Dataset):
    """CCD instruction dataset with uniform 16-keyframe visual sampling.

    Each item returns:
      - frames: float tensor [T, C, H, W] in [0, 1] (or raw uint8 numpy if transform=None
        and as_tensor=False)
      - metadata dict with GT fields and chat messages
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        num_frames: int = 16,
        transform: Callable[[np.ndarray], Any] | None = None,
        as_tensor: bool = True,
        require_video: bool = True,
    ) -> None:
        self.records = load_jsonl(jsonl_path)
        self.num_frames = num_frames
        self.transform = transform
        self.as_tensor = as_tensor
        self.require_video = require_video
        if require_video:
            missing = [r["video_id"] for r in self.records if not Path(r["video_path"]).is_file()]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} videos missing (e.g. {missing[:3]}). "
                    "Re-run process_ccd or set require_video=False."
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.records[index]
        frames = load_video_frames_cv2(rec["video_path"], num_frames=self.num_frames)
        if self.transform is not None:
            frames = self.transform(frames)
        elif self.as_tensor:
            # [T,H,W,C] uint8 → [T,C,H,W] float32
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

        return {
            "video_id": rec["video_id"],
            "video_path": rec["video_path"],
            "frames": frames,
            "messages": rec["messages"],
            "severity": rec["severity"],
            "impact": rec["impact"],
            "start_sec": rec["start_sec"],
            "end_sec": rec["end_sec"],
            "vehicles": rec["vehicles"],
            "weather": rec["weather"],
            "explanation": rec["explanation"],
            "n_vehicles": rec.get("n_vehicles", ""),
            "target_text": rec["messages"][-1]["content"],
        }


def collate_crash_batch(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Simple list collate (variable spatial sizes); keep frames as a list."""
    return {
        "video_id": [b["video_id"] for b in batch],
        "video_path": [b["video_path"] for b in batch],
        "frames": [b["frames"] for b in batch],
        "messages": [b["messages"] for b in batch],
        "target_text": [b["target_text"] for b in batch],
        "severity": [b["severity"] for b in batch],
        "impact": [b["impact"] for b in batch],
        "start_sec": [b["start_sec"] for b in batch],
        "end_sec": [b["end_sec"] for b in batch],
        "vehicles": [b["vehicles"] for b in batch],
        "weather": [b["weather"] for b in batch],
        "explanation": [b["explanation"] for b in batch],
    }
