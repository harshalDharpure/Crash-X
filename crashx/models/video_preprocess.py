"""Shared Qwen2.5-VL video preprocessing helpers (resize + pixel cap)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from PIL import Image


def resize_pil_frames(frames: Sequence[Image.Image], max_side: int = 384) -> list[Image.Image]:
    """Resize frames so the longer side is at most ``max_side`` pixels."""
    if max_side <= 0:
        return list(frames)
    out: list[Image.Image] = []
    for img in frames:
        w, h = img.size
        longest = max(w, h)
        if longest <= max_side:
            out.append(img)
            continue
        scale = max_side / float(longest)
        out.append(img.resize((int(w * scale), int(h * scale)), Image.BILINEAR))
    return out


def resize_frames_np(frames: np.ndarray, max_side: int = 384) -> np.ndarray:
    """Resize uint8 [T,H,W,C] frames to cap the longer spatial side."""
    if max_side <= 0:
        return frames
    from crashx.models.vru_baseline import frames_to_pil

    pil = frames_to_pil(frames)
    resized = resize_pil_frames(pil, max_side=max_side)
    return np.stack([np.asarray(img) for img in resized], axis=0)


def processor_video_kwargs() -> dict[str, Any]:
    """Extra kwargs for Qwen2.5-VL video tokenization to limit per-frame pixels."""
    return {"cap_pixels_per_frame": True}


def call_video_processor(
    processor,
    texts: list[str],
    videos: list[list[Image.Image]],
    *,
    padding: bool = True,
    return_tensors: str = "pt",
) -> dict[str, Any]:
    """Call the processor with video pixel-cap enabled when supported."""
    kwargs = processor_video_kwargs()
    try:
        return processor(
            text=texts,
            videos=videos,
            padding=padding,
            return_tensors=return_tensors,
            **kwargs,
        )
    except TypeError:
        return processor(
            text=texts,
            videos=videos,
            padding=padding,
            return_tensors=return_tensors,
        )
