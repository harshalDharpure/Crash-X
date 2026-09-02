#!/usr/bin/env python3
"""Commercial API baselines for CrashX (GPT-4o / Gemini).

Requires one of:
  export OPENAI_API_KEY=...
  export GEMINI_API_KEY=...   # or GOOGLE_API_KEY

Example:
  PYTHONPATH=. python -m crashx.run_api_baselines --provider openai --limit 50
  PYTHONPATH=. python -m crashx.run_api_baselines --provider gemini --limit 50
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("api_baselines")

from crashx.data.dataset import load_jsonl, load_video_frames_cv2
from crashx.prompts import USER_PROMPT
from crashx.run_experiments import evaluate_condition


STRUCTURED_HINT = (
    " Respond with pipe-delimited fields exactly like: "
    "Severity: ... | Impact: ... | Start: X.XXXs | End: Y.YYYs | "
    "Vehicles: ... | Weather: ... | Explanation: ..."
)


def _frames_to_jpeg_b64(frames, max_side: int = 384, quality: int = 85) -> list[str]:
    out: list[str] = []
    for fr in frames:
        im = Image.fromarray(fr.astype("uint8"))
        w, h = im.size
        scale = max_side / max(w, h)
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
        out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return out


def call_openai(frames_b64: list[str], prompt: str, model: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt + STRUCTURED_HINT}]
    for b64 in frames_b64:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
            }
        )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=512,
        temperature=0,
    )
    return (resp.choices[0].message.content or "").strip()


def call_gemini(frames_b64: list[str], prompt: str, model: str) -> str:
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    gm = genai.GenerativeModel(model)
    parts: list[Any] = [prompt + STRUCTURED_HINT]
    for b64 in frames_b64:
        parts.append({"mime_type": "image/jpeg", "data": base64.b64decode(b64)})
    resp = gm.generate_content(parts, generation_config={"temperature": 0, "max_output_tokens": 512})
    return (getattr(resp, "text", None) or "").strip()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CrashX commercial API baselines")
    p.add_argument("--provider", choices=["openai", "gemini"], required=True)
    p.add_argument("--model", default=None, help="Override model id")
    p.add_argument("--test-jsonl", type=Path, default=Path("crashx/data/splits/test.jsonl"))
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--max-side", type=int, default=384)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sleep", type=float, default=0.5, help="Seconds between API calls")
    p.add_argument("--force-rerun", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set")
        model = args.model or "gpt-4o"
        cond = "ZeroShot-GPT-4o"
        caller = lambda b64, prompt: call_openai(b64, prompt, model)
    else:
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            raise SystemExit("GEMINI_API_KEY or GOOGLE_API_KEY is not set")
        model = args.model or "gemini-1.5-pro"
        cond = "ZeroShot-Gemini-1.5-Pro"
        caller = lambda b64, prompt: call_gemini(b64, prompt, model)

    pred_path = args.results_dir / f"{cond}_predictions.json"
    if pred_path.is_file() and not args.force_rerun:
        logger.info("Skipping %s (exists)", cond)
        return

    records = load_jsonl(args.test_jsonl)
    rows = records[: args.limit] if args.limit else records
    outputs: list[dict[str, Any]] = []
    for rec in tqdm(rows, desc=cond):
        frames = load_video_frames_cv2(rec["video_path"], num_frames=args.num_frames)
        b64 = _frames_to_jpeg_b64(frames, max_side=args.max_side)
        text = caller(b64, USER_PROMPT)
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
        time.sleep(args.sleep)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    with pred_path.open("w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)
    metrics = evaluate_condition(outputs)
    with (args.results_dir / f"{cond}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("%s done → %s metrics=%s", cond, pred_path, metrics)


if __name__ == "__main__":
    main(sys.argv[1:])
