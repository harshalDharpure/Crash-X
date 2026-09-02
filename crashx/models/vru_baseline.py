#!/usr/bin/env python3
"""VRU-Accident-style wrapper for Qwen2.5-VL-7B (base or LoRA CrashLogic-7B).

Provides model/processor loading, 16-frame video preparation, and greedy generation
compatible with the CrashX training and SEASON inference stacks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


def frames_to_pil(frames: np.ndarray | torch.Tensor | Sequence[Any]) -> list[Image.Image]:
    """Convert keyframe array/tensor/list to a list of RGB PIL images."""
    if isinstance(frames, torch.Tensor):
        arr = frames.detach().cpu()
        if arr.ndim == 4 and arr.shape[1] in (1, 3):  # [T,C,H,W]
            arr = arr.permute(0, 2, 3, 1)
        if arr.dtype.is_floating_point:
            arr = (arr.clamp(0, 1) * 255).byte()
        frames_np = arr.numpy()
    elif isinstance(frames, np.ndarray):
        frames_np = frames
        if frames_np.ndim == 4 and frames_np.shape[1] in (1, 3) and frames_np.shape[-1] not in (1, 3):
            frames_np = np.transpose(frames_np, (0, 2, 3, 1))
        if np.issubdtype(frames_np.dtype, np.floating):
            frames_np = (np.clip(frames_np, 0, 1) * 255).astype(np.uint8)
    else:
        return [f if isinstance(f, Image.Image) else Image.fromarray(np.asarray(f)) for f in frames]

    return [Image.fromarray(frames_np[i].astype(np.uint8)) for i in range(frames_np.shape[0])]


class VRUBaselineModel:
    """Thin Video-LLM wrapper following VRU-Accident generation patterns."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        lora_path: str | Path | None = None,
        load_in_4bit: bool = True,
        device_map: str | dict | None = None,
        torch_dtype: torch.dtype | None = None,
        trust_remote_code: bool = True,
        max_side: int = 224,
    ) -> None:
        self.model_id = model_id
        self.lora_path = Path(lora_path) if lora_path else None
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map if device_map is not None else {"": 0}
        self.torch_dtype = torch_dtype or torch.bfloat16
        self.trust_remote_code = trust_remote_code
        self.max_side = max_side
        self.model = None
        self.processor = None

    def load(self) -> "VRUBaselineModel":
        """Load processor + model (optionally with 4-bit quant and LoRA adapters)."""
        from transformers import AutoProcessor, BitsAndBytesConfig

        logger.info("Loading processor from %s", self.model_id)
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
        )

        quant_config = None
        if self.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.torch_dtype,
            )

        model_id_lower = self.model_id.lower()
        if "qwen2-vl" in model_id_lower and "2.5" not in model_id_lower:
            from transformers import Qwen2VLForConditionalGeneration

            model_cls = Qwen2VLForConditionalGeneration
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration

            model_cls = Qwen2_5_VLForConditionalGeneration

        logger.info("Loading model %s (4bit=%s)", self.model_id, self.load_in_4bit)
        self.model = model_cls.from_pretrained(
            self.model_id,
            quantization_config=quant_config,
            device_map=self.device_map,
            torch_dtype=self.torch_dtype if quant_config is None else None,
            trust_remote_code=self.trust_remote_code,
        )

        if self.lora_path is not None:
            from peft import PeftModel

            logger.info("Attaching LoRA adapters from %s", self.lora_path)
            self.model = PeftModel.from_pretrained(self.model, str(self.lora_path))

        self.model.eval()
        return self

    @property
    def device(self) -> torch.device:
        if self.model is None:
            return torch.device("cpu")
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_messages(
        self,
        user_text: str,
        num_frames: int,
        assistant_text: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build Qwen2.5-VL chat messages with a video placeholder."""
        user_content: list[dict[str, Any]] = [
            {"type": "video", "video": f"file:///placeholder.mp4", "nframes": num_frames},
            {"type": "text", "text": user_text},
        ]
        messages = [{"role": "user", "content": user_content}]
        if assistant_text is not None:
            messages.append({"role": "assistant", "content": assistant_text})
        return messages

    def prepare_inputs(
        self,
        frames: np.ndarray | torch.Tensor | Sequence[Any],
        user_text: str,
    ) -> dict[str, Any]:
        """Tokenize prompt + pack video frames for the model forward/generate."""
        if self.processor is None or self.model is None:
            raise RuntimeError("Call load() before prepare_inputs()")

        from crashx.models.video_preprocess import call_video_processor, resize_frames_np

        frames_np = frames if isinstance(frames, np.ndarray) else None
        if frames_np is None and isinstance(frames, torch.Tensor):
            frames_np = frames.detach().cpu().numpy()
        if frames_np is not None and self.max_side > 0:
            frames_np = resize_frames_np(frames_np, max_side=self.max_side)
            pil_frames = frames_to_pil(frames_np)
        else:
            pil_frames = frames_to_pil(frames)
        # Prefer processor video API; fall back to multi-image if needed
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": pil_frames},
                    {"type": "text", "text": user_text},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        try:
            inputs = call_video_processor(
                self.processor,
                [prompt],
                [pil_frames],
            )
        except (TypeError, ValueError):
            # Older processors may expect images= for frame lists
            inputs = self.processor(
                text=[prompt],
                images=pil_frames,
                padding=True,
                return_tensors="pt",
            )

        return {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    @torch.inference_mode()
    def generate_greedy(
        self,
        frames: np.ndarray | torch.Tensor | Sequence[Any],
        user_text: str,
        max_new_tokens: int = 512,
    ) -> str:
        """Standard greedy decoding (baseline / FT without SEASON)."""
        inputs = self.prepare_inputs(frames, user_text)
        input_len = inputs["input_ids"].shape[-1]
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        gen = output_ids[0, input_len:]
        text = self.processor.tokenizer.decode(gen, skip_special_tokens=True)
        return text.strip()

    def forward_logits(
        self,
        inputs: dict[str, Any],
    ) -> torch.Tensor:
        """Return next-token logits [vocab] for contrastive decoding steps."""
        outputs = self.model(**inputs)
        return outputs.logits[:, -1, :]
