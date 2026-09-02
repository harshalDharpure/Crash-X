#!/usr/bin/env python3
"""Multi-model zero-shot backends for CrashX foundation-model benchmark (Option A)."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from crashx.models.vru_baseline import DEFAULT_MODEL_ID, VRUBaselineModel, frames_to_pil

logger = logging.getLogger(__name__)


class FoundationBackend(ABC):
    """Common interface for zero-shot Video-LLM inference."""

    name: str
    model_id: str

    @abstractmethod
    def load(self) -> FoundationBackend:
        ...

    @abstractmethod
    def generate_greedy(
        self,
        frames: np.ndarray,
        user_text: str,
        max_new_tokens: int = 256,
    ) -> str:
        ...

    def unload(self) -> None:
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class QwenVLBackend(FoundationBackend):
    """Qwen2.5-VL-7B-Instruct (existing CrashX stack)."""

    name = "ZeroShot-Qwen2.5-VL-7B"
    model_id = DEFAULT_MODEL_ID

    def __init__(
        self,
        load_in_4bit: bool = True,
        max_side: int = 224,
        device_map: str | dict | None = None,
    ) -> None:
        self.load_in_4bit = load_in_4bit
        self.max_side = max_side
        self.device_map = device_map
        self._wrapper: VRUBaselineModel | None = None

    def load(self) -> QwenVLBackend:
        self._wrapper = VRUBaselineModel(
            model_id=self.model_id,
            lora_path=None,
            load_in_4bit=self.load_in_4bit,
            max_side=self.max_side,
            device_map=self.device_map,
        ).load()
        return self

    def generate_greedy(
        self,
        frames: np.ndarray,
        user_text: str,
        max_new_tokens: int = 256,
    ) -> str:
        if self._wrapper is None:
            raise RuntimeError("Call load() first")
        return self._wrapper.generate_greedy(frames, user_text, max_new_tokens=max_new_tokens)


class Qwen2VL2BBackend(FoundationBackend):
    """Qwen2-VL-2B-Instruct (smaller Qwen family baseline)."""

    name = "ZeroShot-Qwen2-VL-2B"
    model_id = "Qwen/Qwen2-VL-2B-Instruct"

    def __init__(
        self,
        load_in_4bit: bool = True,
        max_side: int = 224,
        device_map: str | dict | None = None,
    ) -> None:
        self.load_in_4bit = load_in_4bit
        self.max_side = max_side
        self.device_map = device_map
        self._wrapper: VRUBaselineModel | None = None

    def load(self) -> Qwen2VL2BBackend:
        self._wrapper = VRUBaselineModel(
            model_id=self.model_id,
            lora_path=None,
            load_in_4bit=self.load_in_4bit,
            max_side=self.max_side,
            device_map=self.device_map,
        ).load()
        return self

    def generate_greedy(
        self,
        frames: np.ndarray,
        user_text: str,
        max_new_tokens: int = 256,
    ) -> str:
        if self._wrapper is None:
            raise RuntimeError("Call load() first")
        return self._wrapper.generate_greedy(frames, user_text, max_new_tokens=max_new_tokens)


class Qwen25VL3BBackend(FoundationBackend):
    """Qwen2.5-VL-3B-Instruct (mid-size open VLM baseline).

    Replaces InternVL in the foundation suite: InternVL2 remote code is incompatible
    with transformers>=4.50 (GenerationMixin / DynamicCache). Qwen2.5-VL-3B runs on the
    same proven CrashX stack as the 7B zero-shot baseline.
    """

    name = "ZeroShot-Qwen2.5-VL-3B"
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

    def __init__(
        self,
        load_in_4bit: bool = True,
        max_side: int = 224,
        device_map: str | dict | None = None,
    ) -> None:
        self.load_in_4bit = load_in_4bit
        self.max_side = max_side
        self.device_map = device_map
        self._wrapper: VRUBaselineModel | None = None

    def load(self) -> Qwen25VL3BBackend:
        self._wrapper = VRUBaselineModel(
            model_id=self.model_id,
            lora_path=None,
            load_in_4bit=self.load_in_4bit,
            max_side=self.max_side,
            device_map=self.device_map,
        ).load()
        return self

    def generate_greedy(
        self,
        frames: np.ndarray,
        user_text: str,
        max_new_tokens: int = 256,
    ) -> str:
        if self._wrapper is None:
            raise RuntimeError("Call load() first")
        return self._wrapper.generate_greedy(frames, user_text, max_new_tokens=max_new_tokens)


class InternVLBackend(FoundationBackend):
    """Deprecated stub: InternVL2 remote code breaks on transformers>=4.50.

    Use Qwen25VL3BBackend (`qwen2.5-vl-3b`) instead.
    """

    name = "ZeroShot-InternVL2.5-8B"
    model_id = "OpenGVLab/InternVL2-8B"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "InternVL2 is incompatible with this transformers version "
            "(GenerationMixin/DynamicCache). Use backend 'qwen2.5-vl-3b' instead."
        )

    def load(self) -> InternVLBackend:
        raise RuntimeError("InternVL2 backend disabled")

    def generate_greedy(
        self,
        frames: np.ndarray,
        user_text: str,
        max_new_tokens: int = 256,
    ) -> str:
        raise RuntimeError("InternVL2 backend disabled")


class LLaVAVideoBackend(FoundationBackend):
    """LLaVA-NeXT-Video-7B backend."""

    name = "ZeroShot-LLaVA-Video-7B"
    model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"

    def __init__(
        self,
        load_in_4bit: bool = True,
        max_side: int = 224,
        device_map: str | dict | None = None,
    ) -> None:
        self.load_in_4bit = load_in_4bit
        self.max_side = max_side
        self.device_map = device_map if device_map is not None else "auto"
        self.model = None
        self.processor = None

    def load(self) -> LLaVAVideoBackend:
        from transformers import BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor

        quant_config = None
        if self.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        logger.info("Loading LLaVA-NeXT-Video %s", self.model_id)
        self.processor = LlavaNextVideoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        if not hasattr(self.processor, "patch_size") or self.processor.patch_size is None:
            self.processor.patch_size = 14
        if not hasattr(self.processor, "vision_feature_select_strategy"):
            self.processor.vision_feature_select_strategy = "default"
        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            quantization_config=quant_config,
            device_map=self.device_map,
        ).eval()
        return self

    def _resize_frames(self, frames: np.ndarray) -> list[Image.Image]:
        pil = frames_to_pil(frames)
        if self.max_side <= 0:
            return pil
        out = []
        for im in pil:
            w, h = im.size
            scale = self.max_side / max(w, h)
            if scale < 1.0:
                im = im.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
            out.append(im)
        return out

    def generate_greedy(
        self,
        frames: np.ndarray,
        user_text: str,
        max_new_tokens: int = 256,
    ) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("Call load() first")

        pil_frames = self._resize_frames(frames)
        prompt = (
            f"{user_text} "
            "Provide Severity, Impact, Start, End, Vehicles, Weather, and Explanation."
        )
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video"},
                ],
            }
        ]
        prompt_text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=prompt_text,
            videos=[pil_frames],
            padding=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        input_len = inputs["input_ids"].shape[-1]
        gen = output_ids[0, input_len:]
        return self.processor.decode(gen, skip_special_tokens=True).strip()


FOUNDATION_REGISTRY: dict[str, type[FoundationBackend]] = {
    "qwen": QwenVLBackend,
    "qwen2-vl-2b": Qwen2VL2BBackend,
    "qwen2.5-vl-3b": Qwen25VL3BBackend,
    "internvl": InternVLBackend,
    "llava-video": LLaVAVideoBackend,
}

FOUNDATION_SPECS: list[dict[str, Any]] = [
    {"name": QwenVLBackend.name, "backend": "qwen"},
    {"name": Qwen2VL2BBackend.name, "backend": "qwen2-vl-2b"},
    {"name": Qwen25VL3BBackend.name, "backend": "qwen2.5-vl-3b"},
    {"name": LLaVAVideoBackend.name, "backend": "llava-video"},
]


def build_backend(
    backend_key: str,
    load_in_4bit: bool = True,
    max_side: int = 224,
    device_map: str | dict | None = None,
) -> FoundationBackend:
    if backend_key not in FOUNDATION_REGISTRY:
        raise ValueError(f"Unknown backend: {backend_key}. Choose from {list(FOUNDATION_REGISTRY)}")
    cls = FOUNDATION_REGISTRY[backend_key]
    if device_map is None:
        device_map = {"": 0}
    return cls(load_in_4bit=load_in_4bit, max_side=max_side, device_map=device_map)


def spec_by_name(condition_name: str) -> dict[str, Any] | None:
    for spec in FOUNDATION_SPECS:
        if spec["name"] == condition_name:
            return spec
    return None
