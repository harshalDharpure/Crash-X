#!/usr/bin/env python3
"""SEASON hybrid contrastive decoding for CrashX.

Default (use_full_season=False): paper PoC with temporal negatives via reverse/shuffle:
    logits_final = (1 + alpha) * logits(V) - alpha * logits(V_neg)

Ablation (use_full_season=True): SEASON-paper path with temporal homogenization,
spatial Gaussian noise, and self-diagnostic JSD weights (w_S, w_T):
    logits = (1+alpha)*logits(V) - alpha*[w_S*logits(V_S) + w_T*logits(V_T)]

alpha=0 reduces to standard greedy decoding.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from crashx.models.vru_baseline import VRUBaselineModel, frames_to_pil

logger = logging.getLogger(__name__)


def _to_frame_array(frames: np.ndarray | torch.Tensor | Sequence[Any]) -> np.ndarray:
    """Normalize input frames to uint8 [T,H,W,C]."""
    if isinstance(frames, torch.Tensor):
        arr = frames.detach().cpu()
        if arr.ndim == 4 and arr.shape[1] in (1, 3):
            arr = arr.permute(0, 2, 3, 1)
        if arr.dtype.is_floating_point:
            arr = (arr.clamp(0, 1) * 255).byte()
        return arr.numpy()
    if isinstance(frames, np.ndarray):
        arr = frames
        if arr.ndim == 4 and arr.shape[1] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (0, 2, 3, 1))
        if np.issubdtype(arr.dtype, np.floating):
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        return arr
    return np.stack([np.asarray(f) for f in frames], axis=0)


def build_temporal_negative(
    frames: np.ndarray,
    mode: str = "reverse",
    seed: int = 0,
) -> np.ndarray:
    """Construct V_neg by reversing or shuffling keyframe order."""
    if mode == "reverse":
        return frames[::-1].copy()
    if mode == "shuffle":
        rng = np.random.RandomState(seed)
        idx = rng.permutation(frames.shape[0])
        return frames[idx].copy()
    raise ValueError(f"Unknown neg_mode: {mode}. Use 'reverse' or 'shuffle'.")


def build_spatial_negative(
    frames: np.ndarray,
    noise_std: float = 0.35,
    seed: int = 0,
) -> np.ndarray:
    """Spatial negative: additive Gaussian noise on RGB frames (SEASON-style)."""
    rng = np.random.RandomState(seed)
    x = frames.astype(np.float32) / 255.0
    noise = rng.randn(*x.shape).astype(np.float32) * noise_std
    x = np.clip(x + noise, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def temporal_homogenize_frames(
    frames: np.ndarray,
    beta: float = 0.5,
) -> np.ndarray:
    """Approximate temporal homogenization in pixel space (ablation-friendly).

    The SEASON paper homogenizes layer-wise vision features. When we cannot hook
    the vision encoder internals cleanly, we blend each frame with the global
    mean frame — preserving spatial content while destroying temporal order cues:

        h_t = (1 - beta) * f_t + beta * mean_t(f)
    """
    beta = float(np.clip(beta, 0.0, 1.0))
    mean_frame = frames.astype(np.float32).mean(axis=0, keepdims=True)
    blended = (1.0 - beta) * frames.astype(np.float32) + beta * mean_frame
    return np.clip(blended, 0, 255).astype(np.uint8)


def jensen_shannon_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """JSD between two discrete distributions (last dim)."""
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    m = 0.5 * (p + q)
    js = 0.5 * (
        F.kl_div(m.log(), p, reduction="none").sum(dim=-1)
        + F.kl_div(m.log(), q, reduction="none").sum(dim=-1)
    )
    return js


class SeasonDecoder:
    """Greedy + SEASON contrastive decoding over a loaded VRUBaselineModel."""

    def __init__(self, backbone: VRUBaselineModel) -> None:
        if backbone.model is None or backbone.processor is None:
            raise RuntimeError("VRUBaselineModel must be load()'ed before SeasonDecoder")
        self.backbone = backbone

    @staticmethod
    def _append_generated_tokens(
        inputs: dict[str, Any],
        token_ids: list[int],
    ) -> dict[str, Any]:
        """Extend multimodal inputs with newly generated text token ids."""
        if not token_ids:
            return inputs
        device = inputs["input_ids"].device
        dtype = inputs["input_ids"].dtype
        new_ids = torch.tensor([token_ids], device=device, dtype=dtype)
        out = dict(inputs)
        out["input_ids"] = torch.cat([inputs["input_ids"], new_ids], dim=-1)
        out["attention_mask"] = torch.ones_like(out["input_ids"])
        if "mm_token_type_ids" in inputs:
            # New decode tokens are plain text (type 0 for Qwen2.5-VL).
            mm_ext = torch.zeros_like(new_ids, dtype=inputs["mm_token_type_ids"].dtype)
            out["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"], mm_ext], dim=-1)
        return out

    def _forward_logits(
        self,
        base_inputs: dict[str, Any],
        generated: list[int],
        *,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run one forward pass and return last-step logits (+ optional attentions)."""
        inputs = self._append_generated_tokens(base_inputs, generated)
        outputs = self.backbone.model(
            **inputs,
            output_attentions=output_attentions,
            use_cache=False,
        )
        attn = outputs.attentions if output_attentions else None
        return outputs.logits[:, -1, :], attn

    def decode_greedy(
        self,
        frames: np.ndarray | torch.Tensor | Sequence[Any],
        user_text: str,
        max_new_tokens: int = 512,
    ) -> str:
        """Standard greedy decoding (equivalent to alpha=0)."""
        return self.backbone.generate_greedy(frames, user_text, max_new_tokens=max_new_tokens)

    @torch.inference_mode()
    def decode_season(
        self,
        frames: np.ndarray | torch.Tensor | Sequence[Any],
        user_text: str,
        alpha: float = 1.0,
        max_new_tokens: int = 512,
        neg_mode: str = "reverse",
        use_full_season: bool = False,
        beta: float = 0.5,
        spatial_noise_std: float = 0.35,
        seed: int = 0,
    ) -> str:
        """Autoregressive SEASON contrastive decoding.

        Parameters
        ----------
        alpha:
            Contrastive strength. alpha=0 → greedy baseline; alpha=1.0 → primary SEASON.
        use_full_season:
            False → V vs V_neg (reverse/shuffle). True → spatial+temporal + JSD weights.
        """
        if abs(alpha) < 1e-8:
            return self.decode_greedy(frames, user_text, max_new_tokens=max_new_tokens)

        frames_np = _to_frame_array(frames)
        tokenizer = self.backbone.processor.tokenizer
        eos_id = tokenizer.eos_token_id

        if use_full_season:
            v_t = temporal_homogenize_frames(frames_np, beta=beta)
            v_s = build_spatial_negative(frames_np, noise_std=spatial_noise_std, seed=seed)
            views = {"O": frames_np, "T": v_t, "S": v_s}
        else:
            v_neg = build_temporal_negative(frames_np, mode=neg_mode, seed=seed)
            views = {"O": frames_np, "N": v_neg}

        # Cache prefilled multimodal inputs per view (video tokens fixed; text grows).
        base_inputs = {
            name: self.backbone.prepare_inputs(vis, user_text) for name, vis in views.items()
        }

        generated: list[int] = []
        for _step in range(max_new_tokens):
            logits_by_view: dict[str, torch.Tensor] = {}
            attn_frame_by_view: dict[str, torch.Tensor | None] = {}

            for name, prefilled in base_inputs.items():
                want_attn = use_full_season
                logits, attentions = self._forward_logits(
                    prefilled,
                    generated,
                    output_attentions=want_attn,
                )
                logits_by_view[name] = logits

                if want_attn:
                    attn_frame_by_view[name] = self._estimate_frame_attention(
                        attentions,
                        num_frames=frames_np.shape[0],
                        device=logits.device,
                    )
                else:
                    attn_frame_by_view[name] = None

            if use_full_season:
                w_s, w_t = self._diagnostic_weights(
                    attn_frame_by_view.get("O"),
                    attn_frame_by_view.get("S"),
                    attn_frame_by_view.get("T"),
                )
                logits = (1.0 + alpha) * logits_by_view["O"] - alpha * (
                    w_s * logits_by_view["S"] + w_t * logits_by_view["T"]
                )
            else:
                logits = (1.0 + alpha) * logits_by_view["O"] - alpha * logits_by_view["N"]

            next_id = int(torch.argmax(logits, dim=-1).item())
            if eos_id is not None and next_id == eos_id:
                break
            generated.append(next_id)

        text = tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip()

    @staticmethod
    def _estimate_frame_attention(
        attentions: tuple[torch.Tensor, ...] | None,
        num_frames: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Heuristic frame-level attention from last-layer cross/self attentions.

        Aggregates attention from the last query token onto a uniform partition of
        the key sequence into ``num_frames`` bins (proxy for per-frame visual mass).
        """
        dev = device or torch.device("cpu")
        uniform = torch.ones(num_frames, device=dev) / num_frames
        if not attentions:
            return uniform
        # attentions[-1]: [B, H, Q, K]
        attn = attentions[-1]
        if attn is None:
            return uniform
        # Mean over heads, take last query position
        a = attn.mean(dim=1)[0, -1]  # [K]
        k = a.numel()
        if k == 0:
            return uniform
        # Soft partition keys into num_frames bins
        edges = torch.linspace(0, k, steps=num_frames + 1, device=a.device)
        masses = []
        for i in range(num_frames):
            lo = int(edges[i].item())
            hi = int(edges[i + 1].item())
            masses.append(a[lo:hi].sum() if hi > lo else a.new_zeros(()))
        masses_t = torch.stack(masses)
        return masses_t / masses_t.sum().clamp_min(1e-8)

    @staticmethod
    def _diagnostic_weights(
        a_o: torch.Tensor | None,
        a_s: torch.Tensor | None,
        a_t: torch.Tensor | None,
    ) -> tuple[float, float]:
        """Compute (w_S, w_T) from JSD of frame-attention distributions."""
        if a_o is None or a_s is None or a_t is None:
            return 0.5, 0.5
        d_s = float(jensen_shannon_divergence(a_o.unsqueeze(0), a_s.unsqueeze(0)).item())
        d_t = float(jensen_shannon_divergence(a_o.unsqueeze(0), a_t.unsqueeze(0)).item())
        denom = d_s + d_t
        if denom < 1e-8:
            return 0.5, 0.5
        return d_s / denom, d_t / denom
