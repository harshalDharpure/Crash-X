# 05 — Table II: Temporal & Forensic Reasoning

**Purpose:** Spatiotemporal localization and forensic fidelity — core CrashX evaluation axes.

---

## Results

| Method | tIoU ↑ | THR@0.25 ↑ | THR@0.50 ↑ | ArgusCost-H ↓ | ArgusCost-O ↓ |
|--------|--------|------------|------------|---------------|---------------|
| Zero-shot Qwen2.5-VL-7B | 0.012 | 0.020 | 0.013 | 0.227 | 0.462 |
| CrashLogic-7B (Greedy) | 0.373 | 0.573 | **0.440** | **0.223** | **0.107** |
| CrashLogic-7B + SEASON | **0.375** | 0.573 | 0.427 | 0.227 | 0.112 |

---

## Best per Metric

| Metric | Best Method | Value | vs Zero-shot |
|--------|-------------|-------|--------------|
| tIoU | SEASON | **0.375** | +30× |
| THR@0.25 | Greedy / SEASON (tie) | **0.573** | +28× |
| THR@0.50 | Greedy | **0.440** | +34× |
| ArgusCost-H | Greedy | **0.223** | −2% (similar) |
| ArgusCost-O | Greedy | **0.107** | **−77%** |

---

## Analysis

### Temporal Reasoning
- Zero-shot **fails** at temporal localization (tIoU ≈ 0.01) — cannot parse crash windows from free-form text.
- Fine-tuning enables **structured timestamp output** → mean tIoU 0.37, with 57% of videos achieving tIoU ≥ 0.25.
- SEASON gives marginal tIoU gain (+0.002) over greedy.

### Forensic Fidelity (ARGUS)
- **ArgusCost-O (omission)** is the strongest differentiator: fine-tuning reduces omission from 46% to 11%.
- Zero-shot omits severity, vehicles, timestamps in ~46% of critical fields.
- **ArgusCost-H (hallucination)** similar across all three (~0.22–0.23) — fine-tuning teaches format but doesn't eliminate conflicting claims.

---

## LaTeX Caption

> *Table II: Spatiotemporal and forensic metrics. tIoU = mean temporal IoU; THR@τ = fraction with tIoU ≥ τ. ArgusCost-H/O = structured hallucination/omission rates (↓ better).*
