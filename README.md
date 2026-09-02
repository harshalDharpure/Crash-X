# CrashX

**CrashX: Benchmarking Spatiotemporal Reasoning and Mitigating Temporal Hallucinations in Traffic Accident Video-LLMs**

Research pipeline for dense crash video captioning on the Car Crash Dataset (CCD), with QLoRA fine-tuning of **Qwen2.5-VL-7B-Instruct** (CrashLogic-7B) and **SEASON** contrastive decoding at inference.

GitHub: [harshalDharpure/Crash-X](https://github.com/harshalDharpure/Crash-X)

---

## Repository Contents

| Path | Description |
|------|-------------|
| `crashx/` | Full Python package (data, training, SEASON, evaluation) |
| `scripts/` | Pipeline runners (`run_full_pipeline.sh`, `run_journal_ablations.sh`) |
| `results/` | All experiment outputs — predictions, metrics, tables, paper docs |
| `results/paper/` | **13 markdown files** — detailed tables, methodology, key findings |
| `results/tables/` | LaTeX + Markdown tables (Tables I–VII) |
| `crashx/data/splits/` | Stratified train/val/test JSONL (1198/150/150) |
| `Car_Crash_Text_Dataset_ground_truth.xlsx` | Excel ground truth (1,500 rows) |
| `outputs/crashlogic_7b_lora/` | LoRA config (weights excluded — see below) |

---

## Quick Start

```bash
pip install -r crashx/requirements.txt
pip install -e .

# Build splits (if needed)
python -m crashx.data.process_ccd \
  --excel Car_Crash_Text_Dataset_ground_truth.xlsx \
  --video-dir video1500 \
  --out-dir crashx/data/splits

# Train CrashLogic-7B
python -m crashx.models.train_qlora \
  --train-jsonl crashx/data/splits/train.jsonl \
  --val-jsonl crashx/data/splits/val.jsonl \
  --output-dir outputs/crashlogic_7b_lora

# Run 3-way experiments
python -m crashx.run_experiments \
  --lora-path outputs/crashlogic_7b_lora \
  --results-dir results

# Generate all paper tables
python -m crashx.run_journal_experiments --tables-only
```

---

## Results Summary (150 test videos)

| Method | BLEU-4 | tIoU | ArgusCost-O | NLI-Score |
|--------|--------|------|-------------|-----------|
| Zero-shot Qwen2.5-VL-7B | 0.016 | 0.012 | 0.462 | 0.047 |
| CrashLogic-7B (Greedy) | **0.142** | 0.373 | **0.107** | -0.005 |
| CrashLogic-7B + SEASON | 0.140 | 0.375 | 0.112 | **0.090** |

Full documentation: [`results/paper/00_INDEX.md`](results/paper/00_INDEX.md)

---

## Data & Model Weights

- **Videos:** Not included (825MB). Place CCD videos in `video1500/000001.mp4` … `0001500.mp4`.
- **LoRA weights:** `adapter_model.safetensors` (~182MB) exceeds GitHub file limit. Train locally (see above) or see `outputs/crashlogic_7b_lora/MODEL_WEIGHTS.md`.

---

## Citations

See [`crashx/README.md`](crashx/README.md) for VRU-Accident, SEASON, and ARGUS references.
