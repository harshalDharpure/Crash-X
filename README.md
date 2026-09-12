# CrashX

Code and evaluation for dense dashcam accident explanation.

We fine-tune Qwen2.5-VL-7B with QLoRA (CrashLogic-7B) and score explanations for
omission vs hallucination, temporal grounding, and NLI faithfulness. Main finding:
adaptation cuts omissions a lot, but hallucination cost barely moves; the strong
tIoU numbers are mostly a dataset prior.

## Data

`Car_Crash_Text_Dataset_ground_truth.xlsx` is our annotation layer (CrashX-1500):
1,500 clips with severity, vehicles, impact, crash window, weather, camera view,
ambiguity notes, and multi-sentence explanations written by human annotators.

The raw videos come from the public Car Crash Dataset (CCD). Put them under
`video1500/` locally (not in this repo). Splits are already in
`crashx/data/splits/` (1198 / 150 / 150, seed 42).

```bash
python -m crashx.data.process_ccd \
  --excel Car_Crash_Text_Dataset_ground_truth.xlsx \
  --video-dir video1500 \
  --out-dir crashx/data/splits
```

## Layout

```text
crashx/                              training, inference, eval
ACCESS_latex_template_20240429/      IEEE Access paper (harshal.zip for Overleaf)
results/                             predictions and metric dumps
scripts/                             experiment shell scripts
outputs/crashlogic_7b_lora/          LoRA config (large weights gitignored)
Car_Crash_Text_Dataset_ground_truth.xlsx
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r crashx/requirements.txt
pip install -e .
```

Needs Python 3.10+ and a CUDA GPU for training/inference.

## Train

```bash
python -m crashx.models.train_qlora \
  --train-jsonl crashx/data/splits/train.jsonl \
  --val-jsonl crashx/data/splits/val.jsonl \
  --output-dir outputs/crashlogic_7b_lora
```

4-bit NF4, LoRA r=16, alpha=32, LR 2e-4, 5 epochs, 8 frames at most 224px.
Adapter weights are gitignored; train locally or see
`outputs/crashlogic_7b_lora/MODEL_WEIGHTS.md`.

## Inference and eval

```bash
python -m crashx.run_experiments \
  --lora-path outputs/crashlogic_7b_lora \
  --results-dir results

python -m crashx.run_journal_experiments --tables-only
```

Eval code is under `crashx/eval/` (BLEU/ROUGE/METEOR/CIDEr/BERTScore, tIoU,
ArgusCost H/O, NLI, bootstrap + Wilcoxon).

## Snapshot (n=150 test)

| System | BLEU-4 | BERTScore | C_O | C_H | tIoU |
|--------|--------|-----------|-----|-----|------|
| Zero-shot Qwen2.5-VL-7B | 0.016 | 0.486 | 0.462 | 0.227 | 0.012 |
| CrashLogic-7B (greedy) | 0.142 | 0.686 | 0.107 | 0.223 | 0.373 |
| CrashLogic + TCD a=0.5 | 0.142 | 0.686 | 0.102 | 0.212 | 0.394 |
| Constant window [3,4]s | — | — | — | — | 0.408 |

Hallucination cost after adaptation is not significant (p=0.81).

## Paper

IEEE Access sources are in `ACCESS_latex_template_20240429/`.
Upload `harshal.zip` to Overleaf; main file is `crashx_access.tex` (pdfLaTeX).
