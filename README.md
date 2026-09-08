# CrashX

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/paper-IEEE%20draft-informational)](paper/CrashX_IEEE_Overleaf.zip)
[![Guide](https://img.shields.io/badge/docs-guide%20briefing-success)](GUIDE_PRESENTATION_BRIEFING.md)

**CrashX: Fine-Tuning and Evaluating Video Language Models for Traffic Accident Explanation**

Research code, evaluation protocol, and IEEE draft for dense **dashcam accident explanation** on **CrashX-1500**, our own human-annotated crash explanation dataset.

> **About the data:** the raw video (1,500 five-second dashcam clips) comes from the public Car Crash Dataset (CCD), which ships anticipation labels only. **Every forensic annotation we train and evaluate on — severity, vehicles by colour and type, impact geometry, crash window, weather, camera view, ambiguity note, and the multi-sentence explanation — was written from scratch by our human annotators** and is released here as [`Car_Crash_Text_Dataset_ground_truth.xlsx`](Car_Crash_Text_Dataset_ground_truth.xlsx) (1,500 rows × 11 columns, 143,968 words of explanation text).

> **Core finding:** QLoRA adaptation of Qwen2.5-VL-7B (**CrashLogic-7B**) sharply reduces *omissions* and improves lexical metrics, but does **not** reduce *hallucinations*. Apparent timestamp accuracy is largely a **dataset prior**. Mild Temporal Contrastive Decoding (TCD / SEASON-style) is not significant under paired tests; strong contrast can hurt.

| Resource | Link |
|----------|------|
| GitHub | https://github.com/harshalDharpure/Crash-X |
| Guide presentation briefing (full walkthrough) | [`GUIDE_PRESENTATION_BRIEFING.md`](GUIDE_PRESENTATION_BRIEFING.md) |
| **IEEE Access Overleaf zip (current target)** | [`ACCESS_latex_template_20240429/harshal.zip`](ACCESS_latex_template_20240429/harshal.zip) |
| IEEE Access LaTeX sources | [`ACCESS_latex_template_20240429/`](ACCESS_latex_template_20240429/) |
| IEEE conference Overleaf zip | [`paper/CrashX_IEEE_Overleaf.zip`](paper/CrashX_IEEE_Overleaf.zip) |
| IEEE conference LaTeX sources | [`paper/ieee_crashx/`](paper/ieee_crashx/) |
| Paper notes / older tables | [`results/paper/00_INDEX.md`](results/paper/00_INDEX.md) |
| Cite this repo | [`CITATION.cff`](CITATION.cff) |

---

## Highlights

0. **CrashX-1500**: 1,500 clips annotated in-house with a nine-field forensic schema plus a ~96-word causal explanation each.
1. **Omission vs hallucination** scored separately (`C_O`, `C_H`) on those structured fields.
2. **CrashLogic-7B**: QLoRA fine-tune of Qwen2.5-VL-7B on 1,198 training clips.
3. **Temporal prior diagnosis**: constant-window baselines beat the model on tIoU; predicted start ≈ uncorrelated with GT.
4. **TCD / SEASON ablation** with paired Wilcoxon + bootstrap CIs on 150 test videos.
5. Release of splits, predictions, metrics scripts, and Overleaf-ready paper package.

### Snapshot (150-video test set)

| System | BLEU-4 ↑ | BERTScore ↑ | C_O ↓ | C_H ↓ | tIoU ↑ |
|--------|----------|-------------|-------|-------|--------|
| Zero-shot Qwen2.5-VL-7B | 0.016 | 0.486 | 0.462 | 0.227 | 0.012 |
| **CrashLogic-7B (greedy)** | **0.142** | **0.686** | **0.107** | 0.223 | 0.373 |
| CrashLogic + TCD α=0.5 | 0.142 | 0.686 | 0.102 | 0.212 | 0.394 |
| Constant window `[3,4]`s | — | — | — | — | **0.408** |

Hallucination cost is **not** significantly improved by adaptation (\(p=0.81\)). See the briefing for full tables.

---

## Repository layout

```text
Crash-X/
├── GUIDE_PRESENTATION_BRIEFING.md   # Complete guide talk notes
├── CITATION.cff                     # Citation metadata
├── LICENSE                          # MIT
├── CONTRIBUTING.md
├── REPOSITORY.md                    # Tagging, releases, maintenance
├── crashx/                          # Python package
│   ├── data/                        # CCD processing + splits
│   ├── models/                      # QLoRA training
│   ├── inference/                   # Greedy + TCD / SEASON
│   ├── eval/                        # Lexical, NLI, ArgusCost, stats
│   └── run_*.py                     # Experiment entry points
├── ACCESS_latex_template_20240429/  # IEEE Access version (current submission target)
│   ├── crashx_access.tex            # Main file
│   ├── harshal.zip                  # Upload to Overleaf
│   └── sections/ tables/ figures/
├── paper/
│   ├── CrashX_IEEE_Overleaf.zip     # IEEE conference version
│   └── ieee_crashx/                 # LaTeX sources
├── results/                         # Predictions, metrics, paper notes
├── scripts/                         # Pipeline shell scripts
├── outputs/crashlogic_7b_lora/      # LoRA config (large weights excluded)
├── Car_Crash_Text_Dataset_ground_truth.xlsx
└── video1500/                       # Local only (gitignored)
```

---

## Installation

```bash
git clone https://github.com/harshalDharpure/Crash-X.git
cd Crash-X
python -m venv .venv && source .venv/bin/activate
pip install -r crashx/requirements.txt
pip install -e .
```

**Requirements:** Python ≥ 3.10, CUDA GPU recommended for training/inference (A100-40GB used in the paper).

---

## Data

| Asset | Location | Notes |
|-------|----------|-------|
| Ground-truth Excel (**our annotations**) | `Car_Crash_Text_Dataset_ground_truth.xlsx` | In repo, 1,500 × 11 |
| Videos | `video1500/000001.mp4` … `0001500.mp4` | **Not** in GitHub (~825MB). Obtain CCD videos separately |
| Splits | `crashx/data/splits/{train,val,test}.jsonl` | 1198 / 150 / 150, seed 42 |

```bash
python -m crashx.data.process_ccd \
  --excel Car_Crash_Text_Dataset_ground_truth.xlsx \
  --video-dir video1500 \
  --out-dir crashx/data/splits
```

---

## Training (CrashLogic-7B)

```bash
python -m crashx.models.train_qlora \
  --train-jsonl crashx/data/splits/train.jsonl \
  --val-jsonl crashx/data/splits/val.jsonl \
  --output-dir outputs/crashlogic_7b_lora
```

Key settings: 4-bit NF4, LoRA \(r{=}16\), \(\alpha{=}32\), LR \(2{\times}10^{-4}\), 5 epochs, 8 frames @ ≤224px.  
Large `adapter_model.safetensors` is gitignored — train locally or see `outputs/crashlogic_7b_lora/MODEL_WEIGHTS.md`.

---

## Inference & evaluation

```bash
# Main systems (greedy / TCD / SEASON)
python -m crashx.run_experiments \
  --lora-path outputs/crashlogic_7b_lora \
  --results-dir results

# Extra baselines / table regeneration
python -m crashx.run_journal_experiments --tables-only
```

Metrics implemented under `crashx/eval/`: BLEU/ROUGE/METEOR/CIDEr/BERTScore, tIoU, ArgusCost-H/O, NLI, paired bootstrap + Wilcoxon.

---

## Paper

**IEEE Access (current target)**

1. Download [`ACCESS_latex_template_20240429/harshal.zip`](ACCESS_latex_template_20240429/harshal.zip)
2. Overleaf → New Project → Upload Project
3. Main file: `crashx_access.tex`, compiler: **pdfLaTeX**, recompile twice (BibTeX in between)

**IEEE conference version**

1. Download [`paper/CrashX_IEEE_Overleaf.zip`](paper/CrashX_IEEE_Overleaf.zip)
2. Main file: `main.tex`, compiler: **pdfLaTeX**, recompile twice

Sources live in [`paper/ieee_crashx/`](paper/ieee_crashx/).  
Presentation / viva briefing: [`GUIDE_PRESENTATION_BRIEFING.md`](GUIDE_PRESENTATION_BRIEFING.md).

---

## Citation

If you use this code, splits, or evaluation protocol, please cite:

```bibtex
@misc{crashx2026,
  title        = {CrashX: Fine-Tuning and Evaluating Video Language Models for Traffic Accident Explanation},
  author       = {Dharpure, Harshal},
  year         = {2026},
  howpublished = {\url{https://github.com/harshalDharpure/Crash-X}},
  note         = {Code, evaluation protocol, and IEEE draft}
}
```

Or use GitHub’s “Cite this repository” (from [`CITATION.cff`](CITATION.cff)).

### Related work we build on
CCD; Qwen2.5-VL; ARGUS (omission/hallucination framing); SEASON / VCD (contrastive decoding); TimeChat / VTimeLLM / LITA (time-aware VLMs). Full bibliography: `paper/ieee_crashx/refs.bib`.

---

## License

Code is released under the [MIT License](LICENSE).  
The **CrashX-1500 annotations** in `Car_Crash_Text_Dataset_ground_truth.xlsx` are our own work and are released with this repository.  
The underlying **CCD videos** are not ours and are not redistributed here; obtain them from the original CCD release under its own license.

---

## Maintainers & contact

- Repository: [harshalDharpure/Crash-X](https://github.com/harshalDharpure/Crash-X)
- See [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`REPOSITORY.md`](REPOSITORY.md) for tags, releases, and how to keep this research repo tidy.
