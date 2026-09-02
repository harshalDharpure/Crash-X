# CrashX: Spatiotemporal Reasoning & Temporal Hallucination Mitigation for Traffic Accident Video-LLMs

Research pipeline for **"CrashX: Benchmarking Spatiotemporal Reasoning and Mitigating Temporal Hallucinations in Traffic Accident Video-LLMs"**.

This codebase adapts the [VRU-Accident](https://github.com/Kimyounggun99/VRU-Accident) evaluation pattern for dense crash captioning, fine-tunes **Qwen2.5-VL-7B-Instruct** with 4-bit QLoRA on the Car Crash Dataset (CCD), applies **SEASON**-style contrastive decoding at inference, and scores outputs with lexical metrics + CrashX-adapted **ARGUS** dual costs.

## Directory layout

```text
crashx/
├── data/
│   ├── process_ccd.py      # Excel → stratified JSONL (1200/150/150)
│   └── dataset.py          # 16-keyframe video sampler
├── models/
│   ├── vru_baseline.py     # Qwen2.5-VL wrapper (VRU-style)
│   └── train_qlora.py      # 4-bit QLoRA SFT → CrashLogic-7B
├── inference/
│   └── season_decoder.py   # Greedy + hybrid SEASON decoding
├── eval/
│   ├── metrics.py          # BLEU-4, ROUGE-L, METEOR, CIDEr, BERTScore, tIoU
│   └── argus_eval.py       # Structured ArgusCost-H / ArgusCost-O
├── run_experiments.py      # Base → FT → FT+SEASON comparison
├── requirements.txt
└── README.md
```

## Data expectations

| Asset | Path |
|-------|------|
| Ground truth Excel | `../Car_Crash_Text_Dataset_ground_truth.xlsx` |
| Videos (1500) | `../video1500/000001.mp4` … `0001500.mp4` |
| Processed splits | `crashx/data/splits/{train,val,test}.jsonl` |
| LoRA checkpoint | `outputs/crashlogic_7b_lora/` |

## Quick start

```bash
# From repo root (vlm_new_gen/)
pip install -r crashx/requirements.txt
pip install -e .   # makes `python -m crashx.*` importable

# 1) Build stratified JSONL splits
python -m crashx.data.process_ccd \
  --excel Car_Crash_Text_Dataset_ground_truth.xlsx \
  --video-dir video1500 \
  --out-dir crashx/data/splits

# 2) Fine-tune CrashLogic-7B (QLoRA, 5 epochs)
python -m crashx.models.train_qlora \
  --train-jsonl crashx/data/splits/train.jsonl \
  --val-jsonl crashx/data/splits/val.jsonl \
  --output-dir outputs/crashlogic_7b_lora

# 3) Run 3-way experiment table on 150 test videos
python -m crashx.run_experiments \
  --test-jsonl crashx/data/splits/test.jsonl \
  --lora-path outputs/crashlogic_7b_lora \
  --alpha 1.0

# Optional: full SEASON paper ablation (homogenization + spatial + JSD weights)
python -m crashx.run_experiments \
  --test-jsonl crashx/data/splits/test.jsonl \
  --lora-path outputs/crashlogic_7b_lora \
  --full-season-ablation
```

## SEASON hybrid decoding

| Flag | Behavior |
|------|----------|
| `use_full_season=False` (default / paper PoC) | Temporal negative via reverse or shuffle of keyframes: \(\mathrm{logits}_{final}=(1+\alpha)\cdot\mathrm{logits}(V)-\alpha\cdot\mathrm{logits}(V_{neg})\) |
| `use_full_season=True` (ablation) | Temporal homogenization + spatial Gaussian noise + self-diagnostic \(w_S,w_T\) (SEASON paper) |

- \(\alpha=0\) → standard greedy baseline  
- \(\alpha=1.0\) → SEASON contrastive strength used in primary experiments  

## Metrics

- **Lexical:** BLEU-4, ROUGE-L, METEOR, CIDEr, BERTScore (`bert-base-uncased`)
- **Temporal:** tIoU between predicted and GT crash windows
- **CrashX ARGUS (structured):** ArgusCost-H (hallucinated claims), ArgusCost-O (omitted GT fields); lower is better
- **Explanation proxy:** BERTScore F1 between predicted and GT Explanation strings

## Citations

```bibtex
@InProceedings{Kim_2025_ICCV,
  author    = {Kim, Younggun and Abdelrahman, Ahmed S. and Abdel-Aty, Mohamed},
  title     = {VRU-Accident: A Vision-Language Benchmark for Video Question Answering
               and Dense Captioning for Accident Scene Understanding},
  booktitle = {ICCV Workshops},
  year      = {2025}
}

@InProceedings{Wu_2026_CVPR,
  author    = {Wu, Chang-Hsun and Chang, Kai-Po and Sheng, Yu-Yang and others},
  title     = {SEASON: Mitigating Temporal Hallucination in Video Large Language Models
               via Self-Diagnostic Contrastive Decoding},
  booktitle = {CVPR},
  year      = {2026}
}

@InProceedings{ARGUS_2025_ICCV,
  title     = {ARGUS: Hallucination and Omission Evaluation in Video-LLMs},
  booktitle = {ICCV},
  year      = {2025}
}
```

## Notes

- Training / SEASON inference expect CUDA with enough VRAM for 4-bit 7B Qwen2.5-VL.
- CPU dry-runs can exercise data + metric modules without loading the VL model.
