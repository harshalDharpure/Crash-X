# CrashX Human Evaluation Pack

- Sample: `sample_50.jsonl` (50 videos, stratified)
- Master: `annotation_master.csv` (200 items = videos × models)
- Blind map: `blind_code_map.json` (**do not show to annotators**)
- Annotator sheets: `annotator_A1.csv` … `annotator_A3.csv`

## Instructions
1. Open the video at `video_path`.
2. Read `model_output` only (ignore blind_code meaning).
3. Score Likert 1–5 for temporal / faithfulness / explanation.
4. Save filled CSVs and run:

```bash
PYTHONPATH=. python -m crashx.eval.build_human_eval_pack \
  --score-kappa results/human_eval/annotator_A1_filled.csv \
               results/human_eval/annotator_A2_filled.csv \
               results/human_eval/annotator_A3_filled.csv
```

Target: Fleiss' κ ≥ 0.65.
