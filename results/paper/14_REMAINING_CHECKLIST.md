# 14 — Remaining A* Checklist Status

Updated: 2026-09-02

| Item | Status | Notes |
|------|--------|-------|
| Foundation baselines (4 open VLMs) | **Done** | Qwen2.5-7B/3B, Qwen2-2B, LLaVA-Video |
| Bootstrap CIs + Wilcoxon | **Done** | Tables VIII–IX |
| TCD α=0.5 primary | **Done** | Table 0 |
| InternVL | Dropped | transformers≥4.50 incompatible |
| GPT-4o / Gemini | **Blocked** | No API keys in environment |
| 16-frame retrain (3 ep, seed 42) | **Running** | `outputs/crashlogic_7b_lora_f16` |
| Multi-seed (3×) full retrain | Deferred | Too slow; report 1 seed + CI first |
| Human-eval pack | **Ready** | `results/human_eval/` (50 vids × 4 models) |
| Human annotators fill κ | **Needs people** | Cannot automate |

## Run commercial baselines when keys exist

```bash
export OPENAI_API_KEY=...
PYTHONPATH=. .venv/bin/python -m crashx.run_api_baselines --provider openai --limit 150

export GEMINI_API_KEY=...
PYTHONPATH=. .venv/bin/python -m crashx.run_api_baselines --provider gemini --limit 150

PYTHONPATH=. .venv/bin/python -m crashx.run_journal_experiments --tables-only
```

## Monitor 16-frame retrain

```bash
tail -f results/logs/phase3_f16_retrain.log
```

## Human annotation

1. Give each annotator `results/human_eval/annotator_A{1,2,3}.csv`
2. Do **not** share `blind_code_map.json`
3. After fill:

```bash
PYTHONPATH=. python -m crashx.eval.build_human_eval_pack \
  --score-kappa results/human_eval/annotator_A1_filled.csv \
               results/human_eval/annotator_A2_filled.csv \
               results/human_eval/annotator_A3_filled.csv
```
