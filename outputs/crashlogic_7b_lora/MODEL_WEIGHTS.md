# CrashLogic-7B LoRA Weights

The fine-tuned adapter `adapter_model.safetensors` (~182 MB) is **not included** in this GitHub repo because it exceeds GitHub's 100 MB per-file limit.

## Reproduce locally

```bash
python -m crashx.models.train_qlora \
  --train-jsonl crashx/data/splits/train.jsonl \
  --val-jsonl crashx/data/splits/val.jsonl \
  --output-dir outputs/crashlogic_7b_lora \
  --epochs 5 --num-frames 8 --max-side 224
```

Training config is saved in `train_config.json`. Final adapter: `adapter_model.safetensors`.

## Included in repo

- `adapter_config.json` — LoRA architecture
- `train_config.json` — hyperparameters
- `tokenizer_config.json`, `processor_config.json`

All experiment **predictions and metrics** in `results/` were generated with this checkpoint and are fully reproducible from the training recipe above.
