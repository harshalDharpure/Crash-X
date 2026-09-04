# Repository maintenance (research project)

How this GitHub repo is organised so it stays usable as a **research codebase + paper companion**.

## Versioning & tags

| Tag | Meaning |
|-----|---------|
| `v0.1.0` | Initial public pipeline (code + early results) |
| `v0.2.0-ieee-draft` | IEEE Overleaf package + omission/hallucination paper rewrite + guide briefing |

Create annotated tags for paper milestones:

```bash
git tag -a v0.2.0-ieee-draft -m "IEEE draft: omission/hallucination framing, Overleaf package, guide briefing"
git push origin v0.2.0-ieee-draft
```

**Do not** retag published versions; cut a new tag (e.g. `v0.2.1-ieee-draft`) after camera-ready fixes.

## Releases (GitHub UI)

For each paper milestone, create a Release from the matching tag and attach:

- `paper/CrashX_IEEE_Overleaf.zip`
- Short changelog (findings + what changed in code/eval)

## What belongs in git

| Include | Exclude |
|---------|---------|
| `crashx/` code | `video1500/*.mp4` |
| Splits JSONL | LoRA `adapter_model.safetensors` |
| Paper LaTeX + zip | `.venv/`, caches, logs |
| Result metrics / prediction JSON (if < GitHub limits) | Secrets / API keys |
| Docs: README, GUIDE briefing, CITATION | Huge raw dumps |

## Canonical documentation map

| Audience | File |
|----------|------|
| First-time visitors | `README.md` |
| Guide / viva / presentation | `GUIDE_PRESENTATION_BRIEFING.md` |
| Academic citation | `CITATION.cff` |
| Overleaf upload | `paper/CrashX_IEEE_Overleaf.zip` |
| Detailed paper notes | `results/paper/00_INDEX.md` |
| Package internals | `crashx/README.md` |

When claims change (e.g. TCD significance), update **README snapshot**, **GUIDE briefing**, and **paper tables** together so they do not contradict.

## GitHub metadata checklist

After each major push:

```bash
gh repo edit harshalDharpure/Crash-X \
  --description "CrashX: evaluating Video-LLM faithfulness for traffic accident explanation (omission vs hallucination, temporal priors, TCD)." \
  --homepage "https://github.com/harshalDharpure/Crash-X" \
  --add-topic video-llm \
  --add-topic traffic-accident \
  --add-topic hallucination \
  --add-topic qlora \
  --add-topic multimodal \
  --add-topic faithfulness \
  --add-topic research
```

## Branch policy

- `main` — stable research snapshot (code that matches the current draft claims)
- Optional: `paper/ieee-draft`, `exp/*` for unfinished experiments

Never force-push `main` after a tagged release.

## Reproducibility statement

Numbers in the IEEE draft regenerate from released per-video outputs under `results/` via `crashx/eval/` scripts. Training requires local CCD videos and a GPU; inference weights may need local training if adapters are not hosted separately.
