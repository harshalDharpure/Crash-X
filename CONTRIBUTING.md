# Contributing to CrashX

Thank you for interest in this research repository.

## Scope

This repo holds **code**, **evaluation scripts**, **result tables**, and an **IEEE paper draft** for CrashX. Large video files and LoRA weight binaries are intentionally excluded.

## Before opening a PR

1. Do not commit secrets, API keys, or `.env` files.
2. Do not commit `video1500/*.mp4` or `adapter_model.safetensors` (see `.gitignore`).
3. Prefer small, focused PRs (eval fix, doc fix, table regeneration).
4. If you change metrics, regenerate affected tables and note the commit/hash in the PR.

## Coding conventions

- Python ≥ 3.10; package under `crashx/`.
- Keep evaluation deterministic where possible (fixed seeds, greedy decode for main tables).
- Document any new metric definition next to the implementation and in `GUIDE_PRESENTATION_BRIEFING.md` if it is paper-facing.

## Issues

Please include:

- What you ran (command + hardware)
- Expected vs actual behaviour
- Whether the issue is about **training**, **inference**, **metrics**, or **paper LaTeX**

## Citation

If your contribution becomes part of a paper or reuse of the protocol, cite the repository via `CITATION.cff`.
