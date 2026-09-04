# CrashX IEEE Paper — Overleaf Upload Guide

## What this folder is

Submission-ready **IEEE conference** (`IEEEtran`) paper package for:

> **CrashX: Domain Adaptation Removes Omissions but Not Hallucinations in Video-LLM Explanations of Traffic Accidents**

## Upload to Overleaf

1. Overleaf → **New Project** → **Upload Project** → select `CrashX_IEEE_Overleaf.zip`.
2. Set the main document to `main.tex` (Menu → Main document).
3. Compiler: **pdfLaTeX** (default). TeX Live 2022 or newer.
4. Click **Recompile** twice (Overleaf runs BibTeX automatically between passes).

## Layout after trim

**Main paper** keeps only the core story:
- `tab_e1` lexical quality
- `tab_ho` omission vs hallucination (central)
- `tab_e3` temporal prior vs constant windows
- `tab_paired` paired statistics
- `tab_fields` slim conflict rates
- `tab_e4` 3-row TCD ablation
- `fig4` temporal prior histogram
- `fig2` 2-video qualitative

**Appendix** (`sections/12_appendix.tex`) holds the rest: dataset/baselines/hyperparams, pipeline figure, full NLI, full fields, no-crash, taxonomy, TCD-diff, full ablation + bar chart, 3-video qualitative.

## Before submission

* Author block is still **Anonymous Authors**.
* Reproducibility URL is a placeholder.
* For a strict 6-page limit you can drop the appendix from the conference PDF and keep it as supplementary material.
