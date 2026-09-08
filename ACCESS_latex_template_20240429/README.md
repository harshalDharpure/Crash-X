# CrashX — IEEE Access submission package

This folder holds the IEEE Access version of the CrashX paper, ported from the
IEEE conference (`IEEEtran`) version in `../paper/ieee_crashx/`.

## Compile

Main file: **`crashx_access.tex`**. Run `./build.sh`, which does
pdflatex → bibtex → pdflatex ×2 and then reports errors, overfull boxes,
page count, and any blank pages.

On Overleaf: upload `harshal.zip` and set
`crashx_access.tex` as the main document.

Current status: **16 pages, 0 errors, 0 blank pages.**

## Do not load tikz or pgfplots in this document

`ieeeaccess.cls` builds on `ltxgrid`, which replaces the LaTeX output routine.
`tikz` and `pgfplots` patch that same output routine, and the two are
incompatible: loading them raises `Extra \else` inside `\@opcol` and emits a
blank page after nearly every page.

The three picture figures are therefore kept as TikZ source in `figures/src/`
and pre-rendered to PDF by `figures/render_figures.sh`, which builds them with
the `standalone` class at exact IEEE Access widths (`\textwidth` 505.12pt,
`\columnwidth` 242.67pt). The paper includes the resulting PDFs with
`\includegraphics`. After editing anything in `figures/src/`, run
`./figures/render_figures.sh` and then rebuild.

Two other class quirks are worked around in the preamble of
`crashx_access.tex`:

- `\textbf{$...$}` (literal math inside `\textbf`) raises
  `Extra }, or forgotten $` with this class. Write `$\mathbf{...}$` instead.
  Macros that use `\ensuremath`, such as `\Hcost`, are safe inside `\textbf`.
- The class bundles Times and Formata without small-caps or slanted shapes, so
  `\textsc` silently degrades to upright roman. The preamble maps the missing
  shapes onto the standard PostScript equivalents.

## Layout

| Path | Contents |
| --- | --- |
| `crashx_access.tex` | Main file: `ieeeaccess` class, preamble, macros, front matter |
| `build.sh` | Full build + error / blank-page report |
| `sections/` | 13 section files (`00_abstract` … `12_appendix`) |
| `tables/` | 16 table files |
| `figures/*.tex` | Float wrappers (2 are colour-coded qualitative tables, 3 include a PDF) |
| `figures/src/*.tikz` | Editable TikZ/pgfplots source for the three picture figures |
| `figures/*.pdf` | Pre-rendered figures, produced by `render_figures.sh` |
| `refs.bib` | 58 references |
| `ieeeaccess.cls`, `IEEEtran.cls`, `IEEEtran.bst`, `spotcolor.sty`, `t1-*`, `*.png` | Official IEEE Access template assets |
| `access_template_reference.tex` | Pristine IEEE Access sample file, kept for reference only |
| `ACCESS_latex_template_20240429/` | Original unmodified template download |

## What changed relative to the conference version

- `IEEEtran` (conference) → `ieeeaccess` class; `IEEEkeywords` → `keywords`;
  added `\history`, `\doi`, `\address`, `\corresp`, `\tfootnote`, `\markboth`,
  `\titlepgskip`, `\PARstart`, author biographies, and `\EOD`.
- Section 3 is rewritten around **CrashX-1500**, our own human-annotated crash
  explanation dataset (`Car_Crash_Text_Dataset_ground_truth.xlsx`): the nine-field
  annotation schema, the annotation protocol, label distributions, and an
  annotation-quality limitation are now stated explicitly.
- Every place that previously credited CCD with the forensic labels now credits
  our annotators; CCD is cited only as the source of the raw video.
- `tables/tab_dataset_splits.tex` moved from the appendix into the main body.
- Abstract cut to 251 words to meet the IEEE Access 150–250 word limit.
- `\paragraph` restored to an unnumbered run-in heading; the class default
  renders it as "a:", "b:", … which reads as a list rather than a heading.
- Fixed a latent pgfplots bug in the temporal-prior figure: symbolic x
  coordinates written as `$[0,1)$` cannot be parsed, because the `)` and `,`
  terminate the coordinate. The labels now use plain keys plus `xticklabels`.
  This bug also broke the conference build and is fixed there too.

## Placeholders to fill before submission

`crashx_access.tex` still contains template values for `\doi`, `\history`,
author names 2 and 3, `\address`, `\corresp` e-mail, the `\tfootnote` funding
statement, and the three biographies.
