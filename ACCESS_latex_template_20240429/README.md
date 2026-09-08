# CrashX — IEEE Access submission package

This folder holds the IEEE Access version of the CrashX paper, ported from the
IEEE conference (`IEEEtran`) version in `../paper/ieee_crashx/`.

## Compile

Main file: **`crashx_access.tex`**

```
pdflatex crashx_access
bibtex   crashx_access
pdflatex crashx_access
pdflatex crashx_access
```

On Overleaf: upload `CrashX_IEEE_Access_Overleaf.zip` and set
`crashx_access.tex` as the main document.

## Layout

| Path | Contents |
| --- | --- |
| `crashx_access.tex` | Main file: `ieeeaccess` class, preamble, macros, front matter |
| `sections/` | 13 section files (`00_abstract` … `12_appendix`) |
| `tables/` | 16 table files |
| `figures/` | 5 TikZ/pgfplots figure files |
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

## Placeholders to fill before submission

`crashx_access.tex` still contains template values for `\doi`, `\history`,
author names 2 and 3, `\address`, `\corresp` e-mail, the `\tfootnote` funding
statement, and the three biographies.
