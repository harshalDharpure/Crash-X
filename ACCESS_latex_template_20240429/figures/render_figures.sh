#!/usr/bin/env bash
# Pre-render the TikZ/pgfplots figures in src/ to standalone PDFs.
#
# Why: tikz and pgfplots patch the LaTeX output routine, which collides with the
# ltxgrid-based output routine in ieeeaccess.cls and injects a blank page after
# almost every page. Rendering the pictures outside the Access document and
# including the resulting PDFs means the paper never loads tikz.
#
# Run this after editing anything in src/, then recompile the paper.
set -euo pipefail

FIGDIR="$(cd "$(dirname "$0")" && pwd)"
BUILD="$FIGDIR/_build"
mkdir -p "$BUILD"

# IEEE Access geometry, measured from ieeeaccess.cls
TEXTWIDTH=505.12177pt
COLUMNWIDTH=242.67355pt

render () {
  local name="$1" width="$2"
  local out="$BUILD/${name}_pic.tex"
  cat > "$out" <<EOF
\\documentclass[border=2pt]{standalone}
\\usepackage[T1]{fontenc}
\\usepackage{times}
\\usepackage{amsmath,amssymb}
\\usepackage{xcolor}
\\usepackage{tikz}
\\usetikzlibrary{arrows.meta,positioning,shapes.geometric,fit,calc}
\\usepackage{pgfplots}
\\pgfplotsset{compat=1.17}
\\newcommand{\\crashlogic}{CrashLogic-7B}
\\newcommand{\\tcd}{TCD}
\\newcommand{\\Hcost}{\\ensuremath{\\mathcal{C}_{\\mathrm{H}}}}
\\newcommand{\\Ocost}{\\ensuremath{\\mathcal{C}_{\\mathrm{O}}}}
\\newcommand{\\code}[1]{\\texttt{#1}}
\\newlength{\\figwidth}\\setlength{\\figwidth}{${width}}
\\begin{document}
\\input{${FIGDIR}/src/${name}.tikz}
\\end{document}
EOF
  ( cd "$BUILD" && pdflatex -interaction=nonstopmode -halt-on-error "${name}_pic.tex" > "${name}.log" 2>&1 ) \
    || { echo "FAILED: $name (see $BUILD/${name}.log)"; grep -m1 -A5 '^! ' "$BUILD/${name}.log"; exit 1; }
  cp "$BUILD/${name}_pic.pdf" "$FIGDIR/${name}.pdf"
  printf '%-22s %s\n' "$name" "$(pdfinfo "$FIGDIR/${name}.pdf" | grep 'Page size')"
}

render fig1_pipeline       "$TEXTWIDTH"
render fig3_ablation_bars  "$TEXTWIDTH"
render fig4_temporal_prior "$COLUMNWIDTH"

echo "PDFs written to $FIGDIR"
