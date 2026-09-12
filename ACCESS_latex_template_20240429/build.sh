#!/usr/bin/env bash
# Full build of the IEEE Access paper: pdflatex -> bibtex -> pdflatex x2.
# Reports errors, page count, and any blank pages.
set -uo pipefail
cd "$(dirname "$0")"
export PATH=/DATA/vaneet_2221cs15/texlive/2026/bin/x86_64-linux:$PATH
MAIN=crashx_access

rm -f $MAIN.aux $MAIN.bbl $MAIN.blg $MAIN.log $MAIN.out $MAIN.toc
pdflatex -interaction=nonstopmode $MAIN.tex > build1.log 2>&1
bibtex $MAIN > build_bib.log 2>&1
pdflatex -interaction=nonstopmode $MAIN.tex > build2.log 2>&1
pdflatex -interaction=nonstopmode $MAIN.tex > build3.log 2>&1

echo "=== errors ==="
grep "^! " build3.log | sort | uniq -c || echo "  none"
echo "=== undefined refs / citations ==="
grep -c "Warning.*undefined" build3.log
echo "=== overfull hbox (worst 6) ==="
grep -o "Overfull \\\\hbox ([0-9.]*pt too wide)" build3.log | sed 's/[^0-9.]*\([0-9.]*\)pt.*/\1/' | sort -rn | head -6 | tr '\n' ' '; echo
echo "=== pages ==="
pdfinfo $MAIN.pdf | grep Pages

echo "=== blank page scan ==="
pdftotext $MAIN.pdf - 2>/dev/null | awk -v RS='\f' '
  { gsub(/[[:space:]]/,"",$0); if (length($0) < 40) printf "  page %d: %d chars\n", NR, length($0); n=NR }
  END { print "  (" n " pages scanned)" }'
