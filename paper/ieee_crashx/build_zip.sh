#!/usr/bin/env bash
# Pack Overleaf-ready zip (no aux/pdf junk).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="$(cd "$ROOT/.." && pwd)/CrashX_IEEE_Overleaf.zip"
TMP="$(mktemp -d)"
STAGE="$TMP/ieee_crashx"

mkdir -p "$STAGE"
rsync -a \
  --exclude='*.aux' --exclude='*.log' --exclude='*.out' \
  --exclude='*.bbl' --exclude='*.blg' --exclude='*.synctex.gz' \
  --exclude='*.pdf' --exclude='.DS_Store' \
  "$ROOT/" "$STAGE/"

# Prefer rsync; fall back to cp if rsync missing
if ! command -v rsync >/dev/null 2>&1; then
  rm -rf "$STAGE"
  mkdir -p "$STAGE"
  cp -a "$ROOT/." "$STAGE/"
  find "$STAGE" -type f \( -name '*.aux' -o -name '*.log' -o -name '*.out' \
    -o -name '*.bbl' -o -name '*.blg' -o -name '*.pdf' -o -name '*.synctex.gz' \) -delete
fi

rm -f "$OUT"
(cd "$TMP" && zip -r -q "$OUT" ieee_crashx)
rm -rf "$TMP"
echo "Wrote $OUT"
ls -lh "$OUT"
