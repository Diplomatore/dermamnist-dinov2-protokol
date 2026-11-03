#!/bin/bash
# birleştir + derle (+ isteğe bağlı pdf denetimi)
# Ortam değişkenleri: MAKALE_WS (varsayılan: betiğin iki üst klasörü = makale/), PYTHON311 (varsayılan: python),
# TECTONIC (varsayılan: tectonic), PDF_DENETIM (isteğe bağlı: pdf_denetim.py yolu).
# Not: makale metni (sections/*.tex) bu depoda yer almaz; betik yeniden üretim zincirini belgelemek için durur.
WS="${MAKALE_WS:-$(cd "$(dirname "$0")/../.." && pwd)}"
S="$WS/paper/araclar"
PY="${PYTHON311:-python}"
cd "$WS" || exit 9
PYTHONIOENCODING=utf-8 "$PY" "$S/birlestir.py" "$WS" || exit 8
cd "$WS/paper" || exit 9
mkdir -p "$WS/notes"
timeout 600 "${TECTONIC:-tectonic}" --keep-logs --keep-intermediates main.tex > "$WS/notes/derleme-cikti.txt" 2>&1
echo "tectonic rc=$?"
grep -n "^!" main.log | head
grep -c "Overfull" main.log | sed 's/^/Overfull: /'
grep -n "LaTeX Warning" main.log | head
grep "Output written" main.log
cd "$WS"
if [ -n "${PDF_DENETIM:-}" ]; then
  PYTHONIOENCODING=utf-8 uv run --python 3.11 --with pymupdf python "$PDF_DENETIM" --ws . 2>/dev/null | sed -n 1,30p
  echo "pdf_denetim rc=${PIPESTATUS[0]}"
fi
