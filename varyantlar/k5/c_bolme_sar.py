#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
c_bolme_sar.py — bir varyant betiğini C1/C2 verisiyle AYNI SÜREÇTE koşar (betik değişmez; K5-ONKAYIT §4.2 "tarif aynen").
  python colab/fp32_kos.py varyantlar/k5/c_bolme_sar.py --veri c1 -- varyantlar/r18_imagenet/ince_ayar.py --seed 0 ... --out {RUN_DIR}/metrics.json
Akış: c_bolme.uygula(veri) → sys.argv/sys.path betiğinkine çevrilir → runpy (run_name="__main__").
"""
import runpy
import sys
from pathlib import Path


def main():
    argv = sys.argv[1:]
    if "--" not in argv or len(argv) < 4 or argv[0] != "--veri":
        raise SystemExit("kullanım: c_bolme_sar.py --veri c1|c2 -- <betik.py> [argümanlar...]")
    veri = argv[1]
    i = argv.index("--")
    betik = Path(argv[i + 1]).resolve()
    if not betik.exists():
        raise SystemExit(f"HATA: betik yok: {betik}")
    if betik.name == "ince_ayar_vit.py" and "--alt-kume" in argv[i + 2:]:
        # ince_ayar_vit.etiket_yukle önbellek varken "ilk N" etiketi okur → yamalı tabakalı alt kümeyle hizasız olurdu
        raise SystemExit("HATA: C verisinde ince_ayar_vit.py --alt-kume desteklenmez (C kuyruklarında kullanılmıyor)")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # varyantlar/
    from k5 import c_bolme
    c_bolme.uygula(veri)
    sys.argv = [str(betik)] + argv[i + 2:]
    sys.path[0] = str(betik.parent)
    runpy.run_path(str(betik), run_name="__main__")


if __name__ == "__main__":
    main()
