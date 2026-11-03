# -*- coding: utf-8 -*-
"""Figür üreticileri için ortak yollar ve .dat yazıcısı (yalnız stdlib).
Veri YALNIZ deney workspace'inin dondurulmuş çıktılarından okunur (brief madde 13);
tahmin_test*.npy dosyaları açılmaz."""
import json
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
DW = WS.parent  # depo kökü = deney çalışma alanı (makale/ = WS)
RC = DW / "runs_colab" if (DW / "runs_colab").is_dir() else DW / "runs_meta" / "runs_colab"  # depoda yalnız JSON damgaları runs_meta/ altında
DATA = WS / "figures" / "data"
DATA.mkdir(parents=True, exist_ok=True)


def jload(rel):
    return json.loads((DW / rel).read_text(encoding="utf-8"))


def dat_yaz(ad, basliklar, satirlar, kaynak):
    """Sekme ayrımlı .dat (pgfplots: col sep=tab). İlk satır yorum: kaynak."""
    yol = DATA / ad
    with yol.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(basliklar) + "\n")
        for s in satirlar:
            f.write("\t".join(str(x) for x in s) + "\n")
    (DATA / (ad + ".kaynak")).write_text(kaynak + "\n", encoding="utf-8")
    print(f"{ad}: {len(satirlar)} satır")


def eng(x, n=2, isaret=True):
    """İngilizce figür metni için sayı: +2.26 / -0.36 (TeX eksi: $-$)."""
    s = f"{x:+.{n}f}" if isaret else f"{x:.{n}f}"
    return s.replace("-", "$-$")
