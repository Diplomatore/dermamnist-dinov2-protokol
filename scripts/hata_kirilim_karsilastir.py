# -*- coding: utf-8 -*-
"""
hata_kirilim_karsilastir.py — K5 "önce/sonra" hata kırılımı (YALNIZ VALIDATION): hata_kirilim.py çıktılarını
(uzun CSV + özet JSON) kollar arasında yan yana dizer. Yeni ölçüm yapmaz; test dosyası açmaz.

Girdi: sonuclar/hata-kirilim-<kol>.csv + sonuclar/hata_kirilim_ozet_<kol>.json (hata_kirilim.py --kosumlar ... --etiket <kol>)
  ve K3.5 yerel baseline'ı sonuclar/hata-kirilim.csv / hata_kirilim_ozet.json (etiket 'baseline').
Çıktı: sonuclar/hata-kirilim-h10-karsilastirma.csv (geniş biçim: anahtar × kol, "ort ± std")
       + stdout'a zor-örnek geçiş sayıları (baseline_colab hep-yanlış → H10).
Kullanım (WS kökü): env/.venv/Scripts/python.exe -I scripts/hata_kirilim_karsilastir.py
"""
import csv
import json
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
S = WS / "sonuclar"
KOLLAR = [("baseline_yerel", "hata-kirilim.csv", "hata_kirilim_ozet.json"),
          ("baseline_colab", "hata-kirilim-baseline_colab.csv", "hata_kirilim_ozet_baseline_colab.json"),
          ("h02_r18in_ft224_colab", "hata-kirilim-h02_r18in_ft224_colab.csv", "hata_kirilim_ozet_h02_r18in_ft224_colab.json"),
          ("h08_r18in_ft224_d4_colab", "hata-kirilim-h08_r18in_ft224_d4_colab.csv", "hata_kirilim_ozet_h08_r18in_ft224_d4_colab.json"),
          ("h10_dinov2b_k4_ft224", "hata-kirilim-h10_dinov2b_k4_ft224.csv", "hata_kirilim_ozet_h10_dinov2b_k4_ft224.json")]
SEC = {"genel", "sinif", "cogunluk", "karisiklik", "sizinti", "kaynak", "dermamnist_c", "seed", "teshis", "son_epoch"}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    tablo, sira = {}, []
    for kol, fcsv, _ in KOLLAR:
        with open(S / fcsv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["bolum"] not in SEC:
                    continue
                k = (r["bolum"], r["olcu"], r["alt_kume"], r["sinif"])
                if k not in tablo:
                    tablo[k] = {}
                    sira.append(k)
                v = float(r["deger_ort"])
                tablo[k][kol] = f"{v:.4f}" + (f" ± {float(r['deger_std']):.4f}" if r["deger_std"] else "")
    out = S / "hata-kirilim-h10-karsilastirma.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bolum", "olcu", "alt_kume", "sinif"] + [k for k, _, _ in KOLLAR])
        for k in sira:
            w.writerow(list(k) + [tablo[k].get(kol, "") for kol, _, _ in KOLLAR])
    # zor örnek geçişleri (val indeksleri; 5 seed'in hepsinde yanlış)
    oz = {kol: json.loads((S / fj).read_text(encoding="utf-8")) for kol, _, fj in KOLLAR}
    hy = {kol: set(o["zor"]["hep_yanlis_indeks"]) for kol, o in oz.items()}
    b, h, r2, r8 = hy["baseline_colab"], hy["h10_dinov2b_k4_ft224"], hy["h02_r18in_ft224_colab"], hy["h08_r18in_ft224_d4_colab"]
    gecis = {"baseline_colab_hep_yanlis": len(b), "h10_hep_yanlis": len(h), "ortak_b_h10": len(b & h),
             "b_duzeltilen_h10": len(b - h), "h10_yeni": len(h - b),
             "h02c_hep_yanlis": len(r2), "h08c_hep_yanlis": len(r8),
             "ortak_h02c_h10": len(r2 & h), "ortak_h08c_h10": len(r8 & h), "ucunun_ortak_h02c_h08c_h10": len(r2 & r8 & h)}
    (S / "hata-kirilim-h10-zor-gecis.json").write_text(json.dumps(gecis, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[karşılaştır] {len(sira)} anahtar × {len(KOLLAR)} kol → {out}")
    print("[zor geçiş]", gecis)
    return 0


if __name__ == "__main__":
    sys.exit(main())
