# -*- coding: utf-8 -*-
"""F3 — sınıf bazında resmî test duyarlılığı (B / H02c / H08c / H10), ort ± seed std.
Kaynak [T] sonuclar/k5_test_sonuc.json: kollar.<kol>.ort.duyarlilik_<sınıf>, kollar.<kol>.std.duyarlilik_<sınıf>.
Sınıf başına test n: data/data_manifest.md 'Dağılım (28 px)' tablosu.
Sınıf farkı GA'ları JSON'da yok; test tahminlerinden türetmek yasak (brief madde 13) → GA çizilmez."""
from _ortak import DW, dat_yaz, jload

SINIF = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
KOLLAR = ["baseline", "h02", "h08", "h10"]
KAY = [-0.3, -0.1, 0.1, 0.3]
T = jload("sonuclar/k5_test_sonuc.json")

n_test = {}
for s in (DW / "data/data_manifest.md").read_text(encoding="utf-8").splitlines():
    for c in SINIF:
        if f"({c}) |" in s and s.startswith("| "):
            n_test[c] = int([x.strip() for x in s.split("|")][5].replace(".", ""))
assert len(n_test) == 7 and sum(n_test.values()) == T["n"]

satir = []
for i, c in enumerate(SINIF):
    for j, kol in enumerate(KOLLAR):
        v = T["kollar"][kol]
        satir.append([f"{i + KAY[j]:.2f}", j, f"{v['ort'][f'duyarlilik_{c}']:.6f}", f"{v['std'][f'duyarlilik_{c}']:.6f}"])
dat_yaz("fig_sinif.dat", ["x", "k", "m", "s"], satir,
        "[T] kollar.{baseline,h02,h08,h10}.ort/std.duyarlilik_<sınıf>")
# x etiketleri + nadir sınıf gölgesi (n < 50: df 23, vasc 29 — veri manifestinden)
dat_yaz("fig_sinif_x.dat", ["x", "lab", "nadir"],
        [[i, "{%s\\\\(n = %d)}" % (c, n_test[c]), 1.0 if n_test[c] < 50 else 0.0] for i, c in enumerate(SINIF)],
        "data/data_manifest.md sınıf dağılımı (test sütunu)")
