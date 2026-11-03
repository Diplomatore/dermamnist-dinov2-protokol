# -*- coding: utf-8 -*-
"""F4 — seçim döngüsü: tur başına kartların val ACC'si (5 seed ort ± std), karar türü, donanım bölgesi.
Kaynaklar (yalnız val; test açılmadan):
  sonuclar/leaderboard.csv               : yerel koşumlar (Tur 0–3, Tur 5 H13), sütun val_acc (seed satırları)
  sonuclar/k5_kuru_val_sonuc.json        : kollar.{h10,h11,h12}.seed.acc (Tur 4, Colab L4)
  gunluk/deney-gunlugu.md satır 11–25    : kart → tur, karar (metin olarak doğrulanır)
Kart/varyant/karar eşlemesi evidence/scripts/build_evidence.py KARTLAR listesiyle aynıdır."""
import csv
import statistics as st
from _ortak import DW, dat_yaz, jload

# (tur, kart, varyant, karar kodu, kaynak, x kayması, etiket konumu)
# karar kodu: 0 baseline/replikasyon, 1 kept (TUT), 2 killed (ÖLDÜR), 3 analysis, 4 inspect → test adayı (lejant: "test candidate")
KART = [
    (0, "B", "baseline", 0, "lb", 0.0, "west"),
    (1, "H01", "h01_dinov2b_lp_r224", 1, "lb", -0.22, "east"),
    (1, "H02", "h02_r18in_ft224", 1, "lb", 0.0, "south"),
    (1, "H03 r28", "h03_dinov2b_lp_r28", 3, "lb", 0.22, "west"),
    (1, "r64", "h03_dinov2b_lp_r64", 3, "lb", 0.22, "west"),
    (1, "r128", "h03_dinov2b_lp_r128", 3, "lb", 0.22, "west"),
    (2, "H04", "h04_h02_logitayar", 1, "lb", -0.22, "east"),
    (2, "H05", "h05_r18in_ft224_cw05", 2, "lb", 0.0, "south"),
    (2, "H06", "h06_h01_logitayar", 3, "lb", 0.22, "west"),
    (3, "H07", "h07_h02_tta8", 2, "lb", -0.22, "east"),
    (3, "H08", "h08_r18in_ft224_d4", 4, "lb", 0.0, "south"),
    (3, "H09", "h09_r18in_ft224_renk02", 2, "lb", 0.22, "west"),
    (4, "H10 (W)", "h10", 1, "kv", -0.22, "south"),
    (4, "H11", "h11", 1, "kv", 0.0, "north"),
    (4, "H12", "h12", 3, "kv", 0.22, "west"),
    (5, "H13", "h13_h02c_h10_lpo", 2, "lb", 0.0, "south"),
]
GUNLUK_KARAR = {"H01": "TUT", "H02": "TUT", "H04": "TUT", "H05": "ÖLDÜR", "H07": "ÖLDÜR", "H08": "İNCELE",
                "H09": "ÖLDÜR", "H10": "TUT", "H11": "TUT", "H13": "ÖLDÜR", "H03": "analiz", "H06": "analiz",
                "H12": "analiz"}

rows = list(csv.DictReader((DW / "sonuclar/leaderboard.csv").open(encoding="utf-8-sig")))
kv = jload("sonuclar/k5_kuru_val_sonuc.json")
gunluk = (DW / "gunluk/deney-gunlugu.md").read_text(encoding="utf-8").splitlines()

# günlük karar metni denetimi (kod ↔ günlük)
for kart, karar in GUNLUK_KARAR.items():
    s = next(x for x in gunluk[10:25] if f"| {kart} " in x or f"| {kart} (" in x)
    h = s.split("|")
    # analiz kartlarında işaret 'Hipotez' sütunundadır ("H03 (analiz)"), diğerlerinde 'Karar' sütununda
    assert (karar in h[3]) if karar == "analiz" else (karar in h[8]), (kart, h[3], h[8])

nokta, en_iyi = [], {}
for tur, kart, var, kod, kay, dx, anc in KART:
    if kay == "lb":
        seeds = [float(r["val_acc"]) for r in rows if r["varyant"] == var]
    else:
        seeds = kv["kollar"][var]["seed"]["acc"]
    assert len(seeds) == 5, kart
    m, s = st.mean(seeds), st.stdev(seeds)
    lab = "{" + kart + "}"
    # etiket kayması (pt; yerleşim, veri değil): işaretçiyi örtmesin
    kx, ky = {"west": (5, 0), "east": (-5, 0), "south": (0, 5), "north": (0, -6)}[anc]
    nokta.append([f"{tur + dx:.2f}", f"{m:.6f}", f"{s:.6f}", kod, lab, anc, kx, ky])
    if kod in (0, 1, 4):  # ölmemiş ve analiz olmayan kartlar → "best surviving card"
        en_iyi[tur] = max(en_iyi.get(tur, 0.0), m)
dat_yaz("fig_tur.dat", ["x", "m", "s", "kod", "lab", "anc", "kx", "ky"], nokta,
        "leaderboard.csv val_acc (yerel), k5_kuru_val_sonuc.json kollar.{h10,h11,h12}.seed.acc (L4)")

# en iyi yaşayan kart (kümülatif maksimum; Tur 5'te TUT yok → taşınır)
bas, kum = [], 0.0
for tur in range(6):
    kum = max(kum, en_iyi.get(tur, 0.0))
    bas.append([tur - 0.45, f"{kum:.6f}"])
    bas.append([tur + 0.45, f"{kum:.6f}"])
dat_yaz("fig_tur_enIyi.dat", ["x", "y"], bas, "fig_tur.dat, kod ∈ {0,1,4} kümülatif maksimum")

# donanım bölgeleri: Tur 0–3 yerel GTX 1650, Tur 4–5 Colab L4 (günlük satır 21–25 '(COLAB' / '(CPU, val)')
assert "(COLAB" in gunluk[21] and "(COLAB" in gunluk[23] and "(CPU, val)" in gunluk[24]
dat_yaz("fig_tur_bolge.dat", ["x0", "x1", "xm", "lab"],
        [[-0.5, 3.5, 1.5, "{Local GTX 1650 (rounds 0--3)}"], [3.5, 5.5, 4.5, "{Colab L4 (rounds 4--5)}"]],
        "gunluk/deney-gunlugu.md satır 11–25 (tur ↔ donanım)")
