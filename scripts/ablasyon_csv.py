# -*- coding: utf-8 -*-
"""
ablasyon_csv.py — K5 ablasyon zinciri B → H01 → H02 → H10 / H08 (karar.md §4) → sonuclar/ablasyon.csv (D5 eksiği).

Yeni koşum YOK, test YENİDEN AÇILMAZ: sayılar yalnız dondurulmuş üç JSON'dan okunur
  val  : sonuclar/k5_kuru_val_sonuc.json  (Colab val, test açılmadan; kollar/kiyas)
  test : sonuclar/k5_test_sonuc.json      (resmî test; kollar/kiyas/aile1/aile2)
  C0   : sonuclar/k5_test_sonuc.json      (C0 alt kümesi; C0.kol/C0.kiyas)
  C1   : sonuclar/k5_c1_sonuc.json        (C1.kol/C1.kiyas/aile_C1_1/aile_C1_2)
JSON'da olmayan değer BOŞ bırakılır (uydurma/türetme yok). Tek istisna `not` sütunundaki "(n)" nokta farkları:
iki kolun JSON'daki ort ACC'sinin farkı, GA'sı yok (karar.md §4 "(n)" ile aynı tanım).
Δ'lar puan (×100), GA %95 görüntü bootstrap'ı (B = 2.000). Holm p tek yönlü, eşleştirilmiş t (aile içinde).
Kullanım (WS kökü): env/.venv/Scripts/python.exe -I scripts/ablasyon_csv.py
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
S = WS / "sonuclar"
F_VAL, F_TEST, F_C1 = "k5_kuru_val_sonuc.json", "k5_test_sonuc.json", "k5_c1_sonuc.json"

# (sıra, basamak, kol, rol, ne değişti, taban kol)  — karar.md §4 tablosu
SATIRLAR = [
    (1, "B baseline_c", "baseline", "zincir", "R18, 28 px, sıfırdan (rakip; val-AUC seçimi)", None),
    (2, "H01c", "h01", "zincir", "+224 px, +DINOv2-B önağırlık, donuk sonda (LR)", "baseline"),
    (3, "H02c", "h02", "zincir", "R18-ImageNet, 224 px, tam ince ayar (eşit önağırlık hakkı)", "baseline"),
    (4, "H10 (W)", "h10", "zincir", "DINOv2-B, 224 px, son 4 blok ince ayar", "h02"),
    (5, "H08c", "h08", "zincir", "H02 + eğitimde D4", "h02"),
    (6, "H10 ↔ H08c", "h10", "karışık kıyas (betimsel, Holm dışı)", "omurga + augmentasyon birlikte", "h08"),
    (7, "H12", "h12", "Pareto", "DINOv2-S donuk sonda", "h01"),
    (8, "H11", "h11", "Pareto", "DINOv2-S tam ince ayar", "h02"),
    (9, "T4 = H08c ⊕ H10", "t4", "ek satır (betimsel; iddia taşımaz)", "seed eşli ½;½ log-olasılık topluluğu", "h10"),
    (10, "baseline_c (son epoch)", "baseline_son", "ikincil (rakip seçim kuralı)", "rakip, son-epoch seçimi", "baseline"),
]
# (n) notları: karar.md §4'teki GA'sız nokta farkları (kol, karşı kol)
N_NOTLARI = {3: ("h02", "h01"), 4: ("h10", "h01"), 8: ("h11", "h12")}


def oku(ad):
    p = S / ad
    return json.loads(p.read_text(encoding="utf-8")), hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def yuv(x, n=4):
    return "" if x is None else round(float(x), n)


def kol_ozet(J, kol, on):
    k = J["kollar"].get(kol)
    if not k:
        return {}
    return {f"{on}_acc_ort": yuv(k["ort"]["acc"]), f"{on}_acc_std": yuv(k["std"]["acc"]),
            f"{on}_auc_ort": yuv(k["ort"]["auc"]), f"{on}_dengeli_ort": yuv(k["ort"]["dengeli_acc"]),
            f"{on}_dengeli_std": yuv(k["std"]["dengeli_acc"]), f"{on}_n_seed": len(k["seed"]["acc"]),
            f"{on}_n_goruntu": J["n"]}


def kiyas_bul(sozluk, kol, taban):
    """sozluk[kol-taban] (yön kol − taban). Yoksa None."""
    return sozluk.get(f"{kol}-{taban}") if taban else None


def delta_acc(kq, on, ek):
    """kiyas biçimi (acc.delta/ga, oran) ya da C0 biçimi (delta_puan/ga_puan)."""
    if kq is None:
        return {}
    if "acc" in kq:
        d, ga = kq["acc"]["delta"] * 100, [g * 100 for g in kq["acc"]["ga"]]
    else:
        d, ga = kq["delta_puan"], kq["ga_puan"]
    return {f"{on}_d{ek}_puan": yuv(d, 2), f"{on}_d{ek}_ga_alt": yuv(ga[0], 2), f"{on}_d{ek}_ga_ust": yuv(ga[1], 2)}


def holm(aileler, kol, taban, on, ek):
    for a in aileler:
        q = a.get(f"{kol}-{taban}")
        if q is not None:
            return {f"{on}_d{ek}_holm_t": f"{q['holm_t']:.3g}", f"{on}_d{ek}_holm_boot": f"{q['holm_boot']:.3g}"}
    return {}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    V, hv = oku(F_VAL)
    T, ht = oku(F_TEST)
    C, hc = oku(F_C1)
    assert V["kip"] == "kuru-val" and T["kip"] == "test" and C["kip"] == "c1", (V["kip"], T["kip"], C["kip"])
    # Bütünlük: C1 JSON'undaki resmî-test bölümü, test JSON'u ile aynı olmalı (aynı dondurulmuş tahminler).
    assert C["kollar"]["h10"]["ort"]["acc"] == T["kollar"]["h10"]["ort"]["acc"]
    c1 = C["C1"]
    satirlar = []
    for sira, bas, kol, rol, ne, taban in SATIRLAR:
        r = {"sira": sira, "basamak": bas, "kol": kol, "rol": rol, "ne_degisti": ne, "taban": taban or ""}
        # val
        r.update(kol_ozet(V, kol, "val"))
        r.update(delta_acc(kiyas_bul(V["kiyas"], kol, taban), "val", "_taban"))
        r.update(delta_acc(kiyas_bul(V["kiyas"], kol, "baseline") if kol not in ("baseline",) else None, "val", "_B"))
        # resmî test
        r.update(kol_ozet(T, kol, "test"))
        r.update(delta_acc(kiyas_bul(T["kiyas"], kol, taban), "test", "_taban"))
        r.update(holm([T["aile1"], T["aile2"]], kol, taban, "test", "_taban") if taban else {})
        r.update(delta_acc(kiyas_bul(T["kiyas"], kol, "baseline") if kol != "baseline" else None, "test", "_B"))
        r.update(holm([T["aile1"]], kol, "baseline", "test", "_B") if kol != "baseline" else {})
        # C0 (resmî test'in lezyon-ayrık alt kümesi, yeniden eğitimsiz)
        k0 = T["C0"]["kol"].get(kol)
        if k0:
            r.update({"c0_acc_ort": yuv(k0["acc_ort"]), "c0_acc_std": yuv(k0["acc_std"]),
                      "c0_dengeli_ort": yuv(k0["dengeli_ort"]), "c0_n_seed": len(T["kosumlar"].get(kol, [])) or "",
                      "c0_n_goruntu": T["C0"]["n"]})
        r.update(delta_acc(kiyas_bul(T["C0"]["kiyas"], kol, taban), "c0", "_taban"))
        r.update(delta_acc(kiyas_bul(T["C0"]["kiyas"], kol, "baseline") if kol != "baseline" else None, "c0", "_B"))
        # C1 (lezyon-ayrık C bölmesi, yeniden eğitim)
        k1 = c1["kol"].get(kol)
        if k1:
            r.update({"c1_acc_ort": yuv(k1["acc_ort"]), "c1_acc_std": yuv(k1["acc_std"]),
                      "c1_auc_ort": yuv(k1["auc_ort"]), "c1_dengeli_ort": yuv(k1["dengeli_ort"]),
                      "c1_n_seed": len(c1["kosumlar"][kol])})
        r.update(delta_acc(kiyas_bul(c1["kiyas"], kol, taban), "c1", "_taban"))
        r.update(holm([c1["aile_C1_1"], c1["aile_C1_2"]], kol, taban, "c1", "_taban") if taban else {})
        r.update(delta_acc(kiyas_bul(c1["kiyas"], kol, "baseline") if kol != "baseline" else None, "c1", "_B"))
        r.update(holm([c1["aile_C1_1"]], kol, "baseline", "c1", "_B") if kol != "baseline" else {})
        # (n) notu
        if sira in N_NOTLARI:
            a, b = N_NOTLARI[sira]
            d = (T["kollar"][a]["ort"]["acc"] - T["kollar"][b]["ort"]["acc"]) * 100
            r["not"] = f"test {a}−{b} {d:+.2f} puan (n: nokta farkı, GA yok)"
        r["kaynak_val"] = f"sonuclar/{F_VAL} (sha256 {hv}…)"
        r["kaynak_test_c0"] = f"sonuclar/{F_TEST} (sha256 {ht}…)"
        r["kaynak_c1"] = f"sonuclar/{F_C1} (sha256 {hc}…)" if (k1 or kiyas_bul(c1["kiyas"], kol, taban)) else ""
        satirlar.append(r)
    alanlar = []
    for r in satirlar:
        for k in r:
            if k not in alanlar:
                alanlar.append(k)
    # Sütunlar değerlendirme bloklarına göre: kimlik | val | test | C0 | C1 | not + kaynaklar
    def grup(k):
        for i, on in enumerate(("val_", "test_", "c0_", "c1_", "not", "kaynak_"), start=1):
            if k.startswith(on):
                return i
        return 0
    alanlar = sorted(alanlar, key=lambda k: (grup(k), alanlar.index(k)))
    cikti = S / "ablasyon.csv"
    with open(cikti, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=alanlar)
        w.writeheader()
        for r in satirlar:
            w.writerow(r)
    print(f"[ablasyon] {len(satirlar)} satır × {len(alanlar)} sütun → {cikti}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
