#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_derivations.py — türetilmiş ifadeler (evidence-contract kural 3).
Yalnız evidence/evidence.json kayıtlarından hesaplar; her türetme kaynak id'leri + formülle
notes/derivations.md'ye yazılır. GA'sı olmayan nokta farkları "(n)" etiketlidir.
Test tahmin dosyalarına dokunulmaz (brief madde 13).
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
WS = Path(__file__).resolve().parents[2]
K = {r["id"]: r for r in json.loads((WS / "evidence" / "evidence.json").read_text(encoding="utf-8"))["kayitlar"]}


def v(i):
    return K[i]["deger"]


def f(x, n=2, isaret=True):
    s = f"{x:+.{n}f}" if isaret else f"{x:.{n}f}"
    return s.replace("-", "−").replace(".", ",")


satir = ["# Türetilmiş değerler (otomatik — `evidence/scripts/build_derivations.py`)", "",
         "*Kaynak: `evidence/evidence.json`. Formüldeki her terim bir kanıt id'sidir. \"(n)\" = GA'sız nokta farkı; "
         "bu değerler için anlamlılık ifadesi kullanılmaz. Puan = ×100.*", ""]


def d(kod, ad, formul, deger, ids, not_=""):
    satir.append(f"| {kod} | {ad} | `{formul}` | **{deger}** | {', '.join(ids)} | {not_} |")


satir += ["## 1. Nokta farkları (resmî test, (n))", "",
          "| Kod | İfade | Formül | Değer | Kaynak id | Not |", "|---|---|---|---|---|---|"]
for kod, a, b in [("D01", "h10", "h01"), ("D02", "h02", "h01"), ("D03", "h11", "h12")]:
    x = 100 * (v(f"T-{a}-acc") - v(f"T-{b}-acc"))
    d(kod, f"{K[f'T-{a}-acc']['yontem']} − {K[f'T-{b}-acc']['yontem']} test ACC", f"100·(T-{a}-acc − T-{b}-acc)",
      f(x) + " puan (n)", [f"T-{a}-acc", f"T-{b}-acc"], "tam değerden; karar.md §4 yuvarlanmış ACC'lerden hesapladığı için ±0,01 fark edebilir (ablasyon.csv ile aynı)")

satir += ["", "## 2. Val → test kayması (karar.md §1.2)", "",
          "| Kod | İfade | Formül | Değer | Kaynak id | Not |", "|---|---|---|---|---|---|"]
for i, kol in enumerate(["baseline", "h01", "h02", "h08", "h10", "h11", "h12", "t4"], start=1):
    x = 100 * (v(f"T-{kol}-acc") - v(f"V-{kol}-acc"))
    d(f"D04.{i}", f"{K[f'T-{kol}-acc']['yontem']} test − val ACC", f"100·(T-{kol}-acc − V-{kol}-acc)",
      f(x) + " puan (n)", [f"T-{kol}-acc", f"V-{kol}-acc"], "W = H10; karar.md §1.2 yuvarlanmış değerlerden −0,07 yazar — makalede tam değerden türetilen bu değer ya da \"≈ −0,1\" kullanılmalı" if kol == "h10" else "")

satir += ["", "## 3. Sınıf duyarlılığı farkları (resmî test; [T] `kollar.*.ort.duyarlilik_*`; GA yok)", "",
          "| Kod | İfade | Formül | Değer | Kaynak id | Not |", "|---|---|---|---|---|---|"]
SIN = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
for cift in [("h08", "h02"), ("h10", "h02"), ("h10", "h08")]:
    a, b = cift
    tot = 0.0
    for s in SIN:
        x = 100 * (v(f"T-{a}-duyarlilik_{s}") - v(f"T-{b}-duyarlilik_{s}"))
        tot += x
        d(f"D05.{a}{b}.{s}", f"{K[f'T-{a}-acc']['yontem']} − {K[f'T-{b}-acc']['yontem']} duyarlılık {s}",
          f"100·(T-{a}-duyarlilik_{s} − T-{b}-duyarlilik_{s})", f(x) + " puan (n)",
          [f"T-{a}-duyarlilik_{s}", f"T-{b}-duyarlilik_{s}"])
    dd = 100 * (v(f"T-{a}-dengeli_acc") - v(f"T-{b}-dengeli_acc"))
    d(f"D05.{a}{b}.sağlama", "Σ Δduyarlılık / 7 = Δdengeli (sağlama)", "Σ_s D05.s / 7", f(tot / 7) + f" (Δdengeli {f(dd)})",
      [f"T-{a}-dengeli_acc", f"T-{b}-dengeli_acc"], "eşit olmalı")
    if cift == ("h08", "h02"):
        ddf = 100 * (v("T-h08-duyarlilik_df") - v("T-h02-duyarlilik_df"))
        dva = 100 * (v("T-h08-duyarlilik_vasc") - v("T-h02-duyarlilik_vasc"))
        katki = (ddf + dva) / 7
        d("D06", "H08c − H02c dengeli Δ'sına df + vasc katkısı", "(D05.h08h02.df + D05.h08h02.vasc)/7",
          f(katki) + " puan", ["T-h08-duyarlilik_df", "T-h02-duyarlilik_df", "T-h08-duyarlilik_vasc",
                                "T-h02-duyarlilik_vasc"])
        d("D07", "df + vasc payı (dengeli Δ içinde)", "D06 / Δdengeli(H08c−H02c)", f"%{katki / dd * 100:.0f}",
          ["T-h08-dengeli_acc", "T-h02-dengeli_acc"], "brief madde 2: %56")
        n_df, n_va = v("DM-test-df"), v("DM-test-vasc")
        d("D08", "seed başına ek doğru görüntü (df, vasc)", "Δdf·n_df/100 , Δvasc·n_vasc/100",
          f"{ddf * n_df / 100:.1f} + {dva * n_va / 100:.1f} ≈ {(ddf * n_df + dva * n_va) / 100:.0f}".replace(".", ","),
          ["DM-test-df", "DM-test-vasc"], "brief madde 2: ≈ 8")

satir += ["", "## 4. Çoğunluk tabanına göre resmî test ACC (C tablolarındaki ACC − 0,7726 karşılığı)", "",
          "| Kod | İfade | Formül | Değer | Kaynak id | Not |", "|---|---|---|---|---|---|"]
taban = v("DM-test-nv") / v("T-meta-n")
d("D09.0", "resmî test çoğunluk tabanı (tam)", "DM-test-nv / T-meta-n", f"{taban:.4f}".replace(".", ","),
  ["DM-test-nv", "T-meta-n"], "veri manifesti 0,669 ile aynı")
for i, kol in enumerate(["baseline", "h01", "h02", "h08", "h10", "h11", "h12"], start=1):
    x = 100 * (v(f"T-{kol}-acc") - taban)
    d(f"D09.{i}", f"{K[f'T-{kol}-acc']['yontem']} resmî test ACC − taban", f"100·(T-{kol}-acc − D09.0)", f(x) + " puan",
      [f"T-{kol}-acc", "D09.0"])

satir += ["", "## 5. Bütçe", "", "| Kod | İfade | Formül | Değer | Kaynak id | Not |", "|---|---|---|---|---|---|"]
d("D10", "Tur-4 bütçesi (tek kalem verilmemiş)", "BUT-tur4 − BUT-tur3", f(v("BUT-tur4") - v("BUT-tur3"), 1, False) + " sa",
  ["BUT-tur4", "BUT-tur3"], "yaklaşık; Tur-4 satırı oturum tabanı, Tur-3 GPU-sa — karışık taban, yalnız kaba sıra")

satir += ["", "## 6. Literatür çıpası (bağlam; aynı koşulda üretilmedi — review 5.1, 2.1)", "",
          "| Kod | İfade | Formül | Değer | Kaynak id | Not |", "|---|---|---|---|---|---|"]
LITN = "farklı koşul (bölme/seçim/augmentasyon farklı ya da bildirilmemiş); sayısal yöntem kıyası değil, yalnız konumlandırma"
d("D11.1", "H10 resmî test ACC − Wu & Papież uçtan uca DINOv2-B ACC", "100·T-h10-acc − LIT-wu2025-e2e-acc",
  f(100 * v("T-h10-acc") - v("LIT-wu2025-e2e-acc")) + " puan", ["T-h10-acc", "LIT-wu2025-e2e-acc"], "hakemsiz ön baskı; " + LITN)
d("D11.2", "H01c resmî test ACC − Wu & Papież doğrusal sonda ACC", "100·T-h01-acc − LIT-wu2025-lp-acc",
  f(100 * v("T-h01-acc") - v("LIT-wu2025-lp-acc")) + " puan", ["T-h01-acc", "LIT-wu2025-lp-acc"], LITN)
d("D11.3", "H02c resmî test ACC − MedMNIST+ ImageNet R18 E2E 224 px ACC", "100·T-h02-acc − LIT-doerrich2025-e2e224-acc",
  f(100 * v("T-h02-acc") - v("LIT-doerrich2025-e2e224-acc")) + " puan", ["T-h02-acc", "LIT-doerrich2025-e2e224-acc"], LITN)
d("D11.4", "Zhang & Li Vanilla R18-224 ACC − H02c resmî test ACC", "100·(LIT-zhang2026-acc − T-h02-acc)",
  f(100 * (v("LIT-zhang2026-acc") - v("T-h02-acc"))) + " puan", ["LIT-zhang2026-acc", "T-h02-acc"],
  "tur-1'de doğrulandı (PMC13413053 Tablo 1(d)); tarif farkı: wd 0,05, ısınma, batch 128, etiket yumuşatma 0,1 — " + LITN)

satir += ["", "## 7. Çoğunluk tabanı farkı (ek satırlar; review 7.5)", "",
          "| Kod | İfade | Formül | Değer | Kaynak id | Not |", "|---|---|---|---|---|---|"]
d("D12", "T4 C0 ACC − C-test çoğunluk tabanı", "100·(T-C0-t4-acc − T-C0-cogunluk)",
  f(100 * (v("T-C0-t4-acc") - v("T-C0-cogunluk"))) + " puan", ["T-C0-t4-acc", "T-C0-cogunluk"], "betimsel ek satır")

satir += ["", "## 8. Eğitim kaybı oranları (val dönemi, K4 yerel koşumları; review B4)", "",
          "| Kod | İfade | Formül | Değer | Kaynak id | Not |", "|---|---|---|---|---|---|"]
for kod, a, ad in [("D13.1", "H08", "D4 (H08) / H02"), ("D13.2", "H09", "renk oynaması (H09) / H02")]:
    sa, sb = K[f"EGK-{a}-ep5"]["seed_degerleri"], K["EGK-H02-ep5"]["seed_degerleri"]
    oran_ort = v(f"EGK-{a}-ep5") / v("EGK-H02-ep5")
    seedli = [x / y for x, y in zip(sa, sb)]
    d(kod, f"epoch 5 eğitim kaybı oranı, {ad}", f"EGK-{a}-ep5 / EGK-H02-ep5 (seed-eşli: s_i/s_i)",
      f"{oran_ort:.1f}× (5 seed ortalaması); s0 {seedli[0]:.1f}×; seed-eşli aralık {min(seedli):.1f}–{max(seedli):.1f}×".replace(".", ","),
      [f"EGK-{a}-ep5", "EGK-H02-ep5"], "tahlil-tur3 '≈21×' / '≈2,5×' değerleri s0 oranıdır (EGK-tahlil-satiri)")
d("D13.3", "epoch 30 eğitim kaybı oranı, D4 (H08) / H02", "EGK-H08-ep30 / EGK-H02-ep30",
  f"{v('EGK-H08-ep30') / v('EGK-H02-ep30'):.1f}×".replace(".", ","), ["EGK-H08-ep30", "EGK-H02-ep30"], "")
d("D13.4", "epoch 30 eğitim kaybı oranı, renk (H09) / H02", "EGK-H09-ep30 / EGK-H02-ep30",
  f"{v('EGK-H09-ep30') / v('EGK-H02-ep30'):.1f}×".replace(".", ","), ["EGK-H09-ep30", "EGK-H02-ep30"], "")

satir += ["", "## 9. Test içi lezyon çokluğu (yalnız meta veri; review 9.1)", "",
          "| Kod | İfade | Formül | Değer | Kaynak id | Not |", "|---|---|---|---|---|---|"]
for kod, ad, pay, payda in [("D14.1", "resmî test", "LES-test-coklu-goruntu", "T-meta-n"),
                            ("D14.2", "S alt kümesi", "LES-S-coklu-goruntu", "T-meta-nS"),
                            ("D14.3", "N alt kümesi", "LES-N-coklu-goruntu", "T-meta-nN"),
                            ("D14.4", "C-test", "LES-Ctest-coklu-goruntu", "T-C0-n")]:
    d(kod, f"{ad}: birden çok görüntülü lezyonlara ait görüntü payı", f"{pay} / {payda}",
      f"%{100 * v(pay) / v(payda):.1f}".replace(".", ","), [pay, payda], "")

satir += ["", "## 10. Bootstrap p çözünürlüğü (review 7.6)", "",
          "| Kod | İfade | Formül | Değer | Kaynak id | Not |", "|---|---|---|---|---|---|"]
B = v("ON-boot-B")
d("D15.1", "en küçük bootstrap p", "1 / (ON-boot-B + 1)", f"{1 / (B + 1):.6f}".replace(".", ","), ["ON-boot-B"], "")
d("D15.2", "Holm sonrası alt sınır (4 karşılaştırmalı Aile 1; 2 karşılaştırmalı Aile 2)", "4/(B+1) ; 2/(B+1)",
  f"{4 / (B + 1):.4f} ; {2 / (B + 1):.4f}".replace(".", ","), ["ON-boot-B"], "tablolardaki 0,002 ve 0,001")

satir += ["", "## 11. Parametre oranları (review 4.8/4.17)", "",
          "| Kod | İfade | Formül | Değer | Kaynak id | Not |", "|---|---|---|---|---|---|"]
d("D16.1", "H10 eğitilen / toplam parametre", "PRM-H10-egitilebilir / PRM-H10-toplam",
  f"%{100 * v('PRM-H10-egitilebilir') / v('PRM-H10-toplam'):.1f}".replace(".", ","), ["PRM-H10-egitilebilir", "PRM-H10-toplam"], "")
d("D16.2", "H10 toplam / H02c toplam parametre", "PRM-H10-toplam / PRM-H02",
  f"{v('PRM-H10-toplam') / v('PRM-H02'):.1f}×".replace(".", ","), ["PRM-H10-toplam", "PRM-H02"], "")

(WS / "notes" / "derivations.md").write_text("\n".join(satir) + "\n", encoding="utf-8")
print(f"{sum(1 for s in satir if s.startswith('| D'))} türetme → notes/derivations.md")
