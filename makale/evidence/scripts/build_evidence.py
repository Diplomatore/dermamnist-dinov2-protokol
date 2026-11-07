#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_evidence.py — DermaMNIST · DINOv2 ↔ ResNet-18 (A2, protokol duyarlılığı) kanıt defteri üreticisi.

Kanıt sözleşmesi (references/evidence-contract.md): her sayı gerçek bir çıktı dosyasından okunur;
kaynak dosya + alan/satır kaydedilir. YENİ SAYI TÜRETİLMEZ (brief madde 13): yalnız dondurulmuş JSON
alanları, koşum metrics.json'ları (C1 C-test'in tek açılışında yazılmış), val leaderboard'ları ve MD/CSV
çıktıları okunur. Test tahmin dosyaları (tahmin_test*.npy) AÇILMAZ.

Tek dönüşüm: oran olarak saklanan farklar (kiyas.*.delta, ga) puana çevrilir (×100); ham değer
`ham_deger` alanında durur. Türetilmiş ifadeler notes/derivations.md'de (build_derivations.py).

Kullanım (proje kökünden):
  python makale_workspace/dermamnist-dinov2-protokol/evidence/scripts/build_evidence.py [--cache <cache_dir>]
Çıktı: WS/evidence/evidence.json (+ --cache verilirse <cache_dir>/evidence_base.json)
Yalnız stdlib (numpy gerekmez).
"""
import argparse
import csv
import hashlib
import json
import re
import statistics as st
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WS = Path(__file__).resolve().parents[2]
DW = WS.parent  # depo kökü = deney çalışma alanı (makale/ = WS)
RC = DW / "runs_colab" if RC.is_dir() else DW / "runs_meta" / "runs_colab"  # depoda yalnız JSON damgaları runs_meta/ altında

KIP_T = "test (bir kez açıldı; 2026-10-11 21:12, ikinci_acilis: null)"
KIP_C0 = "test → C0 yeniden puanlama (resmî test tahminlerinin C-test alt kümesi; aynı tek açılış)"
KIP_C1 = "C1 C-test (yeniden eğitim; C-test kol başına bir kez açıldı, 2026-10-12)"
KIP_V = "val (K4 / K5 kuru-val; test açılmadan)"
KIP_VERI = "veri manifesti"
KIP_G = "deney günlüğü / orkestrasyon"
KIP_LIT = "literatür (aynı koşulda üretilmedi)"

VS_T = "DermaMNIST resmî test (n=2005)"
VS_S = "DermaMNIST resmî test, S alt kümesi (lezyonu train'de; n=693)"
VS_N = "DermaMNIST resmî test, N alt kümesi (n=1312)"
VS_C0 = "C0: C-test alt kümesi, yeniden eğitimsiz (n=1227)"
VS_C1 = "C1: DermaMNIST-C bölmesi C-test, yeniden eğitimli (n=1227)"
VS_V = "DermaMNIST resmî val (n=1003)"
VS_CV = "C-val ∩ resmî val (n=573; lezyon-ayrık, yeniden eğitimsiz)"

KOL_AD = {"baseline": "B", "h01": "H01c", "h02": "H02c", "h08": "H08c", "h10": "H10",
          "h11": "H11", "h12": "H12", "baseline_son": "B_son", "t4": "T4"}
KOL_ETIKET = {
    "t4": ["betimsel/ek", "çoklu karşılaştırma düzeltmesi yok", "iddia taşımaz (KULLANICI KARARI 12)",
           "AUC>0,99 bayrağı denetlenmedi (KARAR 12)"],
    "h11": ["Pareto satırı (betimsel)"], "h12": ["Pareto satırı (betimsel)"],
    "baseline_son": ["ikincil (rakip son-epoch seçimi)"],
    "h10": ["W (val'de seçilmiş kazanan)"], "h02": ["adil kıyas satırı"],
    "h08": ["doğrulanmış güçlü R18 satırı"], "baseline": ["rakip (otoriter)"],
}
METRIK_T = ["acc", "auc", "dengeli_acc", "dengeli_acc_5sinif", "makro_f1", "ece15", "nll", "nv_payi",
            "duyarlilik_akiec", "duyarlilik_bcc", "duyarlilik_bkl", "duyarlilik_df", "duyarlilik_mel",
            "duyarlilik_nv", "duyarlilik_vasc"]
YON = {"acc": "yuksek", "auc": "yuksek", "dengeli_acc": "yuksek", "dengeli_acc_5sinif": "yuksek",
       "makro_f1": "yuksek", "ece15": "dusuk", "nll": "dusuk", "nv_payi": None}

KAYIT = []
IDLER = set()


def ekle(id_, metrik, yontem, veri_seti, deger, birim, dosya, alan, kip, kaynak="yazilim", **ek):
    if id_ in IDLER:
        raise SystemExit(f"tekrarlı id: {id_}")
    if deger is None:
        raise SystemExit(f"değer yok: {id_}")
    IDLER.add(id_)
    r = {"id": id_, "metrik": metrik, "yontem": yontem, "veri_seti": veri_seti, "deger": deger,
         "birim": birim, "kaynak": kaynak, "kaynak_dosya": dosya, "kaynak_alan": alan, "kip": kip,
         "atif": ek.pop("atif", None), "durum": "gecerli"}
    r.update({k: v for k, v in ek.items() if v is not None})
    KAYIT.append(r)
    return r


def jload(rel):
    return json.loads((DW / rel).read_text(encoding="utf-8"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def puan(x):
    return None if x is None else round(100.0 * x, 6)


def gapuan(g):
    return None if g is None else [round(100.0 * g[0], 6), round(100.0 * g[1], 6)]


def satir_bul(rel, desen, baslangic=1):
    """Dosyada desenin geçtiği ilk satırı (1 tabanlı) ve metnini döndürür; yoksa durur."""
    satirlar = (DW / rel).read_text(encoding="utf-8").splitlines()
    for i, s in enumerate(satirlar[baslangic - 1:], start=baslangic):
        if desen in s:
            return i, s
    raise SystemExit(f"desen bulunamadı: {rel}: {desen!r}")


def tr_sayi(s):
    return float(s.replace("−", "-").replace("+", "").replace(",", "."))


SAYI = r"[+−-]?\d+(?:[.,]\d+)?"


def delta_ga(metin):
    """'+1,51 [−0,23; +3,28]' → (1.51, [-0.23, 3.28])"""
    m = re.search(rf"({SAYI})\s*\[({SAYI});\s*({SAYI})\]", metin)
    if not m:
        return None, None
    return tr_sayi(m.group(1)), [tr_sayi(m.group(2)), tr_sayi(m.group(3))]


# ---------------------------------------------------------------- [T] resmî test
def test_kayitlari():
    rel = "sonuclar/k5_test_sonuc.json"
    d = jload(rel)
    assert d["kip"] == "test" and d["ikinci_acilis"] is None
    ekle("T-meta-n", "örnek sayısı", "—", VS_T, d["n"], "görüntü", rel, "n", KIP_T)
    ekle("T-meta-nS", "S alt kümesi büyüklüğü", "—", VS_T, d["n_S"], "görüntü", rel, "n_S", KIP_T)
    ekle("T-meta-nN", "N alt kümesi büyüklüğü", "—", VS_T, d["n_N"], "görüntü", rel, "n_N", KIP_T)
    ekle("T-meta-P6", "P6 özdeşlik kapısı (val argmax + maks |Δp|)", "7 kol × 5 seed", VS_T,
         d["P6"]["gecti"], "mantıksal", rel, "P6.gecti", KIP_T,
         not_=f"{len(d['P6']['satir'])} koşum; hata {len(d['P6']['hatalar'])}",
         p6_kosum_sayisi=len(d["P6"]["satir"]),
         p6_maks_fark=max(v["ozdeslik_p_maks_fark"] for v in d["P6"]["satir"].values()))
    ekle("T-meta-tarih", "test açılış zamanı", "—", VS_T, d["tarih"], "zaman", rel, "tarih", KIP_T,
         betik_sha256=d["betik_sha256"], onkayit_sha256=d["onkayit_sha256"])

    for kol, v in d["kollar"].items():
        for m in METRIK_T:
            if m not in v["ort"]:
                continue
            ekle(f"T-{kol}-{m}", m, KOL_AD[kol], VS_T, v["ort"][m], "oran", rel,
                 f"kollar.{kol}.ort.{m} (+std, seed)", KIP_T, std=v["std"][m],
                 seed_degerleri=v["seed"][m], n_seed=len(v["seed"][m]), yon=YON.get(m, "yuksek"),
                 etiket=KOL_ETIKET.get(kol))
        h4 = v.get("katman_H04")
        if h4:
            for a in ["tau", "acc_ort", "dengeli_ort", "dengeli5_ort"]:
                ekle(f"T-H04-{kol}-{a}", f"H04 logit katmanı {a}", KOL_AD[kol], VS_T, h4[a],
                     "" if a == "tau" else "oran", rel, f"kollar.{kol}.katman_H04.{a}", KIP_T,
                     etiket=["katman (karar vermez)"], seed_degerleri=h4.get("acc_seed") if a == "acc_ort" else None)
        rt = v.get("katman_RT")
        if rt:
            ekle(f"T-RT-{kol}-ece15", "R-T sıcaklık katmanı ECE-15", KOL_AD[kol], VS_T, rt["ece15_ort"], "oran",
                 rel, f"kollar.{kol}.katman_RT.ece15_ort", KIP_T, etiket=["katman (karar vermez)"],
                 T_seed=rt["T"], argmax_degismedi=rt["argmax_degismedi"])
            ekle(f"T-RT-{kol}-nll", "R-T sıcaklık katmanı NLL", KOL_AD[kol], VS_T, rt["nll_ort"], "nat",
                 rel, f"kollar.{kol}.katman_RT.nll_ort", KIP_T, etiket=["katman (karar vermez)"])

    for cift, v in d["kiyas"].items():
        b, a = cift.split("-", 1)
        yon_ad = f"{KOL_AD[b]} − {KOL_AD[a]}"
        et = ["betimsel/ek"] if cift.startswith("t4") else (
            ["betimsel, Holm dışı"] if cift in ("h10-h08", "h11-baseline", "h12-baseline", "h11-h02",
                                                "h12-h01", "baseline_son-baseline") else None)
        for m in ["acc", "dengeli_acc", "dengeli_acc_5sinif"]:
            x = v[m]
            ekle(f"T-K-{cift}-{m}", f"Δ{m} (bootstrap)", yon_ad, VS_T, puan(x["delta"]), "puan", rel,
                 f"kiyas.{cift}.{m}.delta (+ga, p_boot)", KIP_T, ham_deger=x["delta"], ga=gapuan(x["ga"]),
                 p_boot=x["p_boot"], A_deger=x["A"], B_deger=x["B"], etiket=et)
        for m in ["t_acc", "t_dengeli"]:
            if m in v:
                x = v[m]
                ekle(f"T-K-{cift}-{m}", f"{m} seed-eşli fark (tek yönlü t)", yon_ad, VS_T, puan(x["ort"]),
                     "puan", rel, f"kiyas.{cift}.{m}", KIP_T, sd_puan=puan(x["sd"]), p_tek=x["p_tek"],
                     fark_seed_puan=[puan(f) for f in x["fark_seed"]], etiket=et)
        if "acc_S_ham" in v:
            ekle(f"T-K-{cift}-dS", "Δ_S ACC (ham)", yon_ad, VS_S, puan(v["acc_S_ham"]["delta"]), "puan", rel,
                 f"kiyas.{cift}.acc_S_ham", KIP_T, ga=gapuan(v["acc_S_ham"]["ga"]), etiket=et)
            ekle(f"T-K-{cift}-dN", "Δ_N ACC (ham)", yon_ad, VS_N, puan(v["acc_N_ham"]["delta"]), "puan", rel,
                 f"kiyas.{cift}.acc_N_ham", KIP_T, ga=gapuan(v["acc_N_ham"]["ga"]), etiket=et)
            ekle(f"T-K-{cift}-dSdN", "Δ_S − Δ_N (ham; kural nesnesi)", yon_ad, VS_T,
                 puan(v["ham_dS_eksi_dN"]["nokta"]), "puan", rel, f"kiyas.{cift}.ham_dS_eksi_dN", KIP_T,
                 ga=gapuan(v["ham_dS_eksi_dN"]["ga"]), etiket=et)
            ekle(f"T-K-{cift}-Spay", "net kazancın S payı", yon_ad, VS_T, v["S_payi_net"], "oran", rel,
                 f"kiyas.{cift}.S_payi_net", KIP_T, etiket=et)
            for t in ["tab_std_dS", "tab_std_dN", "tab_std_dS_eksi_dN"]:
                ekle(f"T-K-{cift}-{t}", f"{t} (tabaka-std; ikincil, etiket vermez)", yon_ad, VS_T,
                     puan(v[t]["nokta"]), "puan", rel, f"kiyas.{cift}.{t}", KIP_T, ga=gapuan(v[t]["ga"]),
                     etiket=(et or []) + ["ikincil"])
        if "farkli_dogru_goruntu" in v:
            ekle(f"T-K-{cift}-farkli", "farklı doğru görüntü sayısı", yon_ad, VS_T, v["farkli_dogru_goruntu"],
                 "görüntü", rel, f"kiyas.{cift}.farkli_dogru_goruntu", KIP_T, etiket=et)

    for cift, x in d["kiyas_H04_katmani"].items():
        b, a = cift.split("-", 1)
        ekle(f"T-KH04-{cift}-acc", "ΔACC (H04 katmanı)", f"{KOL_AD[b]} − {KOL_AD[a]}", VS_T,
             puan(x["acc"]["delta"]), "puan", rel, f"kiyas_H04_katmani.{cift}.acc", KIP_T,
             ga=gapuan(x["acc"]["ga"]), etiket=["katman (karar vermez)"])
        ekle(f"T-KH04-{cift}-dengeli", "Δdengeli ACC (H04 katmanı)", f"{KOL_AD[b]} − {KOL_AD[a]}", VS_T,
             puan(x["dengeli_acc"]["delta"]), "puan", rel, f"kiyas_H04_katmani.{cift}.dengeli_acc", KIP_T,
             ga=gapuan(x["dengeli_acc"]["ga"]), etiket=["katman (karar vermez)"])

    for aile in ["aile1", "aile2"]:
        marj = "+2,0" if aile == "aile1" else "+0,5"
        for cift, x in d[aile].items():
            b, a = cift.split("-", 1)
            ekle(f"T-{aile.upper()}-{cift}", "ΔACC (Holm ailesi)", f"{KOL_AD[b]} − {KOL_AD[a]}", VS_T,
                 x["delta_puan"], "puan", rel, f"{aile}.{cift}", KIP_T, ga=x["ga_puan"], p_t=x["p_t"],
                 p_boot=x["p_boot"], holm_t=x["holm_t"], holm_boot=x["holm_boot"],
                 kural_gecti=x["kural_gecti"], anlamli=x["anlamli"], marj=marj, aile=aile)

    for ad, x in d["dogrulama"].items():
        k = "H08" if ad.startswith("H08") else "H10"
        ekle(f"T-DOG-{k}", f"doğrulama kuralı {ad}", k, VS_T, x["dogrulandi"], "mantıksal", rel,
             f"dogrulama['{ad}']", KIP_T, delta_puan=x["delta_puan"], ga=x["ga_puan"],
             gorulmemis=x["gorulmemis"])
    x = d["dogrulama"]["H08 (§2.1)"]
    ekle("T-DOG-H08-dengeli5", "H08c − H02c dengeli-5 (df/vasc hariç) Δ", "H08c − H02c", VS_T,
         x["dengeli_5sinif_delta_puan"], "puan", rel, "dogrulama['H08 (§2.1)'].dengeli_5sinif_delta_puan", KIP_T,
         ga=gapuan(d["kiyas"]["h08-h02"]["dengeli_acc_5sinif"]["ga"]))
    ekle("T-G2-H08-bayrak", "görülmemiş bayrağı dengeli_delta>6,5", "H08c − H02c", VS_T,
         x["gorulmemis"]["dengeli_delta>6,5"], "mantıksal", rel, "olcutler.G2_bayraklar['H08 (§2.1)']", KIP_T,
         uyari=("Ön-kayıtlı +6,5 'beklenmedik büyüklük' eşiği aşıldı (dengeli Δ +8,41). Teşhis: arıza yok "
                "(karar.md §2.1; KULLANICI KARARI 12 kabul). Makalede zorunlu beyan cümlesi (brief madde 2). "
                "Eşik gevşetilmedi."))

    for cift, x in d["SN"].items():
        b, a = cift.split("-", 1)
        uy = None
        if cift == "h08-h02":
            uy = ("T-SN2 ('H08 kazancı S/N arasında dengeli dağılır') TUTMADI: ham Δ_S−Δ_N GA'sı 0'ı dışlıyor. "
                  "Betiğin 'desteklendi' etiketi burada 'Δ_S>Δ_N' anlamına gelir; makalede kullanılmaz (brief md 4).")
        ekle(f"T-SN-{cift}", f"S/N kuralı ({x['hipotez']})", f"{KOL_AD[b]} − {KOL_AD[a]}", VS_T,
             x["ham_dS_eksi_dN_puan"], "puan", rel, f"SN.{cift}", KIP_T, ga=x["ham_ga_puan"],
             kural_etiketi=x["kural"], beklenen=x["beklenen"], dS_ham_puan=x["dS_ham_puan"],
             dN_ham_puan=x["dN_ham_puan"], tab_puan=x["tab_dS_eksi_dN_puan"], tab_ga=x["tab_ga_puan"],
             uyari=uy, etiket=["betimsel, Holm dışı"] if x["hipotez"] == "betimsel" else None)

    c0 = d["C0"]
    ekle("T-C0-n", "C-test alt kümesi büyüklüğü", "—", VS_C0, c0["n"], "görüntü", rel, "C0.n", KIP_C0)
    ekle("T-C0-cogunluk", "çoğunluk (nv) tabanı", "—", VS_C0, c0["cogunluk_tabani"], "oran", rel,
         "C0.cogunluk_tabani", KIP_C0)
    for kol, v in c0["kol"].items():
        ekle(f"T-C0-{kol}-acc", "ACC", KOL_AD[kol], VS_C0, v["acc_ort"], "oran", rel, f"C0.kol.{kol}.acc_ort",
             KIP_C0, std=v["acc_std"], n_seed=5, seed_degerleri_yok="C0 kol başına seed değerleri JSON'da yok",
             etiket=KOL_ETIKET.get(kol))
        ekle(f"T-C0-{kol}-dengeli", "dengeli ACC", KOL_AD[kol], VS_C0, v["dengeli_ort"], "oran", rel,
             f"C0.kol.{kol}.dengeli_ort", KIP_C0, etiket=KOL_ETIKET.get(kol))
        ekle(f"T-C0-{kol}-dengeli5", "dengeli-5 ACC", KOL_AD[kol], VS_C0, v["dengeli5_ort"], "oran", rel,
             f"C0.kol.{kol}.dengeli5_ort", KIP_C0, etiket=KOL_ETIKET.get(kol))
        ekle(f"T-C0-{kol}-accmaj", "ACC − çoğunluk tabanı", KOL_AD[kol], VS_C0, v["acc_eksi_cogunluk"], "oran",
             rel, f"C0.kol.{kol}.acc_eksi_cogunluk", KIP_C0, etiket=KOL_ETIKET.get(kol))
    for cift, x in c0["kiyas"].items():
        b, a = cift.split("-", 1)
        uy = None
        if cift == "h10-h02":
            uy = "C0-1 ön-kaydı: ayırt edilemedi (tutmayan ön-kayıt B2)."
        ekle(f"T-C0-K-{cift}", "ΔACC", f"{KOL_AD[b]} − {KOL_AD[a]}", VS_C0, x["delta_puan"], "puan", rel,
             f"C0.kiyas.{cift}", KIP_C0, ga=x["ga_puan"], p_t=x["p_t"], kural=x["kural"],
             kural_etiketi=x["sonuc"], uyari=uy,
             etiket=["betimsel, Holm dışı"] if x["kural"] == "betimsel" else None)
        ekle(f"T-C0-K-{cift}-dengeli", "Δdengeli ACC", f"{KOL_AD[b]} − {KOL_AD[a]}", VS_C0,
             x["dengeli_delta_puan"], "puan", rel, f"C0.kiyas.{cift}.dengeli_delta_puan", KIP_C0,
             ga=x["dengeli_ga_puan"])
    k = c0["KC0"]
    ekle("T-C0-K0-H10", "K0_H10 = Δ_resmî − Δ_C0 (H10 − H02c)", "H10 − H02c", VS_C0, k["K0_H10_puan"], "puan",
         rel, "C0.KC0.K0_H10_puan", KIP_C0, ga=k["K0_H10_ga"])
    ekle("T-C0-K0-H08", "K0_H08 = Δ_resmî − Δ_C0 (H08c − H02c)", "H08c − H02c", VS_C0, k["K0_H08_puan"], "puan",
         rel, "C0.KC0.K0_H08_puan", KIP_C0, ga=k["K0_H08_ga"])
    ekle("T-C0-KC0", "KÇ0 = K0_H10 − K0_H08", "H10 − H08c", VS_C0, k["K0_H10_eksi_K0_H08_puan"], "puan", rel,
         "C0.KC0", KIP_C0, ga=k["ga_puan"], kural_etiketi=k["kural"], beklenen=k["beklenen"],
         uyari="KÇ0 ön-kaydı ayırt edilemedi (tutmayan ön-kayıt B4).")
    o = d["olcutler"]
    ekle("T-olcut-G1", "G1 (W − baseline_c, resmî test)", "H10 − B", VS_T, o["G1"], "mantıksal", rel,
         "olcutler.G1", KIP_T)
    ekle("T-olcut-V", "V (H10 − H02c, resmî test)", "H10 − H02c", VS_T, o["V"], "mantıksal", rel,
         "olcutler.V", KIP_T)
    ekle("T-olcut-W", "W", "H10", VS_V, o["W"], "metin", rel, "olcutler.W", KIP_T)
    return d


# ---------------------------------------------------------------- [C1]
def c1_kayitlari(T):
    rel = "sonuclar/k5_c1_sonuc.json"
    d = jload(rel)
    c = d["C1"]
    # Aile 1 resmî-test Δ'ları [T] ile aynı olmalı (manifest denetimi) — yeniden doğrula
    for cift, x in d["aile1"].items():
        assert abs(x["delta_puan"] - T["aile1"][cift]["delta_puan"]) < 1e-9, cift
    # Faz 5 tur-2 (review-tur2 K11): "C1 yeniden yüklemesinde sayılar aynı kaldı" → id'li, tam alan denetimi
    BOLUMLER = ["n", "n_S", "n_N", "kollar", "n_tabaka", "kiyas", "kiyas_H04_katmani", "aile1", "aile2",
                "dogrulama", "SN", "C0", "olcutler"]
    oz = {"sayisal": 0, "diger": 0, "maks": 0.0, "farkli": [], "yalniz_c1": [], "yalniz_t": []}

    def _kars(a, b, p):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                if k not in b:
                    oz["yalniz_c1"].append(f"{p}.{k}")
                elif k not in a:
                    oz["yalniz_t"].append(f"{p}.{k}")
                else:
                    _kars(a[k], b[k], f"{p}.{k}")
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            for i, (x, y) in enumerate(zip(a, b)):
                _kars(x, y, f"{p}[{i}]")
        elif all(isinstance(z, (int, float)) and not isinstance(z, bool) for z in (a, b)):
            oz["sayisal"] += 1
            oz["maks"] = max(oz["maks"], abs(a - b))
            if a != b:
                oz["farkli"].append(p)
        else:
            oz["diger"] += 1
            if a != b:
                oz["farkli"].append(p)

    for b_ in BOLUMLER:
        _kars(d[b_], T[b_], b_)
    if oz["farkli"] or oz["yalniz_t"]:
        raise SystemExit(f"C1 yeniden yükleme özdeş değil: {oz['farkli'][:5]} {oz['yalniz_t'][:5]}")
    ekle("C1-yeniden-yukleme-ozdes", "C1 kipinde yeniden yüklenen resmî-test sonuçlarının [T] ile özdeşliği "
         "(karşılaştırılan sayısal alan)", "k5_test.py --kip c1 ↔ --kip test", VS_T, oz["sayisal"], "alan", rel,
         ", ".join(BOLUMLER) + " (k5_test_sonuc.json aynı anahtarlarla)", KIP_C1,
         maks_mutlak_fark=oz["maks"], farkli_alan=len(oz["farkli"]), sayisal_olmayan_ozdes=oz["diger"],
         yalniz_c1_dosyasinda=oz["yalniz_c1"], karsi_dosya="sonuclar/k5_test_sonuc.json",
         betik_sha256_ayni=d["betik_sha256"] == T["betik_sha256"],
         onkayit_sha256_ayni=d["onkayit_sha256"] == T["onkayit_sha256"],
         kosumlar_ayni=d["kosumlar"] == T["kosumlar"], tarih_c1=d["tarih"], tarih_test=T["tarih"],
         not_="Yalnız C1 dosyasında olan alanlar (olcutler.G3, olcutler.L) C1'e özgü ölçütlerdir; resmî test "
              "bölümlerinin hiçbir alanı farklı değil. Dayanak software_manifest.md Belirsizlikler 1 ve satır 138.")
    for kol, v in c["kol"].items():
        seed, seed_kaynak = [], []
        for run in c["kosumlar"][kol]:
            mp = DW / "runs_colab" / run / "metrics.json"
            m = json.loads(mp.read_text(encoding="utf-8"))
            seed.append(m["test_acc"])
            seed_kaynak.append(f"runs_colab/{run}/metrics.json:test_acc")
        assert abs(st.mean(seed) - v["acc_ort"]) < 1e-9 and abs(st.stdev(seed) - v["acc_std"]) < 1e-9, kol
        ekle(f"C1-{kol}-acc", "ACC", KOL_AD[kol], VS_C1, v["acc_ort"], "oran", rel, f"C1.kol.{kol}.acc_ort",
             KIP_C1, std=v["acc_std"], seed_degerleri=seed, seed_kaynak=seed_kaynak, n_seed=5,
             etiket=KOL_ETIKET.get(kol))
        for a, m, b in [("auc_ort", "AUC", "oran"), ("dengeli_ort", "dengeli ACC", "oran"),
                        ("dengeli5_ort", "dengeli-5 ACC", "oran"), ("acc_eksi_cogunluk", "ACC − çoğunluk tabanı", "oran")]:
            ekle(f"C1-{kol}-{a.replace('_ort', '').replace('acc_eksi_cogunluk', 'accmaj')}", m, KOL_AD[kol], VS_C1,
                 v[a], b, rel, f"C1.kol.{kol}.{a}", KIP_C1, etiket=KOL_ETIKET.get(kol))
    for cift, v in c["kiyas"].items():
        b, a = cift.split("-", 1)
        yon_ad = f"{KOL_AD[b]} − {KOL_AD[a]}"
        et = ["betimsel, Holm dışı"] if cift == "h10-h08" else None
        for m in ["acc", "dengeli_acc", "dengeli_acc_5sinif"]:
            x = v[m]
            ekle(f"C1-K-{cift}-{m}", f"Δ{m} (bootstrap)", yon_ad, VS_C1, puan(x["delta"]), "puan", rel,
                 f"C1.kiyas.{cift}.{m}", KIP_C1, ham_deger=x["delta"], ga=gapuan(x["ga"]), p_boot=x["p_boot"],
                 etiket=et)
        x = v["t_acc"]
        ekle(f"C1-K-{cift}-t_acc", "t_acc seed-eşli fark", yon_ad, VS_C1, puan(x["ort"]), "puan", rel,
             f"C1.kiyas.{cift}.t_acc", KIP_C1, sd_puan=puan(x["sd"]), p_tek=x["p_tek"],
             fark_seed_puan=[puan(f) for f in x["fark_seed"]], etiket=et)
    for aile in ["aile_C1_1", "aile_C1_2"]:
        for cift, x in c[aile].items():
            b, a = cift.split("-", 1)
            uy = None
            if aile == "aile_C1_2" and cift == "h10-h02":
                uy = ("L = FALSE: GA alt −0,05 → ayırt edilemedi. Holm t 0,018 / boot 0,031 < 0,05 olsa da kural "
                      "geçmedi; 'anlamlı' denmez (karar.md §7.2; brief madde 1).")
            ekle(f"C1-{aile.replace('aile_C1_', 'A')}-{cift}", "ΔACC (C1 Holm ailesi)", f"{KOL_AD[b]} − {KOL_AD[a]}",
                 VS_C1, x["delta_puan"], "puan", rel, f"C1.{aile}.{cift}", KIP_C1, ga=x["ga_puan"],
                 holm_t=x["holm_t"], holm_boot=x["holm_boot"], kural_gecti=x["kural_gecti"],
                 dengeli_puan=x["dengeli_delta_puan"], dengeli_ga=x["dengeli_ga_puan"],
                 dengeli5_puan=x["dengeli5_delta_puan"], dengeli5_ga=x["dengeli5_ga_puan"], uyari=uy,
                 marj="+2,0" if aile == "aile_C1_1" else "+0,5", aile=aile)
    for cift, x in c["i"].items():
        ekle(f"C1-i-{cift}", "C1 (i) kuralı", cift, VS_C1, x["kural"], "metin", rel, f"C1.i.{cift}", KIP_C1,
             delta_puan=x["delta_puan"], ga=x["ga_puan"], dengeli_yon=x["dengeli_yon"])
    k = c["ii_KC"]
    ekle("C1-K-H10", "K_H10 = Δ_resmî − Δ_C1 (H10 − H02)", "H10 − H02c", VS_C1, k["K_H10_puan"], "puan", rel,
         "C1.ii_KC.K_H10_puan", KIP_C1, ga=k["K_H10_ga"], bilesen_K0=k["bilesen_K0_H10"],
         bilesen_K1=k["bilesen_K1_H10"])
    ekle("C1-K-H08", "K_H08 = Δ_resmî − Δ_C1 (H08 − H02)", "H08c − H02c", VS_C1, k["K_H08_puan"], "puan", rel,
         "C1.ii_KC.K_H08_puan", KIP_C1, ga=k["K_H08_ga"], bilesen_K0=k["bilesen_K0_H08"],
         bilesen_K1=k["bilesen_K1_H08"])
    ekle("C1-KC", "KÇ = K_H10 − K_H08", "H10 − H08c", VS_C1, k["K_H10_eksi_K_H08_puan"], "puan", rel,
         "C1.ii_KC", KIP_C1, ga=k["ga_puan"], kural_etiketi=k["kural"],
         uyari="KÇ ön-kaydı ayırt edilemedi (tutmayan ön-kayıt B5).")
    ekle("C1-KC-oran", "K_H10 / Δ_H10(resmî)", "H10 − H02c", VS_C1, k["oran_K_H10_bolu_delta_resmi"], "oran",
         rel, "C1.ii_KC.oran_K_H10_bolu_delta_resmi", KIP_C1)
    ekle("C1-KC-ifade", "KÇ ifade eşiği (oran ≥ 0,5 ∧ KÇ desteklendi)", "H10", VS_C1, k["ifade_esigi"],
         "mantıksal", rel, "C1.ii_KC.ifade_esigi", KIP_C1,
         uyari="İfade eşiği FALSE: 'kazancın yarısından fazlası kayboluyor' cümlesi yazılamaz (B6).")
    ekle("C1-G3", "G3 (W_C1 − baseline_C1)", "H10 − B", VS_C1, c["G3"], "mantıksal", rel, "C1.G3", KIP_C1)
    ekle("C1-L", "L (H10 − H02, C1)", "H10 − H02c", VS_C1, c["L"], "mantıksal", rel, "C1.L", KIP_C1,
         uyari="L = FALSE (brief madde 1: açıkça yazılır).")
    x = c["iii_baseline"]
    ekle("C1-iii", "baseline C1 beklenti [0,79; 0,85] içinde", "B", VS_C1, x["beklenti_[0,79;0,85]"], "mantıksal",
         rel, "C1.iii_baseline", KIP_C1, acc=x["acc"], gorulmemis_087=x["gorulmemis_>0,87_INCELE"])


# ---------------------------------------------------------------- [KV] val (K5 kuru koşusu)
VAL_TANIM_L4 = ("L4 tanımı: Colab L4'te yeniden eğitilmiş kol ('c' sonekli; H10–H12 yalnız L4), 5 seed ortalaması, "
                "K5 kuru-val çıktısı. Tablo tab:ablasyon doğrulama sütunu ve test aşaması bu tanımı kullanır.")
VAL_TANIM_YEREL = ("yerel tanım: seçim döngüsünde yerel GTX 1650'de eğitilmiş kol (Tur 0–3), 5 seed ortalaması, "
                   "leaderboard.csv. Şekil fig:tur ve Bölüm sec:deneykaydi bu tanımı kullanır; L4 karşılığı V-<kol>-acc.")


def val_kayitlari():
    rel = "sonuclar/k5_kuru_val_sonuc.json"
    d = jload(rel)
    for kol, v in d["kollar"].items():
        for m in ["acc", "auc", "dengeli_acc", "dengeli_acc_5sinif", "ece15", "nll"] + \
                 [f"duyarlilik_{s}" for s in ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]]:
            ekle(f"V-{kol}-{m}", m, KOL_AD[kol], VS_V, v["ort"][m], "oran", rel, f"kollar.{kol}.ort.{m}", KIP_V,
                 std=v["std"][m], seed_degerleri=v["seed"][m], n_seed=len(v["seed"][m]),
                 donanim="Colab L4 (FP32, TF32 kapalı)", etiket=KOL_ETIKET.get(kol),
                 tanim=VAL_TANIM_L4)
    for cift, v in d["kiyas"].items():
        b, a = cift.split("-", 1)
        yon_ad = f"{KOL_AD[b]} − {KOL_AD[a]}"
        ekle(f"V-K-{cift}-acc", "ΔACC (bootstrap)", yon_ad, VS_V, puan(v["acc"]["delta"]), "puan", rel,
             f"kiyas.{cift}.acc", KIP_V, ga=gapuan(v["acc"]["ga"]), p_boot=v["acc"]["p_boot"])
        ekle(f"V-K-{cift}-dengeli", "Δdengeli ACC (bootstrap)", yon_ad, VS_V, puan(v["dengeli_acc"]["delta"]),
             "puan", rel, f"kiyas.{cift}.dengeli_acc", KIP_V, ga=gapuan(v["dengeli_acc"]["ga"]))
        if "acc_S_ham" in v:
            ekle(f"V-K-{cift}-dS", "Δ_S ACC (ham)", yon_ad, VS_V, puan(v["acc_S_ham"]["delta"]), "puan", rel,
                 f"kiyas.{cift}.acc_S_ham", KIP_V, ga=gapuan(v["acc_S_ham"]["ga"]))
            ekle(f"V-K-{cift}-dN", "Δ_N ACC (ham)", yon_ad, VS_V, puan(v["acc_N_ham"]["delta"]), "puan", rel,
                 f"kiyas.{cift}.acc_N_ham", KIP_V, ga=gapuan(v["acc_N_ham"]["ga"]))
    for kol, v in d["C0"]["kol"].items():
        ekle(f"V-CV-{kol}-acc", "ACC", KOL_AD[kol], VS_CV, v["acc_ort"], "oran", rel, f"C0.kol.{kol}.acc_ort",
             KIP_V, std=v["acc_std"])
    ekle("V-CV-cogunluk", "çoğunluk tabanı", "—", VS_CV, d["C0"]["cogunluk_tabani"], "oran", rel,
         "C0.cogunluk_tabani", KIP_V)
    # NOT: d["C1"] ve d["C2"] KURU koşu vekilleridir (C1 = C-val ∩ val vekili, C2 = C1 kopyası) → alınmaz.


# ---------------------------------------------------------------- ablasyon.csv çapraz denetim
def ablasyon_capraz():
    rel = "sonuclar/ablasyon.csv"
    satirlar = list(csv.DictReader((DW / rel).open(encoding="utf-8-sig")))
    idx = {r["id"]: r for r in KAYIT}
    esle = {"test_acc_ort": "T-{k}-acc", "val_acc_ort": "V-{k}-acc", "c0_acc_ort": "T-C0-{k}-acc",
            "c1_acc_ort": "C1-{k}-acc"}
    hata = 0
    for i, r in enumerate(satirlar, start=2):
        for sut, sablon in esle.items():
            if not r.get(sut):
                continue
            kid = sablon.format(k=r["kol"])
            if kid not in idx:
                continue
            if abs(round(idx[kid]["deger"], 4) - float(r[sut])) > 6e-5:
                print(f"UYUŞMAZLIK ablasyon.csv satır {i} {sut}: {r[sut]} ↔ {kid} {idx[kid]['deger']}")
                hata += 1
        ekle(f"ABL-{r['sira']}", "ablasyon zinciri basamağı", r["basamak"], "—", r["ne_degisti"], "metin", rel,
             f"satır {i}", KIP_T, rol=r["rol"], taban=r["taban"] or None, kol=r["kol"], not_=r["not"] or None)
    print(f"ablasyon.csv çapraz denetim: {len(satirlar)} satır, uyuşmazlık {hata}")
    if hata:
        raise SystemExit("ablasyon.csv ↔ JSON uyuşmazlığı")


# ---------------------------------------------------------------- replikasyon + literatür
def replikasyon():
    for rel, ortam, kid in [("sonuclar/replikasyon.md", "yerel GTX 1650", "REP-yerel-test_acc"),
                            ("sonuclar/replikasyon-test_auc.md", "yerel GTX 1650", "REP-yerel-test_auc"),
                            ("sonuclar/replikasyon-test_acc_son.md", "yerel GTX 1650", "REP-yerel-test_acc_son"),
                            ("sonuclar/replikasyon-colab.md", "Colab L4", "REP-L4-test_acc")]:
        i, s = satir_bul(rel, "| Bizim baseline")
        m = re.search(r"\*\*(\d\.\d+) ± (\d\.\d+)\*\*", s)
        j, s2 = satir_bul(rel, "| Seed değerleri")
        seeds = [float(x) for x in re.findall(r"\d\.\d+", s2)]
        k, s3 = satir_bul(rel, "kabul aralığı")
        ka = [float(x) for x in re.findall(r"\[(\d\.\d+), (\d\.\d+)\]", s3)[0]]
        metrik = re.search(r"metrik `([^`]+)`", (DW / rel).read_text(encoding="utf-8")).group(1)
        ekle(kid, metrik, "B (rakip replikasyonu)", VS_T, float(m.group(1)), "oran", rel, f"satır {i} (+ seed satır {j})",
             "test (K3 replikasyon kapısı; tek seferlik, rakip için)", std=float(m.group(2)),
             seed_degerleri=seeds, kabul_araligi=ka, kapı="YEŞİL", donanim=ortam)
    rel = "sonuclar/tekrar-ozeti-colab.md"
    for kol, des in [("baseline", "baseline_colab"), ("h01", "h01_dinov2b_lp_r224_colab"),
                     ("h02", "h02_r18in_ft224_colab"), ("h08", "h08_r18in_ft224_d4_colab")]:
        i, s = satir_bul(rel, f"(`{des}`)")
        h = [x.strip() for x in s.split("|")]
        ekle(f"KC1-{kol}", "Colab tekrar kapısı val ACC (Colab ↔ yerel)", KOL_AD[kol], VS_V,
             float(h[3].split("±")[0]), "oran", rel, f"satır {i}", KIP_V, yerel=float(h[4]),
             fark_puan=float(h[5]), tolerans_puan=float(h[6]), uyum=h[7])

    rel = "notlar/literatur-cipa.md"
    LIT = [  # (id, metrik, yöntem, değer, std, birim, atıf, desen)
        ("LIT-yang2023-acc", "test ACC", "ResNet-18 (28), resmî", 0.735, None, "oran", "yang_2023_medmnistv2",
         "| ResNet-18 (28) | **0.917** | **0.735** |"),
        ("LIT-yang2023-auc", "test AUC", "ResNet-18 (28), resmî", 0.917, None, "oran", "yang_2023_medmnistv2",
         "| ResNet-18 (28) | **0.917** | **0.735** |"),
        ("LIT-yang2023-r224-acc", "test ACC", "ResNet-18 (224, 28'den büyütme)", 0.754, None, "oran",
         "yang_2023_medmnistv2", "| ResNet-18 (224) | 0.920 | 0.754 |"),
        ("LIT-doerrich2025-acc", "test ACC", "ResNet-18 E2E 28 px (ImageNet, sıfır dolgu; 3 seed)", 73.38, 0.61, "%",
         "doerrich_2025_medmnistplus", "| **73.38 ± 0.61** | 88.35 ± 0.28 | 3 seed |"),
        ("LIT-doerrich2025-auc", "test AUC", "ResNet-18 E2E 28 px (ImageNet, sıfır dolgu; 3 seed)", 88.35, 0.28, "%",
         "doerrich_2025_medmnistplus", "| **73.38 ± 0.61** | 88.35 ± 0.28 | 3 seed |"),
        ("LIT-medeiros2026-acc", "test ACC", "ResNet-18 28 px sıfırdan (TTA'sız)", 0.742, None, "oran",
         "medeiros_2026_tta", "| 0.742 (TTA'sız) |"),
        ("LIT-abhishek2025-c-acc", "C-test ACC (yayımlanmış C npz)", "ResNet-18 28 px", 0.825, 0.006, "oran",
         "abhishek_2025_dermamnistkalite", "**0.825 ± 0.006 ACC / 0.946 AUC**"),
        ("LIT-wu2025-e2e-acc", "test ACC", "DINOv2 ViT-B/14 uçtan uca, 224 (hakemsiz)", 92.02, 0.32, "%",
         "wu_2025_foundationmedmnist", "**ACC 92.02 ± 0.32 / AUC 98.92 ± 0.11**"),
        ("LIT-wu2025-lp-acc", "test ACC", "DINOv2 ViT-B/14 doğrusal sonda, 224", 83.97, 0.08, "%",
         "wu_2025_foundationmedmnist", "**83.97 ± 0.08 ACC / 96.62 ± 0.02 AUC.**"),
        ("LIT-zhang2026-acc", "test ACC", "ImageNet ResNet-18, DermaMNIST-224 (3 seed)", 0.882, 0.002, "oran",
         "zhang_2026_weightedsampling", "| 0.882 ± 0.002 (AUC 0.973) |"),
        ("LIT-abhishek2025-sizinti-goruntu", "train–test sızan görüntü (Abhishek tanımı)", "—", 886, None, "görüntü",
         "abhishek_2025_dermamnistkalite", "train–test **886 görüntü / 641 lezyon**"),
    ]
    for id_, met, yon, deg, sd, bir, atif, desen in LIT:
        i, _ = satir_bul(rel, desen)
        ekle(id_, met, yon, "DermaMNIST (literatür)", deg, bir, rel, f"satır {i}", KIP_LIT, kaynak="literatur",
             atif=atif, std=sd, bib="notlar/references-cekirdek.bib",
             not_="Aynı koşulda üretilmedi; yalnız bağlam/replikasyon çıpası (sayısal yöntem kıyasına konmaz).")
    i, s = satir_bul("sonuclar/replikasyon.md", "kabul aralığı")
    ekle("REP-kabul-araligi", "replikasyon kabul aralığı (test ACC)", "B", VS_T, [0.7218, 0.754], "oran",
         "sonuclar/replikasyon.md", f"satır {i}", "replikasyon kapısı", tolerans=0.012,
         lit_bant=[0.7338, 0.742])


# ---------------------------------------------------------------- leaderboard (val, seçim döngüsü)
KARTLAR = [  # (tur, kart, varyant, karar, donanim, günlük satır deseni, kaynak)
    (0, "B", "baseline", "replikasyon (YEŞİL)", "GTX 1650", "| 0 | 2026-10-10 | — (replikasyon)", "lb"),
    (1, "H01", "h01_dinov2b_lp_r224", "TUT", "GTX 1650", "| 1 | 2026-10-10/11 | H01 |", "lb"),
    (1, "H02", "h02_r18in_ft224", "TUT", "GTX 1650", "| 1 | 2026-10-10/11 | H02 |", "lb"),
    (1, "H03-r28", "h03_dinov2b_lp_r28", "analiz", "GTX 1650", "| 1 | 2026-10-10 | H03 (analiz)", "lb"),
    (1, "H03-r64", "h03_dinov2b_lp_r64", "analiz", "GTX 1650", "| 1 | 2026-10-10 | H03 (analiz)", "lb"),
    (1, "H03-r128", "h03_dinov2b_lp_r128", "analiz", "GTX 1650", "| 1 | 2026-10-10 | H03 (analiz)", "lb"),
    (2, "H04", "h04_h02_logitayar", "TUT (dengeli ACC hedefli katman)", "GTX 1650 (CPU, post-hoc)",
     "| 2 | 2026-10-11 | H04 |", "lb"),
    (2, "H05", "h05_r18in_ft224_cw05", "ÖLDÜR", "GTX 1650", "| 2 | 2026-10-11 | H05 |", "lb"),
    (2, "H06", "h06_h01_logitayar", "analiz", "GTX 1650 (CPU, post-hoc)", "| 2 | 2026-10-11 | H06 (analiz)", "lb"),
    (3, "H07", "h07_h02_tta8", "ÖLDÜR", "GTX 1650", "| 3 | 2026-10-11 | H07 |", "lb"),
    (3, "H08", "h08_r18in_ft224_d4", "İNCELE → arıza yok → K5 doğrulama adayı", "GTX 1650",
     "| 3 | 2026-10-11 | H08 |", "lb"),
    (3, "H09", "h09_r18in_ft224_renk02", "ÖLDÜR", "GTX 1650", "| 3 | 2026-10-11 | H09 |", "lb"),
    (4, "H10", "h10", "TUT", "Colab L4", "| 4 | 2026-10-11 (COLAB) | H10 |", "kv"),
    (4, "H11", "h11", "TUT", "Colab L4", "| 4 | 2026-10-11 (COLAB) | H11 |", "kv"),
    (4, "H12", "h12", "analiz", "Colab L4", "| 4 | 2026-10-11 (COLAB) | H12 (analiz)", "kv"),
    (5, "H13", "h13_h02c_h10_lpo", "ÖLDÜR", "CPU (Colab L4 tahminleri üzerinde topluluk)",
     "| 5 | 2026-10-11 (CPU, val) | H13 |", "lb"),
]


def leaderboard():
    rel = "sonuclar/leaderboard.csv"
    rows = list(csv.DictReader((DW / rel).open(encoding="utf-8-sig")))
    kv = jload("sonuclar/k5_kuru_val_sonuc.json")
    g_rel = "gunluk/deney-gunlugu.md"
    for tur, kart, var, karar, donanim, desen, kaynak in KARTLAR:
        if kaynak == "lb":
            xs = [r for r in rows if r["varyant"] == var]
            seeds = [float(r["val_acc"]) for r in sorted(xs, key=lambda r: int(r["seed"]))]
            satirno = [i + 2 for i, r in enumerate(rows) if r["varyant"] == var]
            dosya, alan = rel, f"varyant={var}, satır {satirno[0]}–{satirno[-1]}, sütun val_acc"
        else:
            seeds = kv["kollar"][var]["seed"]["acc"]
            dosya, alan = "sonuclar/k5_kuru_val_sonuc.json", f"kollar.{var}.seed.acc"
        assert len(seeds) == 5, (kart, len(seeds))
        gi, gs = satir_bul(g_rel, desen)
        ekle(f"LB-{kart}", "val ACC (5 seed)", kart, VS_V, st.mean(seeds), "oran", dosya, alan, KIP_V,
             std=st.stdev(seeds), seed_degerleri=seeds, tur=tur, karar=karar, donanim=donanim,
             gunluk_satir=f"{g_rel} satır {gi}",
             tanim=(VAL_TANIM_YEREL if "GTX 1650" in donanim else
                    VAL_TANIM_L4 + " (Tur 4 seçim değeri = L4 değeri)" if kaynak == "kv" else
                    "Tur 5: L4 kollarının (H02c, H10) doğrulama tahminleri üzerinde CPU'da topluluk; L4 tanımına bağlı"))
        if kart.startswith("H03-") and kart != "H03-r28":
            continue
        h = [x.strip() for x in gs.split("|")]
        dlt, ga = delta_ga(h[7])
        pm = re.search(r"p(?:_t)?[ =]+(\d+[,.]\d+(?:e-?\d+)?)", h[7].replace("−", "-"))
        ekle(f"G-{kart.split('-')[0]}-delta", "Δ vs ebeveyn (günlük; val)", kart.split("-")[0], VS_V,
             dlt if dlt is not None else h[7], "puan" if dlt is not None else "metin", g_rel,
             f"satır {gi}, sütun 'Δ vs ebeveyn'", KIP_G, ga=ga,
             p=float(pm.group(1).replace(",", ".")) if pm else None, ham_metin=h[7], karar_metin=h[8],
             ana_metrik_metin=h[6], tur=tur,
             tum_delta_ga=[[tr_sayi(m.group(1)), tr_sayi(m.group(2)), tr_sayi(m.group(3))] for m in
                           re.finditer(rf"({SAYI})\s*\[({SAYI});\s*({SAYI})\]", h[7])] or None)


def butce():
    rel = "gunluk/deney-gunlugu.md"
    for kid, desen in [("BUT-tur4", "| Tur-4 sonu birleşik defter"), ("BUT-H12", "| H12 (sekme C"),
                       ("BUT-K5test", "| K5 test (sekme K5T"), ("BUT-C1", "| K5 C1 (C1A"), ("BUT-toplam", "| **Toplam**")]:
        i, s = satir_bul(rel, desen)
        h = [x.strip() for x in s.split("|")]
        sure = [tr_sayi(x) for x in re.findall(r"\d+,\d+", h[2])]
        duvar = [tr_sayi(x) for x in re.findall(r"\d+,\d+", h[3])]
        ekle(kid, "bütçe (koşum süresi toplamı / oturum duvar süresi)", h[1].replace("*", ""), "—",
             duvar[0], "saat (oturum)", rel, f"satır {i}", KIP_G, kosum_suresi_saat=sure[0] if sure else None,
             ust_sinir_saat=32 if "Toplam" in h[1] else None, ham_metin=s.strip())
    for kid, rel2, desen, rx in [
        ("BUT-tur1", "sonuclar/tahlil-tur1.md", "Leaderboard'daki kümülatif toplam", r"≈ (\d+)–(\d+,\d+) / 32"),
        ("BUT-tur2", "sonuclar/tahlil-tur2.md", "**Sayaç:** ≈ 7,5 + 2,3", r"≈ (\d+,\d+) / 32"),
        ("BUT-tur3", "sonuclar/tahlil-tur3.md", "**Tur-3 GPU:**", r"≈ (\d+,\d+)–(\d+,\d+) / 32")]:
        i, s = satir_bul(rel2, desen)
        m = re.search(rx, s)
        deg = tr_sayi(m.group(m.lastindex))
        ekle(kid, "kümülatif bütçe (GPU-sa, üst sınır 32)", kid[-4:], "—", deg, "GPU-saat", rel2, f"satır {i}",
             KIP_G, aralik=[tr_sayi(g) for g in m.groups()], ham_metin=m.group(0))
    i, s = satir_bul(rel, "- Tur sayısı / üst sınır: 5 / 8 (K5'e geçildi")
    ekle("BUT-tur-sayisi", "tur sayısı / üst sınır", "—", "—", 5, "tur", rel, f"satır {i}", KIP_G, ust_sinir=8)


# ---------------------------------------------------------------- başarısız ön-kayıtlar (A tablosu)
def basarisiz():
    rel = "sonuclar/basarisiz-onkayitlar.md"
    for kart in ["H05", "H07", "H09", "H13"]:
        i, s = satir_bul(rel, f"| **{kart}**")
        h = [x.strip() for x in s.split("|")]
        dlt, ga = delta_ga(h[5])
        pm = re.search(r"p_t\s*\*{0,2}(\d+,\d+)", h[5])
        ekle(f"BO-{kart}", "öldürülen kart ölçümü (val)", kart, VS_V, dlt, "puan", rel, f"satır {i}, sütun 'Ölçülen'",
             KIP_V, ga=ga, p_t=tr_sayi(pm.group(1)) if pm else None, olcum_metni=h[5], beklenti=h[3],
             olurme_kurali=h[4], tetiklenen=h[6], mekanizma=h[7], karar="ÖLDÜR")


# ---------------------------------------------------------------- hata kırılımı (val, §9)
def hata_kirilim():
    rel = "sonuclar/hata-kirilim-h10-karsilastirma.csv"
    rows = list(csv.reader((DW / rel).open(encoding="utf-8-sig")))
    bas = rows[0]
    kolmap = {"baseline_colab": "B", "h02_r18in_ft224_colab": "H02c", "h08_r18in_ft224_d4_colab": "H08c",
              "h10_dinov2b_k4_ft224": "H10"}
    alin = {("genel", "acc", "tum", ""), ("genel", "auc", "tum", ""), ("genel", "dengeli_acc", "tum", ""),
            ("genel", "ece15", "tum", ""), ("genel", "makro_f1", "tum", "")}
    for i, r in enumerate(rows[1:], start=2):
        anahtar = tuple(r[:4])
        sinif_duy = r[0] == "sinif" and r[1] == "duyarlilik"
        sn = r[0] == "sizinti" and r[1] == "acc"
        if not (anahtar in alin or sinif_duy or sn):
            continue
        for j, ad in enumerate(bas):
            if ad not in kolmap or not r[j]:
                continue
            m = re.match(r"([\d.]+) ± ([\d.]+)", r[j])
            if not m:
                continue
            ek = f"-{r[3]}" if r[3] else (f"-{r[2]}" if sn else "")
            ekle(f"HA-{kolmap[ad]}-{r[1]}{ek}", f"{r[1]} ({r[2]}{'/' + r[3] if r[3] else ''})", kolmap[ad], VS_V,
                 float(m.group(1)), "oran", rel, f"satır {i}, sütun {ad}", KIP_V, std=float(m.group(2)),
                 donanim="Colab L4")
    for kol, f in [("B", "hata_kirilim_ozet_baseline_colab.json"), ("H02c", "hata_kirilim_ozet_h02_r18in_ft224_colab.json"),
                   ("H08c", "hata_kirilim_ozet_h08_r18in_ft224_d4_colab.json"),
                   ("H10", "hata_kirilim_ozet_h10_dinov2b_k4_ft224.json")]:
        d = jload(f"sonuclar/{f}")
        z = d["zor"]
        ekle(f"HA-{kol}-hep_yanlis", "5 seed'in hepsinde yanlış görüntü sayısı", kol, VS_V,
             len(z["hep_yanlis_indeks"]), "görüntü", f"sonuclar/{f}", "zor.hep_yanlis_indeks (uzunluk)", KIP_V)
    j = jload("sonuclar/hata-kirilim-h10-zor-gecis.json")
    ekle("HA-zor-gecis", "baseline→H10 hep-yanlış geçişi (ham JSON)", "B → H10", VS_V,
         json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in j.items()}, ensure_ascii=False),
         "metin", "sonuclar/hata-kirilim-h10-zor-gecis.json", "tüm alanlar (liste uzunlukları)", KIP_V)


# ---------------------------------------------------------------- veri manifesti
def veri():
    rel = "data/data_manifest.md"
    V = [("DM-resmi-bolme", "resmî bölme train/val/test", [7007, 1003, 2005], "görüntü",
          "- Çoğunluk sınıfı (nv) payı: train %66,98, test %66,88 (1.341/2.005)"),
         ("DM-cogunluk-test", "resmî test çoğunluk (nv) tabanı", 0.669, "oran", "test ACC'si 0,669"),
         ("DM-S-test", "lezyonu train'de olan test görüntüsü (S)", 693, "görüntü",
          "- **Test görüntülerinin %34,56'sının (693/2.005)"),
         ("DM-S-test-oran", "S oranı (test)", 0.3456, "oran", "- **Test görüntülerinin %34,56'sının (693/2.005)"),
         ("DM-S-val", "lezyonu train'de olan val görüntüsü", 348, "görüntü", "Val için oran %34,70 (348/1.003)"),
         ("DM-lezyon-train-test", "train∩test ortak lezyon", 641, "lezyon", "| train∩test | 641 | 1.518 | test'te 693 |"),
         ("DM-C-bolme", "DermaMNIST-C train/val/test", [8215, 573, 1227], "görüntü", "| train | 8.215 | 8.215 ✓"),
         ("DM-cogunluk-C", "C-test çoğunluk (nv) tabanı", 0.7726, "oran", "ACC tabanı C'de **0,7726**"),
         ("DM-C-test-df", "C-test df sayısı", 8, "görüntü", "| 3 | df | 102 | 5 | 8 | 115 |"),
         ("DM-C-test-vasc", "C-test vasc sayısı", 17, "görüntü", "| 6 | vasc | 118 | 7 | 17 | 142 |"),
         ("DM-C-test-akiec", "C-test akiec sayısı", 35, "görüntü", "| 0 | akiec | 273 | 19 | 35 | 327 |"),
         ("DM-C-piksel-rmse", "C-28 ↔ resmî 28 px RMSE medyanı", 14.05, "0–255", "RMSE medyan 14,05")]
    for id_, met, deg, bir, desen in V:
        i, _ = satir_bul(rel, desen)
        ekle(id_, met, "—", "DermaMNIST", deg, bir, rel, f"satır {i}", KIP_VERI)
    test_n = {"akiec": 66, "bcc": 103, "bkl": 220, "df": 23, "mel": 223, "nv": 1341, "vasc": 29}
    for s, n in test_n.items():
        i, line = satir_bul(rel, f"({s}) |")
        h = [x.strip().replace(".", "") for x in line.split("|")]
        assert int(h[3]) == {"akiec": 228, "bcc": 359, "bkl": 769, "df": 80, "mel": 779, "nv": 4693, "vasc": 99}[s]
        assert int(h[5]) == n, (s, h)
        ekle(f"DM-test-{s}", f"resmî test {s} sayısı", "—", VS_T, int(h[5]), "görüntü", rel, f"satır {i}", KIP_VERI,
             train=int(h[3]), val=int(h[4]))


def nitel():
    """F5 nitel panel (Faz 1.5, val; betimsel): aday sayıları ve seçilen görüntü sayısı."""
    yol = WS / "figures" / "data" / "nitel_panel_val.json"
    if not yol.exists():
        print("nitel_panel_val.json yok — NIT kayıtları atlandı")
        return
    j = json.loads(yol.read_text(encoding="utf-8"))
    rel = "../../makale_workspace/dermamnist-dinov2-protokol/figures/data/nitel_panel_val.json"
    for kod, n in j["kategori_aday_sayisi"].items():
        ekle(f"NIT-aday-{kod}", "nitel panel kategori aday sayısı (val)", kod, VS_V, n, "görüntü", rel,
             f"kategori_aday_sayisi.{kod}", KIP_V, etiket=["betimsel, karar taşımaz"],
             aciklama=next(k["aciklama"] for k in j["secim_olcutu"]["kategoriler"] if k["kod"] == kod))
    ekle("NIT-secilen", "nitel panelde gösterilen görüntü sayısı", "—", VS_V, len(j["goruntuler"]), "görüntü", rel,
         "goruntuler (uzunluk)", KIP_V, betik_sha256=j["kaynak"]["betik_sha256"], secim_kurali=j["secim_olcutu"]["kural"])
    b = j.get("bilgi", {})
    if "ucunun_ortak_hep_yanlis_H02c_H08c_H10" in b:
        ekle("NIT-ortak-cekirdek", "H02c, H08c, H10'un ortak hep-yanlış çekirdeği (val)", "H02c ∩ H08c ∩ H10", VS_V,
             b["ucunun_ortak_hep_yanlis_H02c_H08c_H10"], "görüntü", rel,
             "bilgi.ucunun_ortak_hep_yanlis_H02c_H08c_H10", KIP_V)


# ---------------------------------------------------------------- Faz 5 tur-1 ekleri (review.md B4, 4.4, 4.17, 5.1, 9.1)
WS_REL = "../../makale_workspace/dermamnist-dinov2-protokol/"


def satir_bul_ws(rel, desen):
    """Makale WS dosyasında (notes/…) desenin ilk satırı; kaynak_dosya DW'ye göreli yazılır."""
    for i, s in enumerate((WS / rel).read_text(encoding="utf-8").splitlines(), start=1):
        if desen in s:
            return WS_REL + rel, i, s
    raise SystemExit(f"desen bulunamadı: {rel}: {desen!r}")


def _kosumlar(desen, n=5):
    d = sorted(p for p in RC.glob(desen) if p.is_dir())
    if len(d) != n:
        raise SystemExit(f"{desen}: {len(d)} koşum (beklenen {n})")
    return d


def _tek_deger(dizinler, alan):
    """5 koşumun metrics.json alanı; hepsi aynı olmalı (parametre sayısı / C*)."""
    deg = [json.loads((p / "metrics.json").read_text(encoding="utf-8"))[alan] for p in dizinler]
    if len(set(deg)) != 1:
        raise SystemExit(f"{alan} koşumlar arasında farklı: {deg}")
    return deg[0], [p.name for p in dizinler]


def tur1_ekleri():
    # --- (1) parametre sayıları (koşum metrics.json; 5 seed özdeş) — B4, 4.8, 4.17
    PRM = [("PRM-B", "B", "baseline_colab", "param_sayisi", "toplam (MedMNIST resmî ResNet-18, 3×3 kök, 7 sınıf)"),
           ("PRM-H01-sonda", "H01c", "h01_dinov2b_lp_r224_colab", "param_sayisi", "eğitilen sonda (768→7)"),
           ("PRM-H01-omurga", "H01c", "h01_dinov2b_lp_r224_colab", "param_omurga", "donuk DINOv2 ViT-B/14 omurga (başsız)"),
           ("PRM-H02", "H02c", "h02_r18in_ft224_colab", "param_sayisi", "toplam = eğitilen (torchvision ResNet-18, 7 sınıf başı)"),
           ("PRM-H08", "H08c", "h08_r18in_ft224_d4_colab", "param_sayisi", "toplam = eğitilen (H02c ile aynı ağ)"),
           ("PRM-H10-egitilebilir", "H10", "h10_dinov2b_k4_ft224", "param_egitilen", "eğitilen: son 4 blok + norm + baş"),
           ("PRM-H10-toplam", "H10", "h10_dinov2b_k4_ft224", "param_sayisi", "toplam: DINOv2 ViT-B/14 + 768→7 baş"),
           ("PRM-H11", "H11", "h11_dinov2s_ft224", "param_sayisi", "toplam = eğitilen (DINOv2 ViT-S/14 tam ince ayar)"),
           ("PRM-H12-sonda", "H12", "h12_dinov2s_lp_r224", "param_sayisi", "eğitilen sonda (384→7)"),
           ("PRM-H12-omurga", "H12", "h12_dinov2s_lp_r224", "param_omurga", "donuk DINOv2 ViT-S/14 omurga (başsız)")]
    for kid, kol, var, alan, tanim in PRM:
        dz = _kosumlar(f"*_{var}_s[0-4]")
        deg, adlar = _tek_deger(dz, alan)
        ekle(kid, f"parametre sayısı — {tanim}", kol, "—", deg, "parametre",
             f"runs_colab/{adlar[0]}/metrics.json", f"{alan} (5/5 koşumda özdeş: {', '.join(adlar)})",
             "koşum kaydı (K4 Colab eğitimi; test açılmadan)")
    if K_VAL("PRM-H10-toplam") != K_VAL("PRM-H01-omurga") + K_VAL("PRM-H01-sonda"):
        raise SystemExit("PRM sağlaması: H10 toplam ≠ ViT-B omurga + 768→7 baş")

    # --- (2) sonda C* (B4)
    d = jload("sonuclar/k5_dondurma.json")
    for kol in ["h01", "h12"]:
        dz = _kosumlar(f"*_{'h01_dinov2b_lp_r224_colab' if kol == 'h01' else 'h12_dinov2s_lp_r224'}_s[0-4]")
        c_kos, _ = _tek_deger(dz, "secilen_C")
        assert c_kos == d["sonda_C_yildiz"][kol], (kol, c_kos)
        ekle(f"SEL-{kol.upper()}-Cstar", "sonda L2 düzenleme sabiti C* (5 seed ortalamasının argmax'ı, val)",
             KOL_AD[kol], VS_V, d["sonda_C_yildiz"][kol], "—", "sonuclar/k5_dondurma.json",
             f"sonda_C_yildiz.{kol} (koşum metrics.json secilen_C ile aynı)", "dondurma (test öncesi, 2026-10-11 18:54:44)")
    c1 = jload("sonuclar/k5_c1_sonuc.json")["C1"]["kosumlar"]["h01"]
    kaynak = [json.loads((RC / r / "config.json").read_text(encoding="utf-8"))["k5"]["kaynak_run"] for r in c1]
    c_c1, _ = _tek_deger([RC / k for k in kaynak], "secilen_C")
    ekle("SEL-C1-H01-Cstar", "sonda C* (C1: C-val'de seçim)", "H01c", VS_C1, c_c1, "—",
         f"runs_colab/{kaynak[0]}/metrics.json", f"secilen_C (5/5 özdeş; C1 test koşumlarının k5.kaynak_run'ları: {', '.join(kaynak)})",
         "C1 eğitimi (C-test açılmadan)")

    # --- (3) protokol saatleri (B2, B4)
    i, _ = satir_bul("hipotezler/SIRALAMA.md", "## K5 ÖN-KAYDI — H08 TEST DOĞRULAMA KURALI")
    ekle("ON-tarih-H08kural", "H08 test doğrulama kuralının yazılışı", "H08", "—", "2026-10-11T09:42", "zaman",
         "hipotezler/SIRALAMA.md", f"satır {i}", "ön-kayıt (test öncesi)")
    i, _ = satir_bul("hipotezler/SIRALAMA.md", "## ÖN-KAYIT — COLAB TEKRAR KAPISI")
    ekle("ON-tarih-tekrarkapisi", "Colab tekrar kapısı ön-kaydı", "—", "—", "2026-10-11T11:39", "zaman",
         "hipotezler/SIRALAMA.md", f"satır {i}", "ön-kayıt (Colab ölçümlerinden önce)")
    i, _ = satir_bul("gunluk/deney-gunlugu.md", "**KULLANICI KARARI 10 (2026-10-11 18:14):**")
    ekle("ON-tarih-onkayit", "K5 ön-kaydının (K5-ONKAYIT.md) onayı", "—", "—", "2026-10-11T18:14", "zaman",
         "gunluk/deney-gunlugu.md", f"satır {i}", "ön-kayıt (test öncesi)")
    dsha = sha(DW / "sonuclar/k5_dondurma.json")
    assert dsha.startswith("c984fd0a")
    ekle("ON-tarih-dondurma", "dondurma kaydı (k5_dondurma.json) zamanı", "—", "—", d["zaman"], "zaman",
         "sonuclar/k5_dondurma.json", "zaman (+ .sha256 dosyası)", "dondurma (test öncesi)", sha256=dsha,
         onkayit_sha256=d["dosya_sha256"]["hipotezler/K5-ONKAYIT.md"])
    ekb_sha = sha(DW / "hipotezler/K5-ONKAYIT-b.md")
    dondurmada = d["dosya_sha256"].get("hipotezler/K5-ONKAYIT-b.md") == ekb_sha
    i, _ = satir_bul("gunluk/deney-gunlugu.md", "**ŞEF KARARI (test öncesi, 18:55):**")
    j, _ = satir_bul("hipotezler/K5-ONKAYIT-b.md", "`k5_dondurma.py` yeniden koşuldu (yeni SHA256)")
    ekle("ON-tarih-ekb", "ek b (K5-ONKAYIT-b.md; S/N kural nesnesi = ham Δ_S−Δ_N) karar kaydı", "—", "—",
         "2026-10-11T18:55", "zaman", "gunluk/deney-gunlugu.md", f"satır {i}", "ön-kayıt eki (test öncesi)",
         sha256=ekb_sha, dondurmada_sha_var=dondurmada,
         kanit_dondurma="sonuclar/k5_dondurma.json → dosya_sha256['hipotezler/K5-ONKAYIT-b.md']",
         kanit_metin=f"hipotezler/K5-ONKAYIT-b.md satır {j}: 'k5_dondurma.py yeniden koşuldu (yeni SHA256)'",
         uyari=("Günlükteki 18:55 kararın günlüğe yazılış etiketidir. Ek b'nin SHA256'sı 18:54:44 dondurmasının "
                "dosya_sha256 alanındadır ve ek b metni dondurmanın kendisinden sonra yeniden koşulduğunu söyler: "
                "ek b dondurmadan ÖNCE yazıldı ve dondurma onu KAPSAR. 'Dondurmadan bir dakika sonra' ifadesi "
                "kanıtla çelişir." if dondurmada else "ek b SHA'sı dondurmada YOK"))
    for kid, desen, ad in [("ON-ekb-val-ham", "- ham +5,13 [+1,76; +8,93]", "ham"),
                           ("ON-ekb-val-tab", "- tabaka-std +3,92 [−2,25; +10,16]", "tab")]:
        i, s = satir_bul("hipotezler/K5-ONKAYIT-b.md", desen)
        dl, g = delta_ga(s)
        kv = jload("sonuclar/k5_kuru_val_sonuc.json")["SN"]["h10-h02"]
        kd, kg = (kv["ham_dS_eksi_dN_puan"], kv["ham_ga_puan"]) if ad == "ham" else (kv["tab_dS_eksi_dN_puan"], kv["tab_ga_puan"])
        assert abs(kd - dl) < 0.006 and abs(kg[0] - g[0]) < 0.006 and abs(kg[1] - g[1]) < 0.006, (kid, kd, kg)
        ekle(kid, f"ek b yazılmadan önce görülen val Δ_S−Δ_N ({'ham' if ad == 'ham' else 'tabaka-std'}) — M1 beyanı",
             "H10 − H02c", VS_V, round(kd, 6), "puan", "sonuclar/k5_kuru_val_sonuc.json",
             f"SN.h10-h02.{'ham_dS_eksi_dN_puan' if ad == 'ham' else 'tab_dS_eksi_dN_puan'} "
             f"(beyan: hipotezler/K5-ONKAYIT-b.md satır {i})", KIP_V, ga=[round(x, 6) for x in kg])
    i, _ = satir_bul("hipotezler/K5-ONKAYIT.md", "**Güç beyanı:** GA yarı genişliği tahmini ±2,5–3,0")
    ekle("ON-KC-MDE", "önceden beyan edilen en küçük saptanabilir etki (KÇ güç beyanı)", "KÇ (K_H10 − K_H08)",
         "DermaMNIST resmî test → C1", 3.0, "puan (yaklaşık; GA yarı genişliği tahmini ±2,5–3,0)",
         "deney_workspace/3-1-dermamnist-dinov2/hipotezler/K5-ONKAYIT.md", f"satır {i} (§4.3 Güç beyanı)",
         "ön-kayıt (test öncesi)", not_="şef ekledi 2026-10-13; protokol beyanı, ölçüm değil (tur-1'de üreticiye alındı)",
         kapsam="yalnız KÇ kıyası; ana Δ'lar için değil (review 4.9)")
    t = jload("sonuclar/k5_test_sonuc.json")
    m = re.search(r"bootstrap ([\d.]+) RNG (\d+)", t["yontem"])
    B = int(m.group(1).replace(".", ""))
    ekle("ON-boot-B", "görüntü-bootstrap yeniden örnekleme sayısı B", "—", VS_T, B, "örneklem",
         "sonuclar/k5_test_sonuc.json", "yontem", KIP_T, rng=int(m.group(2)),
         p_tanimi="p = (1 + #{Δ* ≤ 0}) / (B + 1)  (k5_test.py:201-221)")

    # --- (4) H10 fp16 token önbelleği (4.4)
    for kid, desen, vs in [("FP16-H10", "*_h10_dinov2b_k4_ft224_s[0-4]", VS_V), ("FP16-H10-C1", "*_h10_c1_s[0-4]", VS_C1)]:
        dz = _kosumlar(desen)
        fark, adlar = _tek_deger(dz, "onbellek_tutarlilik_maks_fark")
        fp16, _ = _tek_deger(dz, "fp16_cls_maks_fark")
        blok, _ = _tek_deger(dz, "donuk_blok")
        log = (dz[0] / "stdout.log").read_text(encoding="utf-8", errors="replace")
        olcek = float(re.search(r"\|CLS\| maks ([\d.]+)", log).group(1))
        has = json.loads((dz[0] / "config.json").read_text(encoding="utf-8")).get("hassasiyet")
        ekle(kid, "fp16 token önbelleği tutarlılığı: maks |ΔCLS| (tam FP32 ağ ↔ önbellek; 8 val görüntüsü)", "H10", vs,
             fark, "CLS birimi", f"runs_colab/{adlar[0]}/metrics.json",
             f"onbellek_tutarlilik_maks_fark (5/5 özdeş); fp16_cls_maks_fark; donuk_blok; |CLS| maks stdout.log",
             "koşum kaydı (eğitim; test açılmadan)", fp16_yuvarlama_payi=fp16, cls_maks_abs=olcek, donuk_blok=blok,
             hassasiyet=has, not_=("Hesap FP32 (TF32 kapalı). Yalnız donuk ilk blokların çıkış belirteçleri fp16 "
                                   "saklanır; test çıkarımı aynı yolu izler (k5_cikarim.py:10-11, :224-228). "
                                   "Sondalar (H01c/H12) float32 öznitelik kullanır."))

    # --- (5) test içi lezyon çokluğu (9.1) — yalnız meta veri (lesion_id eşlemesi); tahmin dosyası açılmaz
    rel = "data/splits/dermamnist_lezyon_esleme.csv"
    rows = list(csv.DictReader((DW / rel).open(encoding="utf-8")))
    tr = {r["lesion_id"] for r in rows if r["split"] == "train"}
    test = [r for r in rows if r["split"] == "test"]
    relc = "data/splits/dermamnist_c_esleme.csv"
    ctest = [r for r in csv.DictReader((DW / relc).open(encoding="utf-8")) if r["split_C"] == "test"]
    kumeler = [("test", VS_T, test, rel, "split == test"),
               ("S", VS_S, [r for r in test if r["lesion_id"] in tr], rel, "split == test ∧ lesion_id ∈ train"),
               ("N", VS_N, [r for r in test if r["lesion_id"] not in tr], rel, "split == test ∧ lesion_id ∉ train"),
               ("Ctest", VS_C1, ctest, relc, "split_C == test")]
    for ad, vs, rr, rl, filtre in kumeler:
        say = {}
        for r in rr:
            say[r["lesion_id"]] = say.get(r["lesion_id"], 0) + 1
        coklu = {k: v for k, v in say.items() if v > 1}
        if ad == "S":
            assert len(say) == 641  # DM-lezyon-train-test ile aynı
        if ad in ("test", "Ctest"):
            assert len(rr) == (2005 if ad == "test" else 1227)
        dag = {}
        for v in say.values():
            dag[v] = dag.get(v, 0) + 1
        ekle(f"LES-{ad}-lezyon", "farklı lezyon sayısı", "—", vs, len(say), "lezyon", rl, f"{filtre}; lesion_id tekil sayımı",
             KIP_VERI, goruntu=len(rr), lezyon_basina_goruntu_dagilimi={str(k): v for k, v in sorted(dag.items())})
        ekle(f"LES-{ad}-coklu-lezyon", "birden çok görüntüsü olan lezyon sayısı", "—", vs, len(coklu), "lezyon", rl,
             f"{filtre}; lesion_id sayısı > 1", KIP_VERI, maks_goruntu=max(say.values()))
        ekle(f"LES-{ad}-coklu-goruntu", "birden çok görüntülü lezyonlara ait görüntü sayısı", "—", vs, sum(coklu.values()),
             "görüntü", rl, f"{filtre}; Σ görüntü (lezyon sayısı > 1)", KIP_VERI, payda=len(rr),
             oran=round(sum(coklu.values()) / len(rr), 6))

    # --- (6) D4 / renk oynaması eğitim kaybı (val dönemi, K4 yerel; B4 tartisma:9, basarisiz:14)
    rel = "sonuclar/tur3_h08h09_ek_olcum.json"
    j = jload(rel)
    for kid, anahtar, kol in [("EGK-H02", "h02_r18in_ft224", "H02"), ("EGK-H08", "h08_r18in_ft224_d4", "H08 (D4)"),
                              ("EGK-H09", "h09_r18in_ft224_renk02", "H09 (renk)")]:
        kos = j[anahtar]
        assert len(kos) == 5
        for ep, k_i in [(5, 1), (30, 3)]:
            seed = [r["tl_e1_5_10_30"][k_i] for r in kos]
            ekle(f"{kid}-ep{ep}", f"eğitim kaybı, epoch {ep} (5 seed ortalaması)", kol, "DermaMNIST resmî train (n=7007)",
                 round(st.mean(seed), 6), "nat (CE)", rel, f"{anahtar}[*].tl_e1_5_10_30[{k_i}]",
                 "val dönemi (K4 Tur 1–3, yerel GTX 1650; test açılmadan)", seed_degerleri=[round(x, 6) for x in seed],
                 kosumlar=[r["run"] for r in kos])
    i, _ = satir_bul("sonuclar/tahlil-tur3.md", "| Train kaybı ep 5 (s0) |")
    ekle("EGK-tahlil-satiri", "tahlil-tur3 ikincil tablo satırı (ep 5, s0): H02 0,012 · H08 0,257 (≈21×) · H09 0,031 (≈2,5×)",
         "H02/H08/H09", "DermaMNIST resmî train (n=7007)", "metin", "metin", "sonuclar/tahlil-tur3.md", f"satır {i}",
         "val dönemi (K4 Tur 3)")

    # --- (7) literatür (B4): metindeki sayılar; doğrulama citation-audit / literatür ekleri
    LIT2 = [  # (id, metrik, yöntem/koşul, değer, std, birim, atıf, kaynak dosya, desen, doğrulama deseni (citation-audit))
        ("LIT-doerrich2025-e2e224-acc", "test ACC", "ImageNet ResNet-18 uçtan uca, 224 px (3 seed)", 82.33, 0.36, "%",
         "doerrich_2025_medmnistplus", "notes/literature.md", "| L3 | ImageNet ResNet-18 E2E, 224 px: 82,33 ACC", "ACC 82,33±0,36"),
        ("LIT-abhishek2025-lezyon-toplam", "DermaMNIST toplam lezyon sayısı", "—", 7470, None, "lezyon",
         "abhishek_2025_dermamnistkalite", "notes/literature.md", "| L6 | DermaMNIST lezyonlarının %13,47'si", "7.470 / 1.006 (%13,47)"),
        ("LIT-abhishek2025-lezyon-coklu", "birden fazla bölmeye düşen lezyon", "—", 1006, None, "lezyon",
         "abhishek_2025_dermamnistkalite", "notes/literature.md", "| L6 | DermaMNIST lezyonlarının %13,47'si", "7.470 / 1.006 (%13,47)"),
        ("LIT-abhishek2025-lezyon-coklu-oran", "birden fazla bölmeye düşen lezyon oranı", "—", 13.47, None, "%",
         "abhishek_2025_dermamnistkalite", "notes/literature.md", "| L6 | DermaMNIST lezyonlarının %13,47'si", "7.470 / 1.006 (%13,47)"),
        ("LIT-yagis2021-sisme", "kesit düzeyinde bölmede test doğruluğu şişmesi (veri setine göre aralık)", "3B MRI, 2B CNN",
         [29, 55], None, "%", "yagis_2021_leakage", "notes/literature.md", "| L11 | Kesit düzeyinde bölme", "%30 / %29 / %48 / %55"),
        ("LIT-feldman2020-etkilenen-acc", "etkilenen test örneklerinde doğruluk (önce → sonra)", "CIFAR-100", [72.14, 45.38],
         None, "%", "feldman_2020_memorize", "notes/literature.md", "| L12 | Etkisi yüksek", "72,14 → 45,38 ve 76,06"),
        ("LIT-huix2024-dinov2-recall", "ISIC 2019 duyarlılık (recall)", "ince ayarlı DINOv2", 0.859, 0.007, "oran",
         "huix_2024_naturalfm", "notes/literature.md", "| L8 | ISIC 2019", "0,859±0,007"),
        ("LIT-huix2024-in1k-recall", "ISIC 2019 duyarlılık (recall)", "ImageNet-1k denetimli temel çizgi", 0.823, 0.008, "oran",
         "huix_2024_naturalfm", "notes/literature.md", "| L8 | ISIC 2019", "0,823±0,008"),
        ("LIT-tupper2024-busi", "döndürme augmentasyonu dengeli doğruluk kazancı (BUSI)", "ImageNet ResNet-18, 5×2 ÇD", 5.4,
         None, "puan", "tupper_2024_ultrasoundaug", "notes/literature.md", "| L10 | Döndürme augmentasyonu", "BUSI +5,4"),
        ("LIT-tupper2024-busbra", "döndürme augmentasyonu dengeli doğruluk kazancı (BUS-BRA patoloji)", "ImageNet ResNet-18, 5×2 ÇD",
         5.1, None, "puan", "tupper_2024_ultrasoundaug", "notes/literature.md", "| L10 | Döndürme augmentasyonu", "BUS-BRA patoloji +5,1"),
        ("LIT-ortizrequena2026-esli-oran", "lezyon eşli test görüntüsü oranı (HAM10000 kaynaklı test)", "ISIC 2018 60/20/20 yeniden bölme",
         39, None, "%", "ortizrequena_2026_skinfm", "notes/literature.md", "| L9 | ISIC 2018", "HAM10000 kaynaklı 1.991 test görüntüsüdür"),
        ("LIT-ortizrequena2026-esli", "lezyon eşli test görüntüsü", "ISIC 2018 60/20/20 yeniden bölme", 779, None, "görüntü",
         "ortizrequena_2026_skinfm", "notes/literature.md", "| L9 | ISIC 2018", "HAM10000 kaynaklı 1.991 test görüntüsüdür"),
        ("LIT-ortizrequena2026-payda", "HAM10000 kaynaklı test görüntüsü (payda)", "ISIC 2018 60/20/20 yeniden bölme", 1991, None,
         "görüntü", "ortizrequena_2026_skinfm", "notes/literature.md", "| L9 | ISIC 2018", "HAM10000 kaynaklı 1.991 test görüntüsüdür"),
        ("LIT-ortizrequena2026-bacc-fark", "eşsiz alt kümede temel model − CNN dengeli doğruluk farkı", "tek seed (42), 1.212 görüntü",
         -0.0055, None, "oran", "ortizrequena_2026_skinfm", "notes/literature.md", "| L9 | ISIC 2018", "−0,0055 [−0,0335; +0,0233]"),
    ]
    for kid, met, yon, deg, sd, bir, atif, rl, desen, dog in LIT2:
        dosya, i, _ = satir_bul_ws(rl, desen)
        _, ia, _ = satir_bul_ws("notes/citation-audit.md", dog)
        ek_ = {"ga": [-0.0335, 0.0233]} if kid.endswith("bacc-fark") else {}
        ekle(kid, met, yon, "literatür", deg, bir, dosya, f"satır {i}", KIP_LIT, kaynak="literatur", atif=atif, std=sd,
             dogrulama_durumu="dogrulandi", dogrulama=f"notes/citation-audit.md satır {ia}",
             not_="Aynı koşulda üretilmedi; yalnız bağlam (sayısal yöntem kıyasına konmaz).", **ek_)
    # Wu AUC (koordinatör, literatür ajanı tur-1) ve Medeiros TTA@50 (hakem doğrulaması)
    i, _ = satir_bul("notlar/literatur-cipa.md", "**ACC 92.02 ± 0.32 / AUC 98.92 ± 0.11**")
    ekle("LIT-wu2025-e2e-auc", "test AUC", "DINOv2 ViT-B/14 uçtan uca, 224 (hakemsiz)", "DermaMNIST (literatür)", 98.92, "%",
         "notlar/literatur-cipa.md", f"satır {i}", KIP_LIT, kaynak="literatur", atif="wu_2025_foundationmedmnist", std=0.11,
         dogrulama_durumu="dogrulandi", dogrulama="literatür ajanı tur-1 (koordinatör iletisi 2026-10-17)",
         not_="Aynı koşulda üretilmedi; yalnız bağlam.")
    dosya, i, _ = satir_bul_ws("notes/review.md", "- TTA@50 ile 0,758 (+1,6).")
    ekle("LIT-medeiros2026-tta50-acc", "test ACC (TTA@50)", "ResNet-18 28 px sıfırdan, TTA@50", "DermaMNIST (literatür)", 0.758,
         "oran", dosya, f"satır {i} (arXiv:2604.09697v1 Tablo 2)", KIP_LIT, kaynak="literatur", atif="medeiros_2026_tta",
         dogrulama_durumu="dogrulandi", dogrulama="hakem tur-1 (review.md §2.2)", not_="Aynı koşulda üretilmedi; yalnız bağlam.")


# ---------------------------------------------------------------- Faz 5 tur-1b: lezyon-kümeli bootstrap (review §9.1(a))
KIP_LKB = ("test sonrası duyarlılık analizi (kullanıcı onayı 2026-10-17; review §9.1(a)); dondurulmuş test/C-test "
           "tahminleri; hiçbir seçim/karar/ön-kayıt sonucu değişmez; yeni eğitim yok")


def lkb_kayitlari():
    rel = "evidence/lkb/lkb_bootstrap_sonuc.json"
    yol = WS / rel
    if not yol.exists():
        print("UYARI: lkb_bootstrap_sonuc.json yok — SENS-LKB-* kayıtları atlandı")
        return
    d = json.loads(yol.read_text(encoding="utf-8"))
    dosya = WS_REL + rel
    v = d["dogrulama_goruntu_duzeyi"]
    if v["uyusmaz"] != 0 or v["maks_mutlak_fark"] > 1e-9:
        raise SystemExit("LKB: görüntü-düzeyi yeniden üretim dondurulmuş değerlerle eşleşmiyor")
    ekle("SENS-LKB-dogrulama", "görüntü-düzeyi bootstrap yeniden üretimi (karşılaştırılan alan sayısı)",
         "k5_test.py fonksiyonları, dondurulmuş tahminler", VS_T, v["karsilastirilan_alan"], "alan", dosya,
         "dogrulama_goruntu_duzeyi", KIP_LKB, maks_mutlak_fark=v["maks_mutlak_fark"], uyusmaz=v["uyusmaz"],
         kapsam=v["kapsam"], tekil_kume_sinamasi=d["tekil_kume_sinamasi"])
    ekle("SENS-LKB-kume-test", "lezyon kümesi sayısı", "resmî test", VS_T, d["kume"]["tam_test"]["n_lezyon"], "lezyon",
         dosya, "kume.tam_test.n_lezyon", KIP_LKB, n_goruntu=d["kume"]["tam_test"]["n_goruntu"])
    ekle("SENS-LKB-kume-Ctest", "lezyon kümesi sayısı", "C-test", VS_C0, d["kume"]["c_test"]["n_lezyon"], "lezyon",
         dosya, "kume.c_test.n_lezyon", KIP_LKB, n_goruntu=d["kume"]["c_test"]["n_goruntu"])
    ekle("SENS-LKB-yontem", "lezyon-kümeli bootstrap yöntemi", "B = 2.000, rng(0)", VS_T, d["yontem"]["B"], "örnek",
         dosya, "yontem", KIP_LKB, aciklama=d["yontem"]["lezyon"], rng=d["yontem"]["rng"])
    VS = {"resmî test, S/N": None, "C0": VS_C0, "KÇ_0": VS_T, "KÇ": VS_C1}
    ref_index = {r["id"]: r for r in KAYIT}
    degisen = []
    for i, r in enumerate(d["ozet"]):
        ref = ref_index.get(r["ref_id"])
        if ref is None:
            raise SystemExit(f"LKB: referans kayıt yok: {r['ref_id']}")
        # görüntü-düzeyi GA ve nokta, defterdeki referans kayıtla aynı olmalı
        if abs(ref["deger"] - r["nokta_puan"]) > 1e-6 or any(abs(a - b) > 1e-6 for a, b in zip(ref["ga"], r["ga_goruntu_puan"])):
            raise SystemExit(f"LKB: {r['ref_id']} defter değeriyle uyuşmuyor")
        vs = ref["veri_seti"] if r["blok"] == "resmî test, S/N" else VS.get(r["blok"], VS_C1 if r["blok"].startswith("C1") else VS_T)
        ek = {k: r[k] for k in ("p_boot_goruntu", "p_boot_lezyon", "holm_boot_goruntu", "holm_boot_lezyon", "holm_t",
                                "anlamli_goruntu", "anlamli_lezyon", "kural_gecti_goruntu", "kural_gecti_lezyon") if r.get(k) is not None}
        uy = None
        if r["isaret_degisti"]:
            degisen.append(f"SENS-LKB-{r['anahtar']}")
            uy = (f"GA'nın sıfıra göre konumu değişti: görüntü '{r['etiket_goruntu']}' → lezyon '{r['etiket_lezyon']}'. "
                  "Duyarlılık bulgusu; karar/ön-kayıt sonucu değişmez.")
        ekle(f"SENS-LKB-{r['anahtar']}", "ΔACC / türev, lezyon-kümeli %95 GA", r["ad"], vs, r["nokta_puan"], "puan", dosya,
             f"ozet[{i}]", KIP_LKB, ga=r["ga_lezyon_puan"], ga_goruntu=r["ga_goruntu_puan"], ref_id=r["ref_id"],
             blok=r["blok"], genislik_orani=r["genislik_orani"], etiket_goruntu=r["etiket_goruntu"],
             etiket_lezyon=r["etiket_lezyon"], isaret_degisti=r["isaret_degisti"], uyari=uy, **ek)
    kb = d["karar_bayraklari_goruntu_lezyon"]
    ekle("SENS-LKB-ozet", "GA'sı sıfıra göre konum değiştiren satır sayısı", "lezyon- vs görüntü-düzeyi bootstrap",
         "DermaMNIST resmî test (n=2005) ve C-test (n=1227)",
         len(degisen), "satır", dosya, "ozet[*].isaret_degisti", KIP_LKB, toplam_satir=len(d["ozet"]), degisen=degisen,
         karar_bayraklari=kb,
         genislik_orani_min_medyan_maks=[min(r["genislik_orani"] for r in d["ozet"]),
                                         st.median(r["genislik_orani"] for r in d["ozet"]),
                                         max(r["genislik_orani"] for r in d["ozet"])],
         genislik_orani_formul="(lezyon GA üst − alt) / (görüntü GA üst − alt), 44 satır",
         anlamli_degisen=[r["anahtar"] for r in d["ozet"] if r.get("anlamli_goruntu") is not None
                          and r["anlamli_goruntu"] != r["anlamli_lezyon"]],
         kural_degisen=[r["anahtar"] for r in d["ozet"] if r.get("kural_gecti_goruntu") is not None
                        and r["kural_gecti_goruntu"] != r["kural_gecti_lezyon"]],
         holm_boot_degisen=[f"SENS-LKB-{r['anahtar']}" for r in d["ozet"] if r.get("holm_boot_goruntu") is not None
                            and r["holm_boot_goruntu"] != r["holm_boot_lezyon"]],
         daralan=[f"SENS-LKB-{r['anahtar']}" for r in d["ozet"] if r["genislik_orani"] < 1])

    # ---- Faz 5 tur-2 (review-tur2 K3): Holm bootstrap p değişiminin id'li özeti
    holm = [r for r in d["ozet"] if r.get("holm_boot_goruntu") is not None]
    holm_deg = [r for r in holm if r["holm_boot_goruntu"] != r["holm_boot_lezyon"]]
    # neden alanındaki çözünürlük tabanı iddiasını doğrula (D15.2): Aile 1 = 4/2001, Aile 2 = 2/2001
    for r in holm:
        if r not in holm_deg:
            taban = 4 / 2001 if r["blok"].endswith(("aile1", "aile_C1_1")) else 2 / 2001
            assert abs(r["holm_boot_goruntu"] - taban) < 1e-12, r["anahtar"]
    ekle("SENS-LKB-holm-ozet", "Holm düzeltmeli bootstrap p'si değişen Holm ailesi kaydı sayısı",
         "lezyon- vs görüntü-düzeyi bootstrap", "DermaMNIST resmî test (n=2005) ve C-test (n=1227)",
         len(holm_deg), "kayıt", dosya, "ozet[*].holm_boot_goruntu / holm_boot_lezyon", KIP_LKB,
         holm_kayit_sayisi=len(holm), ozdes_sayisi=len(holm) - len(holm_deg),
         holm_kayitlari=[f"SENS-LKB-{r['anahtar']}" for r in holm],
         ozdes_degerler={"aile1 (resmî test ve C1; 4 karşılaştırma)": sorted({r["holm_boot_goruntu"] for r in holm
                                                                             if r["blok"].endswith(("aile1", "aile_C1_1"))}),
                         "aile2 (resmî test ve C1; 2 karşılaştırma)": sorted({r["holm_boot_goruntu"] for r in holm
                                                                             if r["blok"].endswith(("aile2", "aile_C1_2"))
                                                                             and r not in holm_deg})},
         degisen=[{"id": f"SENS-LKB-{r['anahtar']}", "holm_boot_goruntu": r["holm_boot_goruntu"],
                   "holm_boot_lezyon": r["holm_boot_lezyon"], "holm_t": r["holm_t"]} for r in holm_deg],
         neden="Özdeş 11 kaydın p'si bootstrap çözünürlük tabanında (Holm çarpanı × 1/2001; D15.2): Aile 1'de "
               "4/2001, Aile 2'de 2/2001. Taban üstündeki tek kayıt C1 H10 − H02c.")

    # ---- Faz 5 tur-2 (review-tur2 K2): genişlik oranı uçları (daralan aralıklar dahil)
    sirali = sorted(d["ozet"], key=lambda r: r["genislik_orani"])
    dar = [r for r in sirali if r["genislik_orani"] < 1]
    ekle("SENS-LKB-daralan", "lezyon-kümeli GA'sı görüntü-düzeyi GA'dan dar olan aralık sayısı",
         "lezyon- vs görüntü-düzeyi bootstrap", "DermaMNIST resmî test (n=2005) ve C-test (n=1227)",
         len(dar), "aralık", dosya, "ozet[*].genislik_orani", KIP_LKB,
         toplam_aralik=len(d["ozet"]),
         daralan=[{"id": f"SENS-LKB-{r['anahtar']}", "genislik_orani": r["genislik_orani"],
                   "daralma_yuzde": 100 * (1 - r["genislik_orani"])} for r in dar],
         en_cok_daralan_id=f"SENS-LKB-{sirali[0]['anahtar']}", en_cok_daralan_oran=sirali[0]["genislik_orani"],
         en_cok_daralan_yuzde=100 * (1 - sirali[0]["genislik_orani"]),
         en_cok_genisleyen_id=f"SENS-LKB-{sirali[-1]['anahtar']}", en_cok_genisleyen_oran=sirali[-1]["genislik_orani"],
         medyan_oran=st.median(r["genislik_orani"] for r in d["ozet"]),
         daralan_bloklari=sorted({r["blok"] for r in dar}),
         not_="'Monte Carlo gürültüsü' nitelemesi "
              "ölçülmedi (B'yi değiştiren ek koşum test tahminlerine yeniden bakmayı gerektirir); yalnız büyüklük (<%1) raporlanır.")

    # ---- Faz 5 tur-2 (review-tur2 K1): tab_lkb kapsamı (44 aralık ↔ 40 tablo satırı)
    disi = [r for r in d["ozet"] if r["blok"] == "resmî test, betimsel"
            and any(k in r["anahtar"].split("-") for k in ("h11", "h12"))]
    ekle("SENS-LKB-tablo-kapsam", "Tablo tab:lkb'de gösterilen aralık sayısı", "lezyon- vs görüntü-düzeyi bootstrap",
         "DermaMNIST resmî test (n=2005) ve C-test (n=1227)", len(d["ozet"]) - len(disi), "aralık", dosya,
         "ozet[*].blok / anahtar", KIP_LKB, toplam_aralik=len(d["ozet"]),
         tablo_disi=[f"SENS-LKB-{r['anahtar']}" for r in disi], tablo_disi_sayisi=len(disi),
         tablo_disi_konum_degisen=sum(bool(r["isaret_degisti"]) for r in disi),
         tablo_disi_genislik_orani=[r["genislik_orani"] for r in disi],
         not_="Tablo dışı dört satır betimsel H11/H12 kıyaslarıdır (DINOv2-S omurga satırları); hiçbirinde GA'nın "
              "sıfıra göre konumu değişmez. Üretici build_tables.py bu listeyi buradan okur.")


# ---------------------------------------------------------------- Revizyon 2026-11-03 (dış eleştiri M3; elestiri-plani R3)
KIP_REV = ("test-sonrası analiz (revizyon 2026-11-03); dondurulmuş C1 C-test tahminleri; yeni eğitim yok; resmî test "
           "tahminleri açılmadı; hiçbir seçim/karar/ön-kayıt sonucu değişmez; Holm dışı, betimsel")
ETIKET_REV = ["test-sonrası analiz (revizyon 2026-11-03)"]


def c1mel_kayitlari():
    rel = "evidence/c1mel/c1_mel_sonuc.json"
    yol = WS / rel
    if not yol.exists():
        print("UYARI: c1_mel_sonuc.json yok — C1-REV-* kayıtları atlandı")
        return
    d = json.loads(yol.read_text(encoding="utf-8"))
    dosya = WS_REL + rel
    v = d["dogrulama"]
    if v["uyusmaz"] != 0 or v["maks_mutlak_fark"] > 1e-9:
        raise SystemExit("C1-REV: k5_c1_sonuc.json yeniden üretimi tutmadı")
    lz = d["lezyon_kumeli"]
    if lz["dogrulama"]["uyusmaz"] != 0 or lz["dogrulama"]["maks_mutlak_fark"] > 1e-9:
        raise SystemExit("C1-REV: LKB lezyon-kümeli yeniden üretimi tutmadı")
    ekle("C1-REV-dogrulama", "dondurulmuş C1 analizinin yeniden üretimi (karşılaştırılan alan sayısı)",
         "k5_test.py fonksiyonları (SHA256 13a71cf1…), dondurulmuş C1 C-test tahminleri", VS_C1,
         v["karsilastirilan_alan"], "alan", dosya, "dogrulama", KIP_REV, etiket=ETIKET_REV,
         maks_mutlak_fark=v["maks_mutlak_fark"], uyusmaz=v["uyusmaz"], kapsam=v["kapsam"],
         lezyon_kumeli_dogrulama=lz["dogrulama"], yontem_ayrinti=d["yontem"], betik=d["betik"])
    ekle("C1-REV-n-mel", "C-test'teki mel görüntü sayısı", "veri (C-test etiketleri)", VS_C1, d["n_mel"], "görüntü",
         dosya, "n_mel", KIP_REV, etiket=ETIKET_REV, n_mel_lezyon=d["n_mel_lezyon"], n_ctest=d["n_ctest"],
         sinif_n=d["sinif_n"],
         uyari=f"mel n = {d['n_mel']} ({d['n_mel_lezyon']} lezyon): tek sınıf duyarlılığının bootstrap GA'ları geniş "
               "(Δ GA genişliği ≈ 10–18 puan); ayırt edilemeyen fark 'eşitlik' diye okunmamalı.")
    ref_id = {"baseline": "C1-baseline-dengeli", "h01": "C1-h01-dengeli", "h02": "C1-h02-dengeli",
              "h08": "C1-h08-dengeli", "h10": "C1-h10-dengeli"}
    for k, r in d["kol"].items():
        m = r["duyarlilik"]["mel"]
        ekle(f"C1-REV-{k}-duyarlilik_mel", "mel duyarlılığı (5 seed ort.; std ddof=1)", KOL_AD[k], VS_C1, m["ort"], "oran",
             dosya, f"kol.{k}.duyarlilik.mel", KIP_REV, etiket=ETIKET_REV, yon="yuksek", std=m["std"], seed=m["seed"],
             n_sinif=d["n_mel"])
        dn = r["dengeli_acc"]
        ekle(f"C1-REV-{k}-dengeli_acc", "dengeli ACC (5 seed ort.; std ddof=1)", KOL_AD[k], VS_C1, dn["ort"], "oran",
             dosya, f"kol.{k}.dengeli_acc", KIP_REV, etiket=ETIKET_REV, yon="yuksek", std=dn["std"], seed=dn["seed"],
             ref_id=ref_id[k], not_="ortalama, dondurulmuş C1 kaydıyla (ref_id) özdeş; std bu revizyonda eklendi")
    AD2 = {"h10-h02": "H10 − H02c", "h10-h08": "H10 − H08c", "h08-h02": "H08c − H02c"}
    for c, e in d["mel_kiyas"].items():
        L = lz["cift"][c]["mel"]
        ekle(f"C1-REV-K-{c}-duyarlilik_mel", "Δ mel duyarlılığı (eşleştirilmiş görüntü-bootstrap)", AD2[c], VS_C1,
             round(100 * e["delta"], 6), "puan", dosya, f"mel_kiyas.{c}", KIP_REV, etiket=ETIKET_REV,
             ham_deger=e["delta"], ga=gapuan(e["ga"]), p_boot=e["p_boot"], ga_genislik_puan=round(100 * e["ga_genislik"], 6),
             konum=e["kural"].split(" (")[0], A=e["A"], B=e["B"], n_gecerli_ornek=e["n_gecerli_ornek"],
             fark_seed_puan=[round(100 * x, 6) for x in e["fark_seed"]], fark_seed_sd_puan=round(100 * e["fark_seed_sd"], 6),
             ga_lezyon=gapuan(L["ga"]), p_boot_lezyon=L["p_boot"], konum_lezyon=L["konum_lezyon"],
             isaret_degisti_lezyon=L["isaret_degisti"],
             uyari=f"mel n = {d['n_mel']}: GA geniş ({100 * e['ga_genislik']:.1f} puan); Holm dışı, test-sonrası.")
    for c, e in d["dengeli_yeniden_uretim"].items():
        kid = f"C1-K-{c}-dengeli_acc"
        ref = next((r for r in KAYIT if r["id"] == kid), None)
        if ref is None or abs(ref["deger"] - e["delta_puan"]) > 1e-5 or \
                any(abs(a - b) > 1e-5 for a, b in zip(ref["ga"], e["ga_puan"])):
            raise SystemExit(f"C1-REV: {kid} yeniden üretilmedi")
        L = lz["cift"][c]["dengeli_acc"]
        ekle(f"C1-REV-K-{c}-dengeli_acc-lezyon", "Δ dengeli ACC, lezyon-kümeli %95 GA", AD2[c], VS_C1,
             ref["deger"], "puan", dosya, f"lezyon_kumeli.cift.{c}.dengeli_acc", KIP_REV, etiket=ETIKET_REV,
             ga=gapuan(L["ga"]), ga_goruntu=ref["ga"], ref_id=kid, p_boot_lezyon=L["p_boot"],
             konum_goruntu=L["konum_goruntu"], konum_lezyon=L["konum_lezyon"], isaret_degisti=L["isaret_degisti"],
             yeniden_uretim_maks_fark_puan=max(abs(ref["deger"] - e["delta_puan"]),
                                                *[abs(a - b) for a, b in zip(ref["ga"], e["ga_puan"])]),
             not_="nokta ve görüntü-düzeyi GA dondurulmuş C1 kaydının (ref_id) aynısı ve yeniden üretildi; lezyon GA "
                  "LKB tam_lezyon.C1.kiyas ile özdeş")


# ---------------------------------------------------------------- Revizyon 2026-11-03: ön-kayıt sapma tablosu (eleştiri M4; R4)
KIP_SAP = "ön-kayıt ↔ gerçekleşen beyanı (revizyon 2026-11-03; kaynak belgelerden satır düzeyinde)"


def _zaman(metin):
    """'*2026-10-11 21:52*' / '(2026-10-11 ~23:10)' → '2026-10-11T21:52' (yoksa yalnız tarih)."""
    m = re.search(r"(20\d\d-\d\d-\d\d)[ T~]*(\d\d:\d\d)?", metin)
    if not m:
        raise SystemExit(f"zaman bulunamadı: {metin[:80]!r}")
    return m.group(1) + (f"T{m.group(2)}" if m.group(2) else "")


def sapma_kayitlari():
    def kay(rel, desen, ws=False, baslangic=1):
        if ws:
            dosya, i, s = satir_bul_ws(rel, desen)
        else:
            i, s = satir_bul(rel, desen, baslangic)
            dosya = rel
        return dosya, i, s

    # (1) H08 "görülmemiş" bayrağı → K5'e taşınma (yol A)
    d, i, s = kay("hipotezler/H08.md", "dengeli ACC Δ > +6,5")
    ekle("SAP-H08-esik", "H08 kartı 'görülmemiş kadar iyi' dengeli ACC Δ eşiği (aşılırsa İNCELE, TUT verilmez)", "H08",
         VS_V, 6.5, "puan", d, f"satır {i}", KIP_SAP, alinti="dengeli ACC Δ > +6,5 … → **İNCELE**, TUT verilmez",
         dosya_mtime="2026-10-11T09:22:33")
    d, i, s = kay("gunluk/deney-gunlugu.md", "| 3 | 2026-10-11 | H08 |")
    m = re.search(r"dengeli ({SAYI}) \[({SAYI}); ({SAYI})\]".format(SAYI=SAYI), s)
    ekle("SAP-H08-val-dengeli", "H08 − H02 dengeli ACC Δ, yerel doğrulama (bayrağı tetikleyen değer)", "H08 − H02",
         VS_V, tr_sayi(m.group(1)), "puan", d, f"satır {i}", KIP_SAP, ga=[tr_sayi(m.group(2)), tr_sayi(m.group(3))],
         not_="yerel GTX 1650 doğrulaması (seçim ortamı); Colab doğrulama değeri V-K-h08-h02-dengeli")
    d, i, s = kay("hipotezler/SIRALAMA.md", "## K5 ÖN-KAYDI — H08 TEST DOĞRULAMA KURALI")
    d2, i2, s2 = kay("orkestrasyon_durumu.md", "## Tur-3 KARAR")
    ekle("SAP-H08-yolA", "H08'in K5 test doğrulama adayı yapılması (yol A; kullanıcı kararı 5)", "H08", "—",
         _zaman(s), "zaman", d, f"satır {i}", KIP_SAP,
         kaynaklar=[{"dosya": d, "satir": i}, {"dosya": d2, "satir": i2},
                    {"dosya": "sonuclar/inceleme-h08.md", "mtime": "2026-10-11T09:34:13", "not": "İNCELE: arıza yok"}])
    d, i, s = kay("sonuclar/karar.md", "## 12. KULLANICI KARARI")
    j, sj = satir_bul("sonuclar/karar.md", "- **G2:** H08", i)
    ekle("SAP-karar12", "K5 karar kaydı (kullanıcı kararı 12): H08 bayrak teşhisi kabul; T4 yalnız ek; ek GPU yok",
         "—", "—", _zaman(s), "zaman", d, f"satır {i}–{i + 4}", KIP_SAP, g2_satiri=j)
    # (2) ek b — val sonrası yazıldı (ON-tarih-ekb, ON-ekb-val-* mevcut); T-SN2 iki ölçüde
    # (3) C1 kuyruğu: A → A-v2 → A-v3 (B, H08c/H10, bir kez)
    for kid, rel, desen in [("SAP-C1-A-a1", "sonuclar/beklenti-k5-c1-A-asama1.md", "*2026-"),
                            ("SAP-C1-Av2-a1", "sonuclar/beklenti-k5-c1-A-v2-asama1.md", "*2026-"),
                            ("SAP-C1-Av3-a1", "sonuclar/beklenti-k5-c1-A-v3-asama1.md", "*2026-"),
                            ("SAP-C1-Av3-a3", "sonuclar/beklenti-k5-c1-A-v3-asama3.md", "*2026-"),
                            ("SAP-C1-B-a1", "sonuclar/beklenti-k5-c1-B-asama1.md", "*2026-"),
                            ("SAP-C1-B-a3", "sonuclar/beklenti-k5-c1-B-asama3.md", "*2026-")]:
        d, i, s = kay(rel, desen)
        _, _, bas = kay(rel, "# Beklenti Kapısı")
        sonuc = "GEÇTİ" if "GEÇTİ" in bas else ("İHLAL" if "İHLAL" in bas else "?")
        ekle(kid, "C1 beklenti kapısı sonucu", rel.split("beklenti-")[1].replace(".md", ""), VS_C1, sonuc, "durum", d,
             f"satır {i}", KIP_SAP, zaman=_zaman(s), baslik=bas.strip())
    # koşum config'lerinden: hangi C1 ölçüm koşumu hangi kuyrukta (H10 → yalnız k5-c1-B)
    kuyruk_kol = {}
    for p in sorted(RC.glob("*_c1*")):
        if not (p / "config.json").exists():
            continue
        cf = json.loads((p / "config.json").read_text(encoding="utf-8"))
        if str(cf.get("kuyruk", "")).startswith("e1-"):
            continue  # E1/H14 (2026-11-07) ayrı deney; K5 C1 kuyruk kaydı değişmesin (E1-* kayıtlarına bakınız)
        if int(cf.get("asama", 0)) == 3:
            kuyruk_kol.setdefault(cf["varyant"], set()).add(cf["kuyruk"])
    kuyruk_kol = {k: sorted(v) for k, v in sorted(kuyruk_kol.items())}
    if kuyruk_kol.get("h10_c1") != ["k5-c1-B"] or kuyruk_kol.get("h10_c1-test") != ["k5-c1-B"]:
        raise SystemExit(f"SAP: H10 C1 kuyruğu beklenenden farklı: {kuyruk_kol}")
    ekle("SAP-C1-kuyruk-kol", "C1 ölçüm (aşama 3) koşumlarının kuyrukları — H10 C1 kuyruğu", "H10", VS_C1, "k5-c1-B",
         "kuyruk", "runs_colab/*_c1*/config.json", "kuyruk, varyant, asama", KIP_SAP, kuyruk_kol=kuyruk_kol,
         not_="A → A-v2 → A-v3 tekrarı baseline/H01c/H02c kuyruğundadır; H08c ve H10 C1 koşumları k5-c1-B'de bir kez koştu")
    d, i, s = kay("gunluk/deney-gunlugu.md", "| 2026-10-11 21:49 | C1-A dumanı")
    d2, i2, s2 = kay("gunluk/deney-gunlugu.md", "| 2026-10-11 22:56 | k5-c1-A-v2")
    ekle("SAP-C1-A-neden", "C1-A / C1-A-v2 durma nedenleri (arıza kayıtları)", "baseline, H01c, H02c", VS_C1, 2, "olay", d,
         f"satır {i}, {i2}", KIP_SAP, olay=[s.strip()[:400], s2.strip()[:400]],
         uygulama_notu="hipotezler/K5-C1-UYGULAMA-NOTU.md (mtime 2026-10-11T23:04:58)")
    # (4) C2 atlandı
    d, i, s = kay("hipotezler/K5-ONKAYIT.md", "**Bütçe kapısı (ön-kayıt; sonuca bakmaz):**")
    ekle("SAP-C2-kapi", "C2 bütçe kapısı: C1 sonunda birleşik defter üst sınırı", "C2", "—", 28.8, "saat", d,
         f"satır {i}", KIP_SAP, alinti="C2, C1 bittiğinde birleşik defter **≤ 28,8** ise başlar")
    d, i, s = kay("gunluk/deney-gunlugu.md", "**KULLANICI KARARI 11")
    ekle("SAP-C2-karar", "C2'nin atlanması (kullanıcı kararı 11)", "C2", "—", _zaman(s), "zaman", d, f"satır {i}",
         KIP_SAP, alinti=s.strip()[:300],
         uyari="Kaynakta kararın gerekçesi yazılı değil ('sınırlılık'); bütçe kapısı işletilmedi.")
    # (5) T4
    d, i, s = kay("hipotezler/K5-ONKAYIT.md", "| (8') | T4 koşullu")
    ekle("SAP-T4-plan", "T4 ön-kaydı: her durumda ek tabloda; iddia yalnız H08 doğrulanırsa; kazanan olamaz", "T4", "—",
         "ek satır", "durum", d, f"satır {i}", KIP_SAP)
    d, i, s = kay("sonuclar/karar.md", "### 2.2 T4 AUC")
    ekle("SAP-T4-auc-denetim", "T4 AUC > 0,99 denetimi (ön-kayıtta T4 için tanımlı değil) yapıldı mı", "T4", VS_T,
         False, "bool", d, f"satır {i}", KIP_SAP, karar_satiri="sonuclar/karar.md §12")
    # (6) LKB — SENS-LKB-* kayıtları mevcut; karar zamanı
    d, i, s = kay("notes/orkestrasyon_durumu.md", "KULLANICI KARARI 2026-10-17: (9.1)", ws=True)
    ekle("SAP-LKB-karar", "lezyon-kümeli bootstrap için kullanıcı kararı (test sonrası duyarlılık)", "—", "—",
         _zaman(s), "zaman", d, f"satır {i}", KIP_SAP)
    d, i, s = kay("hipotezler/K5-ONKAYIT.md", "Test sonrası yapılan her ek analiz")
    ekle("SAP-ON-6-3", "ön-kayıt §6.3: test sonrası ek analiz 'test sonrası keşifsel' etiketiyle ve ayrı bölümde; "
         "yeni kol/seed/alt küme kuralı yok", "—", "—", "§6.3", "madde", d, f"satır {i}", KIP_SAP)
    # (7) H10 + D4 birleşim kartı (koşullu plan)
    d, i, s = kay("hipotezler/SIRALAMA.md", "**H10/H11 + D4 (H08 tarifi):**")
    ekle("SAP-H10D4-plan", "H10/H11 + D4 birleşim kartı koşulu (H08 K5'te doğrulanır ve DINOv2 kolu TUT alırsa)",
         "H10 + D4", "—", "koşullu kart", "durum", d, f"satır {i}–{i + 1}", KIP_SAP,
         alinti="yalnız H08 K5'te doğrulanır ve DINOv2 kolu TUT alırsa birleşim kartı. H10'da önbellek tek görünüm (8 görünüm ≈ 25 GB)")
    d, i, s = kay("gunluk/deney-gunlugu.md", "**KULLANICI KARARI 10")
    ekle("SAP-karar10", "kullanıcı kararı 10: Tur-6..8 yapılmaz, Tur-5 sonrası K5", "—", "—", _zaman(s), "zaman", d,
         f"satır {i}", KIP_SAP)
    # (8) C1 dengeli (ön-kayıtlı yön) + mel (test-sonrası)
    d, i, s = kay("hipotezler/K5-ONKAYIT.md", "**Dengeli ACC C1'in birincil ikincilidir.**")
    ekle("SAP-C1-dengeli-onkayit", "ön-kayıt: C1'de dengeli ACC GA'larıyla raporlanır; yön H10−H02>0, H08−H02>0; "
         "karar vermez", "—", VS_C1, "yön ön-kaydı", "madde", d, f"satır {i}", KIP_SAP)
    # (9) E1 / H14 — beklemede
    d, i, s = kay("notes/elestiri-plani.md", "Yeni deney: **yalnız E1**", ws=True)
    d2, i2, s2 = kay("orkestrasyon_durumu.md", "REVİZYON (2026-11-03)")
    ekle("SAP-E1", "E1 = H10 + D4, C1, test-sonrası keşifsel (H14 ön-kayıt kartı koşumdan önce)", "H10 + D4 (H14)",
         VS_C1, "beklemede", "durum", d, f"satır {i}", KIP_SAP, zaman="2026-11-03",
         kaynaklar=[{"dosya": d, "satir": i}, {"dosya": d2, "satir": i2}],
         not_="sonuç geldiğinde bu kayıt ve tab:sapma satırı güncellenir")


# ---------------------------------------------------------------- Revizyon E1 (2026-11-07): H14 = H10 + eğitim-zamanı D4, C1
KIP_E1 = ("test sonrası keşifsel (revizyon E1, 2026-11-07); H14 C1'de yeniden eğitildi, C-test H14 için seed başına bir "
          "kez açıldı; arşiv kolları (H10, H08c, H02c) dondurulmuş C1 C-test tahminleri; düzeltmesiz, K5 Holm "
          "ailelerine girmez; hiçbir ön-kayıtlı karar değişmez")
ETIKET_E1 = ["test sonrası keşifsel, düzeltmesiz (E1/H14)"]
E1_ORTAM = "evidence/e1/ortam-colab-E1-20261107-155457-57ae.json"  # Drive yz/3-1/env kopyası (sha256 kayıtta)


def e1_kayitlari():
    rel = "sonuclar/e1_h14_sonuc.json"
    if not (DW / rel).exists():
        print("UYARI: e1_h14_sonuc.json yok — E1-* kayıtları atlandı")
        return
    d = jload(rel)
    # --- bütünlük: .sha256 eşleşmeli, kuru özdeşlik geçmiş olmalı, sonuç yeniden yazılmamış olmalı
    beyan = (DW / (rel + ".sha256")).read_text(encoding="utf-8").split()[0]
    if beyan != sha(DW / rel):
        raise SystemExit("E1: e1_h14_sonuc.json SHA256 .sha256 beyanıyla uyuşmuyor")
    if d["kuru"] or not d["kuru_ozdeslik"]["gecti"] or d["yeniden_neden"] is not None:
        raise SystemExit("E1: kuru/özdeşlik/yeniden durumu beklenen değil")
    if d["kart_sha256"] != sha(DW / "hipotezler/H14.md") or d["betik_sha256"] != sha(DW / "scripts/e1_h14_analiz.py"):
        raise SystemExit("E1: kart/betik SHA256 sonuç dosyasındakiyle uyuşmuyor")
    ix = {r["id"]: r for r in KAYIT}

    # (1) kuru koşu özdeşliği (dondurulmuş k5_c1 satırlarının yeniden üretimi)
    oz = d["kuru_ozdeslik"]
    maks = max([v["maks_fark_puan"] for k, v in oz["satirlar"].items() if "maks_fark_puan" in v] +
               [abs(v["fark"]) for k, v in oz["satirlar"].items() if "fark" in v])
    ekle("E1-dogrulama", "E1 analiz betiğiyle dondurulmuş C1 sonuçlarının yeniden üretimi (karşılaştırılan satır)",
         "e1_h14_analiz.py --kuru (k5_test.py işlevleri içe aktarılır)", VS_C1, len(oz["satirlar"]), "satır", rel,
         "kuru_ozdeslik", KIP_E1, etiket=ETIKET_E1, maks_fark_puan=maks, tolerans_puan=oz["tolerans_puan"],
         gecti=oz["gecti"], satirlar=sorted(oz["satirlar"]), karsi_dosya="sonuclar/k5_c1_sonuc.json",
         betik_sha256=d["betik_sha256"], kart_sha256=d["kart_sha256"], k5_test_sha256=d["k5_test_sha256"],
         analiz_tarihi=d["tarih"], yontem_ayrinti=d["yontem"])

    # (2) C-test açılışı: H14 için seed başına bir kez; kural (i) veri_c = 1
    test_kos = sorted(p.name for p in RC.glob("*_h14_c1-test_s*"))
    beyan_kos = sorted(d["kosumlar"]["H14"].values())
    if test_kos != beyan_kos:
        raise SystemExit(f"E1: runs_colab h14 C-test koşumları sonuç dosyasıyla uyuşmuyor: {test_kos}")
    ms = [json.loads((DW / "runs_colab" / k / "metrics.json").read_text(encoding="utf-8")) for k in beyan_kos]
    if any(m["veri_c"] != 1 or m["test_n"] != 1227 for m in ms):
        raise SystemExit("E1: kural (i) — veri_c ≠ 1 ya da test_n ≠ 1227")
    ekle("E1-ctest-acilis", "H14 C-test çıkarım koşumu sayısı (seed başına bir; ikinci açılış yok)", "H14", VS_C1,
         len(test_kos), "koşum", "runs_colab/*_h14_c1-test_s*/metrics.json", "veri_c, test_n; e1_h14_sonuc.json "
         "kosumlar.H14, yeniden_neden", KIP_E1, etiket=ETIKET_E1, kosumlar=beyan_kos, n_seed=d["n_seed"],
         yeniden_neden=d["yeniden_neden"], veri_c=[m["veri_c"] for m in ms],
         not_="kart kural (i): C1 eğitimli modelin resmî test/val çıkarımı yok (veri_c = 1); yeniden açılış yok")

    # (3) kural (f): kontrol kolu Durum A → K1 karşılaştırıcısı arşiv H10 (H10r gerekmedi)
    du = d["durum"]
    a = du["ayrinti"]
    i_od, s_od = satir_bul("orkestrasyon_durumu.md", "E1 BİTTİ")
    ekle("E1-kontrol-durum", "kod yolu kontrolü (kart kural f): kontrol kolu (8 görünüm yolu, D4 kapalı) ↔ Ekim "
         "h10_c1 dumanı", "h14_c1_d4kapali_duman ↔ h10_c1_duman", "C-val ∩ resmî val (n=573)", du["durum"], "durum",
         rel, "durum", KIP_E1, etiket=ETIKET_E1, k1_karsilastirici=du["k1_karsilastirici"], h10r_gerekli=False,
         ilk_adim_loss_fark=a["kontrol_ilk_adim_loss_fark"], argmax_oran=a["kontrol_argmax_oran"],
         p_maks_fark=a["kontrol_p_maks_fark"], val_acc_fark=a["kontrol_val_acc_fark"],
         surum_fark=a["kontrol_surum_fark"], kontrol_run=a["kontrol_run"],
         kural_A="|Δ ilk_adim_loss| ≤ 1e-6 ∧ argmax 573/573 ∧ maks |Δp| ≤ 1e-5 (hipotezler/H14.md satır 48)",
         kaynaklar=[{"dosya": rel, "alan": "durum"}, {"dosya": "orkestrasyon_durumu.md", "satir": i_od}])

    # (4) ortam: koşum config.json'ları (14/14) + Colab oturum ortam kaydı
    h14_cf = {p.name: json.loads((p / "config.json").read_text(encoding="utf-8"))["donanim"]
              for p in sorted(RC.glob("*_h14_c1*"))}
    ekim = json.loads((DW / "runs_colab/20261011-225511_h10_c1_s0/config.json").read_text(encoding="utf-8"))["donanim"]
    torch_s = sorted({c["torch"] for c in h14_cf.values()})
    cudnn_s = sorted({c["cudnn"] for c in h14_cf.values()})
    if len(torch_s) != 1 or len(cudnn_s) != 1:
        raise SystemExit(f"E1: koşumlar arası sürüm farkı {torch_s} {cudnn_s}")
    if f"{ekim['cudnn']} → {cudnn_s[0]}" not in a["kontrol_surum_fark"]:
        raise SystemExit("E1: cuDNN farkı analiz JSON'uyla uyuşmuyor")
    oy = WS / E1_ORTAM
    oc = json.loads(oy.read_text(encoding="utf-8"))
    if oc["torch"] != torch_s[0] or oc["cudnn"] != cudnn_s[0] or oc["oturum"] not in oy.name:
        raise SystemExit("E1: Colab ortam kaydı koşum config'leriyle uyuşmuyor")
    ekle("E1-ortam-cudnn", "E1 koşumlarının cuDNN sürümü (Ekim C1 ölçümleri: karsi_ekim)", "H14 (14 koşum)", "—",
         cudnn_s[0], "sürüm", "runs_colab/*_h14_c1*/config.json", "donanim.cudnn", KIP_E1, etiket=ETIKET_E1,
         karsi_ekim=ekim["cudnn"], karsi_dosya="runs_colab/20261011-225511_h10_c1_s0/config.json",
         n_kosum=len(h14_cf), analiz_beyani=a["kontrol_surum_fark"],
         not_="Kontrol kolu bu farka rağmen Ekim dumanını bit düzeyinde üretti (E1-kontrol-durum).")
    diger = {k: [ekim[k], h14_cf[next(iter(h14_cf))][k]] for k in ("python", "numpy")
             if ekim[k] != h14_cf[next(iter(h14_cf))][k]}
    ekle("E1-ortam-torch", "E1 koşumlarının torch sürümü", "H14 (14 koşum)", "—", torch_s[0], "sürüm",
         "runs_colab/*_h14_c1*/config.json", "donanim.torch", KIP_E1, etiket=ETIKET_E1, karsi_ekim=ekim["torch"],
         ekimle_ayni=torch_s[0] == ekim["torch"], timm=sorted({c["timm"] for c in h14_cf.values()}),
         cuda=sorted({c["cuda"] for c in h14_cf.values()}), diger_farklar_ekim_e1=diger,
         colab_varsayilan_torch=oc["torch_sabitleme"]["colab_torch"], torch_sabitleme=oc["torch_sabitleme"],
         ortam_kaydi=WS_REL + E1_ORTAM, ortam_kaydi_sha256=sha(oy),
         ortam_kaydi_kaynak="<DRIVE_KOK>/3-1/env/ortam-colab-E1-20261107-155457-57ae.json (kopya)",
         uyari=(f"orkestrasyon_durumu.md satır {i_od} 'torch 2.14.1 — sabitleme hedefi 2.11 tutmadı' der; koşum "
                f"kayıtları bununla ÇELİŞİR: 2.14.1 Colab varsayılanıdır, sabitleme başarılı oldu (rc 0) ve 14 koşumun "
                f"hepsi {torch_s[0]} ile koştu. Makalede 'torch 2.14.1' yazılmaz; Ekim'den tek sürüm farkı cuDNN'dir "
                f"(+ python/numpy yama sürümleri)."))

    # (5) bütçe
    i_b, s_b = satir_bul("hipotezler/H14.md", "| Bütçe |")
    if "kart tavanı 5,0 L4-sa" not in s_b or "Tahmin ≈ 1,6 L4-sa" not in s_b:
        raise SystemExit("E1: H14.md bütçe satırı beklenen ifadeleri taşımıyor")
    ekle("E1-butce", "E1 birleşik koşum süresi (kart kural h)", "H14", "—", d["notlar"]["butce_E1_L4sa"], "L4-sa", rel,
         "notlar.butce_E1_L4sa", KIP_E1, etiket=ETIKET_E1, kart_tavani=5.0, kart_tahmini=1.6,
         kart_kaynak=f"hipotezler/H14.md satır {i_b}", oturum_kaynak=f"orkestrasyon_durumu.md satır {i_od}",
         not_="H10r koşulmadı (Durum A); C1 sonu birleşik deftere (BUT-toplam) dahil değildir")

    # (6) H14 mutlak C-test değerleri (5 seed ort.; std = analiz betiğinin kol_ozet'i)
    h = d["kollar"]["H14"]
    seed_alan = {"acc": "test_acc", "auc": "test_auc", "dengeli_acc": "test_dengeli_acc", "nll": "test_nll"}
    for m in ["acc", "auc", "dengeli_acc", "dengeli_acc_5sinif", "duyarlilik_mel", "makro_f1", "ece15", "nll"]:
        ek = {}
        if m in seed_alan:
            sd = [x[seed_alan[m]] for x in ms]
            if abs(st.mean(sd) - h["ort"][m]) > 1e-9:
                raise SystemExit(f"E1: H14 {m} seed ortalaması metrics.json ile uyuşmuyor")
            ek = {"seed_degerleri": sd, "seed_kaynak": [f"runs_colab/{k}/metrics.json:{seed_alan[m]}" for k in beyan_kos],
                  "std_ddof1_metrics": st.stdev(sd)}
        if m == "acc" and abs(st.stdev([x["test_acc"] for x in ms]) - h["std"]["acc"]) > 1e-9:
            raise SystemExit("E1: H14 ACC std (ddof=1) uyuşmuyor")
        ekle(f"E1-H14-{m}", f"{m} (5 seed ort.)", "H14", VS_C1, h["ort"][m], "oran", rel, f"kollar.H14.ort.{m}", KIP_E1,
             etiket=ETIKET_E1, yon=YON.get(m, "yuksek" if m.startswith("duyarlilik") else None), std=h["std"][m],
             std_alan=f"kollar.H14.std.{m}", n_seed=d["n_seed"], **ek)
    ekle("E1-H14-accmaj", "ACC − çoğunluk tabanı", "H14", VS_C1, h["acc_eksi_cogunluk"], "oran", rel,
         "kollar.H14.acc_eksi_cogunluk", KIP_E1, etiket=ETIKET_E1)

    # (7) kıyaslar (Δ = B − A; A = arşiv kolu, B = H14), görüntü-bootstrap B = 2000, rng 0, düzeltmesiz
    ARSIV = {"H10": "h10", "H08c": "h08", "H02c": "h02"}
    REF = {"acc": "C1-{k}-acc", "dengeli_acc": "C1-{k}-dengeli", "dengeli_acc_5sinif": "C1-{k}-dengeli5",
           "mel_duyarlilik": "C1-REV-{k}-duyarlilik_mel"}
    KIY = [("K1: H14 − H10", "E1-K1-h14-h10", "H10", "K1 (kart birincil)"),
           ("K2: H14 − H08c", "E1-K2-h14-h08", "H08c", "K2 (kart birincil)"),
           ("betimsel: H14 − H02c", "E1-BET-h14-h02", "H02c", "betimsel")]
    for anahtar, onek, kol, rol in KIY:
        v = d["kiyaslar"][anahtar]
        if v["n_seed"] != 5:
            raise SystemExit(f"E1: {anahtar} n_seed {v['n_seed']}")
        k = ARSIV[kol]
        for m in ["acc", "dengeli_acc", "dengeli_acc_5sinif", "mel_duyarlilik"]:
            x = v[m]
            rid = REF[m].format(k=k)
            if abs(ix[rid]["deger"] - x["A"]) > 1e-9:
                raise SystemExit(f"E1: {anahtar} {m} A ({x['A']}) ↔ {rid} ({ix[rid]['deger']})")
            if abs(100 * (x["B"] - x["A"]) - x["delta_puan"]) > 1e-9:
                raise SystemExit(f"E1: {anahtar} {m} Δ = B − A değil")
            konum = x["etiket"].split(" (")[0]
            g = x["ga_puan"]
            beklenen = "H14 üstün" if g[0] > 0 else ("H14 geride" if g[1] < 0 else "ayırt edilemedi")
            if konum != beklenen:
                raise SystemExit(f"E1: {anahtar} {m} etiket ↔ GA tutarsız")
            mid = "duyarlilik_mel" if m == "mel_duyarlilik" else m
            ekle(f"{onek}-{mid}", f"Δ{mid} (eşleştirilmiş görüntü-bootstrap)", f"H14 − {kol}", VS_C1,
                 round(x["delta_puan"], 6), "puan", rel, f"kiyaslar.{anahtar}.{m}", KIP_E1,
                 etiket=ETIKET_E1 + [rol], ga=[round(g[0], 6), round(g[1], 6)], konum=konum, A=x["A"], B=x["B"],
                 A_ref_id=rid, p_ust=x.get("p_ust"), p_alt=x.get("p_alt"), n_mel=x.get("n_mel"),
                 farkli_dogru_goruntu=v["farkli_dogru_goruntu"] if m == "acc" else None,
                 not_=("p_ust = (1+#{Δ*≤0})/(B+1), p_alt tersi; düzeltmesiz, bilgi" if "p_ust" in x else None))
        for t, mm in [("t_acc", "ACC"), ("t_dengeli", "dengeli ACC")]:
            x = v[t]
            ekle(f"{onek}-{t}", f"seed-eşli t, {mm} (iki yönlü; bilgi)", f"H14 − {kol}", VS_C1,
                 round(x["ort_puan"], 6), "puan", rel, f"kiyaslar.{anahtar}.{t}", KIP_E1, etiket=ETIKET_E1 + [rol],
                 p_iki_yonlu=x["p_iki_yonlu_bilgi"], fark_seed_puan=[round(f, 6) for f in x["fark_seed_puan"]],
                 n_seed=len(x["fark_seed_puan"]))

    # (8) ön-kayıtlı beklentiler (kart satır 45) ↔ gerçekleşen; mekanizma devrede mi (beklenti-ihlali G2)
    i_e, s_e = satir_bul("hipotezler/H14.md", "| Beklenen etki (yön + büyüklük) |")
    ARALIK = [  # (ad, kartta geçen ifade, alt, üst, gerçekleşen id)
        ("K1 ΔACC", "K1 H14 − H10 (C-test ACC): −0,5 … +2,0 puan", -0.5, 2.0, "E1-K1-h14-h10-acc"),
        ("H14 C-test ACC", "Mutlak H14 C-test ACC **[0,911; 0,936]**", 0.911, 0.936, "E1-H14-acc"),
        ("K2 ΔACC", "K2 H14 − H08c: −1,5 … +1,0", -1.5, 1.0, "E1-K2-h14-h08-acc"),
        ("K1 Δdengeli", "dengeli ACC Δ(H14 − H10) [−2; +8]", -2.0, 8.0, "E1-K1-h14-h10-dengeli_acc"),
        ("K2 Δdengeli", "Δ(H14 − H08c) [−5; +4]", -5.0, 4.0, "E1-K2-h14-h08-dengeli_acc"),
        ("K1 Δmel", "Mel duyarlılığı Δ(H14 − H10) [−5; +12]", -5.0, 12.0, "E1-K1-h14-h10-duyarlilik_mel"),
    ]
    kont = []
    for ad, ifade, lo, hi, rid in ARALIK:
        if ifade not in s_e:
            raise SystemExit(f"E1: kart satır {i_e} '{ifade}' ifadesini taşımıyor")
        val = next(r["deger"] for r in KAYIT if r["id"] == rid)
        kont.append({"ad": ad, "aralik": [lo, hi], "gerceklesen": val, "id": rid, "icinde": lo <= val <= hi})
    if "C-val ACC [0,915; 0,945]" not in s_e or "**> 0,069**" not in s_e:
        raise SystemExit("E1: kart C-val/manipülasyon ifadesi bulunamadı")
    cv = d["egitim"]["cval_acc_ort"]
    kont.append({"ad": "H14 C-val ACC", "aralik": [0.915, 0.945], "gerceklesen": cv,
                 "kaynak_alan": "egitim.cval_acc_ort", "icinde": 0.915 <= cv <= 0.945})
    n = d["notlar"]
    kont.append({"ad": "manipülasyon: s0 ep5 train kaybı > 0,069", "gerceklesen": n["manipulasyon_s0_ep5_loss"],
                 "kaynak_alan": "notlar.manipulasyon_s0_ep5_loss", "icinde": n["manipulasyon_esik_>0,069"]})
    kont.append({"ad": "seçilen epoch ort. > 7,0", "gerceklesen": n["secilen_epoch_ort"],
                 "kaynak_alan": "notlar.secilen_epoch_ort", "icinde": n["secilen_epoch_>7,0"]})
    kont.append({"ad": "ECE ≤ H10 C1", "kaynak_alan": "notlar.ece_H14_<=_K1ref", "icinde": n["ece_H14_<=_K1ref"]})
    ekle("E1-onkayit-beklenti", "kartın ön-kayıtlı beklenti aralıkları içinde kalan madde sayısı",
         "H14", VS_C1, sum(k_["icinde"] for k_ in kont), "madde", "hipotezler/H14.md", f"satır {i_e}", KIP_E1,
         etiket=ETIKET_E1, toplam_madde=len(kont), maddeler=kont, kart_mtime="2026-11-03T22:39:10",
         not_="kart koşumdan önce yazıldı (mtime; sonuç dosyası kart_sha256 ile aynı)")
    eg = d["egitim"]
    sd = eg["seed"]
    if "2 × H10 C1 (0,0345)" not in s_e:
        raise SystemExit("E1: kartta H10 C1 s0 ep5 kaybı (0,0345) bulunamadı")
    H10_EP5 = 0.0345  # hipotezler/H14.md satır i_e: "s0 ep 5 train kaybı > 2 × H10 C1 (0,0345)"
    oran5 = n["manipulasyon_s0_ep5_loss"] / H10_EP5
    ekle("E1-mekanizma", "D4 devrede mi (kart kural g): ihlal eden seed sayısı", "H14", VS_C1,
         len(eg["kural_g_ihlal_seed"]), "seed", rel, "egitim.kural_g_ihlal_seed", KIP_E1, etiket=ETIKET_E1,
         d4_duman_ilk_adim_loss_fark=a["d4_duman_ilk_adim_loss_fark"],
         augment_frekans_maks_sapma=max(s["augment_frekans_maks_sapma"] for s in sd.values()),
         onbellek_gorunum_tutarlilik_maks_fark=max(s["onbellek_gorunum_tutarlilik_maks_fark"] for s in sd.values()),
         onbellek_gorunum_ozdes_cift=sum(s["onbellek_gorunum_ozdes_cift"] for s in sd.values()),
         train_loss_ep5_s0=n["manipulasyon_s0_ep5_loss"], h10_c1_train_loss_ep5_s0=H10_EP5,
         h10_kaynak=f"hipotezler/H14.md satır {i_e} ('2 × H10 C1 (0,0345)')",
         kural_j_notu=n["kural_j_notu"], esikler="augment ≤ 0,01; görünüm ≤ 5e-3; özdeş çift 0; D4 dumanı ≠ kontrol",
         uyari=("Beklenti-ihlali G2 adayı: D4 eklemek ACC'yi ayırt edilebilir biçimde değiştirmedi (K1). Mekanizmanın "
                "devrede olduğu ölçüldü (D4 dumanı ilk adım kaybı kontrolden farklı; augment frekansı ve önbellek "
                f"görünüm denetimleri geçti; s0 ep5 train kaybı H10'unkinin {str(round(oran5, 1)).replace('.', ',')} katı; dengeli ACC K1'de değişti) "
                "→ ACC sonucu bir arıza değil, bulgu olarak raporlanabilir."), train_loss_ep5_oran_h10=oran5)
    ekle("E1-H14-cval-acc", "C-val ACC (5 seed ort.; seçim ölçütü)", "H14", VS_CV, eg["cval_acc_ort"], "oran", rel,
         "egitim.cval_acc_ort", KIP_E1, etiket=ETIKET_E1, seed=[sd[s]["val_acc"] for s in sorted(sd)],
         secilen_epoch=[sd[s]["en_iyi_epoch"] for s in sorted(sd)], secilen_epoch_ort=eg["secilen_epoch_ort"],
         epoch_sure_ort_sn=eg["epoch_sure_ort_sn"], kural_e_az_egitim=eg["kural_e_az_egitim"],
         egitim_kosumlari=[eg["kosumlar"][s] for s in sorted(eg["kosumlar"])])
    gm = d["gorulmemis"]
    ekle("E1-gorulmemis", "kart 'görülmemiş kadar iyi' (İNCELE) bayrağı", "H14", VS_C1, gm["INCELE"], "mantıksal", rel,
         "gorulmemis", KIP_E1, etiket=ETIKET_E1, bayraklar={k_: v_ for k_, v_ in gm.items() if k_ != "INCELE"})

    # (9) ön-kayıt sapma tablosu: E1 gerçekleşen (SAP-E1 'beklemede' kaydı değiştirilmez)
    i_k, s_k = satir_bul("orkestrasyon_durumu.md", "(revizyon E1): H14 kartı yazıldı")
    ekle("SAP-E1-gerceklesen", "E1 / H14 gerçekleşen akış (SAP-E1 'beklemede' kaydının devamı)", "H10 + D4 (H14)",
         VS_C1, "tamamlandı", "durum", "orkestrasyon_durumu.md", f"satır {i_od}", KIP_SAP, zaman=_zaman(s_od),
         kart_zaman=_zaman(s_k), kart_mtime="2026-11-03T22:39:10", onceki_id="SAP-E1",
         kaynaklar=[{"dosya": "orkestrasyon_durumu.md", "satir": i_k}, {"dosya": "orkestrasyon_durumu.md", "satir": i_od},
                    {"dosya": rel, "alan": "tarih", "deger": d["tarih"]}],
         ilgili_id=["E1-kontrol-durum", "E1-ortam-cudnn", "E1-butce", "E1-ctest-acilis", "E1-K2-h14-h08-acc",
                    "E1-K1-h14-h10-acc", "E1-K1-h14-h10-dengeli_acc"])


def val_tanim_esleme():
    """Faz 5 tur-2 (review-tur2 K7): aynı kolun iki doğrulama ACC tanımı (yerel seçim ↔ L4 yeniden koşum)."""
    ix = {r["id"]: r for r in KAYIT}
    ciftler = []
    for kol, lb, v, kc in [("B", "LB-B", "V-baseline-acc", "KC1-baseline"), ("H01/H01c", "LB-H01", "V-h01-acc", "KC1-h01"),
                           ("H02/H02c", "LB-H02", "V-h02-acc", "KC1-h02"), ("H08/H08c", "LB-H08", "V-h08-acc", "KC1-h08")]:
        y, l = ix[lb]["deger"], ix[v]["deger"]
        fark = 100 * (l - y)
        # KC1 kaydındaki (iki ondalıklı) farkla tutarlılık
        assert abs(abs(fark) - ix[kc]["fark_puan"]) < 0.006, (kol, fark, ix[kc]["fark_puan"])
        ciftler.append({"kol": kol, "yerel_id": lb, "yerel": y, "L4_id": v, "L4": l, "L4_eksi_yerel_puan": fark,
                        "kapi_id": kc})
    for r in KAYIT:
        if r["id"].startswith("KC1-"):
            r["tanim"] = "deger = L4 tanımı (V-<kol>-acc, 4 ondalık); yerel = yerel tanım (LB-<kart>, 4 ondalık)"
    ekle("VAL-tanim-esleme", "iki tanımda farklı doğrulama ACC'si olan kol sayısı", "B, H01, H02, H08",
         VS_V, sum(abs(c["L4_eksi_yerel_puan"]) > 1e-9 for c in ciftler), "kol",
         "sonuclar/leaderboard.csv", "LB-* (val_acc) ↔ V-*-acc (k5_kuru_val_sonuc.json kollar.*.ort.acc)", KIP_V,
         ikinci_kaynak_dosya="sonuclar/k5_kuru_val_sonuc.json", ciftler=ciftler, yerel_tanim=VAL_TANIM_YEREL, L4_tanim=VAL_TANIM_L4,
         not_=f"H10, H11, H12 yalnız L4'te koşuldu (tek tanım). Rakip B: yerel {ciftler[0]['yerel']:.4f} (seçim "
              f"döngüsü, Tur 0) ↔ L4 {ciftler[0]['L4']:.4f} (tab:ablasyon). Fark Colab tekrar kapısında ölçülen farktır (KC1-*).")


def K_VAL(kid):
    return next(r["deger"] for r in KAYIT if r["id"] == kid)


def lit_dogrulama_isle():
    """Mevcut LIT kayıtlarına doğrulama durumu (tur-1)."""
    zhang_dosya, zi, _ = satir_bul_ws("notes/literatur-ekleri-tur1.md", "## 2. `zhang_2026_weightedsampling` — 0,882")
    for r in KAYIT:
        if r["id"] == "LIT-zhang2026-acc":
            r["yontem"] = "ImageNet ResNet-18, DermaMNIST-224, Vanilla (CE + etiket yumuşatma 0,1) temel çizgisi (3 seed)"
            r["dogrulama_durumu"] = "dogrulandi"
            r["dogrulama"] = f"{zhang_dosya} satır {zi} (Europe PMC PMC13413053, Tablo 1(d), 'Vanilla' satırı)"
            r["tarif_farki"] = "wd 0,05 (bizde 0,01); 5 epoch ısınma; batch 128 (bizde 64); etiket yumuşatma 0,1; augmentasyon/seçim bildirilmemiş"
        elif r["id"] == "LIT-medeiros2026-acc":
            r["dogrulama_durumu"] = "dogrulandi"
            r["dogrulama"] = "hakem tur-1 (review.md §2.2: arXiv:2604.09697v1 Tablo 2, 0,742)"
            r["kullanim"] = "evidence/tables/tab_replikasyon.tex (\\cite{medeiros_2026_tta}) — bib'den SİLİNMEZ"
        elif r["id"].startswith("LIT-") and "dogrulama_durumu" not in r:
            r["dogrulama_durumu"] = "dogrulandi (Faz 3/4 atıf denetimi)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    a = ap.parse_args()
    T = test_kayitlari()
    c1_kayitlari(T)
    val_kayitlari()
    ablasyon_capraz()
    replikasyon()
    leaderboard()
    val_tanim_esleme()
    butce()
    basarisiz()
    hata_kirilim()
    veri()
    nitel()
    tur1_ekleri()
    lkb_kayitlari()
    c1mel_kayitlari()
    sapma_kayitlari()
    e1_kayitlari()
    lit_dogrulama_isle()
    # not_ → not (Python anahtar sözcüğü çakışması)
    for r in KAYIT:
        if "not_" in r:
            r["not"] = r.pop("not_")
    kaynaklar = sorted({r["kaynak_dosya"] for r in KAYIT if r["kaynak_dosya"].startswith(("sonuclar", "notlar", "gunluk", "data", "hipotezler", "orkestrasyon", WS_REL))})
    out = {
        "deney_meta": {
            "yazilim": str(DW),
            "donanim": "Colab NVIDIA L4 24 GB; Python 3.13.15, torch 2.11.0+cu130, FP32, TF32 kapalı, deterministik "
                       "(karşılaştırılan tüm test/C1 sayıları). Yerel GTX 1650 yalnız K3 replikasyonu ve Tur 1–3 seçimi. "
                       "E1/H14 (2026-11-07): aynı GPU ve torch 2.11.0+cu130, cuDNN 91900 (Ekim 92700); bkz. E1-ortam-*.",
            "tarih": "2026-10-12",
            "tekrar_sayisi": 5,
            "kosul": "Ön-kayıtlı; resmî test bir kez açıldı (2026-10-11 21:12, ikinci_acilis null); C-test kol başına bir "
                     "kez; eşleştirilmiş görüntü-bootstrap (B=2000, rng 0) + tek yönlü seed-eşli t + Holm.",
            "birim_notu": "Δ'lar puan (×100). 'oran' = [0,1]. kiyas.* fraksiyonları ×100 ile puana çevrildi (ham_deger alanı).",
            "uretici": "evidence/scripts/build_evidence.py",
            "kurallar": "Bu üretici test tahmin dosyalarını açmaz (brief madde 13). Tek istisna SENS-LKB-* kayıtlarıdır: kullanıcı onaylı (2026-10-17) test sonrası duyarlılık analizi; tahminleri evidence/scripts/lkb_bootstrap.py okur, bu üretici yalnız onun JSON çıktısını okur. İkinci istisna C1-REV-* kayıtlarıdır: test-sonrası analiz (revizyon 2026-11-03, dış eleştiri M3); yalnız dondurulmuş C1 C-test tahminlerini evidence/scripts/c1_mel_bootstrap.py okur, bu üretici onun JSON çıktısını okur. Üçüncü istisna E1-* kayıtlarıdır: test sonrası keşifsel H14 kolu (revizyon E1, 2026-11-07); tahminleri deney WS scripts/e1_h14_analiz.py okur, bu üretici yalnız sonuclar/e1_h14_sonuc.json ile H14 koşumlarının metrics.json/config.json dosyalarını okur. KV C1/C2 KURU vekilleri alınmadı.",
            "kaynak_sha256": {k: sha(WS / k[len(WS_REL):] if k.startswith(WS_REL) else DW / k) for k in kaynaklar},
        },
        "kayitlar": KAYIT,
    }
    yol = WS / "evidence" / "evidence.json"
    # Şefin koyduğu "durum: terk" işaretlerini koru
    if yol.exists():
        eski = {r["id"]: r for r in json.loads(yol.read_text(encoding="utf-8")).get("kayitlar", [])}
        for r in KAYIT:
            e = eski.get(r["id"])
            if e and e.get("durum") == "terk":
                r["durum"] = "terk"
                if "paradigma" in e:
                    r["paradigma"] = e["paradigma"]
    metin = json.dumps(out, ensure_ascii=False, indent=1)
    yol.write_text(metin, encoding="utf-8")
    print(f"{len(KAYIT)} kayıt → {yol}")
    if a.cache:
        cy = Path(a.cache) / "evidence_base.json"
        cy.write_text(metin, encoding="utf-8")
        print(f"önbellek → {cy}")


if __name__ == "__main__":
    main()
