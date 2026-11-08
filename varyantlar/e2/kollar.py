# -*- coding: utf-8 -*-
"""
e2/kollar.py — H15 (E2, BCN20000) kol kaydı: TEK KAYNAK. Kuyruk yazıcısı, çıkarım betiği, KİLİT ve analiz buradan okur.

Kart (hipotezler/H15.md "Ebeveyn varyant" + §Veri kararları 5): 11 kol × seed {0..4} = 55 kol-seed. Rakip 28 px kolları
(baseline_c, baseline_c1) ağırlık dosyası olmadığı için DIŞLANIR (KULLANICI ONAYI H15, 2026-11-07).
  tur   : k5_cikarim.py türleri (r18 / vit_kismi / vit_tam / sonda)
  veri  : resmi (resmî val 1.003) | c1 (C-val 573; c_bolme yaması)
  rol   : birincil (B1/B2) · ikincil (S4, H14) · betimsel (sıralama, S6)
  ic    : iç test tahminlerinin arşiv varyantı + kuyruk öneki (yalnız analiz; dondurulmuş K5/E1 tahminleri)
"""

SEEDLER = (0, 1, 2, 3, 4)

KOLLAR = {
    # ── resmî bölme kolları (Colab L4, kc1-tekrar / kc2-tur4) ──
    "h10": {"ad": "H10", "tur": "vit_kismi", "veri": "resmi", "rol": "birincil", "hip": "H10",
            "varyant": "h10_dinov2b_k4_ft224",
            "ids": ["20261011-160211", "20261011-161411", "20261011-162611", "20261011-163811", "20261011-165010"],
            "ic": ("h10_dinov2b_k4_ft224-test", "k5-test")},
    "h02c": {"ad": "H02c", "tur": "r18", "veri": "resmi", "rol": "birincil", "hip": "H02",
             "varyant": "h02_r18in_ft224_colab",
             "ids": ["20261011-130855", "20261011-131620", "20261011-132346", "20261011-133112", "20261011-133837"],
             "ic": ("h02_r18in_ft224_colab-test", "k5-test")},
    "h08c": {"ad": "H08c", "tur": "r18", "veri": "resmi", "rol": "ikincil", "hip": "H08",
             "varyant": "h08_r18in_ft224_d4_colab",
             "ids": ["20261011-134604", "20261011-135335", "20261011-140107", "20261011-140837", "20261011-141608"],
             "ic": ("h08_r18in_ft224_d4_colab-test", "k5-test")},
    "h11": {"ad": "H11", "tur": "vit_tam", "veri": "resmi", "rol": "betimsel", "hip": "H11",
            "varyant": "h11_dinov2s_ft224",
            "ids": ["20261011-160117", "20261011-161247", "20261011-162419", "20261011-163552", "20261011-164724"],
            "ic": ("h11_dinov2s_ft224-test", "k5-test")},
    "h01c": {"ad": "H01c", "tur": "sonda", "veri": "resmi", "rol": "betimsel", "hip": "H01",
             "varyant": "h01_dinov2b_lp_r224_colab",
             "ids": ["20261011-125349", "20261011-125744", "20261011-130031", "20261011-130320", "20261011-130608"],
             "ic": ("h01_dinov2b_lp_r224_colab-test", "k5-test")},
    "h12": {"ad": "H12", "tur": "sonda", "veri": "resmi", "rol": "betimsel", "hip": "H12",
            "varyant": "h12_dinov2s_lp_r224",
            "ids": ["20261011-174515", "20261011-174846", "20261011-175135", "20261011-175422", "20261011-175710"],
            "ic": ("h12_dinov2s_lp_r224-test", "k5-test")},
    # ── C1 (lezyon-ayrık, yeniden eğitim) kolları ──
    "h10_c1": {"ad": "h10_c1", "tur": "vit_kismi", "veri": "c1", "rol": "birincil", "hip": "H10", "varyant": "h10_c1",
               "ids": ["20261011-225511", "20261011-230845", "20261011-232219", "20261011-233553", "20261011-234926"],
               "ic": ("h10_c1-test", "k5-c1")},
    "h02_c1": {"ad": "h02_c1", "tur": "r18", "veri": "c1", "rol": "birincil", "hip": "H02", "varyant": "h02_c1",
               "ids": ["20261012-001716", "20261012-002530", "20261012-003344", "20261012-004159", "20261012-005014"],
               "ic": ("h02_c1-test", "k5-c1")},
    "h08_c1": {"ad": "h08_c1", "tur": "r18", "veri": "c1", "rol": "ikincil", "hip": "H08", "varyant": "h08_c1",
               "ids": ["20261011-221301", "20261011-222126", "20261011-222952", "20261011-223818", "20261011-224645"],
               "ic": ("h08_c1-test", "k5-c1")},
    "h14_c1": {"ad": "h14_c1", "tur": "vit_kismi", "veri": "c1", "rol": "ikincil", "hip": "H14", "varyant": "h14_c1",
               "ids": ["20261107-163510", "20261107-164913", "20261107-170310", "20261107-171703", "20261107-173053"],
               "ic": ("h14_c1-test", "e1-h14")},
    "h01_c1": {"ad": "h01_c1", "tur": "sonda", "veri": "c1", "rol": "betimsel", "hip": "H01", "varyant": "h01_c1",
               "ids": ["20261012-000005", "20261012-000407", "20261012-000724", "20261012-001040", "20261012-001358"],
               "ic": ("h01_c1-test", "k5-c1")},
}

# Kuyruk sırası: bütçe tükenirse önce birincil kollar bitmiş olsun (kart K-g; sıra sonucu etkilemez, seçim değildir).
SIRA = ["h10", "h02c", "h10_c1", "h02_c1", "h08c", "h08_c1", "h14_c1", "h11", "h01c", "h12", "h01_c1"]
BIRINCIL = [k for k in SIRA if KOLLAR[k]["rol"] == "birincil"]
RESMI = [k for k in SIRA if KOLLAR[k]["veri"] == "resmi"]
C1 = [k for k in SIRA if KOLLAR[k]["veri"] == "c1"]
VAL_N = {"resmi": 1003, "c1": 573}
assert sorted(SIRA) == sorted(KOLLAR) and len(SIRA) == 11


def run_id(kol: str, seed: int) -> str:
    k = KOLLAR[kol]
    return f"{k['ids'][seed]}_{k['varyant']}_s{seed}"


def e2_varyant(kol: str, kip: str) -> str:
    """kip: duman | onucus | bcn → kuyruk varyant adı (kart taslağıyla aynı biçim)."""
    return {"duman": f"e2_{kol}_duman", "onucus": f"e2_{kol}_onucus", "bcn": f"e2_{kol}-bcn"}[kip]


def kol_of_varyant(varyant: str):
    for k in KOLLAR:
        if varyant in (e2_varyant(k, "duman"), e2_varyant(k, "onucus"), e2_varyant(k, "bcn")):
            return k
    return None
