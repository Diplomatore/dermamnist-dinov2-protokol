#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lkb_bootstrap.py — TEST SONRASI DUYARLILIK ANALİZİ: lezyon-kümeli bootstrap (review.md §9.1(a)).

Kullanıcı onayı: 2026-10-17 (makale-uret Faz 5 tur 1). Hiçbir seçim / karar / ön-kayıt sonucu değişmez;
yeni eğitim yok. DONDURULMUŞ test tahminleri (resmî test 7 kol + baseline_son; C1 C-test 5 kol) okunur.

Yöntem:
  1. Deney betiği `scripts/k5_test.py` (SHA256 13a71cf1…, dondurmadaki betik) modül olarak yüklenir; kıyas, Holm,
     S/N, tabaka-std, C0, KÇ0 ve C1 kod yolları AYNEN onun fonksiyonlarıyla koşulur (kiyas, alt_kiyas, ortak_kc,
     c1_analiz, holm, tabakalar_kur, bolum_meta, c_alt_kume).
  2. Görüntü düzeyi (orijinal `Bootstrap`): sonuç dondurulmuş `k5_test_sonuc.json` / `k5_c1_sonuc.json` ile alan
     alan karşılaştırılır. Eşleşmezse betik DURUR (rc 2) ve lezyon kısmı koşulmaz.
  3. Tekil-küme sınaması: her görüntü kendi kümesi olan `KumeBootstrap`, orijinal W'yi bit düzeyinde üretmeli.
  4. Lezyon düzeyi: yalnız örnekleme birimi değişir. Aynı B = 2.000, her bootstrap örneği için aynı
     `default_rng(0)` politikası (tam test ve alt küme kendi rng(0)'ı ile). L lezyon (sıralı lesion_id kodları)
     yerine konarak çekilir; görüntü ağırlığı = lezyonunun çekilme sayısı. Ortalamalar ağırlıklı (oran kestirimcisi):
     W·f / ΣW. Alt küme (S, N, tabaka, C-test) ortalamaları k5_test.Bootstrap.ort ile aynı formülü kullanır.

Kullanım (deney venv'i; numpy/scipy/sklearn):
  <DW>/env/.venv/Scripts/python.exe -X utf8 -I evidence/scripts/lkb_bootstrap.py
Çıktı: evidence/lkb/lkb_bootstrap_sonuc.json
"""
import csv
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
WS = Path(__file__).resolve().parents[2]
DW = WS.parent  # depo kökü = deney çalışma alanı (makale/ = WS)
RC = DW / "runs_colab" if (DW / "runs_colab").is_dir() else DW / "runs_meta" / "runs_colab"  # depoda yalnız JSON damgaları runs_meta/ altında
RUNS = RC  # tahmin .npy'leri depoda YOK (bkz. makale/README.md)
CIKTI = WS / "evidence" / "lkb" / "lkb_bootstrap_sonuc.json"
DONMUS_K5_SHA = "13a71cf1375fc98a2e8c891bb4f5fdc884e43deb6819ba740aa4035428179b4c"
TOL = 1e-9


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ───────────── k5_test modülü (dondurulmuş betik) ─────────────
k5_yol = DW / "scripts" / "k5_test.py"
if sha256(k5_yol) != DONMUS_K5_SHA:
    raise SystemExit("HATA: k5_test.py SHA256 dondurulmuş değerle aynı değil")
spec = importlib.util.spec_from_file_location("k5_test", k5_yol)
K5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(K5)
ORIJINAL_BOOTSTRAP = K5.Bootstrap

T = json.loads((DW / "sonuclar/k5_test_sonuc.json").read_text(encoding="utf-8"))
C = json.loads((DW / "sonuclar/k5_c1_sonuc.json").read_text(encoding="utf-8"))

# ───────────── tahminler: dondurulmuş JSON'daki koşum adlarıyla ─────────────
DOSYA_SHA = {}
VARYANT_KOS = {}  # varyant → [run adı] (k5_test.yukle_kol yerine; aynı seed sırası)
for kaynak in (T["kosumlar"], C["C1"]["kosumlar"]):
    for kol, adlar in kaynak.items():
        for ad in adlar:
            cf = json.loads((RUNS / ad / "config.json").read_text(encoding="utf-8"))
            VARYANT_KOS.setdefault(cf["varyant"], []).append(ad)


def yukle_kol_dondurulmus(runs, varyant, dosya, etiket_dosya, kuyruk=None):
    adlar = VARYANT_KOS[varyant]
    seedler = [int(json.loads((RUNS / a / "config.json").read_text(encoding="utf-8"))["seed"]) for a in adlar]
    if seedler != [0, 1, 2, 3, 4]:
        raise SystemExit(f"HATA: {varyant} seed sırası {seedler}")
    P, y0 = [], None
    for a in adlar:
        for f in (dosya, etiket_dosya):
            DOSYA_SHA[f"runs_colab/{a}/{f}"] = sha256(RUNS / a / f)
        y = np.load(RUNS / a / etiket_dosya).reshape(-1).astype(np.int64)
        if y0 is not None and not np.array_equal(y, y0):
            raise SystemExit(f"HATA: {varyant}: etiketler farklı")
        y0 = y
        P.append(np.load(RUNS / a / dosya).astype(np.float64))
    return np.stack(P), y0, list(adlar)


K5.yukle_kol = yukle_kol_dondurulmus  # c1_analiz de bunu kullanır


# ───────────── lezyon kümeleri ─────────────
def test_lezyonlari():
    lez = {}
    with open(DW / "data/splits/dermamnist_lezyon_esleme.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "test":
                lez[int(r["index"])] = r["lesion_id"]
    return np.array([lez[i] for i in range(len(lez))])


class KumeBootstrap:
    """k5_test.Bootstrap arayüzü (N, W, ort); örnekleme birimi küme (lezyon)."""

    def __init__(self, kume_etiket):
        self.N = len(kume_etiket)
        _, kod = np.unique(kume_etiket, return_inverse=True)
        L = int(kod.max()) + 1
        rng = np.random.default_rng(K5.RNG)
        idx = rng.integers(0, L, size=(K5.NB, L))
        Ck = np.zeros((K5.NB, L))
        for b in range(K5.NB):
            Ck[b] = np.bincount(idx[b], minlength=L)
        self.W = Ck[:, kod]
        self.L = L
        self.toplam = self.W.sum(1)

    def ort(self, f, m=None):
        if m is None:
            return self.W @ f / self.toplam
        return (self.W[:, m] @ f[m]) / np.maximum(self.W[:, m].sum(1), 1)


# ───────────── analiz (k5_test.main'in istatistik kısmı, aynı sıra) ─────────────
def analiz(fabrika):
    K5.Bootstrap = fabrika
    try:
        meta = K5.bolum_meta("test")
        P, y = {}, None
        for k, v in K5.KOL.items():
            Pk, yk, _ = K5.yukle_kol(RUNS, v + "-test", "tahmin_test.npy", "etiket_test.npy", kuyruk="k5-test")
            if y is None:
                y = yk
            elif not np.array_equal(y, yk):
                raise SystemExit("HATA: etiket")
            P[k] = Pk
        assert len(y) == meta["n"] == 2005
        P["baseline_son"] = K5.yukle_kol(RUNS, K5.KOL["baseline"] + "-test", "tahmin_test_son.npy", "etiket_test.npy")[0]
        P["t4"] = K5.softmax(0.5 * np.log(np.clip(P["h08"], 1e-12, 1)) + 0.5 * np.log(np.clip(P["h10"], 1e-12, 1)))
        s = {}
        BS = K5.Bootstrap(len(y))
        tab = K5.tabakalar_kur(y, meta)
        s["n_tabaka"] = len(tab)
        K_ = {}
        for A, B in K5.AILE1 + K5.AILE2 + K5.BETIMSEL:
            K_[f"{B}-{A}"] = K5.kiyas(P[A], P[B], y, BS, meta["S"], tab)
        s["kiyas"] = K_

        def aile_holm(cifts, kural):
            ad = [f"{B}-{A}" for A, B in cifts]
            pt = [K_[x]["t_acc"]["p_tek"] for x in ad]
            pb = [K_[x]["acc"]["p_boot"] for x in ad]
            ht, hb = K5.holm(pt), K5.holm(pb)
            out = {}
            for i, x in enumerate(ad):
                e = K_[x]["acc"]
                kg = kural(e)
                out[x] = {"delta_puan": 100 * e["delta"], "ga_puan": [100 * e["ga"][0], 100 * e["ga"][1]],
                          "p_t": pt[i], "p_boot": pb[i], "holm_t": ht[i], "holm_boot": hb[i], "kural_gecti": kg,
                          "anlamli": bool(kg and ht[i] < 0.05 and hb[i] < 0.05)}
            return out

        s["aile1"] = aile_holm(K5.AILE1, lambda e: bool(e["delta"] >= 0.020 - 1e-12 and e["ga"][0] > 0))
        s["aile2"] = aile_holm(K5.AILE2, lambda e: bool(e["delta"] >= 0.005 - 1e-12 and e["ga"][0] > 0))
        sn = {}
        for x in ("h10-h02", "h08-h02", "h10-h08"):
            e, eh = K_[x]["tab_std_dS_eksi_dN"], K_[x]["ham_dS_eksi_dN"]
            sn[x] = {"ham_dS_eksi_dN_puan": 100 * eh["nokta"], "ham_ga_puan": [100 * v for v in eh["ga"]],
                     "kural": K5.etiket_kural(eh["ga"]),
                     "tab_dS_eksi_dN_puan": 100 * e["nokta"], "tab_ga_puan": [100 * v for v in e["ga"]]}
        s["SN"] = sn
        idx_c_csv = K5.c_alt_kume("test", "test")
        idx_c = np.sort(idx_c_csv)
        c0 = {"n": int(len(idx_c)), "kiyas": {}}
        for A, B, kural in (("h02", "h10", "C0-1"), ("h02", "h08", "C0-2"), ("h08", "h10", "betimsel"),
                            ("baseline", "h10", "betimsel")):
            r = K5.alt_kiyas(P[A], P[B], y, idx_c)
            c0["kiyas"][f"{B}-{A}"] = {"kural": kural, "delta_puan": 100 * r["acc"]["delta"],
                                       "ga_puan": [100 * v for v in r["acc"]["ga"]],
                                       "sonuc": K5.etiket_kural(r["acc"]["ga"]), "p_t": r.get("t_acc", {}).get("p_tek"),
                                       "dengeli_delta_puan": 100 * r["dengeli_acc"]["delta"],
                                       "dengeli_ga_puan": [100 * v for v in r["dengeli_acc"]["ga"]]}
        dacc = {k: (P[k].argmax(2) == y[None]).mean(0) for k in ("h02", "h08", "h10")}
        k10 = K5.ortak_kc(dacc["h02"], dacc["h10"], dacc["h02"], dacc["h10"], idx_c, BS)
        k08 = K5.ortak_kc(dacc["h02"], dacc["h08"], dacc["h02"], dacc["h08"], idx_c, BS)
        fark = k10["b_K"] - k08["b_K"]
        c0["KC0"] = {"K0_H10_puan": 100 * k10["K"], "K0_H10_ga": [100 * v for v in k10["K_ga"]],
                     "K0_H08_puan": 100 * k08["K"], "K0_H08_ga": [100 * v for v in k08["K_ga"]],
                     "K0_H10_eksi_K0_H08_puan": 100 * (k10["K"] - k08["K"]), "ga_puan": [100 * v for v in K5.ga(fark)],
                     "kural": K5.etiket_kural(K5.ga(fark))}
        s["C0"] = c0
        c1 = K5.c1_analiz("c1", RUNS, P, y, idx_c_csv, BS, False)
        for q in ("_PC", "_yc", "_sira", "kosumlar", "kol", "iii_baseline"):
            c1.pop(q, None)
        s["C1"] = c1
        w = s["aile1"]["h10-baseline"]
        s["olcutler"] = {"G1": bool(w["delta_puan"] >= 2.0 - 1e-9 and w["holm_t"] < 0.05 and w["ga_puan"][0] > 0),
                         "G3": c1["G3"], "L": c1["L"]}
        s["_y"], s["_idx_c"] = y, idx_c
        return s
    finally:
        K5.Bootstrap = ORIJINAL_BOOTSTRAP


# ───────────── karşılaştırma ─────────────
def kiyasla(donmus, yeni, yol, rapor):
    if isinstance(donmus, dict):
        for k, v in donmus.items():
            if k in yeni:
                kiyasla(v, yeni[k], f"{yol}.{k}", rapor)
    elif isinstance(donmus, list):
        if len(donmus) != len(yeni):
            rapor["uyusmaz"].append(f"{yol}: uzunluk")
        for i, (a, b) in enumerate(zip(donmus, yeni)):
            kiyasla(a, b, f"{yol}[{i}]", rapor)
    elif isinstance(donmus, bool) or isinstance(donmus, str) or donmus is None:
        rapor["n"] += 1
        if donmus != yeni:
            rapor["uyusmaz"].append(f"{yol}: {donmus!r} ≠ {yeni!r}")
    else:
        rapor["n"] += 1
        d = abs(float(donmus) - float(yeni))
        rapor["maks"] = max(rapor["maks"], d)
        if d > TOL:
            rapor["uyusmaz"].append(f"{yol}: {donmus} ≠ {yeni}")


def temiz(o):
    return json.loads(json.dumps(o, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else float(x)))


def main():
    t0 = time.time()
    # (1) görüntü düzeyi yeniden üretim
    G = analiz(ORIJINAL_BOOTSTRAP)
    y, idx_c = G.pop("_y"), G.pop("_idx_c")
    rp = {"n": 0, "maks": 0.0, "uyusmaz": []}
    for anahtar in ("kiyas", "aile1", "aile2", "C0"):
        kiyasla(T[anahtar], G[anahtar], f"T.{anahtar}", rp)
    for x, v in T["SN"].items():
        for q in ("ham_dS_eksi_dN_puan", "ham_ga_puan", "tab_dS_eksi_dN_puan", "tab_ga_puan"):
            kiyasla(v[q], G["SN"][x][q], f"T.SN.{x}.{q}", rp)
        if not v["kural"].startswith(G["SN"][x]["kural"]):
            rp["uyusmaz"].append(f"T.SN.{x}.kural")
    kiyasla({q: C["C1"][q] for q in ("kiyas", "aile_C1_1", "aile_C1_2", "i", "ii_KC", "G3", "L")},
            {q: G["C1"][q] for q in ("kiyas", "aile_C1_1", "aile_C1_2", "i", "ii_KC", "G3", "L")}, "C1", rp)
    kiyasla({"kiyas": C["kiyas"], "C0": C["C0"]}, {"kiyas": G["kiyas"], "C0": G["C0"]}, "C1json(resmî)", rp)
    kiyasla(T["olcutler"]["G1"], G["olcutler"]["G1"], "T.olcutler.G1", rp)
    print(f"[görüntü düzeyi] karşılaştırılan alan {rp['n']}, maks |fark| {rp['maks']:.3e}, "
          f"uyuşmaz {len(rp['uyusmaz'])}  ({time.time() - t0:.0f} sn)")
    if rp["uyusmaz"]:
        print("\n".join(rp["uyusmaz"][:40]))
        raise SystemExit(2)

    # (2) tekil-küme sınaması: KumeBootstrap(her görüntü ayrı) == orijinal W (bit düzeyi)
    tek_tam = np.array_equal(KumeBootstrap(np.arange(2005)).W, ORIJINAL_BOOTSTRAP(2005).W)
    tek_c = np.array_equal(KumeBootstrap(np.arange(len(idx_c))).W, ORIJINAL_BOOTSTRAP(len(idx_c)).W)
    print(f"[tekil-küme sınaması] tam test W özdeş: {tek_tam}; C-test W özdeş: {tek_c}")
    if not (tek_tam and tek_c):
        raise SystemExit(3)

    # (3) lezyon düzeyi
    lez = test_lezyonlari()
    assert len(lez) == len(y)
    lez_c = lez[idx_c]
    ust = {}

    def fab(N):
        if N == len(lez):
            return KumeBootstrap(lez)
        if N == len(lez_c):
            return KumeBootstrap(lez_c)
        raise SystemExit(f"HATA: beklenmeyen bootstrap boyutu {N}")

    Lz = analiz(fab)
    Lz.pop("_y"), Lz.pop("_idx_c")
    bs_t, bs_c = KumeBootstrap(lez), KumeBootstrap(lez_c)
    ust["kume"] = {"tam_test": {"n_goruntu": int(len(lez)), "n_lezyon": bs_t.L,
                                "agirlik_toplami_ort": float(bs_t.toplam.mean()),
                                "agirlik_toplami_min_max": [float(bs_t.toplam.min()), float(bs_t.toplam.max())]},
                   "c_test": {"n_goruntu": int(len(lez_c)), "n_lezyon": bs_c.L,
                              "agirlik_toplami_ort": float(bs_c.toplam.mean()),
                              "agirlik_toplami_min_max": [float(bs_c.toplam.min()), float(bs_c.toplam.max())]}}
    print(f"[lezyon düzeyi] tamam ({time.time() - t0:.0f} sn); lezyon: test {bs_t.L}, C-test {bs_c.L}")

    # ───── özet satırları (tablo + defter) ─────
    def et(g):
        return K5.etiket_kural([g[0] / 100, g[1] / 100]).split(" (")[0]

    satir = []

    def ekle(anahtar, ad, blok, ref, nokta, gg, gl, pb_g=None, pb_l=None, **ek):
        nokta, gg, gl = float(nokta), [float(v) for v in gg], [float(v) for v in gl]
        satir.append({"anahtar": anahtar, "ad": ad, "blok": blok, "ref_id": ref, "nokta_puan": nokta,
                      "ga_goruntu_puan": gg, "ga_lezyon_puan": gl,
                      "genislik_orani": (gl[1] - gl[0]) / (gg[1] - gg[0]),
                      "etiket_goruntu": et(gg), "etiket_lezyon": et(gl), "isaret_degisti": et(gg) != et(gl),
                      "p_boot_goruntu": pb_g, "p_boot_lezyon": pb_l, **ek})

    AD = {"baseline": "B", "h01": "H01c", "h02": "H02c", "h08": "H08c", "h10": "H10", "h11": "H11", "h12": "H12",
          "t4": "T4", "baseline_son": "B son-epoch"}

    def cad(c):
        b, a = c.split("-")
        return f"{AD[b]} − {AD[a]}"

    for aile, cifts in (("aile1", K5.AILE1), ("aile2", K5.AILE2)):
        for A, B in cifts:
            c = f"{B}-{A}"
            g, l = G[aile][c], Lz[aile][c]
            ekle(f"T-{c}", cad(c), f"resmî test, {aile}", f"T-{aile.upper()}-{c}", g["delta_puan"], g["ga_puan"],
                 l["ga_puan"], g["p_boot"], l["p_boot"], holm_boot_goruntu=g["holm_boot"], holm_boot_lezyon=l["holm_boot"],
                 holm_t=g["holm_t"], anlamli_goruntu=g["anlamli"], anlamli_lezyon=l["anlamli"],
                 kural_gecti_goruntu=g["kural_gecti"], kural_gecti_lezyon=l["kural_gecti"])
    for A, B in K5.BETIMSEL:
        c = f"{B}-{A}"
        g, l = G["kiyas"][c]["acc"], Lz["kiyas"][c]["acc"]
        ekle(f"T-{c}", cad(c), "resmî test, betimsel", f"T-K-{c}-acc", 100 * g["delta"],
             [100 * v for v in g["ga"]], [100 * v for v in l["ga"]], g["p_boot"], l["p_boot"])
    for c in ("h10-h02", "h08-h02", "h10-h08"):
        for q, alan, isim in (("dS", "acc_S_ham", "Δ_S"), ("dN", "acc_N_ham", "Δ_N"),
                              ("dSdN", "ham_dS_eksi_dN", "Δ_S − Δ_N (ham)"),
                              ("tabstd", "tab_std_dS_eksi_dN", "Δ_S − Δ_N (tabaka-std.)")):
            g, l = G["kiyas"][c][alan], Lz["kiyas"][c][alan]
            nk = g["delta"] if "delta" in g else g["nokta"]
            ref = f"T-K-{c}-{q}" if q != "tabstd" else f"T-K-{c}-tab_std_dS_eksi_dN"
            ekle(f"T-{c}-{q}", f"{cad(c)}: {isim}", "resmî test, S/N", ref, 100 * nk,
                 [100 * v for v in g["ga"]], [100 * v for v in l["ga"]])
    for c in ("h10-h02", "h08-h02", "h10-h08", "h10-baseline"):
        g, l = G["C0"]["kiyas"][c], Lz["C0"]["kiyas"][c]
        ekle(f"C0-{c}", cad(c), "C0", f"T-C0-K-{c}", g["delta_puan"], g["ga_puan"], l["ga_puan"])
    gk, lk = G["C0"]["KC0"], Lz["C0"]["KC0"]
    ekle("C0-K0-H10", "K_0 (H10 − H02c)", "KÇ_0", "T-C0-K0-H10", gk["K0_H10_puan"], gk["K0_H10_ga"], lk["K0_H10_ga"])
    ekle("C0-K0-H08", "K_0 (H08c − H02c)", "KÇ_0", "T-C0-K0-H08", gk["K0_H08_puan"], gk["K0_H08_ga"], lk["K0_H08_ga"])
    ekle("C0-KC0", "KÇ_0 = K_0,H10 − K_0,H08", "KÇ_0", "T-C0-KC0", gk["K0_H10_eksi_K0_H08_puan"], gk["ga_puan"],
         lk["ga_puan"])
    for aile, ref_on in (("aile_C1_1", "C1-A1"), ("aile_C1_2", "C1-A2")):
        for c, g in G["C1"][aile].items():
            l = Lz["C1"][aile][c]
            ekle(f"C1-{c}", cad(c), f"C1, {aile}", f"{ref_on}-{c}", g["delta_puan"], g["ga_puan"], l["ga_puan"],
                 G["C1"]["kiyas"][c]["acc"]["p_boot"], Lz["C1"]["kiyas"][c]["acc"]["p_boot"],
                 holm_boot_goruntu=g["holm_boot"], holm_boot_lezyon=l["holm_boot"], holm_t=g["holm_t"],
                 kural_gecti_goruntu=g["kural_gecti"], kural_gecti_lezyon=l["kural_gecti"])
    g, l = G["C1"]["kiyas"]["h10-h08"]["acc"], Lz["C1"]["kiyas"]["h10-h08"]["acc"]
    ekle("C1-h10-h08", cad("h10-h08"), "C1, betimsel", "C1-K-h10-h08-acc", 100 * g["delta"],
         [100 * v for v in g["ga"]], [100 * v for v in l["ga"]], g["p_boot"], l["p_boot"])
    gk, lk = G["C1"]["ii_KC"], Lz["C1"]["ii_KC"]
    ekle("C1-K-H10", "K (H10 − H02c)", "KÇ", "C1-K-H10", gk["K_H10_puan"], gk["K_H10_ga"], lk["K_H10_ga"])
    ekle("C1-K-H08", "K (H08c − H02c)", "KÇ", "C1-K-H08", gk["K_H08_puan"], gk["K_H08_ga"], lk["K_H08_ga"])
    ekle("C1-KC", "KÇ = K_H10 − K_H08", "KÇ", "C1-KC", gk["K_H10_eksi_K_H08_puan"], gk["ga_puan"], lk["ga_puan"])

    karar = {"G1": [G["olcutler"]["G1"], Lz["olcutler"]["G1"]], "G3": [G["olcutler"]["G3"], Lz["olcutler"]["G3"]],
             "L": [G["olcutler"]["L"], Lz["olcutler"]["L"]],
             "KC_ifade_esigi": [G["C1"]["ii_KC"]["ifade_esigi"], Lz["C1"]["ii_KC"]["ifade_esigi"]]}
    out = {
        "etiket": "test sonrası duyarlılık analizi — hiçbir seçim/karar/ön-kayıt sonucu değişmez; yeni eğitim yok",
        "onay": "kullanıcı, 2026-10-17 (review.md §9.1(a))",
        "tarih": datetime.now().isoformat(timespec="seconds"),
        "betik": "evidence/scripts/lkb_bootstrap.py", "betik_sha256": sha256(__file__),
        "k5_test_sha256": DONMUS_K5_SHA,
        "donmus_kaynak_sha256": {"sonuclar/k5_test_sonuc.json": sha256(DW / "sonuclar/k5_test_sonuc.json"),
                                 "sonuclar/k5_c1_sonuc.json": sha256(DW / "sonuclar/k5_c1_sonuc.json"),
                                 "data/splits/dermamnist_lezyon_esleme.csv":
                                     sha256(DW / "data/splits/dermamnist_lezyon_esleme.csv"),
                                 "data/splits/dermamnist_c_esleme.csv": sha256(DW / "data/splits/dermamnist_c_esleme.csv")},
        "yontem": {"B": K5.NB, "rng": f"numpy.random.default_rng({K5.RNG}) her bootstrap nesnesi için (tam test ve "
                                       "C-test alt kümesi ayrı), görüntü düzeyiyle aynı politika",
                   "goruntu": "k5_test.Bootstrap (dondurulmuş): N görüntü yerine konarak",
                   "lezyon": "L lezyon (np.unique sıralı lesion_id kodu) yerine konarak; görüntü ağırlığı = lezyonun "
                             "çekilme sayısı; ortalama = W·f/ΣW (oran kestirimcisi); alt küme ortalamaları "
                             "k5_test.Bootstrap.ort formülüyle; GA %2,5–%97,5 yüzdelik; p = (1+#{Δ*≤0})/(B+1)",
                   "degismeyen": "nokta kestirimleri, seed-düzeyi t p'leri (Holm t), seçimler"},
        "dogrulama_goruntu_duzeyi": {"karsilastirilan_alan": rp["n"], "maks_mutlak_fark": rp["maks"],
                                     "uyusmaz": len(rp["uyusmaz"]), "tolerans": TOL,
                                     "kapsam": "k5_test_sonuc.json: kiyas(15 çift), aile1, aile2, SN, C0, olcutler.G1; "
                                               "k5_c1_sonuc.json: kiyas, C0, C1.{kiyas, aile_C1_1, aile_C1_2, i, ii_KC, G3, L}"},
        "tekil_kume_sinamasi": {"tam_test_W_ozdes": bool(tek_tam), "c_test_W_ozdes": bool(tek_c)},
        **ust,
        "karar_bayraklari_goruntu_lezyon": karar,
        "ozet": satir,
        "tahmin_dosyalari_sha256": DOSYA_SHA,
        "tam_lezyon": temiz(Lz),
        "sure_sn": round(time.time() - t0, 1),
    }
    CIKTI.parent.mkdir(parents=True, exist_ok=True)
    CIKTI.write_text(json.dumps(temiz(out), ensure_ascii=False, indent=1), encoding="utf-8")
    n_deg = sum(r["isaret_degisti"] for r in satir)
    print(f"{len(satir)} satır; etiket değişen {n_deg}; karar bayrakları {karar}")
    for r in satir:
        print(f"{r['anahtar']:<26} {r['nokta_puan']:+7.2f}  G[{r['ga_goruntu_puan'][0]:+.2f}; {r['ga_goruntu_puan'][1]:+.2f}]"
              f"  L[{r['ga_lezyon_puan'][0]:+.2f}; {r['ga_lezyon_puan'][1]:+.2f}]  x{r['genislik_orani']:.2f}"
              f"  {'DEĞİŞTİ' if r['isaret_degisti'] else ''}")
    print(f"yazıldı: {CIKTI} ({time.time() - t0:.0f} sn)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
