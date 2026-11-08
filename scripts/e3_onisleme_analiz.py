# -*- coding: utf-8 -*-
"""
e3_onisleme_analiz.py — H16 / E3 (BCN ön-işleme hattının B1'e katkısı, resmî val iç-veri kestirimi) ANALİZİ. CPU; keşifsel,
sonradan. TUT/ÖLDÜR VERMEZ; birincil Δ_hat etiketini ve önceden yazılmış cümleleri MEKANİK seçer (hipotezler/H16.md
§Karar kuralı, §İstatistik). K5, E2 (B1/B2, Holm aileleri, e2_bcn_sonuc.*) ve iki kilit (k5_test.KILIT, e2_bcn.KILIT)
DEĞİŞMEZ; BCN etiket CSV'si okunmaz; resmî test ve C-test açılmaz.

Bootstrap ve sınıf duyarlılığı rutinleri `scripts/e2_bcn_analiz.py`'den İÇE AKTARILIR (multinom, ozet, dis_kol; biçim için
tr, ga_tr, p_tr, jsonla, sha) — yeniden yazılmaz. Tek ince sarmalayıcı `deng()`: dis_kol'un sınıf duyarlılıklarından dengeli
doğruluğu yalnız lezyonu bulunan sınıflar üzerinden alır (N alt kümesinde boş sınıf → kart §İstatistik beyanı).
Çekiliş (E2 ile aynı sıra): rng = default_rng(0); (1) cnt = multinom(rng, B, 5) (seed çokluğu; bütün kollar ve bütün hat
kolları aynı); (2) val lezyon kümeleri sınıf tabakalı: sınıf 0..6, tabaka başına multinom(rng, B, L_c) (E2 Cekilis.dis eşi).
N alt kümesi = W'nin N lezyonlarına düşen sütunları (yeni çekiliş yok). S8: cnt = 1 (seed sabit), aynı W.

Sıra (gerçek analiz, BİR KEZ):
 0) KİLİT (`sonuclar/e3_onisleme.KILIT`, `--kilit-ac` ile a3 sonrası yazılır) var; içindeki SHA256'lar (kart = onaylı,
    bu betik, beklenti, beklenen, e3_cikarim, e3_ham_val_hazirla, e2_bcn_analiz, kollar) güncel; `analiz_zamani` yok
    (varsa yalnız --ikinci-acilis "<neden>"; iki sonuç da yazılır, beyan eklenir; K-g).
 1) a3 denetimleri: 4 kol × 5 seed `e3_<kol>` (kuyruk e3-onisleme*), rc 0, kaynak run_id = kollar.py, Durum A/B (H10 ya da
    H02c'de C → DUR; H08c/H11'de C → o kol dışarıda), girdi/sıra SHA, 4 hat geçişi, K-e, K-g sayaçları, bütçe ≤ 2,0 GPU-sa.
 2) K-d KURU KOŞU (hat tahminleri OKUNMADAN): yerel kol val dengelisi ↔ v3.json `orijinal` (kol başına ≤ 0,05 puan);
    Δ_yerel(H10 − H02c) ↔ v3 tam değeri (± 0,05); öz-sınama "hat := yerel" → Δ_hat = 0, GA [0; 0]; eşikler 2,48 / 10,31
    ↔ e2_bcn_sonuc.json (B1 − GA_alt; B1 − iç resmî dengeli) ± 0,005. Biri tutmazsa DUR.
 3) ANCAK BUNDAN SONRA hat tahminleri okunur → birincil Δ_hat + etiket → S1–S10 → İNCELE (i)–(iv) → cümleler.
Kipler:
  --kuru       K-d + uçtan uca KOD sınaması: yerel := arşiv `tahmin_val.npy` (runs_colab), hat := yerel (öz-sınama) ve
               VEKİL hat kolları (arşiv `tahmin_val_son.npy` / `tahmin_val_aucsecim.npy`; 2 görüntü yapay "eksik") — sonuç DEĞİL.
               Çıktı sonuclar/e3_onisleme_kuru_sonuc.{json,md} (+ .sha256). Hat girdisi, KİLİT ve a3 koşumları gerekmez.
  --kilit-ac   a3 bittikten sonra: (1) + (2) (gerçek yerel kollarla) → KİLİT yazılır. Hat tahmini okunmaz.
  (varsayılan) gerçek analiz → sonuclar/e3_onisleme_sonuc.{json,md} + .sha256; KİLİT'e analiz_zamani yazılır.
Kullanım (WS kökü):
  env/.venv/Scripts/python.exe -I scripts/e3_onisleme_analiz.py --kuru
  env/.venv/Scripts/python.exe -I scripts/e3_onisleme_analiz.py --kilit-ac
  env/.venv/Scripts/python.exe -I scripts/e3_onisleme_analiz.py
"""
import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "scripts"))
sys.path.insert(0, str(WS / "varyantlar"))
import e2_bcn_analiz as A  # noqa: E402  (multinom, ozet, dis_kol, tr, ga_tr, p_tr, jsonla, sha — İÇE AKTARILIR)
from e2 import bcn_veri as BV  # noqa: E402
from e2 import kollar as KL  # noqa: E402

KART = WS / "hipotezler" / "H16.md"
ONAYLI_KART_SHA = "f3c6998196e0624801979b01da5226747c8f876502bd8e9c785bd5c6d2157109"   # KULLANICI KARARI 15 (2026-11-08)
BEKLENTI = WS / "kuyruk" / "e3-onisleme.beklenti.json"
BEKLENEN = WS / "kuyruk" / "e3_onisleme_beklenen.json"
KILIT = WS / "sonuclar" / "e3_onisleme.KILIT"
E2_SONUC = WS / "sonuclar" / "e2_bcn_sonuc.json"
V3_JSON = WS / "sonuclar" / "e2_incele_ek" / "v3.json"
ESLEME_CSV = WS / "data" / "splits" / "dermamnist_lezyon_esleme.csv"
NPZ224 = WS / "data" / "medmnist" / "dermamnist_224.npz"
CIKARIM = WS / "varyantlar" / "e3" / "e3_cikarim.py"
HAZIRLA = WS / "scripts" / "e3_ham_val_hazirla.py"
KUYRUK = "e3-onisleme"
KOLLAR = ("h10", "h02c", "h08c", "h11")
HATLAR = ("yerel", "hat", "jpeg", "z125", "z080")
SEEDLER = tuple(KL.SEEDLER)
B_BOOT, RNG = 10_000, 0
M_ESIK, G_ESIK, ESIK_TOL = 2.48, 10.31, 0.005          # kart §Karar kuralı (E2 donmuş sayıları)
B1_DIS = 16.34                                           # e2_bcn_sonuc.json B1 (KISMEN oranı için; kartta yazılı)
KD_TOL = 0.05                                            # puan (K-d)
INC_UST, INC_ALT, INC_VIT_D, INC_R18_UYUM = 8.0, -6.0, -5.0, 0.90
BUTCE_SA = 2.0
KAL_R18 = None                                           # v3.json kal: R18 argmax uyumu (157/160) — çalışma anında okunur
BCN_DM_KANAL = (148.03 - 195.48, 128.69 - 136.70, 123.79 - 142.80)   # v1/v3 kanal ortalamaları (BCN − DermaMNIST train)
NK, NV = 7, 5
SIDDET = {"DIŞLANDI": 0, "KÜÇÜK-BELİRSİZ": 1, "KISMEN": 2, "YORUM DEĞİŞİR": 3}
BILESEN = {"jpeg": "JPEG yeniden sıkıştırma (q = q_BCN)", "z125": "×1,25 ölçek (yakınlaştırma)", "z080": "×0,8 ölçek (uzaklaştırma)"}
tr, ga_tr, p_tr = A.tr, A.ga_tr, A.p_tr


def dur(msg: str):
    print(f"DUR: {msg}", flush=True)
    raise SystemExit(2)


def yuzde(x, n=1) -> str:
    return f"{100 * x:.{n}f}".replace(".", ",")


# ───────────────────────── veri ─────────────────────────
def val_etiket() -> np.ndarray:
    """dermamnist_224.npz → YALNIZ val_labels (K-a)."""
    with np.load(NPZ224, allow_pickle=False) as z:
        return z["val_labels"].reshape(-1).astype(np.int64)


def lezyon_tablosu():
    with open(ESLEME_CSV, encoding="utf-8", newline="") as f:
        satir = list(csv.DictReader(f))
    val = sorted((r for r in satir if r["split"] == "val"), key=lambda r: int(r["index"]))
    if [int(r["index"]) for r in val] != list(range(1003)):
        dur("eşleme CSV'sinde val index kümesi {0..1002} değil")
    train_lez = {r["lesion_id"] for r in satir if r["split"] == "train"}
    lez = np.array([r["lesion_id"] for r in val])
    N = np.array([x not in train_lez for x in lez])
    return lez, N, np.array([r["image_id"] for r in val])


def cfg_kur(y: np.ndarray, lez: np.ndarray) -> dict:
    """dis_kol yapılandırması (E2 Dis.secim eşi): görüntü → lezyon kümesi, küme → sınıf (tek tanı denetimi)."""
    uniq, cl = np.unique(lez, return_inverse=True)
    clcls = np.full(len(uniq), -1)
    for i, c in zip(cl, y):
        if clcls[i] not in (-1, c):
            dur(f"val lezyon kümesi {uniq[i]} birden çok sınıf içeriyor")
        clcls[i] = c
    return {"sel": np.arange(len(y)), "y": y, "cl": cl, "clcls": clcls, "L": len(uniq), "uniq": uniq,
            "n_l": np.bincount(cl, minlength=len(uniq)).astype(np.float64), "n": len(y)}


def cfg_alt(cfg: dict, kume_maske: np.ndarray) -> dict:
    """Küme alt kümesi (S4): görüntüleri bütünüyle düşen kümeler; W sütunları aynı maskeyle alınır."""
    yeni = -np.ones(cfg["L"], np.int64)
    yeni[kume_maske] = np.arange(kume_maske.sum())
    sel = np.where(kume_maske[cfg["cl"]])[0]
    return {"sel": cfg["sel"][sel], "y": cfg["y"][sel], "cl": yeni[cfg["cl"][sel]], "clcls": cfg["clcls"][kume_maske],
            "L": int(kume_maske.sum()), "uniq": cfg["uniq"][kume_maske], "n_l": cfg["n_l"][kume_maske], "n": len(sel)}


def cekilis(cfg: dict, B: int = B_BOOT):
    rng = np.random.default_rng(RNG)
    cnt = A.multinom(rng, B, len(SEEDLER)).astype(np.float64)        # (1) seed çokluğu
    W = np.zeros((B, cfg["L"]))
    for c in range(NK):                                                # (2) sınıf tabakalı lezyon kümesi (Cekilis.dis eşi)
        Lc = np.where(cfg["clcls"] == c)[0]
        if len(Lc):
            W[:, Lc] = A.multinom(rng, B, len(Lc))
    return cnt, W


def deng(Pk: np.ndarray, cfg: dict, W: np.ndarray, cnt: np.ndarray):
    """İnce sarmalayıcı: A.dis_kol → dengeli (yalnız lezyonu olan sınıflar). Dönüş (boot (B,), seed noktaları (5,), duyarlılık (7,))."""
    with np.errstate(invalid="ignore", divide="ignore"):
        rb, r0 = A.dis_kol(Pk, cfg, W, cnt)
    mevcut = np.bincount(cfg["clcls"], minlength=NK) > 0
    return rb[:, mevcut].mean(1), r0[:, mevcut].mean(1), r0.mean(0)


def etiket(nokta: float, ga) -> str:
    if -M_ESIK < ga[0] and ga[1] < M_ESIK:
        return "DIŞLANDI"
    if nokta <= -M_ESIK:
        return "TERS"
    if nokta > G_ESIK:
        return "YORUM DEĞİŞİR"
    if nokta >= M_ESIK:
        return "KISMEN"
    return "KÜÇÜK-BELİRSİZ"


class Veri:
    """y (n,) · lez (n,) · N (n,) · index (n,) · P[kol][hat] (5, n, 7) — hat sırası; Pyerel_tam[kol] (5, 1003, 7)."""

    def __init__(self, yerel_tam: dict, hat: dict | None, index: np.ndarray, kaynak: str):
        self.y_tam = val_etiket()
        lez, N, ids = lezyon_tablosu()
        self.lez_tam, self.N_tam, self.ids_tam = lez, N, ids
        self.yerel_tam = yerel_tam
        self.index = np.asarray(index, np.int64)
        self.y, self.lez, self.N = self.y_tam[self.index], lez[self.index], N[self.index]
        self.hat = hat
        self.kaynak = kaynak

    def P(self, k: str, g: str) -> np.ndarray:
        if g == "yerel":
            return self.yerel_tam[k][:, self.index]
        return self.hat[k][g]


# ───────────────────────── K-d ─────────────────────────
def esik_denetle() -> dict:
    r = json.loads(E2_SONUC.read_text(encoding="utf-8"))
    b1, ga_alt = r["B1"]["nokta"], r["B1"]["ga"][0]
    ic = r["ana"]["S"]["h10-h02"]["ic_R_dengeli"]["nokta"]
    m, g = b1 - ga_alt, b1 - ic
    ok = abs(m - M_ESIK) <= ESIK_TOL and abs(g - G_ESIK) <= ESIK_TOL and abs(b1 - B1_DIS) <= ESIK_TOL
    return {"gecti": bool(ok), "m_kaynak": m, "G_kaynak": g, "B1": b1, "B1_ga_alt": ga_alt, "ic_R_dengeli": ic,
            "m_sabit": M_ESIK, "G_sabit": G_ESIK, "tol": ESIK_TOL, "e2_bcn_sonuc_sha256": A.sha(E2_SONUC)}


def kuru_kosu(V: Veri, kollar) -> dict:
    """K-d: hat tahminleri okunmadan. (a) v3 yeniden üretimi; (b) Δ_yerel; (c) öz-sınama hat := yerel; (d) eşikler."""
    v3 = json.loads(V3_JSON.read_text(encoding="utf-8"))
    cfg_t = cfg_kur(V.y_tam, V.lez_tam)
    bir_W, bir_c = np.ones((1, cfg_t["L"])), np.ones((1, len(SEEDLER)))
    sat, ok = {}, True
    nok = {}
    for k in kollar:
        _, r0, _ = deng(V.yerel_tam[k], cfg_t, bir_W, bir_c)
        nok[k] = 100 * float(r0.mean())
        ref = 100 * float(np.mean([x["dengeli"] for x in v3["val"][k]["orijinal"]]))
        g = abs(nok[k] - ref) <= KD_TOL
        sat[k] = {"bu_puan": nok[k], "v3_puan": ref, "fark": nok[k] - ref, "seed_puan": (100 * r0).tolist(), "gecti": bool(g)}
        ok &= g
    v3d = 100 * (np.mean([x["dengeli"] for x in v3["val"]["h10"]["orijinal"]]) - np.mean([x["dengeli"] for x in v3["val"]["h02c"]["orijinal"]]))
    dy = nok["h10"] - nok["h02c"]
    gy = abs(dy - v3d) <= KD_TOL
    ok &= gy
    # (c) öz-sınama: hat := yerel → Δ_hat ≡ 0 (aynı W, cnt; tam eşitlik beklenir)
    cfg = cfg_kur(V.y, V.lez)
    cnt, W = cekilis(cfg)
    d = {k: deng(V.P(k, "yerel"), cfg, W, cnt) for k in ("h10", "h02c")}
    dh = {k: deng(V.P(k, "yerel").copy(), cfg, W, cnt) for k in ("h10", "h02c")}
    boot = 100 * ((dh["h10"][0] - dh["h02c"][0]) - (d["h10"][0] - d["h02c"][0]))
    nokta = 100 * ((dh["h10"][1].mean() - dh["h02c"][1].mean()) - (d["h10"][1].mean() - d["h02c"][1].mean()))
    oz = A.ozet(nokta, boot)
    go = bool(abs(nokta) <= 1e-12 and abs(oz["ga"][0]) <= 1e-12 and abs(oz["ga"][1]) <= 1e-12)
    ok &= go
    es = esik_denetle()
    ok &= es["gecti"]
    return {"gecti": bool(ok), "tol_puan": KD_TOL, "kol": sat, "delta_yerel": {"bu": dy, "v3": v3d, "fark": dy - v3d, "gecti": bool(gy)},
            "oz_sinama": {"nokta": nokta, "ga": oz["ga"], "gecti": go}, "esikler": es, "kaynak": V.kaynak,
            "v3_sha256": A.sha(V3_JSON)}


# ───────────────────────── birincil + ikinciller ─────────────────────────
def analiz(V: Veri, kollar, beklenen: dict | None) -> dict:
    cfg = cfg_kur(V.y, V.lez)
    cnt, W = cekilis(cfg)
    hatlar = [g for g in HATLAR if g == "yerel" or all(g in V.hat[k] for k in kollar)]
    D = {k: {g: deng(V.P(k, g), cfg, W, cnt) for g in hatlar} for k in kollar}
    # N / S alt kümeleri (aynı W sütunları)
    kumeN = np.array([bool(V.N[np.where(cfg["cl"] == i)[0][0]]) for i in range(cfg["L"])])
    alt = {}
    for ad, mk in (("N", kumeN), ("S", ~kumeN)):
        c_ = cfg_alt(cfg, mk)
        W_ = W[:, mk]
        alt[ad] = {"cfg": c_, "D": {k: {g: deng(V.P(k, g), c_, W_, cnt) for g in hatlar} for k in kollar}}
    # S8: seed sabit
    bir = np.ones_like(cnt)
    D8 = {k: {g: deng(V.P(k, g), cfg, W, bir) for g in ("yerel", "hat")} for k in ("h10", "h02c")}

    def dk(Dx, k, g):            # kol k'nın hat g'deki dengeli değişimi (puan): boot, nokta
        return 100 * (Dx[k][g][0] - Dx[k]["yerel"][0]), 100 * (Dx[k][g][1].mean() - Dx[k]["yerel"][1].mean())

    def dhat(Dx, a, b, g="hat"):  # Δ_g(a − b) − Δ_yerel(a − b)
        bo = 100 * ((Dx[a][g][0] - Dx[b][g][0]) - (Dx[a]["yerel"][0] - Dx[b]["yerel"][0]))
        no = 100 * ((Dx[a][g][1].mean() - Dx[b][g][1].mean()) - (Dx[a]["yerel"][1].mean() - Dx[b]["yerel"][1].mean()))
        return A.ozet(no, bo)

    def dfark(Dx, a, b, g):      # Δ_deng(a − b | g)
        return A.ozet(100 * (Dx[a][g][1].mean() - Dx[b][g][1].mean()), 100 * (Dx[a][g][0] - Dx[b][g][0]))

    R = {"cfg": {"n": cfg["n"], "lezyon": cfg["L"], "sinif_goruntu": np.bincount(cfg["y"], minlength=NK).tolist(),
                 "sinif_lezyon": np.bincount(cfg["clcls"], minlength=NK).tolist(), "kollar": list(kollar), "hatlar": hatlar,
                 "N_goruntu": alt["N"]["cfg"]["n"], "N_lezyon": alt["N"]["cfg"]["L"],
                 "N_sinif_lezyon": np.bincount(alt["N"]["cfg"]["clcls"], minlength=NK).tolist(),
                 "S_goruntu": alt["S"]["cfg"]["n"], "S_lezyon": alt["S"]["cfg"]["L"], "B": B_BOOT, "rng": RNG}}
    if "h10" not in kollar or "h02c" not in kollar:
        dur("birincil kollar (H10, H02c) eksik")
    bir_ = dhat(D, "h10", "h02c")
    bir_["etiket"] = etiket(bir_["nokta"], bir_["ga"])
    R["birincil"] = {"delta_hat": bir_, "delta_yerel": dfark(D, "h10", "h02c", "yerel"), "delta_bcn_hatti": dfark(D, "h10", "h02c", "hat")}
    # S1 kol başına D_k + seed başına dengeli
    R["S1"] = {k: {"D": A.ozet(*dk(D, k, "hat")[::-1]),
                   "dengeli_yerel": 100 * float(D[k]["yerel"][1].mean()), "dengeli_hat": 100 * float(D[k]["hat"][1].mean()),
                   "duyarlilik_yerel": dict(zip(A.SINIF, (100 * D[k]["yerel"][2]).tolist())),
                   "duyarlilik_hat": dict(zip(A.SINIF, (100 * D[k]["hat"][2]).tolist()))} for k in kollar}
    # S2 nokta ikinciller (etiketsiz olanlar + ACC)
    S2 = {}
    for k in kollar:
        Py, Ph = V.P(k, "yerel"), V.P(k, "hat")
        ty, th = Py.argmax(2), Ph.argmax(2)
        uy = (ty == th).mean(1)
        dp = np.abs(Ph.astype(np.float64) - Py.astype(np.float64))
        mat = np.mean([np.bincount(ty[s] * NK + th[s], minlength=NK * NK).reshape(NK, NK) for s in range(len(ty))], 0)
        S2[k] = {"acc_yerel": 100 * float((ty == V.y[None]).mean()), "acc_hat": 100 * float((th == V.y[None]).mean()),
                 "acc_degisim": 100 * float((th == V.y[None]).mean() - (ty == V.y[None]).mean()),
                 "nv_pay_yerel": float((ty == NV).mean()), "nv_pay_hat": float((th == NV).mean()),
                 "nv_pay_degisim": float((th == NV).mean() - (ty == NV).mean()),
                 "uyum_ort": float(uy.mean()), "uyum_min": float(uy.min()), "uyum_seed": uy.tolist(),
                 "p_maks_fark": float(dp.max()), "p_maks_fark_seed": dp.max((1, 2)).tolist(), "p_ort_fark": float(dp.mean()),
                 "donme_matrisi_seed_ort": mat.tolist(), "donme_satir_toplam": (mat.sum(1) - np.diag(mat)).tolist(),
                 "donme_nv_sutunu": (mat[:, NV] - np.eye(NK)[NV] * mat[NV, NV]).tolist(), "donme_toplam": float(mat.sum() - np.trace(mat))}
    R["S2"] = S2
    # S3
    R["S3"] = {}
    for a, b in (("h10", "h08c"), ("h11", "h02c")):
        if a in kollar and b in kollar:
            R["S3"][f"{a}-{b}"] = dhat(D, a, b)
    # S4 N / S
    R["S4"] = {}
    for ad in ("N", "S"):
        Dx = alt[ad]["D"]
        o = dhat(Dx, "h10", "h02c")
        o["etiket"] = etiket(o["nokta"], o["ga"])
        R["S4"][ad] = {"delta_hat": o, "n": alt[ad]["cfg"]["n"], "lezyon": alt[ad]["cfg"]["L"],
                       "bos_sinif": [A.SINIF[c] for c in range(NK) if not (alt[ad]["cfg"]["clcls"] == c).any()],
                       "D": {k: A.ozet(*dk(Dx, k, "hat")[::-1]) for k in kollar}}
    # S5–S7 duyarlılık kolları
    R["S5_S7"] = {}
    for g in ("jpeg", "z125", "z080"):
        if g in hatlar:
            o = dhat(D, "h10", "h02c", g)
            o["etiket_araligi"] = etiket(o["nokta"], o["ga"])
            R["S5_S7"][g] = {"delta": o, "D": {k: A.ozet(*dk(D, k, g)[::-1]) for k in kollar},
                             "uyum": {k: float((V.P(k, "yerel").argmax(2) == V.P(k, g).argmax(2)).mean()) for k in kollar}}
    # S8 seed sabit
    b8 = 100 * ((D8["h10"]["hat"][0] - D8["h02c"]["hat"][0]) - (D8["h10"]["yerel"][0] - D8["h02c"]["yerel"][0]))
    R["S8"] = A.ozet(bir_["nokta"], b8)
    # S9 piksel betimi
    R["S9"] = {"rmse": (beklenen or {}).get("rmse"), "bcn_eksi_dermamnist_kanal": [round(x, 2) for x in BCN_DM_KANAL],
               "kaynak_kanal": "sonuclar/e2_incele_ek/v3_model.py mu_b − mu_d (v1 piksel istatistikleri)"}
    # S10 kalibrasyonla bağ
    v3 = json.loads(V3_JSON.read_text(encoding="utf-8"))
    r18 = [k for k in KL.KOLLAR if KL.KOLLAR[k]["tur"] == "r18"]
    kal_u = sum(v3["kal"][f"{k}/s{s}"]["uyum"] for k in r18 for s in SEEDLER)
    kal_n = 8 * len(r18) * len(SEEDLER)
    r18_val = [k for k in ("h02c", "h08c") if k in kollar]
    don_val = float(np.mean([1 - S2[k]["uyum_ort"] for k in r18_val]))
    don_kal = 1 - kal_u / kal_n
    R["S10"] = {"r18_donme_val": don_val, "r18_donme_kalibrasyon": don_kal, "kalibrasyon": f"{kal_n - kal_u}/{kal_n}",
                "oran": don_val / don_kal if don_kal > 0 else None, "vit_donme_val": float(1 - S2["h10"]["uyum_ort"])}
    # İNCELE (i)–(iv)
    inc = {"i_delta_hat_gt_8": bir_["nokta"] > INC_UST, "ii_delta_hat_lt_-6": bir_["nokta"] < INC_ALT,
           "iii_vit_D_lt_-5": any(R["S1"][k]["D"]["nokta"] < INC_VIT_D for k in ("h10", "h11") if k in kollar),
           "iv_r18_uyum_lt_0_90": any(S2[k]["uyum_ort"] < INC_R18_UYUM for k in r18_val)}
    R["gorulmemis"] = {**{k: bool(v) for k, v in inc.items()}, "INCELE": bool(any(inc.values()))}
    R["beklenti_icinde"] = beklenti_tablosu(R, beklenen)
    R["cumleler"] = cumleler(R, beklenen)
    return R


def beklenti_tablosu(R: dict, beklenen: dict | None) -> list:
    S1, S2 = R["S1"], R["S2"]
    sat = []

    def ek(ad, x, alt, ust):
        sat.append({"nicelik": ad, "olculen": x, "aralik": [alt, ust], "icinde": None if x is None else bool(alt <= x <= ust)})
    ek("RMSE medyanı (hat ↔ yerel)", ((beklenen or {}).get("rmse") or {}).get("hat", {}).get("medyan"), 1.3, 2.5)
    for k in ("h02c", "h08c"):
        if k in S2:
            ek(f"argmax uyumu {k}", S2[k]["uyum_ort"], 0.94, 0.995)
            ek(f"maks|Δp| {k}", S2[k]["p_maks_fark"], 0.4, 0.99)
            ek(f"D_k {k}", S1[k]["D"]["nokta"], -4.0, 1.0)
            ek(f"nv payı değişimi {k}", S2[k]["nv_pay_degisim"], -0.01, 0.03)
    for k, (a, b), (da, db) in (("h10", (0.985, 1.0), (-1.5, 0.5)), ("h11", (0.98, 1.0), (-2.0, 0.5))):
        if k in S2:
            ek(f"argmax uyumu {k}", S2[k]["uyum_ort"], a, b)
            ek(f"maks|Δp| {k}", S2[k]["p_maks_fark"], 0.03, 0.5)
            ek(f"D_k {k}", S1[k]["D"]["nokta"], da, db)
            ek(f"nv payı değişimi {k}", S2[k]["nv_pay_degisim"], -0.01, 0.01)
    ek("Δ_hat (birincil)", R["birincil"]["delta_hat"]["nokta"], -1.5, 4.0)
    ek("Δ_hat GA yarı genişliği", R["birincil"]["delta_hat"]["yari_genislik"], 1.0, 2.5)
    ek("S4 Δ_hat (N)", R["S4"]["N"]["delta_hat"]["nokta"], -2.0, 5.0)
    if "jpeg" in R["S5_S7"]:
        ek("S5 Δ (JPEG)", R["S5_S7"]["jpeg"]["delta"]["nokta"], -1.5, 4.0)
    for g in ("z125", "z080"):
        if g in R["S5_S7"]:
            ek(f"S6/S7 Δ ({g})", R["S5_S7"][g]["delta"]["nokta"], -3.0, 8.0)
    ek("S10 dönme oranı (görülmemiş / eğitim)", R["S10"]["oran"], 1.0, 4.0)
    return sat


def cumleler(R: dict, beklenen: dict | None) -> dict:
    if R["gorulmemis"]["INCELE"]:
        return {"not": "Görülmemiş (İNCELE) bayrağı açık → soruşturma kapanana dek sayı yazılmaz, cümle seçilmez (kart §eşik)."}
    b = R["birincil"]["delta_hat"]
    n = R["cfg"]["n"]
    r = ((beklenen or {}).get("rmse") or {}).get("hat", {}).get("medyan")
    giris = ("Dış farkın bir bölümünün BCN20000 görüntülerini hazırlarken kullandığımız ön-işleme hattından gelip gelmediğini "
             f"sınamak için, resmî doğrulama kümesinin {f'{n:,}'.replace(',', '.')} görüntüsünü HAM10000 özgün dosyalarından aynı "
             "kodla yeniden ürettik. Aynı kolları, yeniden eğitmeden, bu girdide ve MedMNIST+ girdisinde karşılaştırdık. Analiz "
             "hakem önerisiyle, E2 sonuçları görüldükten sonra tasarlandı; test kümesi ve BCN etiketleri açılmadı. Piksel farkı "
             f"küçüktü (görüntü başına RMSE medyanı {tr(r, isaret=False) if r is not None else '—'} gri düzey).")

    def etiket_cumlesi(o):
        e, d, g = o["etiket"], o["nokta"], ga_tr(o["ga"])
        if e == "DIŞLANDI":
            return (f"Ön-işleme hattı DINOv2-B ile ResNet-18 arasındaki dengeli doğruluk farkını {tr(d)} puan {g} değiştirdi. "
                    "Bu aralık, dış farkın kendi belirsizliğinden (±2,48 puan) küçüktür. Dış farkın (+16,34) ön-işleme hattından "
                    "kaynaklanması bu ölçekte dışlanır. Hat etkisi HAM görüntülerinde ölçüldü; BCN'nin kaymış görüntülerindeki "
                    "etkileşimi ölçülmedi.")
        if e == "KÜÇÜK-BELİRSİZ":
            return (f"Ön-işleme hattının farka etkisi nokta kestirimde küçüktü: {tr(d)} puan {g}. Ancak aralık ±2,48 puanlık "
                    "eşdeğerlik bandını aştığından, hattın dış farkın birkaç puanını açıklaması bu veriyle dışlanamadı. Dış farkın "
                    "büyük bölümü hat etkisinin aralığının dışında kalır.")
        if e == "KISMEN":
            return (f"Ön-işleme hattı farkı DINOv2-B lehine {tr(d)} puan {g} büyüttü. Toplamsal bir okumayla bu, dış farkın "
                    f"(+16,34) yaklaşık %{100 * d / B1_DIS:.0f}'ına karşılık gelir. Dış fark iç kestirimin (+6,03) üstünde kalır, "
                    "ama büyüklüğü bu oranda temkinle okunmalıdır. Hat etkisi HAM görüntülerinde ölçüldü.")
        if e == "YORUM DEĞİŞİR":
            return (f"Ön-işleme hattı farkı DINOv2-B lehine {tr(d)} puan {g} büyüttü. Bu, dış farkın iç kestirimi aşan kısmından "
                    "(10,31 puan) büyüktür. Dış merkezdeki 'büyüme' bu nedenle merkez kaymasına değil, kısmen ya da tamamen "
                    "ResNet-18'in ön-işleme farkına duyarlılığına bağlanabilir.")
        return (f"Ön-işleme hattı farkı ResNet-18 lehine {tr(abs(d), isaret=False)} puan {g} kaydırdı. Hat kaynaklı bir şişme "
                "yoktur; dış fark bu açıdan tutucu bir kestirimdir.")

    C = {"giris": giris, "etiket": b["etiket"], "birincil": etiket_cumlesi(b)}
    if b["etiket"] == "YORUM DEĞİŞİR":
        C["kullanici_karari"] = "Başlık ve özet kararı kullanıcıya gider (kart)."
    S2, S1 = R["S2"], R["S1"]
    r18 = [k for k in ("h02c", "h08c") if k in S2]
    u_r18 = float(np.mean([S2[k]["uyum_ort"] for k in r18]))
    C["S1_S2"] = (f"ResNet-18 kollarının tahminleri görüntülerin %{yuzde(1 - u_r18)}'ünde değişti (DINOv2-B'de "
                  f"%{yuzde(1 - S2['h10']['uyum_ort'])}); dengeli doğruluk değişimleri H02c {tr(S1['h02c']['D']['nokta'])}, "
                  f"H10 {tr(S1['h10']['D']['nokta'])} puandı.")
    if R["S10"]["oran"] is not None:
        C["S10"] = (f"Bu oran, daha önce yalnız eğitim görüntülerinde yapılan sınamadakinin (%1,9) "
                    f"{tr(R['S10']['oran'], 1, isaret=False)} katıdır.")
    N = R["S4"]["N"]["delta_hat"]
    if "TERS" not in (N["etiket"], b["etiket"]) and SIDDET[N["etiket"]] > SIDDET[b["etiket"]]:
        c = etiket_cumlesi(N)
        C["S4_N"] = f"Görülmemiş lezyonlarda ({R['S4']['N']['n']:,} görüntü) ".replace(",", ".") + c[0].lower() + c[1:]
    C["S4_N_etiketi"] = N["etiket"]
    for g, o in R["S5_S7"].items():
        if o["delta"]["nokta"] >= M_ESIK:
            C[f"duyarlilik_{g}"] = f"{BILESEN[g]} tek başına Δ_hat'i {tr(o['delta']['nokta'])} puana çıkarıyor; keşifsel."
    C["sinirlilik"] = ("Ön-işleme duyarlılığı doğrulama kümesinde ölçüldü. Bu küme epoch seçiminde kullanıldığı için tamamen "
                       "bağımsız değildir. Ölçülen, yazılım hattının etkisidir; dermatoskop, büyütme ve görüş alanı gibi edinim "
                       "farkları merkez kaymasının parçası sayıldı ve burada ayrıştırılmadı.")
    return C


# ───────────────────────── rapor ─────────────────────────
def rapor_md(R: dict) -> str:
    b = R["birincil"]["delta_hat"]
    L = [f"# H16 / E3 — ön-işleme hattı denetimi{' (KURU VEKİL — SONUÇ DEĞİL)' if R.get('kuru') else ''}",
         f"*{R['tarih']}* · `scripts/e3_onisleme_analiz.py` (SHA256 `{R['betik_sha256'][:16]}…`) · kart SHA256 "
         f"`{R['kart_sha256'][:16]}…` ({'onaylı' if R['kart_onayli'] else 'ONAYLI DEĞİL'}) · keşifsel, sonradan · "
         f"bootstrap B = {B_BOOT}, default_rng({RNG})", "",
         f"**K-d kuru koşu:** {'GEÇTİ' if R['kuru_kosu']['gecti'] else 'DUR'} · kaynak {R['kuru_kosu']['kaynak']}", "",
         "| kol | yerel dengeli (bu) | v3 orijinal | fark (puan) |", "|---|---|---|---|"]
    L += [f"| {k} | {v['bu_puan']:.3f} | {v['v3_puan']:.3f} | {v['fark']:+.4f} |" for k, v in R["kuru_kosu"]["kol"].items()]
    kd = R["kuru_kosu"]
    L += ["", f"Δ_yerel(H10 − H02c) = {kd['delta_yerel']['bu']:+.3f} (v3 {kd['delta_yerel']['v3']:+.3f}; tol ±{KD_TOL}) · öz-sınama "
          f"hat := yerel → Δ_hat {kd['oz_sinama']['nokta']:+.2e}, GA {kd['oz_sinama']['ga']} · eşikler m = "
          f"{kd['esikler']['m_kaynak']:.4f} (sabit 2,48), G = {kd['esikler']['G_kaynak']:.4f} (sabit 10,31) → "
          f"{'tuttu' if kd['esikler']['gecti'] else 'TUTMADI'}", ""]
    L += [f"## Birincil: Δ_hat = {tr(b['nokta'])} puan {ga_tr(b['ga'])} (p {p_tr(b['p'])}) → **{b['etiket']}**",
          f"n = {R['cfg']['n']} görüntü, {R['cfg']['lezyon']} lezyon kümesi · Δ_deng(H10 − H02c): yerel "
          f"{tr(R['birincil']['delta_yerel']['nokta'])} {ga_tr(R['birincil']['delta_yerel']['ga'])}, BCN hattı "
          f"{tr(R['birincil']['delta_bcn_hatti']['nokta'])} {ga_tr(R['birincil']['delta_bcn_hatti']['ga'])}", "",
          "## S1–S2 (kol başına; keşifsel, düzeltmesiz)", "",
          "| kol | D_k (puan) [GA] | dengeli yerel → hat | ACC Δ | nv payı Δ | uyum ort (min) | maks\\|Δp\\| | dönme |",
          "|---|---|---|---|---|---|---|---|"]
    for k in R["cfg"]["kollar"]:
        s1, s2 = R["S1"][k], R["S2"][k]
        L.append(f"| {k} | {tr(s1['D']['nokta'])} {ga_tr(s1['D']['ga'])} | {s1['dengeli_yerel']:.2f} → {s1['dengeli_hat']:.2f} | "
                 f"{tr(s2['acc_degisim'])} | {s2['nv_pay_degisim']:+.4f} | {s2['uyum_ort']:.4f} ({s2['uyum_min']:.4f}) | "
                 f"{s2['p_maks_fark']:.3f} | {s2['donme_toplam']:.1f} |")
    L += ["", "## S3 / S4 / S5–S7 / S8", ""]
    for ad, o in R["S3"].items():
        L.append(f"- S3 {ad}: Δ_hat {tr(o['nokta'])} {ga_tr(o['ga'])}")
    for ad in ("N", "S"):
        o = R["S4"][ad]
        L.append(f"- S4 {ad} ({o['n']} görüntü, {o['lezyon']} lezyon; boş sınıf {o['bos_sinif']}): Δ_hat {tr(o['delta_hat']['nokta'])} "
                 f"{ga_tr(o['delta_hat']['ga'])} → {o['delta_hat']['etiket']}")
    for g, o in R["S5_S7"].items():
        L.append(f"- {g}: Δ {tr(o['delta']['nokta'])} {ga_tr(o['delta']['ga'])} (aralık etiketi {o['delta']['etiket_araligi']}) · "
                 + ", ".join(f"D_{k} {tr(v['nokta'])}" for k, v in o["D"].items()))
    L.append(f"- S8 seed sabit: {tr(R['S8']['nokta'])} {ga_tr(R['S8']['ga'])}")
    L.append(f"- S10: R18 dönme val {yuzde(R['S10']['r18_donme_val'], 2)}% / kalibrasyon {R['S10']['kalibrasyon']} "
             f"({yuzde(R['S10']['r18_donme_kalibrasyon'], 2)}%) → oran {R['S10']['oran'] and round(R['S10']['oran'], 2)}")
    L += ["", f"## İNCELE: {'AÇIK' if R['gorulmemis']['INCELE'] else 'kapalı'} · {R['gorulmemis']}", "",
          "## Beklenti içinde mi (betimsel)", "", "| nicelik | ölçülen | aralık | içinde |", "|---|---|---|---|"]
    L += [f"| {s['nicelik']} | {s['olculen'] if s['olculen'] is None else round(s['olculen'], 4)} | {s['aralik']} | {s['icinde']} |"
          for s in R["beklenti_icinde"]]
    L += ["", "## Önceden yazılmış cümleler (mekanik)", ""]
    L += [f"- **{k}:** {v}" for k, v in R["cumleler"].items()]
    if R.get("ikinci_acilis"):
        L += ["", f"**E3 analizi ikinci kez çalıştı; neden: {R['ikinci_acilis']}**"]
    return "\n".join(L) + "\n"


# ───────────────────────── kaynaklar ─────────────────────────
def kuru_veri() -> tuple:
    """Arşiv (runs_colab) tahminleri: yerel := tahmin_val.npy; vekil hat kolları := tahmin_val_son / tahmin_val_aucsecim."""
    arsiv = WS / "runs_colab"
    yerel, hat = {}, {}
    y = val_etiket()
    for k in KOLLAR:
        Ps, Pn, Pa = [], [], []
        for s in SEEDLER:
            d = arsiv / KL.run_id(k, s)
            if not np.array_equal(np.load(d / "etiket_val.npy").reshape(-1), y):
                dur(f"kuru: {d.name} etiket_val ≠ val_labels")
            Ps.append(np.load(d / "tahmin_val.npy"))
            Pn.append(np.load(d / "tahmin_val_son.npy"))
            Pa.append(np.load(d / "tahmin_val_aucsecim.npy"))
        yerel[k] = np.stack(Ps).astype(np.float64)
        hat[k] = (np.stack(Pn), np.stack(Pa))
    return yerel, hat


def gercek_kosumlar(runs: Path, B: dict) -> tuple:
    """a3: 4 kol × 5 seed. Dönüş (yerel_tam, kayit). Hat tahminleri YÜKLENMEZ (K-d öncesi)."""
    yerel, kayit, durum, sorun = {}, {}, {}, []
    cik_sha = A.sha(CIKARIM)
    bek_sha = A.sha(BEKLENEN)
    for k in KOLLAR:
        Ps, kos = [], []
        for s in SEEDLER:
            aday = []
            for d in sorted(Path(runs).glob(f"*_e3_{k}_s{s}")):
                try:
                    c = json.loads((d / "config.json").read_text(encoding="utf-8"))
                    j = json.loads((d / ".done").read_text(encoding="utf-8"))
                    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    continue
                if c.get("varyant") == f"e3_{k}" and str(c.get("seed")) == str(s) and str(c.get("kuyruk", "")).startswith(KUYRUK) \
                        and j.get("rc") == 0:
                    aday.append((d, c, m))
            if len(aday) != 1:
                sorun.append(f"{k}/s{s}: rc 0 a3 koşumu {len(aday)} adet (1 olmalı)")
                continue
            d, c, m = aday[0]
            e3 = c.get("e3") or {}
            kontrol = {"kaynak_run": e3.get("kaynak_run") == KL.run_id(k, s), "betik": e3.get("betik_sha256") == cik_sha,
                       "beklenen": e3.get("beklenen_sha256") == bek_sha, "girdi_sha": m.get("hat_girdi_sha_eslesti") == 1,
                       "sira_sha": m.get("hat_sira_sha_eslesti") == 1, "gecis4": m.get("hat_gecis_sayisi") == 4,
                       "K-g": all(m.get(x) == 0 for x in ("c1_kol", "bcn_okundu", "resmi_test_okundu", "hat_etiketle_degerlendirildi")),
                       "K-e": m.get("hat_tahmin_sonlu") == 1 and m.get("hat_olasilik_toplam_maks_sapma", 1) <= 1e-4
                       and m.get("hat_tahmin_sinif_maks_pay", 1) <= 0.95, "val_n": m.get("val_n") == 1003,
                       "hat_index": BV.icerik_sha256(np.load(d / "hat_index.npy")) == B["icerik"]["index"]}
            dk = {1: "A", 2: "B", 3: "C"}.get(m.get("ozdeslik_durum_kod"), "C")
            durum[f"{k}/s{s}"] = dk
            if not all(kontrol.values()):
                sorun.append(f"{k}/s{s} ({d.name}): {[x for x, v in kontrol.items() if not v]}")
            Ps.append(np.load(d / "tahmin_val_yerel.npy").astype(np.float64))
            kos.append({"run": d.name, "durum": dk, "sure_sn": json.loads((d / ".done").read_text(encoding="utf-8")).get("sure_sn")})
        if len(Ps) == len(SEEDLER):
            yerel[k] = np.stack(Ps)
        kayit[k] = kos
    if sorun:
        dur("a3 denetimleri: " + " · ".join(sorun))
    C = [x for x, v in durum.items() if v == "C"]
    if any(x.split("/")[0] in ("h10", "h02c") for x in C):
        dur(f"K-c: birincil kolda Durum C {C}")
    disari = sorted({x.split("/")[0] for x in C})
    butce = 0.0
    for d in Path(runs).glob("*_e3_*"):
        try:
            c = json.loads((d / "config.json").read_text(encoding="utf-8"))
            if str(c.get("kuyruk", "")).startswith(KUYRUK):
                butce += float(json.loads((d / ".done").read_text(encoding="utf-8")).get("sure_sn") or 0)
        except Exception:  # noqa: BLE001
            continue
    if butce / 3600 > BUTCE_SA:
        dur(f"K-f: e3 koşumları {butce / 3600:.2f} GPU-sa > {BUTCE_SA}")
    return yerel, {"kosumlar": kayit, "durum": durum, "durum_ozet": dict(Counter(durum.values())), "disari_kol": disari,
                   "butce_gpu_sa": butce / 3600}


def hat_yukle(runs: Path, kayit: dict, kollar) -> dict:
    hat = {}
    for k in kollar:
        hat[k] = {}
        for g in HATLAR[1:]:
            hat[k][g] = np.stack([np.load(Path(runs) / r["run"] / f"tahmin_val_{g}.npy").astype(np.float64) for r in kayit[k]])
    return hat


def kilit_sha() -> dict:
    return {"kart": A.sha(KART), "analiz": A.sha(Path(__file__)), "beklenti": A.sha(BEKLENTI), "beklenen": A.sha(BEKLENEN),
            "e3_cikarim": A.sha(CIKARIM), "e3_ham_val_hazirla": A.sha(HAZIRLA), "e2_bcn_analiz": A.sha(WS / "scripts" / "e2_bcn_analiz.py"),
            "kollar": A.sha(WS / "varyantlar" / "e2" / "kollar.py")}


def yaz(R: dict, out: Path):
    temiz = A.jsonla(R)
    out.write_text(json.dumps(temiz, ensure_ascii=False, indent=1), encoding="utf-8")
    md = rapor_md(temiz)
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    out.with_name(out.name + ".sha256").write_text(f"{A.sha(out)}  {out.name}\n", encoding="utf-8")
    print(md)
    print(f"yazıldı: {out} (+ .md, .sha256)")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kuru", action="store_true")
    ap.add_argument("--kilit-ac", action="store_true")
    ap.add_argument("--runs", default=str(WS / "runs"))
    ap.add_argument("--kilit", default=str(KILIT))
    ap.add_argument("--ikinci-acilis", default=None)
    ap.add_argument("--cikti", default=None)
    ap.add_argument("--beklenen", default=None, help="yalnız kod sınaması: vekil beklenen dosyası")
    a = ap.parse_args()
    global BEKLENEN
    if a.beklenen:
        BEKLENEN = Path(a.beklenen)
    kart_sha = A.sha(KART)
    temel = {"tarih": datetime.now().isoformat(timespec="seconds"), "betik_sha256": A.sha(Path(__file__)), "kart_sha256": kart_sha,
             "kart_onayli": kart_sha == ONAYLI_KART_SHA, "kuru": bool(a.kuru)}
    if not temel["kart_onayli"]:
        dur(f"H16.md SHA256 {kart_sha[:12]} ≠ onaylı {ONAYLI_KART_SHA[:12]}")
    if a.kuru:
        yerel, vekil = kuru_veri()
        V = Veri(yerel, None, np.arange(1003), "arşiv tahmin_val.npy (runs_colab; yerel kol vekili)")
        kd = kuru_kosu(V, KOLLAR)
        print(f"[e3] K-d kuru koşu: {'GEÇTİ' if kd['gecti'] else 'DUR'} · " + ", ".join(
            f"{k} {v['bu_puan']:.3f}/{v['v3_puan']:.3f}" for k, v in kd["kol"].items())
            + f" · Δ_yerel {kd['delta_yerel']['bu']:+.3f} (v3 {kd['delta_yerel']['v3']:+.3f}) · öz-sınama {kd['oz_sinama']}", flush=True)
        if not kd["gecti"]:
            dur("K-d kuru koşu tutmadı")
        # uçtan uca KOD sınaması: vekil hat kolları, 2 görüntü yapay "eksik"
        idx = np.setdiff1d(np.arange(1003), [5, 500])
        hat = {k: {"hat": vekil[k][0][:, idx], "jpeg": vekil[k][1][:, idx], "z125": vekil[k][0][:, idx], "z080": vekil[k][1][:, idx]}
               for k in KOLLAR}
        Vv = Veri(yerel, hat, idx, "VEKİL: hat := tahmin_val_son, jpeg := tahmin_val_aucsecim (sonuç DEĞİL)")
        beklenen = json.loads(BEKLENEN.read_text(encoding="utf-8")) if BEKLENEN.exists() else None
        R = analiz(Vv, KOLLAR, beklenen)
        R.update(temel, kuru_kosu=kd, vekil_not="Hat kolları VEKİLDİR (arşiv son-epoch/AUC-seçim tahminleri); sayılar sonuç değildir.")
        yaz(R, Path(a.cikti) if a.cikti else WS / "sonuclar" / "e3_onisleme_kuru_sonuc.json")
        return 0
    B = json.loads(BEKLENEN.read_text(encoding="utf-8")) if BEKLENEN.exists() else dur(f"{BEKLENEN} yok (a0)")
    yerel, den = gercek_kosumlar(Path(a.runs), B)
    kollar = tuple(k for k in KOLLAR if k not in den["disari_kol"])
    V = Veri(yerel, None, np.arange(1003), "a3 yerel kol (tahmin_val_yerel.npy)")
    kd = kuru_kosu(V, kollar)
    print(f"[e3] K-d: {'GEÇTİ' if kd['gecti'] else 'DUR'} · Δ_yerel {kd['delta_yerel']['bu']:+.3f} · öz-sınama {kd['oz_sinama']['gecti']}")
    if not kd["gecti"]:
        dur("K-d kuru koşu tutmadı — hat tahminleri okunmaz")
    kp = Path(a.kilit)
    if a.kilit_ac:
        yeni = {"kart": "H16 (E3)", "sha256": kilit_sha(), "kosumlar": den["kosumlar"], "durum_ozet": den["durum_ozet"],
                "disari_kol": den["disari_kol"], "butce_gpu_sa": den["butce_gpu_sa"],
                "kuru": {"gecti": kd["gecti"], "delta_yerel": kd["delta_yerel"], "kol": {k: v["fark"] for k, v in kd["kol"].items()}},
                "ikinci_acilis": None}
        if kp.exists():
            eski = json.loads(kp.read_text(encoding="utf-8"))
            if eski.get("sha256") != yeni["sha256"] or eski.get("kosumlar") != A.jsonla(yeni["kosumlar"]) or eski.get("analiz_zamani"):
                dur("KİLİT zaten var ve içeriği farklı / analiz yapılmış — ikinci açılış yalnız kullanıcı kararıyla")
            print(f"KİLİT zaten açık ve aynı içerik ({eski.get('acilis_zamani')})")
            return 0
        yeni["acilis_zamani"] = datetime.now().isoformat(timespec="seconds")
        kp.write_text(json.dumps(A.jsonla(yeni), ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"KİLİT AÇILDI: {kp} · Durum {den['durum_ozet']} · bütçe {den['butce_gpu_sa']:.2f} GPU-sa · dışarıda {den['disari_kol']}")
        return 0
    # ── gerçek analiz (BİR KEZ) ──
    if not kp.exists():
        dur(f"{kp} yok — önce --kilit-ac")
    Kj = json.loads(kp.read_text(encoding="utf-8"))
    if Kj.get("analiz_zamani") and not a.ikinci_acilis:
        dur(f"E3 analizi zaten {Kj['analiz_zamani']} tarihinde yapıldı — yalnız --ikinci-acilis '<neden>'")
    fark = {k: v[:12] for k, v in kilit_sha().items() if Kj.get("sha256", {}).get(k) != v}
    if fark:
        dur(f"KİLİT SHA256'ları güncel dosyalarla aynı değil: {sorted(fark)}")
    if A.jsonla(den["kosumlar"]) != Kj.get("kosumlar"):
        dur("KİLİT koşum listesi güncel a3 koşumlarıyla aynı değil")
    out = Path(a.cikti) if a.cikti else WS / "sonuclar" / "e3_onisleme_sonuc.json"
    if out.exists() and not a.ikinci_acilis:
        dur(f"{out.name} zaten var — yalnız --ikinci-acilis")
    if a.ikinci_acilis and out.exists():
        out = out.with_name(out.stem + f"_ikinci_{datetime.now():%Y%m%d-%H%M%S}.json")
    Kj.setdefault("analiz_acilislari", []).append({"zaman": datetime.now().isoformat(timespec="seconds"), "ikinci_acilis": a.ikinci_acilis})
    Kj["analiz_zamani"] = Kj.get("analiz_zamani") or datetime.now().isoformat(timespec="seconds")
    kp.write_text(json.dumps(Kj, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[e3] K-d ve KİLİT tamam → hat tahminleri AÇILIYOR", flush=True)
    hat = hat_yukle(Path(a.runs), den["kosumlar"], kollar)
    Vh = Veri(yerel, hat, np.load(Path(a.runs) / den["kosumlar"]["h10"][0]["run"] / "hat_index.npy"), "a3 (yerel + 4 hat)")
    R = analiz(Vh, kollar, B)
    R.update(temel, kuru_kosu=kd, a3_denetim=den, ikinci_acilis=a.ikinci_acilis, beklenen_ozet={
        "n": B.get("n"), "eksik": B.get("eksik"), "q_BCN": B.get("q_BCN"), "subsampling": B.get("subsampling"),
        "tani": B.get("tani"), "lisans": B.get("lisans"), "pillow": B.get("pillow")})
    yaz(R, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
