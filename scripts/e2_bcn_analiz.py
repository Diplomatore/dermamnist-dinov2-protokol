# -*- coding: utf-8 -*-
"""
e2_bcn_analiz.py — H15 / E2 (BCN20000 dış merkez) ANALİZİ (hipotezler/H15.md §İstatistik). CPU; TUT/ÖLDÜR VERMEZ, B1/B2
etiketlerini ve önceden yazılmış cümleleri MEKANİK seçer. K5 kararını, K5 Holm ailelerini, k5_*/e1_* sonuçlarını DEĞİŞTİRMEZ.

Sıra (gerçek analiz, kart a3 sonu — BİR KEZ):
 0) `scripts/k5_test.py` SHA256'sı donmuş değerle aynı (metrik/bootstrap işlevleri ondan içe aktarılır); kartın SHA256'sı
    onaylı değerle aynı; `sonuclar/e2_bcn.KILIT` var ve içindeki SHA256'lar (kart, bu betik, beklenti, beklenen, e2_cikarim,
    kopyasız liste, lezyon kümeleri) güncel dosyalarla aynı; KİLİT'te `analiz_zamani` yok (yoksa yalnız --ikinci-acilis).
 1) K-d KURU KOŞU (BCN açılmadan): arşiv tahminleriyle iç değerler k5_test şemasıyla (B = 2.000, rng 0) yeniden üretilir —
    H10 − H02c resmî ACC +2,26 [+1,15; +3,30], dengeli +6,03 [+2,86; +9,37]; C1 +1,12 [−0,05; +2,30], dengeli +5,74
    [+1,21; +10,17]; C0 +0,28 / dengeli +4,45; H10 − H08c resmî −0,36, C1 −0,96. Nokta ve GA uçlarında |fark| > 0,005 puan → DUR.
 2) 55 a3 koşumu (`e2_<kol>-bcn`, kuyruk e2-degerlendirme*): rc 0, Durum A/B (birincil kolda C → DUR), dis_kilit_var,
    bcn_ikinci_cikarim, girdi SHA, kaynak run_id = kollar.py, sıra SHA256; K-g bütçe (Σ koşum ≤ 3,0 L4-sa); K-e dejenere çıktı
    (etikete bakmadan; tek sınıf payı > 0,95 ya da sonlu değil → DUR).
 3) ANCAK BUNDAN SONRA etiketler okunur (data/bcn20000/meta/bcn_20k_train.csv, MD5 = figshare). E1 eşlemesi (AK→akiec, SCC
    dışla) + kopyasız liste (49 çıkarma) + lezyon kümeleri (2 birleştirme).
 4) B1 (Δ_dış,R dengeli, H10 − H02c) ve B2 (D_π = |e_R| − |e_C1|, ACC_π) — Holm m = 2, iki yönlü. İki düzeyli eşleştirilmiş
    bootstrap B = 10.000, numpy.random.default_rng(0), çekiliş sırası: (1) 5 seed indeksi iadeli (bütün kollara, iç ve dış
    tarafa aynı); (2) iç: resmî test görüntüleri iadeli (2.005; K5 şeması), C0/C1 iç değerleri AYNI çekilişin C-test alt
    kümesine düşen çokluk ağırlıklarıyla (ana yol); (3) dış: lezyon kümeleri sınıf tabakalı iadeli (7 tabaka), çokluk
    ağırlıkları bütün kollara aynı. Yapılandırmalar bu sırayla yeni dış çekilişleri alır (seed + iç çekilişleri ortak).
    p = min(1, 2·min((1+#{T*≤0})/(B+1), (1+#{T*≥0})/(B+1))); GA yüzdelik %95. Duyarlılık: seed sabit (yalnız veri).
 5) İkinciller S1–S12, H14, T4 (düzeltmesiz, "keşifsel"); görülmemiş (İNCELE) eşikleri (i)–(iv); önceden yazılmış cümleler.
Kipler:
  --kuru         K-d + VEKİL dış veriyle UÇTAN UCA kod sınaması: "BCN" yerine C-test (1.227; resmî kollar → aynı görüntülerdeki
                 arşiv test tahminleri = C0, C1 kolları → arşiv C-test tahminleri), BCN kodları DermaMNIST etiketinden, akiec'in
                 bir kısmı "SCC" etiketlenir, 3 sahte kopya, 1 sahte lezyon birleştirmesi. Kontrol yapılandırması (E2 eşlemesi,
                 ham küme) iç değerleri BİREBİR vermeli (Δ_dış,R dengeli = C0 +4,45; Δ_dış,C1 ACC_π = +1,12 → e_C1 = 0).
                 BCN dosyası açılmaz. Çıktı sonuclar/e2_bcn_kuru_sonuc.{json,md,svg}.
  --sadece-kuru  yalnız K-d (Colab'da KİLİT satırı çağırır; BCN yok).
  (varsayılan)   gerçek analiz: --runs <E2 koşumlarının runs/ dizini> --kilit <KİLİT> → sonuclar/e2_bcn_sonuc.{json,md,svg} + .sha256.
Kullanım (WS kökü; deney venv'i):
  PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe scripts/e2_bcn_analiz.py --kuru
  PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe scripts/e2_bcn_analiz.py --runs "<DRIVE_KOK>/3-1/runs" --kilit sonuclar/e2_bcn.KILIT
"""
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "scripts"))
sys.path.insert(0, str(WS / "varyantlar"))
import k5_test as K  # noqa: E402  (donmuş: Bootstrap, kiyas, dengeli, auc, ece15, holm, c_alt_kume, kosumlar, yukle_kol, softmax)
from e2 import bcn_veri as BV  # noqa: E402
from e2 import kollar as KL  # noqa: E402

KART = WS / "hipotezler" / "H15.md"
if not KART.exists():                      # Colab paketi: onaylı kartın bayt bayt kopyası (drive_paketle.py E2_KART)
    KART = WS / "kuyruk" / "e2_H15_kart.md"
ONAYLI_KART_SHA = "eeef871456c78ee3686a99958ce6a89d967be6e1ed03ed73489a6cff8ab51b7b"   # KULLANICI ONAYI H15 (2026-11-07)
BEKLENTI = WS / "kuyruk" / "e2-degerlendirme.beklenti.json"
K5_SONUC = WS / "sonuclar" / "k5_test_sonuc.json"
C1_SONUC = WS / "sonuclar" / "k5_c1_sonuc.json"
DONDURMA = WS / "sonuclar" / "k5_dondurma.json"
KILIT = WS / "sonuclar" / "e2_bcn.KILIT"
ETIKET_CSV = WS / "data" / "bcn20000" / "meta" / "bcn_20k_train.csv"
ETIKET_MD5 = "5fd3522a69512dcf5b3c225dc1664ec6"
KUYRUK = "e2-degerlendirme"
B_BOOT, RNG, TOL_PUAN, BUTCE_SA = 10_000, 0, 0.005, 3.0
NK, MEL, NV = 7, 4, 5
EK = "(keşifsel, düzeltmesiz)"
SINIF = K.SINIF
# K-d: kartta yazılı iç değerler (puan); nokta ve GA uçları ±0,005 içinde yeniden üretilmeli
KART_IC = {"R_acc": (2.26, 1.15, 3.30), "R_deng": (6.03, 2.86, 9.37), "C1_acc": (1.12, -0.05, 2.30), "C1_deng": (5.74, 1.21, 10.17),
           "C0_acc": (0.28, None, None), "C0_deng": (4.45, None, None), "R_acc_h08": (-0.36, None, None),
           "C1_acc_h08": (-0.96, None, None)}
# görülmemiş (İNCELE) eşikleri (kart)
INC_DENG_MAKS, INC_B1, INC_ACCPI = 0.70, 12.0, 4.5


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dur(msg: str):
    print(f"DUR: {msg}", flush=True)
    raise SystemExit(2)


def tr(x, n: int = 2, isaret: bool = True) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    s = f"{x:+.{n}f}" if isaret else f"{x:.{n}f}"
    return s.replace("-", "−").replace(".", ",")


def ga_tr(g) -> str:
    return f"[{tr(g[0])}; {tr(g[1])}]"


def p_tr(p) -> str:
    return "—" if p is None else (f"{p:.3g}".replace(".", ",") if p >= 1e-4 else "< 0,0001")


def jsonla(o):
    if isinstance(o, dict):
        return {str(k): jsonla(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonla(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonla(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


# ───────────────────────── iç (arşiv) veri ─────────────────────────
class Ic:
    """Resmî test (2.005) ve C-test (1.227, artan resmî indeks sırası = K5 C0/C1 örneklemi) arşiv tahminleri."""

    def __init__(self, runs: Path):
        self.PR, self.PC, self.kos = {}, {}, {}
        yR = None
        for k in KL.RESMI:
            v, kuy = KL.KOLLAR[k]["ic"]
            P, y, ad = K.yukle_kol(runs, v, "tahmin_test.npy", "etiket_test.npy", kuyruk=kuy)
            if yR is not None and not np.array_equal(y, yR):
                dur(f"iç: {k} resmî test etiketleri diğer kollarla aynı değil")
            yR = y
            self.PR[k], self.kos[k] = P, ad
        idx_csv = K.c_alt_kume("test", "test")
        sira = np.argsort(idx_csv)
        self.idx_c = idx_csv[sira]
        yC = None
        for k in KL.C1:
            v, kuy = KL.KOLLAR[k]["ic"]
            P, y, ad = K.yukle_kol(runs, v, "tahmin_test.npy", "etiket_test.npy", kuyruk=kuy)
            if len(y) != len(idx_csv):
                dur(f"iç: {k} C-test {len(y)} ≠ {len(idx_csv)}")
            P, y = P[:, sira], y[sira]
            if yC is not None and not np.array_equal(y, yC):
                dur(f"iç: {k} C-test etiketleri diğer C1 kollarıyla aynı değil")
            yC = y
            self.PC[k], self.kos[k] = P, ad
        if not np.array_equal(yC, yR[self.idx_c]):
            dur("iç: C-test etiketleri resmî testin aynı indeksleriyle aynı değil (sıra/eşleme)")
        for d in (self.PR, self.PC):
            for k, P in d.items():
                if not (np.isfinite(P).all() and np.allclose(P.sum(2), 1.0, atol=1e-3)):
                    dur(f"iç: {k} olasılıkları sonlu/normalize değil")
        self.yR, self.yC = yR, yC
        self.piR = np.bincount(yR, minlength=NK) / len(yR)     # π_R: resmî test önseli (etiketlerden; elle yazılmaz)
        self.piC = np.bincount(yC, minlength=NK) / len(yC)     # π_C: C-test önseli
        self.PR["t4"] = K.softmax(0.5 * np.log(np.clip(self.PR["h08c"], 1e-12, 1)) + 0.5 * np.log(np.clip(self.PR["h10"], 1e-12, 1)))

    def dogru(self, k):
        """(5, n) bool doğruluk; resmî kollar resmî test, C1 kolları C-test."""
        if k in self.PC:
            return self.PC[k].argmax(2) == self.yC[None]
        return self.PR[k].argmax(2) == self.yR[None]

    def dogru_c0(self, k):
        return self.PR[k][:, self.idx_c].argmax(2) == self.yC[None]


def kuru_ozdeslik(ic: Ic) -> dict:
    """K-d: k5_test şemasıyla (B = 2.000) iç değerlerin yeniden üretimi; donmuş JSON'larla ve kart sayılarıyla."""
    T = json.loads(K5_SONUC.read_text(encoding="utf-8")) if K5_SONUC.exists() else None
    C = json.loads(C1_SONUC.read_text(encoding="utf-8"))["C1"] if C1_SONUC.exists() else None
    BS = K.Bootstrap(len(ic.yR))
    r = K.kiyas(ic.PR["h02c"], ic.PR["h10"], ic.yR, BS)
    r8 = K.kiyas(ic.PR["h08c"], ic.PR["h10"], ic.yR, BS)
    c0 = K.alt_kiyas(ic.PR["h02c"], ic.PR["h10"], ic.yR, ic.idx_c)
    BSc = K.Bootstrap(len(ic.yC))
    c1 = K.kiyas(ic.PC["h02_c1"], ic.PC["h10_c1"], ic.yC, BSc)
    c18 = K.kiyas(ic.PC["h08_c1"], ic.PC["h10_c1"], ic.yC, BSc)

    def pu(e):
        return [100 * e["delta"], 100 * e["ga"][0], 100 * e["ga"][1]]
    bu = {"R_acc": pu(r["acc"]), "R_deng": pu(r["dengeli_acc"]), "C1_acc": pu(c1["acc"]), "C1_deng": pu(c1["dengeli_acc"]),
          "C0_acc": pu(c0["acc"]), "C0_deng": pu(c0["dengeli_acc"]), "R_acc_h08": pu(r8["acc"]), "C1_acc_h08": pu(c18["acc"])}
    donmus = {}
    if T is not None:
        tk = T["kiyas"]
        donmus.update({"R_acc": pu(tk["h10-h02"]["acc"]), "R_deng": pu(tk["h10-h02"]["dengeli_acc"]),
                       "C0_acc": [T["C0"]["kiyas"]["h10-h02"]["delta_puan"], *T["C0"]["kiyas"]["h10-h02"]["ga_puan"]],
                       "C0_deng": [T["C0"]["kiyas"]["h10-h02"]["dengeli_delta_puan"], *T["C0"]["kiyas"]["h10-h02"]["dengeli_ga_puan"]],
                       "R_acc_h08": pu(tk["h10-h08"]["acc"])})
    if C is not None:
        donmus.update({"C1_acc": [C["i"]["h10-h02"]["delta_puan"], *C["i"]["h10-h02"]["ga_puan"]],
                       "C1_deng": [C["aile_C1_2"]["h10-h02"]["dengeli_delta_puan"], *C["aile_C1_2"]["h10-h02"]["dengeli_ga_puan"]],
                       "C1_acc_h08": [C["i"]["h10-h08"]["delta_puan"], *C["i"]["h10-h08"]["ga_puan"]]})
    sat, ok = {}, True
    for k in bu:
        if k not in donmus:            # donmuş JSON bu makinede yok (ör. Colab): yalnız kart sayılarıyla (kartın K-d ölçütü)
            donmus[k] = [float("nan")] * 3
        f_d = max(abs(a - b) for a, b in zip(bu[k], donmus[k])) if np.isfinite(donmus[k][0]) else 0.0
        kart = KART_IC[k]
        f_k = max(abs(a - b) for a, b in zip(bu[k], kart) if b is not None)
        g = bool(f_d <= TOL_PUAN and f_k <= TOL_PUAN + 1e-9)
        sat[k] = {"bu_puan": bu[k], "donmus_puan": donmus[k], "kart_puan": list(kart), "fark_donmus": f_d, "fark_kart": f_k, "gecti": g}
        ok &= g
    return {"gecti": bool(ok), "tolerans_puan": TOL_PUAN, "satirlar": sat, "k5_test_sha256": sha(WS / "scripts" / "k5_test.py"),
            "kosumlar": ic.kos, "donmus_json": {"k5_test_sonuc": T is not None, "k5_c1_sonuc": C is not None}}


def k5_sha_denetle() -> str:
    """k5_test.py donmuş mu (k5_dondurma.json; varsa k5_c1_sonuc.json)."""
    k5s = sha(WS / "scripts" / "k5_test.py")
    don = json.loads(DONDURMA.read_text(encoding="utf-8"))
    c1s = json.loads(C1_SONUC.read_text(encoding="utf-8")).get("betik_sha256") if C1_SONUC.exists() else k5s
    if k5s != don.get("k5_test_sha256") or k5s != c1s:
        dur(f"scripts/k5_test.py SHA256 {k5s[:12]} donmuş değerle aynı değil")
    return k5s


# ───────────────────────── dış veri ─────────────────────────
class Dis:
    """Dış küme (BCN ya da kuru vekil): görüntü sırasıyla ham tanı kodu ve lezyon kimliği + tahminler (kol → {geo: (5,N,7)})."""

    def __init__(self, ad, dx, lez, dislanan, birlesik, P, kaynak):
        self.ad, self.dx, self.lez = np.asarray(ad), np.asarray(dx), np.asarray(lez)
        self.dislanan, self.birlesik, self.P, self.kaynak = set(dislanan), birlesik, P, kaynak
        self.N = len(self.ad)

    def secim(self, esleme: dict, tem: str):
        m = np.array([d in esleme for d in self.dx])
        if tem == "kopyasiz":
            m &= ~np.isin(self.ad, sorted(self.dislanan))
        sel = np.where(m)[0]
        y = np.array([esleme[d] for d in self.dx[sel]], np.int64)
        temsil = {}
        for grup in self.birlesik:
            for u in grup:
                temsil[u] = sorted(grup)[0]
        kume_ad = np.array([temsil.get(x, x) for x in self.lez[sel]])
        uniq, cl = np.unique(kume_ad, return_inverse=True)
        clcls = np.full(len(uniq), -1)
        for i, c in zip(cl, y):
            if clcls[i] not in (-1, c):
                dur(f"dış: lezyon kümesi {uniq[i]} birden çok sınıf içeriyor")
            clcls[i] = c
        return {"sel": sel, "y": y, "cl": cl, "clcls": clcls, "L": len(uniq), "n_lez_ham": len(set(self.lez[sel])),
                "n": len(sel), "n_l": np.bincount(cl, minlength=len(uniq)).astype(np.float64)}


# ───────────────────────── iki düzeyli bootstrap ─────────────────────────
def multinom(rng, B, n, parca=1000):
    W = np.zeros((B, n))
    for i in range(0, B, parca):
        b = min(parca, B - i)
        d = rng.integers(0, n, (b, n))
        W[i:i + b] = np.bincount((d + np.arange(b)[:, None] * n).ravel(), minlength=b * n).reshape(b, n)
    return W


class Cekilis:
    def __init__(self, ic: Ic, B=B_BOOT):
        self.rng = np.random.default_rng(RNG)
        self.B = B
        self.cnt = multinom(self.rng, B, 5).astype(np.float64)          # (1) seed çokluğu (B × 5)
        self.WR = multinom(self.rng, B, len(ic.yR))                       # (2) iç resmî test çokluğu (B × 2.005)
        self.WC = self.WR[:, ic.idx_c]                                    #     aynı çekilişin C-test alt kümesi
        self.bir = np.ones((B, 5))

    def dis(self, cfg):                                                    # (3) dış: sınıf tabakalı lezyon kümesi
        W = np.zeros((self.B, cfg["L"]))
        for c in range(NK):
            Lc = np.where(cfg["clcls"] == c)[0]
            if len(Lc):
                W[:, Lc] = multinom(self.rng, self.B, len(Lc))
        return W


def ic_boot(C, y, W, cnt):
    """(acc_b, dengeli_b) seed-yeniden-örneklemeli; C (5,n) bool, W (B,n). Boş sınıf → dengeli NaN (o yineleme düşer)."""
    Cf = C.astype(np.float64)
    den = W.sum(1)
    acc = (cnt * (W @ Cf.T)).sum(1) / (5 * den)
    rec = []
    for c in range(NK):
        m = y == c
        if not m.any():
            continue
        dc = W[:, m].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            rec.append((cnt * (W[:, m] @ Cf[:, m].T)).sum(1) / (5 * dc))
    return acc, np.mean(np.stack(rec, 1), 1)


def ic_nokta(C, y):
    acc = float(C.mean())
    deng = float(np.mean([np.mean([C[s][y == c].mean() for c in range(NK) if (y == c).any()]) for s in range(len(C))]))
    return acc, deng


def dis_kol(Pk, cfg, W, cnt, lezyon_agirlikli=False):
    """Dış sınıf duyarlılıkları: bootstrap (B × 7) ve nokta (5 × 7, seed başına). Pk (5, N, 7)."""
    C = (Pk[:, cfg["sel"]].argmax(2) == cfg["y"][None]).astype(np.float64)
    S = np.stack([np.bincount(cfg["cl"], weights=C[s], minlength=cfg["L"]) for s in range(len(C))])
    n_l = cfg["n_l"]
    if lezyon_agirlikli:
        S, n_l = S / n_l[None], np.ones_like(n_l)
    rec_b = np.zeros((W.shape[0], NK))
    rec0 = np.zeros((len(C), NK))
    for c in range(NK):
        m = cfg["clcls"] == c
        rec_b[:, c] = (cnt * (W[:, m] @ S[:, m].T)).sum(1) / (5 * (W[:, m] @ n_l[m]))
        rec0[:, c] = S[:, m].sum(1) / n_l[m].sum()
    return rec_b, rec0


def ozet(nokta, boot) -> dict:
    b = np.asarray(boot, np.float64)
    b = b[np.isfinite(b)]
    B = len(b)
    p = min(1.0, 2 * min((1 + np.sum(b <= 0)) / (B + 1), (1 + np.sum(b >= 0)) / (B + 1)))
    return {"nokta": float(nokta), "ga": [float(np.quantile(b, .025)), float(np.quantile(b, .975))], "p": float(p),
            "n_gecerli": int(B), "yari_genislik": float((np.quantile(b, .975) - np.quantile(b, .025)) / 2)}


def etiket_ga(g) -> str:
    return "GA alt > 0" if g[0] > 0 else ("GA üst < 0" if g[1] < 0 else "ayırt edilemedi")


def hesapla(ic: Ic, dis: Dis, cek: Cekilis, esleme: dict, tem: str, geo: str, kollar, seed_sabit=False,
            lezyon_agirlikli=False, W=None) -> dict:
    """Bir yapılandırma için bütün kol duyarlılıkları + B1/B2 + ikinciller (bootstrap)."""
    cfg = dis.secim(esleme, tem)
    if W is None:
        W = cek.dis(cfg)
    cnt = cek.bir if seed_sabit else cek.cnt
    R = {"cfg": {"esleme": "E1" if "SCC" not in esleme else "E2", "kume": tem, "geometri": geo, "n": cfg["n"],
                 "lezyon": cfg["n_lez_ham"], "kume_sayisi": cfg["L"], "seed_sabit": seed_sabit, "lezyon_agirlikli": lezyon_agirlikli,
                 "sinif_goruntu": np.bincount(cfg["y"], minlength=NK).tolist(),
                 "sinif_kume": np.bincount(cfg["clcls"], minlength=NK).tolist()}, "kol": {}}
    rb, r0 = {}, {}
    for k in kollar:
        rb[k], r0[k] = dis_kol(dis.P[k][geo], cfg, W, cnt, lezyon_agirlikli)
    deng_b = {k: rb[k].mean(1) for k in rb}
    deng0 = {k: float(r0[k].mean(1).mean()) for k in r0}
    aR_b = {k: rb[k] @ ic.piR for k in rb}
    aR0 = {k: float((r0[k] @ ic.piR).mean()) for k in r0}
    aC_b = {k: rb[k] @ ic.piC for k in rb}
    aC0 = {k: float((r0[k] @ ic.piC).mean()) for k in r0}
    for k in kollar:
        R["kol"][k] = {"dengeli": deng0[k], "dengeli_seed": r0[k].mean(1).tolist(), "acc_piR": aR0[k], "acc_piC": aC0[k],
                       "duyarlilik": dict(zip(SINIF, r0[k].mean(0).tolist())), "dengeli_ga": ozet(deng0[k], deng_b[k])["ga"]}
    # iç (aynı seed + resmî test çekilişi; C0/C1 aynı çekilişin C-test alt kümesi)
    icv = {}
    for k in [x for x in ("h10", "h02c", "h08c") if x in kollar]:
        C = ic.dogru(k)
        icv[("R", k)] = ic_nokta(C, ic.yR), ic_boot(C, ic.yR, cek.WR, cnt)
        C0 = ic.dogru_c0(k)
        icv[("C0", k)] = ic_nokta(C0, ic.yC), ic_boot(C0, ic.yC, cek.WC, cnt)
    for k in [x for x in ("h10_c1", "h02_c1", "h08_c1") if x in kollar]:
        C = ic.dogru(k)
        icv[("C1", k)] = ic_nokta(C, ic.yC), ic_boot(C, ic.yC, cek.WC, cnt)

    def d_ic(p, a, b, ol):        # iç Δ (puan): ol 0 = ACC, 1 = dengeli
        return 100 * (icv[(p, a)][0][ol] - icv[(p, b)][0][ol]), 100 * (icv[(p, a)][1][ol] - icv[(p, b)][1][ol])

    def d_dis(a, b, ol):          # dış Δ (puan): ol "deng" | "piR" | "piC"
        n_, b_ = {"deng": (deng0, deng_b), "piR": (aR0, aR_b), "piC": (aC0, aC_b)}[ol]
        return 100 * (n_[a] - n_[b]), 100 * (b_[a] - b_[b])

    def fark(x, y_):
        return x[0] - y_[0], x[1] - y_[1]

    def mutlak_fark(e1, e2):
        return abs(e1[0]) - abs(e2[0]), np.abs(e1[1]) - np.abs(e2[1])

    S = {}
    for cift, (a, b), (a1, b1) in (("h10-h02", ("h10", "h02c"), ("h10_c1", "h02_c1")),
                                   ("h10-h08", ("h10", "h08c"), ("h10_c1", "h08_c1"))):
        if not {a, b, a1, b1} <= set(kollar):
            continue
        dR_d, dR_R, dR_C = d_dis(a, b, "deng"), d_dis(a, b, "piR"), d_dis(a, b, "piC")
        dC1_d, dC1_C = d_dis(a1, b1, "deng"), d_dis(a1, b1, "piC")
        iR_acc, iR_deng = d_ic("R", a, b, 0), d_ic("R", a, b, 1)
        iC0_acc, iC0_deng = d_ic("C0", a, b, 0), d_ic("C0", a, b, 1)
        iC1_acc, iC1_deng = d_ic("C1", a1, b1, 0), d_ic("C1", a1, b1, 1)
        eR_pi, eC1_pi = fark(iR_acc, dR_R), fark(iC1_acc, dC1_C)
        eC0_piC, eC0_piR = fark(iC0_acc, dR_C), fark(iC0_acc, dR_R)
        eR_d, eC1_d, eC0_d = fark(iR_deng, dR_d), fark(iC1_deng, dC1_d), fark(iC0_deng, dR_d)
        S[cift] = {
            "dis_R_dengeli": ozet(*dR_d), "dis_R_accpiR": ozet(*dR_R), "dis_R_accpiC": ozet(*dR_C),
            "dis_C1_dengeli": ozet(*dC1_d), "dis_C1_accpiC": ozet(*dC1_C),
            "ic_R_acc": ozet(*iR_acc), "ic_R_dengeli": ozet(*iR_deng), "ic_C0_acc": ozet(*iC0_acc), "ic_C0_dengeli": ozet(*iC0_deng),
            "ic_C1_acc": ozet(*iC1_acc), "ic_C1_dengeli": ozet(*iC1_deng),
            "e_R_pi": ozet(*eR_pi), "e_C1_pi": ozet(*eC1_pi), "e_C0_piC": ozet(*eC0_piC), "e_C0_piR": ozet(*eC0_piR),
            "e_R_dengeli": ozet(*eR_d), "e_C1_dengeli": ozet(*eC1_d), "e_C0_dengeli": ozet(*eC0_d),
            "D_pi": ozet(*mutlak_fark(eR_pi, eC1_pi)), "D_dengeli": ozet(*mutlak_fark(eR_d, eC1_d)),
            "Dp_pi_C0": ozet(*mutlak_fark(eR_pi, eC0_piC)), "Dp_dengeli_C0": ozet(*mutlak_fark(eR_d, eC0_d))}
    R["S"] = S
    if "h14_c1" in kollar:
        R["H14"] = {f"h14_c1-{x}": {"dengeli": ozet(*d_dis("h14_c1", x, "deng")), "accpiC": ozet(*d_dis("h14_c1", x, "piC"))}
                    for x in ("h08_c1", "h10_c1")}
    if "t4" in kollar:
        R["T4"] = {"t4_dengeli": deng0["t4"], "t4_accpiR": aR0["t4"], "t4-h10_dengeli": ozet(*d_dis("t4", "h10", "deng")),
                   "t4-h08c_dengeli": ozet(*d_dis("t4", "h08c", "deng")), "t4-h10_accpiR": ozet(*d_dis("t4", "h10", "piR"))}
    R["_W"] = W
    return R


# ───────────────────────── ikinciller (nokta) ─────────────────────────
def kol_ikincil(Pk, y, ic_std_deng) -> dict:
    t = Pk.argmax(2)
    n = len(Pk)
    hucre = lambda g, h: float(np.mean([((y == g) & (t[s] == h)).sum() for s in range(n)]))  # noqa: E731
    deng = [K.dengeli(y, t[s]) for s in range(n)]
    cift = [float((t[a] != t[b]).mean()) for a in range(n) for b in range(a + 1, n)]
    return {"makro_auc": float(np.mean([K.auc(y, Pk[s]) for s in range(n)])),
            "duyarlilik_mel": float(np.mean([(t[s][y == MEL] == MEL).mean() for s in range(n)])),
            "duyarlilik_nv": float(np.mean([(t[s][y == NV] == NV).mean() for s in range(n)])),
            "ece15": float(np.mean([K.ece15(y, Pk[s]) for s in range(n)])),
            "nll": float(np.mean([-np.mean(np.log(np.clip(Pk[s][np.arange(len(y)), y], 1e-12, 1))) for s in range(n)])),
            "ham_acc": float((t == y[None]).mean()), "tahmin_nv_payi": float((t == NV).mean()),
            "mel_to_nv": hucre(MEL, NV), "nv_to_mel": hucre(NV, MEL), "n_mel": int((y == MEL).sum()), "n_nv": int((y == NV).sum()),
            "dengeli_seed_std_dis": float(np.std(deng, ddof=1)), "dengeli_seed_std_ic": ic_std_deng,
            "seed_cifti_argmax_anlasmazlik": float(np.mean(cift))}


def svg_sacilim(noktalar: list, yol: Path, baslik: str):
    """Miller tarzı ID–OOD saçılımı (probit eksenler); bağımlılıksız SVG."""
    from scipy.stats import norm
    W_, H_, M_ = 560, 460, 60
    xs = [norm.ppf(np.clip(p["ic"], 1e-4, 1 - 1e-4)) for p in noktalar]
    ys = [norm.ppf(np.clip(p["dis"], 1e-4, 1 - 1e-4)) for p in noktalar]
    lo, hi = min(xs + ys) - 0.15, max(xs + ys) + 0.15

    def X(v):
        return M_ + (v - lo) / (hi - lo) * (W_ - 2 * M_)

    def Y(v):
        return H_ - M_ - (v - lo) / (hi - lo) * (H_ - 2 * M_)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W_}" height="{H_}" font-family="sans-serif" font-size="11">',
         f'<text x="{W_ / 2}" y="20" text-anchor="middle" font-size="13">{baslik}</text>',
         f'<line x1="{X(lo)}" y1="{Y(lo)}" x2="{X(hi)}" y2="{Y(hi)}" stroke="#999" stroke-dasharray="4 3"/>',
         f'<rect x="{M_}" y="{M_}" width="{W_ - 2 * M_}" height="{H_ - 2 * M_}" fill="none" stroke="#333"/>']
    for t_ in np.linspace(lo, hi, 5):
        s.append(f'<text x="{X(t_)}" y="{H_ - M_ + 16}" text-anchor="middle">{norm.cdf(t_):.2f}</text>')
        s.append(f'<text x="{M_ - 6}" y="{Y(t_) + 4}" text-anchor="end">{norm.cdf(t_):.2f}</text>')
    s.append(f'<text x="{W_ / 2}" y="{H_ - 18}" text-anchor="middle">iç lezyon-ayrık dengeli doğruluk (probit)</text>')
    s.append(f'<text x="16" y="{H_ / 2}" transform="rotate(-90 16 {H_ / 2})" text-anchor="middle">BCN dengeli doğruluk (probit)</text>')
    for p, x, y in zip(noktalar, xs, ys):
        renk = {"resmi": "#1f77b4", "c1": "#d62728"}[p["veri"]]
        sekil = (f'<rect x="{X(x) - 5}" y="{Y(y) - 5}" width="10" height="10" fill="{renk}"/>' if p["kol"] == "h14_c1"
                 else f'<circle cx="{X(x)}" cy="{Y(y)}" r="5" fill="{renk}"/>')
        s += [sekil, f'<text x="{X(x) + 7}" y="{Y(y) - 6}">{p["ad"]}</text>']
    s.append('<text x="70" y="52" fill="#1f77b4">● resmî bölme kolları</text><text x="230" y="52" fill="#d62728">● C1 kolları (■ h14_c1)</text>')
    s.append("</svg>")
    yol.write_text("\n".join(s), encoding="utf-8")


# ───────────────────────── cümleler ─────────────────────────
def cumleler(R) -> dict:
    if R["gorulmemis"]["INCELE"]:
        return {"not": "Görülmemiş (İNCELE) bayrağı açık → soruşturma kapanana dek sayı yazılmaz, cümle seçilmez (kart)."}
    A = R["ana"]
    b1, b2 = R["B1"], R["B2"]
    dus = [v["dusus_puan"] for v in R["dusus"].values()]
    n_tr, l_tr = f"{A['cfg']['n']:,}".replace(",", "."), f"{A['cfg']['lezyon']:,}".replace(",", ".")
    giris = ("Resmî bölmede ve lezyon-ayrık C1 bölmesinde eğitilmiş kollar, yeniden eğitilmeden, eşik ya da kalibrasyon ayarı "
             "yapılmadan, farklı bir merkezin dermoskopi verisinde (BCN20000, Barselona; "
             f"{n_tr} görüntü, {l_tr} lezyon; SCC dışlandı) bir kez değerlendirildi. "
             + f"Dengeli doğruluk iç lezyon-ayrık değerlere göre {tr(min(abs(x) for x in dus), isaret=False)}–"
               f"{tr(max(abs(x) for x in dus), isaret=False)} puan düştü.")
    d1 = b1["nokta"]
    if b1["etiket"] == "DINOv2 üstün":
        c1 = (f"DINOv2-B'nin ImageNet-ResNet-18'e dengeli doğruluk üstünlüğü dış merkezde korundu: Δ {tr(d1)} puan "
              f"{ga_tr(b1['ga'])} (Holm p {p_tr(b1['holm_p'])}). İç kestirimler resmî bölmede +6,0, lezyon-ayrık C1'de +5,7 puandı.")
    elif b1["etiket"] == "ResNet üstün":
        c1 = (f"Dış merkezde sıralama tersine döndü: ImageNet-ResNet-18, DINOv2-B'yi dengeli doğrulukta {tr(abs(d1), isaret=False)} "
              f"puan {ga_tr(b1['ga'])} geçti. İki iç protokol de dış sıralamayı öngöremedi. Bu, HAM10000→BCN20000 aktarımında "
              "bildirilen sıra dönüşleriyle uyumludur.")
    else:
        c1 = (f"Dış merkezde DINOv2-B ile ImageNet-ResNet-18 dengeli doğrulukta ayırt edilemedi: Δ {tr(d1)} puan {ga_tr(b1['ga'])}. "
              "İki iç protokol de yaklaşık +6 puanlık bir üstünlük öngörmüştü. GA, "
              f"±{tr(b1['yari_genislik'], isaret=False)} puandan küçük farkları dışlamaz. Sonuç ‘fark yok’ değil, ‘bu dış veriyle "
              "belirlenemedi’ diye okunmalıdır.")
    eR, eC = A["S"]["h10-h02"]["e_R_pi"], A["S"]["h10-h02"]["e_C1_pi"]
    if b2["etiket"] == "C1 daha iyi öngördü":
        c2 = (f"Önsel-sabit doğrulukta lezyon-ayrık protokolün öngörü hatası (e_C1 {tr(eC['nokta'])} puan) resmî bölmeninkinden "
              f"(e_R {tr(eR['nokta'])} puan) küçüktü: D {tr(b2['nokta'])} puan {ga_tr(b2['ga'])}, Holm p {p_tr(b2['holm_p'])}. "
              "Dış merkez farkı lezyon-ayrık iç kestirime daha yakın çıktı.")
    elif b2["etiket"] == "resmî daha iyi öngördü":
        c2 = (f"Önsel-sabit doğrulukta resmî bölmenin öngörü hatası (e_R {tr(eR['nokta'])}) lezyon-ayrık protokolünkinden "
              f"(e_C1 {tr(eC['nokta'])}) küçüktü: D {tr(b2['nokta'])} puan {ga_tr(b2['ga'])}, Holm p {p_tr(b2['holm_p'])}. "
              "Lezyon tekrarı içeren resmî bölme, bu dış merkezde farkı daha iyi öngördü.")
    else:
        c2 = (f"İki iç protokolün öngörü hataları ayırt edilemedi (e_R {tr(eR['nokta'])}, e_C1 {tr(eC['nokta'])}; "
              f"D {tr(b2['nokta'])} puan {ga_tr(b2['ga'])}). Bu kıyasın çözünürlüğü iki iç kestirim arasındaki farkla (1,1 puan) "
              "sınırlıdır. Bu nedenle sonuç ‘iki protokol eşit iyi öngördü’ diye değil, ‘bu veriyle hangisinin daha iyi "
              "öngördüğü belirlenemedi’ diye okunmalıdır.")
    uy = lambda e: "uyumlu" if e["ga"][0] <= 0 <= e["ga"][1] else "uyumsuz"  # noqa: E731
    c3 = (f"Dış fark, resmî bölme kestiriminin %95 aralığıyla {uy(eR)} (e_R {tr(eR['nokta'])} {ga_tr(eR['ga'])}) ve lezyon-ayrık "
          f"kestirimin aralığıyla {uy(eC)} (e_C1 {tr(eC['nokta'])} {ga_tr(eC['ga'])}) çıktı (keşifsel, düzeltmesiz).")
    sinir = ("Dış küme BCN20000'in yalnız etiketli eğitim bölümüdür ve bu görüntüler modellerimizin eğitiminde kullanılmadı; ISIC "
             "test etiketleri kullanılmadı; SCC dışlandı (duyarlılıkta akiec'e eşlendi); hasta düzeyinde ayrıklık ölçülemedi, "
             "merkezler coğrafi olarak ayrık; görüntü düzeyi metrik lezyon-kümeli bootstrap ile verildi; rakip 28 px kolları "
             "ağırlık dosyası olmadığı için değerlendirilemedi; klinik bir iddia taşınmaz.")
    return {"giris": giris, "B1_etiketi": b1["etiket"], "B1": c1, "B2_etiketi": b2["etiket"], "B2": c2, "kapsama_S5": c3,
            "sinirliliklar": sinir}


# ───────────────────────── ana analiz ─────────────────────────
def analiz(ic: Ic, dis: Dis, esleme: dict, kuru: bool) -> dict:
    cek = Cekilis(ic)
    kollar = KL.SIRA + ["t4"]
    dis.P["t4"] = {g: K.softmax(0.5 * np.log(np.clip(dis.P["h08c"][g], 1e-12, 1)) + 0.5 * np.log(np.clip(dis.P["h10"][g], 1e-12, 1)))
                   for g in dis.P["h10"]}
    E1, E2 = esleme["E1"], esleme["E2"]
    R = {}
    ana = hesapla(ic, dis, cek, E1, "kopyasiz", "merkez_kare", kollar)
    W_ana = ana.pop("_W")
    R["ana"] = ana
    s = ana["S"]["h10-h02"]
    # B1 / B2 (Holm m = 2, iki yönlü)
    hp = K.holm([s["dis_R_dengeli"]["p"], s["D_pi"]["p"]])
    b1 = {**s["dis_R_dengeli"], "holm_p": hp[0]}
    b2 = {**s["D_pi"], "holm_p": hp[1]}
    b1["etiket"] = ("DINOv2 üstün" if (b1["holm_p"] < 0.05 and b1["nokta"] > 0) else
                    ("ResNet üstün" if (b1["holm_p"] < 0.05 and b1["nokta"] < 0) else "ayırt edilemedi"))
    b2["etiket"] = ("C1 daha iyi öngördü" if (b2["holm_p"] < 0.05 and b2["nokta"] > 0) else
                    ("resmî daha iyi öngördü" if (b2["holm_p"] < 0.05 and b2["nokta"] < 0) else "protokoller ayırt edilemedi"))
    R["B1"], R["B2"] = b1, b2
    # duyarlılık: seed sabit (yalnız veri bootstrap'ı), aynı dış çekiliş
    sd = hesapla(ic, dis, cek, E1, "kopyasiz", "merkez_kare", ["h10", "h02c", "h08c", "h10_c1", "h02_c1", "h08_c1"],
                 seed_sabit=True, W=W_ana)
    R["duyarlilik_seed_sabit"] = {"B1": sd["S"]["h10-h02"]["dis_R_dengeli"], "B2": sd["S"]["h10-h02"]["D_pi"]}
    # ikinciller (keşifsel)
    R["S1"] = {"dis_C1_dengeli": s["dis_C1_dengeli"], "dis_C1_accpiC": s["dis_C1_accpiC"]}
    R["S2"] = {"Dp_pi_C0": s["Dp_pi_C0"], "Dp_dengeli_C0": s["Dp_dengeli_C0"], "e_C0_piC": s["e_C0_piC"], "e_C0_piR": s["e_C0_piR"],
               "e_C0_dengeli": s["e_C0_dengeli"], "e_R_dengeli": s["e_R_dengeli"]}
    R["S3"] = {"D_dengeli": s["D_dengeli"], "not": "çözünürlük sınırı ≈ 0,29 puan; önceden 'bilgilendirici değil' diye beyan edildi"}
    s8 = ana["S"]["h10-h08"]
    R["S4"] = {"B1_esi_dis_R_dengeli_h10-h08c": s8["dis_R_dengeli"], "B2_esi_D_pi": s8["D_pi"], "e_R_pi": s8["e_R_pi"],
               "e_C1_pi": s8["e_C1_pi"], "dis_R_accpiR": s8["dis_R_accpiR"], "dis_C1_accpiC": s8["dis_C1_accpiC"]}
    for x in ("e_R_pi", "e_C1_pi", "e_C0_piC", "e_R_dengeli", "e_C1_dengeli", "e_C0_dengeli"):
        R.setdefault("S5", {})[x] = {**s[x], "kapsama": "uyumlu" if s[x]["ga"][0] <= 0 <= s[x]["ga"][1] else "uyumsuz"}
    yap = {}
    for ad, (es, tem, geo, la) in {"S7_E2_esleme": (E2, "kopyasiz", "merkez_kare", False),
                                   "S8_ham_kume": (E1, "ham", "merkez_kare", False),
                                   "S9_cerceve": (E1, "kopyasiz", "cerceve", False),
                                   "S10_lezyon_agirlikli": (E1, "kopyasiz", "merkez_kare", True)}.items():
        kk = ["h10", "h02c", "h08c", "h10_c1", "h02_c1", "h08_c1"]
        r = hesapla(ic, dis, cek, es, tem, geo, kk, lezyon_agirlikli=la, W=(W_ana if la else None))
        r.pop("_W")
        q = r["S"]["h10-h02"]
        yap[ad] = {"cfg": r["cfg"], "B1": q["dis_R_dengeli"], "kol_dengeli": {k: v["dengeli"] for k, v in r["kol"].items()}}
        if ad in ("S7_E2_esleme", "S9_cerceve"):
            yap[ad].update({"B2": q["D_pi"], "e_R_pi": q["e_R_pi"], "e_C1_pi": q["e_C1_pi"]})
    R.update(yap)
    if kuru:   # kontrol yapılandırması: E2 eşleme + ham küme = C-test'in kendisi → iç değerler BİREBİR
        r = hesapla(ic, dis, cek, E2, "ham", "merkez_kare", ["h10", "h02c", "h10_c1", "h02_c1"])
        r.pop("_W")
        q = r["S"]["h10-h02"]
        C0j = json.loads(K5_SONUC.read_text(encoding="utf-8"))["C0"]["kiyas"]["h10-h02"]
        C1j = json.loads(C1_SONUC.read_text(encoding="utf-8"))["C1"]
        kont = {"dis_R_dengeli = C0 dengeli": (q["dis_R_dengeli"]["nokta"], C0j["dengeli_delta_puan"]),
                "dis_R_accpiC = C0 ACC": (q["dis_R_accpiC"]["nokta"], C0j["delta_puan"]),
                "dis_C1_accpiC = C1 ACC": (q["dis_C1_accpiC"]["nokta"], C1j["i"]["h10-h02"]["delta_puan"]),
                "dis_C1_dengeli = C1 dengeli": (q["dis_C1_dengeli"]["nokta"], C1j["aile_C1_2"]["h10-h02"]["dengeli_delta_puan"]),
                "e_C1_pi = 0": (q["e_C1_pi"]["nokta"], 0.0),
                "e_C0_piC = 0": (q["e_C0_piC"]["nokta"], 0.0)}
        R["kuru_kontrol"] = {k: {"bu": a, "beklenen": b, "fark": abs(a - b), "gecti": bool(abs(a - b) < 1e-9)} for k, (a, b) in kont.items()}
        R["kuru_kontrol_gecti"] = all(v["gecti"] for v in R["kuru_kontrol"].values())
    # kol tablosu + S11/S12 + düşüş + İNCELE + S6
    cfg = dis.secim(E1, "kopyasiz")
    ic_deng = {}
    for k in KL.SIRA:
        C = ic.dogru(k) if k in ic.PC else ic.dogru_c0(k)        # lezyon-ayrık iç: resmî kollar C0, C1 kolları C-test
        ic_deng[k] = ic_nokta(C, ic.yC)[1]
    ic_deng_resmi = {k: ic_nokta(ic.dogru(k), ic.yR)[1] for k in KL.RESMI}
    R["kol_ikincil"], R["dusus"] = {}, {}
    for k in KL.SIRA:
        Pk = dis.P[k]["merkez_kare"][:, cfg["sel"]]
        yi = ic.yC if k in ic.PC else ic.yR
        Pi = ic.PC[k] if k in ic.PC else ic.PR[k]
        ic_std = float(np.std([K.dengeli(yi, Pi[s].argmax(1)) for s in range(len(Pi))], ddof=1))
        R["kol_ikincil"][k] = kol_ikincil(Pk, cfg["y"], ic_std)
        R["dusus"][k] = {"bcn_dengeli": ana["kol"][k]["dengeli"], "ic_lezyon_ayrik_dengeli": ic_deng[k],
                         "dusus_puan": 100 * (ana["kol"][k]["dengeli"] - ic_deng[k])}
    scc = np.where((dis.dx == "SCC") & ~np.isin(dis.ad, sorted(dis.dislanan)))[0]
    R["S12_scc"] = {"n": int(len(scc)), "tahmin_dagilimi": {k: dict(zip(SINIF, np.mean(
        [np.bincount(dis.P[k]["merkez_kare"][s, scc].argmax(1), minlength=NK) / max(len(scc), 1) for s in range(5)], 0).tolist()))
        for k in KL.SIRA} if len(scc) else {}}
    tau_r = stats.kendalltau([ic_deng_resmi[k] for k in KL.RESMI], [ana["kol"][k]["dengeli"] for k in KL.RESMI])
    tau_r0 = stats.kendalltau([ic_deng[k] for k in KL.RESMI], [ana["kol"][k]["dengeli"] for k in KL.RESMI])
    tau_c = stats.kendalltau([ic_deng[k] for k in KL.C1], [ana["kol"][k]["dengeli"] for k in KL.C1])
    R["S6"] = {"tau_b_resmi_k6_ic_resmi_test": float(tau_r.statistic), "tau_b_resmi_k6_ic_C0": float(tau_r0.statistic),
               "tau_b_C1_k5_ic_C_test": float(tau_c.statistic), "not": "yalnız betimsel; p değerine dayalı karar yok (kart S6)",
               "sacilim": [{"kol": k, "ad": KL.KOLLAR[k]["ad"], "veri": KL.KOLLAR[k]["veri"], "ic": ic_deng[k],
                            "dis": ana["kol"][k]["dengeli"]} for k in KL.SIRA]}
    g = {f"(i) {k}: BCN dengeli ≥ iç lezyon-ayrık": bool(R["dusus"][k]["dusus_puan"] >= 0) for k in KL.SIRA}
    g.update({f"(ii) {k}: BCN dengeli > 0,70": bool(ana["kol"][k]["dengeli"] > INC_DENG_MAKS) for k in KL.SIRA})
    g["(iii) |B1 Δ| > 12,0"] = bool(abs(b1["nokta"]) > INC_B1)
    g["(iv) |Δ_dış ACC_π_R (H10 − H02c)| > 4,5"] = bool(abs(s["dis_R_accpiR"]["nokta"]) > INC_ACCPI)
    R["gorulmemis"] = {"bayraklar": g, "INCELE": any(g.values())}
    R["beklenti_icinde"] = {"B1_[-4;+12]": bool(-4 <= b1["nokta"] <= 12), "B2_dis_accpiR_[-4;+6]": bool(-4 <= s["dis_R_accpiR"]["nokta"] <= 6),
                            "D_pi_[-1,5;+1,5]": bool(-1.5 <= b2["nokta"] <= 1.5)}
    R["cumleler"] = cumleler(R)
    if kuru:   # yalnız kod sınaması: İNCELE açık olsa da cümle doldurma yolu çalıştırılır (BCN sonucu DEĞİL)
        R["cumleler_sinama_kuru"] = cumleler({**R, "gorulmemis": {"INCELE": False}})
    return R


def rapor_md(R: dict) -> str:
    A = R["ana"]
    L = [f"# E2 / H15 — BCN20000 dış merkez analizi{' — KURU KOŞU (VEKİL dış veri = C-test; BCN AÇILMADI)' if R['kuru'] else ''}",
         f"*{R['tarih']}* · betik `{R['betik_sha256'][:16]}…` · kart `{R['kart_sha256'][:16]}…` · k5_test `{R['k5_test_sha256'][:16]}…`",
         "", "**Etiketler:** B1/B2 Holm (m = 2, iki yönlü); ikinciller " + EK + ". K5 kararı ve Holm aileleri değişmez.", ""]
    if R["kuru"]:
        L += ["> **KURU KOŞU:** dış küme = C-test (1.227): resmî kollar → arşiv resmî test tahminlerinin C-test satırları (C0), C1 "
              "kolları → arşiv C-test tahminleri. 'SCC', kopya ve lezyon birleştirmesi SAHTEDİR (kod yolu sınaması). Sayılar "
              "BCN sonucu DEĞİLDİR.", ""]
    oz = R["kuru_ozdeslik"]
    L += [f"## 1) K-d kuru koşu özdeşliği → **{'GEÇTİ' if oz['gecti'] else 'DUR'}** (tolerans {oz['tolerans_puan']} puan)",
          "| satır | bu (Δ; GA) | donmuş JSON | kart | maks fark (donmuş / kart) |", "|---|---|---|---|---|"]
    for k, v in oz["satirlar"].items():
        L.append(f"| {k} | {tr(v['bu_puan'][0])}; [{tr(v['bu_puan'][1])}; {tr(v['bu_puan'][2])}] | {tr(v['donmus_puan'][0], 4)} | "
                 f"{tr(v['kart_puan'][0])} | {v['fark_donmus']:.1e} / {v['fark_kart']:.4f} |")
    if R.get("kuru_kontrol"):
        L += ["", f"## 1b) Vekil kontrol yapılandırması (E2 eşleme + ham küme = C-test) → **{'GEÇTİ' if R['kuru_kontrol_gecti'] else 'DUR'}**",
              "| denetim | bu | beklenen | fark |", "|---|---|---|---|"]
        L += [f"| {k} | {v['bu']:+.10f} | {v['beklenen']:+.10f} | {v['fark']:.1e} |" for k, v in R["kuru_kontrol"].items()]
    c = A["cfg"]
    L += ["", f"## 2) Ana yapılandırma: E1 eşleme, kopyasız küme, merkez_kare — n = {c['n']}, lezyon {c['lezyon']} "
              f"(bootstrap kümesi {c['kume_sayisi']}), sınıf görüntü {c['sinif_goruntu']}",
          f"π_R = {np.round(R['piR'], 4).tolist()} · π_C = {np.round(R['piC'], 4).tolist()} (iç test önselleri; etiketlerden)", "",
          "| kol | rol | BCN dengeli [GA] | ACC_π_R | ACC_π_C | ham ACC | makro-AUC | mel duy. | iç lezyon-ayrık dengeli | düşüş (puan) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k in KL.SIRA:
        e, i2, d = A["kol"][k], R["kol_ikincil"][k], R["dusus"][k]
        L.append(f"| {KL.KOLLAR[k]['ad']} | {KL.KOLLAR[k]['rol']} | {e['dengeli']:.4f} [{e['dengeli_ga'][0]:.4f}; {e['dengeli_ga'][1]:.4f}] | "
                 f"{e['acc_piR']:.4f} | {e['acc_piC']:.4f} | {i2['ham_acc']:.4f} | {i2['makro_auc']:.4f} | {i2['duyarlilik_mel']:.4f} | "
                 f"{d['ic_lezyon_ayrik_dengeli']:.4f} | {tr(d['dusus_puan'])} |")
    b1, b2 = R["B1"], R["B2"]
    s = A["S"]["h10-h02"]
    L += ["", "## 3) Birincil aile (Holm m = 2, iki yönlü; B = 10.000 iki düzeyli bootstrap)",
          "| | nokta (puan) | %95 GA | p | Holm p | etiket |", "|---|---|---|---|---|---|",
          f"| **B1** Δ_dış,R dengeli (H10 − H02c) | {tr(b1['nokta'])} | {ga_tr(b1['ga'])} | {p_tr(b1['p'])} | {p_tr(b1['holm_p'])} | **{b1['etiket']}** |",
          f"| **B2** D_π = \\|e_R\\| − \\|e_C1\\| | {tr(b2['nokta'])} | {ga_tr(b2['ga'])} | {p_tr(b2['p'])} | {p_tr(b2['holm_p'])} | **{b2['etiket']}** |",
          "", "| bileşen | nokta | %95 GA |", "|---|---|---|"]
    for x in ("ic_R_acc", "dis_R_accpiR", "e_R_pi", "ic_C1_acc", "dis_C1_accpiC", "e_C1_pi", "ic_R_dengeli", "ic_C1_dengeli", "ic_C0_acc",
              "ic_C0_dengeli", "dis_R_accpiC"):
        L.append(f"| {x} | {tr(s[x]['nokta'])} | {ga_tr(s[x]['ga'])} |")
    sd = R["duyarlilik_seed_sabit"]
    L += [f"- Duyarlılık (seed sabit, yalnız veri bootstrap'ı): B1 {tr(sd['B1']['nokta'])} {ga_tr(sd['B1']['ga'])} · "
          f"B2 {tr(sd['B2']['nokta'])} {ga_tr(sd['B2']['ga'])}",
          f"- Beklenti aralıkları (kart, bilgi): {json.dumps(R['beklenti_icinde'], ensure_ascii=False)}",
          "", "## 4) İkinciller " + EK, "| | nokta (puan) | %95 GA | etiket |", "|---|---|---|---|"]

    def satir(ad, e):
        L.append(f"| {ad} | {tr(e['nokta'])} | {ga_tr(e['ga'])} | {etiket_ga(e['ga'])} |")
    satir("S1 Δ_dış,C1 dengeli (h10_c1 − h02_c1)", R["S1"]["dis_C1_dengeli"])
    satir("S1 Δ_dış,C1 ACC_π_C", R["S1"]["dis_C1_accpiC"])
    satir("S2 D'_π = \\|e_R\\| − \\|e_C0\\| (π_C hedefi)", R["S2"]["Dp_pi_C0"])
    satir("S2 D'_dengeli", R["S2"]["Dp_dengeli_C0"])
    satir("S2 e_C0 (π_R hedefi, bilgi)", R["S2"]["e_C0_piR"])
    satir("S3 D_dengeli (bilgilendirici değil)", R["S3"]["D_dengeli"])
    satir("S4 Δ_dış,R dengeli (H10 − H08c)", R["S4"]["B1_esi_dis_R_dengeli_h10-h08c"])
    satir("S4 D_π (H10 − H08 çifti)", R["S4"]["B2_esi_D_pi"])
    for ad in ("S7_E2_esleme", "S8_ham_kume", "S9_cerceve", "S10_lezyon_agirlikli"):
        satir(f"{ad} B1", R[ad]["B1"])
        if "B2" in R[ad]:
            satir(f"{ad} B2 (D_π)", R[ad]["B2"])
    for k, v in R["ana"].get("H14", {}).items():
        satir(f"H14 {k} dengeli", v["dengeli"])
        satir(f"H14 {k} ACC_π_C", v["accpiC"])
    t4 = R["ana"].get("T4", {})
    if t4:
        L.append(f"| T4 (H08c⊕H10) dengeli {t4['t4_dengeli']:.4f} · ACC_π_R {t4['t4_accpiR']:.4f} | | | |")
        satir("T4 − H10 dengeli", t4["t4-h10_dengeli"])
    L += ["", "**S5 kapsama:** " + " · ".join(f"{k} {tr(v['nokta'])} {ga_tr(v['ga'])} → {v['kapsama']}" for k, v in R["S5"].items()),
          "", f"**S6 Kendall τ_b (betimsel):** resmî k = 6 (iç = resmî test) {R['S6']['tau_b_resmi_k6_ic_resmi_test']:+.3f} · "
              f"(iç = C0) {R['S6']['tau_b_resmi_k6_ic_C0']:+.3f} · C1 k = 5 {R['S6']['tau_b_C1_k5_ic_C_test']:+.3f} · saçılım: "
              f"`{R.get('svg', '—')}`",
          "", "**S11 (kol başına):**", "| kol | ECE-15 | NLL | tahmin nv payı | mel→nv | nv→mel | seed std dış / iç (dengeli) | seed çifti anlaşmazlığı |",
          "|---|---|---|---|---|---|---|---|"]
    for k in KL.SIRA:
        e = R["kol_ikincil"][k]
        L.append(f"| {KL.KOLLAR[k]['ad']} | {e['ece15']:.4f} | {e['nll']:.4f} | {e['tahmin_nv_payi']:.4f} | {e['mel_to_nv']:.1f} | "
                 f"{e['nv_to_mel']:.1f} | {e['dengeli_seed_std_dis']:.4f} / {e['dengeli_seed_std_ic']:.4f} | {e['seed_cifti_argmax_anlasmazlik']:.4f} |")
    if R["S12_scc"]["n"]:
        L += ["", f"**S12 SCC görüntüleri (n = {R['S12_scc']['n']}) tahmin dağılımı (seed ort.):**"]
        L += [f"- {KL.KOLLAR[k]['ad']}: " + ", ".join(f"{c} {v:.3f}" for c, v in d.items()) for k, d in R["S12_scc"]["tahmin_dagilimi"].items()]
    L += ["", f"## 5) Görülmemiş (İNCELE) → **{'AÇIK — sayı yazılmaz, cümle seçilmez' if R['gorulmemis']['INCELE'] else 'kapalı'}**",
          "- açık bayraklar: " + (", ".join(k for k, v in R["gorulmemis"]["bayraklar"].items() if v) or "yok"),
          "", "## 6) Önceden yazılmış cümleler (mekanik seçim)"]
    L += [f"- **{k}:** {v}" for k, v in R["cumleler"].items()]
    if R.get("cumleler_sinama_kuru"):
        L += ["", "### 6b) KURU — cümle doldurma yolu sınaması (İNCELE yok sayılarak; vekil sayılar, makaleye GİRMEZ)"]
        L += [f"- {k}: {v}" for k, v in R["cumleler_sinama_kuru"].items()]
    return "\n".join(L) + "\n"


# ───────────────────────── gerçek veri / vekil ─────────────────────────
def kuru_dis(ic: Ic, beklenen_esleme) -> Dis:
    """VEKİL: C-test (artan resmî indeks) — resmî kollar C0 tahminleri, C1 kolları C-test tahminleri."""
    lez = {}
    with open(WS / "data" / "splits" / "dermamnist_lezyon_esleme.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "test":
                lez[int(r["index"])] = r["lesion_id"]
    kod = {0: "AK", 1: "BCC", 2: "BKL", 3: "DF", 4: "MEL", 5: "NV", 6: "VASC"}
    ad = np.array([f"VEKIL_{i:05d}" for i in ic.idx_c])
    lz = np.array([lez[int(i)] for i in ic.idx_c])
    dx = np.array([kod[int(c)] for c in ic.yC], dtype=object)
    ak_lez = sorted({l_ for l_, c in zip(lz, ic.yC) if c == 0})
    scc_lez = set(ak_lez[::3])                            # akiec lezyonlarının üçte biri "SCC" (E1/E2 yolunu sınar)
    dx = np.array(["SCC" if (c == 0 and l_ in scc_lez) else d for d, c, l_ in zip(dx, ic.yC, lz)])
    nv_lez = sorted({l_ for l_, c in zip(lz, ic.yC) if c == 5})
    P = {k: {"merkez_kare": ic.PR[k][:, ic.idx_c], "cerceve": ic.PR[k][:, ic.idx_c]} for k in KL.RESMI}
    P.update({k: {"merkez_kare": ic.PC[k], "cerceve": ic.PC[k]} for k in KL.C1})
    return Dis(ad, dx, lz, ad[[0, 1, 2]].tolist(), [nv_lez[:2]], P, "VEKİL C-test")


def gercek_dis(runs: Path, B: dict, ikinci: str | None) -> tuple:
    """a3 koşumları → tahminler (ETİKET OKUNMADAN denetlenir: K-c/K-e/K-f/K-g). Dönüş: (P, denetim)."""
    P, den, toplam_sn = {}, {"kosumlar": {}, "durum": {}}, 0.0
    for d in Path(runs).glob("*/"):
        cf, dn = d / "config.json", d / ".done"
        if cf.exists() and dn.exists():
            c = json.loads(cf.read_text(encoding="utf-8"))
            if str(c.get("kuyruk", "")).startswith(KUYRUK):
                toplam_sn += float(json.loads(dn.read_text(encoding="utf-8")).get("sure_sn") or 0)
    den["butce_L4sa"] = toplam_sn / 3600
    if den["butce_L4sa"] > BUTCE_SA:
        dur(f"K-g: E2 koşum süresi {den['butce_L4sa']:.2f} L4-sa > {BUTCE_SA}")
    for k in KL.SIRA:
        v = KL.e2_varyant(k, "bcn")
        bul = {}
        for d in sorted(Path(runs).glob("*/")):
            cf, dn = d / "config.json", d / ".done"
            if not (cf.exists() and dn.exists()):
                continue
            c = json.loads(cf.read_text(encoding="utf-8"))
            if c.get("varyant") != v or not str(c.get("kuyruk", "")).startswith(KUYRUK) or int(c.get("asama", 3)) != 3:
                continue
            if json.loads(dn.read_text(encoding="utf-8")).get("rc") != 0:
                continue
            bul.setdefault(int(c["seed"]), []).append(d)
        if sorted(bul) != list(KL.SEEDLER):
            dur(f"{v}: başarılı a3 seed kümesi {sorted(bul)} ≠ {{0..4}}")
        P[k] = {g: [] for g in BV.GEOMETRILER}
        for s in KL.SEEDLER:
            if len(bul[s]) > 1 and not ikinci:
                dur(f"{v} s{s}: birden çok BCN koşumu {[x.name for x in bul[s]]} (K-f; yalnız --ikinci-acilis)")
            d = bul[s][-1]
            m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
            c = json.loads((d / "config.json").read_text(encoding="utf-8"))
            e2 = c.get("e2") or {}
            durum = {1: "A", 2: "B", 3: "C"}.get(m.get("ozdeslik_durum_kod"), "?")
            den["durum"][f"{k}/s{s}"] = durum
            den["kosumlar"].setdefault(k, []).append(d.name)
            hata = []
            if e2.get("kaynak_run") != KL.run_id(k, s):
                hata.append(f"kaynak {e2.get('kaynak_run')} ≠ {KL.run_id(k, s)}")
            for q, bek in (("dis_kilit_var", 1), ("bcn_npz_sha_eslesti", 1), ("val_kumesi_dogru", 1), ("ozdeslik_gecti", 1),
                           ("bcn_etiket_okundu", 0), ("resmi_test_okundu", 0), ("c_test_okundu", 0), ("bcn_n", B["n"]),
                           ("bcn_geometri_sayisi", 2)):
                if m.get(q) != bek:
                    hata.append(f"{q} = {m.get(q)} ≠ {bek}")
            if m.get("bcn_ikinci_cikarim", 0) != 0 and not ikinci:
                hata.append("bcn_ikinci_cikarim ≠ 0")
            if e2.get("bcn_sira_sha256") != B["sira_sha256"]:
                hata.append("BCN sıra SHA256 uyuşmuyor")
            for g in BV.GEOMETRILER:
                f = d / f"tahmin_bcn_{g}.npy"
                if not f.exists() or (e2.get("bcn_tahmin_sha256") or {}).get(g) != sha(f):
                    hata.append(f"tahmin_bcn_{g}.npy yok ya da SHA256 kayıtla aynı değil")
                    continue
                Pg = np.load(f).astype(np.float64)
                if Pg.shape != (B["n"], NK):
                    hata.append(f"{g} şekil {Pg.shape}")
                P[k][g].append(Pg)
            if durum == "C" and k in KL.BIRINCIL:
                dur(f"K-c: birincil kol {k} s{s} Durum C → E2 tümüyle DUR")
            if hata:
                dur(f"{v} s{s} ({d.name}): " + "; ".join(hata))
        for g in BV.GEOMETRILER:
            P[k][g] = np.stack(P[k][g])
            sonlu = np.isfinite(P[k][g]).all()
            pay = max(np.bincount(P[k][g][s].argmax(1), minlength=NK).max() / B["n"] for s in range(5)) if sonlu else 1.0
            den.setdefault("K-e", {})[f"{k}/{g}"] = {"sonlu": bool(sonlu), "tek_sinif_maks_pay": float(pay)}
            if not sonlu or pay > 0.95:
                dur(f"K-e: {k} {g} dejenere çıktı (sonlu {sonlu}, tek sınıf payı {pay:.3f}) — ön-işleme hatası şüphesi")
    den["durum_ozet"] = {x: sum(1 for v in den["durum"].values() if v == x) for x in "ABC"}
    return P, den


def etiketleri_oku(B: dict) -> tuple:
    """BCN etiketleri — YALNIZ bu noktada (KİLİT ve bütün denetimler sonrası) okunur."""
    if hashlib.md5(ETIKET_CSV.read_bytes()).hexdigest() != ETIKET_MD5:
        dur("bcn_20k_train.csv MD5 figshare değeriyle aynı değil")
    sat = {}
    with open(ETIKET_CSV, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            sat[r["bcn_filename"]] = (r["diagnosis"], r["lesion_id"])
    with np.load(BV.DIZIN / BV.NPZ["merkez_kare"], allow_pickle=False) as z:   # yalnız sıra (bcn_filename)
        ad = z["bcn_filename"].tolist()
    if BV.sira_sha256(ad) != B["sira_sha256"]:
        dur("BCN sıra SHA256 (yerel girdi dosyası) beklenenle aynı değil")
    return ad, np.array([sat[x][0] for x in ad]), np.array([sat[x][1] for x in ad])


def kilit_denetle(kilit: Path, B: dict, ikinci: str | None) -> dict:
    if not kilit.exists():
        dur(f"{kilit} yok — KİLİT olmadan analiz yapılmaz (K-f)")
    Kj = json.loads(kilit.read_text(encoding="utf-8"))
    if Kj.get("analiz_zamani") and not ikinci:
        dur(f"BCN analizi zaten {Kj['analiz_zamani']} tarihinde yapıldı — yalnız --ikinci-acilis '<neden>' (K-f)")
    guncel = {"kart": sha(KART), "analiz": sha(Path(__file__)), "beklenti": sha(BEKLENTI), "beklenen": sha(BV.BEKLENEN),
              "e2_cikarim": sha(WS / "varyantlar" / "e2" / "e2_cikarim.py"),
              "kopyasiz_liste": sha(WS / B["kopyasiz_liste"]["yol"]), "lezyon_kumeleri": sha(WS / B["lezyon_kumeleri"]["yol"])}
    fark = {k: (Kj.get("sha256", {}).get(k, "")[:12], v[:12]) for k, v in guncel.items() if Kj.get("sha256", {}).get(k) != v}
    if fark:
        dur(f"KİLİT SHA256'ları güncel dosyalarla aynı değil: {fark}")
    if guncel["kart"] != ONAYLI_KART_SHA:
        dur("H15.md onaylı SHA256 ile aynı değil")
    if Kj.get("kollar") != {k: [KL.run_id(k, s) for s in KL.SEEDLER] for k in KL.SIRA}:
        dur("KİLİT kol → run_id listesi kollar.py ile aynı değil")
    return Kj


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kuru", action="store_true", help="K-d + vekil dış veriyle uçtan uca kod sınaması (BCN açılmaz)")
    ap.add_argument("--sadece-kuru", action="store_true", help="yalnız K-d (Colab KİLİT satırı)")
    ap.add_argument("--runs", default="runs_colab", help="E2 a3 koşumlarının runs/ dizini (gerçek analiz)")
    ap.add_argument("--arsiv", default="runs_colab", help="iç arşiv (k5-test / k5-c1 / e1-h14 test tahminleri)")
    ap.add_argument("--kilit", default=str(KILIT))
    ap.add_argument("--ikinci-acilis", default=None, help="K-f: ikinci açılışın NEDENİ (iki sonuç da yazılır)")
    ap.add_argument("--cikti", default=None, help="yalnız kod sınaması: çıktı JSON yolu")
    a = ap.parse_args()
    arsiv = Path(a.arsiv) if Path(a.arsiv).is_absolute() else WS / a.arsiv
    runs = Path(a.runs) if Path(a.runs).is_absolute() else WS / a.runs
    k5s = k5_sha_denetle()
    ic = Ic(arsiv)
    oz = kuru_ozdeslik(ic)
    print(f"[e2] K-d kuru koşu: {'GEÇTİ' if oz['gecti'] else 'DUR'} · " + ", ".join(
        f"{k} {v['bu_puan'][0]:+.2f} (fark {v['fark_donmus']:.1e}/{v['fark_kart']:.3f})" for k, v in oz["satirlar"].items()), flush=True)
    if a.sadece_kuru:
        out = Path(a.cikti) if a.cikti else WS / "sonuclar" / "e2_bcn_kd_sonuc.json"
        out.write_text(json.dumps(jsonla({"tarih": datetime.now().isoformat(timespec="seconds"), "betik_sha256": sha(Path(__file__)),
                                          **oz}), ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"yazıldı: {out}")
        return 0 if oz["gecti"] else 2
    if not oz["gecti"]:
        dur("K-d kuru koşu iç değerleri yeniden üretmedi (|fark| > 0,005 puan) — BCN açılmaz")
    esleme = None
    if a.kuru:
        esleme = {"E1": {"NV": 5, "MEL": 4, "BCC": 1, "BKL": 2, "DF": 3, "VASC": 6, "AK": 0}}
        esleme["E2"] = dict(esleme["E1"], SCC=0)
        dis = kuru_dis(ic, esleme)
        den, kilit_j = {"kuru": "BCN tahminleri okunmadı"}, None
        out = Path(a.cikti) if a.cikti else WS / "sonuclar" / "e2_bcn_kuru_sonuc.json"
    else:
        B = BV.beklenen_oku()
        out = Path(a.cikti) if a.cikti else WS / "sonuclar" / "e2_bcn_sonuc.json"
        if out.exists() and not a.ikinci_acilis:
            dur(f"{out.name} zaten var — yalnız --ikinci-acilis '<neden>' (iki sonuç da yazılır)")
        if a.ikinci_acilis and out.exists():
            out = out.with_name(out.stem + f"_ikinci_{datetime.now():%Y%m%d-%H%M%S}.json")
        kilit_j = kilit_denetle(Path(a.kilit), B, a.ikinci_acilis)
        esleme = {k: {kk: int(vv) for kk, vv in v.items()} for k, v in B["esleme"].items()}
        P, den = gercek_dis(runs, B, a.ikinci_acilis)
        print(f"[e2] a3 denetimleri GEÇTİ (Durum {den['durum_ozet']}, bütçe {den['butce_L4sa']:.2f} L4-sa) → etiketler AÇILIYOR", flush=True)
        ad, dx, lez = etiketleri_oku(B)
        kl = json.loads((WS / B["kopyasiz_liste"]["yol"]).read_text(encoding="utf-8"))
        lk = json.loads((WS / B["lezyon_kumeleri"]["yol"]).read_text(encoding="utf-8"))
        dis = Dis(ad, dx, lez, kl["dislanan"], lk["birlesik"], P, "BCN20000 figshare train")
        Kj = json.loads(Path(a.kilit).read_text(encoding="utf-8"))
        Kj.setdefault("analiz_acilislari", []).append({"zaman": datetime.now().isoformat(timespec="seconds"),
                                                       "ikinci_acilis": a.ikinci_acilis, "betik_sha256": sha(Path(__file__))})
        Kj["analiz_zamani"] = Kj.get("analiz_zamani") or datetime.now().isoformat(timespec="seconds")
        Path(a.kilit).write_text(json.dumps(Kj, ensure_ascii=False, indent=1), encoding="utf-8")
    R = analiz(ic, dis, esleme, a.kuru)
    R.update({"kuru": a.kuru, "tarih": datetime.now().isoformat(timespec="seconds"), "betik_sha256": sha(Path(__file__)),
              "kart_sha256": sha(KART), "kart_onayli": sha(KART) == ONAYLI_KART_SHA, "k5_test_sha256": k5s, "kuru_ozdeslik": oz,
              "piR": ic.piR, "piC": ic.piC, "dis_kaynak": dis.kaynak, "a3_denetim": den, "ikinci_acilis": a.ikinci_acilis,
              "yontem": f"iki düzeyli eşleştirilmiş bootstrap B = {B_BOOT}, default_rng({RNG}); çekiliş sırası seed → iç resmî test "
                        "(C0/C1 aynı çekilişin C-test alt kümesi; ana yol) → dış sınıf tabakalı lezyon kümesi (yapılandırma sırasıyla); "
                        "iki yönlü p; Holm m = 2 (B1, B2); ikinciller düzeltmesiz"})
    svg = out.with_suffix(".svg")
    svg_sacilim(R["S6"]["sacilim"], svg, "E2: iç lezyon-ayrık ↔ dış dengeli doğruluk" + (" (KURU VEKİL)" if a.kuru else ""))
    R["svg"] = svg.name
    temiz = jsonla(R)
    out.write_text(json.dumps(temiz, ensure_ascii=False, indent=1), encoding="utf-8")
    md = rapor_md(temiz)
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    out.with_name(out.name + ".sha256").write_text(f"{sha(out)}  {out.name}\n", encoding="utf-8")
    print(md)
    print(f"yazıldı: {out} (+ .md, .svg, .sha256)")
    if a.kuru and not R.get("kuru_kontrol_gecti"):
        print("DUR: vekil kontrol yapılandırması iç değerleri birebir vermedi", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
