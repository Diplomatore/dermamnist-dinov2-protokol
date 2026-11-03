# -*- coding: utf-8 -*-
"""
k5_test.py — K5 analiz betiği (hipotezler/K5-ONKAYIT.md §2, §3, §4.2–4.3, §5, §7). CPU; KARAR VERMEZ, ölçütleri mekanik işaretler.

Kipler
  --kip kuru-val   P5 KURU KOŞUSU: "test" yerine resmî VAL (1.003), "C-test" yerine C-val ∩ val (573) verilir; bütün tablolar
                   üretilir. Kaynak: Colab K4 koşumlarının tahmin_val.npy'leri (runs_colab/ arşivi). Test DOSYASI AÇILMAZ.
                   Beklenen yeniden üretim (P5): H10 − H02c +2,91 [+1,34; +4,61] (tahlil-tur4 §3).
                   C1/C2 kod yolu da kuru koşulur: C1/C2 "tahmini" yerine aynı val tahminleri (vekil; K1 = 0 beklenir).
  --kip test       RESMÎ TEST (§1–§3): k5-test kuyruğunun aşama-3 koşumları (`<kol>-test`, tahmin_test.npy). YALNIZ BİR KEZ:
                   --test-kilidi-acik + k5_dondurma.json SHA256 doğrulaması (bu betiğin ve ön-kaydın SHA256'sı donmuş
                   değerle aynı olmalı) + sonuclar/k5_test.KILIT yoksa. İkinci açılış yalnız --ikinci-acilis "<neden>" (§6.4).
  --kip c1 / c2    C1 / C2 (§4, §5): k5-c1-* / k5-c2 kuyruklarının C-test tahminleri + resmî test sonucu (KÇ için).

İstatistik (§2.4; tahlil_tur4.py ile AYNI yöntem)
  - Seed düzeyi: kesin tek yönlü seed-eşleştirilmiş t (scipy.stats.ttest_rel, alternative="greater"), n = 5.
  - Görüntü bootstrap'ı: B = 2.000, numpy.random.default_rng(0), W = bincount(rng.integers(0, N, (B, N))); aynı W iki kola;
    Δ* = ort_s ACC_B − ort_s ACC_A (seed-ortalamalı görüntü doğruluğundan); %95 yüzdelik GA; p = (1 + #{Δ* ≤ 0}) / (B + 1).
  - Alt küme kıyasları (C0, C1, C2): alt kümenin kendi indeksleri üzerinde ayrı rng(0) bootstrap'ı (tahlil_tur4_cval.py ile aynı).
  - Ortak bootstrap (KÇ0 / KÇ): 2.005 (kuru: 1.003) resmî indeks yeniden örneklenir; alt küme Δ'sı örneklemdeki alt küme
    indekslerinden (W ağırlıklı) hesaplanır.
  - ACC / AUC = medmnist Evaluator tanımı (argmax doğruluğu; makro OvR AUC). Dengeli = sınıf duyarlılıkları ortalaması;
    dengeli-5 = df/vasc hariç. ECE-15, NLL, makro-F1 ikincil.
  - Katmanlar: H04 τ* (donmuş, tam-val) → argmax(log p − τ·log π_train); R-T: seed başına donmuş T → softmax(log p / T).
    İkinci ve üçüncü satır; karar vermez.
  - Holm (α = 0,05, tek yönlü) t p'lerine ve bootstrap p'lerine AYRI ayrı; aile 1 {H01c, H02c, H08c, H10} − baseline_c,
    aile 2 {H08c, H10} − H02c (H13 Tur-5'te ÖLDÜR → aileler 4 ve 2).
  - S/N (§2.6): S = lezyonu resmî train'de olan görüntü; tabaka = sınıf × dx_type × kaynak (N-ağırlıklı; tahlil_tur4_ezber.py).
Kullanım (WS kökü; deney venv'i, scipy/sklearn gerekir):
  PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I scripts/k5_test.py --kip kuru-val --runs runs_colab
  PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I scripts/k5_test.py --kip test --runs "G:/Drive'ım/yz/3-1/runs" --test-kilidi-acik
Çıktı: sonuclar/k5_<kip>_sonuc.json + .md
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
from sklearn.metrics import f1_score, roc_auc_score

WS = Path(__file__).resolve().parents[1]
SINIF = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
K, NV, NB, RNG = 7, 5, 2000, 0
BES = [0, 1, 2, 4, 5]
CTEST_COGUNLUK = 0.7726  # C-test nv payı (948/1.227); data/dermamnist_c/dermamnist_c_analiz.json
ONKAYIT = WS / "hipotezler" / "K5-ONKAYIT.md"
DONDURMA = WS / "sonuclar" / "k5_dondurma.json"
KILIT = WS / "sonuclar" / "k5_test.KILIT"

# kol kısa adı → K4 (val) kaynak varyantı; test kipinde varyant adı + "-test"
KOL = {"baseline": "baseline_colab", "h01": "h01_dinov2b_lp_r224_colab", "h02": "h02_r18in_ft224_colab",
       "h08": "h08_r18in_ft224_d4_colab", "h10": "h10_dinov2b_k4_ft224", "h11": "h11_dinov2s_ft224",
       "h12": "h12_dinov2s_lp_r224"}
AD = {"baseline": "baseline_c", "h01": "H01c", "h02": "H02c", "h08": "H08c", "h10": "H10", "h11": "H11", "h12": "H12",
      "t4": "T4 (H08c⊕H10)", "baseline_son": "baseline_c (son epoch)"}
AILE1 = [("baseline", "h01"), ("baseline", "h02"), ("baseline", "h08"), ("baseline", "h10")]
AILE2 = [("h02", "h08"), ("h02", "h10")]
BETIMSEL = [("h08", "h10"), ("baseline", "h11"), ("baseline", "h12"), ("h02", "h11"), ("h01", "h12"), ("h10", "t4"),
            ("h02", "t4"), ("h08", "t4"), ("baseline", "baseline_son")]
# C1 kolları (§4.2) — kip c1/c2 varyant adları (k5-c1-* / k5-c2 kuyrukları; varyantlar/k5/README.md)
C1_KOL = {"baseline": "baseline_c1", "h01": "h01_c1", "h02": "h02_c1", "h08": "h08_c1", "h10": "h10_c1"}
C2_KOL = {"baseline": "baseline_c2", "h08": "h08_c2", "h10": "h10_c2"}


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ───────────────────────── veri ─────────────────────────
def bolum_meta(bolum: str) -> dict:
    """Resmî bölüm (val | test) indeks sırasıyla: S (lezyonu resmî train'de), dx_type, kaynak. Yalnız metadata CSV."""
    tr, lez, img = set(), {}, {}
    with open(WS / "data/splits/dermamnist_lezyon_esleme.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "train":
                tr.add(r["lesion_id"])
            elif r["split"] == bolum:
                lez[int(r["index"])] = r["lesion_id"]
                img[int(r["index"])] = r["image_id"]
    meta = {}
    with open(WS / "data/meta/HAM10000_metadata.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            meta[r["image_id"]] = r
    n = max(lez) + 1
    return {"S": np.array([lez[i] in tr for i in range(n)]), "dx_type": np.array([meta[img[i]]["dx_type"] for i in range(n)]),
            "kaynak": np.array([meta[img[i]]["dataset"] for i in range(n)]), "n": n}


def c_alt_kume(bolum: str, split_c: str) -> np.ndarray:
    """C bölmesinin (split_c) resmî `bolum`'deki görüntülerinin resmî indeksleri — C CSV satır sırasıyla (= C npz sırası)."""
    with open(WS / "data/splits/dermamnist_c_esleme.csv", encoding="utf-8") as f:
        return np.array([int(r["index_resmi"]) for r in csv.DictReader(f)
                         if r["split_C"] == split_c and r["split_resmi"] == bolum], dtype=np.int64)


def train_logpi() -> np.ndarray:
    with np.load(WS / "data/medmnist/dermamnist.npz") as z:
        ytr = z["train_labels"].reshape(-1).astype(np.int64)
    return np.log(np.bincount(ytr, minlength=K) / len(ytr))


def kosumlar(runs: Path, varyant: str, kuyruk=None, asama=3) -> list:
    out = {}
    for d in sorted(Path(runs).glob("*/")):
        cf, dn = d / "config.json", d / ".done"
        if not (cf.exists() and dn.exists()):
            continue
        c = json.loads(cf.read_text(encoding="utf-8"))
        j = json.loads(dn.read_text(encoding="utf-8"))
        if c.get("varyant") != varyant or j.get("rc") != 0 or int(c.get("asama", 3)) != asama:
            continue
        if kuyruk and not str(c.get("kuyruk", "")).startswith(kuyruk):
            continue
        s = int(c["seed"])
        if s in out:
            raise SystemExit(f"HATA: {varyant} s{s} için birden çok başarılı koşum: {out[s].name}, {d.name}")
        out[s] = d
    return [out[s] for s in sorted(out)]


def yukle_kol(runs: Path, varyant: str, dosya: str, etiket_dosya: str, kuyruk=None):
    dz = kosumlar(runs, varyant, kuyruk)
    if [int(json.loads((d / "config.json").read_text(encoding="utf-8"))["seed"]) for d in dz] != [0, 1, 2, 3, 4]:
        raise SystemExit(f"HATA: {varyant}: seed kümesi {{0..4}} değil ({[d.name for d in dz]})")
    P, y0 = [], None
    for d in dz:
        y = np.load(d / etiket_dosya).reshape(-1).astype(np.int64)
        if y0 is not None and not np.array_equal(y, y0):
            raise SystemExit(f"HATA: {varyant}: etiketler koşumlar arasında farklı")
        y0 = y
        P.append(np.load(d / dosya).astype(np.float64))
    return np.stack(P), y0, [d.name for d in dz]


# ───────────────────────── ölçüler ─────────────────────────
def dengeli(y, t, sinif=range(K)):
    return float(np.mean([(t[y == c] == c).mean() for c in sinif if (y == c).any()]))


def ece15(y, p):
    g, d = p.max(1), (p.argmax(1) == y).astype(float)
    s = np.linspace(0, 1, 16)
    t = 0.0
    for i in range(15):
        m = (g > s[i]) & (g <= s[i + 1]) if i else (g >= s[i]) & (g <= s[i + 1])
        if m.any():
            t += m.mean() * abs(d[m].mean() - g[m].mean())
    return float(t)


def auc(y, p):
    return float(np.mean([roc_auc_score((y == c).astype(int), p[:, c]) for c in range(K) if 0 < (y == c).sum() < len(y)]))


def tah_tau(P, tau, logpi):
    return (np.log(np.maximum(P, 1e-300)) - tau * logpi[None, None, :]).argmax(2)


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def kol_ozet(P, y, tau=None, T=None, logpi=None) -> dict:
    t = P.argmax(2)
    r = {"acc": [float((t[s] == y).mean()) for s in range(len(P))],
         "auc": [auc(y, P[s]) for s in range(len(P))],
         "dengeli_acc": [dengeli(y, t[s]) for s in range(len(P))],
         "dengeli_acc_5sinif": [dengeli(y, t[s], BES) for s in range(len(P))],
         "makro_f1": [float(f1_score(y, t[s], average="macro", labels=list(range(K)), zero_division=0)) for s in range(len(P))],
         "ece15": [ece15(y, P[s]) for s in range(len(P))],
         "nll": [float(-np.mean(np.log(np.clip(P[s][np.arange(len(y)), y], 1e-12, 1)))) for s in range(len(P))],
         "nv_payi": [float((t[s] == NV).mean()) for s in range(len(P))]}
    for c in range(K):
        if (y == c).any():
            r[f"duyarlilik_{SINIF[c]}"] = [float((t[s][y == c] == c).mean()) for s in range(len(P))]
    out = {"ort": {k: float(np.mean(v)) for k, v in r.items()}, "std": {k: float(np.std(v, ddof=1)) for k, v in r.items()},
           "seed": r}
    if tau is not None:
        tk = tah_tau(P, tau, logpi)
        out["katman_H04"] = {"tau": tau, "acc_ort": float(np.mean([(tk[s] == y).mean() for s in range(len(P))])),
                             "dengeli_ort": float(np.mean([dengeli(y, tk[s]) for s in range(len(P))])),
                             "dengeli5_ort": float(np.mean([dengeli(y, tk[s], BES) for s in range(len(P))])),
                             "acc_seed": [float((tk[s] == y).mean()) for s in range(len(P))]}
    if T is not None:
        L = np.log(np.maximum(P, 1e-300))
        PT = np.stack([softmax(L[s] / T[s]) for s in range(len(P))])
        out["katman_RT"] = {"T": list(T), "ece15_ort": float(np.mean([ece15(y, PT[s]) for s in range(len(P))])),
                            "nll_ort": float(np.mean([-np.mean(np.log(np.clip(PT[s][np.arange(len(y)), y], 1e-12, 1)))
                                                      for s in range(len(P))])),
                            "argmax_degismedi": bool(np.array_equal(PT.argmax(2), P.argmax(2)))}
    return out


class Bootstrap:
    def __init__(self, N):
        self.N = N
        rng = np.random.default_rng(RNG)
        idx = rng.integers(0, N, size=(NB, N))
        self.W = np.zeros((NB, N))
        for b in range(NB):
            self.W[b] = np.bincount(idx[b], minlength=N)

    def ort(self, f, m=None):
        if m is None:
            return self.W @ f / self.N
        return (self.W[:, m] @ f[m]) / np.maximum(self.W[:, m].sum(1), 1)


def ga(x):
    return [float(np.quantile(x, .025)), float(np.quantile(x, .975))]


def pboot(x):
    return float((np.sum(x <= 0) + 1) / (NB + 1))


def kiyas(PA, PB, y, BS: Bootstrap, S=None, tabakalar=None) -> dict:
    """B − A; ACC, dengeli, dengeli-5 (bootstrap) + seed t (ACC, dengeli) + S/N (ham, tabaka-std)."""
    dA, dB = (PA.argmax(2) == y[None]).mean(0), (PB.argmax(2) == y[None]).mean(0)
    f = dB - dA
    e = {"acc": {"A": float(dA.mean()), "B": float(dB.mean()), "delta": float(f.mean())}}
    bs = BS.ort(f)
    e["acc"].update({"ga": ga(bs), "p_boot": pboot(bs)})
    oh = np.stack([(y == c).astype(float) for c in range(K)], 1)
    for ad, sin in (("dengeli_acc", list(range(K))), ("dengeli_acc_5sinif", BES)):
        sin = [c for c in sin if (y == c).any()]
        recA = np.array([dA[y == c].mean() for c in sin]); recB = np.array([dB[y == c].mean() for c in sin])
        Wc = BS.W @ oh[:, sin]
        rA = (BS.W @ (dA[:, None] * oh[:, sin])) / np.maximum(Wc, 1)
        rB = (BS.W @ (dB[:, None] * oh[:, sin])) / np.maximum(Wc, 1)
        ok = (Wc > 0).all(1)
        fb = (rB - rA).mean(1)[ok]
        e[ad] = {"A": float(recA.mean()), "B": float(recB.mean()), "delta": float(recB.mean() - recA.mean()),
                 "ga": ga(fb), "p_boot": float((np.sum(fb <= 0) + 1) / (len(fb) + 1)), "n_gecerli_ornek": int(ok.sum())}
    aA, aB = (PA.argmax(2) == y[None]).mean(1), (PB.argmax(2) == y[None]).mean(1)
    xA = np.array([dengeli(y, PA[s].argmax(1)) for s in range(len(PA))])
    xB = np.array([dengeli(y, PB[s].argmax(1)) for s in range(len(PB))])
    for ad, xa, xb in (("t_acc", aA, aB), ("t_dengeli", xA, xB)):
        if len(xa) == len(xb) and len(xa) > 1:
            fd = xb - xa
            p = float(stats.ttest_rel(xb, xa, alternative="greater").pvalue) if np.any(fd != fd[0]) or fd[0] != 0 else 1.0
            e[ad] = {"fark_seed": fd.tolist(), "ort": float(fd.mean()), "sd": float(fd.std(ddof=1)), "p_tek": p}
    e["farkli_dogru_goruntu"] = int((f != 0).sum())
    if S is not None:
        e["acc_S_ham"] = {"delta": float(f[S].mean()), "ga": ga(BS.ort(f, S))}
        e["acc_N_ham"] = {"delta": float(f[~S].mean()), "ga": ga(BS.ort(f, ~S))}
        e["ham_dS_eksi_dN"] = {"nokta": float(f[S].mean() - f[~S].mean()), "ga": ga(BS.ort(f, S) - BS.ort(f, ~S))}
        e["S_payi_net"] = float(f[S].sum() / f.sum()) if f.sum() != 0 else None
    if tabakalar:
        top = {"S": 0.0, "N": 0.0}; w = 0.0
        tb = {"S": np.zeros(NB), "N": np.zeros(NB)}; wb = np.zeros(NB)
        for mS, mN in tabakalar:
            wk = mN.sum()
            top["S"] += wk * f[mS].mean(); top["N"] += wk * f[mN].mean(); w += wk
            sS, sN = BS.W[:, mS].sum(1), BS.W[:, mN].sum(1)
            ok = (sS > 0) & (sN > 0)
            tb["S"] += np.where(ok, sN * (BS.W[:, mS] @ f[mS]) / np.maximum(sS, 1), 0)
            tb["N"] += np.where(ok, sN * (BS.W[:, mN] @ f[mN]) / np.maximum(sN, 1), 0)
            wb += np.where(ok, sN, 0)
        dS, dN = tb["S"] / wb, tb["N"] / wb
        e["tab_std_dS"] = {"nokta": float(top["S"] / w), "ga": ga(dS)}
        e["tab_std_dN"] = {"nokta": float(top["N"] / w), "ga": ga(dN)}
        e["tab_std_dS_eksi_dN"] = {"nokta": float((top["S"] - top["N"]) / w), "ga": ga(dS - dN)}
    return e


def alt_kiyas(PA, PB, y, idx) -> dict:
    """Alt küme (idx, ör. C-test) içinde kendi rng(0) bootstrap'ı (tahlil_tur4_cval.py yöntemi) + seed t."""
    yc = y[idx]
    return kiyas(PA[:, idx], PB[:, idx], yc, Bootstrap(len(idx)))


def holm(p: list) -> list:
    m = len(p)
    o = np.argsort(p)
    adj = np.zeros(m)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i]))
        adj[i] = run
    return adj.tolist()


def tabakalar_kur(y, meta):
    tab = np.array([f"{y[i]}|{meta['dx_type'][i]}|{meta['kaynak'][i]}" for i in range(len(y))])
    S = meta["S"]
    return [((tab == k) & S, (tab == k) & ~S) for k in sorted(set(tab)) if ((tab == k) & S).any() and ((tab == k) & ~S).any()]


def ortak_kc(f_res_A, f_res_B, f_alt_A, f_alt_B, idx_alt, BS: Bootstrap) -> dict:
    """KÇ: K = Δ(resmî, tüm) − Δ(alt küme). f_res_*: (N,) seed-ort görüntü doğruluğu (resmî modeller, tüm indeksler);
    f_alt_*: (N,) aynı biçim, yalnız idx_alt'ta anlamlı (C0: resmî modeller; C1: C1 modelleri resmî indekse yerleştirilmiş)."""
    m = np.zeros(BS.N, dtype=bool); m[idx_alt] = True
    fr, fa = f_res_B - f_res_A, f_alt_B - f_alt_A
    d_res, d_alt = float(fr.mean()), float(fa[m].mean())
    b_res, b_alt = BS.ort(fr), BS.ort(fa, m)
    return {"delta_resmi": d_res, "delta_alt": d_alt, "K": d_res - d_alt, "K_ga": ga(b_res - b_alt),
            "b_K": b_res - b_alt, "delta_resmi_ga": ga(b_res), "delta_alt_ga": ga(b_alt)}


def etiket_kural(ga_):
    return "desteklendi (GA alt > 0)" if ga_[0] > 0 else ("tersi (GA üst < 0)" if ga_[1] < 0 else "ayırt edilemedi")


# ───────────────────────── ana ─────────────────────────
def dondurma_oku(zorunlu: bool):
    if not DONDURMA.exists():
        if zorunlu:
            raise SystemExit(f"HATA: {DONDURMA} yok (P4)")
        return None
    return json.loads(DONDURMA.read_text(encoding="utf-8"))


def tau_T(D, kol, kuru_degerler):
    if D is not None:
        return D["katman"][kol]["tau_tamval"], D["katman"][kol]["T_tamval_seed"]
    return kuru_degerler.get(kol, (None, None))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kip", choices=["kuru-val", "test", "c1", "c2"], required=True)
    ap.add_argument("--runs", required=True, help="koşum klasörlerini içeren dizin (Drive runs/ ya da runs_colab/)")
    ap.add_argument("--test-kilidi-acik", action="store_true")
    ap.add_argument("--ikinci-acilis", default=None, help="§6.4: hata düzeltmesi sonrası ikinci açılışın NEDENİ")
    ap.add_argument("--cikti-ek", default="", help="çıktı dosya adı eki (kuru koşu tekrarları için)")
    a = ap.parse_args()
    runs = Path(a.runs) if Path(a.runs).is_absolute() else WS / a.runs
    kuru = a.kip == "kuru-val"
    D = dondurma_oku(zorunlu=not kuru)
    sonuc = {"kip": a.kip, "tarih": datetime.now().isoformat(timespec="seconds"), "runs": str(runs),
             "betik_sha256": sha256(__file__), "onkayit_sha256": sha256(ONKAYIT),
             "yontem": "bootstrap 2.000 RNG 0 (aynı W iki kola); kesin tek yönlü ttest_rel; Holm t ve bootstrap p'lerine ayrı"}

    if a.kip in ("test", "c1", "c2"):
        if not a.test_kilidi_acik:
            raise SystemExit("HATA: test/C-test kipi yalnız --test-kilidi-acik ile (K5)")
        if D.get("k5_test_sha256") != sonuc["betik_sha256"]:
            raise SystemExit(f"HATA: bu betiğin SHA256'sı donmuş değerle aynı değil ({sonuc['betik_sha256'][:12]} ≠ "
                             f"{str(D.get('k5_test_sha256'))[:12]}) — test öncesi değişiklik yalnız K5-ONKAYIT-b ile")
        if D.get("onkayit_sha256") != sonuc["onkayit_sha256"]:
            raise SystemExit("HATA: K5-ONKAYIT.md donmuş SHA256 ile aynı değil")
        sha_dosya = DONDURMA.with_suffix(".json.sha256")
        if not sha_dosya.exists() or sha_dosya.read_text(encoding="utf-8").split()[0] != sha256(DONDURMA):
            raise SystemExit("HATA: k5_dondurma.json SHA256 damgasıyla uyuşmuyor")
    if a.kip == "test" and KILIT.exists() and not a.ikinci_acilis:
        raise SystemExit(f"HATA: resmî test analizi zaten bir kez koşuldu ({KILIT.read_text(encoding='utf-8')[:200]}). "
                         "§6.4: yalnız hata düzeltmesinden sonra --ikinci-acilis '<neden>' ile; iki sonuç da yazılır.")

    bolum = "val" if kuru else "test"
    meta = bolum_meta(bolum)
    logpi = train_logpi()
    # ── kollar ──
    P, kos = {}, {}
    y = None
    for k, v in KOL.items():
        if kuru:
            Pk, yk, kk = yukle_kol(runs, v, "tahmin_val.npy", "etiket_val.npy")
        elif a.kip == "test":
            Pk, yk, kk = yukle_kol(runs, v + "-test", "tahmin_test.npy", "etiket_test.npy", kuyruk="k5-test")
        else:
            Pk, yk, kk = yukle_kol(runs, v + "-test", "tahmin_test.npy", "etiket_test.npy", kuyruk="k5-test")
        if y is None:
            y = yk
        elif not np.array_equal(y, yk):
            raise SystemExit(f"HATA: {k} etiketleri diğer kollarla aynı değil")
        P[k], kos[k] = Pk, kk
        if not np.allclose(Pk.sum(2), 1.0, atol=1e-3) or not np.isfinite(Pk).all():
            raise SystemExit(f"HATA: {k} olasılıkları sonlu/normalize değil")
    if len(y) != meta["n"]:
        raise SystemExit(f"HATA: etiket sayısı {len(y)} ≠ {bolum} {meta['n']}")
    if kuru:
        P["baseline_son"] = yukle_kol(runs, KOL["baseline"], "tahmin_val_son.npy", "etiket_val.npy")[0]
    else:
        P["baseline_son"] = yukle_kol(runs, KOL["baseline"] + "-test", "tahmin_test_son.npy", "etiket_test.npy", kuyruk="k5-test")[0]
    # T4 = H08c_s ⊕ H10_s (½;½ log-olasılık, kırpma 1e-12) — §1 (8')
    P["t4"] = softmax(0.5 * np.log(np.clip(P["h08"], 1e-12, 1)) + 0.5 * np.log(np.clip(P["h10"], 1e-12, 1)))
    sonuc["kosumlar"] = kos
    if a.kip == "test":
        sonuc["P6"] = p6_denetim(runs, kos)
        if not sonuc["P6"]["gecti"]:
            raise SystemExit(f"HATA: P6 özdeşlik/kaynak denetimi geçmedi — DUR, teşhis: {sonuc['P6']['hatalar'][:5]}")
        # KİLİT: veriler yüklendi ve P6 geçti → resmî test analizi bu noktada açılmış sayılır (§6.1: bir kez)
        KILIT.write_text(json.dumps({"zaman": sonuc["tarih"], "ikinci_acilis": a.ikinci_acilis,
                                     "betik_sha256": sonuc["betik_sha256"]}, ensure_ascii=False) + "\n", encoding="utf-8")
        sonuc["ikinci_acilis"] = a.ikinci_acilis
    sonuc["n"] = int(len(y)); sonuc["n_S"] = int(meta["S"].sum()); sonuc["n_N"] = int((~meta["S"]).sum())

    # ── kol özetleri + katmanlar (donmuş τ*, T) ──
    kuru_kat = {}
    if D is None:  # dondurma henüz yok → kuru koşu katmanları val'den aynı kuralla (LA/SC) hesaplanır
        sys.path.insert(0, str(WS / "varyantlar" / "logit_ayari"))
        import logit_ayari as LA  # noqa: E402
        import sicaklik as SC  # noqa: E402
        taus, Ts = LA.izgara("0:1:0.1"), LA.izgara("0.5:3.0:0.01")
        for k in KOL:
            L = np.log(np.maximum(P[k], 1e-300))
            kuru_kat[k] = (LA.tau_sec(L, logpi, y, np.arange(len(y)), taus, 0.010)[0],
                           [SC.T_sec(L[s], y, np.arange(len(y)), Ts)[0] for s in range(5)])
    ozet = {}
    for k in P:
        tau, T = tau_T(D, k, kuru_kat) if k in KOL else (None, None)
        ozet[k] = kol_ozet(P[k], y, tau, T, logpi)
    sonuc["kollar"] = ozet

    # ── kıyaslar ──
    BS = Bootstrap(len(y))
    tab = tabakalar_kur(y, meta)
    sonuc["n_tabaka"] = len(tab)
    K_ = {}
    for A, B in AILE1 + AILE2 + BETIMSEL:
        K_[f"{B}-{A}"] = kiyas(P[A], P[B], y, BS, meta["S"], tab)
    # ikinci satır (H04 katmanı) kıyasları — betimsel
    K2 = {}
    for A, B in AILE1 + AILE2 + [("h08", "h10")]:
        tA = tah_tau(P[A], ozet[A]["katman_H04"]["tau"], logpi)
        tB = tah_tau(P[B], ozet[B]["katman_H04"]["tau"], logpi)
        PA_ = np.eye(K)[tA]; PB_ = np.eye(K)[tB]
        r = kiyas(PA_, PB_, y, BS)
        K2[f"{B}-{A}"] = {q: r[q] for q in ("acc", "dengeli_acc", "dengeli_acc_5sinif")}
    sonuc["kiyas"] = K_
    sonuc["kiyas_H04_katmani"] = K2

    # ── Holm ──
    def aile_holm(cifts, kural):
        ad = [f"{B}-{A}" for A, B in cifts]
        pt = [K_[x]["t_acc"]["p_tek"] for x in ad]
        pb = [K_[x]["acc"]["p_boot"] for x in ad]
        ht, hb = holm(pt), holm(pb)
        out = {}
        for i, x in enumerate(ad):
            e = K_[x]["acc"]
            kg = kural(e)
            out[x] = {"delta_puan": 100 * e["delta"], "ga_puan": [100 * e["ga"][0], 100 * e["ga"][1]], "p_t": pt[i], "p_boot": pb[i],
                      "holm_t": ht[i], "holm_boot": hb[i], "kural_gecti": kg,
                      "anlamli": bool(kg and ht[i] < 0.05 and hb[i] < 0.05),
                      "ifade": ("anlamlı" if kg and ht[i] < 0.05 and hb[i] < 0.05 else
                                ("doğrulandı (düzeltilmemiş); çoklu karşılaştırma sonrası anlamlılık korunmadı" if kg else
                                 "kural geçmedi"))}
        return out

    sonuc["aile1"] = aile_holm(AILE1, lambda e: bool(e["delta"] >= 0.020 - 1e-12 and e["ga"][0] > 0))
    sonuc["aile2"] = aile_holm(AILE2, lambda e: bool(e["delta"] >= 0.005 - 1e-12 and e["ga"][0] > 0))

    # ── doğrulama kuralları + görülmemiş bayrakları (§2.1–2.2) ──
    h08, h10 = K_["h08-h02"], K_["h10-h02"]
    dog = {"H08 (§2.1)": {"delta_puan": 100 * h08["acc"]["delta"], "ga_puan": [100 * x for x in h08["acc"]["ga"]],
                          "dogrulandi": bool(h08["acc"]["delta"] >= 0.005 - 1e-12 and h08["acc"]["ga"][0] > 0),
                          "r18_satiri": None,
                          "gorulmemis": {"dACC>3": 100 * h08["acc"]["delta"] > 3, "ACC>0,90": ozet["h08"]["ort"]["acc"] > 0.90,
                                         "AUC>0,99": ozet["h08"]["ort"]["auc"] > 0.99,
                                         "dengeli_delta>6,5": 100 * h08["dengeli_acc"]["delta"] > 6.5},
                          "dengeli_5sinif_delta_puan": 100 * h08["dengeli_acc_5sinif"]["delta"]},
           "H10 (§2.2)": {"delta_puan": 100 * h10["acc"]["delta"], "ga_puan": [100 * x for x in h10["acc"]["ga"]],
                          "dogrulandi": bool(h10["acc"]["delta"] >= 0.005 - 1e-12 and h10["acc"]["ga"][0] > 0),
                          "gorulmemis": {"ACC>0,93": ozet["h10"]["ort"]["acc"] > 0.93, "AUC>0,99": ozet["h10"]["ort"]["auc"] > 0.99,
                                         "H10-H02c>+6,0": 100 * h10["acc"]["delta"] > 6.0},
                          "beklenti_[+1,0;+4,5]_icinde": bool(1.0 <= 100 * h10["acc"]["delta"] <= 4.5)}}
    dog["H08 (§2.1)"]["r18_satiri"] = "H08" if dog["H08 (§2.1)"]["dogrulandi"] else "H02"
    sonuc["dogrulama"] = dog
    # S/N (§2.6) — K5-ONKAYIT-b §1: T-SN1/T-SN2 etiketi HAM Δ_S − Δ_N'den (kuralın nesnesi GA(Δ_S − Δ_N); "val +5,1" = ham).
    # Tabaka-standardize Δ_S − Δ_N İKİNCİL satırdır, etiket VERMEZ (tab_* anahtarları; "tab_bilgi_etiketi" yalnız bilgi).
    sn = {}
    for x, hip, bek in (("h10-h02", "T-SN1: Δ_S > Δ_N", [1, 9]), ("h08-h02", "T-SN2: Δ_S ≈ Δ_N", [-3, 4]),
                        ("h10-h08", "betimsel", None)):
        e = K_[x]["tab_std_dS_eksi_dN"]
        eh = K_[x]["ham_dS_eksi_dN"]
        sn[x] = {"hipotez": hip, "kural_kaynagi": "ham Δ_S − Δ_N (K5-ONKAYIT-b §1)",
                 "dS_eksi_dN_puan": 100 * eh["nokta"], "ga_puan": [100 * v for v in eh["ga"]],
                 "kural": etiket_kural(eh["ga"]), "beklenen": bek,
                 "ham_dS_eksi_dN_puan": 100 * eh["nokta"], "ham_ga_puan": [100 * v for v in eh["ga"]],
                 "tab_dS_eksi_dN_puan": 100 * e["nokta"], "tab_ga_puan": [100 * v for v in e["ga"]],
                 "tab_bilgi_etiketi": etiket_kural(e["ga"]) + " (ikincil; etiket vermez)",
                 "dS_tab_puan": 100 * K_[x]["tab_std_dS"]["nokta"], "dN_tab_puan": 100 * K_[x]["tab_std_dN"]["nokta"],
                 "dS_ham_puan": 100 * K_[x]["acc_S_ham"]["delta"], "dN_ham_puan": 100 * K_[x]["acc_N_ham"]["delta"]}
    if sn["h08-h02"]["kural"] == "ayırt edilemedi":
        sn["h08-h02"]["kural"] += " (GA 0'ı kapsıyor → 'dengeli dağılım' ile uyumlu)"
    sonuc["SN"] = sn

    # ── C0 (§3): aynı tahminler, C-test (kuru: C-val) alt kümesi ──
    # C alt kümesi: C0 bootstrap'ı resmî indeks SIRASIYLA (artan; tahlil_tur4_cval.py ile aynı örneklem → kuru koşu
    # C-val sayılarını yeniden üretir). C1/C2 tahminleri C CSV sırasındadır → c1_analiz'de aynı sıraya dizilir.
    idx_c_csv = c_alt_kume(bolum, "test" if not kuru else "val")
    idx_c = np.sort(idx_c_csv)
    yc = y[idx_c]
    c0 = {"n": int(len(idx_c)), "cogunluk_tabani": float((yc == NV).mean()), "kol": {}, "kiyas": {}}
    for k in ["baseline", "h01", "h02", "h08", "h10", "h11", "h12", "t4"]:
        acc = (P[k][:, idx_c].argmax(2) == yc[None]).mean(1)
        c0["kol"][k] = {"acc_ort": float(acc.mean()), "acc_std": float(acc.std(ddof=1)),
                        "dengeli_ort": float(np.mean([dengeli(yc, P[k][s, idx_c].argmax(1)) for s in range(5)])),
                        "dengeli5_ort": float(np.mean([dengeli(yc, P[k][s, idx_c].argmax(1), BES) for s in range(5)])),
                        "acc_eksi_cogunluk": float(acc.mean() - CTEST_COGUNLUK)}
    for A, B, kural in (("h02", "h10", "C0-1"), ("h02", "h08", "C0-2"), ("h08", "h10", "betimsel"), ("baseline", "h10", "betimsel")):
        r = alt_kiyas(P[A], P[B], y, idx_c)
        c0["kiyas"][f"{B}-{A}"] = {"kural": kural, "delta_puan": 100 * r["acc"]["delta"], "ga_puan": [100 * v for v in r["acc"]["ga"]],
                                   "sonuc": etiket_kural(r["acc"]["ga"]), "p_t": r.get("t_acc", {}).get("p_tek"),
                                   "dengeli_delta_puan": 100 * r["dengeli_acc"]["delta"],
                                   "dengeli_ga_puan": [100 * v for v in r["dengeli_acc"]["ga"]]}
    # KÇ0: ortak bootstrap
    dacc = {k: (P[k].argmax(2) == y[None]).mean(0) for k in ("h02", "h08", "h10")}
    k10 = ortak_kc(dacc["h02"], dacc["h10"], dacc["h02"], dacc["h10"], idx_c, BS)
    k08 = ortak_kc(dacc["h02"], dacc["h08"], dacc["h02"], dacc["h08"], idx_c, BS)
    fark = k10["b_K"] - k08["b_K"]
    c0["KC0"] = {"K0_H10_puan": 100 * k10["K"], "K0_H10_ga": [100 * v for v in k10["K_ga"]],
                 "K0_H08_puan": 100 * k08["K"], "K0_H08_ga": [100 * v for v in k08["K_ga"]],
                 "K0_H10_eksi_K0_H08_puan": 100 * (k10["K"] - k08["K"]), "ga_puan": [100 * v for v in ga(fark)],
                 "kural": etiket_kural(ga(fark)), "beklenen": [0.5, 2.5]}
    sonuc["C0"] = c0

    # ── C1 / C2 (§4–§5) ──
    if a.kip in ("c1", "c2") or kuru:
        sonuc["C1"] = c1_analiz(a.kip, runs, P, y, idx_c_csv, BS, kuru)
    if a.kip == "c2" or kuru:
        sonuc["C2"] = c2_analiz(a.kip, runs, sonuc["C1"], kuru)

    # ── §7 ölçütleri (mekanik; W = H10 çünkü H13 ÖLDÜR) ──
    w = sonuc["aile1"]["h10-baseline"]
    sonuc["olcutler"] = {"W": "H10 (H13 Tur-5'te ÖLDÜR)",
                         "G1": bool(w["delta_puan"] >= 2.0 - 1e-9 and w["holm_t"] < 0.05 and w["ga_puan"][0] > 0),
                         "G2_bayraklar": {k: v["gorulmemis"] for k, v in dog.items()},
                         "V": dog["H10 (§2.2)"]["dogrulandi"]}
    if "C1" in sonuc:
        sonuc["olcutler"]["G3"] = sonuc["C1"].get("G3")
        sonuc["olcutler"]["L"] = sonuc["C1"].get("L")

    ek = a.cikti_ek
    out = WS / "sonuclar" / f"k5_{a.kip.replace('-', '_')}{ek}_sonuc.json"
    for q in ("_PC", "_yc", "_sira"):
        sonuc.get("C1", {}).pop(q, None)
    temiz = json.loads(json.dumps(sonuc, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o)))
    out.write_text(json.dumps(temiz, ensure_ascii=False, indent=1), encoding="utf-8")
    md = rapor(sonuc)
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    print(f"yazıldı: {out} (+ .md)")
    return 0


def p6_denetim(runs: Path, kos: dict) -> dict:
    """Test kipi: her `<kol>-test` (a3) koşumu özdeşliği geçmiş mi (ozdeslik_gecti = 1) ve kaynak dosya SHA256'ları aynı seed'in
    `<kol>-ozdeslik` (a1) koşumundakiyle aynı mı (a1 ↔ a3 arasında model dosyası değişmedi)."""
    hatalar, satir = [], {}
    for k, adlar in kos.items():
        a1 = {int(json.loads((d / "config.json").read_text(encoding="utf-8"))["seed"]): d
              for d in kosumlar(runs, KOL[k] + "-ozdeslik", "k5-test", asama=1)}
        for ad in adlar:
            d3 = Path(runs) / ad
            c3 = json.loads((d3 / "config.json").read_text(encoding="utf-8"))
            m3 = json.loads((d3 / "metrics.json").read_text(encoding="utf-8"))
            d1 = a1.get(int(c3["seed"]))
            sha3 = (c3.get("k5") or {}).get("kaynak_sha256") or {}
            sha1 = ((json.loads((d1 / "config.json").read_text(encoding="utf-8")).get("k5") or {}).get("kaynak_sha256")
                    if d1 else None)
            ok = bool(m3.get("ozdeslik_gecti") == 1 and sha1 is not None and
                      all(sha1.get(q) == sha3.get(q) for q in ("best_model.pth", "sonda_agirlik.npz", "tahmin_val.npy")))
            satir[ad] = {"a1": d1.name if d1 else None, "ozdeslik_gecti": m3.get("ozdeslik_gecti"),
                         "ozdeslik_p_maks_fark": m3.get("ozdeslik_p_maks_fark"), "kaynak_sha_ayni": ok}
            if not ok:
                hatalar.append(ad)
    return {"gecti": not hatalar, "hatalar": hatalar, "satir": satir}


def c1_analiz(kip, runs, Pres, y_res, idx_c_res, BS, kuru) -> dict:
    """§4.2–4.3. C1 tahminleri C-test sırasında (C CSV) → resmî indekse (idx_c_res) yerleştirilir.
    kuru: C1 tahmini yerine resmî modellerin aynı (C-val) görüntülerdeki tahminleri (vekil; K1 = 0)."""
    out = {"kol": {}, "kiyas": {}}
    PC = {}
    sira = np.argsort(idx_c_res)          # C CSV sırası → artan resmî indeks (C0 ile aynı bootstrap örneklemi)
    if kuru:
        for k in C1_KOL:
            PC[k] = Pres[k][:, idx_c_res]
        yc = y_res[idx_c_res]
        out["not"] = "KURU: C1 tahmini = resmî modellerin C-val tahminleri (vekil); yalnız kod yolu sınaması"
    else:
        yc = None
        for k, v in C1_KOL.items():
            dosya = "tahmin_test.npy"
            if k == "baseline":   # rakip köprüsü C-test tahminini kendisi yazar (resmî protokol)
                Pk, yk, kk = yukle_kol(runs, v, dosya, "etiket_test.npy", kuyruk="k5-c1")
            else:
                Pk, yk, kk = yukle_kol(runs, v + "-test", dosya, "etiket_test.npy", kuyruk="k5-c1")
            if len(yk) != len(idx_c_res):
                raise SystemExit(f"HATA: C1 {k}: {len(yk)} ≠ {len(idx_c_res)} C-test görüntüsü")
            if not np.array_equal(yk, y_res[idx_c_res]):
                raise SystemExit(f"HATA: C1 {k}: C-test etiketleri resmî testin aynı indeksleriyle aynı değil (sıra/eşleme)")
            yc = yk
            PC[k] = Pk
            out.setdefault("kosumlar", {})[k] = kk
    PC = {k: v[:, sira] for k, v in PC.items()}
    yc = yc[sira]
    idx_c_res = idx_c_res[sira]
    BSc = Bootstrap(len(yc))
    for k in PC:
        acc = (PC[k].argmax(2) == yc[None]).mean(1)
        out["kol"][k] = {"acc_ort": float(acc.mean()), "acc_std": float(acc.std(ddof=1)),
                         "acc_eksi_cogunluk": float(acc.mean() - CTEST_COGUNLUK),
                         "dengeli_ort": float(np.mean([dengeli(yc, PC[k][s].argmax(1)) for s in range(5)])),
                         "dengeli5_ort": float(np.mean([dengeli(yc, PC[k][s].argmax(1), BES) for s in range(5)])),
                         "auc_ort": float(np.mean([auc(yc, PC[k][s]) for s in range(5)]))}
    b = out["kol"]["baseline"]["acc_ort"]
    out["iii_baseline"] = {"acc": b, "beklenti_[0,79;0,85]": bool(0.79 <= b <= 0.85), "gorulmemis_>0,87_INCELE": b > 0.87,
                           "cogunluk_notu": b < CTEST_COGUNLUK + 0.01}
    for A, B in [("baseline", "h01"), ("baseline", "h02"), ("baseline", "h08"), ("baseline", "h10"), ("h02", "h08"),
                 ("h02", "h10"), ("h08", "h10")]:
        out["kiyas"][f"{B}-{A}"] = kiyas(PC[A], PC[B], yc, BSc)
    for ad, cifts, kural in (("aile_C1_1", [("baseline", x) for x in ("h01", "h02", "h08", "h10")],
                              lambda e: e["delta"] >= 0.020 - 1e-12 and e["ga"][0] > 0),
                             ("aile_C1_2", [("h02", "h08"), ("h02", "h10")], lambda e: e["ga"][0] > 0)):
        xs = [f"{B}-{A}" for A, B in cifts]
        pt = [out["kiyas"][x]["t_acc"]["p_tek"] for x in xs]
        pb = [out["kiyas"][x]["acc"]["p_boot"] for x in xs]
        ht, hb = holm(pt), holm(pb)
        out[ad] = {x: {"delta_puan": 100 * out["kiyas"][x]["acc"]["delta"], "ga_puan": [100 * v for v in out["kiyas"][x]["acc"]["ga"]],
                       "holm_t": ht[i], "holm_boot": hb[i], "kural_gecti": bool(kural(out["kiyas"][x]["acc"])),
                       "dengeli_delta_puan": 100 * out["kiyas"][x]["dengeli_acc"]["delta"],
                       "dengeli_ga_puan": [100 * v for v in out["kiyas"][x]["dengeli_acc"]["ga"]],
                       "dengeli5_delta_puan": 100 * out["kiyas"][x]["dengeli_acc_5sinif"]["delta"],
                       "dengeli5_ga_puan": [100 * v for v in out["kiyas"][x]["dengeli_acc_5sinif"]["ga"]]}
                   for i, x in enumerate(xs)}
    out["i"] = {x: {"kural": etiket_kural(out["kiyas"][x]["acc"]["ga"]), "delta_puan": 100 * out["kiyas"][x]["acc"]["delta"],
                    "ga_puan": [100 * v for v in out["kiyas"][x]["acc"]["ga"]],
                    "dengeli_yon": etiket_kural(out["kiyas"][x]["dengeli_acc"]["ga"])}
                for x in ("h10-h02", "h08-h02", "h10-h08")}
    # (ii) KÇ: ortak bootstrap (resmî indeksler); C1 Δ'sı örneklemdeki C-test indekslerinden
    N = BS.N
    dres = {k: (Pres[k].argmax(2) == y_res[None]).mean(0) for k in ("h02", "h08", "h10")}
    dc1 = {}
    for k in ("h02", "h08", "h10"):
        v = np.zeros(N); v[idx_c_res] = (PC[k].argmax(2) == yc[None]).mean(0); dc1[k] = v
    K10 = ortak_kc(dres["h02"], dres["h10"], dc1["h02"], dc1["h10"], idx_c_res, BS)
    K08 = ortak_kc(dres["h02"], dres["h08"], dc1["h02"], dc1["h08"], idx_c_res, BS)
    K0_10 = ortak_kc(dres["h02"], dres["h10"], dres["h02"], dres["h10"], idx_c_res, BS)
    K0_08 = ortak_kc(dres["h02"], dres["h08"], dres["h02"], dres["h08"], idx_c_res, BS)
    fark = K10["b_K"] - K08["b_K"]
    out["ii_KC"] = {"K_H10_puan": 100 * K10["K"], "K_H10_ga": [100 * v for v in K10["K_ga"]],
                    "K_H08_puan": 100 * K08["K"], "K_H08_ga": [100 * v for v in K08["K_ga"]],
                    "K_H10_eksi_K_H08_puan": 100 * (K10["K"] - K08["K"]), "ga_puan": [100 * v for v in ga(fark)],
                    "kural": etiket_kural(ga(fark)).replace("tersi", "çürütüldü"),
                    "bilesen_K0_H10": 100 * K0_10["K"], "bilesen_K1_H10": 100 * (K10["K"] - K0_10["K"]),
                    "bilesen_K0_H08": 100 * K0_08["K"], "bilesen_K1_H08": 100 * (K08["K"] - K0_08["K"]),
                    "oran_K_H10_bolu_delta_resmi": (K10["K"] / K10["delta_resmi"]) if K10["delta_resmi"] != 0 else None}
    oran = out["ii_KC"]["oran_K_H10_bolu_delta_resmi"]
    out["ii_KC"]["ifade_esigi"] = bool(oran is not None and oran >= 0.5 and ga(fark)[0] > 0)
    gw = out["aile_C1_1"]["h10-baseline"]
    out["G3"] = bool(gw["delta_puan"] >= 2.0 - 1e-9 and gw["ga_puan"][0] > 0)
    out["L"] = bool(out["kiyas"]["h10-h02"]["acc"]["ga"][0] > 0)
    out["_PC"] = {k: v for k, v in PC.items()}
    out["_yc"] = yc
    out["_sira"] = sira
    return out


def c2_analiz(kip, runs, c1, kuru) -> dict:
    out = {"kol": {}, "b": {}}
    PC1, yc, sira = c1.pop("_PC"), c1.pop("_yc"), c1.pop("_sira")
    P2 = {}
    if kuru:
        P2 = {k: PC1[k] for k in C2_KOL}
        out["not"] = "KURU: C2 tahmini = C1 vekili (aynı dizi) → C2 − C1 = 0 beklenir; yalnız kod yolu"
    else:
        if kip != "c2":
            return {}
        for k, v in C2_KOL.items():
            if k == "baseline":
                Pk, yk, kk = yukle_kol(runs, v, "tahmin_test.npy", "etiket_test.npy", kuyruk="k5-c2")
            else:
                Pk, yk, kk = yukle_kol(runs, v + "-test", "tahmin_test.npy", "etiket_test.npy", kuyruk="k5-c2")
            Pk, yk = Pk[:, sira], yk[sira]   # C CSV sırası → C1 ile aynı (artan resmî indeks)
            if not np.array_equal(yk, yc):
                raise SystemExit(f"HATA: C2 {k}: C-test etiketleri C1 ile aynı değil")
            P2[k] = Pk
    BSc = Bootstrap(len(yc))
    for k in P2:
        acc = (P2[k].argmax(2) == yc[None]).mean(1)
        out["kol"][k] = {"acc_ort": float(acc.mean()), "acc_std": float(acc.std(ddof=1)),
                         "dengeli_ort": float(np.mean([dengeli(yc, P2[k][s].argmax(1)) for s in range(5)]))}
        r = kiyas(PC1[k], P2[k], yc, BSc)
        out["b"][k] = {"C2_eksi_C1_puan": 100 * r["acc"]["delta"], "ga_puan": [100 * v for v in r["acc"]["ga"]]}
    b = out["kol"]["baseline"]["acc_ort"]
    out["a"] = {"baseline_C2": b, "fark_0,825_puan": 100 * (b - 0.825),
                "sonuc": "yayımlanmış C sonucuyla uyumlu" if abs(b - 0.825) <= 0.020 + 1e-12 else "fark var — olası nedenler yazılır"}
    r = kiyas(P2["h08"], P2["h10"], yc, BSc)
    out["c"] = {"H10_eksi_H08_puan": 100 * r["acc"]["delta"], "ga_puan": [100 * v for v in r["acc"]["ga"]]}
    return out


def rapor(s) -> str:
    def p(x):
        return f"{100 * x:.2f}"
    L = [f"# K5 analiz — kip `{s['kip']}` ({s['tarih']})", "",
         f"n = {s['n']} (S {s['n_S']} / N {s['n_N']}) · tabaka {s.get('n_tabaka')} · betik SHA256 `{s['betik_sha256'][:16]}…` · "
         f"ön-kayıt `{s['onkayit_sha256'][:16]}…`", ""]
    if s["kip"] == "kuru-val":
        L += ["> **KURU KOŞU (P5):** 'test' = resmî VAL, 'C-test' = C-val ∩ val. Test dosyası açılmadı. C1/C2 satırları vekildir.", ""]
    L += ["## Kollar (ort ± std, 5 seed)", "| kol | ACC | AUC | dengeli | dengeli-5 | makro-F1 | ECE | NLL | H04 τ* → ACC / dengeli | R-T ECE |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, o in s["kollar"].items():
        h = o.get("katman_H04"); r = o.get("katman_RT")
        L.append(f"| {AD.get(k, k)} | {o['ort']['acc']:.4f} ± {o['std']['acc']:.4f} | {o['ort']['auc']:.4f} | {o['ort']['dengeli_acc']:.4f} | "
                 f"{o['ort']['dengeli_acc_5sinif']:.4f} | {o['ort']['makro_f1']:.4f} | {o['ort']['ece15']:.4f} | {o['ort']['nll']:.4f} | "
                 + (f"τ*={h['tau']:.1f} → {h['acc_ort']:.4f} / {h['dengeli_ort']:.4f}" if h else "—") + " | "
                 + (f"{r['ece15_ort']:.4f}" if r else "—") + " |")
    for fam, ad in (("aile1", "Aile 1 (rakibe karşı; marj ≥ +2,0 puan ∧ GA alt > 0)"), ("aile2", "Aile 2 (H02c'ye karşı; ≥ +0,5 ∧ GA alt > 0)")):
        L += ["", f"## {ad} — Holm", "| kıyas | Δ ACC (puan) | %95 GA | p_t | p_boot | Holm t | Holm boot | kural | ifade |", "|---|---|---|---|---|---|---|---|---|"]
        for x, e in s[fam].items():
            L.append(f"| {x} | {e['delta_puan']:+.2f} | [{e['ga_puan'][0]:+.2f}; {e['ga_puan'][1]:+.2f}] | {e['p_t']:.3g} | {e['p_boot']:.3g} | "
                     f"{e['holm_t']:.3g} | {e['holm_boot']:.3g} | {'GEÇTİ' if e['kural_gecti'] else 'geçmedi'} | {e['ifade']} |")
    L += ["", "## Doğrulama kuralları ve görülmemiş bayrakları (§2.1–2.2)"]
    for k, e in s["dogrulama"].items():
        L.append(f"- **{k}:** Δ {e['delta_puan']:+.2f} [{e['ga_puan'][0]:+.2f}; {e['ga_puan'][1]:+.2f}] → "
                 f"{'DOĞRULANDI' if e['dogrulandi'] else 'doğrulanmadı'} · bayraklar {e['gorulmemis']}"
                 + (f" · R18 satırı: {e['r18_satiri']}" if e.get("r18_satiri") else "")
                 + (f" · beklenti [+1,0;+4,5] içinde: {e['beklenti_[+1,0;+4,5]_icinde']}" if "beklenti_[+1,0;+4,5]_icinde" in e else ""))
    L += ["", "## S/N ayrışımı (§2.6; kural = HAM Δ_S − Δ_N, K5-ONKAYIT-b §1; tabaka-std ikincil, etiket vermez)",
          "| kıyas | hipotez | ham Δ_S / Δ_N | **ham Δ_S − Δ_N [GA]** | **kural** | ikincil: tab Δ_S / Δ_N | ikincil: tab Δ_S − Δ_N [GA] |",
          "|---|---|---|---|---|---|---|"]
    for x, e in s["SN"].items():
        L.append(f"| {x} | {e['hipotez']} | {e['dS_ham_puan']:+.2f} / {e['dN_ham_puan']:+.2f} | {e['ham_dS_eksi_dN_puan']:+.2f} "
                 f"[{e['ham_ga_puan'][0]:+.2f}; {e['ham_ga_puan'][1]:+.2f}] | {e['kural']} | {e['dS_tab_puan']:+.2f} / {e['dN_tab_puan']:+.2f} | "
                 f"{e['tab_dS_eksi_dN_puan']:+.2f} [{e['tab_ga_puan'][0]:+.2f}; {e['tab_ga_puan'][1]:+.2f}] |")
    c0 = s["C0"]
    L += ["", f"## C0 (§3) — alt küme n = {c0['n']}, çoğunluk tabanı {c0['cogunluk_tabani']:.4f}",
          "| kol | ACC | dengeli | dengeli-5 | ACC − 0,7726 |", "|---|---|---|---|---|"]
    for k, e in c0["kol"].items():
        L.append(f"| {AD.get(k, k)} | {e['acc_ort']:.4f} ± {e['acc_std']:.4f} | {e['dengeli_ort']:.4f} | {e['dengeli5_ort']:.4f} | {e['acc_eksi_cogunluk']:+.4f} |")
    for x, e in c0["kiyas"].items():
        L.append(f"- {e['kural']} {x}: Δ {e['delta_puan']:+.2f} [{e['ga_puan'][0]:+.2f}; {e['ga_puan'][1]:+.2f}] → {e['sonuc']} · "
                 f"dengeli Δ {e['dengeli_delta_puan']:+.2f} [{e['dengeli_ga_puan'][0]:+.2f}; {e['dengeli_ga_puan'][1]:+.2f}]")
    k = c0["KC0"]
    L.append(f"- **KÇ0:** K0_H10 {k['K0_H10_puan']:+.2f}, K0_H08 {k['K0_H08_puan']:+.2f}, fark {k['K0_H10_eksi_K0_H08_puan']:+.2f} "
             f"[{k['ga_puan'][0]:+.2f}; {k['ga_puan'][1]:+.2f}] → {k['kural']} (beklenen [+0,5; +2,5])")
    if s.get("C1"):
        c = s["C1"]
        L += ["", "## C1 (§4)" + (" — KURU VEKİL" if s["kip"] == "kuru-val" else ""), "| kol | ACC | dengeli | dengeli-5 | ACC − 0,7726 |", "|---|---|---|---|---|"]
        for kk, e in c["kol"].items():
            L.append(f"| {kk} | {e['acc_ort']:.4f} | {e['dengeli_ort']:.4f} | {e['dengeli5_ort']:.4f} | {e['acc_eksi_cogunluk']:+.4f} |")
        for x, e in c["i"].items():
            L.append(f"- (i) {x}: Δ {e['delta_puan']:+.2f} [{e['ga_puan'][0]:+.2f}; {e['ga_puan'][1]:+.2f}] → {e['kural']} · dengeli yönü {e['dengeli_yon']}")
        kc = c["ii_KC"]
        L.append(f"- (ii) KÇ: K_H10 {kc['K_H10_puan']:+.2f}, K_H08 {kc['K_H08_puan']:+.2f}, fark {kc['K_H10_eksi_K_H08_puan']:+.2f} "
                 f"[{kc['ga_puan'][0]:+.2f}; {kc['ga_puan'][1]:+.2f}] → {kc['kural']}; ifade eşiği {kc['ifade_esigi']}")
        L.append(f"- (iii) baseline C1 ACC {c['iii_baseline']['acc']:.4f}; G3 {c['G3']}; L {c['L']}")
    if s.get("C2"):
        c = s["C2"]
        L += ["", "## C2 (§5)" + (" — KURU VEKİL" if s["kip"] == "kuru-val" else ""),
              f"- (a) baseline C2 {c['a']['baseline_C2']:.4f} − 0,825 = {c['a']['fark_0,825_puan']:+.2f} puan → {c['a']['sonuc']}"]
        for kk, e in c["b"].items():
            L.append(f"- (b) {kk} C2 − C1 {e['C2_eksi_C1_puan']:+.2f} [{e['ga_puan'][0]:+.2f}; {e['ga_puan'][1]:+.2f}]")
        L.append(f"- (c) H10 − H08 (C2) {c['c']['H10_eksi_H08_puan']:+.2f} [{c['c']['ga_puan'][0]:+.2f}; {c['c']['ga_puan'][1]:+.2f}]")
    L += ["", f"## §7 ölçütleri (mekanik; karar tahlilcide/kullanıcıda): {json.dumps(s['olcutler'], ensure_ascii=False)}"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
