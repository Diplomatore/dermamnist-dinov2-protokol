# -*- coding: utf-8 -*-
"""
tahlil_tur1.py — K4c Tur-1 tahlili (deney-tahlilci, Mod 2). YALNIZ VALIDATION, CPU.

Ne yapar (tekrar üretilebilir; RNG 0, 2.000 yeniden örnekleme):
  1. Eşleştirilmiş görüntü bootstrap'ı (varyantlar/ortak/eslestirilmis_bootstrap.py ile AYNI tanım ve RNG):
     baseline'a karşı her tam varyant; H03 basamakları arası; r224−r28 (T2); H02−H01; H02(aucsecim) kolları.
     Ek uçlar (aynı yeniden örnekleme indeksleriyle): dengeli ACC, makro-F1 değil (doğrusal değil, atlanır),
     molemax-dışı ACC, N (lezyonu train'de olmayan) alt kümesi ACC, tabakalı S−N (sınıf × dx_type × kaynak, N-ağırlıklı).
  2. Seed-eşleştirilmiş t-testleri (leaderboard p'lerinin bağımsız doğrulaması).
  3. Tahminlerden tek tip ikincil metrikler (baseline dahil): ACC, dengeli ACC, makro-F1, ECE-15, NLL, nv payı,
     molemax-dışı ACC, sınıf duyarlılıkları, mel/bkl AUC, tabakalı S−N.
  4. Teşhisler (KARAR VERMEZ, val'de ayarlanır → iyimser): (a) nv payı gerçek paya (0,669) eşitlenince ACC/dengeli ACC;
     (b) sıcaklık ölçekleme sonrası ECE (T val'de uydurulur); (c) r128/r224 öznitelikleriyle lbfgs C∈{0,01; 0,1}
     (H03 dengeli ACC sıçramasının C* ile ilişkisi); (d) koşumlar arası seed bağlılığı ve a2↔a3 tekrar üretilebilirlik.
Çıktı: sonuclar/tahlil_tur1_bootstrap.json, sonuclar/tahlil_tur1_bootstrap.csv, sonuclar/tahlil_tur1_ikincil.csv,
       sonuclar/tahlil_tur1_teshis.json
Kullanım (WS kökü): PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I scripts/tahlil_tur1.py
Test tahminleri/etiketleri AÇILMAZ (M4).
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "varyantlar"))
from ortak.eslestirilmis_bootstrap import cift_fark, eslestirilmis_bootstrap, kosum_bul  # noqa: E402

SINIF = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
K, NV, NB, RNG = 7, 5, 2000, 0
KOL = {  # ad → (varyant, tahmin dosyası)
    "baseline": ("baseline", "tahmin_val.npy"),
    "r28": ("h03_dinov2b_lp_r28", "tahmin_val.npy"),
    "r64": ("h03_dinov2b_lp_r64", "tahmin_val.npy"),
    "r128": ("h03_dinov2b_lp_r128", "tahmin_val.npy"),
    "h01_r224": ("h01_dinov2b_lp_r224", "tahmin_val.npy"),
    "h02": ("h02_r18in_ft224", "tahmin_val.npy"),
    "h02_aucsecim": ("h02_r18in_ft224", "tahmin_val_aucsecim.npy"),
    "h02_son": ("h02_r18in_ft224", "tahmin_val_son.npy"),
}
KIYAS = [  # (A referans, B varyant, etiket)
    ("baseline", "h01_r224", "H01 − B"),
    ("baseline", "h02", "H02 − B"),
    ("baseline", "h02_aucsecim", "H02(aucsecim) − B  [kural b'']"),
    ("baseline", "r28", "r28 − B  [T3]"),
    ("baseline", "r64", "r64 − B"),
    ("baseline", "r128", "r128 − B"),
    ("r28", "r64", "r64 − r28  [T1]"),
    ("r64", "r128", "r128 − r64  [T1]"),
    ("r128", "h01_r224", "r224 − r128  [T1]"),
    ("r28", "h01_r224", "r224 − r28  [T2]"),
    ("h01_r224", "h02", "H02 − H01  [ebeveyn kuralı]"),
    ("h01_r224", "h02_aucsecim", "H02(aucsecim) − H01"),
]


def yukle(varyant, dosya):
    dz = kosum_bul(varyant, WS, dosya)
    P, seed, ad = [], [], []
    y0 = None
    for d in dz:
        y = np.load(d / "etiket_val.npy").reshape(-1).astype(int)
        y0 = y if y0 is None else y0
        assert np.array_equal(y, y0)
        P.append(np.load(d / dosya).astype(np.float64))
        seed.append(int(json.loads((d / "metrics.json").read_text(encoding="utf-8"))["seed"]))
        ad.append(d.name)
    o = np.argsort(seed)
    return np.stack(P)[o], y0, [seed[i] for i in o], [ad[i] for i in o]


def ece15(y, p):
    g, d = p.max(1), (p.argmax(1) == y).astype(float)
    s = np.linspace(0, 1, 16)
    t = 0.0
    for i in range(15):
        m = (g > s[i]) & (g <= s[i + 1]) if i else (g >= s[i]) & (g <= s[i + 1])
        if m.any():
            t += m.mean() * abs(d[m].mean() - g[m].mean())
    return float(t)


def dengeli(y, t):
    return float(np.mean([(t[y == c] == c).mean() for c in range(K)]))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    # ---- meta: S/N, dx_type, kaynak (hata_kirilim.py ile aynı tanım) ----
    npz = np.load(WS / "data/medmnist/dermamnist.npz")
    y_npz = npz["val_labels"].ravel().astype(int)
    N = len(y_npz)
    tr_lez, val_lez, vimg = set(), {}, {}
    with open(WS / "data/splits/dermamnist_lezyon_esleme.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "train":
                tr_lez.add(r["lesion_id"])
            elif r["split"] == "val":
                val_lez[int(r["index"])] = r["lesion_id"]
                vimg[int(r["index"])] = r["image_id"]
    S = np.array([val_lez[i] in tr_lez for i in range(N)])
    meta = {}
    with open(WS / "data/meta/HAM10000_metadata.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            meta[r["image_id"]] = r
    dxt = np.array([meta[vimg[i]]["dx_type"] for i in range(N)])
    kay = np.array([meta[vimg[i]]["dataset"] for i in range(N)])
    MM = kay == "vidir_molemax"
    tab = np.array([f"{y_npz[i]}|{dxt[i]}|{kay[i]}" for i in range(N)])
    tabakalar = []
    for k in sorted(set(tab)):
        m = tab == k
        if (m & S).any() and (m & ~S).any():
            tabakalar.append((m & S, m & ~S))

    # ---- kolları yükle ----
    kol = {}
    for ad, (v, dosya) in KOL.items():
        P, y, seed, kos = yukle(v, dosya)
        assert np.array_equal(y, y_npz), ad
        kol[ad] = {"P": P, "seed": seed, "kos": kos}
    y = y_npz

    # ---- bootstrap ağırlıkları (eslestirilmis_bootstrap ile aynı RNG akışı: rng.integers(0,N,(n,N))) ----
    rng = np.random.default_rng(RNG)
    IDX = rng.integers(0, N, size=(NB, N))
    W = np.zeros((NB, N))
    for b in range(NB):
        W[b] = np.bincount(IDX[b], minlength=N)

    def dbar(P):
        return (P.argmax(2) == y[None]).mean(0)

    def uc_noktalar(d):
        """d: görüntü başına seed-ortalamalı doğruluk (N,). Dönüş: nokta değerleri ve bootstrap dağılımları."""
        out = {}
        out["acc"] = (d.mean(), W @ d / N)
        oh = np.stack([(y == c).astype(float) for c in range(K)], 1)  # (N,K)
        rec_b = (W @ (d[:, None] * oh)) / (W @ oh)  # (NB,K)
        out["dengeli_acc"] = (np.mean([(d[y == c]).mean() for c in range(K)]), rec_b.mean(1))
        for adk, m in [("acc_molemax_disi", ~MM), ("acc_N", ~S), ("acc_S", S), ("acc_N_molemax_disi", ~S & ~MM)]:
            out[adk] = (d[m].mean(), (W[:, m] @ d[m]) / W[:, m].sum(1))
        # tabakalı S−N (N-ağırlıklı)
        top, w = 0.0, 0.0
        topb, wb = np.zeros(NB), np.zeros(NB)
        for mS, mN in tabakalar:
            wk = mN.sum()
            top += wk * (d[mS].mean() - d[mN].mean()); w += wk
            sS, sN = W[:, mS].sum(1), W[:, mN].sum(1)
            ok = (sS > 0) & (sN > 0)
            fark = np.where(ok, (W[:, mS] @ d[mS]) / np.maximum(sS, 1) - (W[:, mN] @ d[mN]) / np.maximum(sN, 1), 0)
            topb += np.where(ok, sN * fark, 0); wb += np.where(ok, sN, 0)
        out["sizinti_tabakali_SN"] = (top / w, topb / wb)
        return out

    U = {ad: uc_noktalar(dbar(k["P"])) for ad, k in kol.items()}

    def ga(a):
        return float(np.quantile(a, .025)), float(np.quantile(a, .975))

    # ---- 1. kıyaslar ----
    sonuc, satirlar = [], []
    for A, B, et in KIYAS:
        r = eslestirilmis_bootstrap(kol[A]["P"], kol[B]["P"], y, NB, RNG)  # referans uygulama (kontrol)
        e = {"kiyas": et, "A": A, "B": B, "seed_A": kol[A]["seed"], "seed_B": kol[B]["seed"],
             "kosum_A": kol[A]["kos"], "kosum_B": kol[B]["kos"], "referans_betik": r}
        for uc in U[A]:
            pa, ba = U[A][uc]
            pb, bb = U[B][uc]
            fb = bb - ba
            lo, hi = ga(fb)
            e[uc] = {"A": float(pa), "B": float(pb), "delta": float(pb - pa), "ga_alt": lo, "ga_ust": hi,
                     "p_tek_B>A": float((np.sum(fb <= 0) + 1) / (NB + 1)),
                     "p_tek_B<A": float((np.sum(fb >= 0) + 1) / (NB + 1))}
            satirlar.append([et, uc, f"{pa:.4f}", f"{pb:.4f}", f"{pb - pa:+.4f}", f"{lo:+.4f}", f"{hi:+.4f}",
                             f"{e[uc]['p_tek_B>A']:.4g}"])
        # tutarlılık: kendi acc bootstrap'ı referans betikle aynı mı
        e["acc_referansla_ayni"] = bool(abs(e["acc"]["ga_alt"] - r["ga_alt"]) < 1e-12 and abs(e["acc"]["ga_ust"] - r["ga_ust"]) < 1e-12)
        # seed-eşleştirilmiş t-testi (ACC), yalnız seed kümeleri aynıysa
        if kol[A]["seed"] == kol[B]["seed"]:
            aA = (kol[A]["P"].argmax(2) == y[None]).mean(1)
            aB = (kol[B]["P"].argmax(2) == y[None]).mean(1)
            t = stats.ttest_rel(aB, aA, alternative="greater")
            t2 = stats.ttest_rel(aB, aA)
            fark = aB - aA
            e["t_testi"] = {"fark_seed": fark.tolist(), "ort": float(fark.mean()), "sd": float(fark.std(ddof=1)),
                            "t": float(t.statistic), "p_tek": float(t.pvalue), "p_iki": float(t2.pvalue),
                            "mde_80": float((2.1318 + 0.8416) * fark.std(ddof=1) / np.sqrt(len(fark)))}
            # dengeli ACC için de
            dA = [dengeli(y, kol[A]["P"][s].argmax(1)) for s in range(len(aA))]
            dB = [dengeli(y, kol[B]["P"][s].argmax(1)) for s in range(len(aB))]
            td = stats.ttest_rel(dB, dA, alternative="greater")
            e["t_testi_dengeli"] = {"ort": float(np.mean(np.array(dB) - np.array(dA))), "p_tek": float(td.pvalue)}
        sonuc.append(e)

    # ---- 2. ikincil metrikler (tek tip, tahminlerden) ----
    ik_satir = []
    ik = {}
    for ad, k in kol.items():
        P = k["P"]
        m = {"acc": [], "dengeli_acc": [], "makro_f1": [], "ece15": [], "nll": [], "nv_payi": [], "auc": [],
             "acc_molemax_disi": [], "ort_guven": []}
        rec = []
        aucs = []
        for s in range(len(P)):
            t = P[s].argmax(1)
            m["acc"].append((t == y).mean()); m["dengeli_acc"].append(dengeli(y, t))
            m["makro_f1"].append(f1_score(y, t, average="macro", labels=list(range(K)), zero_division=0))
            m["ece15"].append(ece15(y, P[s]))
            m["nll"].append(-np.mean(np.log(np.clip(P[s][np.arange(N), y], 1e-12, 1))))
            m["nv_payi"].append((t == NV).mean()); m["ort_guven"].append(P[s].max(1).mean())
            ac = [roc_auc_score((y == c).astype(int), P[s][:, c]) for c in range(K)]
            aucs.append(ac); m["auc"].append(np.mean(ac))
            m["acc_molemax_disi"].append((t[~MM] == y[~MM]).mean())
            rec.append([(t[y == c] == c).mean() for c in range(K)])
        rec, aucs = np.array(rec), np.array(aucs)
        for c in range(K):
            m[f"duyarlilik_{SINIF[c]}"] = rec[:, c].tolist()
            m[f"auc_{SINIF[c]}"] = aucs[:, c].tolist()
        # karışıklık (seed ortalaması)
        CM = np.zeros((len(P), K, K))
        for s in range(len(P)):
            np.add.at(CM[s], (y, P[s].argmax(1)), 1)
        cm = CM.mean(0)
        m["hata_nv_mel"] = [float(c[5, 4] + c[4, 5]) for c in CM]
        m["hata_nv_bkl"] = [float(c[5, 2] + c[2, 5]) for c in CM]
        m["hata_mel_bkl"] = [float(c[4, 2] + c[2, 4]) for c in CM]
        m["toplam_hata"] = [float(N - np.trace(c)) for c in CM]
        m["ucgen_payi"] = [(a + b + c) / t for a, b, c, t in zip(m["hata_nv_mel"], m["hata_nv_bkl"], m["hata_mel_bkl"], m["toplam_hata"])]
        T = P.argmax(2)
        m["seed_cift_anlasmazlik"] = [float((T[i] != T[j]).mean()) for i in range(len(P)) for j in range(i + 1, len(P))] or [0.0]
        ik[ad] = {"ort": {q: float(np.mean(v)) for q, v in m.items()}, "std": {q: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for q, v in m.items()},
                  "karisiklik_ort": cm.tolist(), "n": len(P)}
        for q, v in m.items():
            ik_satir.append([ad, q, f"{np.mean(v):.5f}", f"{np.std(v, ddof=1):.5f}" if len(v) > 1 else "", len(v)])

    # ---- 3. teşhisler ----
    tes = {}
    # (a) nv payını gerçek paya (671/1003) eşitleyen nv log-olasılık kaydırması (seed başına, val'de ayarlı → teşhis)
    hedef = float((y == NV).mean())
    nvesit = {}
    for ad in ["baseline", "r28", "r64", "r128", "h01_r224", "h02"]:
        P = kol[ad]["P"]
        r_acc, r_den = [], []
        for s in range(len(P)):
            L = np.log(np.clip(P[s], 1e-12, 1))
            lo, hi = -20.0, 20.0
            for _ in range(60):
                mid = (lo + hi) / 2
                L2 = L.copy(); L2[:, NV] += mid
                if (L2.argmax(1) == NV).mean() > hedef:
                    hi = mid
                else:
                    lo = mid
            L2 = L.copy(); L2[:, NV] += lo
            t = L2.argmax(1)
            r_acc.append((t == y).mean()); r_den.append(dengeli(y, t))
        nvesit[ad] = {"acc": float(np.mean(r_acc)), "acc_std": float(np.std(r_acc, ddof=1)),
                      "dengeli": float(np.mean(r_den)), "hedef_nv_payi": hedef}
    tes["nv_payi_esitleme"] = nvesit
    # (b) sıcaklık ölçekleme (T val NLL'sinde uydurulur — iyimser; argmax değişmez)
    tsc = {}
    for ad in ["baseline", "h01_r224", "h02"]:
        P = kol[ad]["P"]
        r = []
        for s in range(len(P)):
            L = np.log(np.clip(P[s], 1e-12, 1))
            best = None
            for Tt in np.linspace(0.5, 3.0, 251):
                Q = np.exp(L / Tt); Q /= Q.sum(1, keepdims=True)
                nll = -np.mean(np.log(np.clip(Q[np.arange(N), y], 1e-12, 1)))
                if best is None or nll < best[0]:
                    best = (nll, Tt, ece15(y, Q))
            r.append(best)
        tsc[ad] = {"T": [x[1] for x in r], "ece_sonra_ort": float(np.mean([x[2] for x in r])),
                   "ece_once_ort": ik[ad]["ort"]["ece15"]}
    tes["sicaklik_olcekleme"] = tsc
    # (c) lbfgs C ∈ {0,01; 0,1} r128 ve r224 (C_sk = C/2; StandardScaler yalnız train) — dengeli ACC sıçramasının C*'la ilişkisi
    cl = {}
    for r_ in ["r128", "r224"]:
        z = np.load(WS / f"data/oznitelik/dinov2_vitb14_{r_}.npz")
        Xtr, ytr, Xva, yva = z["train_oz"].astype(np.float64), z["train_etiket"].ravel().astype(int), z["val_oz"].astype(np.float64), z["val_etiket"].ravel().astype(int)
        assert np.array_equal(yva, y)
        mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
        Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
        for C in [0.01, 0.1]:
            lr = LogisticRegression(C=C / 2, solver="lbfgs", max_iter=5000, tol=1e-6).fit(Xtr, ytr)
            t = lr.predict_proba(Xva).argmax(1)
            cl[f"{r_}_C{C}"] = {"acc": float((t == y).mean()), "dengeli": dengeli(y, t), "nv_payi": float((t == NV).mean()),
                               "duyarlilik": [float((t[y == c] == c).mean()) for c in range(K)]}
    tes["lbfgs_C_teshisi"] = cl
    # (d) seed bağlılığı (koşumlar arası) ve a2↔a3 tekrar üretilebilirlik
    sb = {ad: cift_fark(k["P"]) for ad, k in kol.items() if ad in ["r28", "r64", "r128", "h01_r224", "h02"]}
    tes["seed_tahmin_fark_kosumlar_arasi"] = sb
    tekrar = {}
    for a2v, a3v in [("h01_dinov2b_lp_r224_onucus", "h01_dinov2b_lp_r224"), ("h03_dinov2b_lp_r28_onucus", "h03_dinov2b_lp_r28")]:
        P2, _, s2, _ = yukle(a2v, "tahmin_val.npy")
        P3, _, s3, _ = yukle(a3v, "tahmin_val.npy")
        for i, s in enumerate(s2):
            j = s3.index(s)
            tekrar[f"{a3v}_s{s}"] = {"maks_mutlak_fark_a2_a3": float(np.max(np.abs(P2[i] - P3[j]))),
                                     "argmax_ayni_oran": float((P2[i].argmax(1) == P3[j].argmax(1)).mean())}
    tes["a2_a3_tekrar"] = tekrar
    tes["alt_kume_boyutlari"] = {"N_val": int(N), "S": int(S.sum()), "N": int((~S).sum()), "molemax": int(MM.sum()),
                                 "molemax_disi": int((~MM).sum()), "N_molemax_disi": int((~S & ~MM).sum()),
                                 "tabaka_sayisi": len(tabakalar), "tabakali_kapsam_N": float(sum(mN.sum() for _, mN in tabakalar) / (~S).sum())}

    out = WS / "sonuclar"
    (out / "tahlil_tur1_bootstrap.json").write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(out / "tahlil_tur1_bootstrap.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kiyas", "uc_nokta", "A", "B", "delta", "ga_alt_95", "ga_ust_95", "p_tek_B>A"])
        w.writerows(satirlar)
    with open(out / "tahlil_tur1_ikincil.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kol", "olcu", "ort", "std", "n_seed"])
        w.writerows(ik_satir)
    (out / "tahlil_tur1_teshis.json").write_text(json.dumps({"ikincil": ik, "teshis": tes}, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- konsol özeti ----
    for e in sonuc:
        a = e["acc"]
        tt = e.get("t_testi", {})
        print(f"{e['kiyas']:<34} ΔACC {a['delta']*100:+6.2f} GA[{a['ga_alt']*100:+.2f},{a['ga_ust']*100:+.2f}] pbs={a['p_tek_B>A']:.4g}"
              f" | t p={tt.get('p_tek', float('nan')):.4g} sdΔ={tt.get('sd', float('nan'))*100:.2f} MDE={tt.get('mde_80', float('nan'))*100:.2f}"
              f" | Δden {e['dengeli_acc']['delta']*100:+.2f} [{e['dengeli_acc']['ga_alt']*100:+.2f},{e['dengeli_acc']['ga_ust']*100:+.2f}]"
              f" | ΔmmD {e['acc_molemax_disi']['delta']*100:+.2f} [{e['acc_molemax_disi']['ga_alt']*100:+.2f},{e['acc_molemax_disi']['ga_ust']*100:+.2f}]"
              f" | ΔN {e['acc_N']['delta']*100:+.2f} [{e['acc_N']['ga_alt']*100:+.2f},{e['acc_N']['ga_ust']*100:+.2f}]"
              f" | ΔSN {e['sizinti_tabakali_SN']['delta']*100:+.2f} [{e['sizinti_tabakali_SN']['ga_alt']*100:+.2f},{e['sizinti_tabakali_SN']['ga_ust']*100:+.2f}]"
              f" ref={e['acc_referansla_ayni']}")
    for ad in kol:
        u = U[ad]["sizinti_tabakali_SN"]
        print(f"tabakalı S−N {ad:<14} {u[0]*100:+.2f} GA[{ga(u[1])[0]*100:+.2f},{ga(u[1])[1]*100:+.2f}]  N-ACC {U[ad]['acc_N'][0]:.4f} S-ACC {U[ad]['acc_S'][0]:.4f}")
    print(json.dumps(tes, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
