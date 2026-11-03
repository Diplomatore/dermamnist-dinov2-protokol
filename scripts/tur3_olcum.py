# -*- coding: utf-8 -*-
"""
tur3_olcum.py — Tur-3 kart ölçümleri (deney-yürütücü). YALNIZ VALIDATION, CPU. KARAR VERMEZ (karar deney-tahlilcinin);
öldürme kurallarının girdisi olan sayıları üretir.

Yöntem tahlil_tur2.py / tur2_olcum.py ile AYNI:
- Eşleştirilmiş görüntü bootstrap'ı: 2.000, RNG 0, W = bincount(rng.integers(0, N, (2000, N))); uçlar seed-ortalamalı
  görüntü doğruluğundan (ACC, dengeli ACC, molemax-dışı ACC, molemax ACC, nv payı, tabakalı S − N, sınıf duyarlılıkları).
  ACC GA'sı varyantlar/ortak/eslestirilmis_bootstrap.py ile denetlenir.
- Seed-eşleştirilmiş kesin tek yönlü t (scipy.stats.ttest_rel), seed {0..4}.
- Kol başına ikinciller (tahminlerden tek tip): makro-F1, ECE-15, NLL, AUC (+mel/bkl), nv payı, güven − ACC,
  karışıklık hücreleri, seed çifti argmax anlaşmazlığı, toplam hata.
Kullanım (WS kökü): PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I scripts/tur3_olcum.py --kart h07
Çıktı: sonuclar/tur3_<kart>_olcum.json, sonuclar/tur3_<kart>_bootstrap.csv. Test AÇILMAZ (M4).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "varyantlar"))
from ortak.eslestirilmis_bootstrap import eslestirilmis_bootstrap, kosum_bul  # noqa: E402

SINIF = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
K, NV, MEL, BKL, NB, RNG = 7, 5, 4, 2, 2000, 0
KARTLAR = {
    "h07": {
        "kol": {"h02": "h02_r18in_ft224", "h07": "h07_h02_tta8", "h07a_d2": "h07a_h02_tta4d2",
                "h07a_r90": "h07a_h02_tta4r90", "h04": "h04_h02_logitayar", "h07k": "h07k_h07_logitayar",
                "baseline": "baseline", "h07x": "h07x_baseline_tta8", "rt_h02": "rt_h02_sicaklik",
                "rt_h07": "rt_h07_sicaklik"},
        "kiyas": [("h02", "h07", "H07 − H02  [birincil; bootstrap bağlayıcı]"),
                  ("h02", "h07a_d2", "D2 alt kümesi − H02  [ablasyon i]"),
                  ("h02", "h07a_r90", "90° alt kümesi − H02  [ablasyon i]"),
                  ("h07a_r90", "h07a_d2", "D2 − 90°  [ablasyon i, keşifsel yön D2 ≥ 90°]"),
                  ("h04", "h07k", "H07+H04 − H04  [ikinci satır, keşifsel]"),
                  ("h02", "h07k", "H07+H04 − H02  [ikinci satır]"),
                  ("baseline", "h07x", "baseline+TTA − baseline  [M3 yan kolu]")],
    },
    "h08": {"kol": {"h02": "h02_r18in_ft224", "h08": "h08_r18in_ft224_d4", "h04": "h04_h02_logitayar",
                    "h08k": "h08k_h08_logitayar", "rt_h02": "rt_h02_sicaklik", "rt_h08": "rt_h08_sicaklik"},
            "kiyas": [("h02", "h08", "H08 − H02  [birincil]"), ("h04", "h08k", "H08+H04 − H04  [keşifsel]"),
                      ("h02", "h08k", "H08+H04 − H02")]},
    "h09": {"kol": {"h02": "h02_r18in_ft224", "h09": "h09_r18in_ft224_renk02", "h04": "h04_h02_logitayar",
                    "h09k": "h09k_h09_logitayar", "rt_h02": "rt_h02_sicaklik", "rt_h09": "rt_h09_sicaklik"},
            "kiyas": [("h02", "h09", "H09 − H02  [birincil]"), ("h04", "h09k", "H09+H04 − H04  [keşifsel]"),
                      ("h02", "h09k", "H09+H04 − H02")]},
}


def yukle(varyant, dosya="tahmin_val.npy"):
    dz = kosum_bul(varyant, WS, dosya)
    P, seed, ad, y0 = [], [], [], None
    for d in dz:
        c = json.loads((d / "config.json").read_text(encoding="utf-8"))
        if int(c.get("asama", 3)) != 3:
            continue
        y = np.load(d / "etiket_val.npy").reshape(-1).astype(int)
        y0 = y if y0 is None else y0
        assert np.array_equal(y, y0)
        P.append(np.load(d / dosya).astype(np.float64))
        seed.append(int(c["seed"]))
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--kart", required=True, choices=sorted(KARTLAR))
    a = ap.parse_args()
    KOL, KIYAS = KARTLAR[a.kart]["kol"], KARTLAR[a.kart]["kiyas"]
    npz = np.load(WS / "data/medmnist/dermamnist.npz")
    y = npz["val_labels"].ravel().astype(int)
    N = len(y)
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
    tab = np.array([f"{y[i]}|{dxt[i]}|{kay[i]}" for i in range(N)])
    tabakalar = []
    for k in sorted(set(tab)):
        m = tab == k
        if (m & S).any() and (m & ~S).any():
            tabakalar.append((m & S, m & ~S))

    kol = {}
    for ad, v in KOL.items():
        P, yy, seed, kos = yukle(v)
        assert np.array_equal(yy, y), ad
        assert seed == [0, 1, 2, 3, 4], (ad, seed)
        kol[ad] = {"P": P, "kos": kos}

    rng = np.random.default_rng(RNG)
    IDX = rng.integers(0, N, size=(NB, N))
    W = np.zeros((NB, N))
    for b in range(NB):
        W[b] = np.bincount(IDX[b], minlength=N)
    oh = np.stack([(y == c).astype(float) for c in range(K)], 1)

    def uc_noktalar(P):
        d = (P.argmax(2) == y[None]).mean(0)
        out = {"acc": (d.mean(), W @ d / N)}
        rec_b = (W @ (d[:, None] * oh)) / (W @ oh)
        rec = np.array([(d[y == c]).mean() for c in range(K)])
        out["dengeli_acc"] = (rec.mean(), rec_b.mean(1))
        for c in range(K):
            out[f"duyarlilik_{SINIF[c]}"] = (rec[c], rec_b[:, c])
        out["acc_molemax_disi"] = (d[~MM].mean(), (W[:, ~MM] @ d[~MM]) / W[:, ~MM].sum(1))
        out["acc_molemax"] = (d[MM].mean(), (W[:, MM] @ d[MM]) / W[:, MM].sum(1))
        nvp = (P.argmax(2) == NV).mean(0)
        out["nv_payi"] = (nvp.mean(), W @ nvp / N)
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

    U = {ad: uc_noktalar(k["P"]) for ad, k in kol.items()}

    def ga(x):
        return float(np.quantile(x, .025)), float(np.quantile(x, .975))

    sonuc, satirlar = [], []
    for A, B, et in KIYAS:
        r = eslestirilmis_bootstrap(kol[A]["P"], kol[B]["P"], y, NB, RNG)
        e = {"kiyas": et, "A": A, "B": B, "kosum_A": kol[A]["kos"], "kosum_B": kol[B]["kos"]}
        for uc in U[A]:
            pa, ba = U[A][uc]
            pb, bb = U[B][uc]
            fb = bb - ba
            lo, hi = ga(fb)
            e[uc] = {"A": float(pa), "B": float(pb), "delta": float(pb - pa), "ga_alt": lo, "ga_ust": hi,
                     "p_tek_B>A": float((np.sum(fb <= 0) + 1) / (NB + 1))}
            satirlar.append([et, uc, f"{pa:.4f}", f"{pb:.4f}", f"{pb - pa:+.4f}", f"{lo:+.4f}", f"{hi:+.4f}",
                             f"{e[uc]['p_tek_B>A']:.4g}"])
        e["acc_referansla_ayni"] = bool(abs(e["acc"]["ga_alt"] - r["ga_alt"]) < 1e-12 and abs(e["acc"]["ga_ust"] - r["ga_ust"]) < 1e-12)
        e["karar_degisen_goruntu_seed_ort"] = float((kol[A]["P"].argmax(2) != kol[B]["P"].argmax(2)).sum(1).mean())
        aA = (kol[A]["P"].argmax(2) == y[None]).mean(1)
        aB = (kol[B]["P"].argmax(2) == y[None]).mean(1)
        dA = np.array([dengeli(y, kol[A]["P"][s].argmax(1)) for s in range(5)])
        dB = np.array([dengeli(y, kol[B]["P"][s].argmax(1)) for s in range(5)])
        for ad, xa, xb in [("t_acc", aA, aB), ("t_dengeli", dA, dB)]:
            f = xb - xa
            e[ad] = {"A_seed": xa.tolist(), "B_seed": xb.tolist(), "fark_seed": f.tolist(), "ort": float(f.mean()),
                     "sd": float(f.std(ddof=1)),
                     "p_tek_B>A": float(stats.ttest_rel(xb, xa, alternative="greater").pvalue),
                     "p_tek_B<A": float(stats.ttest_rel(xb, xa, alternative="less").pvalue)}
        sonuc.append(e)

    ik = {}
    for ad, k in kol.items():
        P = k["P"]
        m = {q: [] for q in ["acc", "dengeli_acc", "makro_f1", "ece15", "nll", "auc", "auc_mel", "auc_bkl", "nv_payi",
                             "guven_eksi_acc", "acc_molemax_disi", "acc_molemax", "mel_nv", "bkl_nv", "nv_mel",
                             "nv_bkl", "toplam_hata"]}
        for s in range(len(P)):
            t = P[s].argmax(1)
            m["acc"].append((t == y).mean()); m["dengeli_acc"].append(dengeli(y, t))
            m["makro_f1"].append(f1_score(y, t, average="macro", labels=list(range(K)), zero_division=0))
            m["ece15"].append(ece15(y, P[s]))
            m["nll"].append(-np.mean(np.log(np.clip(P[s][np.arange(N), y], 1e-12, 1))))
            ac = [roc_auc_score((y == c).astype(int), P[s][:, c]) for c in range(K)]
            m["auc"].append(np.mean(ac)); m["auc_mel"].append(ac[MEL]); m["auc_bkl"].append(ac[BKL])
            m["nv_payi"].append((t == NV).mean())
            m["guven_eksi_acc"].append(P[s].max(1).mean() - (t == y).mean())
            m["acc_molemax_disi"].append((t[~MM] == y[~MM]).mean()); m["acc_molemax"].append((t[MM] == y[MM]).mean())
            cm = np.zeros((K, K)); np.add.at(cm, (y, t), 1)
            m["mel_nv"].append(cm[MEL, NV]); m["bkl_nv"].append(cm[BKL, NV])
            m["nv_mel"].append(cm[NV, MEL]); m["nv_bkl"].append(cm[NV, BKL]); m["toplam_hata"].append(N - np.trace(cm))
        T = P.argmax(2)
        m["seed_cift_anlasmazlik"] = [float((T[i] != T[j]).mean()) for i in range(5) for j in range(i + 1, 5)]
        ik[ad] = {"ort": {q: float(np.mean(v)) for q, v in m.items()}, "std": {q: float(np.std(v, ddof=1)) for q, v in m.items()},
                  "kosumlar": k["kos"]}

    ek = {}
    if a.kart == "h07":
        G = np.stack([np.load(WS / "runs" / d / "tahmin_val_gorunum.npy").astype(np.float64) for d in kol["h07"]["kos"]])  # (5,8,N,K)
        acc_g = (G.argmax(3) == y[None, None]).mean(2)  # (5, 8)
        ek["gorunum_acc_seed_x_gorunum"] = acc_g.tolist()
        ek["gorunum_acc_ort"] = acc_g.mean(0).tolist()
        ek["donusmus_eksi_kimlik_ort_puan"] = float(100 * (acc_g[:, 1:].mean() - acc_g[:, 0].mean()))
        ek["donusmus_eksi_kimlik_gorunum_puan"] = (100 * (acc_g[:, 1:].mean(0) - acc_g[:, 0].mean())).tolist()
        ek["gorunumler_arasi_argmax_anlasmazlik_ort"] = float(np.mean([(G[s].argmax(2) != G[s, 0].argmax(1)[None]).any(0).mean() for s in range(5)]))
        ek["kimlik_ozdeslik"] = [json.loads((WS / "runs" / d / "metrics.json").read_text(encoding="utf-8"))["ozdeslik_kimlik_maks_fark"] for d in kol["h07"]["kos"]]
        ek["h07x_kimlik_ozdeslik"] = [json.loads((WS / "runs" / d / "metrics.json").read_text(encoding="utf-8"))["ozdeslik_kimlik_maks_fark"] for d in kol["h07x"]["kos"]]
        mk = [json.loads((WS / "runs" / d / "metrics.json").read_text(encoding="utf-8")) for d in kol["h07k"]["kos"]]
        ek["h07k_tau"] = {"tau_kat": [mk[0][f"tau_kat_{i}"] for i in range(1, 6)], "tau_tamval": mk[0]["tau_tamval"],
                         "val_dengeli_acc_tamval": float(np.mean([x["val_dengeli_acc_tamval"] for x in mk])),
                         "val_acc_tamval": float(np.mean([x["val_acc_tamval"] for x in mk]))}
        rt = [json.loads((WS / "runs" / d / "metrics.json").read_text(encoding="utf-8")) for d in kol["rt_h07"]["kos"]]
        ek["rt_h07"] = {"oof_ece_ort": float(np.mean([x["val_ece"] for x in rt])), "oof_ece_std": float(np.std([x["val_ece"] for x in rt], ddof=1)),
                        "ham_ece_ort": float(np.mean([x["val_ece_ham"] for x in rt])), "T_tamval": [x["T_tamval"] for x in rt]}
        sure = [json.loads((WS / "runs" / d / "metrics.json").read_text(encoding="utf-8")) for d in kol["h07"]["kos"]]
        ek["cikarim_sure_sn"] = [x["cikarim_sure_sn"] for x in sure]
        ek["cikarim_img_s"] = [x["cikarim_img_s"] for x in sure]
    out = {"kart": a.kart, "kiyaslar": sonuc, "ikincil": ik, "ek": ek,
           "yontem": "eşleştirilmiş görüntü bootstrap 2.000 RNG 0 (tahlil_tur2 ile aynı W); kesin tek yönlü t scipy ttest_rel"}
    (WS / "sonuclar" / f"tur3_{a.kart}_olcum.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(WS / "sonuclar" / f"tur3_{a.kart}_bootstrap.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["kiyas", "uc", "A", "B", "delta", "ga_alt", "ga_ust", "p_tek_B>A"]); w.writerows(satirlar)
    for e in sonuc:
        print(f"{e['kiyas']}: ACC {e['acc']['A']:.4f}→{e['acc']['B']:.4f} Δ {100*e['acc']['delta']:+.2f} "
              f"[{100*e['acc']['ga_alt']:+.2f}; {100*e['acc']['ga_ust']:+.2f}] p_t(B>A) {e['t_acc']['p_tek_B>A']:.4g} "
              f"sd_Δ {100*e['t_acc']['sd']:.2f} | dengeli Δ {100*e['dengeli_acc']['delta']:+.2f} "
              f"[{100*e['dengeli_acc']['ga_alt']:+.2f}; {100*e['dengeli_acc']['ga_ust']:+.2f}] | "
              f"molemax-dışı Δ {100*e['acc_molemax_disi']['delta']:+.2f} [{100*e['acc_molemax_disi']['ga_alt']:+.2f}; "
              f"{100*e['acc_molemax_disi']['ga_ust']:+.2f}] | S−N Δ {100*e['sizinti_tabakali_SN']['delta']:+.2f} "
              f"[{100*e['sizinti_tabakali_SN']['ga_alt']:+.2f}; {100*e['sizinti_tabakali_SN']['ga_ust']:+.2f}] | "
              f"karar değişen {e['karar_degisen_goruntu_seed_ort']:.1f} | ref GA aynı {e['acc_referansla_ayni']}")
    print(json.dumps({k: {q: round(v, 4) for q, v in ik[k]["ort"].items()} for k in ik}, ensure_ascii=False))
    print(json.dumps(ek, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
