# -*- coding: utf-8 -*-
"""
tahlil_tur4.py — Tur-4 tahlil ölçümleri (deney-tahlilci, K4c). YALNIZ VALIDATION, CPU. KARAR VERMEZ; tahlil-tur4.md'nin
sayılarını üretir. Test tahmini/etiketi OKUNMAZ (M4): yalnız tahmin_val.npy / etiket_val.npy / metrics.json / gecmis.json.

Ortam kuralı (SIRALAMA "ÖN-KAYIT — COLAB TEKRAR KAPISI"): kıyaslar YALNIZ Colab kollarıyla
  baseline_colab, h01_dinov2b_lp_r224_colab, h02_r18in_ft224_colab, h08_r18in_ft224_d4_colab, h10_dinov2b_k4_ft224,
  h11_dinov2s_ft224 — hepsi aşama 3, seed 0–4, kök = Colab Drive eşitlemesi (--kok).
Yöntem tur3_olcum.py / tahlil_tur2.py ile AYNI:
- Eşleştirilmiş görüntü bootstrap'ı 2.000, RNG 0, W = bincount(rng.integers(0, N, (2000, N))); uç noktalar seed-ortalamalı
  görüntü doğruluğundan. ACC GA'sı varyantlar/ortak/eslestirilmis_bootstrap.py ile denetlenir (acc_referansla_ayni).
- Seed-eşleştirilmiş kesin tek yönlü t (scipy.stats.ttest_rel), seed {0..4}.
- Ek uç noktalar (Tur-4 ezber sorusu): S (lezyonu train'de) ve N alt kümelerinde ham ACC, molemax-dışı S/N, tabakalı S − N
  (sınıf × dx_type × kaynak), df/vasc hariç 5 sınıflı dengeli ACC.
- Katmanlar (analiz olarak, koşum üretmeden): H04 logit ayarı OOF (logit_ayari.oof_tau; aynı katlar.json, τ 0:1:0,1,
  kısıt = kolun kendi ham ACC'si − 0,010) ve R-T (sicaklik.T_sec; aynı katlar, T 0,5:3,0:0,01, OOF ECE-15).
Kullanım (WS kökü): PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I scripts/tahlil_tur4.py --kok "<DRIVE_KOK>/3-1"
Çıktı: sonuclar/tahlil_tur4_olcum.json, sonuclar/tahlil_tur4_bootstrap.csv, sonuclar/tahlil_tur4_ikincil.csv.
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
sys.path.insert(0, str(WS / "varyantlar" / "logit_ayari"))
from ortak.eslestirilmis_bootstrap import eslestirilmis_bootstrap, kosum_bul  # noqa: E402
import logit_ayari as LA  # noqa: E402
import sicaklik as SC  # noqa: E402

SINIF = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
K, NV, MEL, BKL, NB, RNG = 7, 5, 4, 2, 2000, 0
BES = [0, 1, 2, 4, 5]  # df (3) ve vasc (6) hariç
KOL = {"baseline": "baseline_colab", "h01": "h01_dinov2b_lp_r224_colab", "h02": "h02_r18in_ft224_colab",
       "h08": "h08_r18in_ft224_d4_colab", "h10": "h10_dinov2b_k4_ft224", "h11": "h11_dinov2s_ft224"}
KATMAN_KOL = ["h01", "h02", "h08", "h10", "h11"]
KIYAS = [("h02", "h10", "H10 − H02c  [BİRİNCİL karar; bootstrap (b') bağlayıcı]"),
         ("h02", "h11", "H11 − H02c  [BİRİNCİL karar; bootstrap (b') bağlayıcı]"),
         ("h01", "h10", "H10 − H01c  [temsil, karar vermez: sonda → son-4-blok]"),
         ("h08", "h10", "H10 − H08c  [bilgi; farklı aileler, K5 kazanan seçimi]"),
         ("h10", "h11", "H11 − H10  [Pareto; yön ön-kaydı yok]"),
         ("h01", "h11", "H11 − H01c  [çapraz: boyut + ince ayar paketi]"),
         ("h08", "h11", "H11 − H08c  [bilgi]"),
         ("h02", "h08", "H08c − H02c  [referans: Colab'da H08 adaylığı]"),
         ("h02", "h01", "H01c − H02c  [referans]"),
         ("baseline", "h10", "H10 − baseline_c  [zincir bilgisi]"),
         ("h02k", "h10k", "H10+H04 − H02c+H04  [ikinci satır]"),
         ("h02k", "h11k", "H11+H04 − H02c+H04  [ikinci satır]"),
         ("h08k", "h10k", "H10+H04 − H08c+H04  [ikinci satır, bilgi]"),
         ("h10", "h10k", "H10+H04 − H10  [katmanın kendi bedeli/kazancı]")]


def yukle(kok, varyant, dosya="tahmin_val.npy"):
    P, seed, ad, y0, M = [], [], [], None, []
    for d in kosum_bul(varyant, kok, dosya):
        c = json.loads((d / "config.json").read_text(encoding="utf-8"))
        if int(c.get("asama", 3)) != 3:
            continue
        y = np.load(d / "etiket_val.npy").reshape(-1).astype(int)
        y0 = y if y0 is None else y0
        assert np.array_equal(y, y0)
        P.append(np.load(d / dosya).astype(np.float64))
        seed.append(int(c["seed"]))
        ad.append(d)
        M.append(json.loads((d / "metrics.json").read_text(encoding="utf-8")))
    o = np.argsort(seed)
    return np.stack(P)[o], y0, [seed[i] for i in o], [ad[i] for i in o], [M[i] for i in o]


def ece15(y, p):
    g, d = p.max(1), (p.argmax(1) == y).astype(float)
    s = np.linspace(0, 1, 16)
    t = 0.0
    for i in range(15):
        m = (g > s[i]) & (g <= s[i + 1]) if i else (g >= s[i]) & (g <= s[i + 1])
        if m.any():
            t += m.mean() * abs(d[m].mean() - g[m].mean())
    return float(t)


def dengeli(y, t, sinif=range(K)):
    return float(np.mean([(t[y == c] == c).mean() for c in sinif]))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--kok", required=True)
    a = ap.parse_args()
    kok = Path(a.kok)
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

    # ───── kollar + bütünlük denetimleri ─────
    kol, denetim = {}, {}
    for ad, v in KOL.items():
        P, yy, seed, dz, M = yukle(kok, v)
        assert np.array_equal(yy, y), ad
        assert seed == [0, 1, 2, 3, 4], (ad, seed)
        kol[ad] = {"P": P, "kos": [d.name for d in dz], "dz": dz, "M": M}
        acc_npy = (P.argmax(2) == y[None]).mean(1)
        dn = {"npy_metrics_acc_maks_fark": float(np.max(np.abs(acc_npy - np.array([m["val_acc"] for m in M])))),
              "test_anahtari": sum(1 for m in M for k in m if k.startswith("test")),
              "seed_cift_maks_p_fark_min": float(min(np.max(np.abs(P[i] - P[j])) for i in range(5) for j in range(i + 1, 5)))}
        if ad in ("h10", "h11"):
            G = [json.loads((d / "gecmis.json").read_text(encoding="utf-8")) for d in dz]
            cf = [json.loads((d / "config.json").read_text(encoding="utf-8")) for d in dz]
            dn["ebeveyn"] = [c.get("ebeveyn") for c in cf]
            dn["en_iyi_epoch"] = [m["en_iyi_epoch"] for m in M]
            dn["gecmis_eniyi_val_acc_esit"] = [abs(max(e["val_acc"] for e in g) - m["val_acc"]) < 1e-12 for g, m in zip(G, M)]
            dn["kural_e_15_ve_son3_artis"] = sum(1 for g, m in zip(G, M) if m["en_iyi_epoch"] == 15 and
                                               g[-1]["val_acc"] > g[-2]["val_acc"] > g[-3]["val_acc"])
            dn["train_loss_ep5"] = [g[4]["train_loss"] for g in G]
            dn["train_loss_son"] = [g[-1]["train_loss"] for g in G]
            dn["val_acc_egrisi_ort"] = np.mean([[e["val_acc"] for e in g] for g in G], 0).tolist()
            dn["val_acc_son_ort"] = float(np.mean([m["val_acc_son"] for m in M]))
            dn["val_acc_aucsecim_ort"] = float(np.mean([m["val_acc_aucsecim"] for m in M]))
            for q in ("epoch_sure_ort_sn", "egitim_sure_sn", "egitim_img_s_kararli", "param_egitilen", "param_sayisi",
                      "tepe_gpu_bellek_mb", "n_train", "val_auc", "val_acc", "onbellek_tutarlilik_maks_fark"):
                dn[q] = [m.get(q) for m in M]
        if ad in ("h02", "h08"):
            G = [json.loads((d / "gecmis.json").read_text(encoding="utf-8")) for d in dz]
            dn["train_loss_ep5"] = [g[4]["train_loss"] for g in G]
            dn["en_iyi_epoch"] = [m.get("en_iyi_epoch") for m in M]
            dn["epoch_sure_ort_sn"] = [m.get("epoch_sure_ort_sn") for m in M]
        denetim[ad] = dn

    # ───── katmanlar: H04 logit ayarı OOF + R-T (analiz; koşum üretmez) ─────
    KJ = json.loads(LA.KATLAR.read_text(encoding="utf-8"))
    kat = np.array(KJ["kat"], dtype=np.int64)
    pi = LA.train_pi()
    logpi = np.log(pi)
    taus = LA.izgara("0:1:0.1")
    Ts = LA.izgara("0.5:3.0:0.01")
    katman = {}
    for ad in KATMAN_KOL:
        P = kol[ad]["P"]
        L = np.log(np.maximum(P, 1e-300))
        tau_k, _ = LA.oof_tau(L, logpi, y, kat, taus, 0.010)
        tau_t, _ = LA.tau_sec(L, logpi, y, np.arange(N), taus, 0.010)
        Poof = np.stack([LA.softmax(L[s] - tau_k[kat][:, None] * logpi[None, :]) for s in range(5)])
        kol[ad + "k"] = {"P": Poof, "kos": kol[ad]["kos"]}
        ece_oof, ece_ham, ece_tam, Tt = [], [], [], []
        for s in range(5):
            Tk = np.array([SC.T_sec(L[s], y, np.where(kat != k)[0], Ts)[0] for k in range(int(kat.max()) + 1)])
            p_oof = LA.softmax(L[s] / Tk[kat][:, None])
            assert np.array_equal(p_oof.argmax(1), P[s].argmax(1))
            T_t = SC.T_sec(L[s], y, np.arange(N), Ts)[0]
            ece_oof.append(ece15(y, p_oof)); ece_ham.append(ece15(y, P[s])); ece_tam.append(ece15(y, LA.softmax(L[s] / T_t)))
            Tt.append(T_t)
        t_oof = Poof.argmax(2)
        katman[ad] = {"tau_kat": tau_k.tolist(), "tau_tamval": tau_t,
                      "oof_acc_ort": float(np.mean([(t_oof[s] == y).mean() for s in range(5)])),
                      "oof_dengeli_ort": float(np.mean([dengeli(y, t_oof[s]) for s in range(5)])),
                      "oof_dengeli_std": float(np.std([dengeli(y, t_oof[s]) for s in range(5)], ddof=1)),
                      "oof_dengeli5_ort": float(np.mean([dengeli(y, t_oof[s], BES) for s in range(5)])),
                      "ham_acc_ort": float((P.argmax(2) == y[None]).mean()),
                      "ham_dengeli_ort": float(np.mean([dengeli(y, P[s].argmax(1)) for s in range(5)])),
                      "rt_oof_ece_ort": float(np.mean(ece_oof)), "rt_oof_ece_std": float(np.std(ece_oof, ddof=1)),
                      "rt_ham_ece_ort": float(np.mean(ece_ham)), "rt_tamval_ece_ort": float(np.mean(ece_tam)), "rt_T_tamval": Tt}

    # ───── bootstrap ─────
    rng = np.random.default_rng(RNG)
    IDX = rng.integers(0, N, size=(NB, N))
    W = np.zeros((NB, N))
    for b in range(NB):
        W[b] = np.bincount(IDX[b], minlength=N)
    oh = np.stack([(y == c).astype(float) for c in range(K)], 1)

    def alt(d, m):
        return d[m].mean(), (W[:, m] @ d[m]) / W[:, m].sum(1)

    def uc_noktalar(P):
        d = (P.argmax(2) == y[None]).mean(0)
        out = {"acc": (d.mean(), W @ d / N)}
        rec_b = (W @ (d[:, None] * oh)) / (W @ oh)
        rec = np.array([(d[y == c]).mean() for c in range(K)])
        out["dengeli_acc"] = (rec.mean(), rec_b.mean(1))
        out["dengeli_acc_5sinif"] = (rec[BES].mean(), rec_b[:, BES].mean(1))
        for c in range(K):
            out[f"duyarlilik_{SINIF[c]}"] = (rec[c], rec_b[:, c])
        out["acc_molemax_disi"] = alt(d, ~MM)
        out["acc_molemax"] = alt(d, MM)
        out["acc_S_ham"] = alt(d, S)
        out["acc_N_ham"] = alt(d, ~S)
        out["acc_S_molemaxdisi"] = alt(d, S & ~MM)
        out["acc_N_molemaxdisi"] = alt(d, ~S & ~MM)
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

    kol_SN = {ad: {"nokta": float(U[ad]["sizinti_tabakali_SN"][0]), "ga": ga(U[ad]["sizinti_tabakali_SN"][1]),
                   "acc_S": float(U[ad]["acc_S_ham"][0]), "acc_N": float(U[ad]["acc_N_ham"][0]),
                   "acc_S_mmd": float(U[ad]["acc_S_molemaxdisi"][0]), "acc_N_mmd": float(U[ad]["acc_N_molemaxdisi"][0])}
              for ad in kol}
    # seed başına tabakalı S − N (metrics.json; eşleşme denetimi)
    for ad in KOL:
        kol_SN[ad]["seed_metrics"] = [m.get("val_sizinti_tabakali_SN") for m in kol[ad]["M"]]

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
        # kazancın S/N ayrışımı: Δ = (n_S/N)Δ_S + (n_N/N)Δ_N
        dA = (kol[A]["P"].argmax(2) == y[None]).mean(0)
        dB = (kol[B]["P"].argmax(2) == y[None]).mean(0)
        top = (dB - dA).sum()
        e["ayrisim"] = {"n_S": int(S.sum()), "n_N": int((~S).sum()),
                        "S_katkisi_puan": float(100 * (dB - dA)[S].sum() / N), "N_katkisi_puan": float(100 * (dB - dA)[~S].sum() / N),
                        "S_payi": float((dB - dA)[S].sum() / top) if top != 0 else None,
                        "S_goruntu_payi": float(S.mean()),
                        "net_kazanilan_goruntu_S": float((dB - dA)[S].sum()), "net_kazanilan_goruntu_N": float((dB - dA)[~S].sum())}
        aA = (kol[A]["P"].argmax(2) == y[None]).mean(1)
        aB = (kol[B]["P"].argmax(2) == y[None]).mean(1)
        xA = np.array([dengeli(y, kol[A]["P"][s].argmax(1)) for s in range(5)])
        xB = np.array([dengeli(y, kol[B]["P"][s].argmax(1)) for s in range(5)])
        for ad, xa, xb in [("t_acc", aA, aB), ("t_dengeli", xA, xB)]:
            f = xb - xa
            e[ad] = {"A_seed": xa.tolist(), "B_seed": xb.tolist(), "fark_seed": f.tolist(), "ort": float(f.mean()),
                     "sd": float(f.std(ddof=1)),
                     "p_tek_B>A": float(stats.ttest_rel(xb, xa, alternative="greater").pvalue),
                     "p_tek_B<A": float(stats.ttest_rel(xb, xa, alternative="less").pvalue)}
        sonuc.append(e)

    ik = {}
    for ad in KOL:
        P = kol[ad]["P"]
        m = {q: [] for q in ["acc", "dengeli_acc", "dengeli_acc_5sinif", "makro_f1", "ece15", "nll", "auc", "auc_mel",
                             "auc_bkl", "nv_payi", "guven_eksi_acc", "acc_molemax_disi", "acc_molemax", "mel_nv", "bkl_nv",
                             "nv_mel", "nv_bkl", "toplam_hata", "duyarlilik_mel"]}
        for s in range(5):
            t = P[s].argmax(1)
            m["acc"].append((t == y).mean()); m["dengeli_acc"].append(dengeli(y, t)); m["dengeli_acc_5sinif"].append(dengeli(y, t, BES))
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
            m["duyarlilik_mel"].append((t[y == MEL] == MEL).mean())
        T_ = P.argmax(2)
        m["seed_cift_anlasmazlik"] = [float((T_[i] != T_[j]).mean()) for i in range(5) for j in range(i + 1, 5)]
        ik[ad] = {"ort": {q: float(np.mean(v)) for q, v in m.items()}, "std": {q: float(np.std(v, ddof=1)) for q, v in m.items()},
                  "kosumlar": kol[ad]["kos"]}

    # ───── bütçe (L4): .done sure_sn, kuyruk bazında ─────
    butce = {}
    for d in sorted((kok / "runs").glob("*/")):
        cf, dn = d / "config.json", d / ".done"
        if not (cf.exists() and dn.exists()):
            continue
        c = json.loads(cf.read_text(encoding="utf-8")); j = json.loads(dn.read_text(encoding="utf-8"))
        q = c.get("kuyruk", "?")
        butce.setdefault(q, {"n": 0, "sure_sn": 0.0})
        butce[q]["n"] += 1; butce[q]["sure_sn"] += float(j.get("sure_sn") or 0)
    for q in butce:
        butce[q]["sa"] = butce[q]["sure_sn"] / 3600

    out = {"kok": str(kok), "kiyaslar": sonuc, "ikincil": ik, "katman": katman, "denetim": denetim, "kol_SN": kol_SN,
           "butce_kosum_toplami": butce, "n_S": int(S.sum()), "n_N": int((~S).sum()), "n_tabaka": len(tabakalar),
           "yontem": "eşleştirilmiş görüntü bootstrap 2.000 RNG 0 (tur3_olcum ile aynı W); kesin tek yönlü t scipy ttest_rel; "
                     "H04 OOF = logit_ayari.oof_tau (katlar.json, τ 0:1:0,1, kısıt 0,010); R-T = sicaklik.T_sec (OOF ECE-15)"}
    (WS / "sonuclar" / "tahlil_tur4_olcum.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    with open(WS / "sonuclar" / "tahlil_tur4_bootstrap.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["kiyas", "uc", "A", "B", "delta", "ga_alt", "ga_ust", "p_tek_B>A"]); w.writerows(satirlar)
    with open(WS / "sonuclar" / "tahlil_tur4_ikincil.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        qs = list(ik["h10"]["ort"])
        w.writerow(["kol"] + [f"{q}_ort" for q in qs] + [f"{q}_std" for q in qs])
        for ad in ik:
            w.writerow([KOL[ad]] + [f"{ik[ad]['ort'][q]:.5f}" for q in qs] + [f"{ik[ad]['std'][q]:.5f}" for q in qs])
    for e in sonuc:
        print(f"{e['kiyas']}: ACC {e['acc']['A']:.4f}→{e['acc']['B']:.4f} Δ {100*e['acc']['delta']:+.2f} "
              f"[{100*e['acc']['ga_alt']:+.2f}; {100*e['acc']['ga_ust']:+.2f}] p_t {e['t_acc']['p_tek_B>A']:.3g} sd {100*e['t_acc']['sd']:.2f} "
              f"| den Δ {100*e['dengeli_acc']['delta']:+.2f} [{100*e['dengeli_acc']['ga_alt']:+.2f}; {100*e['dengeli_acc']['ga_ust']:+.2f}] "
              f"5s {100*e['dengeli_acc_5sinif']['delta']:+.2f} [{100*e['dengeli_acc_5sinif']['ga_alt']:+.2f}; {100*e['dengeli_acc_5sinif']['ga_ust']:+.2f}] "
              f"| S {100*e['acc_S_ham']['delta']:+.2f} [{100*e['acc_S_ham']['ga_alt']:+.2f}; {100*e['acc_S_ham']['ga_ust']:+.2f}] "
              f"N {100*e['acc_N_ham']['delta']:+.2f} [{100*e['acc_N_ham']['ga_alt']:+.2f}; {100*e['acc_N_ham']['ga_ust']:+.2f}] "
              f"S-N(tab) Δ {100*e['sizinti_tabakali_SN']['delta']:+.2f} [{100*e['sizinti_tabakali_SN']['ga_alt']:+.2f}; {100*e['sizinti_tabakali_SN']['ga_ust']:+.2f}] "
              f"| S payı {e['ayrisim']['S_payi']} | ref {e['acc_referansla_ayni']}")
    print(json.dumps({k: {q: round(v, 4) for q, v in ik[k]["ort"].items()} for k in ik}, ensure_ascii=False))
    print(json.dumps(katman, ensure_ascii=False, default=float))
    print(json.dumps({k: {q: v for q, v in d.items() if q not in ("val_acc_egrisi_ort",)} for k, d in denetim.items()}, ensure_ascii=False, default=float))
    print(json.dumps(kol_SN, ensure_ascii=False, default=float))
    print(json.dumps(butce, ensure_ascii=False))


if __name__ == "__main__":
    main()
