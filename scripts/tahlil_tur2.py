# -*- coding: utf-8 -*-
"""
tahlil_tur2.py — K4c Tur-2 tahlili (deney-tahlilci, Mod 2). YALNIZ VALIDATION, CPU, GPU KULLANMAZ.

Ne yapar (tekrar üretilebilir; RNG 0, 2.000 yeniden örnekleme; tahlil_tur1.py ile AYNI bootstrap tanımı):
  1. Kıyaslar (eşleştirilmiş görüntü bootstrap'ı + kesin tek yönlü seed-eşleştirilmiş t, scipy):
     H05 − H02 (birincil; H05 kartı), aucsecim ve son-epoch kolları, H05 − H04 (kart ikincil 11, keşifsel),
     H05 − H02@τ=0,5 (Bayes eşdeğeri post-hoc, seçimsiz), H04 − H02 ve H06 − H01 (tur2_cpu_olcum doğrulaması).
     Uçlar: ACC, dengeli ACC, molemax-dışı ACC, tabakalı S − N, sınıf duyarlılıkları.
  2. Kol başına ikincil metrikler (tahminlerden tek tip): makro-F1, ECE-15, NLL, AUC, nv payı, güven − ACC,
     karışıklık hücreleri, sınıf başına tahmin sayısı.
  3. H04 denetimleri: OOF kararlarının katlar.json + τ_k ile bağımsız yeniden hesabı; seed başına τ = 1 zarfı;
     s3 sınıf ayrışımı; ACC kısıtının bootstrap olasılığı.
  4. H02 τ eğrisi (tam val, seçimsiz, betimsel): ACC ↔ dengeli ACC ödünleşim sınırı ve H05'in bu sınıra göre yeri.
  5. H05 epoch eğrisi (gecmis.json val_dengeli_acc; Byrd & Lipton sönme sorusu).
Çıktı: sonuclar/tahlil_tur2_bootstrap.json, sonuclar/tahlil_tur2_bootstrap.csv, sonuclar/tahlil_tur2_ikincil.csv,
       sonuclar/tahlil_tur2_teshis.json
Kullanım (WS kökü): PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I scripts/tahlil_tur2.py
Test tahminleri/etiketleri AÇILMAZ (M4).
"""
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
KOL = {
    "h02": ("h02_r18in_ft224", "tahmin_val.npy"),
    "h02_auc": ("h02_r18in_ft224", "tahmin_val_aucsecim.npy"),
    "h02_son": ("h02_r18in_ft224", "tahmin_val_son.npy"),
    "h05": ("h05_r18in_ft224_cw05", "tahmin_val.npy"),
    "h05_auc": ("h05_r18in_ft224_cw05", "tahmin_val_aucsecim.npy"),
    "h05_son": ("h05_r18in_ft224_cw05", "tahmin_val_son.npy"),
    "h04": ("h04_h02_logitayar", "tahmin_val.npy"),
    "h01": ("h01_dinov2b_lp_r224", "tahmin_val.npy"),
    "h06": ("h06_h01_logitayar", "tahmin_val.npy"),
}
KIYAS = [
    ("h02", "h05", "H05 − H02  [birincil]"),
    ("h02_auc", "h05_auc", "H05(aucsecim) − H02(aucsecim)"),
    ("h02_son", "h05_son", "H05(son ep) − H02(son ep)"),
    ("h04", "h05", "H05 − H04(OOF)  [kart ikincil 11]"),
    ("h02_tau05", "h05", "H05 − H02@τ=0,5  [eğitim-zamanı vs post-hoc eşdeğer]"),
    ("h02", "h02_tau05", "H02@τ=0,5 − H02  [teşhis, seçimsiz]"),
    ("h02", "h04", "H04 − H02  [doğrulama]"),
    ("h01", "h06", "H06 − H01  [doğrulama]"),
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


def softmax(L):
    L = L - L.max(1, keepdims=True)
    e = np.exp(L)
    return e / e.sum(1, keepdims=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    npz = np.load(WS / "data/medmnist/dermamnist.npz")
    y_npz = npz["val_labels"].ravel().astype(int)
    ytr = npz["train_labels"].ravel().astype(int)
    pi = np.bincount(ytr, minlength=K) / len(ytr)
    logpi = np.log(pi)
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

    kol = {}
    for ad, (v, dosya) in KOL.items():
        P, y, seed, kos = yukle(v, dosya)
        assert np.array_equal(y, y_npz), ad
        assert seed == [0, 1, 2, 3, 4], (ad, seed)
        kol[ad] = {"P": P, "seed": seed, "kos": kos}
    y = y_npz
    # türetilmiş kol: H02 + sabit τ = 0,5 (H05'in β = 0,5'inin Bayes eşdeğeri; seçim YOK → iyimserlik yok)
    L02 = np.log(np.maximum(kol["h02"]["P"], 1e-300))
    kol["h02_tau05"] = {"P": np.stack([softmax(L02[s] - 0.5 * logpi) for s in range(5)]), "seed": kol["h02"]["seed"],
                        "kos": ["türetilmiş: h02 tahmin_val − 0,5·log π_train"]}

    rng = np.random.default_rng(RNG)
    IDX = rng.integers(0, N, size=(NB, N))
    W = np.zeros((NB, N))
    for b in range(NB):
        W[b] = np.bincount(IDX[b], minlength=N)
    oh = np.stack([(y == c).astype(float) for c in range(K)], 1)

    def dbar(P):
        return (P.argmax(2) == y[None]).mean(0)

    def uc_noktalar(d, P):
        out = {}
        out["acc"] = (d.mean(), W @ d / N)
        rec_b = (W @ (d[:, None] * oh)) / (W @ oh)
        rec = np.array([(d[y == c]).mean() for c in range(K)])
        out["dengeli_acc"] = (rec.mean(), rec_b.mean(1))
        for c in range(K):
            out[f"duyarlilik_{SINIF[c]}"] = (rec[c], rec_b[:, c])
        out["acc_molemax_disi"] = (d[~MM].mean(), (W[:, ~MM] @ d[~MM]) / W[:, ~MM].sum(1))
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

    U = {ad: uc_noktalar(dbar(k["P"]), k["P"]) for ad, k in kol.items()}

    def ga(a):
        return float(np.quantile(a, .025)), float(np.quantile(a, .975))

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
                     "p_tek_B>A": float((np.sum(fb <= 0) + 1) / (NB + 1)),
                     "p_tek_B<A": float((np.sum(fb >= 0) + 1) / (NB + 1))}
            satirlar.append([et, uc, f"{pa:.4f}", f"{pb:.4f}", f"{pb - pa:+.4f}", f"{lo:+.4f}", f"{hi:+.4f}",
                             f"{e[uc]['p_tek_B>A']:.4g}"])
        e["acc_referansla_ayni"] = bool(abs(e["acc"]["ga_alt"] - r["ga_alt"]) < 1e-12 and abs(e["acc"]["ga_ust"] - r["ga_ust"]) < 1e-12)
        e["farkli_dogru_goruntu"] = r["farkli_dogru_goruntu"]
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
                     "p_tek_B<A": float(stats.ttest_rel(xb, xa, alternative="less").pvalue),
                     "mde_80": float((2.1318 + 0.8416) * f.std(ddof=1) / np.sqrt(5))}
        sonuc.append(e)

    # ---- ikincil metrikler ----
    ik, ik_satir = {}, []
    for ad, k in kol.items():
        P = k["P"]
        m = {q: [] for q in ["acc", "dengeli_acc", "makro_f1", "ece15", "nll", "auc", "nv_payi", "guven_eksi_acc",
                             "acc_molemax_disi", "auc_mel", "auc_bkl", "mel_nv", "bkl_nv", "nv_mel", "nv_bkl",
                             "toplam_hata"]}
        rec, npred = [], []
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
            m["acc_molemax_disi"].append((t[~MM] == y[~MM]).mean())
            cm = np.zeros((K, K)); np.add.at(cm, (y, t), 1)
            m["mel_nv"].append(cm[MEL, NV]); m["bkl_nv"].append(cm[BKL, NV])
            m["nv_mel"].append(cm[NV, MEL]); m["nv_bkl"].append(cm[NV, BKL]); m["toplam_hata"].append(N - np.trace(cm))
            rec.append([(t[y == c] == c).mean() for c in range(K)]); npred.append(np.bincount(t, minlength=K))
        rec, npred = np.array(rec), np.array(npred)
        for c in range(K):
            m[f"duyarlilik_{SINIF[c]}"] = rec[:, c].tolist()
            m[f"tahmin_sayisi_{SINIF[c]}"] = npred[:, c].tolist()
        T = P.argmax(2)
        m["seed_cift_anlasmazlik"] = [float((T[i] != T[j]).mean()) for i in range(5) for j in range(i + 1, 5)]
        ik[ad] = {"ort": {q: float(np.mean(v)) for q, v in m.items()}, "std": {q: float(np.std(v, ddof=1)) for q, v in m.items()},
                  "seed": {q: [float(x) for x in v] for q, v in m.items() if len(v) == 5}}
        for q, v in m.items():
            ik_satir.append([ad, q, f"{np.mean(v):.5f}", f"{np.std(v, ddof=1):.5f}", len(v)])

    tes = {"pi_train": pi.tolist()}
    # ---- H04 denetimleri ----
    KJ = json.loads((WS / "varyantlar/logit_ayari/katlar.json").read_text(encoding="utf-8"))
    kat = np.array(KJ["kat"])
    h04_kos = kol["h04"]["kos"]
    tau_k = json.loads((WS / "runs" / h04_kos[0] / "tau_secim.json").read_text(encoding="utf-8"))["tau_kat"]
    tau_i = np.array([tau_k[k] for k in kat])
    uyusmaz = []
    for s in range(5):
        t_yeni = (L02[s] - tau_i[:, None] * logpi[None]).argmax(1)
        uyusmaz.append(int((t_yeni != kol["h04"]["P"][s].argmax(1)).sum()))
    tes["h04_oof_bagimsiz_yeniden_hesap"] = {"tau_kat": tau_k, "argmax_uyusmaz_goruntu_seed": uyusmaz}
    # seed başına τ ızgarası (tam val, seçimsiz, betimsel) + sınıf ayrışımı
    izgara = np.round(np.arange(0, 1.0001, 0.1), 1)
    egri = {"tau": izgara.tolist(), "acc_ort": [], "dengeli_ort": [], "acc_seed": [], "dengeli_seed": []}
    for tau in izgara:
        acc_s = [((L02[s] - tau * logpi).argmax(1) == y).mean() for s in range(5)]
        den_s = [dengeli(y, (L02[s] - tau * logpi).argmax(1)) for s in range(5)]
        egri["acc_ort"].append(float(np.mean(acc_s))); egri["dengeli_ort"].append(float(np.mean(den_s)))
        egri["acc_seed"].append([float(a) for a in acc_s]); egri["dengeli_seed"].append([float(a) for a in den_s])
    tes["h02_tau_egrisi_tamval_betimsel"] = egri
    d02 = np.array(egri["dengeli_seed"][0])
    tes["h04_seed_zarfi"] = {
        "dengeli_delta_tau07_tamval_seed": (np.array(egri["dengeli_seed"][7]) - d02).tolist(),
        "dengeli_delta_tau1_seed": (np.array(egri["dengeli_seed"][10]) - d02).tolist(),
        "dengeli_delta_oof_seed": (np.array(ik["h04"]["seed"]["dengeli_acc"]) - d02).tolist(),
        "h02_dengeli_seed": d02.tolist()}
    sinif_ayr = {}
    for s in range(5):
        t0, t1 = kol["h02"]["P"][s].argmax(1), kol["h04"]["P"][s].argmax(1)
        sinif_ayr[f"s{s}"] = {SINIF[c]: [int((t0[y == c] == c).sum()), int((t1[y == c] == c).sum()), int((y == c).sum())] for c in range(K)}
        sinif_ayr[f"s{s}"]["dengeli_katki_puan"] = {SINIF[c]: float(100 * (((t1[y == c] == c).mean()) - ((t0[y == c] == c).mean())) / K) for c in range(K)}
    tes["h04_sinif_ayrisimi_dogru_sayisi_[h02,h04,destek]"] = sinif_ayr
    # Grubbs benzeri: s3 Δ'nın seed Δ dağılımındaki z'si
    dd = np.array(tes["h04_seed_zarfi"]["dengeli_delta_oof_seed"])
    tes["h04_s3_z"] = float((dd[3] - dd.mean()) / dd.std(ddof=1))
    tes["h04_s3_z_kritik_grubbs_n5_a05"] = 1.715
    # ACC kısıtı belirsizliği: bootstrap Δacc < −0,010 olasılığı
    fb = U["h04"]["acc"][1] - U["h02"]["acc"][1]
    tes["h04_acc_kisit"] = {"delta": float(U["h04"]["acc"][0] - U["h02"]["acc"][0]),
                            "P_boot_delta_lt_-0.010": float(np.mean(fb < -0.010)),
                            "ga95": list(ga(fb)), "pay_puan": float(100 * (U["h04"]["acc"][0] - U["h02"]["acc"][0] + 0.010)),
                            "pay_goruntu_karari_5seed_toplam": float((U["h04"]["acc"][0] - (U["h02"]["acc"][0] - 0.010)) * N * 5),
                            "seed_kisit_alti": [int(a < b - 0.010) for a, b in zip(ik["h04"]["seed"]["acc"], ik["h02"]["seed"]["acc"])]}
    # ---- H05'in H02 τ sınırına göre yeri (betimsel; aynı ACC'de post-hoc ne kadar dengeli veriyor) ----
    acc05, den05 = ik["h05"]["ort"]["acc"], ik["h05"]["ort"]["dengeli_acc"]
    ao, do = np.array(egri["acc_ort"]), np.array(egri["dengeli_ort"])
    # ACC'ye göre doğrusal ara değer (ACC τ ile azalan)
    o = np.argsort(ao)
    den_ayni_acc = float(np.interp(acc05, ao[o], do[o]))
    tes["h05_sinira_gore"] = {"h05_acc": acc05, "h05_dengeli": den05,
                              "h02_sinirinda_ayni_acc_dengeli": den_ayni_acc,
                              "fark_puan": 100 * (den05 - den_ayni_acc)}
    # ---- H05 epoch eğrisi ----
    ep = []
    for d in kol["h05"]["kos"]:
        g = json.loads((WS / "runs" / d / "gecmis.json").read_text(encoding="utf-8"))
        ep.append([(r["epoch"], r["val_acc"], r["val_dengeli_acc"]) for r in g])
    ep = np.array(ep)  # (5, 30, 3)
    tes["h05_epoch_egrisi"] = {
        "dengeli_ort_epoch": ep[:, :, 2].mean(0).tolist(), "acc_ort_epoch": ep[:, :, 1].mean(0).tolist(),
        "dengeli_ep1_5": float(ep[:, :5, 2].mean()), "dengeli_ep6_15": float(ep[:, 5:15, 2].mean()),
        "dengeli_ep16_30": float(ep[:, 15:, 2].mean()), "acc_ep1_5": float(ep[:, :5, 1].mean()),
        "acc_ep16_30": float(ep[:, 15:, 1].mean()),
        "secilen_epoch": [int(json.loads((WS / "runs" / d / "metrics.json").read_text(encoding="utf-8"))["en_iyi_epoch"]) for d in kol["h05"]["kos"]],
        "dengeli_eniyi_epoch_basina_max": ep[:, :, 2].max(1).tolist()}
    # test anahtarı denetimi (M4)
    test_say = 0
    for ad in ["h05", "h04", "h06"]:
        for d in kol[ad]["kos"]:
            for fn in ["metrics.json", "gecmis.json"]:
                p = WS / "runs" / d / fn
                if p.exists():
                    test_say += p.read_text(encoding="utf-8").count('"test')
            test_say += len(list((WS / "runs" / d).glob("*test*")))
    tes["test_anahtar_dosya_sayisi"] = test_say

    out = WS / "sonuclar"
    (out / "tahlil_tur2_bootstrap.json").write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(out / "tahlil_tur2_bootstrap.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["kiyas", "uc", "A", "B", "delta", "ga_alt", "ga_ust", "p_tek_B>A"]); w.writerows(satirlar)
    with open(out / "tahlil_tur2_ikincil.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["kol", "metrik", "ort", "std", "n"]); w.writerows(ik_satir)
    tes["ikincil"] = ik
    (out / "tahlil_tur2_teshis.json").write_text(json.dumps(tes, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- konsol özeti ----
    for e in sonuc:
        print(f"\n== {e['kiyas']}  (acc GA referansla aynı: {e['acc_referansla_ayni']}; karar değişen/seed {e['karar_degisen_goruntu_seed_ort']:.1f})")
        for uc in ["acc", "dengeli_acc", "acc_molemax_disi", "nv_payi", "sizinti_tabakali_SN"] + [f"duyarlilik_{c}" for c in SINIF]:
            v = e[uc]
            print(f"  {uc:24s} A {v['A']:.4f} B {v['B']:.4f} Δ {100*v['delta']:+.2f} [{100*v['ga_alt']:+.2f}; {100*v['ga_ust']:+.2f}] pB>A {v['p_tek_B>A']:.4f}")
        for ad in ["t_acc", "t_dengeli"]:
            t = e[ad]
            print(f"  {ad}: Δseed {[round(100*x,2) for x in t['fark_seed']]} ort {100*t['ort']:+.2f} sd {100*t['sd']:.2f} p>{t['p_tek_B>A']:.4f} p<{t['p_tek_B<A']:.4f} MDE {100*t['mde_80']:.2f}")
    print("\nikincil (ort):")
    for ad in ik:
        o_ = ik[ad]["ort"]
        print(f"  {ad:10s} acc {o_['acc']:.4f} den {o_['dengeli_acc']:.4f} f1 {o_['makro_f1']:.4f} ece {o_['ece15']:.4f} nll {o_['nll']:.4f} auc {o_['auc']:.4f} nv {o_['nv_payi']:.4f} g-a {o_['guven_eksi_acc']:+.4f} mm {o_['acc_molemax_disi']:.4f} mel→nv {o_['mel_nv']:.1f} nv→mel {o_['nv_mel']:.1f} bkl→nv {o_['bkl_nv']:.1f} nv→bkl {o_['nv_bkl']:.1f} hata {o_['toplam_hata']:.1f} anlaşmazlık {o_['seed_cift_anlasmazlik']:.3f}")
        print("             duy " + " ".join(f"{c} {o_['duyarlilik_'+c]:.3f}" for c in SINIF) + " | tahmin " + " ".join(f"{o_['tahmin_sayisi_'+c]:.1f}" for c in SINIF))
    print("\nteşhis:", json.dumps({k: v for k, v in tes.items() if k not in ("ikincil", "h02_tau_egrisi_tamval_betimsel", "h04_sinif_ayrisimi_dogru_sayisi_[h02,h04,destek]")}, ensure_ascii=False)[:4000])
    print("τ eğrisi:", [(t, round(a, 4), round(d, 4)) for t, a, d in zip(egri["tau"], egri["acc_ort"], egri["dengeli_ort"])])
    print("s3 ayrışım:", sinif_ayr["s3"])


if __name__ == "__main__":
    main()
