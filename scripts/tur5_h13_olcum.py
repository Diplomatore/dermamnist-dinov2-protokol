# -*- coding: utf-8 -*-
"""
tur5_h13_olcum.py — H13 (Tur-5) kart ölçümleri (deney-yürütücü; YALNIZ VAL, CPU). KARAR VERMEZ (karar tahlilcide).
Tanımlar tahlil_tur4.py / tahlil_tur4_ezber.py / tahlil_tur4_cval.py ile AYNI (o betikler değiştirilmedi; KOL listeleri
sabit olduğundan aynı tanımlar burada H13 kollarıyla yeniden kuruldu):
- Eşleştirilmiş görüntü bootstrap'ı 2.000, RNG 0, W = bincount(rng.integers(0, N, (2000, N))); ACC GA'sı
  varyantlar/ortak/eslestirilmis_bootstrap.py ile denetlenir (acc_referansla_ayni).
- Seed-eşleştirilmiş kesin tek yönlü t (scipy.stats.ttest_rel, alternative="greater"), seed {0..4}.
- S/N: S = lezyonu train'de olan val görüntüsü; tabakalı S − N (sınıf × dx_type × kaynak, N-ağırlıklı); tabaka-std Δ_S/Δ_N.
- C-val: resmî val ∩ C-val (split_C == val ve split_resmi == val), alt küme içi bootstrap 2.000 RNG 0.
- H04 OOF katmanı H10 için logit_ayari.oof_tau ile (tahlil_tur4 ile aynı) yeniden hesaplanır; H13 için h13k_h13_logitayar koşumları okunur.
Kollar: Colab üyeleri runs_colab/ (Drive arşiv kopyası); topluluk kolları runs/ (yerel, aşama 3, rc = 0).
Kullanım (WS kökü): PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I scripts/tur5_h13_olcum.py
Çıktı: sonuclar/tur5_h13_olcum.json
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
sys.path.insert(0, str(WS / "varyantlar" / "logit_ayari"))
from ortak.eslestirilmis_bootstrap import eslestirilmis_bootstrap  # noqa: E402
import logit_ayari as LA  # noqa: E402

K, NV, MEL, BKL, NB, RNG = 7, 5, 4, 2, 2000, 0
SINIF = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
BES = [0, 1, 2, 4, 5]
COLAB = WS / "runs_colab"
YEREL = WS / "runs"
KOL = {  # ad: (kök, varyant)
    "h10": (COLAB, "h10_dinov2b_k4_ft224"), "h02": (COLAB, "h02_r18in_ft224_colab"), "h08": (COLAB, "h08_r18in_ft224_d4_colab"),
    "h01": (COLAB, "h01_dinov2b_lp_r224_colab"), "bas": (COLAB, "baseline_colab"),
    "h13": (YEREL, "h13_h02c_h10_lpo"), "h13k": (YEREL, "h13k_h13_logitayar"), "rt13": (YEREL, "rt_h13_sicaklik"),
    "op": (YEREL, "h13x_h02c_h10_op"), "t2": (YEREL, "h13x_t2_h01c"), "t4": (YEREL, "h13x_t4_h08c_h10_lpo"),
    "m3c": (YEREL, "h13x_m3_baseline_h02c_lpo"), "hom2": (YEREL, "h13x_abl_h10_hom2"),
    "t3_h10": (YEREL, "h13x_t3_h10_hom5"), "t3_h02": (YEREL, "h13x_t3_h02c_hom5"), "m3_bas5": (YEREL, "h13x_m3_baseline_hom5"),
}


def yukle(kok: Path, varyant: str):
    P, seed, ad, M, C = [], [], [], [], []
    y0 = None
    for d in sorted(kok.glob("*/")):
        cf, dn = d / "config.json", d / ".done"
        if not (cf.exists() and dn.exists() and (d / "tahmin_val.npy").exists()):
            continue
        c = json.loads(cf.read_text(encoding="utf-8"))
        j = json.loads(dn.read_text(encoding="utf-8"))
        if c.get("varyant") != varyant or j.get("rc") != 0 or int(c.get("asama", 3)) != 3:
            continue
        y = np.load(d / "etiket_val.npy").reshape(-1).astype(int)
        y0 = y if y0 is None else y0
        assert np.array_equal(y, y0)
        P.append(np.load(d / "tahmin_val.npy").astype(np.float64))
        seed.append(str(c["seed"]))
        ad.append(d.name)
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        M.append({k: v for k, v in m.items() if not k.startswith("test")})  # test anahtarları okunmaz (baseline_c rakip köprüsü)
        C.append(c)
    o = np.argsort(seed)
    return {"P": np.stack(P)[o], "y": y0, "seed": [seed[i] for i in o], "kos": [ad[i] for i in o],
            "M": [M[i] for i in o], "C": [C[i] for i in o]}


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
    y = np.load(WS / "data/medmnist/dermamnist.npz")["val_labels"].ravel().astype(int)
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
    tabakalar = [((tab == k) & S, (tab == k) & ~S) for k in sorted(set(tab)) if ((tab == k) & S).any() and ((tab == k) & ~S).any()]
    with open(WS / "data/splits/dermamnist_c_esleme.csv", encoding="utf-8") as f:
        cv = {int(r["index_resmi"]) for r in csv.DictReader(f) if r["split_C"] == "val" and r["split_resmi"] == "val"}
    CV = np.array([i in cv for i in range(N)])

    kol = {ad: yukle(k, v) for ad, (k, v) in KOL.items()}
    bes = ["0", "1", "2", "3", "4"]
    denetim = {}
    for ad, k in kol.items():
        assert np.array_equal(k["y"], y), ad
        n_bek = 1 if ad.startswith(("t3_", "m3_bas5")) else 5
        assert len(k["seed"]) == n_bek, (ad, k["seed"])
        if n_bek == 5:
            assert k["seed"] == bes, (ad, k["seed"])
        acc_npy = (k["P"].argmax(2) == y[None]).mean(1)
        denetim[ad] = {"kosumlar": k["kos"], "seed": k["seed"],
                       "npy_metrics_acc_maks_fark": float(np.max(np.abs(acc_npy - np.array([m["val_acc"] for m in k["M"]])))),
                       "ebeveyn": [c.get("ebeveyn") for c in k["C"]]}
    # H13 eşleme denetimi: config.ebeveyn = aynı seed H10 run_id; üyeler = aynı seed H10/H02c
    eb_ok = [kol["h13"]["C"][s]["ebeveyn"] == kol["h10"]["kos"][s] for s in range(5)]
    uy_ok = [[u["run_id"] for u in kol["h13"]["C"][s]["uyeler"]] == [kol["h10"]["kos"][s], kol["h02"]["kos"][s]] for s in range(5)]
    denetim["h13_eslesme"] = {"ebeveyn_h10_ayni_seed": eb_ok, "uyeler_h10_h02c_ayni_seed": uy_ok}
    assert all(eb_ok) and all(uy_ok)

    # H04 OOF katmanı: H10 (yeniden hesap; tahlil_tur4 ile aynı) → "h10k"
    KJ = json.loads(LA.KATLAR.read_text(encoding="utf-8"))
    kat = np.array(KJ["kat"], dtype=np.int64)
    logpi = np.log(LA.train_pi())
    taus = LA.izgara("0:1:0.1")
    L10 = np.log(np.maximum(kol["h10"]["P"], 1e-300))
    tk10, _ = LA.oof_tau(L10, logpi, y, kat, taus, 0.010)
    kol["h10k"] = {"P": np.stack([LA.softmax(L10[s] - tk10[kat][:, None] * logpi[None, :]) for s in range(5)]),
                   "kos": kol["h10"]["kos"], "seed": bes}
    L02 = np.log(np.maximum(kol["h02"]["P"], 1e-300))
    tk02, _ = LA.oof_tau(L02, logpi, y, kat, taus, 0.010)
    kol["h02k"] = {"P": np.stack([LA.softmax(L02[s] - tk02[kat][:, None] * logpi[None, :]) for s in range(5)]),
                   "kos": kol["h02"]["kos"], "seed": bes}

    rng = np.random.default_rng(RNG)
    IDX = rng.integers(0, N, size=(NB, N))
    W = np.zeros((NB, N))
    for b in range(NB):
        W[b] = np.bincount(IDX[b], minlength=N)
    oh = np.stack([(y == c).astype(float) for c in range(K)], 1)

    def ga(x):
        return [float(np.quantile(x, .025)), float(np.quantile(x, .975))]

    def alt(d, m):
        return d[m].mean(), (W[:, m] @ d[m]) / W[:, m].sum(1)

    def uc(P):
        d = (P.argmax(2) == y[None]).mean(0)
        o = {"acc": (d.mean(), W @ d / N)}
        rec_b = (W @ (d[:, None] * oh)) / (W @ oh)
        rec = np.array([(d[y == c]).mean() for c in range(K)])
        o["dengeli_acc"] = (rec.mean(), rec_b.mean(1))
        o["dengeli_acc_5sinif"] = (rec[BES].mean(), rec_b[:, BES].mean(1))
        for c in range(K):
            o[f"duyarlilik_{SINIF[c]}"] = (rec[c], rec_b[:, c])
        o["acc_molemax_disi"] = alt(d, ~MM)
        o["acc_S_ham"] = alt(d, S)
        o["acc_N_ham"] = alt(d, ~S)
        o["acc_cval"] = alt(d, CV)
        nvp = (P.argmax(2) == NV).mean(0)
        o["nv_payi"] = (nvp.mean(), W @ nvp / N)
        top = w = 0.0
        topb, wb = np.zeros(NB), np.zeros(NB)
        for mS, mN in tabakalar:
            wk = mN.sum()
            top += wk * (d[mS].mean() - d[mN].mean()); w += wk
            sS, sN = W[:, mS].sum(1), W[:, mN].sum(1)
            ok = (sS > 0) & (sN > 0)
            fk = np.where(ok, (W[:, mS] @ d[mS]) / np.maximum(sS, 1) - (W[:, mN] @ d[mN]) / np.maximum(sN, 1), 0)
            topb += np.where(ok, sN * fk, 0); wb += np.where(ok, sN, 0)
        o["sizinti_tabakali_SN"] = (top / w, topb / wb)
        return o

    def std_tab(f):
        top = {"S": 0.0, "N": 0.0}; w = 0.0
        topb = {"S": np.zeros(NB), "N": np.zeros(NB)}; wb = np.zeros(NB)
        for mS, mN in tabakalar:
            wk = mN.sum()
            top["S"] += wk * f[mS].mean(); top["N"] += wk * f[mN].mean(); w += wk
            sS, sN = W[:, mS].sum(1), W[:, mN].sum(1)
            ok = (sS > 0) & (sN > 0)
            topb["S"] += np.where(ok, sN * (W[:, mS] @ f[mS]) / np.maximum(sS, 1), 0)
            topb["N"] += np.where(ok, sN * (W[:, mN] @ f[mN]) / np.maximum(sN, 1), 0)
            wb += np.where(ok, sN, 0)
        return {k: (top[k] / w, topb[k] / wb) for k in ("S", "N")}

    U = {ad: uc(k["P"]) for ad, k in kol.items() if ad not in ("rt13",)}

    def kiyas(A, B, et, tlik=True):
        PA, PB = kol[A]["P"], kol[B]["P"]
        r = eslestirilmis_bootstrap(PA, PB, y, NB, RNG)
        e = {"kiyas": et, "A": A, "B": B, "kosum_A": kol[A]["kos"], "kosum_B": kol[B]["kos"]}
        for q in U[A]:
            pa, ba = U[A][q]; pb, bb = U[B][q]
            fb = bb - ba
            e[q] = {"A": float(pa), "B": float(pb), "delta": float(pb - pa), "ga": ga(fb),
                    "p_tek_B>A": float((np.sum(fb <= 0) + 1) / (NB + 1))}
        e["acc_referansla_ayni"] = bool(abs(e["acc"]["ga"][0] - r["ga_alt"]) < 1e-12 and abs(e["acc"]["ga"][1] - r["ga_ust"]) < 1e-12)
        e["acc_bootstrap_ref"] = {k: r[k] for k in ("delta", "ga_alt", "ga_ust", "p_tek", "farkli_dogru_goruntu")}
        dA = (PA.argmax(2) == y[None]).mean(0); dB = (PB.argmax(2) == y[None]).mean(0); f = dB - dA
        st = std_tab(f)
        e["tab_std_dS"] = {"nokta": float(100 * st["S"][0]), "ga": [100 * x for x in ga(st["S"][1])]}
        e["tab_std_dN"] = {"nokta": float(100 * st["N"][0]), "ga": [100 * x for x in ga(st["N"][1])]}
        e["tab_std_dS_eksi_dN"] = {"nokta": float(100 * (st["S"][0] - st["N"][0])), "ga": [100 * x for x in ga(st["S"][1] - st["N"][1])]}
        top = f.sum()
        e["ayrisim"] = {"S_katkisi_puan": float(100 * f[S].sum() / N), "N_katkisi_puan": float(100 * f[~S].sum() / N),
                        "S_payi": float(f[S].sum() / top) if top != 0 else None,
                        "net_goruntu_S": float(f[S].sum()), "net_goruntu_N": float(f[~S].sum())}
        # C-val alt küme içi bootstrap (tahlil_tur4_cval ile aynı: alt küme içinde yeniden örnekleme, RNG 0)
        idx = np.where(CV)[0]; n = len(idx)
        Ic = np.random.default_rng(0).integers(0, n, size=(NB, n))
        fc = f[idx]
        e["cval"] = {"n": n, "dACC_puan": float(100 * fc.mean()), "ga": [100 * x for x in ga(fc[Ic].mean(1))]}
        if tlik and len(PA) == 5 and len(PB) == 5:
            aA = (PA.argmax(2) == y[None]).mean(1); aB = (PB.argmax(2) == y[None]).mean(1)
            xA = np.array([dengeli(y, PA[s].argmax(1)) for s in range(5)])
            xB = np.array([dengeli(y, PB[s].argmax(1)) for s in range(5)])
            cA = (PA[:, idx].argmax(2) == y[idx][None]).mean(1); cB = (PB[:, idx].argmax(2) == y[idx][None]).mean(1)
            for q, xa, xb in [("t_acc", aA, aB), ("t_dengeli", xA, xB), ("t_cval", cA, cB)]:
                fd = xb - xa
                e[q] = {"A_seed": xa.tolist(), "B_seed": xb.tolist(), "fark_seed": fd.tolist(), "ort": float(fd.mean()),
                        "sd": float(fd.std(ddof=1)),
                        "p_tek_B>A": float(stats.ttest_rel(xb, xa, alternative="greater").pvalue),
                        "p_tek_B<A": float(stats.ttest_rel(xb, xa, alternative="less").pvalue)}
        e["karar_degisen_goruntu_ort"] = float((PA.argmax(2) != PB.argmax(2)).sum(1).mean()) if len(PA) == len(PB) else None
        return e

    KIY = [("h10", "h13", "H13 − H10  [BİRİNCİL; öldürme (b)/(b') bağlayıcı]"),
           ("h02", "h13", "H13 − H02c  [ikincil (1)]"),
           ("h08", "h13", "H13 − H08c  [ikincil (5) C-val için; bilgi]"),
           ("h10", "op", "H13x op (olasılık ort.) − H10  [ikincil (7)]"),
           ("h13", "op", "op − lpo  [ikincil (7)]"),
           ("hom2", "h13", "H13 − E_hom2 (H10_s ⊕ H10_s+1)  [ablasyon (i); çiftler örtüşür]"),
           ("h10", "hom2", "E_hom2 − H10  [ablasyon (i) bilgi]"),
           ("h10", "t2", "T2 (H10 ⊕ H02c ⊕ H01c) − H10  [betimsel]"),
           ("h13", "t2", "T2 − H13  [betimsel]"),
           ("h10", "t4", "T4 (H10 ⊕ H08c) − H10  [koşullu, betimsel]"),
           ("h13", "t4", "T4 − H13  [betimsel]"),
           ("h02", "m3c", "M3: baseline_c ⊕ H02c − H02c  [betimsel]"),
           ("bas", "m3c", "M3: baseline_c ⊕ H02c − baseline_c  [betimsel]"),
           ("h10", "t3_h10", "T3: H10 hom5 (n = 1) − H10 tek-model ort.  [betimsel]"),
           ("h02", "t3_h02", "T3: H02c hom5 (n = 1) − H02c tek-model ort.  [betimsel]"),
           ("bas", "m3_bas5", "M3: baseline_c hom5 (n = 1) − baseline_c tek-model ort.  [betimsel]"),
           ("h10k", "h13k", "H13 + H04 − H10 + H04 (OOF)  [ikinci satır; karar vermez]"),
           ("h13", "h13k", "H13 + H04 − H13  [katmanın kendi bedeli/kazancı]")]
    sonuc = {e[2]: kiyas(*e) for e in KIY}

    # ikincil metrik ortalamaları (tek-model/topluluk; tahlil_tur4 ik ile aynı tanımlar)
    ik = {}
    for ad in ("h10", "h02", "h08", "h01", "bas", "h13", "op", "t2", "t4", "m3c", "hom2", "t3_h10", "t3_h02", "m3_bas5", "h13k", "h10k"):
        P = kol[ad]["P"]
        m = {q: [] for q in ["acc", "dengeli_acc", "dengeli_acc_5sinif", "makro_f1", "ece15", "nll", "auc", "auc_mel", "auc_bkl",
                             "nv_payi", "acc_molemax_disi", "mel_nv", "toplam_hata", "acc_S", "acc_N", "acc_cval", "sizinti_tabakali_SN"]}
        for s in range(len(P)):
            t = P[s].argmax(1)
            m["acc"].append((t == y).mean()); m["dengeli_acc"].append(dengeli(y, t)); m["dengeli_acc_5sinif"].append(dengeli(y, t, BES))
            m["makro_f1"].append(f1_score(y, t, average="macro", labels=list(range(K)), zero_division=0))
            m["ece15"].append(ece15(y, P[s])); m["nll"].append(-np.mean(np.log(np.clip(P[s][np.arange(N), y], 1e-12, 1))))
            ac = [roc_auc_score((y == c).astype(int), P[s][:, c]) for c in range(K)]
            m["auc"].append(np.mean(ac)); m["auc_mel"].append(ac[MEL]); m["auc_bkl"].append(ac[BKL])
            m["nv_payi"].append((t == NV).mean()); m["acc_molemax_disi"].append((t[~MM] == y[~MM]).mean())
            m["mel_nv"].append(int(((y == MEL) & (t == NV)).sum())); m["toplam_hata"].append(int((t != y).sum()))
            m["acc_S"].append((t[S] == y[S]).mean()); m["acc_N"].append((t[~S] == y[~S]).mean()); m["acc_cval"].append((t[CV] == y[CV]).mean())
            dd = (t == y)
            top = w = 0.0
            for mS, mN in tabakalar:
                wk = mN.sum(); top += wk * (dd[mS].mean() - dd[mN].mean()); w += wk
            m["sizinti_tabakali_SN"].append(top / w)
        ik[ad] = {"n": len(P), "ort": {q: float(np.mean(v)) for q, v in m.items()},
                  "std": {q: (float(np.std(v, ddof=1)) if len(v) > 1 else None) for q, v in m.items()}}

    # çeşitlilik (ikincil 6): çapraz aile seed-eşleşik vs aile içi (s ≠ s')
    def an_iy(P1, P2):
        t1, t2 = P1.argmax(1), P2.argmax(1)
        return float((t1 != t2).mean()), float(((t1 != y) & (t2 != y)).mean())
    T10, T02 = kol["h10"]["P"], kol["h02"]["P"]
    capraz = [an_iy(T10[s], T02[s]) for s in range(5)]
    ic10 = [an_iy(T10[i], T10[j]) for i in range(5) for j in range(i + 1, 5)]
    ic02 = [an_iy(T02[i], T02[j]) for i in range(5) for j in range(i + 1, 5)]
    capraz_tum = [an_iy(T10[i], T02[j]) for i in range(5) for j in range(5)]
    cesit = {"capraz_seed_eslesik": {"anlasmazlik": float(np.mean([a for a, _ in capraz])), "ikisi_de_yanlis": float(np.mean([b for _, b in capraz])),
                                     "seed": [list(x) for x in capraz]},
             "capraz_tum_25_cift": {"anlasmazlik": float(np.mean([a for a, _ in capraz_tum])), "ikisi_de_yanlis": float(np.mean([b for _, b in capraz_tum]))},
             "aile_ici_h10_10_cift": {"anlasmazlik": float(np.mean([a for a, _ in ic10])), "ikisi_de_yanlis": float(np.mean([b for _, b in ic10]))},
             "aile_ici_h02c_10_cift": {"anlasmazlik": float(np.mean([a for a, _ in ic02])), "ikisi_de_yanlis": float(np.mean([b for _, b in ic02]))},
             "hata_orani_h10": float(1 - (T10.argmax(2) == y[None]).mean()), "hata_orani_h02c": float(1 - (T02.argmax(2) == y[None]).mean())}
    # bağımsız hatalar altında beklenen "ikisi de yanlış" (bilgi)
    cesit["bagimsiz_beklenen_ikisi_de_yanlis"] = cesit["hata_orani_h10"] * cesit["hata_orani_h02c"]

    # katman özetleri
    Mk = kol["h13k"]["M"]; Mr = kol["rt13"]["M"]
    katman = {"h13k": {"oof_acc": [m["val_acc"] for m in Mk], "oof_dengeli": [m["val_dengeli_acc"] for m in Mk],
                       "oof_dengeli_ort": float(np.mean([m["val_dengeli_acc"] for m in Mk])),
                       "oof_dengeli_std": float(np.std([m["val_dengeli_acc"] for m in Mk], ddof=1)),
                       "oof_acc_ort": float(np.mean([m["val_acc"] for m in Mk])),
                       "oof_dengeli5": float(np.mean([dengeli(y, kol["h13k"]["P"][s].argmax(1), BES) for s in range(5)])),
                       "tau_kat": [Mk[0][f"tau_kat_{i}"] for i in range(1, 6)], "tau_tamval": Mk[0]["tau_tamval"],
                       "dengeli_tamval_ort": float(np.mean([m["val_dengeli_acc_tamval"] for m in Mk])),
                       "katlar_sha256": Mk[0]["katlar_sha256"], "tau0_ozdeslik_fark": [m["tau0_ozdeslik_fark"] for m in Mk]},
              "h10k_yeniden": {"oof_dengeli_ort": ik["h10k"]["ort"]["dengeli_acc"], "oof_acc_ort": ik["h10k"]["ort"]["acc"], "tau_kat": tk10.tolist()},
              "rt13": {"oof_ece": [m["val_ece"] for m in Mr], "oof_ece_ort": float(np.mean([m["val_ece"] for m in Mr])),
                       "oof_ece_std": float(np.std([m["val_ece"] for m in Mr], ddof=1)),
                       "T_tamval": [m.get("T_tamval") for m in Mr], "ham_ece_ort": ik["h13"]["ort"]["ece15"]}}

    # maliyet (M7)
    M13 = kol["h13"]["M"]
    maliyet = {"param_toplam": M13[0]["param_sayisi"], "cpu_birlesim_sn_ort": float(np.mean([m["egitim_sure_sn"] for m in M13])),
               "h10_epoch_val_sn_ort": float(np.mean([m.get("epoch_val_sn_ort", float("nan")) for m in kol["h10"]["M"]])),
               "h10_param": kol["h10"]["M"][0].get("param_sayisi"), "h02c_param": kol["h02"]["M"][0].get("param_sayisi")}

    # görülmemiş eşikleri (kart; karar değil — sayı)
    b = sonuc[KIY[0][2]]
    gorulmemis = {"delta_vs_h10_puan": 100 * b["acc"]["delta"], "esik_delta": 2.0, "ort_acc": ik["h13"]["ort"]["acc"], "esik_acc": 0.93,
                  "ort_auc": ik["h13"]["ort"]["auc"], "esik_auc": 0.99}
    out = {"kiyaslar": sonuc, "ikincil": ik, "cesitlilik": cesit, "katman": katman, "maliyet": maliyet, "gorulmemis": gorulmemis,
           "denetim": denetim, "n_S": int(S.sum()), "n_N": int((~S).sum()), "n_cval": int(CV.sum()), "n_tabaka": len(tabakalar),
           "yontem": "eşleştirilmiş görüntü bootstrap 2.000 RNG 0 (tahlil_tur4 ile aynı W); kesin tek yönlü t scipy ttest_rel; "
                     "S/N tabaka-std ve tabakalı S − N tahlil_tur4(_ezber) ile aynı; C-val alt küme içi bootstrap RNG 0 (tahlil_tur4_cval)"}
    (WS / "sonuclar" / "tur5_h13_olcum.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    def p(x):
        return f"{100 * x:+.2f}"
    for et, e in sonuc.items():
        s = (f"{et}: ACC {e['acc']['A']:.4f}→{e['acc']['B']:.4f} Δ {p(e['acc']['delta'])} [{p(e['acc']['ga'][0])}; {p(e['acc']['ga'][1])}] "
             f"p_bs {e['acc']['p_tek_B>A']:.3g}")
        if "t_acc" in e:
            s += f" | t: sd {100 * e['t_acc']['sd']:.2f} p_t {e['t_acc']['p_tek_B>A']:.3g} seed {[round(100 * x, 2) for x in e['t_acc']['fark_seed']]}"
        s += (f" | den Δ {p(e['dengeli_acc']['delta'])} [{p(e['dengeli_acc']['ga'][0])}; {p(e['dengeli_acc']['ga'][1])}]"
              f" 5s {p(e['dengeli_acc_5sinif']['delta'])} [{p(e['dengeli_acc_5sinif']['ga'][0])}; {p(e['dengeli_acc_5sinif']['ga'][1])}]"
              f" | S {p(e['acc_S_ham']['delta'])} [{p(e['acc_S_ham']['ga'][0])}; {p(e['acc_S_ham']['ga'][1])}]"
              f" N {p(e['acc_N_ham']['delta'])} [{p(e['acc_N_ham']['ga'][0])}; {p(e['acc_N_ham']['ga'][1])}]"
              f" tabStd ΔS {e['tab_std_dS']['nokta']:+.2f} ΔN {e['tab_std_dN']['nokta']:+.2f} ΔS−ΔN {e['tab_std_dS_eksi_dN']['nokta']:+.2f} "
              f"[{e['tab_std_dS_eksi_dN']['ga'][0]:+.2f}; {e['tab_std_dS_eksi_dN']['ga'][1]:+.2f}]"
              f" | Δ(tabS−N) {p(e['sizinti_tabakali_SN']['delta'])} [{p(e['sizinti_tabakali_SN']['ga'][0])}; {p(e['sizinti_tabakali_SN']['ga'][1])}]"
              f" | C-val {e['cval']['dACC_puan']:+.2f} [{e['cval']['ga'][0]:+.2f}; {e['cval']['ga'][1]:+.2f}]"
              + (f" p_t {e['t_cval']['p_tek_B>A']:.3g}" if "t_cval" in e else "")
              + f" | ref {e['acc_referansla_ayni']}")
        print(s)
    print(json.dumps({k: {q: round(v, 4) for q, v in d["ort"].items()} for k, d in ik.items()}, ensure_ascii=False))
    print(json.dumps(cesit, ensure_ascii=False))
    print(json.dumps(katman, ensure_ascii=False, default=float))
    print(json.dumps(maliyet, ensure_ascii=False), json.dumps(gorulmemis, ensure_ascii=False))


if __name__ == "__main__":
    main()
