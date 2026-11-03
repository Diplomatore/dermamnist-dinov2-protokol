# -*- coding: utf-8 -*-
"""
inceleme_h08.py — H08 İNCELE soruşturması (deney-tahlilci, Tur-3 K4c). YALNIZ VALIDATION, CPU. Test AÇILMAZ (M4).

H08 kartı "görülmemiş" eşiği: dengeli ACC Δ > +6,5 → İNCELE. Ölçülen Δ +8,06 [+4,97; +11,79]. Bu betik kartın
soruşturma sırasındaki ölçülebilir testleri üretir:
  (a) train/val karışması: piksel düzeyinde train↔val yakın-kopya taraması, KİMLİK ve 7 D4 dönüşümü altında
      (D4 augmentasyonunun "döndürülmüş kopya" yoluyla val'i train'e sokup sokmadığı). 28 px resmî npz, RMSE (0–255).
      Aynı-lezyon (S) çiftlerinde en iyi hizalayan dönüşüm. Kazancın S/N ve yakın-kopya alt kümelerine dağılımı.
  (b) seçim iyimserliği: son epoch ve val-AUC seçimli kollarda ACC/dengeli Δ (bootstrap); H08 epoch başına dengeli.
  (c) kazancın kaynağı: sınıf ayrışımı (Δdengeli = (1/7) Σ_c Δduyarlılık_c), görüntü birimiyle Δ, df/vasc hariç
      5-sınıf dengeli, en etkili k görüntüyü çıkarınca Δ, çoğunluk-oyu görüntü dönüşleri, seed tutarlılığı.
  (d) eşik: Δdengeli bootstrap dağılımı, P*(Δ ≤ 6,5), seed bileşeni.
  (e) mekanizma: H02'nin seçimsiz τ (logit ayarı) eğrisi — H08 noktası bu eğrinin erişebileceği bölgede mi
      (önsel kayması) yoksa dışında mı (sıralama/temsil); sınıf başına bire-karşı-hepsi AUC Δ.
Bootstrap: 2.000, RNG 0, W = bincount(rng.integers(0, N, (2000, N))) — tur3_olcum.py ile aynı W.
Kullanım (WS kökü): PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I scripts/inceleme_h08.py
Çıktı: sonuclar/inceleme_h08.json (+ ekrana özet).
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "varyantlar"))
from ortak.eslestirilmis_bootstrap import kosum_bul  # noqa: E402

SINIF = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
K, NV, NB, RNG = 7, 5, 2000, 0
ESIK_DENGELI = 0.065


def yukle(varyant, dosya="tahmin_val.npy"):
    P, seed, ad, y0, gec, met = [], [], [], None, [], []
    for d in kosum_bul(varyant, WS, dosya):
        c = json.loads((d / "config.json").read_text(encoding="utf-8"))
        if int(c.get("asama", 3)) != 3:
            continue
        y = np.load(d / "etiket_val.npy").reshape(-1).astype(int)
        y0 = y if y0 is None else y0
        assert np.array_equal(y, y0)
        P.append(np.load(d / dosya).astype(np.float64))
        seed.append(int(c["seed"]))
        ad.append(d.name)
        gec.append(json.loads((d / "gecmis.json").read_text(encoding="utf-8")))
        met.append(json.loads((d / "metrics.json").read_text(encoding="utf-8")))
    o = np.argsort(seed)
    return (np.stack(P)[o], y0, [seed[i] for i in o], [ad[i] for i in o], [gec[i] for i in o], [met[i] for i in o])


def dengeli(y, t):
    return float(np.mean([(t[y == c] == c).mean() for c in range(K)]))


def d4_np(x, k):
    """x: (N, H, W, C) numpy; dihedral.py ile aynı tanım (uzamsal eksenler 1, 2)."""
    if k == 0:
        return x
    if k in (1, 2, 3):
        return np.rot90(x, k, axes=(1, 2))
    if k == 4:
        return x[:, :, ::-1]
    if k == 5:
        return x[:, ::-1]
    if k == 6:
        return x.transpose(0, 2, 1, 3)
    if k == 7:
        return np.rot90(x, 1, axes=(1, 2))[:, :, ::-1]
    raise ValueError(k)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out = {}
    npz = np.load(WS / "data/medmnist/dermamnist.npz")
    y = npz["val_labels"].ravel().astype(int)
    N = len(y)
    n_c = np.bincount(y, minlength=K)
    out["destek"] = dict(zip(SINIF, n_c.tolist()))

    # ── meta: S (lezyonu train'de), kaynak ──
    tr_lez, val_lez, vimg, tr_by_lez = set(), {}, {}, {}
    with open(WS / "data/splits/dermamnist_lezyon_esleme.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "train":
                tr_lez.add(r["lesion_id"])
                tr_by_lez.setdefault(r["lesion_id"], []).append(int(r["index"]))
            elif r["split"] == "val":
                val_lez[int(r["index"])] = r["lesion_id"]
                vimg[int(r["index"])] = r["image_id"]
    S = np.array([val_lez[i] in tr_lez for i in range(N)])
    meta = {}
    with open(WS / "data/meta/HAM10000_metadata.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            meta[r["image_id"]] = r
    kay = np.array([meta[vimg[i]]["dataset"] for i in range(N)])
    MM = kay == "vidir_molemax"

    # ── kollar ──
    A = yukle("h02_r18in_ft224")
    B = yukle("h08_r18in_ft224_d4")
    for X in (A, B):
        assert np.array_equal(X[1], y) and X[2] == [0, 1, 2, 3, 4]
    PA, PB = A[0], B[0]
    out["kosum_h02"], out["kosum_h08"] = A[3], B[3]

    rng = np.random.default_rng(RNG)
    IDX = rng.integers(0, N, size=(NB, N))
    W = np.zeros((NB, N))
    for b in range(NB):
        W[b] = np.bincount(IDX[b], minlength=N)
    oh = np.stack([(y == c).astype(float) for c in range(K)], 1)

    def dbar(P):
        return (P.argmax(2) == y[None]).mean(0)

    def dengeli_bs(d, siniflar=range(K)):
        rec_b = (W @ (d[:, None] * oh)) / (W @ oh)
        rec = np.array([d[y == c].mean() for c in range(K)])
        s = list(siniflar)
        return rec[s].mean(), rec_b[:, s].mean(1)

    def kiyas(PA_, PB_, ad):
        dA, dB = dbar(PA_), dbar(PB_)
        r = {}
        a0, ab = dA.mean(), W @ dA / N
        b0, bb = dB.mean(), W @ dB / N
        r["acc"] = {"A": a0, "B": b0, "delta": b0 - a0, "ga": list(np.quantile(bb - ab, [.025, .975]))}
        for et, sn in (("dengeli", range(K)), ("dengeli_5sinif_dfvasc_haric", [0, 1, 2, 4, 5])):
            a0, ab = dengeli_bs(dA, sn)
            b0, bb = dengeli_bs(dB, sn)
            fb = bb - ab
            r[et] = {"A": a0, "B": b0, "delta": b0 - a0, "ga": list(np.quantile(fb, [.025, .975])),
                     "bs_sd": float(fb.std(ddof=1)), "P*(delta<=0)": float((fb <= 0).mean()),
                     "P*(delta<=esik_6.5)": float((fb <= ESIK_DENGELI).mean())}
        r = json.loads(json.dumps(r, default=float))
        out[ad] = r
        return r

    # (d) birincil kol + eşik dağılımı
    r0 = kiyas(PA, PB, "kiyas_secilen_epoch")
    # (b) seçim kuralı kolları
    PA_son, PB_son = yukle("h02_r18in_ft224", "tahmin_val_son.npy")[0], yukle("h08_r18in_ft224_d4", "tahmin_val_son.npy")[0]
    PA_auc, PB_auc = (yukle("h02_r18in_ft224", "tahmin_val_aucsecim.npy")[0],
                      yukle("h08_r18in_ft224_d4", "tahmin_val_aucsecim.npy")[0])
    kiyas(PA_son, PB_son, "kiyas_son_epoch")
    kiyas(PA_auc, PB_auc, "kiyas_aucsecim")
    # H08 epoch başına dengeli (gecmis.json; H02'de bu anahtar yok)
    ep_den = np.array([[g["val_dengeli_acc"] for g in gs] for gs in B[4]])  # (5, 30)
    ep_acc_B = np.array([[g["val_acc"] for g in gs] for gs in B[4]])
    ep_acc_A = np.array([[g["val_acc"] for g in gs] for gs in A[4]])
    out["h08_epoch_dengeli"] = {
        "secilen_epoch": [m["en_iyi_epoch"] for m in B[5]],
        "secilen_epoch_dengeli": [float(ep_den[s, B[5][s]["en_iyi_epoch"] - 1]) for s in range(5)],
        "metrics_dengeli": [m["val_dengeli_acc"] for m in B[5]],
        "ep21_30_ort_seed": ep_den[:, 20:].mean(1).tolist(), "ep21_30_ort": float(ep_den[:, 20:].mean()),
        "ep11_30_ort": float(ep_den[:, 10:].mean()), "maks_epoch_ort": float(ep_den.max(1).mean()),
        "son_epoch_ort": float(ep_den[:, -1].mean()),
        "secilen_epoch_dengelinin_seed_icindeki_yuzdeligi": [float((ep_den[s] <= ep_den[s, B[5][s]["en_iyi_epoch"] - 1]).mean())
                                                               for s in range(5)],
        "acc_ep21_30_ort_h08": float(ep_acc_B[:, 20:].mean()), "acc_ep21_30_ort_h02": float(ep_acc_A[:, 20:].mean()),
    }

    # (c) sınıf ayrışımı (görüntü birimi)
    tA, tB = PA.argmax(2), PB.argmax(2)
    cA, cB = (tA == y[None]), (tB == y[None])
    sinif = {}
    for c in range(K):
        m = y == c
        gA, gB = cA[:, m].sum(1), cB[:, m].sum(1)
        sinif[SINIF[c]] = {"n": int(m.sum()), "dogru_h02_seed": gA.tolist(), "dogru_h08_seed": gB.tolist(),
                           "delta_goruntu_ort": float((gB - gA).mean()),
                           "delta_duyarlilik_puan": float(100 * (gB - gA).mean() / m.sum()),
                           "dengeli_katki_puan": float(100 * (gB - gA).mean() / m.sum() / K)}
    out["sinif_ayrisimi"] = sinif
    out["dengeli_katki_df_vasc_puan"] = sinif["df"]["dengeli_katki_puan"] + sinif["vasc"]["dengeli_katki_puan"]
    out["dengeli_seed"] = {"h02": [dengeli(y, tA[s]) for s in range(5)], "h08": [dengeli(y, tB[s]) for s in range(5)]}
    out["dengeli_seed"]["delta"] = [b - a for a, b in zip(out["dengeli_seed"]["h02"], out["dengeli_seed"]["h08"])]
    # seed başına sınıf katkısı (df+vasc)
    out["seed_dfvasc_katki_puan"] = [float(100 * sum((cB[s, y == c].sum() - cA[s, y == c].sum()) / n_c[c] for c in (3, 6)) / K)
                                     for s in range(5)]

    # görüntü başına dengeli katkısı → en etkili k görüntüyü çıkar
    dA, dB = dbar(PA), dbar(PB)
    katki = (dB - dA) / n_c[y] / K
    o = np.argsort(-katki)
    loo = {}
    for k in (1, 2, 3, 4, 5, 8, 10, 15, 20):
        m = np.ones(N, bool)
        m[o[:k]] = False
        rec = np.array([(dB[m & (y == c)] - dA[m & (y == c)]).mean() for c in range(K)])
        loo[str(k)] = {"delta_dengeli_puan": float(100 * rec.mean()),
                       "cikarilan_siniflar": [SINIF[y[i]] for i in o[:k]]}
    out["en_etkili_k_cikarilinca"] = loo
    pos = np.sort(katki[katki > 0])[::-1]
    cum = np.cumsum(pos) / katki.sum()
    out["toplam_delta_dengeli_puan"] = float(100 * katki.sum())
    out["delta_yarisini_aciklayan_goruntu_sayisi"] = int(np.searchsorted(cum, 0.5) + 1)
    out["pozitif_katkili_goruntu"] = int((katki > 0).sum()); out["negatif_katkili_goruntu"] = int((katki < 0).sum())

    # çoğunluk oyu dönüşleri (≥ 3/5 seed)
    mA, mB = cA.sum(0) >= 3, cB.sum(0) >= 3
    out["cogunluk_donus"] = {SINIF[c]: {"h02_yanlis_h08_dogru": int(((~mA) & mB & (y == c)).sum()),
                                        "h02_dogru_h08_yanlis": int((mA & (~mB) & (y == c)).sum())} for c in range(K)}

    # S / N ve molemax ayrımı
    out["S_N"] = {"n_S": int(S.sum()), "n_N": int((~S).sum()),
                  "dACC_S_puan": float(100 * (dB[S] - dA[S]).mean()), "dACC_N_puan": float(100 * (dB[~S] - dA[~S]).mean()),
                  "dACC_S_molemaxdisi_puan": float(100 * (dB[S & ~MM] - dA[S & ~MM]).mean()),
                  "dACC_N_molemaxdisi_puan": float(100 * (dB[~S & ~MM] - dA[~S & ~MM]).mean()),
                  "n_S_molemaxdisi": int((S & ~MM).sum()), "n_N_molemaxdisi": int((~S & ~MM).sum())}
    # nadir sınıflarda S payı
    out["S_payi_sinif"] = {SINIF[c]: float(S[y == c].mean()) for c in range(K)}
    out["S_N_nadir"] = {SINIF[c]: {"dRec_S_puan": float(100 * (dB[S & (y == c)] - dA[S & (y == c)]).mean()) if (S & (y == c)).any() else None,
                                   "dRec_N_puan": float(100 * (dB[~S & (y == c)] - dA[~S & (y == c)]).mean()) if (~S & (y == c)).any() else None,
                                   "n_S": int((S & (y == c)).sum()), "n_N": int((~S & (y == c)).sum())} for c in range(K)}

    # (e) sıralama vs önsel: AUC (bire-karşı-hepsi) ve H02 seçimsiz τ eğrisi
    auc = {}
    for c in range(K):
        a_ = [roc_auc_score(y == c, PA[s][:, c]) for s in range(5)]
        b_ = [roc_auc_score(y == c, PB[s][:, c]) for s in range(5)]
        auc[SINIF[c]] = {"h02": float(np.mean(a_)), "h08": float(np.mean(b_)), "delta_puan": float(100 * (np.mean(b_) - np.mean(a_)))}
    out["auc_ovr"] = auc
    ytr = npz["train_labels"].ravel().astype(int)
    pi = np.bincount(ytr, minlength=K) / len(ytr)
    egri = []
    for tau in np.round(np.arange(0, 1.01, 0.05), 2):
        acc_s, den_s, nv_s = [], [], []
        for s in range(5):
            t = (np.log(PA[s] + 1e-12) - tau * np.log(pi)[None]).argmax(1)
            acc_s.append((t == y).mean()); den_s.append(dengeli(y, t)); nv_s.append((t == NV).mean())
        egri.append({"tau": float(tau), "acc": float(np.mean(acc_s)), "dengeli": float(np.mean(den_s)), "nv_payi": float(np.mean(nv_s))})
    out["h02_tau_egrisi_tamval_seçimsiz"] = egri
    h08_acc, h08_den = float(dB.mean()), float(np.mean(out["dengeli_seed"]["h08"]))
    out["h08_nokta"] = {"acc": h08_acc, "dengeli": h08_den}
    out["h02_egrisinde_acc>=h08_olan_tau"] = [e["tau"] for e in egri if e["acc"] >= h08_acc]
    out["h02_egrisi_maks_dengeli"] = max(e["dengeli"] for e in egri)
    out["h02_egrisi_maks_acc"] = max(e["acc"] for e in egri)

    # (a) piksel düzeyi train↔val yakın-kopya taraması (28 px, RMSE 0–255), kimlik + 7 D4 dönüşümü
    Xtr = npz["train_images"].astype(np.float64)
    Xva = npz["val_images"].astype(np.float64).reshape(N, -1)
    nva = (Xva ** 2).sum(1)
    D = Xva.shape[1]
    rmse_k = np.zeros((8, N)); arg_k = np.zeros((8, N), dtype=int)
    for k in range(8):
        T = np.ascontiguousarray(d4_np(Xtr, k)).reshape(len(Xtr), -1)
        d2 = nva[:, None] + (T ** 2).sum(1)[None] - 2 * Xva @ T.T
        d2 = np.maximum(d2, 0) / D
        arg_k[k] = d2.argmin(1); rmse_k[k] = np.sqrt(d2.min(1))
    r_id = rmse_k[0]; r_rot = rmse_k[1:].min(0); k_rot = rmse_k[1:].argmin(0) + 1
    esikler = (1, 3, 5, 8, 12)
    yk = {}
    for e in esikler:
        m_id, m_rot = r_id <= e, r_rot <= e
        yk[f"rmse<={e}"] = {"kimlik": int(m_id.sum()), "d4_kimlik_disi": int(m_rot.sum()),
                            "yalniz_d4_ile": int((m_rot & ~m_id).sum()),
                            "yalniz_d4_ile_S": int((m_rot & ~m_id & S).sum())}
    out["yakin_kopya_28px"] = yk
    out["rmse_kimlik_dagilim"] = {q: float(np.quantile(r_id, q)) for q in (0.0, 0.01, 0.05, 0.5)}
    out["rmse_d4disi_dagilim"] = {q: float(np.quantile(r_rot, q)) for q in (0.0, 0.01, 0.05, 0.5)}
    # yalnız D4 dönüşümüyle yakın olan val görüntüleri: kazanç orada mı?
    for e in (5, 8):
        m = (r_rot <= e) & (r_rot < r_id - 1.0)
        out[f"d4_yakin_kopya_rmse<={e}_kazanc"] = {"n": int(m.sum()),
                                                  "dACC_puan": float(100 * (dB[m] - dA[m]).mean()) if m.any() else None,
                                                  "siniflar": np.bincount(y[m], minlength=K).tolist()}
    # aynı-lezyon çiftleri: en iyi hizalayan dönüşüm
    cift = []
    Xva_img = npz["val_images"].astype(np.float64)
    for i in np.where(S)[0]:
        for j in tr_by_lez.get(val_lez[i], []):
            r_ = [float(np.sqrt(((d4_np(Xtr[j:j + 1], k)[0] - Xva_img[i]) ** 2).mean())) for k in range(8)]
            cift.append((int(i), int(j), r_))
    R = np.array([c[2] for c in cift])
    kbest = R.argmin(1)
    out["ayni_lezyon_ciftleri"] = {
        "n_cift": len(cift), "n_val_goruntu": int(S.sum()),
        "en_iyi_donusum_sayim": dict(zip(["kimlik", "rot90", "rot180", "rot270", "yatay", "dikey", "transpoz", "ters_transpoz"],
                                         np.bincount(kbest, minlength=8).tolist())),
        "rmse_kimlik_medyan": float(np.median(R[:, 0])), "rmse_eniyi_medyan": float(np.median(R.min(1))),
        "kimlik_disi_eniyi_ve_kazanc>=3": int(((kbest != 0) & (R[:, 0] - R.min(1) >= 3)).sum()),
    }
    # bu "döndürülmüş aynı-lezyon" val görüntülerinde kazanç
    vi = sorted({cift[t][0] for t in range(len(cift)) if kbest[t] != 0 and R[t, 0] - R[t].min() >= 3})
    vi = np.array(vi, dtype=int)
    S_kalan = S.copy(); S_kalan[vi] = False
    out["ayni_lezyon_donuk_hizali_kazanc"] = {
        "n_val": int(len(vi)),
        "dACC_puan": float(100 * (dB[vi] - dA[vi]).mean()) if len(vi) else None,
        "dACC_S_diger_puan": float(100 * (dB[S_kalan] - dA[S_kalan]).mean()),
        "siniflar": np.bincount(y[vi], minlength=K).tolist() if len(vi) else None}

    p = WS / "sonuclar" / "inceleme_h08.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
