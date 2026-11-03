# -*- coding: utf-8 -*-
"""
hata_kirilim.py — K3.5 hata analizi kırılımı (YALNIZ VALIDATION). K5'te kazanan için aynı betik koşar
("önce/sonra" figürü): --varyant <ad> veya --kosumlar <dizin...>.

Kural (M4 / tahlilci): test tahminleri (tahmin_test*.npy, etiket_test.npy) bu betikte AÇILMAZ;
gecmis.json'dan yalnız train_* / val_* anahtarları okunur (test_* anahtarları atılır).

Girdi (koşum dizini başına): config.json (varyant), .done (rc), tahmin_val.npy (seçilen epoch softmax),
  tahmin_val_son.npy (son epoch; varsa), etiket_val.npy, metrics.json (en_iyi_epoch), gecmis.json (varsa).
Ek: data/splits/dermamnist_lezyon_esleme.csv (val görüntüsünün lezyonu train'de var mı → S/N),
  data/medmnist/dermamnist.npz val_labels (etiket hizası denetimi).

Çıktı: sonuclar/hata-kirilim.csv (uzun biçim: bolum, olcu, alt_kume, sinif, deger_ort, deger_std, n_seed, not)
       sonuclar/hata_kirilim_ozet.json (rapor için ham sayılar)

Tanımlar (varyantlar/ortak/degerlendir.py ile aynı):
  ACC = argmax doğruluğu; AUC = makro OvR (sınıf başına roc_auc_score ortalaması = medmnist getAUC);
  dengeli ACC = sınıf duyarlılıklarının ortalaması; ECE = 15 eşit kutu, en yüksek sınıf güveni (Guo vd. 2017).
Kullanım (WS kökünden):
  PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I scripts/hata_kirilim.py --varyant baseline
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

SINIF = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
K = 7
NADIR = [3, 6, 0]  # df, vasc, akiec (val destek 12, 14, 33)
NV = 5


def ece(y, p, n_kutu=15):
    guven = p.max(1)
    dogru = (p.argmax(1) == y).astype(float)
    s = np.linspace(0, 1, n_kutu + 1)
    t = 0.0
    kutular = []
    for i in range(n_kutu):
        m = (guven > s[i]) & (guven <= s[i + 1]) if i > 0 else (guven >= s[i]) & (guven <= s[i + 1])
        if m.any():
            t += m.mean() * abs(dogru[m].mean() - guven[m].mean())
            kutular.append((i, int(m.sum()), float(dogru[m].mean()), float(guven[m].mean())))
        else:
            kutular.append((i, 0, float("nan"), float("nan")))
    return float(t), kutular


def auc_makro(y, p):
    return float(np.mean([roc_auc_score((y == c).astype(int), p[:, c]) for c in range(K)]))


def dengeli(y, t):
    return float(np.mean([(t[y == c] == c).mean() for c in range(K) if (y == c).any()]))


def ozet_olcu(y, p):
    t = p.argmax(1)
    e, kut = ece(y, p)
    n = len(y)
    return {
        "acc": float((t == y).mean()),
        "auc": auc_makro(y, p),
        "dengeli_acc": dengeli(y, t),
        "makro_f1": float(f1_score(y, t, average="macro", labels=list(range(K)), zero_division=0)),
        "ece15": e,
        "nll": float(-np.mean(np.log(np.clip(p[np.arange(n), y], 1e-12, 1)))),
        "brier": float(np.mean(np.sum((p - np.eye(K)[y]) ** 2, 1))),
        "ort_guven": float(p.max(1).mean()),
        "tahmin_nv_payi": float((t == NV).mean()),
        "_kutular": kut,
    }


def kosumlari_bul(ws: Path, varyant: str):
    out = []
    for d in sorted((ws / "runs").iterdir()):
        if not d.is_dir() or not (d / "config.json").exists() or not (d / ".done").exists():
            continue
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        done = json.loads((d / ".done").read_text(encoding="utf-8"))
        if cfg.get("varyant") == varyant and done.get("rc") == 0 and (d / "tahmin_val.npy").exists():
            out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--varyant", default="baseline")
    ap.add_argument("--kosumlar", nargs="*", default=None, help="açık koşum dizinleri (varyant aramasını ezer)")
    ap.add_argument("--cikti-csv", default=None)
    ap.add_argument("--cikti-json", default=None)
    ap.add_argument("--etiket", default=None, help="CSV/JSON'daki varyant etiketi (vars. --varyant)")
    a = ap.parse_args()
    ws = Path(a.ws)
    etiket_ad = a.etiket or a.varyant
    kos = [Path(k) for k in a.kosumlar] if a.kosumlar else kosumlari_bul(ws, a.varyant)
    assert kos, "koşum bulunamadı"

    # --- etiketler ve lezyon eşlemesi (yalnız val + train lezyon kümesi) ---
    npz = np.load(ws / "data/medmnist/dermamnist.npz")
    y_npz = npz["val_labels"].ravel().astype(int)
    tr_lez, val_lez, val_dx = set(), {}, {}
    with open(ws / "data/splits/dermamnist_lezyon_esleme.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "train":
                tr_lez.add(r["lesion_id"])
            elif r["split"] == "val":
                val_lez[int(r["index"])] = r["lesion_id"]
                val_dx[int(r["index"])] = r["dx"]
    N = len(y_npz)
    S = np.array([val_lez[i] in tr_lez for i in range(N)])  # lezyonu train'de olan
    assert all(SINIF[y_npz[i]] == val_dx[i] for i in range(N)), "lezyon CSV dx ↔ npz etiket uyuşmuyor"

    P, Pson, seeds, en_iyi, gecmis = [], [], [], [], []
    for d in kos:
        y = np.load(d / "etiket_val.npy").ravel().astype(int)
        assert np.array_equal(y, y_npz), f"{d.name}: etiket_val npz ile hizasız"
        P.append(np.load(d / "tahmin_val.npy").astype(np.float64))
        if (d / "tahmin_val_son.npy").exists():
            Pson.append(np.load(d / "tahmin_val_son.npy").astype(np.float64))
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        seeds.append(int(m.get("seed", -1)))
        en_iyi.append(m.get("en_iyi_epoch"))
        if (d / "gecmis.json").exists():
            g = json.loads((d / "gecmis.json").read_text(encoding="utf-8"))
            gecmis.append([{k: v for k, v in e.items() if not k.startswith("test")} for e in g])  # test atılır
    y = y_npz
    P = np.stack(P)
    Pson = np.stack(Pson) if len(Pson) == len(kos) else None
    ns = len(P)
    T = P.argmax(2)  # (seed, N)
    D = (T == y[None])  # doğru mu

    satir = []

    def ekle(bolum, olcu, degerler, alt="tum", sinif="", notu=""):
        v = np.asarray(degerler, dtype=float)
        satir.append([etiket_ad, bolum, olcu, alt, sinif, f"{np.nanmean(v):.6f}",
                      f"{np.nanstd(v, ddof=1):.6f}" if len(v) > 1 else "", len(v), notu])

    oz = {"varyant": etiket_ad, "kosumlar": [d.name for d in kos], "seed": seeds, "en_iyi_epoch": en_iyi,
          "N_val": int(N), "S_say": int(S.sum()), "N_say": int((~S).sum())}

    # ---------------- 1-2. genel ve sınıf bazlı ----------------
    genel = [ozet_olcu(y, P[s]) for s in range(ns)]
    for k in ["acc", "auc", "dengeli_acc", "makro_f1", "ece15", "nll", "brier", "ort_guven", "tahmin_nv_payi"]:
        ekle("genel", k, [g[k] for g in genel], notu="secilen epoch (val-AUC)")
    oz["genel"] = {k: [g[k] for g in genel] for k in genel[0] if not k.startswith("_")}
    destek = np.bincount(y, minlength=K)
    oz["destek"] = destek.tolist()
    oz["cogunluk_taban_acc"] = float(destek[NV] / N)
    rec = np.array([recall_score(y, T[s], labels=list(range(K)), average=None, zero_division=0) for s in range(ns)])
    pre = np.array([precision_score(y, T[s], labels=list(range(K)), average=None, zero_division=0) for s in range(ns)])
    f1 = np.array([f1_score(y, T[s], labels=list(range(K)), average=None, zero_division=0) for s in range(ns)])
    tahmin_say = np.array([np.bincount(T[s], minlength=K) for s in range(ns)])
    auc_sinif = np.array([[roc_auc_score((y == c).astype(int), P[s][:, c]) for c in range(K)] for s in range(ns)])
    for c in range(K):
        ekle("sinif", "duyarlilik", rec[:, c], sinif=SINIF[c], notu=f"destek={destek[c]}")
        ekle("sinif", "kesinlik", pre[:, c], sinif=SINIF[c], notu="tahmin yoksa 0 (zero_division)")
        ekle("sinif", "f1", f1[:, c], sinif=SINIF[c])
        ekle("sinif", "auc_ovr", auc_sinif[:, c], sinif=SINIF[c])
        ekle("sinif", "tahmin_sayisi", tahmin_say[:, c], sinif=SINIF[c], notu=f"gercek={destek[c]}")
        hata_c = [(T[s][y == c] != c).sum() for s in range(ns)]
        ekle("sinif", "hata_sayisi", hata_c, sinif=SINIF[c])
        ekle("sinif", "hata_kutlesi_payi", [h / (N - D[s].sum()) for s, h in enumerate(hata_c)], sinif=SINIF[c],
             notu="toplam yanlışların bu gerçek sınıftan gelen payı")
    oz["sinif"] = {"duyarlilik": rec.tolist(), "kesinlik": pre.tolist(), "f1": f1.tolist(),
                   "auc": auc_sinif.tolist(), "tahmin_say": tahmin_say.tolist()}
    # karışıklık matrisi (sayım; seed ortalaması ve std)
    CM = np.zeros((ns, K, K))
    for s in range(ns):
        np.add.at(CM[s], (y, T[s]), 1)
    cm_ort, cm_std = CM.mean(0), CM.std(0, ddof=1)
    for i in range(K):
        for j in range(K):
            ekle("karisiklik", f"gercek={SINIF[i]}|tahmin={SINIF[j]}", CM[:, i, j], sinif=SINIF[i])
    oz["karisiklik_ort"] = cm_ort.tolist()
    oz["karisiklik_std"] = cm_std.tolist()
    # nv'ye çöken hatalar
    nv_ye = [((T[s] == NV) & (y != NV)).sum() for s in range(ns)]
    toplam_hata = [int(N - D[s].sum()) for s in range(ns)]
    ekle("cogunluk", "nv_disi_ornegin_nv_tahmini", nv_ye, notu="gerçek≠nv iken tahmin=nv (adet)")
    ekle("cogunluk", "hatalarin_nv_tahmini_payi", np.array(nv_ye) / np.array(toplam_hata))
    ekle("cogunluk", "nv_disi_duyarlilik", [D[s][y != NV].mean() for s in range(ns)],
         notu="nv dışı 332 örnekte doğruluk")
    ekle("cogunluk", "nadir3_duyarlilik_ort", rec[:, NADIR].mean(1), notu="df+vasc+akiec makro duyarlılık")
    ekle("cogunluk", "nadir3_hata_orani_havuz",
         [1 - D[s][np.isin(y, NADIR)].mean() for s in range(ns)], notu="59 nadir örnek havuzu")

    # ---------------- 3. kalibrasyon ----------------
    kut_say = np.array([[k[1] for k in g["_kutular"]] for g in genel])
    kut_acc = np.array([[k[2] for k in g["_kutular"]] for g in genel])
    kut_guv = np.array([[k[3] for k in g["_kutular"]] for g in genel])
    for i in range(15):
        lo, hi = i / 15, (i + 1) / 15
        ekle("kalibrasyon", "kutu_sayisi", kut_say[:, i], sinif=f"({lo:.3f},{hi:.3f}]")
        ekle("kalibrasyon", "kutu_dogruluk", kut_acc[:, i], sinif=f"({lo:.3f},{hi:.3f}]")
        ekle("kalibrasyon", "kutu_ort_guven", kut_guv[:, i], sinif=f"({lo:.3f},{hi:.3f}]")
    asiri = [(P[s].max(1) - D[s]).mean() for s in range(ns)]  # ort güven − acc
    ekle("kalibrasyon", "guven_eksi_acc", asiri, notu=">0 aşırı güven")
    yg_yanlis = [((P[s].max(1) >= 0.9) & ~D[s]).sum() for s in range(ns)]
    ekle("kalibrasyon", "guven>=0.9_yanlis_adet", yg_yanlis)
    yg = [(P[s].max(1) >= 0.9).sum() for s in range(ns)]
    ekle("kalibrasyon", "guven>=0.9_adet", yg)
    ekle("kalibrasyon", "guven>=0.9_dogruluk", [D[s][P[s].max(1) >= 0.9].mean() if yg[s] else np.nan for s in range(ns)])
    # sınıf bazlı: nv tahminlerinin güveni, yanlış nv tahminlerinin güveni
    ekle("kalibrasyon", "yanlis_nv_tahmin_ort_guven",
         [P[s].max(1)[(T[s] == NV) & ~D[s]].mean() for s in range(ns)])
    oz["kalibrasyon"] = {"kut_say": kut_say.tolist(), "kut_acc": kut_acc.tolist(), "kut_guv": kut_guv.tolist(),
                         "guven_eksi_acc": asiri, "yg_yanlis": yg_yanlis, "yg": yg}
    if Pson is not None:
        son = [ozet_olcu(y, Pson[s]) for s in range(ns)]
        for k in ["acc", "auc", "dengeli_acc", "makro_f1", "ece15", "nll", "ort_guven", "tahmin_nv_payi"]:
            ekle("son_epoch", k, [g[k] for g in son], notu="son epoch (seçimsiz; baseline 100, H02 30)")
        oz["son_epoch"] = {k: [g[k] for g in son] for k in son[0] if not k.startswith("_")}
        Tson = Pson.argmax(2)
        oz["son_epoch"]["sinif_duyarlilik"] = [[float((Tson[s][y == c] == c).mean()) for c in range(K)]
                                               for s in range(ns)]
        for c in range(K):
            ekle("son_epoch", "duyarlilik", [(Tson[s][y == c] == c).mean() for s in range(ns)], sinif=SINIF[c])

    # ---------------- 4. lezyon sızıntısı S/N ----------------
    accS = [D[s][S].mean() for s in range(ns)]
    accN = [D[s][~S].mean() for s in range(ns)]
    ekle("sizinti", "acc", accS, alt="S_lezyon_trainde", notu=f"n={S.sum()}")
    ekle("sizinti", "acc", accN, alt="N_lezyon_trainde_yok", notu=f"n={(~S).sum()}")
    ekle("sizinti", "acc_farki_S_eksi_N", np.array(accS) - np.array(accN))
    # sınıf bileşimi düzeltmesi: S'nin sınıf-koşullu doğruluğu N'nin sınıf oranlarıyla
    wN = np.bincount(y[~S], minlength=K) / (~S).sum()
    adj = []
    for s in range(ns):
        cs = [(c, D[s][S & (y == c)].mean()) for c in range(K) if (S & (y == c)).any()]
        w = np.array([wN[c] for c, _ in cs]); w = w / w.sum()
        adj.append(float(np.sum(w * np.array([v for _, v in cs]))))
    ekle("sizinti", "acc_S_Nagirlikli", adj, notu="S sınıf-koşullu ACC, N sınıf oranlarıyla")
    ekle("sizinti", "acc_farki_Nagirlikli_S_eksi_N", np.array(adj) - np.array(accN))
    ekle("sizinti", "dengeli_acc", [dengeli(y[S], T[s][S]) for s in range(ns)], alt="S_lezyon_trainde")
    ekle("sizinti", "dengeli_acc", [dengeli(y[~S], T[s][~S]) for s in range(ns)], alt="N_lezyon_trainde_yok")
    ekle("sizinti", "nv_payi_gercek", [np.mean(y[S] == NV)], alt="S_lezyon_trainde")
    ekle("sizinti", "nv_payi_gercek", [np.mean(y[~S] == NV)], alt="N_lezyon_trainde_yok")
    ekle("sizinti", "ece15", [ece(y[S], P[s][S])[0] for s in range(ns)], alt="S_lezyon_trainde")
    ekle("sizinti", "ece15", [ece(y[~S], P[s][~S])[0] for s in range(ns)], alt="N_lezyon_trainde_yok")
    ekle("sizinti", "ort_guven", [P[s][S].max(1).mean() for s in range(ns)], alt="S_lezyon_trainde")
    ekle("sizinti", "ort_guven", [P[s][~S].max(1).mean() for s in range(ns)], alt="N_lezyon_trainde_yok")
    for c in range(K):
        nS, nN = int((S & (y == c)).sum()), int((~S & (y == c)).sum())
        if nS:
            ekle("sizinti", "duyarlilik", [D[s][S & (y == c)].mean() for s in range(ns)], alt="S_lezyon_trainde",
                 sinif=SINIF[c], notu=f"n={nS}")
        if nN:
            ekle("sizinti", "duyarlilik", [D[s][~S & (y == c)].mean() for s in range(ns)], alt="N_lezyon_trainde_yok",
                 sinif=SINIF[c], notu=f"n={nN}")
    # sınıf içi permütasyon testi (görüntü düzeyinde, seed-ortalamalı doğruluk; S/N etiketi sınıf içinde karıştırılır)
    dbar = D.mean(0)
    def istat(Smask):
        a_s = []
        for c in range(K):
            mS, mN = Smask & (y == c), ~Smask & (y == c)
            if mS.any() and mN.any():
                a_s.append(wN[c] * (dbar[mS].mean() - dbar[mN].mean()))
        return float(np.sum(a_s))
    gozlenen = istat(S)
    rng = np.random.default_rng(0)
    perm = []
    for _ in range(10000):
        Sp = S.copy()
        for c in range(K):
            idx = np.where(y == c)[0]
            Sp[idx] = rng.permutation(S[idx])
        perm.append(istat(Sp))
    perm = np.array(perm)
    p_perm = float((np.sum(perm >= gozlenen) + 1) / (len(perm) + 1))
    ekle("sizinti", "sinif_ici_fark_istatistigi", [gozlenen], notu=f"perm p(tek yönlü)={p_perm:.4g}; 10000 perm, RNG0")
    # bootstrap GA (görüntü düzeyi, sınıf-tabakalı) sınıf-düzeltilmiş fark
    bs = []
    for _ in range(2000):
        idx = np.concatenate([rng.choice(np.where(y == c)[0], (y == c).sum()) for c in range(K)])
        yb, Sb, db = y[idx], S[idx], dbar[idx]
        a_s = []
        for c in range(K):
            mS, mN = Sb & (yb == c), ~Sb & (yb == c)
            if mS.any() and mN.any():
                a_s.append(wN[c] * (db[mS].mean() - db[mN].mean()))
        bs.append(np.sum(a_s))
    oz["sizinti"] = {"accS": accS, "accN": accN, "accS_Nagirlikli": adj, "sinif_ici_istat": gozlenen,
                     "p_perm": p_perm, "bs_ga": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                     "S_sinif": np.bincount(y[S], minlength=K).tolist(),
                     "N_sinif": np.bincount(y[~S], minlength=K).tolist()}
    # --- karıştırıcı denetimi: HAM10000 dx_type / dataset (kaynak) ve tabakalı S−N ---
    meta = {}
    mpath = ws / "data/meta/HAM10000_metadata.csv"
    if mpath.exists():
        with open(mpath, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                meta[r["image_id"]] = r
        with open(ws / "data/splits/dermamnist_lezyon_esleme.csv", encoding="utf-8") as f:
            vimg = {int(r["index"]): r["image_id"] for r in csv.DictReader(f) if r["split"] == "val"}
        dxt = np.array([meta[vimg[i]]["dx_type"] for i in range(N)])
        kay = np.array([meta[vimg[i]]["dataset"] for i in range(N)])
        for ad, arr in [("dx_type", dxt), ("kaynak", kay)]:
            for v in sorted(set(arr)):
                m = arr == v
                ekle("kaynak", f"acc|{ad}={v}", [D[s][m].mean() for s in range(ns)], notu=f"n={m.sum()} S={int((m & S).sum())}")
                if (m & S).any() and (m & ~S).any():
                    ekle("kaynak", f"acc|{ad}={v}", [D[s][m & S].mean() for s in range(ns)], alt="S_lezyon_trainde")
                    ekle("kaynak", f"acc|{ad}={v}", [D[s][m & ~S].mean() for s in range(ns)], alt="N_lezyon_trainde_yok")
        mm = kay == "vidir_molemax"
        ekle("kaynak", "acc|molemax_disi", [D[s][~mm].mean() for s in range(ns)], notu=f"n={(~mm).sum()}")
        ekle("kaynak", "dengeli_acc|molemax_disi", [dengeli(y[~mm], T[s][~mm]) for s in range(ns)])
        tab = np.array([f"{y[i]}|{dxt[i]}|{kay[i]}" for i in range(N)])

        def tabakali(Dv, Smask):
            top = w = 0.0
            for k in sorted(set(tab)):  # sıralı: PYTHONHASHSEED'den bağımsız (tahlil-tur1)
                m = tab == k
                if (m & Smask).any() and (m & ~Smask).any():
                    wk = (m & ~Smask).sum()
                    top += wk * (Dv[m & Smask].mean() - Dv[m & ~Smask].mean()); w += wk
            return top / w, w / (~Smask).sum()
        tf = [tabakali(D[s].astype(float), S)[0] for s in range(ns)]
        g_t, kaps = tabakali(dbar, S)
        ekle("sizinti", "acc_farki_tabakali_sinif_dxtype_kaynak", tf,
             notu=f"N-ağırlıklı; tabakaların kapsadığı N payı={kaps:.3f}")
        pt = []
        for _ in range(10000):
            Sp = S.copy()
            for k in sorted(set(tab)):  # sıralı: PYTHONHASHSEED'den bağımsız (tahlil-tur1)
                idx = np.where(tab == k)[0]
                Sp[idx] = rng.permutation(S[idx])
            pt.append(tabakali(dbar, Sp)[0])
        pt = np.array(pt)
        p_ust = float((np.sum(pt >= g_t) + 1) / (len(pt) + 1))
        p_iki = float((np.sum(np.abs(pt) >= abs(g_t)) + 1) / (len(pt) + 1))
        ekle("sizinti", "tabakali_fark_permutasyon", [g_t], notu=f"p_tek(S>N)={p_ust:.4g}; p_iki={p_iki:.4g}; 10000 perm")
        oz["sizinti"].update({"tabakali_fark": g_t, "tabakali_seed": tf, "tabakali_kapsam": kaps,
                              "tabakali_p_tek": p_ust, "tabakali_p_iki": p_iki,
                              "follow_up_S": int(((dxt == "follow_up") & S).sum()),
                              "follow_up_n": int((dxt == "follow_up").sum())})
        if Pson is not None:
            Dson_ = Pson.argmax(2) == y[None]
            tfs = [tabakali(Dson_[s].astype(float), S)[0] for s in range(ns)]
            ekle("sizinti_son_epoch", "acc_farki_tabakali_sinif_dxtype_kaynak", tfs, notu="son epoch (baseline: 100, train_acc=1.0)")
            oz["sizinti"]["tabakali_son_seed"] = tfs
    cpath = ws / "data/splits/dermamnist_c_esleme.csv"
    if cpath.exists():
        cval = np.zeros(N, bool)
        with open(cpath, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["split_resmi"] == "val" and r["split_C"] == "val":
                    cval[int(r["index_resmi"])] = True
        ekle("dermamnist_c", "acc|C_val_altkumesi", [D[s][cval].mean() for s in range(ns)],
             notu=f"resmî val'in C-val'de kalan {cval.sum()} görüntüsü; model resmî train'de eğitildi")
        ekle("dermamnist_c", "acc|C_traine_tasinan_val", [D[s][~cval].mean() for s in range(ns)], notu=f"n={(~cval).sum()}")
        ekle("dermamnist_c", "dengeli_acc|C_val_altkumesi", [dengeli(y[cval], T[s][cval]) for s in range(ns)])
        ekle("dermamnist_c", "nv_payi|C_val_altkumesi", [float(np.mean(y[cval] == NV))])
        ekle("dermamnist_c", "cogunluk_taban|C_val_altkumesi", [float(np.mean(y[cval] == NV))])
        oz["dermamnist_c"] = {"n": int(cval.sum()), "acc": [float(D[s][cval].mean()) for s in range(ns)],
                              "acc_tasinan": [float(D[s][~cval].mean()) for s in range(ns)],
                              "nv_payi": float(np.mean(y[cval] == NV))}
    p_iki_sinif = float((np.sum(np.abs(perm) >= abs(gozlenen)) + 1) / (len(perm) + 1))
    oz["sizinti"]["p_perm_iki_sinif_ici"] = p_iki_sinif
    ekle("sizinti", "sinif_ici_fark_permutasyon_iki_yonlu", [gozlenen], notu=f"p_iki={p_iki_sinif:.4g}")
    if Pson is not None:
        Dson = Pson.argmax(2) == y[None]
        aS = [Dson[s][S].mean() for s in range(ns)]; aN = [Dson[s][~S].mean() for s in range(ns)]
        ekle("sizinti_son_epoch", "acc", aS, alt="S_lezyon_trainde")
        ekle("sizinti_son_epoch", "acc", aN, alt="N_lezyon_trainde_yok")
        ekle("sizinti_son_epoch", "acc_farki_S_eksi_N", np.array(aS) - np.array(aN))
        oz["sizinti_son"] = {"accS": aS, "accN": aN}

    # ---------------- 5. seed varyansı ve seçim oynaklığı ----------------
    ekle("seed", "en_iyi_epoch", en_iyi)
    ciftler = [(i, j) for i in range(ns) for j in range(i + 1, ns)]
    anlasmazlik = [(T[i] != T[j]).mean() for i, j in ciftler]
    ekle("seed", "ciftli_tahmin_anlasmazligi", anlasmazlik, notu="seed çiftleri arasında farklı argmax oranı (10 çift)")
    if gecmis:
        maxacc = [max(e["val_acc"] for e in g) for g in gecmis]
        argmaxacc = [int(np.argmax([e["val_acc"] for e in g])) + 1 for g in gecmis]
        maxauc = [max(e["val_auc"] for e in g) for g in gecmis]
        ekle("seed", "val_acc_epoch_maks", maxacc, notu="100 epoch içinde en yüksek val_acc (iyimser tavan)")
        ekle("seed", "val_acc_maks_epoch", argmaxacc)
        ekle("seed", "val_auc_epoch_maks", maxauc)
        # epoch 5-20 penceresinde val_acc'nin epochtan epocha oynaklığı
        pencere = [np.std([e["val_acc"] for e in g if 5 <= e["epoch"] <= 20], ddof=1) for g in gecmis]
        ekle("seed", "val_acc_std_epoch5_20", pencere, notu="seçimin düştüğü pencerede epoch-epoch oynaklık")
        komsu = []
        for g, ei in zip(gecmis, en_iyi):
            v = [e["val_acc"] for e in g if abs(e["epoch"] - ei) <= 2]
            komsu.append(max(v) - min(v))
        ekle("seed", "val_acc_secilen±2epoch_aralik", komsu)
        tr_son = [g[-1].get("train_acc") for g in gecmis]
        ekle("seed", "train_acc_son_epoch", tr_son)
        # val_loss her geçmişte olmayabilir (H02 ince_ayar.py gecmis.json'u yazmaz; tahlil-tur1 2026-10-11 düzeltmesi)
        vl_min_ep = ([int(np.argmin([e["val_loss"] for e in g])) + 1 for g in gecmis]
                     if all("val_loss" in e for g in gecmis for e in g) else [])
        if vl_min_ep:
            ekle("seed", "val_loss_min_epoch", vl_min_ep)
        oz["gecmis_val_acc"] = [[e["val_acc"] for e in g] for g in gecmis]
        oz["gecmis_val_auc"] = [[e["val_auc"] for e in g] for g in gecmis]
        oz["gecmis_train_acc"] = [[e.get("train_acc") for e in g] for g in gecmis]
        oz["seed_ek"] = {"maxacc": maxacc, "argmaxacc": argmaxacc, "maxauc": maxauc, "pencere_std": pencere,
                         "komsu": komsu, "tr_son": tr_son, "vl_min_ep": vl_min_ep}
    # val ACC örneklem SE ve 5-seed eşleştirilmiş en küçük saptanabilir etki (MDE, %80 güç, tek yönlü α=.05)
    acc_ort = float(np.mean([g["acc"] for g in genel])); acc_sd = float(np.std([g["acc"] for g in genel], ddof=1))
    se_binom = math.sqrt(acc_ort * (1 - acc_ort) / N)
    t4, z80 = 2.1318, 0.8416  # t_{0.95,4}; z_{0.80}
    oz["mde"] = {"se_binom": se_binom, "acc_sd": acc_sd,
                 "mde_esli_bagimsiz_varyans": (t4 + z80) * math.sqrt(2) * acc_sd / math.sqrt(ns),
                 "mde_esli_sonda_ebeveyn": (t4 + z80) * acc_sd / math.sqrt(ns)}
    ekle("seed", "val_acc_binom_SE", [se_binom], notu="sqrt(p(1-p)/1003)")

    # ---------------- 6. zor örnekler ----------------
    yanlis_say = (~D).sum(0)
    hep_yanlis = yanlis_say == ns
    hep_dogru = yanlis_say == 0
    for c in range(K):
        m = y == c
        ekle("zor", "hep_yanlis_adet", [int((hep_yanlis & m).sum())], sinif=SINIF[c], notu=f"destek={m.sum()}")
        ekle("zor", "hep_yanlis_adet", [int((hep_yanlis & m & S).sum())], alt="S_lezyon_trainde", sinif=SINIF[c])
        ekle("zor", "hep_yanlis_adet", [int((hep_yanlis & m & ~S).sum())], alt="N_lezyon_trainde_yok", sinif=SINIF[c])
        ekle("zor", "hep_dogru_adet", [int((hep_dogru & m).sum())], sinif=SINIF[c])
    ekle("zor", "hep_yanlis_adet", [int(hep_yanlis.sum())])
    ekle("zor", "hep_dogru_adet", [int(hep_dogru.sum())])
    ekle("zor", "kararsiz_adet", [int(((yanlis_say > 0) & (yanlis_say < ns)).sum())], notu="1-4 seed yanlış")
    ekle("zor", "hep_yanlis_orani", [hep_yanlis[S].mean()], alt="S_lezyon_trainde")
    ekle("zor", "hep_yanlis_orani", [hep_yanlis[~S].mean()], alt="N_lezyon_trainde_yok")
    # hep-yanlışlar neye tahmin ediliyor (çoğunluk oyu)
    oy = np.array([np.bincount(T[:, i], minlength=K).argmax() for i in range(N)])
    hy_hedef = np.zeros((K, K), int)
    np.add.at(hy_hedef, (y[hep_yanlis], oy[hep_yanlis]), 1)
    for i in range(K):
        for j in range(K):
            if hy_hedef[i, j]:
                ekle("zor", f"hep_yanlis|gercek={SINIF[i]}|oy={SINIF[j]}", [hy_hedef[i, j]], sinif=SINIF[i])
    # hep-yanlışlarda ortalama güven (yanlış sınıfa güven)
    ekle("zor", "hep_yanlis_ort_guven", [P[:, hep_yanlis].max(2).mean()])
    oz["zor"] = {"hep_yanlis": int(hep_yanlis.sum()), "hep_dogru": int(hep_dogru.sum()),
                 "hep_yanlis_sinif": np.bincount(y[hep_yanlis], minlength=K).tolist(),
                 "hep_yanlis_S_sinif": np.bincount(y[hep_yanlis & S], minlength=K).tolist(),
                 "hep_yanlis_N_sinif": np.bincount(y[hep_yanlis & ~S], minlength=K).tolist(),
                 "hep_yanlis_hedef": hy_hedef.tolist(),
                 "hep_yanlis_S_orani": float(hep_yanlis[S].mean()), "hep_yanlis_N_orani": float(hep_yanlis[~S].mean()),
                 "hep_yanlis_indeks": np.where(hep_yanlis)[0].tolist()}

    # ---------------- 7. teşhis (parametresiz, val'de ayar YOK) ----------------
    # (a) 5-seed olasılık ortalaması (A9 tavan göstergesi)
    Pens = P.mean(0)
    ens = ozet_olcu(y, Pens)
    for k in ["acc", "auc", "dengeli_acc", "makro_f1", "ece15"]:
        ekle("teshis", f"ens5_{k}", [ens[k]], notu="5 seed olasılık ortalaması; teşhis, kart sonucu değil")
    # (b) train önsel düzeltmesi p(c|x)/π_train(c) (logit ayarı τ=1, Menon vd. 2021) — parametre val'de seçilmez
    pi = np.bincount(npz["train_labels"].ravel().astype(int), minlength=K) / len(npz["train_labels"])
    oz["pi_train"] = pi.tolist()
    adj_ol = []
    for s in range(ns):
        q = P[s] / pi[None]; q = q / q.sum(1, keepdims=True)
        adj_ol.append(ozet_olcu(y, q))
    for k in ["acc", "dengeli_acc", "makro_f1", "auc", "tahmin_nv_payi"]:
        ekle("teshis", f"onsel_duzeltme_{k}", [g[k] for g in adj_ol],
             notu="p/π_train, τ=1 sabit (Menon 2021); val'de ayar yok; teşhis")
    oz["teshis"] = {"ens": {k: v for k, v in ens.items() if not k.startswith("_")},
                    "onsel": {k: [g[k] for g in adj_ol] for k in adj_ol[0] if not k.startswith("_")}}

    cikti_csv = Path(a.cikti_csv) if a.cikti_csv else ws / "sonuclar/hata-kirilim.csv"
    cikti_json = Path(a.cikti_json) if a.cikti_json else ws / "sonuclar/hata_kirilim_ozet.json"
    with open(cikti_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["varyant", "bolum", "olcu", "alt_kume", "sinif", "deger_ort", "deger_std", "n_seed", "not"])
        w.writerows(satir)
    for g in genel:
        g.pop("_kutular", None)
    cikti_json.write_text(json.dumps(oz, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"yazıldı: {cikti_csv} ({len(satir)} satır), {cikti_json}; koşum={len(kos)} seed={seeds}")


if __name__ == "__main__":
    main()
