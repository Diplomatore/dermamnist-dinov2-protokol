# -*- coding: utf-8 -*-
"""K2 veri hazırlığı — 3.1 DermaMNIST (MedMNIST v3 / MedMNIST+).

Aşamalar:
  indir  : medmnist paketiyle DermaMNIST 28/64/128/224 .npz (download=True, resmî MD5 kontrolü
           torchvision.download_url içinde) + sızıntı denetimi için iki metadata CSV:
             - HAM10000_metadata (Harvard Dataverse, doi:10.7910/DVN/DBW86T, dosya id 4338392)
             - dermamnist_split_info.csv (Abhishek vd. 2025, GitHub, sabit commit) — DermaMNIST
               index -> HAM10000 image_id eşlemesi (MedMNIST yazarlarından alınmış dosya adları)
  analiz : SHA256/MD5/boyut, split sayıları, sınıf dağılımı, şekil/dtype, boyutlar arası sıra
           tutarlılığı, tam-piksel hash çakışması, 28 px yakın-kopya (RMSE + dHash) ve lezyon-id
           çakışması. Sonuç: data/data_manifest.json (+ yardımcı CSV/PNG).

Resmî split DEĞİŞTİRİLMEZ; yalnız ölçülür. Test görüntüleri yalnız istatistik (hash/mesafe) için okunur.

Kullanım (WS kökünden):
  env/.venv/Scripts/python.exe scripts/veri_hazirla.py --asama hepsi
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
DATA = WS / "data"
ROOT = DATA / "medmnist"
META = DATA / "meta"
SPLITS = DATA / "splits"
RAPOR = DATA / "raporlar"
os.environ.setdefault("HF_HOME", str(DATA / "hf_cache"))
os.environ.setdefault("TORCH_HOME", str(DATA / "torch_cache"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np  # noqa: E402

BOYUTLAR = [28, 64, 128, 224]
SPLIT_ADLARI = ["train", "val", "test"]
DATAVERSE_META_URL = "https://dataverse.harvard.edu/api/access/datafile/4338392?format=original"
GH_COMMIT = "ac655ac5a5b93264c57d69d6d520778d915bfc89"
SPLITINFO_URL = (
    "https://raw.githubusercontent.com/kakumarabhishek/Corrected-Skin-Image-Datasets/"
    f"{GH_COMMIT}/DermaMNIST/DermaMNIST_Analysis/dermamnist_split_info.csv"
)
# HAM10000 dx kodu -> MedMNIST etiket indeksi (INFO['dermamnist']['label'] sırası)
DX2LBL = {"akiec": 0, "bcc": 1, "bkl": 2, "df": 3, "mel": 4, "nv": 5, "vasc": 6}


def dosya_hash(p, algo):
    h = hashlib.new(algo)
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def npz_yolu(s):
    return ROOT / ("dermamnist.npz" if s == 28 else f"dermamnist_{s}.npz")


# ----------------------------------------------------------------------------- indir
def indir(boyutlar):
    import medmnist

    ROOT.mkdir(parents=True, exist_ok=True)
    META.mkdir(parents=True, exist_ok=True)
    kp = DATA / "indirme_kaydi.json"
    kayit = json.loads(kp.read_text(encoding="utf-8")) if kp.exists() else {}
    for s in boyutlar:
        t0 = time.time()
        onceden = npz_yolu(s).exists()
        ds = medmnist.DermaMNIST(split="train", size=s, root=str(ROOT), download=True)
        kayit.setdefault(str(s), {"dosya": npz_yolu(s).name, "kosumlar": []})
        kayit[str(s)]["kosumlar"].append({"tarih": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                          "sure_sn": round(time.time() - t0, 1),
                                          "durum": "onceden_var_md5_dogrulandi" if onceden else "indirildi"})
        print(f"[indir] {s}px tamam ({kayit[str(s)]['kosumlar'][-1]}) -> {npz_yolu(s)}", flush=True)
        del ds
    for ad, url in [("HAM10000_metadata.csv", DATAVERSE_META_URL),
                    ("dermamnist_split_info.csv", SPLITINFO_URL)]:
        hedef = META / ad
        if not hedef.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(hedef, "wb") as f:
                f.write(r.read())
        kayit[ad] = {"url": url}
        print(f"[indir] {ad} tamam", flush=True)
    (DATA / "indirme_kaydi.json").write_text(json.dumps(kayit, indent=1, ensure_ascii=False), encoding="utf-8")


# ----------------------------------------------------------------------------- yardımcılar
def img_hash(arr):
    return np.array([hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest() for a in arr])


def nn_rmse(A, B, ayni=False, parca=1024):
    """A'nın her satırı için B'deki en yakın komşu RMSE'si (piksel 0-255) ve indeksi."""
    A = A.reshape(len(A), -1).astype(np.float64)
    B = B.reshape(len(B), -1).astype(np.float64)
    d = A.shape[1]
    nb = (B ** 2).sum(1)
    en_d = np.empty(len(A))
    en_i = np.empty(len(A), dtype=np.int64)
    for i in range(0, len(A), parca):
        a = A[i:i + parca]
        D = (a ** 2).sum(1)[:, None] + nb[None, :] - 2 * a @ B.T
        if ayni:
            idx = np.arange(i, i + len(a))
            D[np.arange(len(a)), idx] = np.inf
        j = D.argmin(1)
        en_i[i:i + len(a)] = j
        en_d[i:i + len(a)] = np.sqrt(np.maximum(D[np.arange(len(a)), j], 0) / d)
    return en_d, en_i


def dhash(arr):
    """28x28x3 -> gri -> 9x8 (PIL bilinear) -> 64 bit fark hash'i (bool dizi)."""
    from PIL import Image

    out = np.empty((len(arr), 64), dtype=bool)
    for k, a in enumerate(arr):
        g = np.asarray(Image.fromarray(a).convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
        out[k] = (g[:, 1:] > g[:, :-1]).ravel()
    return out


def nn_hamming(HA, HB, ayni=False):
    a = HA.astype(np.int32)
    b = HB.astype(np.int32)
    D = a @ (1 - b).T + (1 - a) @ b.T
    if ayni:
        np.fill_diagonal(D, 10 ** 6)
    j = D.argmin(1)
    return D[np.arange(len(a)), j], j


def ozet(x):
    x = np.asarray(x, dtype=float)
    q = np.percentile(x, [0, 1, 5, 25, 50, 75, 95])
    return {"min": round(q[0], 3), "p1": round(q[1], 3), "p5": round(q[2], 3), "p25": round(q[3], 3),
            "medyan": round(q[4], 3), "p75": round(q[5], 3), "p95": round(q[6], 3),
            "ort": round(float(x.mean()), 3)}


# ----------------------------------------------------------------------------- analiz
def analiz(boyutlar):
    import medmnist
    import pandas as pd
    from PIL import Image

    info = medmnist.INFO["dermamnist"]
    sinif_ad = info["label"]
    sonuc = {"veri_seti": "DermaMNIST", "medmnist_surum": medmnist.__version__,
             "numpy_surum": np.__version__, "zenodo_kayit": "10.5281/zenodo.10519652",
             "info_lisans": info["license"], "info_n_samples": info["n_samples"], "dosyalar": {},
             "boyutlar": {}}
    if (DATA / "indirme_kaydi.json").exists():
        sonuc["indirme_kaydi"] = json.loads((DATA / "indirme_kaydi.json").read_text(encoding="utf-8"))

    veri = {}
    for s in boyutlar:
        p = npz_yolu(s)
        md5 = dosya_hash(p, "md5")
        resmi = info["MD5" if s == 28 else f"MD5_{s}"]
        sonuc["dosyalar"][p.name] = {"bayt": p.stat().st_size, "sha256": dosya_hash(p, "sha256"),
                                     "md5": md5, "md5_resmi": resmi, "md5_eslesme": md5 == resmi}
        z = np.load(p)
        b = {"anahtarlar": sorted(z.files), "split": {}}
        veri[s] = {}
        for sp in SPLIT_ADLARI:
            X = z[f"{sp}_images"]
            y = z[f"{sp}_labels"]
            sayim = np.bincount(y.ravel(), minlength=7)
            b["split"][sp] = {
                "n": int(len(X)), "goruntu_sekli": list(X.shape[1:]), "goruntu_dtype": str(X.dtype),
                "etiket_sekli": list(y.shape[1:]), "etiket_dtype": str(y.dtype),
                "piksel_min": int(X.min()), "piksel_max": int(X.max()),
                "beklenen_n": info["n_samples"][sp], "n_eslesme": int(len(X)) == info["n_samples"][sp],
                "sinif_sayim": {f"{k} {sinif_ad[str(k)]}": int(sayim[k]) for k in range(7)},
                "dengesizlik_orani_max_min": round(float(sayim.max() / sayim.min()), 2),
            }
            veri[s][sp] = (X, y.ravel())
        # split'ler arası / içi tam-piksel kopya
        hs = {sp: img_hash(veri[s][sp][0]) for sp in SPLIT_ADLARI}
        tam = {}
        for sp in SPLIT_ADLARI:
            u, c = np.unique(hs[sp], return_counts=True)
            tam[f"{sp}_ici_kopya_goruntu"] = int((c[c > 1]).sum())
        for a_, b_ in [("train", "val"), ("train", "test"), ("val", "test")]:
            ortak = set(hs[a_]) & set(hs[b_])
            tam[f"{a_}∩{b_}_hash"] = len(ortak)
            tam[f"{b_}_ornek_{a_}_ile_ayni"] = int(np.isin(hs[b_], list(ortak)).sum())
        b["tam_piksel_kopya"] = tam
        sonuc["boyutlar"][str(s)] = b
        print(f"[analiz] {s}px: n={[b['split'][sp]['n'] for sp in SPLIT_ADLARI]} md5_ok="
              f"{sonuc['dosyalar'][p.name]['md5_eslesme']} tam_kopya={tam}", flush=True)

    # boyutlar arası tutarlılık (aynı sıra / aynı etiket)
    if 28 in veri:
        tut = {}
        for s in boyutlar:
            if s == 28:
                continue
            r = {}
            for sp in SPLIT_ADLARI:
                Xs, ys = veri[s][sp]
                X28, y28 = veri[28][sp]
                kucuk = np.stack([np.asarray(Image.fromarray(a).resize((28, 28), Image.BICUBIC)) for a in Xs])
                e = np.sqrt(((kucuk.astype(float) - X28.astype(float)) ** 2).reshape(len(Xs), -1).mean(1))
                rng = np.random.default_rng(0)
                perm = rng.permutation(len(Xs))
                e_kar = np.sqrt(((kucuk.astype(float) - X28[perm].astype(float)) ** 2).reshape(len(Xs), -1).mean(1))
                r[sp] = {"etiket_ayni": bool((ys == y28).all()),
                         "rmse_ayni_indeks_medyan": round(float(np.median(e)), 2),
                         "rmse_ayni_indeks_max": round(float(e.max()), 2),
                         "rmse_karisik_indeks_medyan": round(float(np.median(e_kar)), 2)}
            tut[str(s)] = r
        sonuc["boyutlar_arasi_tutarlilik_(28e_kucultulmus)"] = tut
        print("[analiz] boyut tutarlılığı:", json.dumps(tut, ensure_ascii=False), flush=True)

    # ---------------- lezyon-id eşlemesi
    meta = pd.read_csv(META / "HAM10000_metadata.csv")
    si = pd.read_csv(META / "dermamnist_split_info.csv")
    m = si.merge(meta[["lesion_id", "image_id", "dx"]], on="image_id", how="left")
    lez = {"ham10000_goruntu": int(len(meta)), "ham10000_lezyon": int(meta.lesion_id.nunique()),
           "split_info_satir": int(len(si)), "eslesmeyen_image_id": int(m.lesion_id.isna().sum())}
    etiket_uyum = 0
    toplam = 0
    for sp in SPLIT_ADLARI:
        y28 = veri[28][sp][1]
        mm = m[m.split == sp].sort_values("index")
        assert (mm["index"].values == np.arange(len(mm))).all()
        uy = (mm.dx.map(DX2LBL).values == y28)
        etiket_uyum += int(uy.sum())
        toplam += len(mm)
    lez["etiket_dx_uyumu"] = f"{etiket_uyum}/{toplam}"
    lez_split = m.groupby("lesion_id").split.agg(lambda s: frozenset(s))
    lez_ad = {}
    for a_, b_ in [("train", "val"), ("train", "test"), ("val", "test")]:
        L = set(lez_split[lez_split.map(lambda f: a_ in f and b_ in f)].index)
        img = m[m.lesion_id.isin(L) & m.split.isin([a_, b_])]
        lez_ad[f"{a_}∩{b_}"] = {"lezyon": len(L), "goruntu_(iki_bolumdeki_toplam)": int(len(img)),
                               f"{b_}_goruntu_sizan": int((img.split == b_).sum())}
    L3 = lez_split[lez_split.map(lambda f: len(f) == 3)]
    coklu = lez_split[lez_split.map(lambda f: len(f) > 1)]
    lez["capraz"] = lez_ad
    lez["uc_bolumde_birden_lezyon"] = int(len(L3))
    lez["uc_bolumde_birden_goruntu"] = int(m.lesion_id.isin(L3.index).sum())
    lez["birden_fazla_bolumde_lezyon"] = int(len(coklu))
    lez["birden_fazla_bolumde_lezyon_orani"] = round(len(coklu) / m.lesion_id.nunique(), 4)
    for sp in ["val", "test"]:
        tr_lez = set(m[m.split == "train"].lesion_id)
        mm = m[m.split == sp]
        lez[f"{sp}_goruntu_lezyonu_trainde_olan"] = int(mm.lesion_id.isin(tr_lez).sum())
        lez[f"{sp}_goruntu_lezyonu_trainde_olan_orani"] = round(float(mm.lesion_id.isin(tr_lez).mean()), 4)
    goruntu_per_lez = meta.groupby("lesion_id").size()
    lez["ham_lezyon_basina_goruntu_dagilimi"] = {str(k): int(v) for k, v in goruntu_per_lez.value_counts().sort_index().items()}
    # split içi tam-piksel kopyaların kimliği (28 px)
    ic = []
    for sp in SPLIT_ADLARI:
        hs = img_hash(veri[28][sp][0])
        u, inv, c = np.unique(hs, return_inverse=True, return_counts=True)
        mm = m[m.split == sp].sort_values("index")
        for g in np.where(c > 1)[0]:
            idx = np.where(inv == g)[0]
            ic.append({"split": sp, "indeksler": idx.tolist(), "image_id": mm.image_id.values[idx].tolist(),
                       "lesion_id": mm.lesion_id.values[idx].tolist(),
                       "etiket": veri[28][sp][1][idx].tolist()})
    lez["split_ici_tam_kopya_gruplari_28px"] = ic
    sonuc["lezyon_sizintisi"] = lez
    SPLITS.mkdir(parents=True, exist_ok=True)
    m.sort_values(["split", "index"]).to_csv(SPLITS / "dermamnist_lezyon_esleme.csv", index=False)
    print("[analiz] lezyon:", json.dumps(lez, ensure_ascii=False), flush=True)

    # ---------------- 28 px yakın-kopya
    X = {sp: veri[28][sp][0] for sp in SPLIT_ADLARI}
    les = {sp: m[m.split == sp].sort_values("index").lesion_id.values for sp in SPLIT_ADLARI}
    yk = {"yontem": "28x28x3 ham piksel RMSE (0-255) en yakın komşu + 64-bit dHash (gri, 9x8) Hamming en yakın komşu"}
    dist = {}
    for sorgu, ref in [("test", "train"), ("val", "train"), ("test", "val"), ("train", "train")]:
        ayni = sorgu == ref
        d, j = nn_rmse(X[sorgu], X[ref], ayni=ayni)
        hd, hj = nn_hamming(dhash(X[sorgu]), dhash(X[ref]), ayni=ayni)
        ayni_lez_nn = les[sorgu] == les[ref][j]
        kayit = {"rmse_nn": ozet(d), "dhash_nn": ozet(hd),
                 "rmse_esik_alti": {str(t): int((d <= t).sum()) for t in [0, 2, 5, 10, 15, 20]},
                 "dhash_esik_alti": {str(t): int((hd <= t).sum()) for t in [0, 2, 4, 6]},
                 "rmse_nn_ayni_lezyon": int(ayni_lez_nn.sum()),
                 "dhash_nn_ayni_lezyon": int((les[sorgu] == les[ref][hj]).sum())}
        if not ayni:
            ref_lez = set(les[ref])
            sizan = np.isin(les[sorgu], list(ref_lez))
            kayit["lezyonu_refte_olan_sorgu"] = int(sizan.sum())
            kayit["rmse_nn_yakalama_orani_(lezyon_sizintisi)"] = round(float(ayni_lez_nn[sizan].mean()), 4)
            kayit["rmse_nn_sizan_medyan"] = round(float(np.median(d[sizan])), 2)
            kayit["rmse_nn_sizmayan_medyan"] = round(float(np.median(d[~sizan])), 2)
            # eşik altı çiftlerin lezyon kesinliği
            kayit["rmse_esik_alti_ayni_lezyon"] = {str(t): int(((d <= t) & ayni_lez_nn).sum()) for t in [5, 10, 15, 20]}
            h_ayni = les[sorgu] == les[ref][hj]
            kayit["dhash_esik_alti_ayni_lezyon"] = {str(t): int(((hd <= t) & h_ayni).sum()) for t in [0, 2, 4, 6]}
        dist[f"{sorgu}->{ref}"] = kayit
        if sorgu == "val":
            sira = np.argsort(d)[:16]
            izgara = np.concatenate([np.concatenate([X["val"][k], X["train"][j[k]]], 0) for k in sira], 1)
            RAPOR.mkdir(parents=True, exist_ok=True)
            Image.fromarray(izgara).resize((izgara.shape[1] * 4, izgara.shape[0] * 4), Image.NEAREST).save(
                RAPOR / "en_yakin_16_val_train.png")
            pd.DataFrame({"val_idx": sira, "train_idx": j[sira], "rmse": d[sira].round(2),
                          "ayni_lezyon": ayni_lez_nn[sira]}).to_csv(RAPOR / "en_yakin_16_val_train.csv", index=False)
        print(f"[analiz] {sorgu}->{ref}: {json.dumps(kayit, ensure_ascii=False)}", flush=True)
    yk["sonuclar"] = dist
    sonuc["yakin_kopya_28px"] = yk
    sonuc["toplam_bayt"] = int(sum(v["bayt"] for v in sonuc["dosyalar"].values()))
    sonuc["meta_dosyalari"] = {f.name: {"bayt": f.stat().st_size, "sha256": dosya_hash(f, "sha256")}
                               for f in sorted(META.glob("*.csv"))}
    sonuc["meta_kaynak"] = {"HAM10000_metadata.csv": DATAVERSE_META_URL, "dermamnist_split_info.csv": SPLITINFO_URL}
    (DATA / "data_manifest.json").write_text(json.dumps(sonuc, indent=1, ensure_ascii=False), encoding="utf-8")
    print("[analiz] yazıldı:", DATA / "data_manifest.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--asama", choices=["indir", "analiz", "hepsi"], default="hepsi")
    ap.add_argument("--boyutlar", type=int, nargs="+", default=BOYUTLAR)
    a = ap.parse_args()
    if a.asama in ("indir", "hepsi"):
        indir(a.boyutlar)
    if a.asama in ("analiz", "hepsi"):
        analiz(a.boyutlar)
