# -*- coding: utf-8 -*-
"""K2 eki — DermaMNIST-C (Abhishek, Jain, Hamarneh, Sci Data 2025) indirme + doğrulama.

Kaynak: Zenodo 10.5281/zenodo.12739457 (v1.1, kavram DOI 10.5281/zenodo.11101337).
Dosya listesi ve resmî MD5 Zenodo REST API'sinden (`/api/records/12739457`) alınır; API yanıtı
`data/dermamnist_c/zenodo_12739457.json` olarak saklanır.

Aşamalar:
  indir  : seçilen dosyaları indirir (HTTP Range ile KALDIĞI YERDEN devam eder; `.part` dosyası),
           MD5'i Zenodo değeriyle doğrular (uyuşmazsa dosyayı siler, çıkış kodu 2), SHA256 hesaplar.
           Kayıt: data/dermamnist_c/indirme_kaydi_c.json (dosya başına bayt, süre, hız, MD5, SHA256).
  analiz : npz anahtarları/şekil/dtype, split sayıları (beklenen 8215/573/1227), sınıf dağılımı,
           DermaMNIST-C.csv ile npz tutarlılığı, resmî MedMNIST split'iyle görüntü (ISIC id) kesişimi,
           HAM10000 lezyon-ayrıklığı, tam-piksel hash çakışması, resmî 28 px ile piksel özdeşliği.
           Çıktı: data/dermamnist_c/dermamnist_c_analiz.json (+ C eşleme CSV'si data/splits/).

Varsayılan dosyalar: DermaMNIST-C.csv + dermamnist_corrected_28.npz. 224 px için `--boyutlar 28 224`.
DermaMNIST-E dosyaları (dermamnist_extended_*.npz, DermaMNIST-E.csv) İNDİRİLMEZ (yalnız listelenir).

Kullanım (WS kökünden; GPU kullanmaz):
  PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe scripts/indir_dermamnist_c.py --asama indir --boyutlar 28
  PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe scripts/indir_dermamnist_c.py --asama indir --boyutlar 224
  PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe scripts/indir_dermamnist_c.py --asama analiz --boyutlar 28
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
DATA = WS / "data"
KOK = DATA / "dermamnist_c"
META = DATA / "meta"
SPLITS = DATA / "splits"
MEDMNIST_ROOT = DATA / "medmnist"
RECORD_ID = 12739457
API = f"https://zenodo.org/api/records/{RECORD_ID}"
UA = {"User-Agent": "K2-veri-hazirlayici/1.0 (akademik; urllib)"}
SPLIT_ADLARI = ["train", "val", "test"]
BEKLENEN = {"train": 8215, "val": 573, "test": 1227}
DX2LBL = {"akiec": 0, "bcc": 1, "bkl": 2, "df": 3, "mel": 4, "nv": 5, "vasc": 6}

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def dosya_hash(p, algo):
    h = hashlib.new(algo)
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def api_kaydi(yenile=False):
    KOK.mkdir(parents=True, exist_ok=True)
    yol = KOK / f"zenodo_{RECORD_ID}.json"
    if yenile or not yol.exists():
        req = urllib.request.Request(API, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            ham = r.read()
        yol.write_bytes(ham)
        log(f"Zenodo API yanıtı kaydedildi: {yol} ({len(ham)} B)")
    return json.loads(yol.read_text(encoding="utf-8"))


def indir_tek(url, hedef, beklenen_bayt, beklenen_md5):
    """Range ile devam eden indirme. Dönüş: dict (bayt, süre, hız, md5, sha256, durum)."""
    if hedef.exists() and hedef.stat().st_size == beklenen_bayt:
        md5 = dosya_hash(hedef, "md5")
        if md5 == beklenen_md5:
            log(f"{hedef.name}: zaten var, MD5 eşleşti -> atlandı")
            return {"durum": "mevcut", "bayt": beklenen_bayt, "md5": md5,
                    "sha256": dosya_hash(hedef, "sha256")}
        log(f"{hedef.name}: MD5 uyuşmadı ({md5}), yeniden indirilecek")
        hedef.unlink()
    part = hedef.with_name(hedef.name + ".part")
    bas = part.stat().st_size if part.exists() else 0
    if bas > beklenen_bayt:
        part.unlink()
        bas = 0
    t0 = time.time()
    alinan = 0
    deneme = 0
    while True:
        bas = part.stat().st_size if part.exists() else 0
        if bas == beklenen_bayt:
            break
        h = dict(UA)
        if bas:
            h["Range"] = f"bytes={bas}-"
        try:
            req = urllib.request.Request(url, headers=h)
            with urllib.request.urlopen(req, timeout=120) as r:
                kod = r.status
                if bas and kod != 206:
                    log(f"sunucu Range desteklemedi (HTTP {kod}); baştan")
                    part.unlink()
                    bas = 0
                mod = "ab" if bas else "wb"
                son_rapor = time.time()
                with open(part, mod) as f:
                    while True:
                        blk = r.read(1 << 16)
                        if not blk:
                            break
                        f.write(blk)
                        alinan += len(blk)
                        if time.time() - son_rapor > 15:
                            simdi = bas + (f.tell() if mod == "wb" else 0)
                            gecen = time.time() - t0
                            hiz = alinan / max(gecen, 1e-9)
                            boy = part.stat().st_size
                            kalan = (beklenen_bayt - boy) / max(hiz, 1)
                            log(f"{hedef.name}: {boy}/{beklenen_bayt} B "
                                f"({100*boy/beklenen_bayt:.1f}%), {hiz/1024:.0f} KB/s, kalan ~{kalan/60:.1f} dk")
                            son_rapor = time.time()
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            deneme += 1
            log(f"{hedef.name}: bağlantı hatası ({e!r}); deneme {deneme}/20, 10 sn sonra devam")
            if deneme >= 20:
                raise
            time.sleep(10)
            continue
        if (part.stat().st_size if part.exists() else 0) < beklenen_bayt:
            deneme += 1
            log(f"{hedef.name}: akış erken bitti; deneme {deneme}/20")
            if deneme >= 20:
                raise RuntimeError("akış tekrar tekrar erken bitti")
            time.sleep(5)
    sure = time.time() - t0
    md5 = dosya_hash(part, "md5")
    if md5 != beklenen_md5:
        part.unlink()
        raise RuntimeError(f"{hedef.name}: MD5 UYUŞMADI ölçülen={md5} resmî={beklenen_md5}; dosya silindi")
    part.replace(hedef)
    sha = dosya_hash(hedef, "sha256")
    hiz = alinan / max(sure, 1e-9)
    log(f"{hedef.name}: tamam {beklenen_bayt} B, {sure:.1f} sn, {hiz/1024:.0f} KB/s, MD5 eşleşti, SHA256={sha}")
    return {"durum": "indirildi", "bayt": beklenen_bayt, "bu_oturumda_alinan_bayt": alinan,
            "sure_sn": round(sure, 1), "ort_hiz_KBps": round(hiz / 1024, 1), "md5": md5, "sha256": sha}


def asama_indir(boyutlar, yenile_api):
    kayit = api_kaydi(yenile_api)
    dosyalar = {f["key"]: f for f in kayit["files"]}
    log("Zenodo dosya listesi:")
    for k, f in dosyalar.items():
        log(f"  {k}  {f['size']} B  {f['checksum']}")
    secim = ["DermaMNIST-C.csv"] + [f"dermamnist_corrected_{b}.npz" for b in boyutlar]
    kayit_yolu = KOK / "indirme_kaydi_c.json"
    ozet = json.loads(kayit_yolu.read_text(encoding="utf-8")) if kayit_yolu.exists() else {}
    ozet.setdefault("kaynak", {"zenodo_record": RECORD_ID, "doi": kayit.get("doi"),
                               "version": kayit["metadata"].get("version"),
                               "lisans_kayit": kayit["metadata"].get("license")})
    ozet.setdefault("dosyalar", {})
    for k in secim:
        f = dosyalar[k]
        md5_resmi = f["checksum"].split(":", 1)[1]
        url = f["links"]["self"]
        log(f"indiriliyor: {k} ({f['size']} B) <- {url}")
        t_bas = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        sonuc = indir_tek(url, KOK / k, f["size"], md5_resmi)
        sonuc.update({"url": url, "md5_resmi": md5_resmi, "md5_eslesti": sonuc["md5"] == md5_resmi,
                      "baslangic": t_bas})
        ozet["dosyalar"][k] = sonuc
        kayit_yolu.write_text(json.dumps(ozet, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"kayıt: {kayit_yolu}")


# ----------------------------------------------------------------------------- analiz
def asama_analiz(boyutlar):
    import numpy as np
    import pandas as pd

    sonuc = {"tarih": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")}
    csv_c = pd.read_csv(KOK / "DermaMNIST-C.csv")
    sonuc["csv_sutunlar"] = list(csv_c.columns)
    sonuc["csv_satir"] = int(len(csv_c))
    log("DermaMNIST-C.csv sütunlar:", list(csv_c.columns), "satır:", len(csv_c))
    ham = pd.read_csv(META / "HAM10000_metadata.csv")
    resmi = pd.read_csv(META / "dermamnist_split_info.csv")  # split,index,image_id (resmî MedMNIST)
    id2les = dict(zip(ham.image_id, ham.lesion_id))
    id2dx = dict(zip(ham.image_id, ham.dx))
    id2resmi = dict(zip(resmi.image_id, resmi.split))
    id2resmi_idx = dict(zip(resmi.image_id, resmi["index"]))

    for b in boyutlar:
        yol = KOK / f"dermamnist_corrected_{b}.npz"
        if not yol.exists():
            log(f"{yol.name} yok, atlandı")
            continue
        with np.load(yol, allow_pickle=False) as dz:  # bir kez belleğe al (NpzFile tekrar açmasın)
            d = {k: dz[k] for k in dz.files}
        r = {"anahtarlar": {k: {"shape": list(d[k].shape), "dtype": str(d[k].dtype)} for k in d}}
        log(f"{yol.name} anahtarlar:", r["anahtarlar"])
        r["split_sayilari"] = {s: int(d[f"{s}_images"].shape[0]) for s in SPLIT_ADLARI}
        r["split_beklenen_eslesti"] = r["split_sayilari"] == BEKLENEN
        r["sinif_sayilari"] = {}
        for s in SPLIT_ADLARI:
            y = d[f"{s}_labels"].reshape(-1)
            r["sinif_sayilari"][s] = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
        for s in SPLIT_ADLARI:
            x = d[f"{s}_images"]
            r.setdefault("piksel", {})[s] = {"min": int(x.min()), "max": int(x.max())}
        # tam-piksel hash çakışması
        hs = {s: [hashlib.sha1(np.ascontiguousarray(im).tobytes()).hexdigest() for im in d[f"{s}_images"]]
              for s in SPLIT_ADLARI}
        r["tam_piksel_cakisma"] = {
            "train∩val": len(set(hs["train"]) & set(hs["val"])),
            "train∩test": len(set(hs["train"]) & set(hs["test"])),
            "val∩test": len(set(hs["val"]) & set(hs["test"])),
            "split_ici_kopya": {s: len(hs[s]) - len(set(hs[s])) for s in SPLIT_ADLARI},
        }
        sonuc[f"npz_{b}"] = r
        if b != 28 and (KOK / "dermamnist_corrected_28.npz").exists():
            r["boyutlar_arasi_sira"] = boyut_sira_denetimi(d, b)
        if b == 28:
            sonuc["csv_npz_tutarlilik"] = csv_npz_tutarlilik(csv_c, d, id2dx)
            # resmî 28 px ile piksel özdeşliği (aynı image_id)
            sonuc["resmi28_piksel"] = resmi_piksel_karsilastir(csv_c, d)
        del d

    # --- CSV tabanlı kimlik analizi
    sonuc["kimlik"] = kimlik_analizi(csv_c, id2les, id2dx, id2resmi, id2resmi_idx)
    cikti = KOK / "dermamnist_c_analiz.json"
    cikti.write_text(json.dumps(sonuc, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"analiz yazıldı: {cikti}")


def boyut_sira_denetimi(d, b, n=300, seed=0):
    """Büyük boyut -> 28 px bicubic küçültme; aynı indeksteki C-28 ile RMSE ve en yakın komşu eşleşmesi.
    Yalnız train ve val (test açılmaz)."""
    import numpy as np
    from PIL import Image
    with np.load(KOK / "dermamnist_corrected_28.npz") as z:
        k28 = {s: z[f"{s}_images"] for s in ("train", "val")}
    out = {}
    for s in ("train", "val"):
        X, Y = d[f"{s}_images"], k28[s]
        if len(X) != len(Y):
            out[s] = f"uzunluk farklı {len(X)} != {len(Y)}"
            continue
        sel = np.random.default_rng(seed).choice(len(X), size=min(n, len(X)), replace=False)
        P = Y.reshape(len(Y), -1).astype(np.float32)
        Pn = (P ** 2).sum(1)
        ayni, rm_ayni, rm_nn = 0, [], []
        for j in sel:
            k = np.asarray(Image.fromarray(X[j]).resize((28, 28), Image.BICUBIC)).reshape(-1).astype(np.float32)
            dist = np.maximum(Pn - 2 * P @ k + (k ** 2).sum(), 0)
            nn = int(np.argmin(dist))
            ayni += int(nn == j)
            rm_ayni.append(float(np.sqrt(dist[j] / k.size)))
            rm_nn.append(float(np.sqrt(dist[nn] / k.size)))
        out[s] = {"orneklem": int(len(sel)), "nn_ayni_indeks": ayni,
                  "rmse_ayni_indeks_medyan": float(np.median(rm_ayni)), "rmse_nn_medyan": float(np.median(rm_nn))}
    return out


def _csv_kolonlari(csv_c):
    cols = {c.lower(): c for c in csv_c.columns}
    def bul(*adaylar):
        for a in adaylar:
            for lc, c in cols.items():
                if lc == a:
                    return c
        for a in adaylar:
            for lc, c in cols.items():
                if a in lc:
                    return c
        return None
    return {"split": bul("split", "partition", "set"), "index": bul("index", "idx"),
            "image_id": bul("image_id", "image", "isic_id", "filename"),
            "lesion_id": bul("lesion_id", "lesion"), "label": bul("label", "dx", "diagnosis")}


def _split_norm(s):
    s = str(s).lower()
    return {"valid": "val", "validation": "val", "training": "train", "testing": "test"}.get(s, s)


def csv_npz_tutarlilik(csv_c, d, id2dx):
    import numpy as np
    k = _csv_kolonlari(csv_c)
    out = {"kolon_eslemesi": k}
    if k["split"] is None:
        out["not"] = "CSV'de split sütunu bulunamadı"
        return out
    sp = csv_c[k["split"]].map(_split_norm)
    out["csv_split_sayilari"] = {s: int((sp == s).sum()) for s in SPLIT_ADLARI}
    if k["image_id"] is not None:
        for s in SPLIT_ADLARI:
            alt = csv_c[sp == s]
            if k["index"] is not None:
                alt = alt.sort_values(k["index"])
            y = d[f"{s}_labels"].reshape(-1)
            if len(alt) != len(y):
                out.setdefault("uyumsuz", {})[s] = f"csv {len(alt)} != npz {len(y)}"
                continue
            dx_lbl = np.array([DX2LBL.get(id2dx.get(i, ""), -1) for i in alt[k["image_id"]]])
            out.setdefault("etiket_dx_uyumu", {})[s] = f"{int((dx_lbl == y).sum())}/{len(y)}"
    return out


def resmi_piksel_karsilastir(csv_c, d):
    import numpy as np
    k = _csv_kolonlari(csv_c)
    rp = MEDMNIST_ROOT / "dermamnist.npz"
    if not rp.exists() or k["image_id"] is None or k["split"] is None:
        return {"not": "resmî 28 px veya CSV kimlik sütunu yok"}
    import pandas as pd
    resmi = pd.read_csv(META / "dermamnist_split_info.csv")
    with np.load(rp) as Rz:  # NpzFile her erişimde diziyi yeniden açar -> bir kez belleğe al
        R = {f"{s}_images": Rz[f"{s}_images"] for s in SPLIT_ADLARI}
    resmi_img = {}
    id2ri = {iid: (s, int(i)) for s, i, iid in zip(resmi.split, resmi["index"], resmi.image_id)}
    sp = csv_c[k["split"]].map(_split_norm)
    ozdes = 0
    toplam = 0
    rmse = []
    for s in SPLIT_ADLARI:
        alt = csv_c[sp == s]
        if k["index"] is not None:
            alt = alt.sort_values(k["index"])
        X = d[f"{s}_images"]
        if len(alt) != len(X):
            continue
        for j, iid in enumerate(alt[k["image_id"]]):
            if iid not in id2ri:
                continue
            rs, ri = id2ri[iid]
            a = X[j].astype(np.int32)
            bb = R[f"{rs}_images"][ri].astype(np.int32)
            toplam += 1
            if a.shape == bb.shape and np.array_equal(a, bb):
                ozdes += 1
            elif a.shape == bb.shape:
                rmse.append(float(np.sqrt(((a - bb) ** 2).mean())))
    del resmi_img
    out = {"karsilastirilan": toplam, "piksel_ozdes": ozdes}
    if rmse:
        out["farkli_olanlarda_rmse_medyan"] = float(np.median(rmse))
        out["farkli_olanlarda_rmse_max"] = float(np.max(rmse))
    return out


def kimlik_analizi(csv_c, id2les, id2dx, id2resmi, id2resmi_idx):
    import pandas as pd
    k = _csv_kolonlari(csv_c)
    out = {"kolon_eslemesi": k}
    if k["split"] is None or k["image_id"] is None:
        out["not"] = "CSV'de split/image_id sütunu yok"
        return out
    sp = csv_c[k["split"]].map(_split_norm)
    ids = csv_c[k["image_id"]].astype(str)
    out["benzersiz_image_id"] = int(ids.nunique())
    out["HAM_eslesmeyen_image_id"] = int((~ids.isin(id2les.keys())).sum())
    out["resmi_spliti_olmayan_image_id"] = int((~ids.isin(id2resmi.keys())).sum())
    out["resmi_toplam_C_de_olmayan"] = int(len(set(id2resmi) - set(ids)))
    if k["lesion_id"] is not None:
        les = csv_c[k["lesion_id"]].astype(str)
        out["csv_lesion_HAM_uyumu"] = f"{int((les.values == ids.map(id2les).astype(str).values).sum())}/{len(ids)}"
    les = ids.map(id2les)
    L = {s: set(les[sp == s].dropna()) for s in SPLIT_ADLARI}
    I = {s: set(ids[sp == s]) for s in SPLIT_ADLARI}
    out["lezyon_sayisi"] = {s: len(L[s]) for s in SPLIT_ADLARI}
    out["lezyon_cakisma"] = {"train∩val": len(L["train"] & L["val"]), "train∩test": len(L["train"] & L["test"]),
                             "val∩test": len(L["val"] & L["test"]),
                             "uc_bolum": len(L["train"] & L["val"] & L["test"])}
    out["image_id_cakisma"] = {"train∩val": len(I["train"] & I["val"]), "train∩test": len(I["train"] & I["test"]),
                               "val∩test": len(I["val"] & I["test"])}
    # resmî split -> C split geçiş matrisi
    tab = pd.crosstab(ids.map(id2resmi).fillna("yok"), sp)
    out["resmi_to_C_gecis"] = {r: {c: int(tab.loc[r, c]) for c in tab.columns} for r in tab.index}
    # C test'inin resmî train/val ile görüntü kesişimi (aynı ISIC kimliği)
    C_test = I["test"]
    out["C_test_resmi_kaynak"] = {s: int(sum(id2resmi.get(i) == s for i in C_test)) for s in SPLIT_ADLARI}
    out["C_train_resmi_kaynak"] = {s: int(sum(id2resmi.get(i) == s for i in I["train"])) for s in SPLIT_ADLARI}
    out["C_val_resmi_kaynak"] = {s: int(sum(id2resmi.get(i) == s for i in I["val"])) for s in SPLIT_ADLARI}
    # dx dağılımı (HAM metadata'dan)
    out["dx_dagilimi"] = {s: ids[sp == s].map(id2dx).value_counts().to_dict() for s in SPLIT_ADLARI}
    out["dx_dagilimi"] = {s: {kk: int(v) for kk, v in dd.items()} for s, dd in out["dx_dagilimi"].items()}
    # eşleme CSV'si (yeni split DEĞİL; DermaMNIST-C resmî split'inin kaydı)
    SPLITS.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"split_C": sp.values, "image_id": ids.values, "lesion_id": les.values,
                       "dx": ids.map(id2dx).values, "split_resmi": ids.map(id2resmi).values,
                       "index_resmi": ids.map(id2resmi_idx).values})
    if k["index"] is not None:
        df.insert(1, "index_C", csv_c[k["index"]].values)
    df.to_csv(SPLITS / "dermamnist_c_esleme.csv", index=False)
    out["esleme_csv"] = "data/splits/dermamnist_c_esleme.csv"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asama", choices=["indir", "analiz", "hepsi"], default="hepsi")
    ap.add_argument("--boyutlar", type=int, nargs="+", default=[28], choices=[28, 224])
    ap.add_argument("--api-yenile", action="store_true")
    a = ap.parse_args()
    if a.asama in ("indir", "hepsi"):
        asama_indir(a.boyutlar, a.api_yenile)
    if a.asama in ("analiz", "hepsi"):
        asama_analiz(a.boyutlar)


if __name__ == "__main__":
    main()
