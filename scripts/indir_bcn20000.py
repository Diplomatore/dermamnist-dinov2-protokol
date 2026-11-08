# -*- coding: utf-8 -*-
"""E2 dış doğrulama verisi — BCN20000 (Hernández-Pérez vd., Sci Data 11, 641, 2024) indirme + denetim.

Kalıcı kaynak: figshare 10.6084/m9.figshare.24140028 (v1). Dosya listesi ve resmî MD5 figshare
REST API'sinden (`https://api.figshare.com/v2/articles/24140028`) alınır, yanıt saklanır.
Çapraz denetim kaynakları: ISIC Archive API v2 (koleksiyon 249 = BCN20000, 212 = HAM10000) ve
ISIC 2019 Challenge eğitim meta verisi (HAM10000 + BCN_20000 + MSK; yalnız kimlik/çakışma için).

Aşamalar (WS kökünden; GPU kullanmaz; tüm çıktılar data/bcn20000/ altında):
  meta        : figshare API + bcn_20k_{train,test}.csv (MD5 doğrulamalı) + ISIC 2019 eğitim CSV'leri
                + ISIC koleksiyon 249/212 meta verisi (sayfalı API; ~290 istek).
  zipliste    : train/test zip'lerinin merkezi dizini HTTP Range ile okunur (zip İNDİRİLMEZ) ->
                dosya sayısı, toplam boyut, CRC listesi (meta/zip_icerik_{train,test}.csv).
  ornek       : train zip'inden HTTP Range ile seed'li, sınıf-tabakalı ≤N görüntü çıkarır (CRC doğrulamalı)
                + kalibrasyon için ≤K HAM10000 görüntüsü (yalnız DermaMNIST TRAIN bölümünden) ISIC API'den.
  kalibrasyon : HAM orijinalinden aday ön-işlemelerle 28/224 px üretip resmî dermamnist{,_224}.npz
                train görüntüsüyle RMSE kıyası -> MedMNIST ön-işlemesinin ölçülerek belirlenmesi.
  analiz      : sınıf dağılımı, lezyon başına görüntü, tarih aralığı, HAM↔BCN çakışma denetimi
                (ISIC id, lesion_id, kaynak merkez, ISIC 2019 lezyon öneki, piksel-NN 28 px), örnek
                görüntü özellikleri. Çıktı: data/bcn20000/bcn20000_analiz.json
  onisleme    : mevcut görüntülerden (örnek ya da tam zip) 28/224 px npz üretir (iki geometri seçeneği).
  indir       : BCN_20k_train.zip tam indirme (Range ile devam, MD5 doğrulama). UZUN (≈1,7 sa @1 MB/s) ->
                yalnız kuyruk/Colab üzerinden.
  nn_tam      : tam BCN train 28 px -> DermaMNIST 10.015 görüntü piksel-NN (RMSE ≤5 = kopya eşiği, ana manifest).

Kullanım:
  PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I scripts/indir_bcn20000.py --asama meta
  PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I scripts/indir_bcn20000.py --asama zipliste
  PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I scripts/indir_bcn20000.py --asama ornek --n-ornek 40 --n-ham 8
  PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I scripts/indir_bcn20000.py --asama kalibrasyon
  PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I scripts/indir_bcn20000.py --asama analiz
  PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I scripts/indir_bcn20000.py --asama onisleme --kaynak ornek
  PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I scripts/indir_bcn20000.py --asama indir      # uzun
  PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I scripts/indir_bcn20000.py --asama onisleme --kaynak tam
  PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I scripts/indir_bcn20000.py --asama nn_tam   # HAM piksel-NN
"""
import argparse
import csv
import hashlib
import io
import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[1]
DATA = WS / "data"
KOK = DATA / "bcn20000"
META = KOK / "meta"
ORNEK = KOK / "ornek"
HAMKAL = KOK / "kalibrasyon_ham"
ZIPDIR = KOK / "zip"  # tam zip buraya (BCN_20k_train.zip)
MEDMNIST_ROOT = DATA / "medmnist"
FIG_ID = 24140028
FIG_API = f"https://api.figshare.com/v2/articles/{FIG_ID}"
ISIC_API = "https://api.isic-archive.com/api/v2"
ISIC2019 = "https://isic-archive.s3.amazonaws.com/challenges/2019/"
UA = {"User-Agent": "K2-veri-hazirlayici/1.0 (akademik; urllib)"}
BCN_SINIFLAR = ["NV", "MEL", "BCC", "BKL", "AK", "SCC", "DF", "VASC"]
DM_ADLAR = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

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


def http_get(url, headers=None, timeout=120, deneme=5):
    h = dict(UA)
    if headers:
        h.update(headers)
    for k in range(deneme):
        try:
            req = urllib.request.Request(url, headers=h)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if k == deneme - 1:
                raise
            log(f"  yeniden deneme {k + 1}: {e}")
            time.sleep(2 * (k + 1))


def json_yaz(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def kayit_guncelle(anahtar, deger):
    p = KOK / "indirme_kaydi_bcn.json"
    d = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    d[anahtar] = deger
    json_yaz(p, d)


# ───────────────────────── meta ─────────────────────────
def figshare_kaydi(yenile=False):
    p = META / f"figshare_{FIG_ID}.json"
    if yenile or not p.exists():
        _, ham = http_get(FIG_API)
        META.mkdir(parents=True, exist_ok=True)
        p.write_bytes(ham)
        log(f"figshare API yanıtı kaydedildi: {p} ({len(ham)} B)")
    return json.loads(p.read_text(encoding="utf-8"))


def fig_dosya(rec, ad):
    for f in rec["files"]:
        if f["name"] == ad:
            return f
    raise KeyError(ad)


def isic_koleksiyon(cid, hedef):
    """ISIC API v2 sayfalı arama; ham sonuçlar jsonl olarak saklanır."""
    if hedef.exists():
        n = sum(1 for _ in open(hedef, encoding="utf-8"))
        log(f"ISIC koleksiyon {cid}: mevcut ({n} satır) -> atlandı")
        return n
    url = f"{ISIC_API}/images/search/?collections={cid}&limit=100"
    tmp = hedef.with_suffix(".part")
    n, sayfa, t0 = 0, 0, time.time()
    with open(tmp, "w", encoding="utf-8") as f:
        while url:
            _, ham = http_get(url, timeout=60)
            d = json.loads(ham.decode("utf-8"))
            for r in d["results"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
            sayfa += 1
            url = d.get("next")
            if sayfa % 25 == 0:
                log(f"  koleksiyon {cid}: sayfa {sayfa}, {n}/{d.get('count')} ({time.time() - t0:.0f} sn)")
    tmp.replace(hedef)
    log(f"ISIC koleksiyon {cid}: {n} kayıt, {sayfa} sayfa, {time.time() - t0:.1f} sn")
    return n


def asama_meta(a):
    META.mkdir(parents=True, exist_ok=True)
    rec = figshare_kaydi(yenile=True)
    sonuc = {"zaman": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
             "figshare": {"doi": rec["doi"], "lisans": rec["license"]["name"], "boyut_toplam": rec["size"],
                          "yayim": rec["published_date"], "dosyalar": []}}
    for ad in ["bcn_20k_train.csv", "bcn_20k_test.csv"]:
        f = fig_dosya(rec, ad)
        p = META / ad
        if not (p.exists() and dosya_hash(p, "md5") == f["computed_md5"]):
            _, ham = http_get(f["download_url"])
            p.write_bytes(ham)
        md5 = dosya_hash(p, "md5")
        ok = md5 == f["computed_md5"]
        log(f"{ad}: {p.stat().st_size} B, MD5 {md5} {'eşleşti' if ok else 'UYUŞMADI'}")
        if not ok:
            p.unlink()
            raise SystemExit(2)
        sonuc["figshare"]["dosyalar"].append({"ad": ad, "bayt": p.stat().st_size, "md5": md5, "md5_resmi": f["computed_md5"],
                                               "sha256": dosya_hash(p, "sha256")})
    for f in rec["files"]:
        if f["name"].endswith(".zip"):
            sonuc["figshare"]["dosyalar"].append({"ad": f["name"], "bayt_resmi": f["size"], "md5_resmi": f["computed_md5"],
                                                   "durum": "indirilmedi (meta aşaması)"})
    sonuc["isic2019"] = []
    for ad in ["ISIC_2019_Training_Metadata.csv", "ISIC_2019_Training_GroundTruth.csv"]:
        p = META / ad
        if not p.exists():
            _, ham = http_get(ISIC2019 + ad)
            if ham[:20].lstrip().lower().startswith(b"<!doctype"):
                raise SystemExit(f"{ad}: HTML yönlendirme döndü")
            p.write_bytes(ham)
        sonuc["isic2019"].append({"ad": ad, "url": ISIC2019 + ad, "bayt": p.stat().st_size, "sha256": dosya_hash(p, "sha256")})
    # Cassidy vd. 2022 kopya listesi (GitHub, sabit commit)
    CAS = "https://raw.githubusercontent.com/mmu-dermatology-research/isic_duplicate_removal_strategy/b785b75fc1b1440d7be7b17d2cb2348de6c76102/"
    sonuc["cassidy"] = []
    for ad, uzak in [("cassidy_06_all_train_duplicates.txt", "file_lists/06%20-%20all_train_duplicates_deleted_(all%20but%20newest).txt"),
                     ("cassidy_README.md", "README.md")]:
        p = META / ad
        if not p.exists():
            _, ham = http_get(CAS + uzak)
            p.write_bytes(ham)
        sonuc["cassidy"].append({"ad": ad, "url": CAS + uzak, "bayt": p.stat().st_size, "sha256": dosya_hash(p, "sha256")})
    sonuc["isic_koleksiyon"] = {}
    for cid in a.koleksiyonlar:
        _, ham = http_get(f"{ISIC_API}/collections/{cid}/")
        bilgi = json.loads(ham.decode("utf-8"))
        hedef = META / f"isic_koleksiyon_{cid}.jsonl"
        n = isic_koleksiyon(cid, hedef)
        sonuc["isic_koleksiyon"][str(cid)] = {"ad": bilgi["name"], "kayit": n, "public": bilgi["public"],
                                              "sha256_jsonl": dosya_hash(hedef, "sha256")}
    kayit_guncelle("meta", sonuc)
    log("meta aşaması tamam")


# ───────────────────────── HTTP Range ile zip okuma ─────────────────────────
class HttpRangeDosya(io.RawIOBase):
    """figshare ndownloader -> (10 sn geçerli imzalı) R2 yönlendirmesi; her okuma yeni Range isteği."""

    def __init__(self, url, boyut):
        self.url, self.boyut, self.poz, self.istek, self.bayt = url, boyut, 0, 0, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.poz

    def seek(self, ofs, whence=0):
        self.poz = ofs if whence == 0 else (self.poz + ofs if whence == 1 else self.boyut + ofs)
        return self.poz

    def readinto(self, b):
        if self.poz >= self.boyut:
            return 0
        son = min(self.poz + len(b), self.boyut) - 1
        kod, ham = http_get(self.url, headers={"Range": f"bytes={self.poz}-{son}"})
        if kod != 206:
            raise IOError(f"Range desteklenmedi (HTTP {kod})")
        n = len(ham)
        b[:n] = ham
        self.poz += n
        self.istek += 1
        self.bayt += n
        return n


def uzak_zip(rec, ad, tampon=1 << 20):
    f = fig_dosya(rec, ad)
    ham = HttpRangeDosya(f["download_url"], f["size"])
    return zipfile.ZipFile(io.BufferedReader(ham, buffer_size=tampon)), ham


def asama_zipliste(a):
    rec = figshare_kaydi()
    ozet = {}
    for ad in ["BCN_20k_train.zip", "BCN_20k_test.zip"]:
        t0 = time.time()
        z, ham = uzak_zip(rec, ad)
        bilgiler = z.infolist()
        hedef = META / f"zip_icerik_{ad.split('_')[-1].replace('.zip', '')}.csv"
        with open(hedef, "w", newline="", encoding="utf-8") as fo:
            w = csv.writer(fo)
            w.writerow(["ad", "boyut", "sikistirilmis", "crc32", "tarih"])
            for i in bilgiler:
                w.writerow([i.filename, i.file_size, i.compress_size, f"{i.CRC:08x}", "%04d-%02d-%02d" % i.date_time[:3]])
        dosyalar = [i for i in bilgiler if not i.is_dir()]
        uz = Counter(Path(i.filename).suffix.lower() for i in dosyalar)
        ozet[ad] = {"giris": len(bilgiler), "dosya": len(dosyalar), "klasor": len(bilgiler) - len(dosyalar),
                    "uzanti": dict(uz), "acik_boyut_toplam": sum(i.file_size for i in dosyalar),
                    "sikistirilmis_toplam": sum(i.compress_size for i in dosyalar),
                    "zip_boyut": fig_dosya(rec, ad)["size"], "range_istek": ham.istek, "range_bayt": ham.bayt,
                    "sure_sn": round(time.time() - t0, 1), "liste": str(hedef.relative_to(WS))}
        log(f"{ad}: {ozet[ad]}")
    kayit_guncelle("zipliste", ozet)


# ───────────────────────── örnek ─────────────────────────
def csv_oku(p):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def asama_ornek(a):
    rec = figshare_kaydi()
    tr = csv_oku(META / "bcn_20k_train.csv")
    rng = random.Random(a.seed)
    sinif = defaultdict(list)
    for r in tr:
        sinif[r["diagnosis"]].append(r["bcn_filename"])
    k = max(1, a.n_ornek // len(sinif))
    secim = []
    for s in sorted(sinif):
        secim += [(s, x) for x in rng.sample(sorted(sinif[s]), min(k, len(sinif[s])))]
    secim = secim[: a.n_ornek]
    ORNEK.mkdir(parents=True, exist_ok=True)
    (ORNEK / "orijinal").mkdir(exist_ok=True)
    z, ham = uzak_zip(rec, "BCN_20k_train.zip", tampon=1 << 21)
    adlar = {Path(i.filename).name: i for i in z.infolist() if not i.is_dir()}
    t0 = time.time()
    kayit = []
    for s, fn in secim:
        info = adlar[fn]
        hedef = ORNEK / "orijinal" / fn
        if not hedef.exists():
            veri = z.read(info)  # zipfile CRC32'yi doğrular
            hedef.write_bytes(veri)
        kayit.append({"bcn_filename": fn, "diagnosis": s, "bayt": hedef.stat().st_size, "crc32": f"{info.CRC:08x}",
                      "sha256": dosya_hash(hedef, "sha256")})
    log(f"BCN örnek: {len(kayit)} görüntü, {time.time() - t0:.1f} sn, Range istek {ham.istek}, {ham.bayt} B")
    json_yaz(ORNEK / "ornek_listesi.json", {"seed": a.seed, "sinif_basi": k, "kaynak": "BCN_20k_train.zip (HTTP Range)",
                                            "goruntuler": kayit})
    # HAM kalibrasyon: yalnız DermaMNIST TRAIN bölümünden, seed'li
    es = csv_oku(DATA / "splits" / "dermamnist_lezyon_esleme.csv")
    trn = [r for r in es if r["split"] == "train"]
    hsec = random.Random(a.seed).sample(trn, a.n_ham)
    HAMKAL.mkdir(parents=True, exist_ok=True)
    hk = []
    for r in hsec:
        iid = r["image_id"]
        hedef = HAMKAL / f"{iid}.jpg"
        if not hedef.exists():
            _, j = http_get(f"{ISIC_API}/images/{iid}/")
            meta = json.loads(j.decode("utf-8"))
            _, img = http_get(meta["files"]["full"]["url"])
            hedef.write_bytes(img)
            (HAMKAL / f"{iid}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        hk.append({"image_id": iid, "split": "train", "index": int(r["index"]), "dx": r["dx"], "bayt": hedef.stat().st_size})
    json_yaz(HAMKAL / "kalibrasyon_listesi.json", {"seed": a.seed, "kaynak": "ISIC API v2 /images/<id>/ files.full", "goruntuler": hk})
    log(f"HAM kalibrasyon: {len(hk)} görüntü (yalnız DermaMNIST train)")


# ───────────────────────── ön-işleme adayları ─────────────────────────
def adaylar():
    from PIL import Image
    from skimage.transform import resize as sk_resize

    def pil(yontem):
        def f(img, S):
            return np.asarray(img.resize((S, S), yontem))
        return f

    def ski(aa, order):
        def f(img, S):
            x = np.asarray(img).astype(np.float64) / 255.0
            y = sk_resize(x, (S, S), order=order, anti_aliasing=aa, mode="reflect")
            return np.clip(np.round(y * 255), 0, 255).astype(np.uint8)
        return f

    return {"PIL_NEAREST": pil(Image.NEAREST), "PIL_BILINEAR": pil(Image.BILINEAR), "PIL_BICUBIC": pil(Image.BICUBIC),
            "PIL_LANCZOS": pil(Image.LANCZOS), "PIL_BOX": pil(Image.BOX),
            "skimage_o1_aa": ski(True, 1), "skimage_o3_aa": ski(True, 3), "skimage_o1_noaa": ski(False, 1)}


def geometri(img, tur):
    """'gerdir': doğrudan kare boyuta (en-boy bozulur); 'merkez_kare': kısa kenara merkez kırp;
    'merkez_4x3': 4:3 merkez kırpma (HAM 600x450 geometrisini taklit) sonra gerdir."""
    w, h = img.size
    if tur == "gerdir":
        return img
    if tur == "merkez_kare":
        s = min(w, h)
        l, t = (w - s) // 2, (h - s) // 2
        return img.crop((l, t, l + s, t + s))
    if tur == "merkez_4x3":
        if w * 3 >= h * 4:
            nw, nh = h * 4 // 3, h
        else:
            nw, nh = w, w * 3 // 4
        l, t = (w - nw) // 2, (h - nh) // 2
        return img.crop((l, t, l + nw, t + nh))
    raise ValueError(tur)


def asama_kalibrasyon(a):
    from PIL import Image
    kl = json.loads((HAMKAL / "kalibrasyon_listesi.json").read_text(encoding="utf-8"))
    idx = [g["index"] for g in kl["goruntuler"]]
    ref = {}
    with np.load(MEDMNIST_ROOT / "dermamnist.npz") as d:
        ref[28] = d["train_images"][idx]
    p224 = MEDMNIST_ROOT / "dermamnist_224.npz"
    if p224.exists():
        with np.load(p224) as d:
            X = d["train_images"]
            ref[224] = X[idx]
            del X
    ad = adaylar()
    sonuc = {"n": len(idx), "boyut": {}, "orijinal_boyutlar": []}
    imgs = []
    for g in kl["goruntuler"]:
        im = Image.open(HAMKAL / f"{g['image_id']}.jpg").convert("RGB")
        imgs.append(im)
        sonuc["orijinal_boyutlar"].append(list(im.size))
    for S, R in ref.items():
        tablo = {}
        for geo in ["gerdir", "merkez_kare"]:
            for k, f in ad.items():
                rm = []
                for im, r in zip(imgs, R):
                    y = f(geometri(im, geo), S).astype(np.float64)
                    rm.append(float(np.sqrt(np.mean((y - r.astype(np.float64)) ** 2))))
                tablo[f"{geo}|{k}"] = {"rmse_ort": round(float(np.mean(rm)), 3), "rmse_maks": round(float(np.max(rm)), 3)}
        en = min(tablo, key=lambda t: tablo[t]["rmse_ort"])
        sonuc["boyut"][str(S)] = {"adaylar": tablo, "en_iyi": en, "en_iyi_rmse": tablo[en]}
        log(f"{S} px en iyi: {en} {tablo[en]}")
        for t, v in sorted(tablo.items(), key=lambda kv: kv[1]["rmse_ort"])[:6]:
            log(f"   {t:35s} {v}")
    # yatay/dikey yansıtma denetimi (sıra/yön kontrolü)
    json_yaz(KOK / "kalibrasyon_onisleme.json", sonuc)


# ───────────────────────── analiz ─────────────────────────
def isic_oku(cid):
    p = META / f"isic_koleksiyon_{cid}.jsonl"
    return [json.loads(s) for s in open(p, encoding="utf-8")] if p.exists() else []


def klin(r, k):
    return (r.get("metadata", {}).get("clinical", {}) or {}).get(k)


def asama_analiz(a):
    out = {"zaman": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")}
    tr = csv_oku(META / "bcn_20k_train.csv")
    te = csv_oku(META / "bcn_20k_test.csv")
    out["csv"] = {"train_satir": len(tr), "test_satir": len(te),
                  "train_split_degerleri": dict(Counter(r["split"] for r in tr)),
                  "test_split_degerleri": dict(Counter(r["split"] for r in te)),
                  "test_tani_dolu": sum(1 for r in te if r["diagnosis"].strip()),
                  "train_tani_bos": sum(1 for r in tr if not r["diagnosis"].strip()),
                  "dosya_adi_tekil": len({r["bcn_filename"] for r in tr + te}),
                  "train_test_dosya_kesisim": len({r["bcn_filename"] for r in tr} & {r["bcn_filename"] for r in te})}
    # sınıf dağılımı (görüntü ve lezyon)
    sg = Counter(r["diagnosis"] for r in tr)
    lez_sinif = defaultdict(set)
    for r in tr:
        lez_sinif[r["lesion_id"]].add(r["diagnosis"])
    sl = Counter(next(iter(v)) for v in lez_sinif.values() if len(v) == 1)
    out["sinif"] = {"goruntu": dict(sg.most_common()), "lezyon_tek_tanili": dict(sl.most_common()),
                    "lezyon_coklu_tanili": sum(1 for v in lez_sinif.values() if len(v) > 1),
                    "dengesizlik_max_min": round(max(sg.values()) / min(sg.values()), 2)}
    lg = Counter(r["lesion_id"] for r in tr)
    out["lezyon"] = {"train_lezyon": len(lg), "goruntu_per_lezyon_hist": dict(sorted(Counter(lg.values()).items())),
                     "maks": max(lg.values()), "ort": round(len(tr) / len(lg), 3)}
    lt = Counter(r["lesion_id"] for r in te)
    out["lezyon"]["test_lezyon"] = len(lt)
    out["lezyon"]["test_goruntu_per_lezyon_hist"] = dict(sorted(Counter(lt.values()).items()))
    out["lezyon"]["train_test_lesion_id_kesisim_ham"] = len(set(lg) & set(lt))
    out["lezyon"]["lesion_id_bicim"] = {"train_ornek": tr[0]["lesion_id"], "test_ornek": te[0]["lesion_id"]}
    norm = lambda s: s.replace("BCN_", "").replace("BCN", "").lstrip("0")
    out["lezyon"]["train_test_lesion_id_kesisim_rakam_normalize"] = len({norm(x) for x in lg} & {norm(x) for x in lt})
    tarih = sorted(r["capture_date"] for r in tr if r["capture_date"])
    tarih_te = sorted(r["capture_date"] for r in te if r["capture_date"])
    out["tarih"] = {"train": [tarih[0], tarih[-1]], "test": [tarih_te[0], tarih_te[-1]],
                    "train_bos": len(tr) - len(tarih), "test_bos": len(te) - len(tarih_te)}
    out["demografi_train"] = {"cinsiyet": dict(Counter(r["sex"] or "bos" for r in tr)),
                              "bolge": dict(Counter(r["anatom_site_general"] or "bos" for r in tr).most_common())}

    # ISIC 2019 eğitim kümesi: kaynak ayrımı
    m19 = csv_oku(META / "ISIC_2019_Training_Metadata.csv")
    g19 = {r["image"]: r for r in csv_oku(META / "ISIC_2019_Training_GroundTruth.csv")}
    on = Counter((r["lesion_id"].split("_")[0] if r["lesion_id"] else "BOS") for r in m19)
    bcn19 = [r for r in m19 if r["lesion_id"].startswith("BCN_")]
    ham19 = [r for r in m19 if r["lesion_id"].startswith("HAM_")]
    sinif_kol = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]

    def gt(i):
        r = g19[i]
        return [k for k in sinif_kol if float(r[k]) == 1.0][0]
    out["isic2019"] = {"satir": len(m19), "lesion_id_oneki": dict(on), "bcn_satir": len(bcn19), "ham_satir": len(ham19),
                       "bcn_sinif": dict(Counter(gt(r["image"]) for r in bcn19).most_common()),
                       "bcn_lezyon": len({r["lesion_id"] for r in bcn19}),
                       "bcn_ham_lesion_kesisim": len({r["lesion_id"] for r in bcn19} & {r["lesion_id"] for r in ham19}),
                       "bcn_ham_image_kesisim": len({r["image"] for r in bcn19} & {r["image"] for r in ham19})}
    # DermaMNIST image_id'leri
    es = csv_oku(DATA / "splits" / "dermamnist_lezyon_esleme.csv")
    dm_img = {r["image_id"] for r in es}
    dm_lez = {r["lesion_id"] for r in es}
    hammeta = csv_oku(DATA / "meta" / "HAM10000_metadata.csv")
    out["ham10000"] = {"kaynak_dataset_alani": dict(Counter(r["dataset"] for r in hammeta))}
    out["isic2019"]["dermamnist_image_in_isic2019"] = len(dm_img & {r["image"] for r in m19})
    out["isic2019"]["dermamnist_image_in_isic2019_ham_onekli"] = len(dm_img & {r["image"] for r in ham19})
    out["isic2019"]["dermamnist_image_in_isic2019_bcn_onekli"] = len(dm_img & {r["image"] for r in bcn19})
    out["isic2019"]["dermamnist_lesion_in_bcn_onekli"] = len(dm_lez & {r["lesion_id"] for r in bcn19})
    # figshare train ↔ ISIC2019 BCN lezyon eşleşmesi
    fl = set(lg)
    il = {r["lesion_id"] for r in bcn19}
    norm2 = lambda s: s.split("_")[-1].lstrip("0")
    out["isic2019"]["figshare_train_lezyon_vs_isic2019_bcn_lezyon"] = {
        "figshare": len(fl), "isic2019": len(il), "ham_kesisim": len(fl & il),
        "rakam_normalize_kesisim": len({norm2(x) for x in fl} & {norm2(x) for x in il}),
        "ornek_isic2019": sorted(il)[:3], "ornek_figshare": sorted(fl)[:3]}

    # ISIC Archive koleksiyonları
    b = isic_oku(249)
    h = isic_oku(212)
    if b and h:
        bid, hid = {r["isic_id"] for r in b}, {r["isic_id"] for r in h}
        bl = {klin(r, "lesion_id") for r in b} - {None}
        hl = {klin(r, "lesion_id") for r in h} - {None}
        bp = {klin(r, "patient_id") for r in b} - {None}
        hp = {klin(r, "patient_id") for r in h} - {None}
        d3 = Counter(klin(r, "diagnosis_3") or klin(r, "diagnosis_2") or klin(r, "diagnosis_1") or "YOK" for r in b)
        b19 = {r["image"] for r in bcn19}
        egitim = [r for r in b if r["isic_id"] in b19]
        diger = [r for r in b if r["isic_id"] not in b19]
        dtam = lambda rs: dict(Counter("|".join(str(klin(r, k) or "") for k in ["diagnosis_1", "diagnosis_2", "diagnosis_3"])
                                       for r in rs).most_common())
        out["isic_arsiv"] = {
            "bcn_kayit": len(b), "ham_kayit": len(h),
            "isic_id_kesisim": len(bid & hid),
            "lesion_id_kesisim": len(bl & hl), "bcn_lesion_id_dolu": sum(1 for r in b if klin(r, "lesion_id")),
            "ham_lesion_id_dolu": sum(1 for r in h if klin(r, "lesion_id")),
            "patient_id_kesisim": len(bp & hp), "bcn_patient_id_dolu": sum(1 for r in b if klin(r, "patient_id")),
            "ham_patient_id_dolu": sum(1 for r in h if klin(r, "patient_id")),
            "bcn_atif": dict(Counter(r.get("attribution") for r in b)),
            "ham_atif": dict(Counter(r.get("attribution") for r in h)),
            "bcn_lisans": dict(Counter(r.get("copyright_license") for r in b)),
            "ham_lisans": dict(Counter(r.get("copyright_license") for r in h)),
            "bcn_piksel": dict(Counter(f"{r['metadata']['acquisition'].get('pixels_x')}x{r['metadata']['acquisition'].get('pixels_y')}" for r in b).most_common(10)),
            "bcn_image_type": dict(Counter(r["metadata"]["acquisition"].get("image_type") for r in b)),
            "bcn_dermamnist_image_kesisim": len(bid & dm_img),
            "ham_koleksiyon_dermamnist_kapsama": len(hid & dm_img),
            "bcn_isic2019_egitim_kesisim": len(bid & b19), "bcn_isic2019_disi": len(diger),
            "bcn_tani_d3": dict(d3.most_common()),
            "bcn_isic2019egitim_tani": dtam(egitim), "bcn_isic2019disi_tani": dtam(diger),
            "bcn_isic2019disi_tani_dolu": sum(1 for r in diger if klin(r, "diagnosis_1")),
            "bcn_isic2019disi_lezyon": len({klin(r, "lesion_id") for r in diger}),
            "bcn_isic2019egitim_lezyon": len({klin(r, "lesion_id") for r in egitim}),
            "bcn_egitim_disi_lezyon_kesisim": len({klin(r, "lesion_id") for r in diger} & {klin(r, "lesion_id") for r in egitim}),
            "bcn_isic_lezyon_toplam": len(bl), "bcn_isic_patient_toplam": len(bp),
            "bcn_patient_goruntu_hist": dict(sorted(Counter(Counter(klin(r, "patient_id") for r in b if klin(r, "patient_id")).values()).items())[:15]),
            "bcn_isic_lezyon_goruntu_hist": dict(sorted(Counter(Counter(klin(r, "lesion_id") for r in b).values()).items())),
            "bcn_isic_egitim_lezyon_goruntu_hist": dict(sorted(Counter(Counter(klin(r, "lesion_id") for r in egitim).values()).items())),
            "bcn_dosya_boyutu_ort": round(float(np.mean([r["files"]["full"]["size"] for r in b])), 0),
            "bcn_dosya_boyutu_toplam": int(sum(r["files"]["full"]["size"] for r in b)),
            "bcn_egitim_dosya_boyutu_toplam": int(sum(r["files"]["full"]["size"] for r in egitim)),
        }
    # örnek görüntü özellikleri + piksel-NN (28 px, kalibre ön-işleme) DermaMNIST 10.015'e karşı
    ol = ORNEK / "ornek_listesi.json"
    if ol.exists():
        from PIL import Image
        kal = json.loads((KOK / "kalibrasyon_onisleme.json").read_text(encoding="utf-8"))
        geo, yon = kal["boyut"]["28"]["en_iyi"].split("|")
        f = adaylar()[yon]
        lst = json.loads(ol.read_text(encoding="utf-8"))["goruntuler"]
        boy, kare, kenar, X = Counter(), 0, [], []
        for g in lst:
            im = Image.open(ORNEK / "orijinal" / g["bcn_filename"]).convert("RGB")
            boy[f"{im.size[0]}x{im.size[1]}"] += 1
            arr = np.asarray(im)
            k = max(8, arr.shape[0] // 20)
            kose = np.concatenate([arr[:k, :k].reshape(-1, 3), arr[:k, -k:].reshape(-1, 3), arr[-k:, :k].reshape(-1, 3), arr[-k:, -k:].reshape(-1, 3)])
            kenar.append(float(kose.mean()))
            X.append(f(geometri(im, "merkez_4x3" if geo == "gerdir" else geo), 28))
        X = np.stack(X).reshape(len(X), -1).astype(np.float32)
        with np.load(MEDMNIST_ROOT / "dermamnist.npz") as d:
            R = np.concatenate([d[f"{s}_images"] for s in ["train", "val", "test"]]).reshape(-1, 28 * 28 * 3).astype(np.float32)
        d2 = (X ** 2).sum(1)[:, None] + (R ** 2).sum(1)[None, :] - 2 * X @ R.T
        nn = np.sqrt(np.maximum(d2.min(1), 0) / (28 * 28 * 3))
        out["ornek"] = {"n": len(lst), "boyutlar": dict(boy), "kose_ort_parlaklik_medyan": round(float(np.median(kenar)), 1),
                        "koyu_kose_(<40)": int(sum(1 for v in kenar if v < 40)),
                        "dosya_bayt_ort": round(float(np.mean([g["bayt"] for g in lst])), 0),
                        "nn_rmse_dermamnist10015": {"min": round(float(nn.min()), 2), "medyan": round(float(np.median(nn)), 2),
                                                    "maks": round(float(nn.max()), 2), "esik5_alti": int((nn <= 5).sum()),
                                                    "esik10_alti": int((nn <= 10).sum())},
                        "nn_onisleme": f"{'merkez_4x3+gerdir' if geo == 'gerdir' else geo}|{yon}"}
    # zip içeriği ↔ CSV
    for ad, rows in [("train", tr), ("test", te)]:
        zp = META / f"zip_icerik_{ad}.csv"
        if zp.exists():
            zn = {Path(r["ad"]).name for r in csv_oku(zp)}
            cn = {r["bcn_filename"] for r in rows}
            eksik = sorted(cn - zn)
            out.setdefault("zip_csv", {})[ad] = {"zip": len(zn), "csv": len(cn), "ortak": len(zn & cn),
                                                 "csvde_var_zipte_yok": len(eksik), "zipte_var_csvde_yok": len(zn - cn),
                                                 "eksik_ornek": eksik[:40]}
    # zip CRC32+boyut ile ikili-özdeş kopyalar (görüntü indirmeden)
    zt, ze = META / "zip_icerik_train.csv", META / "zip_icerik_test.csv"
    if zt.exists() and ze.exists():
        lab = {r["bcn_filename"]: r for r in tr}
        anah = lambda r: (r["crc32"], r["boyut"])
        grp = defaultdict(list)
        for r in csv_oku(zt):
            grp[anah(r)].append(Path(r["ad"]).name)
        dup = [v for v in grp.values() if len(v) > 1]
        ce = Counter(anah(r) for r in csv_oku(ze))
        out["zip_crc_kopya"] = {
            "train_grup": len(dup), "train_fazla_goruntu": sum(len(v) - 1 for v in dup),
            "train_ayni_lezyon": sum(1 for v in dup if len({lab[x]["lesion_id"] for x in v}) == 1),
            "train_farkli_lezyon": sum(1 for v in dup if len({lab[x]["lesion_id"] for x in v}) > 1),
            "train_farkli_tani": [[(x, lab[x]["lesion_id"], lab[x]["diagnosis"]) for x in v] for v in dup if len({lab[x]["diagnosis"] for x in v}) > 1],
            "train_farkli_lezyon_gruplar": [[(x, lab[x]["lesion_id"], lab[x]["diagnosis"]) for x in v] for v in dup if len({lab[x]["lesion_id"] for x in v}) > 1],
            "test_grup": sum(1 for c in ce.values() if c > 1), "train_test_kesisim": len(set(grp) & set(ce))}
    # sınıf eşleme seçenekleri (KARAR DEĞİL; ön-kayıt kartı seçer). Sayılar görüntü / lezyon.
    BIREBIR = {"NV": "nv", "MEL": "mel", "BCC": "bcc", "BKL": "bkl", "DF": "df", "VASC": "vasc"}
    secenek = {"E1_AK>akiec_SCC_disla": dict(BIREBIR, AK="akiec"),
               "E2_AK+SCC>akiec": dict(BIREBIR, AK="akiec", SCC="akiec"),
               "E3_AK_ve_SCC_disla_6sinif": dict(BIREBIR)}
    kopya_fazla = set()
    if zt.exists():
        for v in dup:
            kopya_fazla.update(sorted(v)[1:])
        celiski = {x for v in dup if len({lab[y]["diagnosis"] for y in v}) > 1 for x in v}
    es_out = {}
    for ad, m in secenek.items():
        for tem in ["ham", "kopyasiz"]:
            rows = [r for r in tr if r["diagnosis"] in m]
            if tem == "kopyasiz":
                rows = [r for r in rows if r["bcn_filename"] not in kopya_fazla and r["bcn_filename"] not in celiski]
            g = Counter(m[r["diagnosis"]] for r in rows)
            lz = Counter(m[d] for d in {r["lesion_id"]: r["diagnosis"] for r in rows}.values())
            es_out[f"{ad}|{tem}"] = {"goruntu": len(rows), "lezyon": len({r["lesion_id"] for r in rows}),
                                     "dislanan_goruntu": len(tr) - len(rows),
                                     "sinif_goruntu": {k: g.get(k, 0) for k in DM_ADLAR},
                                     "sinif_lezyon": {k: lz.get(k, 0) for k in DM_ADLAR},
                                     "cogunluk_payi": round(max(g.values()) / len(rows), 4)}
    out["esleme_secenekleri_figshare_train"] = es_out
    out["kopya_temizleme"] = {"kopya_fazla_goruntu": len(kopya_fazla), "celiskili_etiket_goruntu": len(celiski)}
    # Cassidy vd. 2022 (Med Image Anal 75:102305) ikili-özdeş kopya silme listesi 06 (eğitim kümeleri)
    cp = META / "cassidy_06_all_train_duplicates.txt"
    if cp.exists():
        sil = [Path(x.strip()).stem.replace("_downsampled", "") for x in open(cp, encoding="utf-8") if x.strip()]
        yil = Counter(x.split("/")[5] if len(x.split("/")) > 5 else "?" for x in open(cp, encoding="utf-8") if x.strip())
        out["cassidy06"] = {"satir": len(sil), "yil_klasoru": dict(yil),
                            "bcn_isic2019_id_silinen": len(set(sil) & {r["image"] for r in bcn19}),
                            "dermamnist_id_silinen": len(set(sil) & dm_img)}
    json_yaz(KOK / "bcn20000_analiz.json", out)
    log(json.dumps(out, ensure_ascii=False, indent=1)[:6000])


# ───────────────────────── ön-işleme ─────────────────────────
_Z = None


def _isci_baslat(zip_yol):
    global _Z
    _Z = zipfile.ZipFile(zip_yol)


def _isci_isle(arg):
    """Tam veri: zip'ten oku -> geometri -> S px (kalibre interpolasyon). Süreç havuzunda koşar."""
    from PIL import Image
    ic_ad, geo, yon, S = arg
    im = Image.open(io.BytesIO(_Z.read(ic_ad))).convert("RGB")
    return adaylar()[yon](geometri(im, geo), S)


def asama_onisleme(a):
    from PIL import Image
    kal = json.loads((KOK / "kalibrasyon_onisleme.json").read_text(encoding="utf-8"))
    tr = {r["bcn_filename"]: r for r in csv_oku(META / "bcn_20k_train.csv")}
    if a.kaynak == "ornek":
        lst = [g["bcn_filename"] for g in json.loads((ORNEK / "ornek_listesi.json").read_text(encoding="utf-8"))["goruntuler"]]
        oku = lambda fn: Image.open(ORNEK / "orijinal" / fn).convert("RGB")
        cikti = ORNEK
    else:
        zip_yol = Path(a.zip_yol) if a.zip_yol else ZIPDIR / "BCN_20k_train.zip"
        z = zipfile.ZipFile(zip_yol)
        adlar = {Path(i.filename).name: i for i in z.infolist() if not i.is_dir()}
        lst = sorted(fn for fn in tr if fn in adlar)
        oku = lambda fn: Image.open(io.BytesIO(z.read(adlar[fn]))).convert("RGB")
        cikti = Path(a.cikti_dizin) if a.cikti_dizin else KOK
    for S in a.boyutlar:
        geo_ref, yon = kal["boyut"][str(S) if str(S) in kal["boyut"] else "28"]["en_iyi"].split("|")
        f = adaylar()[yon]
        for geo in a.geometriler:
            X = np.zeros((len(lst), S, S, 3), np.uint8)
            if a.kaynak == "tam" and a.isci > 1:
                from concurrent.futures import ProcessPoolExecutor
                with ProcessPoolExecutor(a.isci, initializer=_isci_baslat, initargs=(str(zip_yol),)) as ex:
                    for i, y in enumerate(ex.map(_isci_isle, [(adlar[fn].filename, geo, yon, S) for fn in lst], chunksize=16)):
                        X[i] = y
                        if i % 1000 == 0:
                            log(f"  {S}px {geo}: {i}/{len(lst)}")
            else:
                for i, fn in enumerate(lst):
                    X[i] = f(geometri(oku(fn), geo), S)
                    if a.kaynak == "tam" and i % 1000 == 0:
                        log(f"  {S}px {geo}: {i}/{len(lst)}")
            ad = f"bcn20000_{'ornek' if a.kaynak == 'ornek' else 'train'}_{geo}_{S}.npz"
            np.savez_compressed(cikti / ad, images=X, bcn_filename=np.array(lst), diagnosis=np.array([tr[fn]["diagnosis"] for fn in lst]),
                                lesion_id=np.array([tr[fn]["lesion_id"] for fn in lst]))
            json_yaz(cikti / ad.replace(".npz", ".json"), {"n": len(lst), "boyut": S, "geometri": geo, "interpolasyon": yon,
                                                           "kalibrasyon_ref": f"{geo_ref}|{yon}", "sha256": dosya_hash(cikti / ad, "sha256")})
            log(f"yazıldı {ad} {X.shape}")
    return {"n": len(lst), "boyutlar": len(a.boyutlar)}


# ───────────────────────── tam veri piksel-NN (HAM çakışması) ─────────────────────────
def asama_nn_tam(a):
    """BCN train (merkez_kare, 28 px, kalibre interpolasyon) -> DermaMNIST 28 px 10.015 görüntü (train+val+test;
    DermaMNIST-C train resmî val/test görüntülerini de içerdiği için hepsi eğitim-ilişkili sayılır). Yalnız istatistik."""
    B = np.load(Path(a.nn_girdi) if a.nn_girdi else KOK / "bcn20000_train_merkez_kare_28.npz")
    X = B["images"].reshape(len(B["images"]), -1).astype(np.float32)
    with np.load(MEDMNIST_ROOT / "dermamnist.npz") as d:
        parca = [(s_, d[f"{s_}_images"]) for s_ in ["train", "val", "test"]]
    R = np.concatenate([p_ for _, p_ in parca]).reshape(-1, 28 * 28 * 3).astype(np.float32)
    etiket = np.concatenate([[f"{s_}:{i}" for i in range(len(p_))] for s_, p_ in parca])
    rn = (R ** 2).sum(1)
    nn_d, nn_i = np.zeros(len(X), np.float32), np.zeros(len(X), np.int64)
    for i in range(0, len(X), 1024):
        x = X[i:i + 1024]
        d2 = (x ** 2).sum(1)[:, None] + rn[None, :] - 2 * x @ R.T
        nn_i[i:i + 1024] = d2.argmin(1)
        nn_d[i:i + 1024] = np.sqrt(np.maximum(d2.min(1), 0) / X.shape[1])
    sira = np.argsort(nn_d)[:30]
    I = B["images"].astype(np.float32)
    kose = np.concatenate([I[:, :3, :3], I[:, :3, -3:], I[:, -3:, :3], I[:, -3:, -3:]], 1).reshape(len(I), -1).mean(1)
    with np.load(MEDMNIST_ROOT / "dermamnist.npz") as d:
        H = d["train_images"].astype(np.float32)
    hk = np.concatenate([H[:, :3, :3], H[:, :3, -3:], H[:, -3:, :3], H[:, -3:, -3:]], 1).reshape(len(H), -1).mean(1)
    out = {"n_bcn": int(len(X)), "n_ref": int(len(R)),
           "nn_rmse": {"min": round(float(nn_d.min()), 2), "p1": round(float(np.percentile(nn_d, 1)), 2),
                       "medyan": round(float(np.median(nn_d)), 2)},
           "koyu_kose_orani_bcn_28px(<40)": round(float((kose < 40).mean()), 4),
           "koyu_kose_orani_dermamnist_train_28px(<40)": round(float((hk < 40).mean()), 4),
           "esik5_alti": int((nn_d <= 5).sum()), "esik10_alti": int((nn_d <= 10).sum()), "esik15_alti": int((nn_d <= 15).sum()),
           "en_yakin_30": [{"bcn": str(B["bcn_filename"][j]), "dx": str(B["diagnosis"][j]), "ref": str(etiket[nn_i[j]]),
                            "rmse": round(float(nn_d[j]), 2)} for j in sira]}
    json_yaz(Path(a.nn_cikti) if a.nn_cikti else KOK / "bcn20000_nn_tam.json", out)
    log(json.dumps({k: v for k, v in out.items() if k != "en_yakin_30"}, ensure_ascii=False))
    return {k: v for k, v in out.items() if k != "en_yakin_30"}


# ───────────────────────── tam indirme ─────────────────────────
def asama_indir(a):
    rec = figshare_kaydi()
    f = fig_dosya(rec, "BCN_20k_train.zip")
    ZIPDIR.mkdir(parents=True, exist_ok=True)
    hedef = ZIPDIR / f["name"]
    part = hedef.with_name(hedef.name + ".part")
    if hedef.exists() and hedef.stat().st_size == f["size"] and dosya_hash(hedef, "md5") == f["computed_md5"]:
        log("zaten var, MD5 eşleşti")
        return {"bayt": f["size"], "md5_eslesti": 1, "sure_sn": 0}
    t0 = time.time()
    while True:
        bas = part.stat().st_size if part.exists() else 0
        if bas >= f["size"]:
            break
        h = dict(UA)
        h["Range"] = f"bytes={bas}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(f["download_url"], headers=h), timeout=120) as r:
                if bas and r.status != 206:
                    part.unlink()
                    continue
                with open(part, "ab" if bas else "wb") as fo:
                    son = time.time()
                    while True:
                        blk = r.read(1 << 20)
                        if not blk:
                            break
                        fo.write(blk)
                        if time.time() - son > 30:
                            n = part.stat().st_size
                            log(f"  {n / 1e9:.2f}/{f['size'] / 1e9:.2f} GB, ort {n / max(time.time() - t0, 1) / 1e6:.2f} MB/s")
                            son = time.time()
        except Exception as e:
            log(f"kesinti: {e}; 10 sn sonra devam")
            time.sleep(10)
    md5 = dosya_hash(part, "md5")
    if md5 != f["computed_md5"]:
        part.unlink()
        raise SystemExit(f"MD5 UYUŞMADI {md5} != {f['computed_md5']}")
    part.replace(hedef)
    kayit_guncelle("tam_indirme", {"dosya": f["name"], "bayt": f["size"], "md5": md5, "sha256": dosya_hash(hedef, "sha256"),
                                   "sure_sn": round(time.time() - t0, 1)})
    log("tam indirme tamam, MD5 eşleşti")
    return {"bayt": f["size"], "md5_eslesti": 1, "sure_sn": round(time.time() - t0, 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--asama", required=True, choices=["meta", "zipliste", "ornek", "kalibrasyon", "analiz", "onisleme", "indir", "nn_tam"])
    p.add_argument("--nn-girdi", default=None, help="nn_tam girdisi (varsayılan tam train 28 px npz; duman testi için örnek)")
    p.add_argument("--nn-cikti", default=None)
    p.add_argument("--metrics", default=None, help="kuyruk/kosum.py için düz JSON özet ({RUN_DIR}/metrics.json)")
    p.add_argument("--koleksiyonlar", type=int, nargs="+", default=[249, 212])
    p.add_argument("--n-ornek", type=int, default=40)
    p.add_argument("--n-ham", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--kaynak", choices=["ornek", "tam"], default="ornek")
    p.add_argument("--boyutlar", type=int, nargs="+", default=[28, 224])
    p.add_argument("--zip-yol", default=None, help="tam ön-işleme zip'i (varsayılan data/bcn20000/zip/BCN_20k_train.zip)")
    p.add_argument("--cikti-dizin", default=None)
    p.add_argument("--isci", type=int, default=1, help="tam ön-işleme süreç sayısı")
    p.add_argument("--geometriler", nargs="+", default=["merkez_kare"],
                   help="merkez_kare = MedMNIST yöntemi (kalibrasyonla ölçüldü; BCN 1024x1024 karede özdeşlik)")
    a = p.parse_args()
    t0 = time.time()
    r = {"meta": asama_meta, "zipliste": asama_zipliste, "ornek": asama_ornek, "kalibrasyon": asama_kalibrasyon,
         "analiz": asama_analiz, "onisleme": asama_onisleme, "indir": asama_indir, "nn_tam": asama_nn_tam}[a.asama](a)
    if a.metrics:
        m = {"asama": a.asama, "sure_sn": round(time.time() - t0, 1)}
        for k, v in (r or {}).items():
            if isinstance(v, dict):
                m.update({f"{k}_{kk}": vv for kk, vv in v.items()})
            else:
                m[k] = v
        json_yaz(Path(a.metrics), m)


if __name__ == "__main__":
    main()
