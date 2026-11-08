# -*- coding: utf-8 -*-
"""
e3_ham_val_hazirla.py — H16 (E3) a0 VERİ KAPISI ve ETİKETSİZ HAT GİRDİSİ (CPU + ağ; GPU ve model YOK; hipotezler/H16.md §a0).

Resmî val'in 1.003 görüntüsü HAM10000'in ISIC özgün JPEG'lerinden, E2'nin BCN 224 girdisini üreten AYNI işlevlerle
(`scripts/indir_bcn20000.py` → `geometri`, `adaylar`; İÇE AKTARILIR, kopyalanmaz) yeniden üretilir. Model çıktısına bakılmaz.
Aşamalar (--asama hepsi = sırayla):
  liste    K-a: indirme listesi YALNIZ split == "val" satırlarından; liste ∩ split == "test" image_id kümesi = 0, aksi DUR.
           K-b: val satırı = 1.003, index kümesi = {0..1002}, image_id tekil; aksi DUR. N/S lezyon ayrımı (N: lezyonun
           split == "train" satırı yok). Yalnız `dermamnist_lezyon_esleme.csv` okunur (test satırlarının yalnız image_id'si).
  indir    ISIC API v2 `/images/<id>/` meta + `files.full` (IB.http_get, IB.ISIC_API) → data/ham_val_isic/<id>.{jpg,json};
           görüntü başına lisans alanı. Yeniden başlatılabilir (var olan dosya atlanır), atomik yazım, iş parçacığı havuzu.
           İndirilemeyen > 10 → DUR; ≤ 10 → listelenir, iki kolda da dışlanır.
  qbcn     S5 için q_BCN: BCN_20k_train.zip'teki sıralı (ad) ilk 500 .jpg dosyasının YALNIZ JPEG başlığı (nicemleme tabloları,
           kroma alt-örneklemesi). Zip data/bcn20000/zip/ altında varsa yerelden, yoksa figshare HTTP Range ile
           (IB.HttpRangeDosya; dosya başına tek Range isteği). IJG ölçek ters çevirimi: Pillow'un quality = 1..100 için ürettiği
           tablolarla en yakın eşleşme; medyan. Piksel çözülmez, `bcn_20k_train.csv` AÇILMAZ, modele hiçbir şey verilmez.
  hazirla  Dört hat (index sırasıyla): hat = IB.adaylar()[yon](IB.geometri(im, geo), 224), geo|yon kalibrasyon_onisleme.json
           boyut["224"]["en_iyi"]'den (= merkez_kare|PIL_BILINEAR; değilse DUR). jpeg = merkez kare (450²) → Pillow JPEG
           (quality = q_BCN, subsampling = BCN) → çöz → hat. z125 = merkez 360² kırp → adaylar()[yon]. z080 = merkez 562²
           (dikeyde yansıtmalı dolgu, numpy "reflect") → adaylar()[yon]. Görüntü başına RMSE (hat ↔ dermamnist_224 val_images)
           ve kanal ortalaması farkı; K-b RMSE kapısı (medyan ∈ [1,0; 3,0], RMSE > 15 sayısı = 0); ISIC meta tanısı ↔ val_labels
           (taksonomi tablosu; uyuşmazlık > %2 → DUR). npz anahtarları: images, index, image_id — ETİKET YOK.
  beklenen kuyruk/e3_onisleme_beklenen.json (dondurulur; varsa ve farklıysa DUR) + data/ham_val_isic/e3_a0_rapor.{md,json}.
Sınama (--sinama DIZIN): 8 HAM kalibrasyon (TRAIN) görüntüsüyle aynı hat; birincil hat çıktısı E2'nin bcn_e2_kal_224.npz
  görüntüleriyle BAYT BAYT aynı olmalı ("aynı işlev" kanıtı). Ağ yalnız --qbcn-n > 0 ise (q_BCN yolunun sınaması).
Çıkış: 0 GEÇTİ · 2 DUR (K-a / K-b) · 3 beklenen dosyası zaten var ve farklı.
Kullanım (WS kökü):
  env/.venv/Scripts/python.exe -I scripts/e3_ham_val_hazirla.py --asama hepsi --metrics {RUN_DIR}/metrics.json
  env/.venv/Scripts/python.exe -I scripts/e3_ham_val_hazirla.py --sinama <geçici dizin> [--qbcn-n 5]
"""
import argparse
import csv
import hashlib
import io
import json
import os
import struct
import sys
import time
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "scripts"))
sys.path.insert(0, str(WS / "varyantlar"))
import indir_bcn20000 as IB  # noqa: E402  (geometri, adaylar, http_get, ISIC_API, HttpRangeDosya, figshare_kaydi; DEĞİŞTİRİLMEZ)
from e2 import bcn_veri as BV  # noqa: E402  (icerik_sha256, sira_sha256, dosya_sha256, ETIKET_ANAHTARLARI)

HEDEF = WS / "data" / "ham_val_isic"
ESLEME_CSV = WS / "data" / "splits" / "dermamnist_lezyon_esleme.csv"
NPZ224 = WS / "data" / "medmnist" / "dermamnist_224.npz"
KAL_JSON = WS / "data" / "bcn20000" / "kalibrasyon_onisleme.json"
KAL_HAM = WS / "data" / "bcn20000" / "kalibrasyon_ham"
E2_KAL_NPZ = WS / "data" / "bcn20000" / "e2_girdi" / "bcn_e2_kal_224.npz"
E2_BEKLENEN = WS / "kuyruk" / "e2_bcn_beklenen.json"
BEKLENEN = WS / "kuyruk" / "e3_onisleme_beklenen.json"
KART = WS / "hipotezler" / "H16.md"
DM = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]          # DermaMNIST indeks sırası
HATLAR = ("hat", "jpeg", "z125", "z080")
NPZ_AD = {g: f"e3_val_{g}_224.npz" for g in HATLAR}
N_VAL, EKSIK_MAKS, RMSE_MEDYAN, RMSE_UC, TANI_UYUSMAZ_MAKS = 1003, 10, (1.0, 3.0), 15.0, 0.02
BEKLENEN_YON = "merkez_kare|PIL_BILINEAR"
Q_VARSAYILAN = 75
Z125, Z080 = 1.25, 0.8

# ISIC v2 taksonomisi → DermaMNIST (HAM10000 dx). Önce diagnosis_3 (tam ad), sonra diagnosis_2. Eşlenemeyen = uyuşmaz sayılır.
TANI3 = {"Nevus": "nv", "Melanoma, NOS": "mel", "Melanoma in situ": "mel", "Melanoma Invasive": "mel",
         "Basal cell carcinoma": "bcc", "Pigmented benign keratosis": "bkl", "Seborrheic keratosis": "bkl",
         "Solar lentigo": "bkl", "Lichen planus like keratosis": "bkl", "Dermatofibroma": "df",
         "Solar or actinic keratosis": "akiec", "Squamous cell carcinoma in situ": "akiec",
         "Squamous cell carcinoma, NOS": "akiec", "Hemangioma": "vasc", "Angiokeratoma": "vasc",
         "Pyogenic granuloma": "vasc", "Hemorrhage": "vasc", "Angioma": "vasc"}
TANI2 = {"Benign melanocytic proliferations": "nv", "Malignant melanocytic proliferations (Melanoma)": "mel",
         "Malignant adnexal epithelial proliferations - Follicular": "bcc", "Benign epidermal proliferations": "bkl",
         "Flat melanotic pigmentations - not melanocytic nevus": "bkl", "Indeterminate epidermal proliferations": "akiec",
         "Malignant epidermal proliferations": "akiec", "Benign soft tissue proliferations - Fibro-histiocytic": "df",
         "Benign soft tissue proliferations - Vascular": "vasc"}

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:  # noqa: BLE001
    pass


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sha(p) -> str:
    return BV.dosya_sha256(Path(p))


def dur(msg, R=None):
    log(f"DUR: {msg}")
    if R is not None:
        R.setdefault("dur", []).append(msg)
    raise SystemExit(2)


def json_yaz(p: Path, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, p)


class ValNpz:
    """dermamnist_224.npz'den YALNIZ val_images / val_labels (K-a: başka anahtar okunursa sayılır ve reddedilir)."""
    IZINLI = ("val_images", "val_labels")

    def __init__(self, yol=NPZ224):
        self.z = np.load(yol, allow_pickle=False)
        self.okunan = Counter()

    def __getitem__(self, k):
        self.okunan[k] += 1
        if k not in self.IZINLI:
            raise SystemExit(f"HATA (K-a): dermamnist_224.npz → {k!r} okunamaz (yalnız {self.IZINLI})")
        return self.z[k]

    def test_okundu(self) -> int:
        return int(sum(v for k, v in self.okunan.items() if k.startswith("test")))


def tani_esle(meta: dict):
    c = (meta.get("metadata") or {}).get("clinical") or {}
    d3, d2 = c.get("diagnosis_3"), c.get("diagnosis_2")
    if d3 in TANI3:
        return TANI3[d3], d2, d3
    if d3 and "nevus" in str(d3).lower():
        return "nv", d2, d3
    return TANI2.get(d2), d2, d3


# ───────────────────────── liste (K-a / K-b sayım) ─────────────────────────
def liste(R: dict) -> list:
    with open(ESLEME_CSV, encoding="utf-8", newline="") as f:
        satir = list(csv.DictReader(f))
    val = sorted((r for r in satir if r["split"] == "val"), key=lambda r: int(r["index"]))
    test_id = {r["image_id"] for r in satir if r["split"] == "test"}        # yalnız kimlik (K-a kesişimi)
    train_lez = {r["lesion_id"] for r in satir if r["split"] == "train"}
    kesisim = sorted({r["image_id"] for r in val} & test_id)
    if kesisim:
        dur(f"K-a: indirme listesi ∩ test image_id = {len(kesisim)} > 0 ({kesisim[:5]})", R)
    idx = [int(r["index"]) for r in val]
    if len(val) != N_VAL or sorted(idx) != list(range(N_VAL)) or len({r["image_id"] for r in val}) != N_VAL:
        dur(f"K-b: val satırı {len(val)} / index kümesi / image_id tekilliği bozuk", R)
    lez = sorted({r["lesion_id"] for r in val})
    N = sorted(l_ for l_ in lez if l_ not in train_lez)
    R["liste"] = {"n_val": len(val), "test_kesisim": 0, "lezyon_val": len(lez), "lezyon_N": len(N), "lezyon_S": len(lez) - len(N),
                  "goruntu_N": sum(1 for r in val if r["lesion_id"] not in train_lez), "esleme_csv_sha256": sha(ESLEME_CSV),
                  "liste_sha256": BV.sira_sha256([r["image_id"] for r in val])}
    R["_N_lezyon"] = N
    log(f"liste: val {len(val)} · test kesişimi 0 · lezyon {len(lez)} (N {len(N)} / S {len(lez) - len(N)})")
    return val


# ───────────────────────── indir ─────────────────────────
def _indir_bir(iid: str, dizin: Path) -> dict:
    jpg, js = dizin / f"{iid}.jpg", dizin / f"{iid}.json"
    if jpg.exists() and js.exists():
        return {"image_id": iid, "durum": "vardi"}
    try:
        _, j = IB.http_get(f"{IB.ISIC_API}/images/{iid}/", timeout=60)
        meta = json.loads(j.decode("utf-8"))
        if meta.get("isic_id") != iid:
            return {"image_id": iid, "durum": "hata", "neden": f"isic_id {meta.get('isic_id')} ≠ {iid}"}
        _, img = IB.http_get(meta["files"]["full"]["url"], timeout=120)
        from PIL import Image
        Image.open(io.BytesIO(img)).verify()
        t1, t2 = jpg.with_name(jpg.name + ".tmp"), js.with_name(js.name + ".tmp")
        t1.write_bytes(img)
        t2.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(t1, jpg)
        os.replace(t2, js)
        return {"image_id": iid, "durum": "indi"}
    except Exception as e:  # noqa: BLE001
        return {"image_id": iid, "durum": "hata", "neden": f"{type(e).__name__}: {e}"[:300]}


def indir(R: dict, val: list, dizin: Path, isci: int):
    dizin.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sonuc = []
    with ThreadPoolExecutor(isci) as ex:
        for i, s in enumerate(ex.map(lambda r: _indir_bir(r["image_id"], dizin), val)):
            sonuc.append(s)
            if i % 100 == 0:
                log(f"  indir {i}/{len(val)} ({time.time() - t0:.0f}s)")
    eksik = sorted(s["image_id"] for s in sonuc if s["durum"] == "hata")
    R["indir"] = {"n": len(val), "durum": dict(Counter(s["durum"] for s in sonuc)), "eksik": eksik,
                  "hatalar": [s for s in sonuc if s["durum"] == "hata"][:20], "isci": isci,
                  "sure_sn": round(time.time() - t0, 1), "kaynak": f"{IB.ISIC_API}/images/<id>/ → files.full"}
    log(f"indir: {R['indir']['durum']} · eksik {len(eksik)} · {R['indir']['sure_sn']}s")
    if len(eksik) > EKSIK_MAKS:
        dur(f"K-b: indirilemeyen görüntü {len(eksik)} > {EKSIK_MAKS}", R)


# ───────────────────────── q_BCN (yalnız JPEG başlığı) ─────────────────────────
def ijg_tablolari() -> dict:
    """Pillow'un quality = 1..100 için yazdığı nicemleme tabloları (okunduğu sırayla; dosyadan okunan tablolarla aynı düzen)."""
    from PIL import Image
    ref = {}
    for q in range(1, 101):
        b = io.BytesIO()
        Image.new("RGB", (16, 16), (128, 128, 128)).save(b, "JPEG", quality=q)
        qt = Image.open(io.BytesIO(b.getvalue())).quantization
        ref[q] = (np.asarray(qt[0], np.int64), np.asarray(qt[1], np.int64) if 1 in qt else None)
    return ref


def q_kestir(veri: bytes, ref: dict) -> dict:
    from PIL import Image, JpegImagePlugin
    im = Image.open(io.BytesIO(veri))                 # yalnız başlık; piksel ÇÖZÜLMEZ (load çağrılmaz)
    qt = im.quantization
    t0 = np.asarray(qt[0], np.int64)
    t1 = np.asarray(qt[1], np.int64) if 1 in qt else None
    uzak = {}
    for q, (r0, r1) in ref.items():
        d = int(np.abs(t0 - r0).sum())
        if t1 is not None and r1 is not None:
            d += int(np.abs(t1 - r1).sum())
        uzak[q] = d
    qb = min(uzak, key=lambda q: (uzak[q], -q))
    return {"q": int(qb), "uzaklik": uzak[qb], "tam": int(uzak[qb] == 0), "alt_ornek": int(JpegImagePlugin.get_sampling(im)),
            "tablo_sayisi": len(qt), "boyut": f"{im.size[0]}x{im.size[1]}"}


def _yerel_baslik(info, ham: "IB.HttpRangeDosya", bayt: int) -> bytes:
    """Zip yerel başlığı + ilk `bayt` veri baytı, TEK Range isteğiyle; sıkıştırılmışsa kısmi açma."""
    ham.seek(info.header_offset)
    n = 30 + len(info.orig_filename.encode("utf-8")) + 1024 + bayt
    buf = bytearray(n)
    k = ham.readinto(buf)
    buf = bytes(buf[:k])
    if buf[:4] != b"PK\x03\x04":
        raise IOError("yerel başlık imzası yok")
    yontem, = struct.unpack("<H", buf[8:10])
    ad_n, ek_n = struct.unpack("<HH", buf[26:30])
    veri = buf[30 + ad_n + ek_n:]
    if yontem == 0:
        return veri
    if yontem == 8:
        return zlib.decompressobj(-15).decompress(veri)
    raise IOError(f"desteklenmeyen sıkıştırma {yontem}")


def _yerel_zip_baslik(z, info) -> dict:
    """Yerel zip: üyenin yalnız ilk 64 KB'ı (gerekirse 256 KB, en son tamamı) — JPEG başlığı; piksel çözülmez."""
    for bayt in (1 << 16, 1 << 18, None):
        try:
            with z.open(info) as f:
                veri = f.read() if bayt is None else f.read(bayt)
            return {"ad": Path(info.filename).name, **q_kestir(veri, IJG), "okunan_bayt": len(veri)}
        except Exception as e:  # noqa: BLE001
            son = e
    return {"ad": Path(info.filename).name, "hata": f"{type(son).__name__}: {son}"[:200]}


IJG = {}


def qbcn(R: dict, n: int, isci: int, kayit_dizin: Path, uzak_zorla: bool = False) -> dict:
    ref = ijg_tablolari()
    IJG.clear()
    IJG.update(ref)
    zip_yol = IB.ZIPDIR / "BCN_20k_train.zip"
    t0 = time.time()
    if zip_yol.exists() and not uzak_zorla:
        import zipfile
        z = zipfile.ZipFile(zip_yol)
        bilgi = sorted((i for i in z.infolist() if not i.is_dir() and i.filename.lower().endswith(".jpg")),
                       key=lambda i: Path(i.filename).name)[:n]
        sonuc = [_yerel_zip_baslik(z, i) for i in bilgi]
        kaynak = {"tur": "yerel zip (üye başına ilk 64 KB)", "yol": str(zip_yol.relative_to(WS)).replace("\\", "/"),
                  "zip_bayt": zip_yol.stat().st_size}
    else:
        import zipfile
        rec = IB.figshare_kaydi()                      # data/bcn20000/meta/figshare_24140028.json (yerelde; yazmaz)
        f = IB.fig_dosya(rec, "BCN_20k_train.zip")
        ham0 = IB.HttpRangeDosya(f["download_url"], f["size"])
        z = zipfile.ZipFile(io.BufferedReader(ham0, buffer_size=1 << 20))
        bilgi = sorted((i for i in z.infolist() if not i.is_dir() and i.filename.lower().endswith(".jpg")),
                       key=lambda i: Path(i.filename).name)[:n]

        def bir(info):
            ham = IB.HttpRangeDosya(f["download_url"], f["size"])
            for bayt in (1 << 16, 1 << 18, info.compress_size):
                try:
                    return {"ad": Path(info.filename).name, **q_kestir(_yerel_baslik(info, ham, bayt), ref), "range_istek": ham.istek}
                except Exception as e:  # noqa: BLE001  (başlık kesik → daha büyük parça)
                    son = e
            return {"ad": Path(info.filename).name, "hata": f"{type(son).__name__}: {son}"[:200]}
        with ThreadPoolExecutor(isci) as ex:
            sonuc = list(ex.map(bir, bilgi))
        kaynak = {"tur": "figshare HTTP Range", "doi": rec.get("doi"), "dosya": f["name"], "boyut": f["size"],
                  "md5_resmi": f["computed_md5"], "merkezi_dizin_range_istek": ham0.istek}
    iyi = [s for s in sonuc if "q" in s]
    if not iyi:
        R["qbcn"] = {"kaynak": kaynak, "n": 0, "q_BCN": Q_VARSAYILAN, "varsayilan": True, "hatalar": sonuc[:10]}
        log(f"q_BCN: başlık okunamadı → sabit {Q_VARSAYILAN} (beyan)")
        return R["qbcn"]
    qs = np.array([s["q"] for s in iyi])
    alt = Counter(s["alt_ornek"] for s in iyi)
    ss = alt.most_common(1)[0][0]
    R["qbcn"] = {"kaynak": kaynak, "n_istenen": n, "n": len(iyi), "hata": len(sonuc) - len(iyi),
                 "q_medyan_ham": float(np.median(qs)), "q_BCN": int(np.floor(np.median(qs) + 0.5)), "varsayilan": False,
                 "q_dagilim": {str(k): v for k, v in sorted(Counter(qs.tolist()).items())},
                 "tam_eslesme_orani": round(float(np.mean([s["tam"] for s in iyi])), 4),
                 "uzaklik_maks": int(max(s["uzaklik"] for s in iyi)),
                 "alt_ornek_dagilim": {str(k): v for k, v in alt.items()}, "subsampling": int(ss),
                 "boyut_dagilim": dict(Counter(s["boyut"] for s in iyi).most_common(5)),
                 "ilk_ad": iyi[0]["ad"], "son_ad": iyi[-1]["ad"], "sure_sn": round(time.time() - t0, 1),
                 "yontem": "Pillow quality 1..100 tablolarıyla en yakın (L1) eşleşme (IJG ölçek ters çevirimi), medyan; "
                           "alt-örnekleme çoğunluk (Pillow get_sampling: 0 = 4:4:4, 1 = 4:2:2, 2 = 4:2:0)"}
    if ss not in (0, 1, 2):
        dur(f"q_BCN: alt-örnekleme kodu {ss} Pillow'a verilemez", R)
    kayit_dizin.mkdir(parents=True, exist_ok=True)
    json_yaz(kayit_dizin / "qbcn_dosya.json", sonuc)
    R["qbcn"]["dosya_kaydi_sha256"] = sha(kayit_dizin / "qbcn_dosya.json")
    log(f"q_BCN = {R['qbcn']['q_BCN']} (medyan {R['qbcn']['q_medyan_ham']}, n {len(iyi)}, tam eşleşme "
        f"{R['qbcn']['tam_eslesme_orani']}), subsampling {ss}, {R['qbcn']['sure_sn']}s")
    return R["qbcn"]


# ───────────────────────── hatlar ─────────────────────────
def kal_yon():
    kal = json.loads(KAL_JSON.read_text(encoding="utf-8"))
    en = kal["boyut"]["224"]["en_iyi"]
    if en != BEKLENEN_YON:
        raise SystemExit(f"DUR: kalibrasyon_onisleme.json 224 en_iyi {en} ≠ BCN hattı {BEKLENEN_YON}")
    geo, yon = en.split("|")
    return geo, yon


def merkez_kare_kenar(im, kenar: int):
    """Merkezden kenar×kenar bölge; kaynak dışına taşan kısım numpy 'reflect' ile doldurulur (S7). Dolgu miktarı döner."""
    from PIL import Image
    a = np.asarray(im)
    h, w = a.shape[:2]
    pad = [[0, 0], [0, 0], [0, 0]]
    for ax, L in ((0, h), (1, w)):
        if L < kenar:
            fazla = kenar - L
            pad[ax] = [fazla // 2, fazla - fazla // 2]
    if any(sum(p) for p in pad):
        a = np.pad(a, pad, mode="reflect")
    h, w = a.shape[:2]
    t, l_ = (h - kenar) // 2, (w - kenar) // 2
    return Image.fromarray(np.ascontiguousarray(a[t:t + kenar, l_:l_ + kenar])), [pad[0], pad[1]]


def hatlar_uret(im, geo: str, yon: str, q: int, ss: int) -> tuple:
    """im: PIL RGB (çözülmüş HAM JPEG). Dönüş: ({hat: 224² uint8}, bilgi)."""
    from PIL import Image
    f = IB.adaylar()[yon]
    out = {"hat": f(IB.geometri(im, geo), 224)}
    kare = IB.geometri(im, geo)                                          # 450² (HAM 600×450)
    b = io.BytesIO()
    kare.save(b, "JPEG", quality=int(q), subsampling=int(ss))
    out["jpeg"] = f(IB.geometri(Image.open(io.BytesIO(b.getvalue())).convert("RGB"), geo), 224)
    s = min(im.size)
    z1, p1 = merkez_kare_kenar(im, int(round(s / Z125)))
    z0, p0 = merkez_kare_kenar(im, int(s / Z080))                       # 450 / 0,8 = 562,5 → 562 (kart)
    out["z125"], out["z080"] = f(z1, 224), f(z0, 224)
    return out, {"boyut": f"{im.size[0]}x{im.size[1]}", "jpeg_bayt": b.tell(), "z125_kenar": z1.size[0], "z080_kenar": z0.size[0],
                 "z080_dolgu": p0}


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)))


def hazirla(R: dict, kayitlar: list, kaynak_dizin: Path, ref: np.ndarray, cikti: Path, q: int, ss: int, etiket_val=None) -> dict:
    """kayitlar: [(index, image_id)] (sıra = çıktı sırası); ref: (len, 224, 224, 3) aynı sıradaki MedMNIST+ görüntüleri."""
    from PIL import Image
    import PIL
    geo, yon = kal_yon()
    X = {g: [] for g in HATLAR}
    idx, ids, bilgi, rm = [], [], [], {g: [] for g in HATLAR}
    kanal = {g: [] for g in HATLAR}
    tani, lisans, boyut = [], Counter(), Counter()
    eksik = []
    t0 = time.time()
    for j, (i, iid) in enumerate(kayitlar):
        p = kaynak_dizin / f"{iid}.jpg"
        if not p.exists():
            eksik.append(iid)
            continue
        im = Image.open(p).convert("RGB")
        Y, b = hatlar_uret(im, geo, yon, q, ss)
        for g in HATLAR:
            X[g].append(Y[g])
            rm[g].append(rmse(Y[g], ref[j]))
            kanal[g].append(Y[g].reshape(-1, 3).astype(np.float64).mean(0) - ref[j].reshape(-1, 3).astype(np.float64).mean(0))
        idx.append(int(i))
        ids.append(iid)
        bilgi.append({"index": int(i), "image_id": iid, **b, "rmse_hat": round(rm["hat"][-1], 4)})
        boyut[b["boyut"]] += 1
        js = kaynak_dizin / f"{iid}.json"
        if js.exists():
            meta = json.loads(js.read_text(encoding="utf-8"))
            lisans[str(meta.get("copyright_license"))] += 1
            if etiket_val is not None:
                dm, d2, d3 = tani_esle(meta)
                tani.append({"index": int(i), "isic": dm, "val": DM[int(etiket_val[i])], "d2": d2, "d3": d3})
        if j % 200 == 0:
            log(f"  hazırla {j}/{len(kayitlar)} ({time.time() - t0:.0f}s)")
    if len(eksik) > EKSIK_MAKS:
        dur(f"K-b: kaynak dizinde olmayan görüntü {len(eksik)} > {EKSIK_MAKS}", R)
    cikti.mkdir(parents=True, exist_ok=True)
    idx_a, ids_a = np.asarray(idx, np.int64), np.asarray(ids)
    for g in HATLAR:
        A = np.stack(X[g]).astype(np.uint8)
        np.savez_compressed(cikti / NPZ_AD[g], images=A, index=idx_a, image_id=ids_a)
        with np.load(cikti / NPZ_AD[g], allow_pickle=False) as z:
            fazla = BV.ETIKET_ANAHTARLARI & set(z.files)
            if fazla or set(z.files) != {"images", "index", "image_id"}:
                dur(f"{NPZ_AD[g]} anahtarları {z.files} (etiket anahtarı olmamalı)", R)
    ozet = {}
    for g in HATLAR:
        r = np.asarray(rm[g])
        ozet[g] = {"medyan": round(float(np.median(r)), 4), "ort": round(float(r.mean()), 4),
                   "p95": round(float(np.percentile(r, 95)), 4), "maks": round(float(r.max()), 4),
                   "min": round(float(r.min()), 4), "n_gt5": int((r > 5).sum()), "n_gt15": int((r > RMSE_UC).sum()),
                   "kanal_ort_fark_RGB": [round(float(x), 3) for x in np.mean(kanal[g], 0)]}
    uc = sorted(bilgi, key=lambda b: -b["rmse_hat"])[:15]
    R["hazirla"] = {"n": len(idx), "eksik": eksik, "geo": geo, "yon": yon, "q": int(q), "subsampling": int(ss),
                    "pillow": PIL.__version__, "numpy": np.__version__, "boyut_dagilim": dict(boyut), "rmse": ozet,
                    "rmse_uc_15": uc, "lisans": dict(lisans), "sure_sn": round(time.time() - t0, 1),
                    "kalibrasyon_onisleme_sha256": sha(KAL_JSON), "indir_bcn20000_sha256": sha(WS / "scripts" / "indir_bcn20000.py")}
    if tani:
        uy = [t for t in tani if t["isic"] != t["val"]]
        R["hazirla"]["tani"] = {"n": len(tani), "uyusmaz": len(uy), "oran": round(len(uy) / len(tani), 4),
                                "eslenemedi": sum(1 for t in tani if t["isic"] is None),
                                "tablo": {f"{a}|{b}": c for (a, b), c in Counter((t["d3"], t["val"]) for t in tani).most_common()},
                                "uyusmaz_ornek": uy[:20]}
    R["_npz"] = (idx_a, ids_a)
    json_yaz(cikti / "e3_hat_goruntu_kaydi.json", bilgi)
    log(f"hazırla: n {len(idx)} · eksik {len(eksik)} · RMSE hat medyan {ozet['hat']['medyan']} maks {ozet['hat']['maks']} · "
        f"jpeg {ozet['jpeg']['medyan']} · z125 {ozet['z125']['medyan']} · z080 {ozet['z080']['medyan']} · {R['hazirla']['sure_sn']}s")
    return ozet


def kapilar_hazirla(R: dict):
    o = R["hazirla"]["rmse"]["hat"]
    if not (RMSE_MEDYAN[0] <= o["medyan"] <= RMSE_MEDYAN[1]):
        dur(f"K-b: RMSE medyanı {o['medyan']} ∉ [{RMSE_MEDYAN[0]}; {RMSE_MEDYAN[1]}] (hiza/eşleme şüphesi)", R)
    if o["n_gt15"] > 0:
        dur(f"K-b: RMSE > {RMSE_UC} olan görüntü {o['n_gt15']} > 0 (hiza/eşleme şüphesi)", R)
    t = R["hazirla"].get("tani")
    if t is None or t["oran"] > TANI_UYUSMAZ_MAKS:
        dur(f"K-b: ISIC meta tanısı ↔ val_labels uyuşmazlığı {t and t['oran']} > {TANI_UYUSMAZ_MAKS}", R)
    R["K-b"] = {"gecti": True, "rmse_medyan": o["medyan"], "rmse_gt15": 0, "tani_uyusmaz_oran": t["oran"],
                "eksik": len(R["hazirla"]["eksik"])}


# ───────────────────────── beklenen ─────────────────────────
def beklenen(R: dict, cikti: Path, hedef: Path, rapor_dizin: Path):
    B = {"_aciklama": "H16 (E3) a0 donmuş veri beklentisi (scripts/e3_ham_val_hazirla.py). e3_cikarim.py ve e3_onisleme_analiz.py "
                      "bu dosyayla doğrular; uyuşmazlık → DUR. Model çıktısına bakılmadan yazıldı; koşum sonrası DEĞİŞTİRİLMEZ.",
         "zaman": datetime.now().isoformat(timespec="seconds"), "kart_sha256": sha(KART) if KART.exists() else None,
         "betik_sha256": sha(Path(__file__)), "n": int(R["hazirla"]["n"]), "n_val": N_VAL,
         "eksik": R["hazirla"]["eksik"] + [x for x in R.get("indir", {}).get("eksik", []) if x not in R["hazirla"]["eksik"]],
         "dosyalar": {}, "icerik": {}}
    idx_a, ids_a = R["_npz"]
    for g in HATLAR:
        p = cikti / NPZ_AD[g]
        B["dosyalar"][NPZ_AD[g]] = {"sha256": sha(p), "bayt": p.stat().st_size, "hat": g}
        with np.load(p, allow_pickle=False) as z:
            B["icerik"][g] = BV.icerik_sha256(z["images"])
            if not (np.array_equal(z["index"], idx_a) and np.array_equal(z["image_id"], ids_a)):
                dur(f"{NPZ_AD[g]} index/image_id sırası diğer hatlarla aynı değil", R)
    B["icerik"]["index"] = BV.icerik_sha256(idx_a)
    B["sira_sha256"] = BV.sira_sha256(ids_a.tolist())
    B["index_artan"] = bool(np.all(np.diff(idx_a) > 0))
    B.update({"geo": R["hazirla"]["geo"], "yon": R["hazirla"]["yon"], "q_BCN": R["hazirla"]["q"],
              "subsampling": R["hazirla"]["subsampling"], "pillow": R["hazirla"]["pillow"], "numpy": R["hazirla"]["numpy"],
              "qbcn": {k: v for k, v in R.get("qbcn", {}).items() if k != "hatalar"},
              "rmse": R["hazirla"]["rmse"], "tani": {k: v for k, v in (R["hazirla"].get("tani") or {}).items() if k != "uyusmaz_ornek"},
              "lisans": R["hazirla"]["lisans"], "boyut_dagilim": R["hazirla"]["boyut_dagilim"],
              "liste": R.get("liste"), "lezyon_N": R.get("_N_lezyon"),
              "kapilar": {"K-a": {"test_kesisim": 0, "test_okundu": R.get("test_okundu", 0)}, "K-b": R.get("K-b")},
              "kaynak_sha256": {"kalibrasyon_onisleme": R["hazirla"]["kalibrasyon_onisleme_sha256"],
                                "indir_bcn20000": R["hazirla"]["indir_bcn20000_sha256"], "esleme_csv": sha(ESLEME_CSV),
                                "dermamnist_224_npz_json": None},
              "dizin": str(cikti.relative_to(WS)) if cikti.is_relative_to(WS) else str(cikti)})
    if hedef.exists():
        eski = json.loads(hedef.read_text(encoding="utf-8"))
        if eski.get("dosyalar") != B["dosyalar"] or eski.get("icerik") != B["icerik"]:
            log(f"DUR: {hedef.name} zaten var ve içerik farklı — dondurulmuş dosya değiştirilmez (rc 3)")
            raise SystemExit(3)
        log(f"{hedef.name} zaten var ve aynı içerik — yeniden yazılmadı")
    else:
        json_yaz(hedef, B)
    q = R.get("qbcn", {})
    md = ["# H16 / E3 — a0 veri kapısı raporu", f"*{B['zaman']}* · `scripts/e3_ham_val_hazirla.py` · beklenen `{hedef.name}` "
          f"SHA256 `{sha(hedef)[:16]}…`", "",
          f"- **K-a:** indirme listesi yalnız `split == \"val\"`; test image_id kesişimi **0**; test anahtarı okunmadı.",
          f"- **K-b:** n = **{B['n']}** / {N_VAL} · eksik {len(B['eksik'])} {B['eksik'][:10]} · RMSE (hat ↔ MedMNIST+ val) medyan "
          f"**{B['rmse']['hat']['medyan']}** (kapı [1,0; 3,0]), %95 {B['rmse']['hat']['p95']}, maks {B['rmse']['hat']['maks']}, "
          f"> 15: {B['rmse']['hat']['n_gt15']} · ISIC tanı ↔ val_labels uyuşmazlığı {B['tani'].get('oran')} (kapı ≤ 0,02; "
          f"eşlenemeyen {B['tani'].get('eslenemedi')})",
          f"- **Hat:** `{B['geo']}|{B['yon']}` (kalibrasyon_onisleme.json; BCN hattıyla aynı) · Pillow {B['pillow']} · boyutlar "
          f"{B['boyut_dagilim']}",
          f"- **q_BCN = {B['q_BCN']}** (kaynak {q.get('kaynak', {}).get('tur')}, n {q.get('n')}, medyan {q.get('q_medyan_ham')}, "
          f"tam eşleşme {q.get('tam_eslesme_orani')}, dağılım {q.get('q_dagilim')}); subsampling {B['subsampling']}",
          f"- **Lisans:** {B['lisans']} — `data/ham_val_isic/` depoya girmez (K-h).",
          f"- **N/S:** lezyon N {B['liste']['lezyon_N']} / S {B['liste']['lezyon_S']} · görüntü N {B['liste']['goruntu_N']}",
          "", "| hat | RMSE medyan | ort | %95 | maks | > 5 | kanal ort. farkı (R, G, B) |", "|---|---|---|---|---|---|---|"]
    md += [f"| {g} | {v['medyan']} | {v['ort']} | {v['p95']} | {v['maks']} | {v['n_gt5']} | {v['kanal_ort_fark_RGB']} |"
           for g, v in B["rmse"].items()]
    md += ["", "| dosya | bayt | SHA256 |", "|---|---|---|"]
    md += [f"| `{k}` | {v['bayt']:,} | `{v['sha256']}` |" for k, v in B["dosyalar"].items()]
    md += ["", f"Sıra SHA256 `{B['sira_sha256']}` · içerik SHA256: " + ", ".join(f"{k} `{v[:16]}…`" for k, v in B["icerik"].items())]
    rapor_dizin.mkdir(parents=True, exist_ok=True)
    (rapor_dizin / "e3_a0_rapor.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    json_yaz(rapor_dizin / "e3_a0_rapor.json", {k: v for k, v in R.items() if not k.startswith("_")})
    log(f"beklenen: {hedef} · rapor {rapor_dizin / 'e3_a0_rapor.md'}")
    return B


# ───────────────────────── sınama (8 HAM TRAIN kalibrasyon görüntüsü) ─────────────────────────
def sinama(a) -> int:
    d = Path(a.sinama)
    d.mkdir(parents=True, exist_ok=True)
    R = {"zaman": datetime.now().isoformat(timespec="seconds"), "sinama": True}
    kl = json.loads((KAL_HAM / "kalibrasyon_listesi.json").read_text(encoding="utf-8"))["goruntuler"]
    assert all(g["split"] == "train" for g in kl)
    kayit = [(int(g["index"]), g["image_id"]) for g in kl]
    with np.load(NPZ224, allow_pickle=False) as z:
        ref = z["train_images"][[k for k, _ in kayit]]             # yalnız sınama: 8 TRAIN görüntüsü
    q, ss = Q_VARSAYILAN, 2
    if a.qbcn_n:
        o = qbcn(R, a.qbcn_n, a.isci, d, uzak_zorla=a.qbcn_uzak)
        q, ss = o["q_BCN"], o.get("subsampling", 2)
    hazirla(R, kayit, KAL_HAM, ref, d, q, ss)
    with np.load(d / NPZ_AD["hat"], allow_pickle=False) as z:
        Xh = z["images"]
    with np.load(E2_KAL_NPZ, allow_pickle=False) as z:
        Xe2, ie2 = z["images"], z["idx"]
    B2 = json.loads(E2_BEKLENEN.read_text(encoding="utf-8"))
    ayni = bool(np.array_equal(Xh, Xe2) and ie2.tolist() == [k for k, _ in kayit])
    R["sinama_sonuc"] = {"birincil_hat_e2_kal_ile_bayt_ayni": ayni,
                         "icerik_sha_e2_beklenen": BV.icerik_sha256(Xh) == B2["icerik"]["kal_images"],
                         "rmse_hat_ort": R["hazirla"]["rmse"]["hat"]["ort"], "kalibrasyon_beklenen_rmse_ort": 1.76,
                         "q": q, "subsampling": ss}
    json_yaz(d / "sinama_sonuc.json", {k: v for k, v in R.items() if not k.startswith("_")})
    log(f"SINAMA: birincil hat ↔ E2 bcn_e2_kal_224 bayt bayt {'AYNI' if ayni else 'FARKLI'} · içerik SHA "
        f"{R['sinama_sonuc']['icerik_sha_e2_beklenen']} · RMSE ort {R['sinama_sonuc']['rmse_hat_ort']} (kalibrasyon 1,76)")
    return 0 if ayni else 2


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--asama", choices=["liste", "indir", "qbcn", "hazirla", "hepsi"], default="hepsi")
    p.add_argument("--isci", type=int, default=6, help="indirme / Range iş parçacığı")
    p.add_argument("--qbcn-n", type=int, default=500)
    p.add_argument("--qbcn-uzak", action="store_true", help="yalnız sınama: yerel zip olsa da HTTP Range yolunu kullan")
    p.add_argument("--sinama", default=None, help="8 HAM TRAIN kalibrasyon görüntüsüyle kod sınaması (çıktı dizini)")
    p.add_argument("--metrics", default=None)
    a = p.parse_args()
    t0 = time.time()
    if a.sinama:
        if a.qbcn_n == 500:
            a.qbcn_n = 0
        return sinama(a)
    R = {"zaman": datetime.now().isoformat(timespec="seconds"), "betik_sha256": sha(Path(__file__)),
         "kart_sha256": sha(KART) if KART.exists() else None}
    ara = HEDEF / "e3_a0_ara.json"
    if a.asama != "hepsi" and ara.exists():
        R = {**json.loads(ara.read_text(encoding="utf-8")), **R}
    rc = 0
    try:
        val = liste(R)
        if a.asama in ("indir", "hepsi"):
            indir(R, val, HEDEF, a.isci)
        if a.asama in ("qbcn", "hepsi"):
            qbcn(R, a.qbcn_n, a.isci, HEDEF)
        if a.asama in ("hazirla", "hepsi"):
            if "qbcn" not in R:
                dur("hazırla: q_BCN yok (önce --asama qbcn)", R)
            V = ValNpz()
            Xv, yv = V["val_images"], V["val_labels"].reshape(-1)
            kayit = [(int(r["index"]), r["image_id"]) for r in val]
            ref = Xv[[k for k, _ in kayit]]
            hazirla(R, kayit, HEDEF, ref, HEDEF, R["qbcn"]["q_BCN"], R["qbcn"].get("subsampling", 2), etiket_val=yv)
            R["test_okundu"] = V.test_okundu()
            if R["test_okundu"]:
                dur("K-a: test anahtarı okundu", R)
            kapilar_hazirla(R)
            beklenen(R, HEDEF, BEKLENEN, HEDEF)
    except SystemExit as e:
        rc = int(e.code) if isinstance(e.code, int) else 2
    finally:
        json_yaz(ara, {k: v for k, v in R.items() if k != "_npz"})
    if a.metrics:
        h = R.get("hazirla", {})
        m = {"asama": a.asama, "gecti": int(rc == 0), "sure_sn": round(time.time() - t0, 1), "n_val": R.get("liste", {}).get("n_val", 0),
             "test_kesisim": 0 if "liste" in R else -1, "test_okundu": R.get("test_okundu", 0),
             "eksik": len(R.get("indir", {}).get("eksik", [])), "hat_n": h.get("n", 0),
             "q_bcn": R.get("qbcn", {}).get("q_BCN", -1), "q_bcn_n": R.get("qbcn", {}).get("n", 0),
             "rmse_hat_medyan": (h.get("rmse") or {}).get("hat", {}).get("medyan", -1.0),
             "rmse_hat_maks": (h.get("rmse") or {}).get("hat", {}).get("maks", -1.0),
             "tani_uyusmaz_oran": (h.get("tani") or {}).get("oran", -1.0)}
        Path(a.metrics).write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"a0 {a.asama} {'GEÇTİ' if rc == 0 else f'DUR rc {rc}'} ({time.time() - t0:.0f}s)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
