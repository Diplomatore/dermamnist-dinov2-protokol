# -*- coding: utf-8 -*-
"""
k5/c_bolme.py — K5 C1/C2 VERİ YAMASI (K5-ONKAYIT §4.1, §4.2, §5). Varyant kodu DEĞİŞMEZ; `ortak.veri` modülünün
yol/doğrulama/metadata fonksiyonları süreç içinde C bölmesine yönlendirilir ("tek fark veri dosyası ve bölmedir").

veri = "c1": resmî MedMNIST(+) pikselleri + C bölmesi → data/dermamnist_c_resmipiksel_{28,224}.npz
             (scripts/c_bolme_npz.py yapar; yanındaki .json meta dosya/içerik SHA256'sını taşır; içerik SHA256'sı
             kuyruk/k5_c_bolme_beklenen.json ile — şef test geçişinden SONRA yazar — aynı olmalı, değilse DUR).
veri = "c2": C'nin yayımlanmış npz'si → data/dermamnist_c/dermamnist_corrected_{28,224}.npz (Zenodo 12739457 MD5).
Yamalanan (ortak.veri): npz_yolu, npz_dogrula, resmi_md5, etiket_tutarliligi, sizinti_maskesi, val_meta.
  - yukle() değişmez (modül içi npz_yolu çağrısı yamalı olanı bulur); test kilidi (test_izin) AYNEN geçerli.
  - val_meta(): C-val (573) sırasıyla S (lezyonu C-train'de; C lezyon-ayrık → hepsi False), dx_type, kaynak.
  - Öznitelik/token önbellekleri ayrı dizine: DERMA_OZN_DIZIN = data/oznitelik_<veri> (resmî önbellekle karışmaz) —
    bu yüzden uygula() dinov2_ozn / dinov2_ft modülleri İÇE AKTARILMADAN ÖNCE çağrılmalıdır.
Kullanım: from k5 import c_bolme; c_bolme.uygula("c1")   (ya da varyantlar/k5/c_bolme_sar.py ile bir betiği sararak)
"""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
VARYANTLAR = WS / "varyantlar"
C_ESLEME = WS / "data" / "splits" / "dermamnist_c_esleme.csv"
HAM_META = WS / "data" / "meta" / "HAM10000_metadata.csv"
BEKLENEN = WS / "kuyruk" / "k5_c_bolme_beklenen.json"
# Zenodo 12739457 resmî MD5 (colab/hat/veri_listesi.json ile aynı)
C2_MD5 = {28: "96e2862980877d45d9cacb5c25de6950", 224: "84920fb70c83b234c295b6f0d4ae2bc0"}
VERILER = ("c1", "c2")


def npz_yolu_c(veri: str, size: int) -> Path:
    if veri == "c1":
        return WS / "data" / f"dermamnist_c_resmipiksel_{size}.npz"
    if veri == "c2":
        return WS / "data" / "dermamnist_c" / f"dermamnist_corrected_{size}.npz"
    raise ValueError(veri)


def dosya_hash(yol: Path):
    h5, h256 = hashlib.md5(), hashlib.sha256()
    with open(yol, "rb") as f:
        for parca in iter(lambda: f.read(1 << 22), b""):
            h5.update(parca)
            h256.update(parca)
    return h5.hexdigest(), h256.hexdigest()


def icerik_sha256(z) -> str:
    """npz içeriğinin deterministik SHA256'sı (zip başlığındaki zaman damgasından bağımsız): anahtar sıralı;
    her dizi için ad, dtype, şekil ve C-sıralı baytlar."""
    h = hashlib.sha256()
    for k in sorted(z.files):
        a = np.ascontiguousarray(z[k])
        h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def c_satirlar(split_c: str) -> list:
    with open(C_ESLEME, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["split_C"] == split_c]


ALT_KUME_RNG = 20261011


def alt_kume_sec(y: np.ndarray, n: int, bolum: str) -> np.ndarray:
    """Sınıf-tabakalı, deterministik alt küme (yalnız duman): sınıf payı ∝ bölümdeki pay (en büyük kalan yöntemi; mevcut her
    sınıftan en az 1 örnek, n ≥ sınıf sayısıysa). RNG = default_rng(20261011 + bölüm kodu). Dönüş: artan sıralı indeksler."""
    y = np.asarray(y).reshape(-1)
    if n >= len(y):
        return np.arange(len(y))
    rng = np.random.default_rng(ALT_KUME_RNG + {"train": 0, "val": 1, "test": 2}[bolum])
    siniflar, say = np.unique(y, return_counts=True)
    hedef = n * say / len(y)
    k = np.floor(hedef).astype(int)
    if n >= len(siniflar):
        k = np.maximum(k, 1)
    while k.sum() > n:
        k[np.argmax(k)] -= 1
    kalan = hedef - np.floor(hedef)
    for i in np.argsort(-kalan):
        if k.sum() >= n:
            break
        if k[i] < say[i]:
            k[i] += 1
    sec = np.concatenate([rng.choice(np.where(y == c)[0], size=int(ki), replace=False) for c, ki in zip(siniflar, k) if ki > 0])
    return np.sort(sec)


_DOGRULANDI = {}


def dogrula(veri: str, size: int) -> dict:
    """npz_dogrula eşdeğeri. c1: dosya SHA256 = meta .json (aynı oturumda c_bolme_npz.py yazdı) ve meta'daki içerik
    SHA256'sı = kuyruk/k5_c_bolme_beklenen.json; c2: MD5 = Zenodo. Uyuşmazlık → SystemExit(2)."""
    anahtar = (veri, size)
    if anahtar in _DOGRULANDI:
        return _DOGRULANDI[anahtar]
    yol = npz_yolu_c(veri, size)
    if not yol.exists():
        raise SystemExit(f"HATA: C veri dosyası yok: {yol} (c1: önce scripts/c_bolme_npz.py; c2: DermaMNIST-C indirmesi)")
    md5, sha = dosya_hash(yol)
    bilgi = {"npz": yol.name, "kok": str(yol.parent), "md5": md5, "sha256": sha, "bayt": yol.stat().st_size, "veri": veri}
    if veri == "c2":
        if md5 != C2_MD5[size]:
            raise SystemExit(f"HATA: {yol.name} MD5 {md5} ≠ Zenodo {C2_MD5[size]}")
        bilgi["md5_resmi"] = C2_MD5[size]
    else:
        meta_yol = yol.with_suffix(".json")
        if not meta_yol.exists():
            raise SystemExit(f"HATA: {meta_yol} yok (c_bolme_npz.py denetimi geçmemiş)")
        meta = json.loads(meta_yol.read_text(encoding="utf-8"))
        if meta.get("dosya_sha256") != sha or not meta.get("denetim_gecti"):
            raise SystemExit(f"HATA: {yol.name} SHA256 meta ile uyuşmuyor ya da denetim geçmemiş")
        if not BEKLENEN.exists():
            raise SystemExit(f"HATA: {BEKLENEN} yok — C1 verisinin donmuş içerik SHA256'sı (şef, test geçişinden sonra)")
        bek = json.loads(BEKLENEN.read_text(encoding="utf-8"))
        if bek["icerik_sha256"].get(str(size)) != meta.get("icerik_sha256"):
            raise SystemExit(f"HATA: {yol.name} içerik SHA256 {meta.get('icerik_sha256')} ≠ donmuş "
                             f"{bek['icerik_sha256'].get(str(size))}")
        bilgi.update({"md5_resmi": md5, "icerik_sha256": meta["icerik_sha256"], "beklenen": str(BEKLENEN.name)})
    _DOGRULANDI[anahtar] = bilgi
    return bilgi


def uygula(veri: str, run_dir: Path | None = None) -> dict:
    if veri not in VERILER:
        raise SystemExit(f"HATA: bilinmeyen veri {veri!r} ({VERILER})")
    if "dinov2_ozn.ozn_cikar" in sys.modules or "dinov2_ft.ince_ayar_vit" in sys.modules:
        raise SystemExit("HATA: c_bolme.uygula() öznitelik modülleri içe aktarılmadan ÖNCE çağrılmalı (OZN_DIZIN)")
    oz = WS / "data" / f"oznitelik_{veri}"
    oz.mkdir(parents=True, exist_ok=True)
    os.environ["DERMA_OZN_DIZIN"] = str(oz)
    if str(VARYANTLAR) not in sys.path:
        sys.path.insert(0, str(VARYANTLAR))
    from ortak import veri as V

    def npz_yolu(size, kok=None):
        return npz_yolu_c(veri, size)

    def npz_dogrula(size, kok=None):
        return dogrula(veri, size)

    def resmi_md5(size):
        return dogrula(veri, size)["md5_resmi"]

    def etiket_tutarliligi(size, bolumler=("train", "val"), kok=None):
        if size == 28:
            return True
        with np.load(npz_yolu_c(veri, 28)) as a, np.load(npz_yolu_c(veri, size)) as b:
            return all(np.array_equal(a[f"{s}_labels"], b[f"{s}_labels"]) for s in bolumler)

    def sizinti_maskesi(bolum="val"):
        if bolum == "test":
            raise V.TestKilidiHatasi("test sızıntı alt kümesi yalnız K5 analizinde")
        tr = {r["lesion_id"] for r in c_satirlar("train")}
        return np.array([r["lesion_id"] in tr for r in c_satirlar(bolum)], dtype=bool)

    meta_onbellek = {}

    def val_meta():
        if not meta_onbellek:
            meta = {}
            with open(HAM_META, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    meta[r["image_id"]] = r
            sat = c_satirlar("val")
            meta_onbellek.update({"S": sizinti_maskesi("val"),
                                  "dx_type": np.array([meta[r["image_id"]]["dx_type"] for r in sat]),
                                  "kaynak": np.array([meta[r["image_id"]]["dataset"] for r in sat])})
        return meta_onbellek

    _yukle_ozgun = V.yukle

    def yukle(size, bolumler=("train", "val"), alt_kume=None, test_izin=False, kok=None):
        """alt_kume YOKSA özgün yukle() AYNEN. alt_kume=N (yalnız duman/hata ayıklama) → C npz sınıf sırasıyla dizili olduğundan
        (HAM10000 metadata sırası; yayımlanmış C npz ile aynı) "ilk N" tek sınıf verir → bunun yerine SINIF-TABAKALI, sabit
        RNG'li alt küme (alt_kume_sec). idx = seçilen gerçek indeksler (val_meta hizası korunur)."""
        D = _yukle_ozgun(size, bolumler, alt_kume=None, test_izin=test_izin)
        if alt_kume:
            for b in bolumler:
                sec = alt_kume_sec(D[b]["y"], alt_kume, b)
                D[b] = {"x": np.ascontiguousarray(D[b]["x"][sec]), "y": D[b]["y"][sec], "idx": D[b]["idx"][sec]}
        return D

    V.yukle = yukle
    V.npz_yolu, V.npz_dogrula, V.resmi_md5 = npz_yolu, npz_dogrula, resmi_md5
    V.etiket_tutarliligi, V.sizinti_maskesi, V.val_meta = etiket_tutarliligi, sizinti_maskesi, val_meta
    V._VAL_META = None
    V.VERI_KOK = npz_yolu_c(veri, 28).parent
    kayit = {"veri": veri, "aciklama": {"c1": "resmî MedMNIST(+) pikselleri + DermaMNIST-C bölmesi (K5-ONKAYIT §4.1)",
                                         "c2": "DermaMNIST-C yayımlanmış npz (Zenodo 12739457; K5-ONKAYIT §5)"}[veri],
             "npz_28": str(npz_yolu_c(veri, 28)), "npz_224": str(npz_yolu_c(veri, 224)), "ozn_dizin": str(oz),
             "c_esleme_sha256": hashlib.sha256(C_ESLEME.read_bytes()).hexdigest()}
    rd = run_dir or (Path(os.environ["RUN_DIR"]) if os.environ.get("RUN_DIR") else None)
    if rd is not None:
        Path(rd).mkdir(parents=True, exist_ok=True)
        (Path(rd) / "c_bolme.json").write_text(json.dumps(kayit, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[c_bolme] veri={veri} · ortak.veri yamalandı · DERMA_OZN_DIZIN={oz}", flush=True)
    return kayit
