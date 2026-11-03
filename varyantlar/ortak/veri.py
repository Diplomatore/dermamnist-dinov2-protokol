# -*- coding: utf-8 -*-
"""
ortak/veri.py — DermaMNIST(+) npz yükleyici (28/64/128/224), MD5 doğrulaması, lezyon sızıntı maskesi.

Kurallar:
- Yalnız resmî split (V3). Kök: WS/data/medmnist. İndirme YOK (download=False eşdeğeri).
- TEST KİLİDİ (M4): `yukle(..., bolumler=...)` "test" isterse `test_izin=True` olmadan hata verir.
  K4 varyantları test görüntüsü/etiketi okumaz; test yalnız K5'te `-test` son ekli varyantlarda.
- Etiketler (N,1) uint8 → (N,) int64.
"""
import csv
import hashlib
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
VERI_KOK = WS / "data" / "medmnist"
LEZYON_CSV = WS / "data" / "splits" / "dermamnist_lezyon_esleme.csv"
HAM_META_CSV = WS / "data" / "meta" / "HAM10000_metadata.csv"
DATA_FLAG = "dermamnist"
SINIF_SAYISI = 7
SINIF_KISA = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
BOYUTLAR = (28, 64, 128, 224)


class TestKilidiHatasi(RuntimeError):
    pass


def npz_adi(size: int) -> str:
    return f"{DATA_FLAG}{'' if size == 28 else '_' + str(size)}.npz"


def npz_yolu(size: int, kok: Path = VERI_KOK) -> Path:
    return Path(kok) / npz_adi(size)


def dosya_hash(yol: Path):
    h5, h256 = hashlib.md5(), hashlib.sha256()
    with open(yol, "rb") as f:
        for parca in iter(lambda: f.read(1 << 20), b""):
            h5.update(parca)
            h256.update(parca)
    return h5.hexdigest(), h256.hexdigest()


def resmi_md5(size: int) -> str:
    from medmnist import INFO
    return INFO[DATA_FLAG]["MD5" + ("" if size == 28 else f"_{size}")]


def npz_dogrula(size: int, kok: Path = VERI_KOK) -> dict:
    """MD5 resmî değerle eşleşmeli; değilse SystemExit(2). Dönüş: {npz, md5, md5_resmi, sha256, bayt}."""
    if size not in BOYUTLAR:
        raise SystemExit(f"HATA: desteklenmeyen boyut {size} (izinli {BOYUTLAR})")
    yol = npz_yolu(size, kok)
    if not yol.exists():
        raise SystemExit(f"HATA: veri yok: {yol} (indirme tamamlanmamış olabilir)")
    md5, sha256 = dosya_hash(yol)
    md5_r = resmi_md5(size)
    if md5 != md5_r:
        raise SystemExit(f"HATA: {yol.name} MD5 {md5} ≠ resmî {md5_r} (yarım/bozuk indirme?)")
    return {"npz": yol.name, "kok": str(Path(kok)), "md5": md5, "md5_resmi": md5_r, "sha256": sha256,
            "bayt": yol.stat().st_size}


def yukle(size: int, bolumler=("train", "val"), alt_kume: int | None = None, test_izin: bool = False,
          kok: Path = VERI_KOK) -> dict:
    """{bolum: {"x": uint8 (N,H,W,3), "y": int64 (N,), "idx": int64 (N,) resmî indeks}}.
    alt_kume=N → her bölümün İLK N örneği (duman/hata ayıklama; ölçüm değildir)."""
    if "test" in bolumler and not test_izin:
        raise TestKilidiHatasi("test bölümü K4'te okunamaz (M4). Yalnız K5 '-test' varyantlarında test_izin=True.")
    yol = npz_yolu(size, kok)
    out = {}
    with np.load(yol) as z:
        for b in bolumler:
            x = z[f"{b}_images"]
            y = z[f"{b}_labels"].reshape(-1).astype(np.int64)
            idx = np.arange(len(y), dtype=np.int64)
            if alt_kume:
                x, y, idx = x[:alt_kume], y[:alt_kume], idx[:alt_kume]
            if x.ndim == 3:  # gri ise 3 kanala (DermaMNIST zaten RGB)
                x = np.repeat(x[..., None], 3, axis=-1)
            out[b] = {"x": np.ascontiguousarray(x), "y": y, "idx": idx}
    return out


def etiket_tutarliligi(size: int, bolumler=("train", "val"), kok: Path = VERI_KOK) -> bool:
    """H03 operasyonel DUR koşulu: etiket dizileri 28 px resmî dosyayla bayt bayt aynı olmalı.
    K4'te yalnız train/val etiketleri karşılaştırılır (test etiketleri K5'te)."""
    if size == 28:
        return True
    with np.load(npz_yolu(28, kok)) as a, np.load(npz_yolu(size, kok)) as b:
        return all(np.array_equal(a[f"{s}_labels"], b[f"{s}_labels"]) for s in bolumler)


def sizinti_maskesi(bolum: str = "val") -> np.ndarray:
    """bool (N_bolum,), resmî indeks sırasıyla: True = görüntünün lezyonu (lesion_id) train'de de var (S kümesi).
    Kaynak: data/splits/dermamnist_lezyon_esleme.csv (Abhishek vd. 2025 eşlemesi + HAM10000 metadata)."""
    if bolum == "test":
        raise TestKilidiHatasi("test sızıntı alt kümesi yalnız K5'te")
    tr_lez, satirlar = set(), []
    with open(LEZYON_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "train":
                tr_lez.add(r["lesion_id"])
            elif r["split"] == bolum:
                satirlar.append((int(r["index"]), r["lesion_id"]))
    n = max(i for i, _ in satirlar) + 1
    m = np.zeros(n, dtype=bool)
    for i, lez in satirlar:
        m[i] = lez in tr_lez
    return m


_VAL_META = None


def val_meta() -> dict:
    """Resmî val indeks sırasıyla (N_val,) diziler — scripts/hata_kirilim.py (K3.5, tahlilci) tanımlarının AYNISI:
      "S"       bool: lezyonu (lesion_id) train'de de var,
      "dx_type" str : HAM10000_metadata.csv dx_type (histo / follow_up / consensus / confocal),
      "kaynak"  str : HAM10000_metadata.csv dataset (vidir_molemax / vidir_modern / rosendahl / vienna_dias).
    Eşleme: dermamnist_lezyon_esleme.csv (split, index, image_id) → HAM10000 image_id. Test için ÇAĞRILMAZ (M4)."""
    global _VAL_META
    if _VAL_META is None:
        meta = {}
        with open(HAM_META_CSV, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                meta[r["image_id"]] = r
        vimg = {}
        with open(LEZYON_CSV, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["split"] == "val":
                    vimg[int(r["index"])] = r["image_id"]
        n = max(vimg) + 1
        _VAL_META = {"S": sizinti_maskesi("val"),
                     "dx_type": np.array([meta[vimg[i]]["dx_type"] for i in range(n)]),
                     "kaynak": np.array([meta[vimg[i]]["dataset"] for i in range(n)])}
    return _VAL_META
