#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
e2/bcn_veri.py — H15 (E2) BCN20000 MODEL GİRDİSİ: ETİKETSİZ dosyalar, SHA256 denetimi ve Colab'da açma (npz → npy).

Neden ayrı, etiketsiz girdi: `scripts/indir_bcn20000.py --asama onisleme` çıktısı (`bcn20000_train_merkez_kare_224.npz`)
`diagnosis` ve `lesion_id` anahtarlarını da taşır. Çıkarım tarafına etiket hiç gitmesin diye a0 (`scripts/e2_bcn_hazirla.py`)
yalnız piksel + dosya adı içeren türev dosyalar yazar (kart a1: "betik diagnosis alanını okumaz"):
  bcn_e2_merkez_kare_224.npz   images (12.413, 224, 224, 3) uint8 · bcn_filename (12.413,) — ANA geometri
  bcn_e2_cerceve_fark_224.npz  idx (k,) int64 · images (k, 224, 224, 3) — yalnız yazarların kırpıcısının DEĞİŞTİRDİĞİ satırlar;
                               kırpılmayan satırlar merkez_kare ile bayt bayt aynıdır (a0 bunu sınar). Açılınca tam dizi kurulur.
  bcn_e2_kal_224.npz           images (8, 224, 224, 3) · idx (8,) — 8 HAM kalibrasyon görüntüsü, BCN ön-işleme yoluyla
                               (DermaMNIST TRAIN indeksleri; a1 ön-işleme hattı denetimi)
Bütün dosya SHA256'ları ve dizi içerik SHA256'ları `kuyruk/e2_bcn_beklenen.json`'da donar (a0). Colab'da npz bir kez
`acik/` altına düz .npy olarak açılır (her satır bellek eşlemeli okur; her satır içerik SHA256'sını yeniden doğrular).

İçerik SHA256 tanımı (dosya zaman damgasından bağımsız): sha256(dtype + "|" + şekil + "|" + C-sıralı baytlar).

Kullanım (WS kökü):
  python varyantlar/e2/bcn_veri.py ac      [--dizin data/bcn20000/e2_girdi] [--beklenen kuyruk/e2_bcn_beklenen.json]
  python varyantlar/e2/bcn_veri.py dogrula [...]     (dosya SHA256; çıkış 0/2)
  python varyantlar/e2/bcn_veri.py kapi --out {RUN_DIR}/metrics.json   (a1 VERİ KAPISI satırı: K-a nn_tam = 0, K-b sayılar,
        dosya + açık .npy içerik SHA256'ları, n = 12.413; metrik anahtarları `kapi_*`; çıkış 0 GEÇTİ / 2 DUR)
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
DIZIN = WS / "data" / "bcn20000" / "e2_girdi"
BEKLENEN = WS / "kuyruk" / "e2_bcn_beklenen.json"
GEOMETRILER = ("merkez_kare", "cerceve")
N_BCN = 12413
NPZ = {"merkez_kare": "bcn_e2_merkez_kare_224.npz", "cerceve_fark": "bcn_e2_cerceve_fark_224.npz", "kal": "bcn_e2_kal_224.npz"}
ETIKET_ANAHTARLARI = {"diagnosis", "lesion_id", "dx", "label", "labels", "etiket"}


def dosya_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for parca in iter(lambda: f.read(1 << 22), b""):
            h.update(parca)
    return h.hexdigest()


def icerik_sha256(a: np.ndarray, satir_parca: int = 512) -> str:
    """dtype + şekil + C-sıralı baytlar (bellek eşlemeli dizide parça parça; tam dizi kopyalanmaz)."""
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(b"|"); h.update(str(tuple(a.shape)).encode()); h.update(b"|")
    if a.ndim == 0 or a.shape[0] == 0:
        h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()
    for i in range(0, a.shape[0], satir_parca):
        h.update(np.ascontiguousarray(a[i:i + satir_parca]).tobytes())
    return h.hexdigest()


def sira_sha256(adlar) -> str:
    return hashlib.sha256("\n".join(str(x) for x in adlar).encode("utf-8")).hexdigest()


def beklenen_oku(yol: Path = BEKLENEN) -> dict:
    if not Path(yol).exists():
        raise SystemExit(f"HATA: {yol} yok — a0 (scripts/e2_bcn_hazirla.py) yazmadan E2 çıkarımı yapılmaz (K-b)")
    return json.loads(Path(yol).read_text(encoding="utf-8"))


def acik_dizin(dizin: Path) -> Path:
    return Path(dizin) / "acik"


def npy_yolu(dizin: Path, geo: str) -> Path:
    return acik_dizin(dizin) / f"bcn_e2_{geo}_224.npy"


def sira_yolu(dizin: Path) -> Path:
    return acik_dizin(dizin) / "bcn_e2_sira.npy"


def _etiketsiz(z, ad: str):
    fazla = ETIKET_ANAHTARLARI & set(z.files)
    if fazla:
        raise SystemExit(f"HATA: {ad} etiket anahtarı taşıyor {sorted(fazla)} — E2 model girdisi etiketsiz olmalı (K-h/a1)")


def dosyalari_dogrula(dizin: Path, B: dict) -> dict:
    """Taşınan npz dosyalarının dosya SHA256'sı = beklenen (Drive → Colab kopyası bütün mü)."""
    out = {}
    for ad, bilgi in B["dosyalar"].items():
        p = Path(dizin) / ad
        if not p.exists():
            raise SystemExit(f"HATA: {p} yok")
        s = dosya_sha256(p)
        out[ad] = {"sha256": s, "eslesti": s == bilgi["sha256"], "bayt": p.stat().st_size}
        if s != bilgi["sha256"]:
            raise SystemExit(f"HATA (K-b): {ad} SHA256 {s[:16]}… ≠ beklenen {bilgi['sha256'][:16]}…")
    return out


def ac(dizin: Path = DIZIN, beklenen: Path = BEKLENEN, log=print) -> dict:
    """npz → acik/*.npy (idempotent: içerik SHA256 tutuyorsa yeniden yazmaz). Dönüş: özet."""
    dizin = Path(dizin)
    B = beklenen_oku(beklenen)
    ozet = {"dosyalar": dosyalari_dogrula(dizin, B)}
    acik_dizin(dizin).mkdir(parents=True, exist_ok=True)
    hedef_mk, hedef_cv, hedef_sira = npy_yolu(dizin, "merkez_kare"), npy_yolu(dizin, "cerceve"), sira_yolu(dizin)

    def tamam(p, anahtar):
        if not p.exists():
            return False
        a = np.load(p, mmap_mode="r", allow_pickle=False)
        return icerik_sha256(a) == B["icerik"][anahtar]

    if tamam(hedef_mk, "merkez_kare") and tamam(hedef_cv, "cerceve") and tamam(hedef_sira, "sira"):
        log("[bcn_veri] açık .npy dosyaları zaten var ve içerik SHA256 tutuyor")
        ozet["acildi"] = False
        return ozet
    with np.load(dizin / NPZ["merkez_kare"], allow_pickle=False) as z:
        _etiketsiz(z, NPZ["merkez_kare"])
        X = z["images"]
        adlar = z["bcn_filename"]
    if len(X) != B["n"] or X.shape[1:] != (224, 224, 3) or X.dtype != np.uint8:
        raise SystemExit(f"HATA (K-b): merkez_kare şekli {X.shape} {X.dtype} ≠ ({B['n']}, 224, 224, 3) uint8")
    with np.load(dizin / NPZ["cerceve_fark"], allow_pickle=False) as z:
        _etiketsiz(z, NPZ["cerceve_fark"])
        idx, Xf = z["idx"].astype(np.int64), z["images"]
    for ad, a, anahtar in (("merkez_kare", X, "merkez_kare"), ("sira", adlar, "sira")):
        s = icerik_sha256(a)
        if s != B["icerik"][anahtar]:
            raise SystemExit(f"HATA (K-b): {ad} içerik SHA256 {s[:16]}… ≠ beklenen")
    if sira_sha256(adlar.tolist()) != B["sira_sha256"]:
        raise SystemExit("HATA (K-b): bcn_filename sırası beklenenle aynı değil")
    for p, a in ((hedef_mk, X), (hedef_sira, adlar)):
        tmp = p.with_name(p.stem + ".tmp.npy")
        np.save(tmp, a, allow_pickle=False)
        os.replace(tmp, p)
    C = X.copy()
    C[idx] = Xf
    del X
    s = icerik_sha256(C)
    if s != B["icerik"]["cerceve"]:
        raise SystemExit(f"HATA (K-b): cerceve içerik SHA256 {s[:16]}… ≠ beklenen")
    tmp = hedef_cv.with_name(hedef_cv.stem + ".tmp.npy")
    np.save(tmp, C, allow_pickle=False)
    os.replace(tmp, hedef_cv)
    ozet.update({"acildi": True, "n": int(len(C)), "cerceve_degisen": int(len(idx))})
    log(f"[bcn_veri] açıldı: {acik_dizin(dizin)} (n={len(C)}, çerçeve kırpılan {len(idx)})")
    return ozet


def geometri_oku(dizin: Path, geo: str, B: dict, ilk: int | None = None):
    """(bellek eşlemeli images, bcn_filename listesi, içerik_sha_eslesti). ilk=N → yalnız ilk N satır (a1 duman)."""
    if geo not in GEOMETRILER:
        raise SystemExit(f"HATA: geometri {geo!r} ∉ {GEOMETRILER}")
    X = np.load(npy_yolu(dizin, geo), mmap_mode="r", allow_pickle=False)
    adlar = np.load(sira_yolu(dizin), allow_pickle=False)
    if ilk is not None:
        return X[:ilk], adlar[:ilk].tolist(), None   # duman: tam içerik denetimi a1 veri kapısında
    ok = (icerik_sha256(X) == B["icerik"][geo]) and (icerik_sha256(adlar) == B["icerik"]["sira"]) \
        and sira_sha256(adlar.tolist()) == B["sira_sha256"]
    return X, adlar.tolist(), bool(ok)


def kal_oku(dizin: Path, B: dict):
    with np.load(Path(dizin) / NPZ["kal"], allow_pickle=False) as z:
        _etiketsiz(z, NPZ["kal"])
        X, idx = z["images"], z["idx"].astype(np.int64)
    ok = icerik_sha256(X) == B["icerik"]["kal_images"] and idx.tolist() == B["kal_idx"]
    return X, idx, bool(ok)


def kapi(dizin: Path, beklenen: Path) -> dict:
    """a1 veri kapısı (CPU): a0'ın donmuş kapı sonuçları + Colab'daki kopyanın bütünlüğü. Etiket OKUNMAZ."""
    B = beklenen_oku(beklenen)
    ka, kb = (B.get("kapilar") or {}).get("K-a") or {}, (B.get("kapilar") or {}).get("K-b") or {}
    s = (B.get("sayilar") or {}).get("E1|kopyasiz") or {}
    m = {"kapi_ka_gecti": int(bool(ka.get("gecti")) and int(ka.get("esik5_alti", -1)) == 0),
         "kapi_nn_esik5_alti": int(ka.get("esik5_alti", -1)), "kapi_nn_esik10_alti_bilgi": int(ka.get("esik10_alti_bilgi", -1)),
         "kapi_kb_gecti": int(bool(kb.get("gecti")) and s.get("goruntu") == 11933 and s.get("lezyon") == 3451),
         "kapi_n": int(B.get("n", -1))}
    try:
        dosyalari_dogrula(dizin, B)
        m["kapi_dosya_sha_eslesti"] = 1
    except SystemExit as e:
        print(e)
        m["kapi_dosya_sha_eslesti"] = 0
    ok = 1
    for geo in GEOMETRILER:
        p = npy_yolu(dizin, geo)
        if not p.exists():
            ok = 0
            continue
        X = np.load(p, mmap_mode="r", allow_pickle=False)
        ok &= int(icerik_sha256(X) == B["icerik"][geo] and len(X) == B["n"])
    ad = np.load(sira_yolu(dizin), allow_pickle=False) if sira_yolu(dizin).exists() else np.array([])
    ok &= int(icerik_sha256(ad) == B["icerik"]["sira"] and sira_sha256(ad.tolist()) == B["sira_sha256"])
    m["kapi_acik_icerik_eslesti"] = ok
    m["kapi_kal_eslesti"] = int(kal_oku(dizin, B)[2])
    m["kapi_cerceve_orani"] = float((B.get("cerceve") or {}).get("oran", float("nan")))
    m["kapi_gecti"] = int(all(m[k] for k in ("kapi_ka_gecti", "kapi_kb_gecti", "kapi_dosya_sha_eslesti", "kapi_acik_icerik_eslesti",
                                            "kapi_kal_eslesti")) and m["kapi_n"] == N_BCN and m["kapi_nn_esik5_alti"] == 0)
    return m


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("islem", choices=["ac", "dogrula", "kapi"])
    p.add_argument("--dizin", default=str(DIZIN))
    p.add_argument("--beklenen", default=str(BEKLENEN))
    p.add_argument("--out", default=None, help="kapi: {RUN_DIR}/metrics.json")
    a = p.parse_args()
    if a.islem == "kapi":
        m = kapi(Path(a.dizin), Path(a.beklenen))
        if a.out:
            Path(a.out).write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(m, ensure_ascii=False, indent=1))
        print("VERİ KAPISI " + ("GEÇTİ" if m["kapi_gecti"] else "DUR (K-a/K-b) — E2 başlamaz"))
        return 0 if m["kapi_gecti"] else 2
    if a.islem == "ac":
        print(json.dumps(ac(Path(a.dizin), Path(a.beklenen)), ensure_ascii=False, indent=1))
        return 0
    B = beklenen_oku(Path(a.beklenen))
    print(json.dumps(dosyalari_dogrula(Path(a.dizin), B), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
