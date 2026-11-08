# -*- coding: utf-8 -*-
"""
e2/cerceve.py — H15 S9 duyarlılığı: yazarların siyah çerçeve kırpıcısı (`ucuncu/image_cropper.py` @7ea35f5) + ANA 224 px yolu.

Kart (H15 §Veri kararları 3): "yazarların siyah çerçeve kırpıcısı (`image_cropper.py` @7ea35f5, cv2), ardından aynı 224 px yolu".
Yazarın `crop_files` döngüsünün görüntü başına mantığı (satır satır):
    img = cv2.imread(f, cv2.IMREAD_COLOR); gray = cv2.cvtColor(img, BGR2GRAY); _, thresh = cv2.threshold(gray, 60, 255, 0)
    try: x,y,w,h = obtain_bb(thresh); r = intensity_ratio(gray, x,y,w,h); if r > 1.1: img = img[y:y+h, x:x+w] (+ imwrite)
    except: (kırpma yok)
Burada:
  - `obtain_bb` modülden OLDUĞU GİBİ çağrılır (cv2.findContours; ≥100 px kenarlı SON dış kontur; yoksa UnboundLocalError →
    yazarın `except` dalı = kırpma yok).
  - `intensity_ratio` yerine VEKTÖREL EŞDEĞERİ kullanılır: cii = Σ(kırpık gri) / (yükseklik + genişlik), fii = Σ(gri) / (H + W),
    oran = cii / fii (yazarın paydaları aynen; alan değil, kenar toplamı). Gerekçe: (1) özgün işlev 1024² görüntüde iki Python
    döngüsüyle ≈ 2·10⁶ skaler toplama yapar (12.413 görüntüde saatler); (2) numpy ≥ 2'de (NEP 50) `0 + np.uint8` uint8 kalır ve
    toplam TAŞAR — özgün işlev numpy 2.2.6'da yanlış sonuç verir. Vektörel sürüm int64 toplar (numpy 1.x davranışı = yazarın
    kodunun yazıldığı dönemdeki anlamı). Eşdeğerlik: `es_degerlik_sina()` özgün işlevi int64 girdiyle çağırıp birebir karşılaştırır.
  - Pikseller ANA yolla aynı çözücüden gelir (PIL, `.convert("RGB")`); kırpma kutusu yazarın cv2 çözümlemesinden. Yazarın
    `cv2.imwrite` (JPEG yeniden kodlama) adımı UYGULANMAZ: S9 yalnız geometrinin (çerçevenin) etkisini ayırmalı. Ardından
    `indir_bcn20000.geometri(.., "merkez_kare")` + `PIL_BILINEAR` 224 px (ANA yolun aynısı; içe aktarılır, değiştirilmez).
  - EXIF yön etiketi: cv2.IMREAD_COLOR yönü uygular, PIL uygulamaz. Görüntü boyutları ya da yön farklıysa kutu PIL pikselinden
    (RGB→gri, cv2) hesaplanır ve `yon_uyusmaz` sayılır (a0 raporunda; 1024² dermoskopide 0 beklenir).
"""
import io
import sys
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
ESIK_GRI = 60          # yazarın cv2.threshold eşiği
ESIK_ORAN = 1.1        # yazarın kırpma kararı
_Z = None


def _ic():
    sys.path.insert(0, str(Path(__file__).resolve().parent / "ucuncu"))
    import image_cropper as IC  # noqa: E402  (bayt bayt kopya; KAYNAK.md)
    return IC


def _ib():
    sys.path.insert(0, str(WS / "scripts"))
    import indir_bcn20000 as IB  # noqa: E402  (geometri + adaylar; DEĞİŞTİRİLMEZ, yalnız içe aktarılır)
    return IB


def oran_vek(gray: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    crop = gray[y:y + h, x:x + w]
    cii = np.int64(crop.sum(dtype=np.int64)) / (crop.shape[0] + crop.shape[1])
    fii = np.int64(gray.sum(dtype=np.int64)) / (gray.shape[0] + gray.shape[1])
    return float(cii / fii)


def karar(bgr: np.ndarray) -> dict:
    """Yazarın görüntü başına kararı. Dönüş: {kirp: bool, kutu: (x,y,w,h) | None, oran: float | None, neden: str}."""
    import cv2
    IC = _ic()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, ESIK_GRI, 255, 0)
    try:
        x, y, w, h = IC.obtain_bb(thresh)
    except UnboundLocalError:
        return {"kirp": False, "kutu": None, "oran": None, "neden": "kutu_yok"}
    r = oran_vek(gray, x, y, w, h)
    return {"kirp": bool(r > ESIK_ORAN), "kutu": (int(x), int(y), int(w), int(h)), "oran": r,
            "neden": "kirp" if r > ESIK_ORAN else "oran_dusuk"}


def isle_bayt(veri: bytes) -> tuple:
    """JPEG baytları → (224 px uint8 | None (kırpılmadı), bilgi). Kırpılmayan görüntü için None döner: ANA merkez_kare
    satırı aynen kullanılır (a0 bunu örneklemle bayt bayt sınar)."""
    import cv2
    from PIL import Image
    IB = _ib()
    im = Image.open(io.BytesIO(veri))
    exif_yon = None
    try:
        exif_yon = im.getexif().get(274)
    except Exception:  # noqa: BLE001
        pass
    rgb = np.asarray(im.convert("RGB"))
    bgr = cv2.imdecode(np.frombuffer(veri, np.uint8), cv2.IMREAD_COLOR)
    yon_uyusmaz = bgr is None or bgr.shape[:2] != rgb.shape[:2] or exif_yon not in (None, 1)
    if yon_uyusmaz:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    k = karar(bgr)
    bilgi = {"boyut": f"{rgb.shape[1]}x{rgb.shape[0]}", "yon_uyusmaz": bool(yon_uyusmaz), **k}
    if not k["kirp"]:
        return None, bilgi
    x, y, w, h = k["kutu"]
    parca = Image.fromarray(np.ascontiguousarray(rgb[y:y + h, x:x + w]))
    return IB.adaylar()["PIL_BILINEAR"](IB.geometri(parca, "merkez_kare"), 224), bilgi


def merkez_kare_bayt(veri: bytes) -> np.ndarray:
    """ANA yol (indir_bcn20000._isci_isle ile aynı): PIL RGB → merkez_kare → PIL_BILINEAR 224."""
    from PIL import Image
    IB = _ib()
    im = Image.open(io.BytesIO(veri)).convert("RGB")
    return IB.adaylar()["PIL_BILINEAR"](IB.geometri(im, "merkez_kare"), 224)


# ── süreç havuzu (Windows: modül düzeyi işlevler) ──
def isci_baslat(zip_yol: str):
    global _Z
    import zipfile
    _Z = zipfile.ZipFile(zip_yol)


def isci(arg):
    i, ic_ad, kontrol = arg
    veri = _Z.read(ic_ad)          # zipfile CRC32'yi doğrular
    x, bilgi = isle_bayt(veri)
    mk = merkez_kare_bayt(veri) if kontrol else None
    return i, x, bilgi, mk


def es_degerlik_sina(n: int = 25, tohum: int = 0) -> dict:
    """oran_vek ↔ özgün intensity_ratio (int64 girdiyle; numpy 1.x anlamı) birebir; ayrıca özgün işlevin uint8 girdide
    numpy ≥ 2'de taştığını belgeler. Küçük rastgele dizilerde (Python döngüsü hızlı kalsın)."""
    IC = _ic()
    rng = np.random.default_rng(tohum)
    esit, tasma = 0, 0
    for _ in range(n):
        H, W = rng.integers(20, 60, 2)
        g = rng.integers(0, 256, (H, W)).astype(np.uint8)
        x, y = int(rng.integers(0, W // 2)), int(rng.integers(0, H // 2))
        w, h = int(rng.integers(1, W - x)), int(rng.integers(1, H - y))
        ozgun = IC.intensity_ratio(g.astype(np.int64), x, y, w, h)
        esit += int(float(ozgun) == oran_vek(g, x, y, w, h))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                tasma += int(float(IC.intensity_ratio(g, x, y, w, h)) != oran_vek(g, x, y, w, h))
            except Exception:  # noqa: BLE001
                tasma += 1
    return {"n": n, "esit": esit, "uint8_ozgun_farkli": tasma, "numpy": np.__version__}
