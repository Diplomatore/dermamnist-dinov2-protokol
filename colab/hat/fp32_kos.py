#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fp32_kos.py — Colab hattı FP32 SARMALAYICISI (yerel kod değişmeden TF32'yi kapatır).

Neden: Ön-kayıt (SIRALAMA.md "ÖN-KAYIT — COLAB TEKRAR KAPISI") FP32, TF32 KAPALI, AMP YOK diyor.
  - Varyant betikleri (ortak/kayit.py seed_bagla) TF32'yi zaten kapatıyor.
  - Rakip köprüsü scripts/rakip_kos.py KAPATMIYOR: torch varsayılanı cudnn.allow_tf32=True → L4'te (Ada)
    konvolüsyonlar TF32 ile koşardı. GTX 1650'de (Turing) TF32 donanımı yok → yerel koşumlar zaten IEEE FP32.
  Bu sarmalayıcı her kuyruk komutunun önüne konur:  python colab/fp32_kos.py <betik.py> <betik argümanları...>
  Aynı süreçte (runpy; PID değişmez → kosum.py --kural öldürmesi doğrudan eğitimi öldürür):
    1) torch.backends.cuda.matmul.allow_tf32 = False, torch.backends.cudnn.allow_tf32 = False
       (+ torch ≥ 2.9'da yeni API: cuda.matmul.fp32_precision = cudnn.conv.fp32_precision = "ieee"),
       torch.set_float32_matmul_precision("highest"); cudnn.deterministic=True, benchmark=False
       (betik kendi seed_bagla'sında aynısını tekrar yapar; rakip_kos.py de deterministic=True yapar).
    2) Durumu {RUN_DIR}/fp32_durum.json'a yazar (başlangıç + atexit'te bitiş durumu, paket sürümü).
    3) Betiği __main__ olarak koşar (sys.argv, sys.path[0], __file__ betiğinkiyle aynı).
  Ortam düzeyinde ayrıca defter NVIDIA_TF32_OVERRIDE=0 (cuBLAS/cuDNN kütüphane düzeyinde TF32 yasağı) ve
  CUBLAS_WORKSPACE_CONFIG=:4096:8 koyar; ikisi de buraya kaydedilir.
Yerel parite: GTX 1650'de TF32 yok → bu sarmalayıcı yerelde aritmetiği DEĞİŞTİRMEZ (yalnız bayrak yazar).
Defter ortam hücresi TF32'nin gerçekten kapalı olduğunu SAYISAL olarak sınar (fp64 referansına göre göreli hata).
"""
import atexit
import json
import os
import runpy
import sys
import time
from pathlib import Path


def _durum(torch) -> dict:
    d = {}
    for ad, f in (("cuda.matmul.allow_tf32", lambda: torch.backends.cuda.matmul.allow_tf32),
                  ("cudnn.allow_tf32", lambda: torch.backends.cudnn.allow_tf32),
                  ("float32_matmul_precision", lambda: torch.get_float32_matmul_precision()),
                  ("cudnn.deterministic", lambda: torch.backends.cudnn.deterministic),
                  ("cudnn.benchmark", lambda: torch.backends.cudnn.benchmark),
                  ("cuda.matmul.fp32_precision", lambda: torch.backends.cuda.matmul.fp32_precision),
                  ("cudnn.conv.fp32_precision", lambda: torch.backends.cudnn.conv.fp32_precision)):
        try:
            d[ad] = f()
        except Exception as e:  # noqa: BLE001 — eski torch'ta yeni API yok
            d[ad] = f"yok ({type(e).__name__})"
    return d


def tf32_kapat(torch) -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for nesne in ("cuda.matmul", "cudnn.conv"):
        try:
            hedef = torch.backends.cuda.matmul if nesne == "cuda.matmul" else torch.backends.cudnn.conv
            hedef.fp32_precision = "ieee"
        except Exception:  # noqa: BLE001 — torch < 2.9
            pass
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("kullanım: python colab/fp32_kos.py <betik.py> [argümanlar...]")
    betik = Path(sys.argv[1]).resolve()
    if not betik.exists():
        raise SystemExit(f"HATA: betik yok: {betik}")
    import torch
    tf32_kapat(torch)
    kayit = {"zaman": time.strftime("%Y-%m-%dT%H:%M:%S"), "betik": str(betik), "torch": torch.__version__,
             "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
             "ortam": {k: os.environ.get(k) for k in ("NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG",
                                                      "HF_HUB_OFFLINE", "TZ", "PYTHONHASHSEED")},
             "baslangic": _durum(torch)}
    paket = Path(__file__).resolve().parents[1] / "PAKET.json"
    if paket.exists():
        try:
            kayit["paket_surum"] = json.loads(paket.read_text(encoding="utf-8")).get("surum")
        except Exception:  # noqa: BLE001
            pass
    rd = os.environ.get("RUN_DIR")
    hedef = Path(rd) / "fp32_durum.json" if rd else None

    def yaz():
        if hedef is None:
            return
        try:
            tmp = hedef.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(kayit, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, hedef)
        except OSError:
            pass

    yaz()

    def bitis():
        kayit["bitis"] = _durum(torch)
        kayit["bitis_zaman"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        yaz()

    atexit.register(bitis)
    print(f"[fp32_kos] TF32 kapalı: {kayit['baslangic']} · NVIDIA_TF32_OVERRIDE={kayit['ortam']['NVIDIA_TF32_OVERRIDE']}",
          flush=True)
    sys.argv = [str(betik)] + sys.argv[2:]
    sys.path[0] = str(betik.parent)
    runpy.run_path(str(betik), run_name="__main__")


if __name__ == "__main__":
    main()
