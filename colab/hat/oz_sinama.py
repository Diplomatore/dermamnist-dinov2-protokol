#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
oz_sinama.py — Colab ortamının SAYISAL öz-sınaması (ölçüm koşumu DEĞİL; leaderboard'a girmez).

--tf32 : FP32 matmul + konvolüsyonun fp64 referansına göre göreli hatası. IEEE FP32 ≈ 1e-6, TF32 ≈ 1e-3 (10 bit mantis).
         Eşik 1e-4: altı = TF32 KAPALI (ön-kayıt: FP32, TF32 kapalı). --seed-bagla verilirse ölçümden önce
         varyant betiklerinin yaptığı gibi ortak/kayit.seed_bagla(0) çağrılır (eski API bayraklarının yeniden yazılması
         yeni API durumunu bozmasın diye). Sarmalayıcısız + NVIDIA_TF32_OVERRIDE'sız koşu "güç" sınamasıdır: TF32'yi
         görebildiğimizi gösterir (L4'te varsayılan cudnn.allow_tf32=True).
Çıktı: JSON (stdout son satırı + --cikti).
"""
import argparse
import json
import sys
from pathlib import Path


def olc(torch) -> dict:
    g = torch.Generator(device="cpu").manual_seed(1234)
    a = torch.randn(2048, 2048, generator=g)
    b = torch.randn(2048, 2048, generator=g)
    x = torch.randn(16, 64, 56, 56, generator=g)
    w = torch.randn(64, 64, 3, 3, generator=g) * 0.05
    dev = torch.device("cuda:0")
    a32, b32, x32, w32 = (t.to(dev) for t in (a, b, x, w))
    c32 = (a32 @ b32).double().cpu()
    c64 = (a.double() @ b.double())
    y32 = torch.nn.functional.conv2d(x32, w32, padding=1).double().cpu()
    y64 = torch.nn.functional.conv2d(x.double().to(dev), w.double().to(dev), padding=1).cpu()
    mm = float((c32 - c64).abs().max() / c64.abs().max())
    cv = float((y32 - y64).abs().max() / y64.abs().max())
    return {"matmul_goreli_hata": mm, "conv_goreli_hata": cv}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--seed-bagla", action="store_true")
    p.add_argument("--etiket", default="")
    p.add_argument("--cikti", default=None)
    p.add_argument("--esik", type=float, default=1e-4)
    a = p.parse_args()
    import torch
    if not torch.cuda.is_available():
        print("HATA: CUDA yok"); return 97
    if a.seed_bagla:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "varyantlar"))
        from ortak.kayit import seed_bagla
        seed_bagla(0)
    r = {"etiket": a.etiket, "torch": torch.__version__, "cihaz": torch.cuda.get_device_name(0),
         "cuda.matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
         "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32}
    for ad, f in (("cuda.matmul.fp32_precision", lambda: torch.backends.cuda.matmul.fp32_precision),
                  ("cudnn.conv.fp32_precision", lambda: torch.backends.cudnn.conv.fp32_precision)):
        try:
            r[ad] = f()
        except Exception as e:  # noqa: BLE001
            r[ad] = f"yok ({type(e).__name__})"
    r.update(olc(torch))
    r["esik"] = a.esik
    r["ieee"] = bool(r["matmul_goreli_hata"] < a.esik and r["conv_goreli_hata"] < a.esik)
    s = json.dumps(r, ensure_ascii=False)
    if a.cikti:
        Path(a.cikti).parent.mkdir(parents=True, exist_ok=True)
        Path(a.cikti).write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    print(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
