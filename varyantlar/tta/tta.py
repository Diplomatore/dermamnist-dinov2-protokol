#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tta.py — H07: test-zamanı artırma (TTA). Ebeveynin seçilmiş-epoch ağırlıkları (best_model.pth) ile val'de
8 dihedral (D4) görünümün softmax olasılıklarının ARİTMETİK ortalaması. Eğitim YOK. TEST YOK (M4).

Kart karşılığı (H07 "Değişen tek şey" + "Uygulama notu"):
- p(c|x) = (1/8) Σ_{g∈D4} softmax(f(g·x)). D4 tanımı `ortak/dihedral.py` (H08 ile ortak tek kaynak).
- h02_r18in_ft224: torchvision resnet18(weights=None), fc = Linear(512, 7), load_state_dict(strict=True), eval();
  dermamnist_224.npz val (MD5 doğrulamalı); normalizasyon ince_ayar.py hazirla() ile AYNI (uint8 → GPU →
  permute(0,3,1,2).contiguous().float()/255 → ImageNet). Görünümler normalizasyondan SONRA GPU'da; her biri
  .contiguous(). cudnn.deterministic=True, benchmark=False. val bs 128.
- baseline (M3 yan kolu, `h07x_baseline_tta8`): rakibin models.ResNet18 sınıfı (rakip/experiments/MedMNIST2D, SALT-OKUNUR
  içe aktarma; rakip dosyasına yazılmaz), resmi_cikti/**/best_model.pth ['net'], 28 px, ToTensor + Normalize(0,5; 0,5)
  eşdeğeri (x/255 − 0,5)/0,5.
Denetimler (öldürme (c)/(d)): kimlik görünümü ↔ ebeveynin tahmin_val.npy max|Δp| (≤ 1e-4) ve kimlik val_acc ↔
ebeveynin metrics.json val_acc (tam eşit); görünümler arası max|Δp| > 0 (dönüşüm devrede). İhlalde metrics yine
yazılır (teşhis için) ve rc = 3 döner (DUR).

Çıktılar ({RUN_DIR}): metrics.json (val_*; TEST ANAHTARI YOK) · tahmin_val.npy (8 görünüm ortalaması, float32) ·
  tahmin_val_gorunum.npy (8 × N × 7, float32; ablasyon i/ii) · etiket_val.npy · config.json eki.
Kullanım (WS kökü; kosum.py altında):
  env/.venv/Scripts/python.exe varyantlar/tta/tta.py --ebeveyn-varyant h02_r18in_ft224 --seed 0 --gorunum d4 \
      --val-bs 128 --out {RUN_DIR}/metrics.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
os.environ.setdefault("TORCH_HOME", str(WS / "data" / "torch_cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ortak import dihedral as D4  # noqa: E402
from ortak import veri as V  # noqa: E402
from ortak.degerlendir import _acc, degerlendir  # noqa: E402
from ortak.kayit import (cihaz_sec, config_ekle, donanim_bilgisi, metrics_yaz, seed_bagla,  # noqa: E402
                         tahmin_kaydet)

IMAGENET_ORT = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RAKIP_DIR = WS / "rakip" / "experiments" / "MedMNIST2D"
OZDESLIK_TOL = 1e-4  # kart (c)


def ebeveyn_bul(varyant: str, seed: int) -> Path:
    """runs/ altında varyant adı eşleşen, damgalı rc=0, aşama 3, bu seed'in TEK koşumu (tahmin_val.npy var)."""
    bul = []
    for d in sorted((WS / "runs").glob("*/")):
        cfg, done = d / "config.json", d / ".done"
        if not (cfg.exists() and done.exists() and (d / "tahmin_val.npy").exists()):
            continue
        c = json.loads(cfg.read_text(encoding="utf-8"))
        j = json.loads(done.read_text(encoding="utf-8"))
        if c.get("varyant") == varyant and j.get("rc") == 0 and int(c.get("asama", 3)) == 3 and int(c["seed"]) == seed:
            bul.append(d)
    if len(bul) != 1:
        raise SystemExit(f"HATA: {varyant} seed {seed} için {len(bul)} damgalı aşama-3 koşumu: {[b.name for b in bul]}")
    return bul[0]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ebeveyn-varyant", required=True, choices=["h02_r18in_ft224", "baseline"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--gorunum", default="d4", choices=["d4"], help="kart: 8 dihedral görünüm (alt kümeler tta_alt.py)")
    p.add_argument("--val-bs", type=int, default=128)
    p.add_argument("--out", required=True)
    p.add_argument("--cihaz", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--cpu-izin", action="store_true")
    p.add_argument("--alt-kume", type=int, default=None, help="hata ayıklama: val'in ilk N örneği (ölçüm değildir)")
    a = p.parse_args()

    import torch
    import torch.nn as nn
    out = Path(a.out).resolve()
    run_dir = out.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    if not D4.oz_sinama():
        print("DUR: D4 öz-sınaması başarısız (8 dönüşüm ayrık/kapalı değil)")
        return 3
    cihaz, cihaz_ad = cihaz_sec(a.cihaz, a.cpu_izin)
    seed_bagla(a.seed)

    eb_dir = ebeveyn_bul(a.ebeveyn_varyant, a.seed)
    eb_m = json.loads((eb_dir / "metrics.json").read_text(encoding="utf-8"))
    rakip = a.ebeveyn_varyant == "baseline"
    size = 28 if rakip else 224
    if rakip:
        adaylar = sorted((eb_dir / "resmi_cikti").glob("*/*/best_model.pth"))
        if len(adaylar) != 1:
            raise SystemExit(f"HATA: {eb_dir.name} resmi_cikti altında {len(adaylar)} best_model.pth")
        agirlik = adaylar[0]
        sys.path.insert(0, str(RAKIP_DIR))
        from models import ResNet18  # rakibin model sınıfı (salt-okunur içe aktarma)
        model = ResNet18(in_channels=3, num_classes=V.SINIF_SAYISI)
        sd = torch.load(agirlik, map_location="cpu", weights_only=True)["net"]
        ort_t, std_t, norm_ad = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), "(x/255 − 0,5)/0,5 (rakip ToTensor + Normalize(.5,.5))"
        model_ad = "rakip models.ResNet18 (3x3 stem, maxpool yok), 28 px"
    else:
        from torchvision.models import resnet18
        agirlik = eb_dir / "best_model.pth"
        model = resnet18(weights=None)
        model.fc = nn.Linear(512, V.SINIF_SAYISI)
        sd = torch.load(agirlik, map_location="cpu", weights_only=True)
        ort_t, std_t, norm_ad = IMAGENET_ORT, IMAGENET_STD, "ImageNet (GPU'da; ince_ayar.py hazirla() ile aynı)"
        model_ad = "torchvision resnet18(weights=None) + fc Linear(512,7), 224 px"
    model.load_state_dict(sd, strict=True)
    model.to(cihaz).eval()
    n_param = int(sum(q.numel() for q in model.parameters()))
    agirlik_sha = V.dosya_hash(agirlik)[1]

    veri_bilgi = V.npz_dogrula(size)
    Dv = V.yukle(size, ("val",), alt_kume=a.alt_kume)["val"]
    xva, yva = torch.from_numpy(Dv["x"]), Dv["y"]
    y_eb = np.load(eb_dir / "etiket_val.npy").reshape(-1).astype(np.int64)
    if not np.array_equal(y_eb[:len(yva)], yva):
        print("DUR: ebeveynin etiket_val.npy ↔ npz val etiketleri uyuşmuyor (sıra/hizalama)")
        return 3
    p_eb = np.load(eb_dir / "tahmin_val.npy").astype(np.float32)[:len(yva)]
    ort = torch.tensor(ort_t, device=cihaz).view(1, 3, 1, 1)
    std = torch.tensor(std_t, device=cihaz).view(1, 3, 1, 1)

    def hazirla(xb_u8):  # ince_ayar.py hazirla() ile aynı (rakip kolu için yalnız ort/std farklı)
        x = xb_u8.to(cihaz, non_blocking=True).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
        return (x - ort) / std

    N = len(yva)
    P = np.zeros((8, N, V.SINIF_SAYISI), dtype=np.float32)
    if cihaz.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, N, a.val_bs):
            x0 = hazirla(xva[i:i + a.val_bs])
            for k in range(8):
                P[k, i:i + a.val_bs] = torch.softmax(model(D4.donustur(x0, k)), dim=1).float().cpu().numpy()
    if cihaz.type == "cuda":
        torch.cuda.synchronize()
    sure = round(time.time() - t0, 3)

    p_tta = P.astype(np.float64).mean(axis=0).astype(np.float32)  # olasılıkların aritmetik ortalaması
    tahmin_kaydet(run_dir, p_tta, yva)
    np.save(run_dir / "tahmin_val_gorunum.npy", P)

    # ── denetimler (öldürme c/d) ──
    oz_maks = float(np.max(np.abs(P[0] - p_eb)))
    acc_kimlik = _acc(yva, P[0])
    oz_acc = abs(acc_kimlik - float(eb_m["val_acc"])) if a.alt_kume is None else float("nan")
    gor_maks = float(max(np.max(np.abs(P[k] - P[0])) for k in range(1, 8)))
    am = P.argmax(axis=2)  # (8, N)
    anlasmazlik = float((am != am[0:1]).any(axis=0).mean())
    acc_g = [_acc(yva, P[k]) for k in range(8)]

    m = degerlendir(yva, p_tta, on="val", val_idx=Dv["idx"])
    alt_grup = m.pop("alt_grup")
    metrics = {
        **m,
        **{f"val_acc_gorunum_{k}": acc_g[k] for k in range(8)},
        "val_acc_gorunum_donusmus_ort_fark": float(np.mean(acc_g[1:]) - acc_g[0]),
        "val_acc_d2_ort": float(np.mean([acc_g[k] for k in D4.D2])),
        "val_acc_r90_ort": float(np.mean([acc_g[k] for k in D4.R90])),
        "gorunum_arasi_argmax_anlasmazlik": anlasmazlik,
        "ozdeslik_kimlik_maks_fark": oz_maks,
        "ozdeslik_kimlik_acc_fark": oz_acc,
        "gorunum_arasi_maks_fark": gor_maks,
        "karar_degisen_n": int((p_tta.argmax(1) != p_eb.argmax(1)).sum()),
        "ebeveyn_val_acc": float(eb_m["val_acc"]),
        "val_acc_delta": float(m["val_acc"] - float(eb_m["val_acc"])),
        "gorunum_sayisi": 8,
        "cikarim_sure_sn": sure, "cikarim_img_s": round(8 * N / max(sure, 1e-9), 1),
        "egitim_sure_sn": sure, "sure_sn": sure, "param_sayisi": n_param, "giris_cozunurluk": size,
        "seed": a.seed, "cihaz": cihaz_ad, "alt_grup": alt_grup,
    }
    if cihaz.type == "cuda":
        metrics["tepe_gpu_bellek_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    if a.alt_kume:
        metrics["hata_ayiklama"] = 1
    config_ekle(run_dir, {
        "donanim": {**donanim_bilgisi(), "cihaz": cihaz_ad}, "veri": veri_bilgi,
        "model": {"ad": model_ad, "agirlik": str(agirlik.relative_to(WS)).replace("\\", "/"), "agirlik_sha256": agirlik_sha,
                  "param_sayisi": n_param, "ebeveyn_run": eb_dir.name},
        "protokol": {"tta": "D4 8 görünüm, softmax olasılıklarının aritmetik ortalaması", "gorunumler": D4.AD,
                     "d2_alt_kume": list(D4.D2), "r90_alt_kume": list(D4.R90), "normalizasyon": norm_ad,
                     "gorunum_uretimi": "normalizasyondan sonra GPU'da torch.rot90/flip/transpose + .contiguous()",
                     "val_bs": a.val_bs, "egitim": "yok", "cudnn_deterministic": True, "cudnn_benchmark": False,
                     "size": size, "alt_kume": a.alt_kume},
    })
    metrics_yaz(out, metrics)
    print(f"[tta] {a.ebeveyn_varyant} s{a.seed} ← {eb_dir.name} · TTA val_acc {m['val_acc']:.4f} "
          f"(ebeveyn {float(eb_m['val_acc']):.4f}) · görünüm ACC {' '.join(f'{x:.3f}' for x in acc_g)} · "
          f"kimlik max|Δp| {oz_maks:.2e} · görünümler arası {gor_maks:.3f} · {sure:.1f}s ({8 * N / max(sure, 1e-9):.0f} img/s)")
    dur = []
    if not oz_maks <= OZDESLIK_TOL:
        dur.append(f"kimlik görünümü ebeveynden sapıyor (max|Δp| {oz_maks:.3e} > {OZDESLIK_TOL})")
    if a.alt_kume is None and oz_acc != 0.0:
        dur.append(f"kimlik val_acc ≠ ebeveyn val_acc (fark {oz_acc:.3e})")
    if gor_maks == 0.0:
        dur.append("görünümler arası max|Δp| = 0 (dönüşüm devrede değil)")
    if dur:
        print("DUR (kart öldürme c/d): " + " · ".join(dur))
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
