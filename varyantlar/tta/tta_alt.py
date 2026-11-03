#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tta_alt.py — H07 ablasyon (i) (karar vermez): H07 koşumunun KAYDEDİLMİŞ görünüm başına olasılıklarından
(tahmin_val_gorunum.npy, 8 × N × 7) bir alt kümenin ortalaması. Ek GPU/çıkarım YOK (kart: "hepsi kaydedilen görünüm
başına olasılıklardan CPU'da hesaplanır"). TEST YOK.
  --alt d2  : {kimlik, 180°, yatay çevirme, dikey çevirme} (sıkıştırma eksenini koruyan 4 görünüm)
  --alt r90 : {90°, 270°, transpoz, ters-transpoz} (eksenleri değiştiren 4 görünüm)
İndeksler `ortak/dihedral.py` D2 / R90. Kaynak koşum: --kaynak-varyant'ın aynı seed'li aşama-3 koşumu.
Kullanım (WS kökü; kosum.py altında, CPU):
  env/.venv/Scripts/python.exe varyantlar/tta/tta_alt.py --kaynak-varyant h07_h02_tta8 --alt d2 --seed 0 --out {RUN_DIR}/metrics.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from ortak import dihedral as D4  # noqa: E402
from ortak import veri as V  # noqa: E402
from ortak.degerlendir import degerlendir  # noqa: E402
from ortak.kayit import config_ekle, metrics_yaz, tahmin_kaydet  # noqa: E402
from tta import ebeveyn_bul  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kaynak-varyant", required=True)
    p.add_argument("--alt", required=True, choices=["d2", "r90"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    t0 = time.time()
    out = Path(a.out).resolve()
    run_dir = out.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    kd = ebeveyn_bul(a.kaynak_varyant, a.seed)
    km = json.loads((kd / "metrics.json").read_text(encoding="utf-8"))
    P = np.load(kd / "tahmin_val_gorunum.npy")
    y = np.load(kd / "etiket_val.npy").reshape(-1).astype(np.int64)
    with np.load(V.npz_yolu(28)) as z:
        if not np.array_equal(z["val_labels"].reshape(-1).astype(np.int64), y):
            print("DUR: etiket hizası bozuk")
            return 3
    alt = D4.D2 if a.alt == "d2" else D4.R90
    p_alt = P[list(alt)].astype(np.float64).mean(axis=0).astype(np.float32)
    # özdeşlik: 8 görünüm ortalaması kaydedilen TTA ile aynı mı (kaynak bütünlüğü)
    p8 = P.astype(np.float64).mean(axis=0).astype(np.float32)
    oz = float(np.max(np.abs(p8 - np.load(kd / "tahmin_val.npy"))))
    tahmin_kaydet(run_dir, p_alt, y)
    m = degerlendir(y, p_alt, on="val", val_idx=np.arange(len(y)))
    alt_grup = m.pop("alt_grup")
    sure = round(time.time() - t0, 3)
    metrics = {**m, "alt_kume_kod": 1 if a.alt == "d2" else 2, "gorunum_sayisi": len(alt),
               "kaynak_tta8_val_acc": float(km["val_acc"]), "kaynak_ozdeslik_maks_fark": oz,
               "karar_degisen_n_vs_tta8": int((p_alt.argmax(1) != p8.argmax(1)).sum()),
               "egitim_sure_sn": sure, "sure_sn": sure, "param_sayisi": int(km.get("param_sayisi", 0)),
               "seed": a.seed, "cihaz": "cpu", "alt_grup": alt_grup}
    config_ekle(run_dir, {"protokol": {"ablasyon": f"H07 (i) {a.alt}: görünümler {[D4.AD[k] for k in alt]}",
                                       "kaynak_run": kd.name, "gpu": "yok (kaydedilmiş olasılıklar)"}})
    metrics_yaz(out, metrics)
    print(f"[tta_alt] {a.alt} s{a.seed} ← {kd.name}: val_acc {m['val_acc']:.4f} (tta8 {float(km['val_acc']):.4f}) · "
          f"kaynak özdeşlik {oz:.1e}")
    return 0 if oz == 0.0 else 3


if __name__ == "__main__":
    sys.exit(main())
