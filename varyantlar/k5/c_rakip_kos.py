#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
c_rakip_kos.py — RAKİBİ (scripts/rakip_kos.py → MedMNIST/experiments@70b6b3a, DOKUNULMAZ) C1/C2 verisiyle koşar
(K5-ONKAYIT §4.2 baseline satırı: "rakip kodu, yalnız veri yolu override", 28 px, 100 epoch, seçim resmî val-AUC).

Yapılan (aynı süreç, runpy):
  1) C verisi doğrulanır (c_bolme.dogrula: c1 içerik SHA256 donmuş değer / c2 Zenodo MD5).
  2) data/k5_rakip_<veri>/dermamnist.npz → C npz'nin 28 px dosyasına sembolik bağ (medmnist dosya adı sabittir).
  3) medmnist.INFO["dermamnist"]["MD5"] süreç içinde doğrulanmış C dosyasının MD5'ine çevrilir (rakip_kos.py'nin resmî-MD5
     denetimi C dosyasını reddetmesin diye; denetim c_bolme.dogrula ile YAPILDI) ve INFO["dermamnist"]["n_samples"] C bölme
     boyutlarına (dosyadan; 8.215/573/1.227) çevrilir (medmnist dataset __len__ assert'i; 2026-10-11 C1-A dumanı hatası).
     INFO'nun başka alanı değişmez.
  4) scripts/rakip_kos.py --veri-kok data/k5_rakip_<veri> + kalan argümanlar (seed, epoch, out) aynen.
Not: rakip protokolü test bölümünü her epoch değerlendirir (seçime katmaz); burada "test" = C-test (1.227). Rakibin
yazdığı test_* anahtarları C-test değerleridir ve C1/C2 analizinde (scripts/k5_test.py --kip c1|c2) kullanılır.
Kullanım: python colab/fp32_kos.py varyantlar/k5/c_rakip_kos.py --veri c1 -- --seed 0 --epoch 100 --out {RUN_DIR}/metrics.json
"""
import json
import os
import runpy
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[2]


def main():
    argv = sys.argv[1:]
    if len(argv) < 3 or argv[0] != "--veri" or "--" not in argv:
        raise SystemExit("kullanım: c_rakip_kos.py --veri c1|c2 -- <rakip_kos.py argümanları>")
    veri = argv[1]
    rest = argv[argv.index("--") + 1:]
    if "--veri-kok" in rest or "--size" in rest:
        raise SystemExit("HATA: --veri-kok/--size bu sarmalayıcıda verilmez (28 px, C kökü)")
    sys.path.insert(0, str(WS / "varyantlar"))
    from k5 import c_bolme
    bilgi = c_bolme.dogrula(veri, 28)
    kok = WS / "data" / f"k5_rakip_{veri}"
    kok.mkdir(parents=True, exist_ok=True)
    bag = kok / "dermamnist.npz"
    hedef = c_bolme.npz_yolu_c(veri, 28)
    if bag.is_symlink() or bag.exists():
        if bag.resolve() != hedef.resolve():
            bag.unlink()
    if not bag.exists():
        try:
            bag.symlink_to(hedef)
        except OSError:   # Windows'ta sembolik bağ izni yoksa kopya (yalnız yerel sınama)
            import shutil
            shutil.copyfile(hedef, bag)
    import medmnist
    import numpy as np
    eski = medmnist.INFO["dermamnist"]["MD5"]
    eski_n = dict(medmnist.INFO["dermamnist"]["n_samples"])
    medmnist.INFO["dermamnist"]["MD5"] = bilgi["md5"]
    # medmnist/dataset.py __len__: assert info["n_samples"][split] == imgs.shape[0] → C bölme boyutları (dosyadan okunur;
    # C1 = C2 = 8.215/573/1.227). Yalnız süreç içi INFO; rakip kodu ve paket dosyası değişmez (K5-ONKAYIT-b §4).
    with np.load(hedef) as z:
        yeni_n = {s: int(z[f"{s}_labels"].shape[0]) for s in ("train", "val", "test")}
    medmnist.INFO["dermamnist"]["n_samples"] = yeni_n
    rd = os.environ.get("RUN_DIR")
    if rd:
        Path(rd).mkdir(parents=True, exist_ok=True)
        (Path(rd) / "c_bolme.json").write_text(json.dumps(
            {"veri": veri, "npz": str(hedef), "dogrulama": bilgi, "medmnist_md5_resmi_ozgun": eski,
             "medmnist_md5_surec_ici": bilgi["md5"], "medmnist_n_samples_ozgun": eski_n,
             "medmnist_n_samples_surec_ici": yeni_n, "veri_kok": str(kok)}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[c_rakip_kos] veri={veri} · {hedef.name} md5 {bilgi['md5']} · rakip_kos.py --veri-kok {kok}", flush=True)
    betik = WS / "scripts" / "rakip_kos.py"
    sys.argv = [str(betik), "--veri-kok", str(kok)] + rest
    sys.path[0] = str(betik.parent)
    runpy.run_path(str(betik), run_name="__main__")


if __name__ == "__main__":
    main()
