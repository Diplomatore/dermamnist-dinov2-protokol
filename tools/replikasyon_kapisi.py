#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
replikasyon_kapisi.py — Faz K3 kapısı: rakip (baseline) koşumumuz literatür değerini
tolerans içinde tekrarlıyor mu? Tekrarlamıyorsa manipülasyona GEÇİLMEZ; önce kurulum düzeltilir.
(Çürük baseline'a karşı kazanmak saman adamdır — deney-sozlesmesi.md K3.)

Kullanım:
  python replikasyon_kapisi.py --kok <ws> --metrik accuracy --yon max \
      --lit 0.927 --lit-ust 0.935 --tolerans 0.01 --kaynak "Devlin 2019 / Clusy 2025 reprod."
  --lit tek değer veya --lit/--lit-ust ile aralık. --tolerans mutlak (metrik biriminde).
  --baseline varyant adı (varsayılan 'baseline').

Karar:
  YEŞİL  : ort. baseline, [lit-tol, lit_ust+tol] içinde → manipülasyon açılır
  SARI   : aralığın DIŞINDA ama tolerans×2 içinde → kullanıcıya bildir, gerekçeli devam olabilir
  KIRMIZI: daha kötü (kurulum hatası şüphesi) veya daha İYİ (sızıntı/protokol şüphesi) → DUR
Çıkış kodu: 0 yeşil, 1 sarı, 2 kırmızı, 3 veri yok. Karar sonuclar/replikasyon.md'ye yazılır.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def baseline_degerleri(kok: Path, varyant: str, metrik: str):
    vals = []
    for d in (kok / "runs").glob("*/"):
        done, cfg = d / ".done", d / "config.json"
        if not (done.exists() and cfg.exists()):
            continue
        c = json.loads(cfg.read_text(encoding="utf-8")); j = json.loads(done.read_text(encoding="utf-8"))
        if c.get("varyant") == varyant and j.get("rc") == 0 and j.get("metrics_var"):
            m = j.get("metrics") or {}
            if metrik in m and isinstance(m[metrik], (int, float)):
                vals.append((str(c.get("seed")), float(m[metrik])))
    return vals


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kok", required=True); p.add_argument("--metrik", required=True)
    p.add_argument("--yon", choices=["max", "min"], default="max")
    p.add_argument("--lit", type=float, required=True, help="literatür değeri (alt sınır)")
    p.add_argument("--lit-ust", type=float, default=None, help="literatür aralığı üst sınırı")
    p.add_argument("--tolerans", type=float, required=True, help="mutlak tolerans, metrik biriminde")
    p.add_argument("--kaynak", required=True, help="literatür değerinin künyesi/URL'si")
    p.add_argument("--baseline", default="baseline"); p.add_argument("--min-seed", type=int, default=3)
    a = p.parse_args()

    kok = Path(a.kok).resolve()
    vals = baseline_degerleri(kok, a.baseline, a.metrik)
    lit_alt, lit_ust = a.lit, (a.lit_ust if a.lit_ust is not None else a.lit)
    if lit_ust < lit_alt:
        lit_alt, lit_ust = lit_ust, lit_alt
    out = kok / "sonuclar" / "replikasyon.md"; out.parent.mkdir(parents=True, exist_ok=True)

    if not vals:
        out.write_text(f"# Replikasyon Kapısı — VERİ YOK\n*{datetime.now():%Y-%m-%d %H:%M}*\n\n"
                       f"`{a.baseline}` varyantı için damgalı başarılı koşum bulunamadı.\n", encoding="utf-8")
        print("VERİ YOK — baseline koşumu damgalanmamış."); return 3

    n = len(vals); mu = mean(v for _, v in vals); sd = stdev([v for _, v in vals]) if n > 1 else 0.0
    alt_k, ust_k = lit_alt - a.tolerans, lit_ust + a.tolerans
    if alt_k <= mu <= ust_k:
        karar, rc, aciklama = "YEŞİL", 0, "Baseline literatür aralığını tolerans içinde tekrarlıyor. Manipülasyon fazı açılabilir."
    elif (lit_alt - 2 * a.tolerans) <= mu <= (lit_ust + 2 * a.tolerans):
        karar, rc = "SARI", 1
        aciklama = ("Baseline literatür aralığının dışında ama 2×tolerans içinde. Protokol farkı (split, epoch, "
                    "önişleme) belgelenmeli; kullanıcı onayıyla devam edilebilir.")
    else:
        karar, rc = "KIRMIZI", 2
        daha_iyi = (mu > ust_k) if a.yon == "max" else (mu < alt_k)
        if daha_iyi:
            aciklama = ("Baseline literatürden BELİRGİN İYİ: test sızıntısı, yanlış split, yanlış metrik veya "
                        "farklı değerlendirme protokolü şüphesi. Bu bir başarı değil, kırmızı bayraktır (sonuc-tahlili I-grubu).")
        else:
            aciklama = ("Baseline literatürden BELİRGİN KÖTÜ: kurulum/hiperparametre/sürüm hatası şüphesi. "
                        "Manipülasyona geçilmez; önce kurulum düzeltilir (beklenti-ihlali.md).")
    seed_notu = "" if n >= a.min_seed else f"\n> ⚠️ Yalnız {n} seed; kapı için en az {a.min_seed} gerekir. Karar geçici."
    if n < a.min_seed and rc == 0:
        rc = 1; karar = "SARI (seed yetersiz)"

    md = f"""# Replikasyon Kapısı — {karar}
*{datetime.now():%Y-%m-%d %H:%M}* · metrik `{a.metrik}` ({'yüksek iyi' if a.yon=='max' else 'düşük iyi'})

| | değer |
|---|---|
| Literatür değeri/aralığı | {lit_alt:g} – {lit_ust:g} |
| Kaynak | {a.kaynak} |
| Tolerans (mutlak) | ±{a.tolerans:g} → kabul aralığı [{alt_k:g}, {ust_k:g}] |
| Bizim baseline ({a.baseline}) | **{mu:.4g} ± {sd:.3g}** (n={n} seed: {', '.join(s for s,_ in sorted(vals))}) |
| Seed değerleri | {', '.join(f'{v:.4g}' for _, v in sorted(vals))} |

**Karar:** {karar} — {aciklama}{seed_notu}

Bu tablo makalenin Deneysel Kurulum bölümüne "baseline replikasyonu" olarak girer
(literatür değeri `kaynak: literatur` etiketiyle evidence.json'a alınır).
"""
    out.write_text(md, encoding="utf-8"); print(md); print(f"yazıldı: {out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
