# Replikasyon Kapısı — YEŞİL
*2026-10-10 19:54* · metrik `test_auc` (yüksek iyi)

| | değer |
|---|---|
| Literatür değeri/aralığı | 0.917 – 0.917 |
| Kaynak | Yang 2023 T3 (0.917) |
| Tolerans (mutlak) | ±0.01 → kabul aralığı [0.907, 0.927] |
| Bizim baseline (baseline) | **0.9148 ± 0.00338** (n=5 seed: 0, 1, 2, 3, 4) |
| Seed değerleri | 0.91, 0.9144, 0.9195, 0.9149, 0.9149 |

**Karar:** YEŞİL — Baseline literatür aralığını tolerans içinde tekrarlıyor. Manipülasyon fazı açılabilir.

Bu tablo makalenin Deneysel Kurulum bölümüne "baseline replikasyonu" olarak girer
(literatür değeri `kaynak: literatur` etiketiyle evidence.json'a alınır).
