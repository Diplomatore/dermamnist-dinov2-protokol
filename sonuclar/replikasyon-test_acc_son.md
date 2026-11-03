# Replikasyon Kapısı — YEŞİL
*2026-10-10 19:53* · metrik `test_acc_son` (yüksek iyi)

| | değer |
|---|---|
| Literatür değeri/aralığı | 0.7338 – 0.742 |
| Kaynak | aynı çıpa; son-epoch (2023-05 öncesi best_model hatası) |
| Tolerans (mutlak) | ±0.012 → kabul aralığı [0.7218, 0.754] |
| Bizim baseline (baseline) | **0.7395 ± 0.00763** (n=5 seed: 0, 1, 2, 3, 4) |
| Seed değerleri | 0.7317, 0.7451, 0.7312, 0.7476, 0.7416 |

**Karar:** YEŞİL — Baseline literatür aralığını tolerans içinde tekrarlıyor. Manipülasyon fazı açılabilir.

Bu tablo makalenin Deneysel Kurulum bölümüne "baseline replikasyonu" olarak girer
(literatür değeri `kaynak: literatur` etiketiyle evidence.json'a alınır).
