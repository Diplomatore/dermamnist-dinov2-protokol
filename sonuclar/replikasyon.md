# Replikasyon Kapısı — YEŞİL
*2026-10-10 19:54* · metrik `test_acc` (yüksek iyi)

| | değer |
|---|---|
| Literatür değeri/aralığı | 0.7338 – 0.742 |
| Kaynak | Yang 2023 Sci Data 10:41 T3 (0.735); Doerrich 2025 Sci Rep 15:7669 S4 (73.38±0.61); Medeiros 2026 arXiv:2604.09697 T2 (0.742) |
| Tolerans (mutlak) | ±0.012 → kabul aralığı [0.7218, 0.754] |
| Bizim baseline (baseline) | **0.7296 ± 0.0176** (n=5 seed: 0, 1, 2, 3, 4) |
| Seed değerleri | 0.7067, 0.7197, 0.7501, 0.7436, 0.7277 |

**Karar:** YEŞİL — Baseline literatür aralığını tolerans içinde tekrarlıyor. Manipülasyon fazı açılabilir.

Bu tablo makalenin Deneysel Kurulum bölümüne "baseline replikasyonu" olarak girer
(literatür değeri `kaynak: literatur` etiketiyle evidence.json'a alınır).

## Ek kapılar (aynı 5 koşum; ayrıntı ayrı dosyalarda)
- `test_auc`: 0.9148 ± 0.0034 → YEŞİL [0.907, 0.927] — `replikasyon-test_auc.md`
- `test_acc_son` (100. epoch; resmî kodun 2023-05 öncesi `best_model = model` referans hatasıyla raporlanan büyüklük): 0.7395 ± 0.0076 → YEŞİL — `replikasyon-test_acc_son.md`
- Not: val-AUC ile seçilen epoch (resmî niyet) seed'ler arası daha oynak (std 0.0176 vs 0.0076); en iyi epoch'lar erken (aşırı uyum başlamadan).
