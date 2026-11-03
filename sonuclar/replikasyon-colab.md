# Replikasyon Kapısı — YEŞİL
*2026-10-11 15:27* · metrik `test_acc` (yüksek iyi)

| | değer |
|---|---|
| Literatür değeri/aralığı | 0.7338 – 0.742 |
| Kaynak | Yang 2023 Sci Data 10:41 T3 (0.735); Doerrich 2025 Sci Rep 15:7669 S4 (73.38±0.61); Medeiros 2026 arXiv:2604.09697 T2 (0.742) — COLAB L4 tekrarı |
| Tolerans (mutlak) | ±0.012 → kabul aralığı [0.7218, 0.754] |
| Bizim baseline (baseline_colab) | **0.7395 ± 0.00526** (n=5 seed: 0, 1, 2, 3, 4) |
| Seed değerleri | 0.7436, 0.7317, 0.7411, 0.7441, 0.7367 |

**Karar:** YEŞİL — Baseline literatür aralığını tolerans içinde tekrarlıyor. Manipülasyon fazı açılabilir.

Bu tablo makalenin Deneysel Kurulum bölümüne "baseline replikasyonu" olarak girer
(literatür değeri `kaynak: literatur` etiketiyle evidence.json'a alınır).
