# Colab tekrar kapısı — ÖZET (BİLGİ; kapı kararı şefte, SIRALAMA.md ön-kaydı)
*2026-10-11T15:26:33* · kuyruklar: kc1-tekrar

| kol | n | Colab val_acc ort ± std | yerel | |Δ| puan | tolerans | uyum |
|---|---|---|---|---|---|---|
| baseline (`baseline_colab`) | 5 | 0.7458 ± 0.0074 | 0.7396 | 0.62 | 2.0 | UYUMLU |
| H01 (`h01_dinov2b_lp_r224_colab`) | 5 | 0.8223 ± 0.0013 | 0.8223 | 0.00 | 1.5 | UYUMLU |
| H02 (`h02_r18in_ft224_colab`) | 5 | 0.8656 ± 0.0019 | 0.8672 | 0.16 | 1.5 | UYUMLU |
| H08 (`h08_r18in_ft224_d4_colab`) | 5 | 0.8865 ± 0.0033 | 0.8893 | 0.28 | 1.5 | UYUMLU |

Sıralama H08 > H02 > H01: **KORUNDU** (0.8865 / 0.8656 / 0.8223)

baseline_colab test_acc ort 0.7395 (n=5) ∈ [0,7218; 0,754]: **EVET** (karar verici: replikasyon_kapisi.py)

FP32 denetimi: fp32_durum.json eksik 0 [] · TF32 açık görülen 0 []
