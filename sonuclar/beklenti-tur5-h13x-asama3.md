# Beklenti Kapısı — kuyruk `tur5-h13x` aşama 3 → **GEÇTİ**
*2026-10-11 18:08* · metrik `val_acc` (max) · beklenti: `tur5-h13x.beklenti.json` · 28 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h13x_h02c_h10_op | 5 | 0.8975 | 0.9023, 0.9003, 0.8933, 0.8993, 0.8923 | 4.7 |
| h13x_t2_h01c | 5 | 0.8909 | 0.8903, 0.8933, 0.8853, 0.9013, 0.8843 | 4.8 |
| h13x_t4_h08c_h10_lpo | 5 | 0.9095 | 0.9013, 0.9143, 0.9163, 0.9073, 0.9083 | 4.7 |
| h13x_m3_baseline_h02c_lpo | 5 | 0.8616 | 0.8624, 0.8574, 0.8624, 0.8564, 0.8694 | 4.7 |
| h13x_abl_h10_hom2 | 5 | 0.8953 | 0.8963, 0.9023, 0.9003, 0.8883, 0.8893 | 4.7 |
| h13x_t3_h10_hom5 | 1 | 0.8973 | 0.8973 | 4.7 |
| h13x_t3_h02c_hom5 | 1 | 0.8764 | 0.8764 | 4.7 |
| h13x_m3_baseline_hom5 | 1 | 0.7747 | 0.7747 | 4.7 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 28 başarılı koşum (en az 28) |
| 🟢 OK | kollar_farkli | kollar birbirinden farklı |
| 🟢 OK | farkli_seedler_farkli | h13x_h02c_h10_op: seed'ler farklı |
| 🟢 OK | farkli_seedler_farkli | h13x_t2_h01c: seed'ler farklı |
| 🟢 OK | farkli_seedler_farkli | h13x_t4_h08c_h10_lpo: seed'ler farklı |
| 🟢 OK | farkli_seedler_farkli | h13x_m3_baseline_h02c_lpo: seed'ler farklı |
| 🟢 OK | farkli_seedler_farkli | h13x_abl_h10_hom2: seed'ler farklı |
| 🟢 OK | aralik | h13x_h02c_h10_op: ort 0.8975 ∈ [0.7, 0.93] (alt = baseline_c ≈ 0,75 − pay; üst = H13 görülmemiş ACC) |
| 🟢 OK | aralik | h13x_t2_h01c: ort 0.8909 ∈ [0.7, 0.93] (alt = baseline_c ≈ 0,75 − pay; üst = H13 görülmemiş ACC) |
| 🟢 OK | aralik | h13x_t4_h08c_h10_lpo: ort 0.9095 ∈ [0.7, 0.93] (alt = baseline_c ≈ 0,75 − pay; üst = H13 görülmemiş ACC) |
| 🟢 OK | aralik | h13x_m3_baseline_h02c_lpo: ort 0.8616 ∈ [0.7, 0.93] (alt = baseline_c ≈ 0,75 − pay; üst = H13 görülmemiş ACC) |
| 🟢 OK | aralik | h13x_abl_h10_hom2: ort 0.8953 ∈ [0.7, 0.93] (alt = baseline_c ≈ 0,75 − pay; üst = H13 görülmemiş ACC) |
| 🟢 OK | aralik | h13x_t3_h10_hom5: ort 0.8973 ∈ [0.7, 0.93] (alt = baseline_c ≈ 0,75 − pay; üst = H13 görülmemiş ACC) |
| 🟢 OK | aralik | h13x_t3_h02c_hom5: ort 0.8764 ∈ [0.7, 0.93] (alt = baseline_c ≈ 0,75 − pay; üst = H13 görülmemiş ACC) |
| 🟢 OK | aralik | h13x_m3_baseline_hom5: ort 0.7747 ∈ [0.7, 0.93] (alt = baseline_c ≈ 0,75 − pay; üst = H13 görülmemiş ACC) |
| 🟢 OK | ikincil_aralik | h13x_h02c_h10_op/s0 etiket_hizasi=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_t2_h01c/s0 etiket_hizasi=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_t4_h08c_h10_lpo/s0 etiket_hizasi=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_m3_baseline_h02c_lpo/s0 etiket_hizasi=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_abl_h10_hom2/s0 etiket_hizasi=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_h02c_h10_op/s1 etiket_hizasi=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_t2_h01c/s1 etiket_hizasi=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_t4_h08c_h10_lpo/s1 etiket_hizasi=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_m3_baseline_h02c_lpo/s1 etiket_hizasi=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_abl_h10_hom2/s1 etiket_hizasi=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_h02c_h10_op/s2 etiket_hizasi=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_t2_h01c/s2 etiket_hizasi=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_t4_h08c_h10_lpo/s2 etiket_hizasi=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_m3_baseline_h02c_lpo/s2 etiket_hizasi=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_abl_h10_hom2/s2 etiket_hizasi=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_h02c_h10_op/s3 etiket_hizasi=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_t2_h01c/s3 etiket_hizasi=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_t4_h08c_h10_lpo/s3 etiket_hizasi=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_m3_baseline_h02c_lpo/s3 etiket_hizasi=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_abl_h10_hom2/s3 etiket_hizasi=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_h02c_h10_op/s4 etiket_hizasi=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_t2_h01c/s4 etiket_hizasi=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_t4_h08c_h10_lpo/s4 etiket_hizasi=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_m3_baseline_h02c_lpo/s4 etiket_hizasi=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_abl_h10_hom2/s4 etiket_hizasi=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_t3_h10_hom5/s0-4 etiket_hizasi=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_t3_h02c_hom5/s0-4 etiket_hizasi=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_m3_baseline_hom5/s0-4 etiket_hizasi=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_h02c_h10_op/s0 seed_denet_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_t2_h01c/s0 seed_denet_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_t4_h08c_h10_lpo/s0 seed_denet_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_m3_baseline_h02c_lpo/s0 seed_denet_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_abl_h10_hom2/s0 seed_denet_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_h02c_h10_op/s1 seed_denet_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_t2_h01c/s1 seed_denet_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_t4_h08c_h10_lpo/s1 seed_denet_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_m3_baseline_h02c_lpo/s1 seed_denet_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_abl_h10_hom2/s1 seed_denet_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_h02c_h10_op/s2 seed_denet_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_t2_h01c/s2 seed_denet_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_t4_h08c_h10_lpo/s2 seed_denet_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_m3_baseline_h02c_lpo/s2 seed_denet_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_abl_h10_hom2/s2 seed_denet_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_h02c_h10_op/s3 seed_denet_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_t2_h01c/s3 seed_denet_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_t4_h08c_h10_lpo/s3 seed_denet_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_m3_baseline_h02c_lpo/s3 seed_denet_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_abl_h10_hom2/s3 seed_denet_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_h02c_h10_op/s4 seed_denet_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_t2_h01c/s4 seed_denet_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_t4_h08c_h10_lpo/s4 seed_denet_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_m3_baseline_h02c_lpo/s4 seed_denet_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_abl_h10_hom2/s4 seed_denet_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_t3_h10_hom5/s0-4 seed_denet_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_t3_h02c_hom5/s0-4 seed_denet_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_m3_baseline_hom5/s0-4 seed_denet_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13x_h02c_h10_op/s0 satir_toplam_maks_sapma=4.326e-08 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13x_t2_h01c/s0 satir_toplam_maks_sapma=4.408e-08 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13x_t4_h08c_h10_lpo/s0 satir_toplam_maks_sapma=4.076e-08 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13x_m3_baseline_h02c_lpo/s0 satir_toplam_maks_sapma=4.18e-08 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13x_abl_h10_hom2/s0 satir_toplam_maks_sapma=4.054e-08 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13x_h02c_h10_op/s1 satir_toplam_maks_sapma=4.191e-08 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13x_t2_h01c/s1 satir_toplam_maks_sapma=4.025e-08 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13x_t4_h08c_h10_lpo/s1 satir_toplam_maks_sapma=3.405e-08 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13x_m3_baseline_h02c_lpo/s1 satir_toplam_maks_sapma=3.848e-08 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13x_abl_h10_hom2/s1 satir_toplam_maks_sapma=3.633e-08 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13x_h02c_h10_op/s2 satir_toplam_maks_sapma=4.251e-08 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13x_t2_h01c/s2 satir_toplam_maks_sapma=3.792e-08 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13x_t4_h08c_h10_lpo/s2 satir_toplam_maks_sapma=3.951e-08 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13x_m3_baseline_h02c_lpo/s2 satir_toplam_maks_sapma=4.111e-08 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13x_abl_h10_hom2/s2 satir_toplam_maks_sapma=3.365e-08 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13x_h02c_h10_op/s3 satir_toplam_maks_sapma=4.208e-08 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13x_t2_h01c/s3 satir_toplam_maks_sapma=3.893e-08 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13x_t4_h08c_h10_lpo/s3 satir_toplam_maks_sapma=4.059e-08 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13x_m3_baseline_h02c_lpo/s3 satir_toplam_maks_sapma=3.97e-08 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13x_abl_h10_hom2/s3 satir_toplam_maks_sapma=4.204e-08 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13x_h02c_h10_op/s4 satir_toplam_maks_sapma=4.258e-08 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13x_t2_h01c/s4 satir_toplam_maks_sapma=3.731e-08 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13x_t4_h08c_h10_lpo/s4 satir_toplam_maks_sapma=4.069e-08 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13x_m3_baseline_h02c_lpo/s4 satir_toplam_maks_sapma=3.766e-08 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13x_abl_h10_hom2/s4 satir_toplam_maks_sapma=4.093e-08 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13x_t3_h10_hom5/s0-4 satir_toplam_maks_sapma=3.801e-08 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13x_t3_h02c_hom5/s0-4 satir_toplam_maks_sapma=3.679e-08 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13x_m3_baseline_hom5/s0-4 satir_toplam_maks_sapma=4.278e-08 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13x_h02c_h10_op/s0 kirp=1e-12 ∈ [1e-12,1e-12] |
| 🟢 OK | ikincil_aralik | h13x_t2_h01c/s0 kirp=1e-12 ∈ [1e-12,1e-12] |
| 🟢 OK | ikincil_aralik | h13x_t4_h08c_h10_lpo/s0 kirp=1e-12 ∈ [1e-12,1e-12] |
| 🟢 OK | ikincil_aralik | h13x_m3_baseline_h02c_lpo/s0 kirp=1e-12 ∈ [1e-12,1e-12] |
| 🟢 OK | ikincil_aralik | h13x_abl_h10_hom2/s0 kirp=1e-12 ∈ [1e-12,1e-12] |
| 🟢 OK | ikincil_aralik | h13x_h02c_h10_op/s1 kirp=1e-12 ∈ [1e-12,1e-12] |
| 🟢 OK | ikincil_aralik | h13x_t2_h01c/s1 kirp=1e-12 ∈ [1e-12,1e-12] |
| 🟢 OK | ikincil_aralik | h13x_t4_h08c_h10_lpo/s1 kirp=1e-12 ∈ [1e-12,1e-12] |
| 🟢 OK | ikincil_aralik | h13x_m3_baseline_h02c_lpo/s1 kirp=1e-12 ∈ [1e-12,1e-12] |
| 🟢 OK | ikincil_aralik | h13x_abl_h10_hom2/s1 kirp=1e-12 ∈ [1e-12,1e-12] |
| 🟢 OK | ikincil_aralik | h13x_h02c_h10_op/s2 kirp=1e-12 ∈ [1e-12,1e-12] |
| 🟢 OK | ikincil_aralik | h13x_t2_h01c/s2 kirp=1e-12 ∈ [1e-12,1e-12] |
| 🟢 OK | ikincil_aralik | h13x_t4_h08c_h10_lpo/s2 kirp=1e-12 ∈ [1e-12,1e-12] |
| 🟢 OK | ikincil_aralik | h13x_m3_baseline_h02c_lpo/s2 kirp=1e-12 ∈ [1e-12,1e-12] |
| 🟢 OK | ikincil_aralik | h13x_abl_h10_hom2/s2 kirp=1e-12 ∈ [1e-12,1e-12] |
| 🟢 OK | ikincil_aralik | h13x_h02c_h10_op/s3 kirp=1e-12 ∈ [1e-12,1e-12] |
| 🟢 OK | ikincil_aralik | h13x_t2_h01c/s3 kirp=1e-12 ∈ [1e-12,1e-12] |
| 🟢 OK | ikincil_aralik | h13x_t4_h08c_h10_lpo/s3 kirp=1e-12 ∈ [1e-12,1e-12] |
| 🟢 OK | ikincil_aralik | h13x_m3_baseline_h02c_lpo/s3 kirp=1e-12 ∈ [1e-12,1e-12] |
| 🟢 OK | ikincil_aralik | h13x_abl_h10_hom2/s3 kirp=1e-12 ∈ [1e-12,1e-12] |
| 🟢 OK | ikincil_aralik | h13x_h02c_h10_op/s4 kirp=1e-12 ∈ [1e-12,1e-12] |
| 🟢 OK | ikincil_aralik | h13x_t2_h01c/s4 kirp=1e-12 ∈ [1e-12,1e-12] |
| 🟢 OK | ikincil_aralik | h13x_t4_h08c_h10_lpo/s4 kirp=1e-12 ∈ [1e-12,1e-12] |
| 🟢 OK | ikincil_aralik | h13x_m3_baseline_h02c_lpo/s4 kirp=1e-12 ∈ [1e-12,1e-12] |
| 🟢 OK | ikincil_aralik | h13x_abl_h10_hom2/s4 kirp=1e-12 ∈ [1e-12,1e-12] |
| 🟢 OK | ikincil_aralik | h13x_t3_h10_hom5/s0-4 kirp=1e-12 ∈ [1e-12,1e-12] |
| 🟢 OK | ikincil_aralik | h13x_t3_h02c_hom5/s0-4 kirp=1e-12 ∈ [1e-12,1e-12] |
| 🟢 OK | ikincil_aralik | h13x_m3_baseline_hom5/s0-4 kirp=1e-12 ∈ [1e-12,1e-12] |
| 🟢 OK | ikincil_aralik | h13x_h02c_h10_op/s0 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h13x_t2_h01c/s0 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h13x_t4_h08c_h10_lpo/s0 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h13x_m3_baseline_h02c_lpo/s0 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h13x_abl_h10_hom2/s0 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h13x_h02c_h10_op/s1 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h13x_t2_h01c/s1 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h13x_t4_h08c_h10_lpo/s1 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h13x_m3_baseline_h02c_lpo/s1 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h13x_abl_h10_hom2/s1 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h13x_h02c_h10_op/s2 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h13x_t2_h01c/s2 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h13x_t4_h08c_h10_lpo/s2 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h13x_m3_baseline_h02c_lpo/s2 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h13x_abl_h10_hom2/s2 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h13x_h02c_h10_op/s3 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h13x_t2_h01c/s3 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h13x_t4_h08c_h10_lpo/s3 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h13x_m3_baseline_h02c_lpo/s3 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h13x_abl_h10_hom2/s3 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h13x_h02c_h10_op/s4 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h13x_t2_h01c/s4 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h13x_t4_h08c_h10_lpo/s4 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h13x_m3_baseline_h02c_lpo/s4 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h13x_abl_h10_hom2/s4 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h13x_t3_h10_hom5/s0-4 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h13x_t3_h02c_hom5/s0-4 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h13x_m3_baseline_hom5/s0-4 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h13x_h02c_h10_op/s0 egitim_sure_sn=4.02 ∈ [0.0,300] |
| 🟢 OK | ikincil_aralik | h13x_t2_h01c/s0 egitim_sure_sn=4.131 ∈ [0.0,300] |
| 🟢 OK | ikincil_aralik | h13x_t4_h08c_h10_lpo/s0 egitim_sure_sn=4.121 ∈ [0.0,300] |
| 🟢 OK | ikincil_aralik | h13x_m3_baseline_h02c_lpo/s0 egitim_sure_sn=4.08 ∈ [0.0,300] |
| 🟢 OK | ikincil_aralik | h13x_abl_h10_hom2/s0 egitim_sure_sn=4.088 ∈ [0.0,300] |
| 🟢 OK | ikincil_aralik | h13x_h02c_h10_op/s1 egitim_sure_sn=4.094 ∈ [0.0,300] |
| 🟢 OK | ikincil_aralik | h13x_t2_h01c/s1 egitim_sure_sn=4.072 ∈ [0.0,300] |
| 🟢 OK | ikincil_aralik | h13x_t4_h08c_h10_lpo/s1 egitim_sure_sn=4.08 ∈ [0.0,300] |
| 🟢 OK | ikincil_aralik | h13x_m3_baseline_h02c_lpo/s1 egitim_sure_sn=4.1 ∈ [0.0,300] |
| 🟢 OK | ikincil_aralik | h13x_abl_h10_hom2/s1 egitim_sure_sn=4.088 ∈ [0.0,300] |
| 🟢 OK | ikincil_aralik | h13x_h02c_h10_op/s2 egitim_sure_sn=4.077 ∈ [0.0,300] |
| 🟢 OK | ikincil_aralik | h13x_t2_h01c/s2 egitim_sure_sn=4.277 ∈ [0.0,300] |
| 🟢 OK | ikincil_aralik | h13x_t4_h08c_h10_lpo/s2 egitim_sure_sn=4.142 ∈ [0.0,300] |
| 🟢 OK | ikincil_aralik | h13x_m3_baseline_h02c_lpo/s2 egitim_sure_sn=4.101 ∈ [0.0,300] |
| 🟢 OK | ikincil_aralik | h13x_abl_h10_hom2/s2 egitim_sure_sn=4.092 ∈ [0.0,300] |
| 🟢 OK | ikincil_aralik | h13x_h02c_h10_op/s3 egitim_sure_sn=4.071 ∈ [0.0,300] |
| 🟢 OK | ikincil_aralik | h13x_t2_h01c/s3 egitim_sure_sn=4.1 ∈ [0.0,300] |
| 🟢 OK | ikincil_aralik | h13x_t4_h08c_h10_lpo/s3 egitim_sure_sn=4.076 ∈ [0.0,300] |
| 🟢 OK | ikincil_aralik | h13x_m3_baseline_h02c_lpo/s3 egitim_sure_sn=4.086 ∈ [0.0,300] |
| 🟢 OK | ikincil_aralik | h13x_abl_h10_hom2/s3 egitim_sure_sn=4.116 ∈ [0.0,300] |
| 🟢 OK | ikincil_aralik | h13x_h02c_h10_op/s4 egitim_sure_sn=4.101 ∈ [0.0,300] |
| 🟢 OK | ikincil_aralik | h13x_t2_h01c/s4 egitim_sure_sn=4.079 ∈ [0.0,300] |
| 🟢 OK | ikincil_aralik | h13x_t4_h08c_h10_lpo/s4 egitim_sure_sn=4.115 ∈ [0.0,300] |
| 🟢 OK | ikincil_aralik | h13x_m3_baseline_h02c_lpo/s4 egitim_sure_sn=4.064 ∈ [0.0,300] |
| 🟢 OK | ikincil_aralik | h13x_abl_h10_hom2/s4 egitim_sure_sn=4.254 ∈ [0.0,300] |
| 🟢 OK | ikincil_aralik | h13x_t3_h10_hom5/s0-4 egitim_sure_sn=4.138 ∈ [0.0,300] |
| 🟢 OK | ikincil_aralik | h13x_t3_h02c_hom5/s0-4 egitim_sure_sn=4.085 ∈ [0.0,300] |
| 🟢 OK | ikincil_aralik | h13x_m3_baseline_hom5/s0-4 egitim_sure_sn=4.118 ∈ [0.0,300] |
