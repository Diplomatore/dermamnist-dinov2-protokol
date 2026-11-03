# Beklenti Kapısı — kuyruk `kc1-tekrar` aşama 3 → **GEÇTİ**
*2026-10-11 15:26* · metrik `val_acc` (max) · beklenti: `kc1-tekrar.beklenti.json` · 20 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h01_dinov2b_lp_r224_colab | 5 | 0.8223 | 0.8245, 0.8215, 0.8215, 0.8215, 0.8225 | 179.1 |
| h02_r18in_ft224_colab | 5 | 0.8656 | 0.8684, 0.8664, 0.8644, 0.8654, 0.8634 | 443.7 |
| h08_r18in_ft224_d4_colab | 5 | 0.8865 | 0.8843, 0.8893, 0.8903, 0.8863, 0.8824 | 448.9 |
| baseline_colab | 5 | 0.7458 | 0.7547, 0.7498, 0.7478, 0.7368, 0.7398 | 752.6 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 20 başarılı koşum (en az 20) |
| 🟢 OK | kollar_farkli | kollar birbirinden farklı |
| 🟢 OK | aralik | baseline_colab: ort 0.7458 ∈ [0.7196, 0.7596] (SIRALAMA ön-kaydı: |val_acc_Colab − 0,7396| ≤ 2,0 puan (yerel seed std 2,05)) |
| 🟢 OK | aralik | h01_dinov2b_lp_r224_colab: ort 0.8223 ∈ [0.8073, 0.8373] (SIRALAMA ön-kaydı: |Δ| ≤ 1,5 puan, yerel 0,8223) |
| 🟢 OK | aralik | h02_r18in_ft224_colab: ort 0.8656 ∈ [0.8522, 0.8822] (SIRALAMA ön-kaydı: |Δ| ≤ 1,5 puan, yerel 0,8672) |
| 🟢 OK | aralik | h08_r18in_ft224_d4_colab: ort 0.8865 ∈ [0.8743, 0.9043] (SIRALAMA ön-kaydı: |Δ| ≤ 1,5 puan, yerel 0,8893) |
| 🟢 OK | ikincil_aralik | baseline_colab/s0 test_acc=0.7436 ∈ [0.7098,0.766] |
| 🟢 OK | ikincil_aralik | baseline_colab/s1 test_acc=0.7317 ∈ [0.7098,0.766] |
| 🟢 OK | ikincil_aralik | baseline_colab/s2 test_acc=0.7411 ∈ [0.7098,0.766] |
| 🟢 OK | ikincil_aralik | baseline_colab/s3 test_acc=0.7441 ∈ [0.7098,0.766] |
| 🟢 OK | ikincil_aralik | baseline_colab/s4 test_acc=0.7367 ∈ [0.7098,0.766] |
| 🟢 OK | ikincil_aralik | baseline_colab/s0 test_auc=0.9142 ∈ [0.897,0.937] |
| 🟢 OK | ikincil_aralik | baseline_colab/s1 test_auc=0.9127 ∈ [0.897,0.937] |
| 🟢 OK | ikincil_aralik | baseline_colab/s2 test_auc=0.9173 ∈ [0.897,0.937] |
| 🟢 OK | ikincil_aralik | baseline_colab/s3 test_auc=0.9116 ∈ [0.897,0.937] |
| 🟢 OK | ikincil_aralik | baseline_colab/s4 test_auc=0.918 ∈ [0.897,0.937] |
| 🟢 OK | ikincil_aralik | baseline_colab/s0 secim_val_auc_farki=0 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | baseline_colab/s1 secim_val_auc_farki=0 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | baseline_colab/s2 secim_val_auc_farki=0 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | baseline_colab/s3 secim_val_auc_farki=0 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | baseline_colab/s4 secim_val_auc_farki=0 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab/s0 val_auc=0.9563 ∈ [0.88,0.99] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab/s1 val_auc=0.9563 ∈ [0.88,0.99] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab/s2 val_auc=0.9563 ∈ [0.88,0.99] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab/s3 val_auc=0.9563 ∈ [0.88,0.99] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab/s4 val_auc=0.9561 ∈ [0.88,0.99] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_colab/s0 val_auc=0.9651 ∈ [0.88,0.99] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_colab/s1 val_auc=0.9691 ∈ [0.88,0.99] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_colab/s2 val_auc=0.9684 ∈ [0.88,0.99] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_colab/s3 val_auc=0.9765 ∈ [0.88,0.99] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_colab/s4 val_auc=0.9692 ∈ [0.88,0.99] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab/s0 val_auc=0.9782 ∈ [0.88,0.99] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab/s1 val_auc=0.9756 ∈ [0.88,0.99] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab/s2 val_auc=0.9811 ∈ [0.88,0.99] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab/s3 val_auc=0.9802 ∈ [0.88,0.99] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab/s4 val_auc=0.9761 ∈ [0.88,0.99] |
| 🟢 OK | ikincil_aralik | baseline_colab/s0 val_auc=0.9214 ∈ [0.88,0.99] |
| 🟢 OK | ikincil_aralik | baseline_colab/s1 val_auc=0.924 ∈ [0.88,0.99] |
| 🟢 OK | ikincil_aralik | baseline_colab/s2 val_auc=0.9188 ∈ [0.88,0.99] |
| 🟢 OK | ikincil_aralik | baseline_colab/s3 val_auc=0.9229 ∈ [0.88,0.99] |
| 🟢 OK | ikincil_aralik | baseline_colab/s4 val_auc=0.9245 ∈ [0.88,0.99] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab/s0 yakinsama_fark=0.002193 ∈ [0.0,0.005] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab/s0 seed_tahmin_en_kucuk_fark=0.04417 ∈ [1e-06,1.0] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab/s1 seed_tahmin_en_kucuk_fark=0.04105 ∈ [1e-06,1.0] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab/s2 seed_tahmin_en_kucuk_fark=0.04417 ∈ [1e-06,1.0] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab/s3 seed_tahmin_en_kucuk_fark=0.04878 ∈ [1e-06,1.0] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab/s4 seed_tahmin_en_kucuk_fark=0.04105 ∈ [1e-06,1.0] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab/s0 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab/s1 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab/s2 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab/s3 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab/s4 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_colab/s0 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_colab/s1 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_colab/s2 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_colab/s3 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_colab/s4 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab/s0 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab/s1 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab/s2 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab/s3 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab/s4 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_colab/s0 epoch_sure_ort_sn=14.07 ∈ [2.4,60] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_colab/s1 epoch_sure_ort_sn=14.08 ∈ [2.4,60] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_colab/s2 epoch_sure_ort_sn=14.08 ∈ [2.4,60] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_colab/s3 epoch_sure_ort_sn=14.08 ∈ [2.4,60] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_colab/s4 epoch_sure_ort_sn=14.1 ∈ [2.4,60] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab/s0 epoch_sure_ort_sn=14.27 ∈ [2.4,60] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab/s1 epoch_sure_ort_sn=14.27 ∈ [2.4,60] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab/s2 epoch_sure_ort_sn=14.25 ∈ [2.4,60] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab/s3 epoch_sure_ort_sn=14.26 ∈ [2.4,60] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab/s4 epoch_sure_ort_sn=14.22 ∈ [2.4,60] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab/s0 augment_toplam_ornek=2.102e+05 ∈ [210210,210210] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab/s1 augment_toplam_ornek=2.102e+05 ∈ [210210,210210] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab/s2 augment_toplam_ornek=2.102e+05 ∈ [210210,210210] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab/s3 augment_toplam_ornek=2.102e+05 ∈ [210210,210210] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab/s4 augment_toplam_ornek=2.102e+05 ∈ [210210,210210] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab/s0 augment_frekans_maks_sapma=0.0009883 ∈ [0.0,0.01] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab/s1 augment_frekans_maks_sapma=0.001554 ∈ [0.0,0.01] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab/s2 augment_frekans_maks_sapma=0.001564 ∈ [0.0,0.01] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab/s3 augment_frekans_maks_sapma=0.001198 ∈ [0.0,0.01] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab/s4 augment_frekans_maks_sapma=0.001343 ∈ [0.0,0.01] |
