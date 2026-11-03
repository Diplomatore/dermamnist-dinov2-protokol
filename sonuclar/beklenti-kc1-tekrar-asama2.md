# Beklenti Kapısı — kuyruk `kc1-tekrar` aşama 2 → **GEÇTİ**
*2026-10-11 12:53* · metrik `val_acc` (max) · beklenti: `kc1-tekrar.beklenti.json` · 8 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| baseline_colab_onucus | 2 | 0.7797 | 0.7836, 0.7757 | 158.6 |
| h01_dinov2b_lp_r224_colab_onucus | 2 | 0.823 | 0.8245, 0.8215 | 179.7 |
| h02_r18in_ft224_colab_onucus | 2 | 0.8599 | 0.8624, 0.8574 | 92.3 |
| h08_r18in_ft224_d4_colab_onucus | 2 | 0.8709 | 0.8674, 0.8744 | 93.2 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 8 başarılı koşum (en az 8) |
| 🟢 OK | kollar_farkli | kollar birbirinden farklı |
| 🟢 OK | aralik | baseline_colab_onucus: ort 0.7797 ∈ [0.669, 0.8] (k3 a2: alt val çoğunluk tabanı 0,669, üst 28 px tavanı + marj; yerel 0,7816/0,7817) |
| 🟢 OK | aralik | h01_dinov2b_lp_r224_colab_onucus: ort 0.823 ∈ [0.7596, 0.88] (tur1 a2 / H01 kartı: alt B + 0,02, üst görülmemiş 0,88; yerel 0,8245/0,8215) |
| 🟢 OK | aralik | h02_r18in_ft224_colab_onucus: ort 0.8599 ∈ [0.72, 0.9] (tur1 a2 (5 epoch); yerel 0,8604/0,8584) |
| 🟢 OK | aralik | h08_r18in_ft224_d4_colab_onucus: ort 0.8709 ∈ [0.72, 0.9] (tur3 a2 (5 epoch)) |
| 🟢 OK | ikincil_aralik | baseline_colab_onucus/s0 val_auc=0.9296 ∈ [0.85,0.99] |
| 🟢 OK | ikincil_aralik | baseline_colab_onucus/s1 val_auc=0.9313 ∈ [0.85,0.99] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab_onucus/s0 val_auc=0.9563 ∈ [0.85,0.99] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab_onucus/s1 val_auc=0.9563 ∈ [0.85,0.99] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_colab_onucus/s0 val_auc=0.9657 ∈ [0.85,0.99] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_colab_onucus/s1 val_auc=0.9736 ∈ [0.85,0.99] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab_onucus/s0 val_auc=0.9746 ∈ [0.85,0.99] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab_onucus/s1 val_auc=0.9763 ∈ [0.85,0.99] |
| 🟢 OK | ikincil_aralik | baseline_colab_onucus/s0 secim_val_auc_farki=0 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | baseline_colab_onucus/s1 secim_val_auc_farki=0 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab_onucus/s0 yakinsama_fark=0.001496 ∈ [0.0,0.005] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab_onucus/s1 yakinsama_fark=0.001496 ∈ [0.0,0.005] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab_onucus/s0 seed_tahmin_en_kucuk_fark=0.05358 ∈ [1e-06,1.0] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab_onucus/s1 seed_tahmin_en_kucuk_fark=0.05358 ∈ [1e-06,1.0] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab_onucus/s0 ayni_seed_tahmin_maks_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab_onucus/s1 ayni_seed_tahmin_maks_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab_onucus/s0 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab_onucus/s1 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_colab_onucus/s0 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_colab_onucus/s1 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab_onucus/s0 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab_onucus/s1 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab_onucus/s0 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab_onucus/s1 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_colab_onucus/s0 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_colab_onucus/s1 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab_onucus/s0 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab_onucus/s1 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab_onucus/s0 param_omurga=8.572e+07 ∈ [85724928,85724928] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab_onucus/s1 param_omurga=8.572e+07 ∈ [85724928,85724928] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab_onucus/s0 oznitelik_boyut=768 ∈ [768,768] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab_onucus/s1 oznitelik_boyut=768 ∈ [768,768] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab_onucus/s0 oznitelik_determinizm_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab_onucus/s1 oznitelik_determinizm_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab_onucus/s0 oznitelik_img_s=139 ∈ [10,1000] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab_onucus/s1 oznitelik_img_s=139 ∈ [10,1000] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_colab_onucus/s0 ilk_adim_loss=1.852 ∈ [1.5,2.6] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_colab_onucus/s1 ilk_adim_loss=2.134 ∈ [1.5,2.6] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab_onucus/s0 ilk_adim_loss=1.845 ∈ [1.5,2.6] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab_onucus/s1 ilk_adim_loss=2.117 ∈ [1.5,2.6] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_colab_onucus/s0 epoch_sure_ort_sn=14.3 ∈ [2.4,60] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_colab_onucus/s1 epoch_sure_ort_sn=14.27 ∈ [2.4,60] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab_onucus/s0 epoch_sure_ort_sn=14.44 ∈ [2.4,60] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab_onucus/s1 epoch_sure_ort_sn=14.42 ∈ [2.4,60] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_colab_onucus/s0 val_acc_aucsecim=0.8594 ∈ [0.69,0.9] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_colab_onucus/s1 val_acc_aucsecim=0.8574 ∈ [0.69,0.9] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab_onucus/s0 val_acc_aucsecim=0.8674 ∈ [0.69,0.9] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab_onucus/s1 val_acc_aucsecim=0.8744 ∈ [0.69,0.9] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab_onucus/s0 augment_d4=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab_onucus/s1 augment_d4=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab_onucus/s0 augment_toplam_ornek=3.504e+04 ∈ [35035,35035] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab_onucus/s1 augment_toplam_ornek=3.504e+04 ∈ [35035,35035] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab_onucus/s0 augment_frekans_maks_sapma=0.001873 ∈ [0.0,0.01] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab_onucus/s1 augment_frekans_maks_sapma=0.003322 ∈ [0.0,0.01] |
| 🟢 OK | ikincil_aralik | baseline_colab_onucus/s0 egitim_sure_sn=154.3 ∈ [10,1500] |
| 🟢 OK | ikincil_aralik | baseline_colab_onucus/s1 egitim_sure_sn=152.5 ∈ [10,1500] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab_onucus/s0 egitim_sure_sn=213.9 ∈ [10,1500] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab_onucus/s1 egitim_sure_sn=138.2 ∈ [10,1500] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_colab_onucus/s0 egitim_sure_sn=71.58 ∈ [10,1500] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_colab_onucus/s1 egitim_sure_sn=71.4 ∈ [10,1500] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab_onucus/s0 egitim_sure_sn=72.27 ∈ [10,1500] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab_onucus/s1 egitim_sure_sn=72.16 ∈ [10,1500] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab_onucus/s0 tepe_gpu_bellek_mb=667.8 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab_onucus/s1 tepe_gpu_bellek_mb=89.5 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_colab_onucus/s0 tepe_gpu_bellek_mb=1669 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_colab_onucus/s1 tepe_gpu_bellek_mb=1669 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab_onucus/s0 tepe_gpu_bellek_mb=1669 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab_onucus/s1 tepe_gpu_bellek_mb=1669 ∈ [1,22500] |
