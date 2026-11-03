# Beklenti Kapısı — kuyruk `kc1-tekrar` aşama 1 → **GEÇTİ**
*2026-10-11 12:36* · metrik `val_acc` (max) · beklenti: `kc1-tekrar.beklenti.json` · 4 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| baseline_colab_duman | 1 | 0.6869 | 0.6869 | 32.5 |
| h01_dinov2b_lp_r224_colab_duman | 1 | 0.6523 | 0.6523 | 34.8 |
| h02_r18in_ft224_colab_duman | 1 | 0.8026 | 0.8026 | 35.1 |
| h08_r18in_ft224_d4_colab_duman | 1 | 0.8116 | 0.8116 | 35.5 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 4 başarılı koşum (en az 4) |
| 🟢 OK | aralik | baseline_colab_duman: ort 0.6869 ∈ [0.619, 0.8] (k3-baseline-yerel-v2.beklenti.json a1: alt = val çoğunluk tabanı 0,669 − 0,05 (3 epoch), üst = 28 px tavanı ~0,78 + marj (literatur-cipa §4e); yerel duman 0,7089) |
| 🟢 OK | aralik | h01_dinov2b_lp_r224_colab_duman: ort 0.6523 ∈ [0.55, 0.95] (tur1.beklenti.json a1 (KOMUTLAR §5): 256 train, tek C; yerel duman 0,652) |
| 🟢 OK | aralik | h02_r18in_ft224_colab_duman: ort 0.8026 ∈ [0.62, 0.9] (tur1.beklenti.json a1: çoğunluk 0,669 − 0,05; üst H02 görülmemiş 0,90; yerel duman 0,8116) |
| 🟢 OK | aralik | h08_r18in_ft224_d4_colab_duman: ort 0.8116 ∈ [0.62, 0.9] (tur3.beklenti.json a1 (H08); yerel duman 0,8096) |
| 🟢 OK | kollar_farkli | kollar birbirinden farklı |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_colab_duman/s0 ozdeslik_ilk_adim_loss_fark=2.384e-07 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab_duman/s0 ozdeslik_ilk_adim_loss_fark=3.576e-07 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | baseline_colab_duman/s0 secim_val_auc_farki=0 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | baseline_colab_duman/s0 val_auc=0.892 ∈ [0.6,0.99] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab_duman/s0 val_auc=0.8557 ∈ [0.6,0.99] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_colab_duman/s0 val_auc=0.9624 ∈ [0.6,0.99] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab_duman/s0 val_auc=0.9533 ∈ [0.6,0.99] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab_duman/s0 ayni_seed_tahmin_maks_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab_duman/s0 seed_tahmin_en_kucuk_fark=0.1389 ∈ [1e-06,1.0] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab_duman/s0 param_omurga=8.572e+07 ∈ [85724928,85724928] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab_duman/s0 oznitelik_boyut=768 ∈ [768,768] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab_duman/s0 oznitelik_determinizm_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab_duman/s0 oznitelik_img_s=124.9 ∈ [10,1000] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_colab_duman/s0 ilk_adim_loss=1.852 ∈ [1.5,2.6] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab_duman/s0 ilk_adim_loss=1.845 ∈ [1.5,2.6] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_colab_duman/s0 epoch_sure_ort_sn=14.34 ∈ [2.4,60] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab_duman/s0 epoch_sure_ort_sn=14.71 ∈ [2.4,60] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab_duman/s0 augment_d4=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab_duman/s0 augment_toplam_ornek=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | baseline_colab_duman/s0 egitim_sure_sn=27.2 ∈ [1,900] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab_duman/s0 egitim_sure_sn=31.98 ∈ [1,900] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_colab_duman/s0 egitim_sure_sn=14.35 ∈ [1,900] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab_duman/s0 egitim_sure_sn=14.72 ∈ [1,900] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_colab_duman/s0 tepe_gpu_bellek_mb=667.8 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_colab_duman/s0 tepe_gpu_bellek_mb=1669 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_colab_duman/s0 tepe_gpu_bellek_mb=1669 ∈ [1,22500] |
