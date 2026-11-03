# Beklenti Kapısı — kuyruk `tur1` aşama 2 → **GEÇTİ**
*2026-10-10 20:57* · metrik `val_acc` (max) · beklenti: `tur1.beklenti.json` · 6 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h03_dinov2b_lp_r28_onucus | 2 | 0.7747 | 0.7737, 0.7757 | 95.3 |
| h01_dinov2b_lp_r224_onucus | 2 | 0.823 | 0.8245, 0.8215 | 103.0 |
| h02_r18in_ft224_onucus | 2 | 0.8594 | 0.8604, 0.8584 | 279.4 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 6 başarılı koşum (en az 6) |
| 🟢 OK | aralik | h03_dinov2b_lp_r28_onucus: ort 0.7747 ∈ [0.69, 0.8] (H03 kartı uygulama notu (r28 aralik [0,69; 0,80]): alt = operasyonel DUR 0,69 (çoğunluk 0,669 + 2), üst = r28 görülmemiş eşiği 0,80 (literatur-cipa §4b/e). Kart beklentisi [0,730; 0,780]'ın gevşetilmişi) |
| 🟢 OK | aralik | h01_dinov2b_lp_r224_onucus: ort 0.823 ∈ [0.7596, 0.88] (H01 kartı: alt = öldürme (b) eşiği B + 0,02 (hata-analizi §8), üst = görülmemiş eşiği 0,88 (literatur-cipa §4b). Kart beklentisi [0,820; 0,860]'ın gevşetilmişi; kart uygulama notu a2–3 aralığıyla aynı) |
| 🟢 OK | aralik | h02_r18in_ft224_onucus: ort 0.8594 ∈ [0.72, 0.9] (varyantlar/KOMUTLAR.md §5 h02 a2 (20:10, H02 sonucu yokken): 5 epoch kısaltılmış → alt ≈ rakip testi 0,735 − 0,015; üst = H02 görülmemiş eşiği 0,90 (gevşetilmez)) |
| 🟢 OK | kollar_farkli | kollar birbirinden farklı |
| 🟢 OK | ikincil_aralik | h03_dinov2b_lp_r28_onucus/s0 val_auc=0.9324 ∈ [0.88,0.99] |
| 🟢 OK | ikincil_aralik | h03_dinov2b_lp_r28_onucus/s1 val_auc=0.9323 ∈ [0.88,0.99] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_onucus/s0 val_auc=0.9563 ∈ [0.88,0.99] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_onucus/s1 val_auc=0.9563 ∈ [0.88,0.99] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_onucus/s0 val_auc=0.9665 ∈ [0.88,0.99] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_onucus/s1 val_auc=0.9737 ∈ [0.88,0.99] |
| 🟢 OK | ikincil_aralik | h03_dinov2b_lp_r28_onucus/s0 yakinsama_fark=0.004985 ∈ [0.0,0.005] |
| 🟢 OK | ikincil_aralik | h03_dinov2b_lp_r28_onucus/s1 yakinsama_fark=0.004985 ∈ [0.0,0.005] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_onucus/s0 yakinsama_fark=0.001496 ∈ [0.0,0.005] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_onucus/s1 yakinsama_fark=0.001496 ∈ [0.0,0.005] |
| 🟢 OK | ikincil_aralik | h03_dinov2b_lp_r28_onucus/s0 seed_tahmin_en_kucuk_fark=0.03659 ∈ [1e-06,1.0] |
| 🟢 OK | ikincil_aralik | h03_dinov2b_lp_r28_onucus/s1 seed_tahmin_en_kucuk_fark=0.03659 ∈ [1e-06,1.0] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_onucus/s0 seed_tahmin_en_kucuk_fark=0.05358 ∈ [1e-06,1.0] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_onucus/s1 seed_tahmin_en_kucuk_fark=0.05358 ∈ [1e-06,1.0] |
| 🟢 OK | ikincil_aralik | h03_dinov2b_lp_r28_onucus/s0 ayni_seed_tahmin_maks_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h03_dinov2b_lp_r28_onucus/s1 ayni_seed_tahmin_maks_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_onucus/s0 ayni_seed_tahmin_maks_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_onucus/s1 ayni_seed_tahmin_maks_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h03_dinov2b_lp_r28_onucus/s0 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h03_dinov2b_lp_r28_onucus/s1 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_onucus/s0 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_onucus/s1 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_onucus/s0 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_onucus/s1 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h03_dinov2b_lp_r28_onucus/s0 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h03_dinov2b_lp_r28_onucus/s1 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_onucus/s0 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_onucus/s1 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_onucus/s0 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_onucus/s1 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h03_dinov2b_lp_r28_onucus/s0 param_omurga=8.572e+07 ∈ [85724928,85724928] |
| 🟢 OK | ikincil_aralik | h03_dinov2b_lp_r28_onucus/s1 param_omurga=8.572e+07 ∈ [85724928,85724928] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_onucus/s0 param_omurga=8.572e+07 ∈ [85724928,85724928] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_onucus/s1 param_omurga=8.572e+07 ∈ [85724928,85724928] |
| 🟢 OK | ikincil_aralik | h03_dinov2b_lp_r28_onucus/s0 oznitelik_boyut=768 ∈ [768,768] |
| 🟢 OK | ikincil_aralik | h03_dinov2b_lp_r28_onucus/s1 oznitelik_boyut=768 ∈ [768,768] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_onucus/s0 oznitelik_boyut=768 ∈ [768,768] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_onucus/s1 oznitelik_boyut=768 ∈ [768,768] |
| 🟢 OK | ikincil_aralik | h03_dinov2b_lp_r28_onucus/s0 oznitelik_determinizm_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h03_dinov2b_lp_r28_onucus/s1 oznitelik_determinizm_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_onucus/s0 oznitelik_determinizm_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_onucus/s1 oznitelik_determinizm_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h03_dinov2b_lp_r28_onucus/s0 oznitelik_img_s=24.74 ∈ [10,1000] |
| 🟢 OK | ikincil_aralik | h03_dinov2b_lp_r28_onucus/s1 oznitelik_img_s=24.74 ∈ [10,1000] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_onucus/s0 oznitelik_img_s=25.4 ∈ [10,1000] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_onucus/s1 oznitelik_img_s=25.4 ∈ [10,1000] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_onucus/s0 epoch_sure_ort_sn=49.35 ∈ [10,90] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_onucus/s1 epoch_sure_ort_sn=51.68 ∈ [10,90] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_onucus/s0 ilk_adim_loss=1.852 ∈ [1.5,2.5] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_onucus/s1 ilk_adim_loss=2.134 ∈ [1.5,2.5] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_onucus/s0 val_acc_aucsecim=0.8584 ∈ [0.69,0.9] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_onucus/s1 val_acc_aucsecim=0.8584 ∈ [0.69,0.9] |
| 🟢 OK | ikincil_aralik | h03_dinov2b_lp_r28_onucus/s0 egitim_sure_sn=91.29 ∈ [10,900] |
| 🟢 OK | ikincil_aralik | h03_dinov2b_lp_r28_onucus/s1 egitim_sure_sn=89.54 ∈ [10,900] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_onucus/s0 egitim_sure_sn=97.37 ∈ [10,900] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_onucus/s1 egitim_sure_sn=97.73 ∈ [10,900] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_onucus/s0 egitim_sure_sn=246.8 ∈ [10,900] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_onucus/s1 egitim_sure_sn=258.4 ∈ [10,900] |
| 🟢 OK | ikincil_aralik | h03_dinov2b_lp_r28_onucus/s0 tepe_gpu_bellek_mb=42.5 ∈ [1,3900] |
| 🟢 OK | ikincil_aralik | h03_dinov2b_lp_r28_onucus/s1 tepe_gpu_bellek_mb=42.5 ∈ [1,3900] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_onucus/s0 tepe_gpu_bellek_mb=42.5 ∈ [1,3900] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_onucus/s1 tepe_gpu_bellek_mb=42.5 ∈ [1,3900] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_onucus/s0 tepe_gpu_bellek_mb=1742 ∈ [1,3900] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_onucus/s1 tepe_gpu_bellek_mb=1742 ∈ [1,3900] |
