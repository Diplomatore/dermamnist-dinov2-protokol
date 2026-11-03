# Beklenti Kapısı — kuyruk `tur3` aşama 2 → **GEÇTİ**
*2026-10-11 04:59* · metrik `val_acc` (max) · beklenti: `tur3.beklenti.json` · 4 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h08_r18in_ft224_d4_onucus | 2 | 0.8699 | 0.8674, 0.8724 | 260.1 |
| h09_r18in_ft224_renk02_onucus | 2 | 0.8589 | 0.8574, 0.8604 | 267.5 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 4 başarılı koşum (en az 4) |
| 🟢 OK | farkli_seedler_farkli | h08_r18in_ft224_d4_onucus: seed'ler farklı |
| 🟢 OK | farkli_seedler_farkli | h09_r18in_ft224_renk02_onucus: seed'ler farklı |
| 🟢 OK | kollar_farkli | kollar birbirinden farklı |
| 🟢 OK | aralik | h08_r18in_ft224_d4_onucus: ort 0.8699 ∈ [0.72, 0.9] (H08/H09 a2: alt ≈ rakip testi 0,735 − 0,015, üst = H02 tavanı) |
| 🟢 OK | aralik | h09_r18in_ft224_renk02_onucus: ort 0.8589 ∈ [0.72, 0.9] (H08/H09 a2: alt ≈ rakip testi 0,735 − 0,015, üst = H02 tavanı) |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_onucus/s0 epoch_sure_ort_sn=47.68 ∈ [10,90] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_onucus/s1 epoch_sure_ort_sn=48.66 ∈ [10,90] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02_onucus/s0 epoch_sure_ort_sn=49.36 ∈ [10,90] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02_onucus/s1 epoch_sure_ort_sn=49.38 ∈ [10,90] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_onucus/s0 val_auc=0.9732 ∈ [0.88,0.99] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_onucus/s1 val_auc=0.9775 ∈ [0.88,0.99] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02_onucus/s0 val_auc=0.9706 ∈ [0.88,0.99] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02_onucus/s1 val_auc=0.9705 ∈ [0.88,0.99] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_onucus/s0 augment_frekans_maks_sapma=0.001873 ∈ [0.0,0.01] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_onucus/s1 augment_frekans_maks_sapma=0.003322 ∈ [0.0,0.01] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_onucus/s0 augment_toplam_ornek=3.504e+04 ∈ [35035,35035] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_onucus/s1 augment_toplam_ornek=3.504e+04 ∈ [35035,35035] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02_onucus/s0 renk_carpan_b_ort=1 ∈ [0.99,1.01] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02_onucus/s1 renk_carpan_b_ort=1.001 ∈ [0.99,1.01] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02_onucus/s0 renk_carpan_b_min=0.8 ∈ [0.8,0.85] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02_onucus/s1 renk_carpan_b_min=0.8 ∈ [0.8,0.85] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02_onucus/s0 renk_carpan_b_max=1.2 ∈ [1.15,1.2] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02_onucus/s1 renk_carpan_b_max=1.2 ∈ [1.15,1.2] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02_onucus/s0 renk_carpan_c_ort=1.001 ∈ [0.99,1.01] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02_onucus/s1 renk_carpan_c_ort=1 ∈ [0.99,1.01] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02_onucus/s0 renk_carpan_c_min=0.8 ∈ [0.8,0.85] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02_onucus/s1 renk_carpan_c_min=0.8 ∈ [0.8,0.85] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02_onucus/s0 renk_carpan_c_max=1.2 ∈ [1.15,1.2] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02_onucus/s1 renk_carpan_c_max=1.2 ∈ [1.15,1.2] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02_onucus/s0 renk_carpan_s_ort=0.9998 ∈ [0.99,1.01] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02_onucus/s1 renk_carpan_s_ort=1 ∈ [0.99,1.01] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02_onucus/s0 renk_carpan_s_min=0.8 ∈ [0.8,0.85] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02_onucus/s1 renk_carpan_s_min=0.8 ∈ [0.8,0.85] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02_onucus/s0 renk_carpan_s_max=1.2 ∈ [1.15,1.2] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02_onucus/s1 renk_carpan_s_max=1.2 ∈ [1.15,1.2] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_onucus/s0 param_sayisi=1.118e+07 ∈ [11180103,11180103] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_onucus/s1 param_sayisi=1.118e+07 ∈ [11180103,11180103] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02_onucus/s0 param_sayisi=1.118e+07 ∈ [11180103,11180103] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02_onucus/s1 param_sayisi=1.118e+07 ∈ [11180103,11180103] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_onucus/s0 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_onucus/s1 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02_onucus/s0 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02_onucus/s1 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_onucus/s0 tepe_gpu_bellek_mb=1742 ∈ [1,3900] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_onucus/s1 tepe_gpu_bellek_mb=1742 ∈ [1,3900] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02_onucus/s0 tepe_gpu_bellek_mb=1742 ∈ [1,3900] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02_onucus/s1 tepe_gpu_bellek_mb=1742 ∈ [1,3900] |
