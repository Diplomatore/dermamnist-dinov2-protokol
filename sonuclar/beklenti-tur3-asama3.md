# Beklenti Kapısı — kuyruk `tur3` aşama 3 → **GEÇTİ**
*2026-10-11 09:14* · metrik `val_acc` (max) · beklenti: `tur3.beklenti.json` · 10 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h08_r18in_ft224_d4 | 5 | 0.8893 | 0.8893, 0.8933, 0.8933, 0.8833, 0.8873 | 1495.2 |
| h09_r18in_ft224_renk02 | 5 | 0.8704 | 0.8714, 0.8714, 0.8744, 0.8644, 0.8704 | 1512.2 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 10 başarılı koşum (en az 10) |
| 🟢 OK | farkli_seedler_farkli | h08_r18in_ft224_d4: seed'ler farklı |
| 🟢 OK | farkli_seedler_farkli | h09_r18in_ft224_renk02: seed'ler farklı |
| 🟢 OK | kollar_farkli | kollar birbirinden farklı |
| 🟢 OK | aralik | h08_r18in_ft224_d4: ort 0.8893 ∈ [0.8, 0.9] (H08/H09 a3: alt 0,80 bozuk eğitim alarmı, üst 0,90 = H02 tavanı (gevşetilmez)) |
| 🟢 OK | aralik | h09_r18in_ft224_renk02: ort 0.8704 ∈ [0.8, 0.9] (H08/H09 a3: alt 0,80 bozuk eğitim alarmı, üst 0,90 = H02 tavanı (gevşetilmez)) |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4/s0 val_auc=0.9781 ∈ [0.93,0.99] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4/s1 val_auc=0.9761 ∈ [0.93,0.99] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4/s2 val_auc=0.978 ∈ [0.93,0.99] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4/s3 val_auc=0.9808 ∈ [0.93,0.99] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4/s4 val_auc=0.9796 ∈ [0.93,0.99] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s0 val_auc=0.9698 ∈ [0.93,0.99] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s1 val_auc=0.9714 ∈ [0.93,0.99] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s2 val_auc=0.9706 ∈ [0.93,0.99] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s3 val_auc=0.9682 ∈ [0.93,0.99] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s4 val_auc=0.9686 ∈ [0.93,0.99] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4/s0 epoch_sure_ort_sn=48.97 ∈ [10,90] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4/s1 epoch_sure_ort_sn=48.98 ∈ [10,90] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4/s2 epoch_sure_ort_sn=48.98 ∈ [10,90] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4/s3 epoch_sure_ort_sn=48.99 ∈ [10,90] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4/s4 epoch_sure_ort_sn=49.55 ∈ [10,90] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s0 epoch_sure_ort_sn=49.99 ∈ [10,90] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s1 epoch_sure_ort_sn=49.72 ∈ [10,90] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s2 epoch_sure_ort_sn=49.55 ∈ [10,90] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s3 epoch_sure_ort_sn=49.54 ∈ [10,90] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s4 epoch_sure_ort_sn=49.56 ∈ [10,90] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4/s0 augment_frekans_maks_sapma=0.0009883 ∈ [0.0,0.01] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4/s1 augment_frekans_maks_sapma=0.001554 ∈ [0.0,0.01] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4/s2 augment_frekans_maks_sapma=0.001564 ∈ [0.0,0.01] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4/s3 augment_frekans_maks_sapma=0.001198 ∈ [0.0,0.01] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4/s4 augment_frekans_maks_sapma=0.001343 ∈ [0.0,0.01] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4/s0 augment_toplam_ornek=2.102e+05 ∈ [210210,210210] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4/s1 augment_toplam_ornek=2.102e+05 ∈ [210210,210210] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4/s2 augment_toplam_ornek=2.102e+05 ∈ [210210,210210] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4/s3 augment_toplam_ornek=2.102e+05 ∈ [210210,210210] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4/s4 augment_toplam_ornek=2.102e+05 ∈ [210210,210210] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s0 renk_carpan_b_ort=1 ∈ [0.99,1.01] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s1 renk_carpan_b_ort=1 ∈ [0.99,1.01] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s2 renk_carpan_b_ort=0.9996 ∈ [0.99,1.01] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s3 renk_carpan_b_ort=0.9995 ∈ [0.99,1.01] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s4 renk_carpan_b_ort=1 ∈ [0.99,1.01] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s0 renk_carpan_b_min=0.8 ∈ [0.8,0.85] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s1 renk_carpan_b_min=0.8 ∈ [0.8,0.85] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s2 renk_carpan_b_min=0.8 ∈ [0.8,0.85] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s3 renk_carpan_b_min=0.8 ∈ [0.8,0.85] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s4 renk_carpan_b_min=0.8 ∈ [0.8,0.85] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s0 renk_carpan_b_max=1.2 ∈ [1.15,1.2] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s1 renk_carpan_b_max=1.2 ∈ [1.15,1.2] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s2 renk_carpan_b_max=1.2 ∈ [1.15,1.2] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s3 renk_carpan_b_max=1.2 ∈ [1.15,1.2] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s4 renk_carpan_b_max=1.2 ∈ [1.15,1.2] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s0 renk_carpan_c_ort=0.9999 ∈ [0.99,1.01] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s1 renk_carpan_c_ort=0.9999 ∈ [0.99,1.01] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s2 renk_carpan_c_ort=1 ∈ [0.99,1.01] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s3 renk_carpan_c_ort=0.9995 ∈ [0.99,1.01] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s4 renk_carpan_c_ort=1 ∈ [0.99,1.01] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s0 renk_carpan_c_min=0.8 ∈ [0.8,0.85] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s1 renk_carpan_c_min=0.8 ∈ [0.8,0.85] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s2 renk_carpan_c_min=0.8 ∈ [0.8,0.85] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s3 renk_carpan_c_min=0.8 ∈ [0.8,0.85] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s4 renk_carpan_c_min=0.8 ∈ [0.8,0.85] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s0 renk_carpan_c_max=1.2 ∈ [1.15,1.2] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s1 renk_carpan_c_max=1.2 ∈ [1.15,1.2] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s2 renk_carpan_c_max=1.2 ∈ [1.15,1.2] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s3 renk_carpan_c_max=1.2 ∈ [1.15,1.2] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s4 renk_carpan_c_max=1.2 ∈ [1.15,1.2] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s0 renk_carpan_s_ort=1 ∈ [0.99,1.01] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s1 renk_carpan_s_ort=0.9999 ∈ [0.99,1.01] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s2 renk_carpan_s_ort=0.9995 ∈ [0.99,1.01] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s3 renk_carpan_s_ort=0.9999 ∈ [0.99,1.01] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s4 renk_carpan_s_ort=1 ∈ [0.99,1.01] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s0 renk_carpan_s_min=0.8 ∈ [0.8,0.85] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s1 renk_carpan_s_min=0.8 ∈ [0.8,0.85] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s2 renk_carpan_s_min=0.8 ∈ [0.8,0.85] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s3 renk_carpan_s_min=0.8 ∈ [0.8,0.85] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s4 renk_carpan_s_min=0.8 ∈ [0.8,0.85] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s0 renk_carpan_s_max=1.2 ∈ [1.15,1.2] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s1 renk_carpan_s_max=1.2 ∈ [1.15,1.2] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s2 renk_carpan_s_max=1.2 ∈ [1.15,1.2] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s3 renk_carpan_s_max=1.2 ∈ [1.15,1.2] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s4 renk_carpan_s_max=1.2 ∈ [1.15,1.2] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4/s0 param_sayisi=1.118e+07 ∈ [11180103,11180103] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4/s1 param_sayisi=1.118e+07 ∈ [11180103,11180103] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4/s2 param_sayisi=1.118e+07 ∈ [11180103,11180103] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4/s3 param_sayisi=1.118e+07 ∈ [11180103,11180103] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4/s4 param_sayisi=1.118e+07 ∈ [11180103,11180103] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s0 param_sayisi=1.118e+07 ∈ [11180103,11180103] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s1 param_sayisi=1.118e+07 ∈ [11180103,11180103] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s2 param_sayisi=1.118e+07 ∈ [11180103,11180103] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s3 param_sayisi=1.118e+07 ∈ [11180103,11180103] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s4 param_sayisi=1.118e+07 ∈ [11180103,11180103] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4/s0 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4/s1 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4/s2 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4/s3 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4/s4 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s0 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s1 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s2 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s3 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02/s4 n_train=7007 ∈ [7007,7007] |
