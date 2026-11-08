# Beklenti Kapısı — kuyruk `e3-onisleme` aşama 2 → **GEÇTİ**
*2026-11-08 15:59* · metrik `val_acc` (max) · beklenti: `e3-onisleme.beklenti.json` · 4 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| e3_h10_onucus | 1 | 0.8943 | 0.8943 | 208.7 |
| e3_h02c_onucus | 1 | 0.8684 | 0.8684 | 23.2 |
| e3_h08c_onucus | 1 | 0.8843 | 0.8843 | 23.3 |
| e3_h11_onucus | 1 | 0.8654 | 0.8654 | 61.2 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 4 başarılı koşum (en az 4) |
| 🟢 OK | aralik | e3_h02c_onucus: ort 0.8684 ∈ [0.83, 0.9] (H15 a2) |
| 🟢 OK | aralik | e3_h08c_onucus: ort 0.8843 ∈ [0.85, 0.92] (H15 a2) |
| 🟢 OK | aralik | e3_h10_onucus: ort 0.8943 ∈ [0.86, 0.93] (H15 a2) |
| 🟢 OK | aralik | e3_h11_onucus: ort 0.8654 ∈ [0.84, 0.92] (H15 a2) |
| 🟢 OK | ikincil_aralik | e3_h10_onucus/s0 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e3_h02c_onucus/s0 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e3_h08c_onucus/s0 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e3_h11_onucus/s0 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e3_h10_onucus/s0 ozdeslik_p_maks_fark=8.507e-05 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e3_h02c_onucus/s0 ozdeslik_p_maks_fark=9.477e-06 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e3_h08c_onucus/s0 ozdeslik_p_maks_fark=6.914e-06 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e3_h11_onucus/s0 ozdeslik_p_maks_fark=2.45e-05 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e3_h10_onucus/s0 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | e3_h02c_onucus/s0 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | e3_h08c_onucus/s0 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | e3_h11_onucus/s0 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | e3_h10_onucus/s0 hat_n=1003 ∈ [993,1003] |
| 🟢 OK | ikincil_aralik | e3_h02c_onucus/s0 hat_n=1003 ∈ [993,1003] |
| 🟢 OK | ikincil_aralik | e3_h08c_onucus/s0 hat_n=1003 ∈ [993,1003] |
| 🟢 OK | ikincil_aralik | e3_h11_onucus/s0 hat_n=1003 ∈ [993,1003] |
| 🟢 OK | ikincil_aralik | e3_h10_onucus/s0 hat_gecis_sayisi=4 ∈ [4,4] |
| 🟢 OK | ikincil_aralik | e3_h02c_onucus/s0 hat_gecis_sayisi=4 ∈ [4,4] |
| 🟢 OK | ikincil_aralik | e3_h08c_onucus/s0 hat_gecis_sayisi=4 ∈ [4,4] |
| 🟢 OK | ikincil_aralik | e3_h11_onucus/s0 hat_gecis_sayisi=4 ∈ [4,4] |
| 🟢 OK | ikincil_aralik | e3_h10_onucus/s0 hat_girdi_sha_eslesti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e3_h02c_onucus/s0 hat_girdi_sha_eslesti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e3_h08c_onucus/s0 hat_girdi_sha_eslesti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e3_h11_onucus/s0 hat_girdi_sha_eslesti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e3_h10_onucus/s0 hat_tahmin_sonlu=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e3_h02c_onucus/s0 hat_tahmin_sonlu=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e3_h08c_onucus/s0 hat_tahmin_sonlu=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e3_h11_onucus/s0 hat_tahmin_sonlu=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e3_h10_onucus/s0 hat_olasilik_toplam_maks_sapma=1.801e-07 ∈ [0.0,0.0001] |
| 🟢 OK | ikincil_aralik | e3_h02c_onucus/s0 hat_olasilik_toplam_maks_sapma=1.836e-07 ∈ [0.0,0.0001] |
| 🟢 OK | ikincil_aralik | e3_h08c_onucus/s0 hat_olasilik_toplam_maks_sapma=1.885e-07 ∈ [0.0,0.0001] |
| 🟢 OK | ikincil_aralik | e3_h11_onucus/s0 hat_olasilik_toplam_maks_sapma=1.723e-07 ∈ [0.0,0.0001] |
| 🟢 OK | ikincil_aralik | e3_h10_onucus/s0 hat_tahmin_sinif_maks_pay=0.7926 ∈ [0.0,0.95] |
| 🟢 OK | ikincil_aralik | e3_h02c_onucus/s0 hat_tahmin_sinif_maks_pay=0.7398 ∈ [0.0,0.95] |
| 🟢 OK | ikincil_aralik | e3_h08c_onucus/s0 hat_tahmin_sinif_maks_pay=0.7119 ∈ [0.0,0.95] |
| 🟢 OK | ikincil_aralik | e3_h11_onucus/s0 hat_tahmin_sinif_maks_pay=0.6939 ∈ [0.0,0.95] |
| 🟢 OK | ikincil_aralik | e3_h10_onucus/s0 hat_argmax_uyum=0.9651 ∈ [0.5,1.0] |
| 🟢 OK | ikincil_aralik | e3_h02c_onucus/s0 hat_argmax_uyum=0.9561 ∈ [0.5,1.0] |
| 🟢 OK | ikincil_aralik | e3_h08c_onucus/s0 hat_argmax_uyum=0.9681 ∈ [0.5,1.0] |
| 🟢 OK | ikincil_aralik | e3_h11_onucus/s0 hat_argmax_uyum=0.9681 ∈ [0.5,1.0] |
| 🟢 OK | ikincil_aralik | e3_h10_onucus/s0 hat_etiketle_degerlendirildi=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e3_h02c_onucus/s0 hat_etiketle_degerlendirildi=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e3_h08c_onucus/s0 hat_etiketle_degerlendirildi=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e3_h11_onucus/s0 hat_etiketle_degerlendirildi=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e3_h10_onucus/s0 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e3_h02c_onucus/s0 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e3_h08c_onucus/s0 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e3_h11_onucus/s0 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e3_h10_onucus/s0 bcn_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e3_h02c_onucus/s0 bcn_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e3_h08c_onucus/s0 bcn_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e3_h11_onucus/s0 bcn_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e3_h10_onucus/s0 c1_kol=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e3_h02c_onucus/s0 c1_kol=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e3_h08c_onucus/s0 c1_kol=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e3_h11_onucus/s0 c1_kol=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e3_h10_onucus/s0 cikarim_sure_sn=192.3 ∈ [1,900] |
| 🟢 OK | ikincil_aralik | e3_h02c_onucus/s0 cikarim_sure_sn=8.3 ∈ [1,900] |
| 🟢 OK | ikincil_aralik | e3_h08c_onucus/s0 cikarim_sure_sn=8.3 ∈ [1,900] |
| 🟢 OK | ikincil_aralik | e3_h11_onucus/s0 cikarim_sure_sn=46.5 ∈ [1,900] |
| 🟢 OK | ikincil_aralik | e3_h10_onucus/s0 tepe_gpu_bellek_mb=489.2 ∈ [1,3800] |
| 🟢 OK | ikincil_aralik | e3_h02c_onucus/s0 tepe_gpu_bellek_mb=505.6 ∈ [1,3800] |
| 🟢 OK | ikincil_aralik | e3_h08c_onucus/s0 tepe_gpu_bellek_mb=505.6 ∈ [1,3800] |
| 🟢 OK | ikincil_aralik | e3_h11_onucus/s0 tepe_gpu_bellek_mb=172.1 ∈ [1,3800] |
