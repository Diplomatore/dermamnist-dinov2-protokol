# Beklenti Kapısı — kuyruk `e3-onisleme` aşama 1 → **GEÇTİ**
*2026-11-08 15:53* · metrik `val_acc` (max) · beklenti: `e3-onisleme.beklenti.json` · 2 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| e3_h02c_duman | 1 | 0.8684 | 0.8684 | 14.4 |
| e3_h10_duman | 1 | 0.8943 | 0.8943 | 89.1 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 2 başarılı koşum (en az 2) |
| 🟢 OK | aralik | e3_h02c_duman: ort 0.8684 ∈ [0.83, 0.9] (H15 a2 H02c) |
| 🟢 OK | aralik | e3_h10_duman: ort 0.8943 ∈ [0.86, 0.93] (H15 a2 H10 (val 0,8947)) |
| 🟢 OK | ikincil_aralik | e3_h02c_duman/s0 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e3_h10_duman/s0 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e3_h02c_duman/s0 ozdeslik_p_maks_fark=9.477e-06 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e3_h10_duman/s0 ozdeslik_p_maks_fark=8.507e-05 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e3_h02c_duman/s0 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | e3_h10_duman/s0 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | e3_h02c_duman/s0 hat_n=1003 ∈ [993,1003] |
| 🟢 OK | ikincil_aralik | e3_h10_duman/s0 hat_n=1003 ∈ [993,1003] |
| 🟢 OK | ikincil_aralik | e3_h02c_duman/s0 hat_gecis_sayisi=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e3_h10_duman/s0 hat_gecis_sayisi=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e3_h02c_duman/s0 hat_girdi_sha_eslesti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e3_h10_duman/s0 hat_girdi_sha_eslesti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e3_h02c_duman/s0 hat_tahmin_sonlu=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e3_h10_duman/s0 hat_tahmin_sonlu=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e3_h02c_duman/s0 hat_olasilik_toplam_maks_sapma=1.669e-07 ∈ [0.0,0.0001] |
| 🟢 OK | ikincil_aralik | e3_h10_duman/s0 hat_olasilik_toplam_maks_sapma=1.601e-07 ∈ [0.0,0.0001] |
| 🟢 OK | ikincil_aralik | e3_h02c_duman/s0 hat_tahmin_sinif_maks_pay=0.7198 ∈ [0.0,0.95] |
| 🟢 OK | ikincil_aralik | e3_h10_duman/s0 hat_tahmin_sinif_maks_pay=0.6959 ∈ [0.0,0.95] |
| 🟢 OK | ikincil_aralik | e3_h02c_duman/s0 hat_argmax_uyum=0.9561 ∈ [0.5,1.0] |
| 🟢 OK | ikincil_aralik | e3_h10_duman/s0 hat_argmax_uyum=0.9651 ∈ [0.5,1.0] |
| 🟢 OK | ikincil_aralik | e3_h02c_duman/s0 hat_etiketle_degerlendirildi=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e3_h10_duman/s0 hat_etiketle_degerlendirildi=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e3_h02c_duman/s0 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e3_h10_duman/s0 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e3_h02c_duman/s0 bcn_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e3_h10_duman/s0 bcn_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e3_h02c_duman/s0 c1_kol=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e3_h10_duman/s0 c1_kol=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e3_h02c_duman/s0 tepe_gpu_bellek_mb=505.6 ∈ [1,3800] |
| 🟢 OK | ikincil_aralik | e3_h10_duman/s0 tepe_gpu_bellek_mb=489.2 ∈ [1,3800] |
