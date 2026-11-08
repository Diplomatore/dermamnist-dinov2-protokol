# Beklenti Kapısı — kuyruk `e2-degerlendirme` aşama 1 → **GEÇTİ**
*2026-11-07 23:38* · metrik `val_acc` (max) · beklenti: `e2-degerlendirme.beklenti.json` · 3 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| e2_h10_duman | 1 | 0.8943 | 0.8943 | 37.9 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 1 başarılı koşum (en az 1) |
| 🟢 OK | aralik | e2_h10_duman: ort 0.8943 ∈ [0.86, 0.93] (H10 Colab val ort. 0,8947 (karar.md §1.2); özdeşlik ayrıca ozdeslik_* ile) |
| 🟢 OK | ikincil_aralik | e2_h10_duman/s0 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h10_duman/s0 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h10_duman/s0 ozdeslik_p_maks_fark=0 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h10_duman/s0 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h10_duman/s0 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h10_duman/s0 bcn_n=48 ∈ [48,48] |
| 🟢 OK | ikincil_aralik | e2_h10_duman/s0 bcn_tahmin_sonlu=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h10_duman/s0 bcn_olasilik_toplam_maks_sapma=1.29e-07 ∈ [0.0,0.0001] |
| 🟢 OK | ikincil_aralik | e2_h10_duman/s0 bcn_tahmin_sinif_maks_pay=0.4375 ∈ [0.0,0.95] |
| 🟢 OK | ikincil_aralik | e2_h10_duman/s0 kal_argmax_uyum=8 ∈ [7,8] |
| 🟢 OK | ikincil_aralik | e2_h10_duman/s0 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h10_duman/s0 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h10_duman/s0 tepe_gpu_bellek_mb=938.6 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | e2_h10_duman/s0 kal_girdi_sha_eslesti=1 ∈ [1,1] |
