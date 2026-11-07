# Beklenti Kapısı — kuyruk `e1-h14` aşama 1 → **GEÇTİ**
*2026-11-07 16:25* · metrik `val_acc` (max) · beklenti: `e1-h14.beklenti.json` · 3 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h14_c1_d4kapali_duman | 1 | 0.8918 | 0.8918 | 461.5 |
| h14_c1_duman | 1 | 0.8883 | 0.8883 | 77.3 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 2 başarılı koşum (en az 2) |
| 🟢 OK | kollar_farkli | kollar birbirinden farklı |
| 🟢 OK | aralik | h14_c1_d4kapali_duman: ort 0.8918 ∈ [0.88, 0.9] (h10_c1_duman_s0 val_acc 0,8918 ± ≈ 6 görüntü (Durum A'da tam eşit beklenir)) |
| 🟢 OK | aralik | h14_c1_duman: ort 0.8883 ∈ [0.68, 0.96] (k5-c1-B a1 ile aynı (C-val çoğunluk 0,782 − 0,10; üst §4.2 0,96)) |
| 🟢 OK | ikincil_aralik | h14_c1_d4kapali_duman/s0 kontrol_ilk_adim_loss_fark=0 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | h14_c1_d4kapali_duman/s0 kontrol_argmax_oran=1 ∈ [0.9947,1.0] |
| 🟢 OK | ikincil_aralik | h14_c1_d4kapali_duman/s0 kontrol_p_maks_fark=1.341e-06 ∈ [0.0,0.05] |
| 🟢 OK | ikincil_aralik | h14_c1_d4kapali_duman/s0 ilk_adim_loss=2.029 ∈ [1.5,2.6] |
| 🟢 OK | ikincil_aralik | h14_c1_duman/s0 ilk_adim_loss=2.061 ∈ [1.5,2.6] |
| 🟢 OK | ikincil_aralik | h14_c1_duman/s0 augment_d4=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h14_c1_duman/s0 augment_toplam_ornek=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h14_c1_d4kapali_duman/s0 onbellek_gorunum_sayisi=8 ∈ [8,8] |
| 🟢 OK | ikincil_aralik | h14_c1_duman/s0 onbellek_gorunum_sayisi=8 ∈ [8,8] |
| 🟢 OK | ikincil_aralik | h14_c1_d4kapali_duman/s0 onbellek_gorunum_tutarlilik_maks_fark=0.002195 ∈ [0.0,0.005] |
| 🟢 OK | ikincil_aralik | h14_c1_duman/s0 onbellek_gorunum_tutarlilik_maks_fark=0.002195 ∈ [0.0,0.005] |
| 🟢 OK | ikincil_aralik | h14_c1_d4kapali_duman/s0 onbellek_gorunum_ozdes_cift=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | h14_c1_duman/s0 onbellek_gorunum_ozdes_cift=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | h14_c1_d4kapali_duman/s0 onbellek_tutarlilik_maks_fark=0.001514 ∈ [0.0,0.005] |
| 🟢 OK | ikincil_aralik | h14_c1_duman/s0 onbellek_tutarlilik_maks_fark=0.001514 ∈ [0.0,0.005] |
| 🟢 OK | ikincil_aralik | h14_c1_d4kapali_duman/s0 disk_bos_gb_once=243.8 ∈ [40,100000] |
| 🟢 OK | ikincil_aralik | h14_c1_d4kapali_duman/s0 param_egitilen=2.836e+07 ∈ [28364551,28364551] |
| 🟢 OK | ikincil_aralik | h14_c1_duman/s0 param_egitilen=2.836e+07 ∈ [28364551,28364551] |
| 🟢 OK | ikincil_aralik | h14_c1_d4kapali_duman/s0 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h14_c1_duman/s0 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h14_c1_d4kapali_duman/s0 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h14_c1_duman/s0 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h14_c1_d4kapali_duman/s0 tepe_gpu_bellek_mb=2445 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h14_c1_duman/s0 tepe_gpu_bellek_mb=2444 ∈ [1,22500] |
