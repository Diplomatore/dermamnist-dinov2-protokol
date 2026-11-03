# Beklenti Kapısı — kuyruk `kc2-tur4-h10` aşama 1 → **GEÇTİ**
*2026-10-11 15:53* · metrik `val_acc` (max) · beklenti: `kc2-tur4-h10.beklenti.json` · 1 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h10_dinov2b_k4_ft224_duman | 1 | 0.7956 | 0.7956 | 112.0 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 1 başarılı koşum (en az 1) |
| 🟢 OK | aralik | h10_dinov2b_k4_ft224_duman: ort 0.7956 ∈ [0.62, 0.93] (H10.md uygulama notu a1) |
| 🟢 OK | ikincil_aralik | h10_dinov2b_k4_ft224_duman/s0 param_egitilen=2.836e+07 ∈ [28364551,28364551] |
| 🟢 OK | ikincil_aralik | h10_dinov2b_k4_ft224_duman/s0 param_sayisi=8.573e+07 ∈ [85730311,85730311] |
| 🟢 OK | ikincil_aralik | h10_dinov2b_k4_ft224_duman/s0 ilk_adim_loss=1.713 ∈ [1.7117894759178163,1.7137894759178163] |
| 🟢 OK | ikincil_aralik | h10_dinov2b_k4_ft224_duman/s0 onbellek_tutarlilik_maks_fark=0.002426 ∈ [0.0,0.005] |
| 🟢 OK | ikincil_aralik | h10_dinov2b_k4_ft224_duman/s0 egitim_sure_sn=45.53 ∈ [20,600] |
| 🟢 OK | ikincil_aralik | h10_dinov2b_k4_ft224_duman/s0 tepe_gpu_bellek_mb=2432 ∈ [1,22500] |
