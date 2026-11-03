# Beklenti Kapısı — kuyruk `k5-c1-B` aşama 1 → **GEÇTİ**
*2026-10-11 22:00* · metrik `val_acc` (max) · beklenti: `k5-c1-B.beklenti.json` · 3 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h08_c1_duman | 1 | 0.9005 | 0.9005 | 31.1 |
| h10_c1_duman | 1 | 0.8918 | 0.8918 | 116.6 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 2 başarılı koşum (en az 2) |
| 🟢 OK | kollar_farkli | kollar birbirinden farklı |
| 🟢 OK | aralik | h08_c1_duman: ort 0.9005 ∈ [0.68, 0.96] (C-val çoğunluk 0,782 − 0,10 (1 epoch / 256 örnek sonda); üst §4.2 0,96) |
| 🟢 OK | aralik | h10_c1_duman: ort 0.8918 ∈ [0.68, 0.96] (C-val çoğunluk 0,782 − 0,10 (1 epoch / 256 örnek sonda); üst §4.2 0,96) |
| 🟢 OK | ikincil_aralik | h08_c1_duman/s0 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h10_c1_duman/s0 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h08_c1_duman/s0 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h10_c1_duman/s0 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h08_c1_duman/s0 tepe_gpu_bellek_mb=1669 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h10_c1_duman/s0 tepe_gpu_bellek_mb=2432 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h10_c1_duman/s0 onbellek_tutarlilik_maks_fark=0.001514 ∈ [0.0,0.005] |
| 🟢 OK | ikincil_aralik | h10_c1_duman/s0 param_egitilen=2.836e+07 ∈ [11180103,28364551] |
| 🟢 OK | ikincil_aralik | h08_c1_duman/s0 ilk_adim_loss=2.047 ∈ [1.5,2.6] |
| 🟢 OK | ikincil_aralik | h10_c1_duman/s0 ilk_adim_loss=2.029 ∈ [1.5,2.6] |
| 🟢 OK | ikincil_aralik | h08_c1_duman/s0 augment_d4=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h08_c1_duman/s0 augment_toplam_ornek=8215 ∈ [8215,8215] |
