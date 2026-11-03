# Beklenti Kapısı — kuyruk `k5-c1-A-v3` aşama 1 → **GEÇTİ**
*2026-10-11 23:46* · metrik `val_acc` (max) · beklenti: `k5-c1-A-v3.beklenti.json` · 4 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h01_c1_duman | 1 | 0.8866 | 0.8866 | 281.7 |
| h02_c1_duman | 1 | 0.9058 | 0.9058 | 31.6 |
| baseline_c1_duman | 1 | 0.808 | 0.808 | 34.5 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 3 başarılı koşum (en az 3) |
| 🟢 OK | kollar_farkli | kollar birbirinden farklı |
| 🟢 OK | aralik | h01_c1_duman: ort 0.8866 ∈ [0.68, 0.96] (C-val çoğunluk 0,782 − 0,10 (1 epoch / 256 örnek sonda); üst §4.2 0,96) |
| 🟢 OK | aralik | h02_c1_duman: ort 0.9058 ∈ [0.68, 0.96] (C-val çoğunluk 0,782 − 0,10 (1 epoch / 256 örnek sonda); üst §4.2 0,96) |
| 🟢 OK | aralik | baseline_c1_duman: ort 0.808 ∈ [0.68, 0.9] (C-val çoğunluk tabanı 0,782 (448/573) − 0,10 duman payı (3 epoch); üst = §4.2 baseline üst 0,90) |
| 🟢 OK | ikincil_aralik | h01_c1_duman/s0 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h02_c1_duman/s0 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h01_c1_duman/s0 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h02_c1_duman/s0 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h01_c1_duman/s0 tepe_gpu_bellek_mb=667.8 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h02_c1_duman/s0 tepe_gpu_bellek_mb=1669 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | baseline_c1_duman/s0 secim_val_auc_farki=0 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h02_c1_duman/s0 ilk_adim_loss=2.028 ∈ [1.5,2.6] |
| 🟢 OK | ikincil_aralik | h01_c1_duman/s0 param_omurga=8.572e+07 ∈ [85724928,85724928] |
| 🟢 OK | ikincil_aralik | h01_c1_duman/s0 oznitelik_determinizm_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h01_c1_duman/s0 ayni_seed_tahmin_maks_fark=0 ∈ [0.0,0.0] |
