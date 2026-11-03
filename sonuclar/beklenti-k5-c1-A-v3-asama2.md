# Beklenti Kapısı — kuyruk `k5-c1-A-v3` aşama 2 → **GEÇTİ**
*2026-10-12 00:00* · metrik `val_acc` (max) · beklenti: `k5-c1-A-v3.beklenti.json` · 6 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h01_c1_onucus | 2 | 0.8962 | 0.8953, 0.897 | 135.3 |
| h02_c1_onucus | 2 | 0.9197 | 0.918, 0.9215 | 95.2 |
| baseline_c1_onucus | 2 | 0.8482 | 0.8464, 0.8499 | 170.1 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 6 başarılı koşum (en az 6) |
| 🟢 OK | kollar_farkli | kollar birbirinden farklı |
| 🟢 OK | aralik | h01_c1_onucus: ort 0.8962 ∈ [0.74, 0.96] (5 epoch / iki-seed C*: §4.2 alt 0,78 − 0,04; üst 0,96) |
| 🟢 OK | aralik | h02_c1_onucus: ort 0.9197 ∈ [0.74, 0.96] (5 epoch / iki-seed C*: §4.2 alt 0,78 − 0,04; üst 0,96) |
| 🟢 OK | aralik | baseline_c1_onucus: ort 0.8482 ∈ [0.7, 0.9] (20 epoch: §4.2 alt 0,74 − 0,04 ön-uçuş payı; üst 0,90) |
| 🟢 OK | ikincil_aralik | h01_c1_onucus/s0 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h01_c1_onucus/s1 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h02_c1_onucus/s0 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h02_c1_onucus/s1 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h01_c1_onucus/s0 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h01_c1_onucus/s1 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h02_c1_onucus/s0 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h02_c1_onucus/s1 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h01_c1_onucus/s0 tepe_gpu_bellek_mb=92 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h01_c1_onucus/s1 tepe_gpu_bellek_mb=92 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h02_c1_onucus/s0 tepe_gpu_bellek_mb=1669 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h02_c1_onucus/s1 tepe_gpu_bellek_mb=1669 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | baseline_c1_onucus/s0 secim_val_auc_farki=0 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | baseline_c1_onucus/s1 secim_val_auc_farki=0 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h01_c1_onucus/s0 val_auc=0.9774 ∈ [0.8,0.995] |
| 🟢 OK | ikincil_aralik | h01_c1_onucus/s1 val_auc=0.977 ∈ [0.8,0.995] |
| 🟢 OK | ikincil_aralik | h02_c1_onucus/s0 val_auc=0.981 ∈ [0.8,0.995] |
| 🟢 OK | ikincil_aralik | h02_c1_onucus/s1 val_auc=0.9832 ∈ [0.8,0.995] |
| 🟢 OK | ikincil_aralik | baseline_c1_onucus/s0 val_auc=0.9583 ∈ [0.8,0.995] |
| 🟢 OK | ikincil_aralik | baseline_c1_onucus/s1 val_auc=0.9588 ∈ [0.8,0.995] |
| 🟢 OK | ikincil_aralik | h01_c1_onucus/s0 yakinsama_fark=0.0008726 ∈ [0.0,0.005] |
| 🟢 OK | ikincil_aralik | h01_c1_onucus/s1 yakinsama_fark=0.0008726 ∈ [0.0,0.005] |
| 🟢 OK | ikincil_aralik | h01_c1_onucus/s0 seed_tahmin_en_kucuk_fark=0.05035 ∈ [1e-06,1.0] |
| 🟢 OK | ikincil_aralik | h01_c1_onucus/s1 seed_tahmin_en_kucuk_fark=0.05035 ∈ [1e-06,1.0] |
