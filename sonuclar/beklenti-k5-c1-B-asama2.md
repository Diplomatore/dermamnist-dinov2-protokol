# Beklenti Kapısı — kuyruk `k5-c1-B` aşama 2 → **GEÇTİ**
*2026-10-11 22:13* · metrik `val_acc` (max) · beklenti: `k5-c1-B.beklenti.json` · 4 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h08_c1_onucus | 2 | 0.925 | 0.925, 0.925 | 96.2 |
| h10_c1_onucus | 2 | 0.9267 | 0.9302, 0.9232 | 282.8 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 4 başarılı koşum (en az 4) |
| 🟢 OK | kollar_farkli | kollar birbirinden farklı |
| 🟢 OK | aralik | h08_c1_onucus: ort 0.925 ∈ [0.74, 0.96] (5 epoch / iki-seed C*: §4.2 alt 0,78 − 0,04; üst 0,96) |
| 🟢 OK | aralik | h10_c1_onucus: ort 0.9267 ∈ [0.74, 0.96] (5 epoch / iki-seed C*: §4.2 alt 0,78 − 0,04; üst 0,96) |
| 🟢 OK | ikincil_aralik | h08_c1_onucus/s0 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h08_c1_onucus/s1 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h10_c1_onucus/s0 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h10_c1_onucus/s1 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h08_c1_onucus/s0 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h08_c1_onucus/s1 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h10_c1_onucus/s0 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h10_c1_onucus/s1 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h08_c1_onucus/s0 tepe_gpu_bellek_mb=1669 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h08_c1_onucus/s1 tepe_gpu_bellek_mb=1669 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h10_c1_onucus/s0 tepe_gpu_bellek_mb=2432 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h10_c1_onucus/s1 tepe_gpu_bellek_mb=2432 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h10_c1_onucus/s0 onbellek_tutarlilik_maks_fark=0.001514 ∈ [0.0,0.005] |
| 🟢 OK | ikincil_aralik | h10_c1_onucus/s1 onbellek_tutarlilik_maks_fark=0.001514 ∈ [0.0,0.005] |
| 🟢 OK | ikincil_aralik | h10_c1_onucus/s0 param_egitilen=2.836e+07 ∈ [11180103,28364551] |
| 🟢 OK | ikincil_aralik | h10_c1_onucus/s1 param_egitilen=2.836e+07 ∈ [11180103,28364551] |
| 🟢 OK | ikincil_aralik | h08_c1_onucus/s0 val_auc=0.9886 ∈ [0.8,0.995] |
| 🟢 OK | ikincil_aralik | h08_c1_onucus/s1 val_auc=0.9885 ∈ [0.8,0.995] |
| 🟢 OK | ikincil_aralik | h10_c1_onucus/s0 val_auc=0.9804 ∈ [0.8,0.995] |
| 🟢 OK | ikincil_aralik | h10_c1_onucus/s1 val_auc=0.9818 ∈ [0.8,0.995] |
