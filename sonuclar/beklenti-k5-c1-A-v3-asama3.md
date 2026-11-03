# Beklenti Kapısı — kuyruk `k5-c1-A-v3` aşama 3 → **GEÇTİ**
*2026-10-12 02:10* · metrik `val_acc` (max) · beklenti: `k5-c1-A-v3.beklenti.json` · 25 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h01_c1 | 5 | 0.8946 | 0.8953, 0.897, 0.8935, 0.8935, 0.8935 | 204.2 |
| h02_c1 | 5 | 0.9201 | 0.925, 0.9162, 0.9197, 0.925, 0.9145 | 492.4 |
| baseline_c1 | 5 | 0.8419 | 0.8464, 0.8412, 0.8499, 0.8342, 0.8377 | 819.1 |
| h01_c1-test | 5 | 0.8946 | 0.8953, 0.897, 0.8935, 0.8935, 0.8935 | 18.7 |
| h02_c1-test | 5 | 0.9201 | 0.925, 0.9162, 0.9197, 0.925, 0.9145 | 15.7 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 25 başarılı koşum (en az 15) |
| 🟢 OK | aralik | h01_c1: ort 0.8946 ∈ [0.78, 0.96] (K5-ONKAYIT §4.2 M8 (C-val nv payı %78; ezber azaldığından alt uç geniş)) |
| 🟢 OK | aralik | h01_c1-test: ort 0.8946 ∈ [0.78, 0.96] (K5-ONKAYIT §4.2 M8 (C-val nv payı %78; ezber azaldığından alt uç geniş) (C-test satırının val_acc'si = aynı modelin C-val yeniden çıkarımı)) |
| 🟢 OK | aralik | h02_c1: ort 0.9201 ∈ [0.78, 0.96] (K5-ONKAYIT §4.2 M8 (C-val nv payı %78; ezber azaldığından alt uç geniş)) |
| 🟢 OK | aralik | h02_c1-test: ort 0.9201 ∈ [0.78, 0.96] (K5-ONKAYIT §4.2 M8 (C-val nv payı %78; ezber azaldığından alt uç geniş) (C-test satırının val_acc'si = aynı modelin C-val yeniden çıkarımı)) |
| 🟢 OK | aralik | baseline_c1: ort 0.8419 ∈ [0.74, 0.9] (K5-ONKAYIT §4.2 M8 (C-val nv payı %78; ezber azaldığından alt uç geniş)) |
| 🟢 OK | ikincil_aralik | h01_c1/s0 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h01_c1/s1 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h01_c1/s2 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h01_c1/s3 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h01_c1/s4 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h02_c1/s0 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h02_c1/s1 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h02_c1/s2 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h02_c1/s3 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h02_c1/s4 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h01_c1/s0 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h01_c1/s1 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h01_c1/s2 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h01_c1/s3 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h01_c1/s4 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h02_c1/s0 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h02_c1/s1 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h02_c1/s2 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h02_c1/s3 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h02_c1/s4 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h01_c1-test/s0 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h01_c1-test/s1 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h01_c1-test/s2 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h01_c1-test/s3 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h01_c1-test/s4 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h02_c1-test/s0 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h02_c1-test/s1 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h02_c1-test/s2 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h02_c1-test/s3 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h02_c1-test/s4 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h01_c1/s0 tepe_gpu_bellek_mb=92 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h01_c1/s1 tepe_gpu_bellek_mb=92 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h01_c1/s2 tepe_gpu_bellek_mb=92 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h01_c1/s3 tepe_gpu_bellek_mb=92 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h01_c1/s4 tepe_gpu_bellek_mb=92 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h02_c1/s0 tepe_gpu_bellek_mb=1669 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h02_c1/s1 tepe_gpu_bellek_mb=1669 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h02_c1/s2 tepe_gpu_bellek_mb=1669 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h02_c1/s3 tepe_gpu_bellek_mb=1669 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h02_c1/s4 tepe_gpu_bellek_mb=1669 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | baseline_c1/s0 secim_val_auc_farki=0 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | baseline_c1/s1 secim_val_auc_farki=0 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | baseline_c1/s2 secim_val_auc_farki=0 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | baseline_c1/s3 secim_val_auc_farki=0 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | baseline_c1/s4 secim_val_auc_farki=0 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h01_c1-test/s0 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h01_c1-test/s1 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h01_c1-test/s2 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h01_c1-test/s3 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h01_c1-test/s4 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h02_c1-test/s0 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h02_c1-test/s1 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h02_c1-test/s2 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h02_c1-test/s3 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h02_c1-test/s4 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h01_c1-test/s0 ozdeslik_p_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h01_c1-test/s1 ozdeslik_p_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h01_c1-test/s2 ozdeslik_p_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h01_c1-test/s3 ozdeslik_p_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h01_c1-test/s4 ozdeslik_p_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h02_c1-test/s0 ozdeslik_p_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h02_c1-test/s1 ozdeslik_p_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h02_c1-test/s2 ozdeslik_p_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h02_c1-test/s3 ozdeslik_p_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h02_c1-test/s4 ozdeslik_p_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h01_c1-test/s0 test_n=1227 ∈ [1227,1227] |
| 🟢 OK | ikincil_aralik | h01_c1-test/s1 test_n=1227 ∈ [1227,1227] |
| 🟢 OK | ikincil_aralik | h01_c1-test/s2 test_n=1227 ∈ [1227,1227] |
| 🟢 OK | ikincil_aralik | h01_c1-test/s3 test_n=1227 ∈ [1227,1227] |
| 🟢 OK | ikincil_aralik | h01_c1-test/s4 test_n=1227 ∈ [1227,1227] |
| 🟢 OK | ikincil_aralik | h02_c1-test/s0 test_n=1227 ∈ [1227,1227] |
| 🟢 OK | ikincil_aralik | h02_c1-test/s1 test_n=1227 ∈ [1227,1227] |
| 🟢 OK | ikincil_aralik | h02_c1-test/s2 test_n=1227 ∈ [1227,1227] |
| 🟢 OK | ikincil_aralik | h02_c1-test/s3 test_n=1227 ∈ [1227,1227] |
| 🟢 OK | ikincil_aralik | h02_c1-test/s4 test_n=1227 ∈ [1227,1227] |
| 🟢 OK | ikincil_aralik | h01_c1-test/s0 test_sonlu=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h01_c1-test/s1 test_sonlu=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h01_c1-test/s2 test_sonlu=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h01_c1-test/s3 test_sonlu=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h01_c1-test/s4 test_sonlu=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h02_c1-test/s0 test_sonlu=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h02_c1-test/s1 test_sonlu=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h02_c1-test/s2 test_sonlu=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h02_c1-test/s3 test_sonlu=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h02_c1-test/s4 test_sonlu=1 ∈ [1,1] |
