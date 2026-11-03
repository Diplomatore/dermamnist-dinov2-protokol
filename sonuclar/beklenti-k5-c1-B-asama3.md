# Beklenti Kapısı — kuyruk `k5-c1-B` aşama 3 → **GEÇTİ**
*2026-10-12 00:07* · metrik `val_acc` (max) · beklenti: `k5-c1-B.beklenti.json` · 20 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h08_c1 | 5 | 0.9309 | 0.9337, 0.9302, 0.9319, 0.9302, 0.9284 | 504.0 |
| h10_c1 | 5 | 0.926 | 0.9319, 0.925, 0.9197, 0.9284, 0.925 | 811.8 |
| h08_c1-test | 5 | 0.9309 | 0.9337, 0.9302, 0.9319, 0.9302, 0.9284 | 19.0 |
| h10_c1-test | 5 | 0.926 | 0.9319, 0.925, 0.9197, 0.9284, 0.925 | 29.8 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 20 başarılı koşum (en az 10) |
| 🟢 OK | aralik | h08_c1: ort 0.9309 ∈ [0.78, 0.96] (K5-ONKAYIT §4.2 M8 (C-val nv payı %78; ezber azaldığından alt uç geniş)) |
| 🟢 OK | aralik | h08_c1-test: ort 0.9309 ∈ [0.78, 0.96] (K5-ONKAYIT §4.2 M8 (C-val nv payı %78; ezber azaldığından alt uç geniş) (C-test satırının val_acc'si = aynı modelin C-val yeniden çıkarımı)) |
| 🟢 OK | aralik | h10_c1: ort 0.926 ∈ [0.78, 0.96] (K5-ONKAYIT §4.2 M8 (C-val nv payı %78; ezber azaldığından alt uç geniş)) |
| 🟢 OK | aralik | h10_c1-test: ort 0.926 ∈ [0.78, 0.96] (K5-ONKAYIT §4.2 M8 (C-val nv payı %78; ezber azaldığından alt uç geniş) (C-test satırının val_acc'si = aynı modelin C-val yeniden çıkarımı)) |
| 🟢 OK | ikincil_aralik | h08_c1/s0 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h08_c1/s1 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h08_c1/s2 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h08_c1/s3 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h08_c1/s4 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h10_c1/s0 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h10_c1/s1 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h10_c1/s2 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h10_c1/s3 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h10_c1/s4 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h08_c1/s0 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h08_c1/s1 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h08_c1/s2 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h08_c1/s3 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h08_c1/s4 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h10_c1/s0 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h10_c1/s1 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h10_c1/s2 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h10_c1/s3 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h10_c1/s4 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h08_c1-test/s0 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h08_c1-test/s1 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h08_c1-test/s2 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h08_c1-test/s3 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h08_c1-test/s4 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h10_c1-test/s0 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h10_c1-test/s1 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h10_c1-test/s2 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h10_c1-test/s3 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h10_c1-test/s4 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h08_c1/s0 tepe_gpu_bellek_mb=1669 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h08_c1/s1 tepe_gpu_bellek_mb=1669 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h08_c1/s2 tepe_gpu_bellek_mb=1669 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h08_c1/s3 tepe_gpu_bellek_mb=1669 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h08_c1/s4 tepe_gpu_bellek_mb=1669 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h10_c1/s0 tepe_gpu_bellek_mb=2432 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h10_c1/s1 tepe_gpu_bellek_mb=2432 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h10_c1/s2 tepe_gpu_bellek_mb=2432 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h10_c1/s3 tepe_gpu_bellek_mb=2432 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h10_c1/s4 tepe_gpu_bellek_mb=2432 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h10_c1/s0 onbellek_tutarlilik_maks_fark=0.001514 ∈ [0.0,0.005] |
| 🟢 OK | ikincil_aralik | h10_c1/s1 onbellek_tutarlilik_maks_fark=0.001514 ∈ [0.0,0.005] |
| 🟢 OK | ikincil_aralik | h10_c1/s2 onbellek_tutarlilik_maks_fark=0.001514 ∈ [0.0,0.005] |
| 🟢 OK | ikincil_aralik | h10_c1/s3 onbellek_tutarlilik_maks_fark=0.001514 ∈ [0.0,0.005] |
| 🟢 OK | ikincil_aralik | h10_c1/s4 onbellek_tutarlilik_maks_fark=0.001514 ∈ [0.0,0.005] |
| 🟢 OK | ikincil_aralik | h10_c1/s0 param_egitilen=2.836e+07 ∈ [11180103,28364551] |
| 🟢 OK | ikincil_aralik | h10_c1/s1 param_egitilen=2.836e+07 ∈ [11180103,28364551] |
| 🟢 OK | ikincil_aralik | h10_c1/s2 param_egitilen=2.836e+07 ∈ [11180103,28364551] |
| 🟢 OK | ikincil_aralik | h10_c1/s3 param_egitilen=2.836e+07 ∈ [11180103,28364551] |
| 🟢 OK | ikincil_aralik | h10_c1/s4 param_egitilen=2.836e+07 ∈ [11180103,28364551] |
| 🟢 OK | ikincil_aralik | h08_c1-test/s0 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h08_c1-test/s1 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h08_c1-test/s2 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h08_c1-test/s3 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h08_c1-test/s4 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h10_c1-test/s0 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h10_c1-test/s1 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h10_c1-test/s2 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h10_c1-test/s3 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h10_c1-test/s4 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h08_c1-test/s0 ozdeslik_p_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h08_c1-test/s1 ozdeslik_p_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h08_c1-test/s2 ozdeslik_p_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h08_c1-test/s3 ozdeslik_p_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h08_c1-test/s4 ozdeslik_p_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h10_c1-test/s0 ozdeslik_p_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h10_c1-test/s1 ozdeslik_p_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h10_c1-test/s2 ozdeslik_p_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h10_c1-test/s3 ozdeslik_p_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h10_c1-test/s4 ozdeslik_p_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h08_c1-test/s0 test_n=1227 ∈ [1227,1227] |
| 🟢 OK | ikincil_aralik | h08_c1-test/s1 test_n=1227 ∈ [1227,1227] |
| 🟢 OK | ikincil_aralik | h08_c1-test/s2 test_n=1227 ∈ [1227,1227] |
| 🟢 OK | ikincil_aralik | h08_c1-test/s3 test_n=1227 ∈ [1227,1227] |
| 🟢 OK | ikincil_aralik | h08_c1-test/s4 test_n=1227 ∈ [1227,1227] |
| 🟢 OK | ikincil_aralik | h10_c1-test/s0 test_n=1227 ∈ [1227,1227] |
| 🟢 OK | ikincil_aralik | h10_c1-test/s1 test_n=1227 ∈ [1227,1227] |
| 🟢 OK | ikincil_aralik | h10_c1-test/s2 test_n=1227 ∈ [1227,1227] |
| 🟢 OK | ikincil_aralik | h10_c1-test/s3 test_n=1227 ∈ [1227,1227] |
| 🟢 OK | ikincil_aralik | h10_c1-test/s4 test_n=1227 ∈ [1227,1227] |
| 🟢 OK | ikincil_aralik | h08_c1-test/s0 test_sonlu=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h08_c1-test/s1 test_sonlu=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h08_c1-test/s2 test_sonlu=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h08_c1-test/s3 test_sonlu=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h08_c1-test/s4 test_sonlu=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h10_c1-test/s0 test_sonlu=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h10_c1-test/s1 test_sonlu=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h10_c1-test/s2 test_sonlu=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h10_c1-test/s3 test_sonlu=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h10_c1-test/s4 test_sonlu=1 ∈ [1,1] |
