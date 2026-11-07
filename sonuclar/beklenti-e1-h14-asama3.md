# Beklenti Kapısı — kuyruk `e1-h14` aşama 3 → **GEÇTİ**
*2026-11-07 17:47* · metrik `val_acc` (max) · beklenti: `e1-h14.beklenti.json` · 10 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h14_c1 | 5 | 0.9277 | 0.925, 0.9302, 0.918, 0.9354, 0.9302 | 832.6 |
| h14_c1-test | 5 | 0.9277 | 0.925, 0.9302, 0.918, 0.9354, 0.9302 | 30.3 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 10 başarılı koşum (en az 10) |
| 🟢 OK | farkli_seedler_farkli | h14_c1: seed'ler farklı |
| 🟢 OK | farkli_seedler_farkli | h14_c1-test: seed'ler farklı |
| 🟢 OK | aralik | h14_c1: ort 0.9277 ∈ [0.78, 0.96] (K5-ONKAYIT §4.2 M8) |
| 🟢 OK | aralik | h14_c1-test: ort 0.9277 ∈ [0.78, 0.96] (K5-ONKAYIT §4.2 M8 (C-test satırının val_acc'si = C-val yeniden çıkarımı)) |
| 🟢 OK | ikincil_aralik | h14_c1/s0 epoch_sure_ort_sn=54.84 ∈ [30,110] |
| 🟢 OK | ikincil_aralik | h14_c1/s1 epoch_sure_ort_sn=54.42 ∈ [30,110] |
| 🟢 OK | ikincil_aralik | h14_c1/s2 epoch_sure_ort_sn=54.08 ∈ [30,110] |
| 🟢 OK | ikincil_aralik | h14_c1/s3 epoch_sure_ort_sn=53.94 ∈ [30,110] |
| 🟢 OK | ikincil_aralik | h14_c1/s4 epoch_sure_ort_sn=54 ∈ [30,110] |
| 🟢 OK | ikincil_aralik | h14_c1/s0 augment_frekans_maks_sapma=0.002012 ∈ [0.0,0.01] |
| 🟢 OK | ikincil_aralik | h14_c1/s1 augment_frekans_maks_sapma=0.001736 ∈ [0.0,0.01] |
| 🟢 OK | ikincil_aralik | h14_c1/s2 augment_frekans_maks_sapma=0.002401 ∈ [0.0,0.01] |
| 🟢 OK | ikincil_aralik | h14_c1/s3 augment_frekans_maks_sapma=0.001657 ∈ [0.0,0.01] |
| 🟢 OK | ikincil_aralik | h14_c1/s4 augment_frekans_maks_sapma=0.001086 ∈ [0.0,0.01] |
| 🟢 OK | ikincil_aralik | h14_c1/s0 augment_toplam_ornek=1.232e+05 ∈ [123225,123225] |
| 🟢 OK | ikincil_aralik | h14_c1/s1 augment_toplam_ornek=1.232e+05 ∈ [123225,123225] |
| 🟢 OK | ikincil_aralik | h14_c1/s2 augment_toplam_ornek=1.232e+05 ∈ [123225,123225] |
| 🟢 OK | ikincil_aralik | h14_c1/s3 augment_toplam_ornek=1.232e+05 ∈ [123225,123225] |
| 🟢 OK | ikincil_aralik | h14_c1/s4 augment_toplam_ornek=1.232e+05 ∈ [123225,123225] |
| 🟢 OK | ikincil_aralik | h14_c1/s0 onbellek_gorunum_tutarlilik_maks_fark=0.002195 ∈ [0.0,0.005] |
| 🟢 OK | ikincil_aralik | h14_c1/s1 onbellek_gorunum_tutarlilik_maks_fark=0.002195 ∈ [0.0,0.005] |
| 🟢 OK | ikincil_aralik | h14_c1/s2 onbellek_gorunum_tutarlilik_maks_fark=0.002195 ∈ [0.0,0.005] |
| 🟢 OK | ikincil_aralik | h14_c1/s3 onbellek_gorunum_tutarlilik_maks_fark=0.002195 ∈ [0.0,0.005] |
| 🟢 OK | ikincil_aralik | h14_c1/s4 onbellek_gorunum_tutarlilik_maks_fark=0.002195 ∈ [0.0,0.005] |
| 🟢 OK | ikincil_aralik | h14_c1/s0 param_egitilen=2.836e+07 ∈ [28364551,28364551] |
| 🟢 OK | ikincil_aralik | h14_c1/s1 param_egitilen=2.836e+07 ∈ [28364551,28364551] |
| 🟢 OK | ikincil_aralik | h14_c1/s2 param_egitilen=2.836e+07 ∈ [28364551,28364551] |
| 🟢 OK | ikincil_aralik | h14_c1/s3 param_egitilen=2.836e+07 ∈ [28364551,28364551] |
| 🟢 OK | ikincil_aralik | h14_c1/s4 param_egitilen=2.836e+07 ∈ [28364551,28364551] |
| 🟢 OK | ikincil_aralik | h14_c1/s0 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h14_c1/s1 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h14_c1/s2 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h14_c1/s3 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h14_c1/s4 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h14_c1/s0 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h14_c1/s1 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h14_c1/s2 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h14_c1/s3 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h14_c1/s4 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h14_c1-test/s0 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h14_c1-test/s1 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h14_c1-test/s2 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h14_c1-test/s3 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h14_c1-test/s4 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h14_c1-test/s0 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h14_c1-test/s1 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h14_c1-test/s2 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h14_c1-test/s3 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h14_c1-test/s4 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h14_c1-test/s0 ozdeslik_p_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h14_c1-test/s1 ozdeslik_p_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h14_c1-test/s2 ozdeslik_p_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h14_c1-test/s3 ozdeslik_p_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h14_c1-test/s4 ozdeslik_p_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h14_c1-test/s0 veri_c=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h14_c1-test/s1 veri_c=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h14_c1-test/s2 veri_c=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h14_c1-test/s3 veri_c=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h14_c1-test/s4 veri_c=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h14_c1-test/s0 test_n=1227 ∈ [1227,1227] |
| 🟢 OK | ikincil_aralik | h14_c1-test/s1 test_n=1227 ∈ [1227,1227] |
| 🟢 OK | ikincil_aralik | h14_c1-test/s2 test_n=1227 ∈ [1227,1227] |
| 🟢 OK | ikincil_aralik | h14_c1-test/s3 test_n=1227 ∈ [1227,1227] |
| 🟢 OK | ikincil_aralik | h14_c1-test/s4 test_n=1227 ∈ [1227,1227] |
| 🟢 OK | ikincil_aralik | h14_c1-test/s0 test_sonlu=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h14_c1-test/s1 test_sonlu=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h14_c1-test/s2 test_sonlu=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h14_c1-test/s3 test_sonlu=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h14_c1-test/s4 test_sonlu=1 ∈ [1,1] |
