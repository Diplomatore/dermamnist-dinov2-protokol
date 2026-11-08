# Beklenti Kapısı — kuyruk `e2-degerlendirme` aşama 2 → **GEÇTİ**
*2026-11-08 00:02* · metrik `val_acc` (max) · beklenti: `e2-degerlendirme.beklenti.json` · 55 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| e2_h10_onucus | 5 | 0.8947 | 0.8943, 0.8923, 0.9053, 0.8943, 0.8873 | 27.6 |
| e2_h02c_onucus | 5 | 0.8656 | 0.8684, 0.8664, 0.8644, 0.8654, 0.8634 | 19.4 |
| e2_h10_c1_onucus | 5 | 0.926 | 0.9319, 0.925, 0.9197, 0.9284, 0.925 | 25.9 |
| e2_h02_c1_onucus | 5 | 0.9201 | 0.925, 0.9162, 0.9197, 0.925, 0.9145 | 18.5 |
| e2_h08c_onucus | 5 | 0.8865 | 0.8843, 0.8893, 0.8903, 0.8863, 0.8824 | 18.5 |
| e2_h08_c1_onucus | 5 | 0.9309 | 0.9337, 0.9302, 0.9319, 0.9302, 0.9284 | 18.7 |
| e2_h14_c1_onucus | 5 | 0.9277 | 0.925, 0.9302, 0.918, 0.9354, 0.9302 | 29.3 |
| e2_h11_onucus | 5 | 0.88 | 0.8654, 0.8843, 0.8873, 0.8794, 0.8833 | 24.6 |
| e2_h01c_onucus | 5 | 0.8223 | 0.8245, 0.8215, 0.8215, 0.8215, 0.8225 | 28.1 |
| e2_h12_onucus | 5 | 0.8187 | 0.8195, 0.8185, 0.8175, 0.8185, 0.8195 | 20.5 |
| e2_h01_c1_onucus | 5 | 0.8946 | 0.8953, 0.897, 0.8935, 0.8935, 0.8935 | 28.4 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 55 başarılı koşum (en az 55) |
| 🟢 OK | farkli_seedler_farkli | e2_h10_onucus: seed'ler farklı |
| 🟢 OK | farkli_seedler_farkli | e2_h02c_onucus: seed'ler farklı |
| 🟢 OK | farkli_seedler_farkli | e2_h10_c1_onucus: seed'ler farklı |
| 🟢 OK | farkli_seedler_farkli | e2_h02_c1_onucus: seed'ler farklı |
| 🟢 OK | farkli_seedler_farkli | e2_h08c_onucus: seed'ler farklı |
| 🟢 OK | farkli_seedler_farkli | e2_h08_c1_onucus: seed'ler farklı |
| 🟢 OK | farkli_seedler_farkli | e2_h14_c1_onucus: seed'ler farklı |
| 🟢 OK | farkli_seedler_farkli | e2_h11_onucus: seed'ler farklı |
| 🟢 OK | farkli_seedler_farkli | e2_h01c_onucus: seed'ler farklı |
| 🟢 OK | farkli_seedler_farkli | e2_h12_onucus: seed'ler farklı |
| 🟢 OK | farkli_seedler_farkli | e2_h01_c1_onucus: seed'ler farklı |
| 🟢 OK | aralik | e2_h01c_onucus: ort 0.8223 ∈ [0.79, 0.85] (H01c Colab val 0,8223) |
| 🟢 OK | aralik | e2_h02c_onucus: ort 0.8656 ∈ [0.83, 0.9] (H02c Colab val 0,8656) |
| 🟢 OK | aralik | e2_h08c_onucus: ort 0.8865 ∈ [0.85, 0.92] (H08c Colab val 0,8865) |
| 🟢 OK | aralik | e2_h10_onucus: ort 0.8947 ∈ [0.86, 0.93] (H10 Colab val 0,8947) |
| 🟢 OK | aralik | e2_h11_onucus: ort 0.88 ∈ [0.84, 0.92] (H11 Colab val 0,8800) |
| 🟢 OK | aralik | e2_h12_onucus: ort 0.8187 ∈ [0.79, 0.85] (H12 Colab val 0,8187) |
| 🟢 OK | aralik | e2_h01_c1_onucus: ort 0.8946 ∈ [0.82, 0.94] (C-val; h01_c1 değeri kartta yok, geniş alarm) |
| 🟢 OK | aralik | e2_h02_c1_onucus: ort 0.9201 ∈ [0.86, 0.96] (C-val; h02_c1 değeri kartta yok, geniş alarm (üst K5-ONKAYIT §4.2 0,96)) |
| 🟢 OK | aralik | e2_h08_c1_onucus: ort 0.9309 ∈ [0.9, 0.96] (h08_c1 C-val 0,9309) |
| 🟢 OK | aralik | e2_h10_c1_onucus: ort 0.926 ∈ [0.9, 0.95] (h10_c1 C-val 0,926) |
| 🟢 OK | aralik | e2_h14_c1_onucus: ort 0.9277 ∈ [0.9, 0.96] (h14_c1 C-val 0,9277) |
| 🟢 OK | ikincil_aralik | e2_h10_onucus/s0 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h10_onucus/s1 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h10_onucus/s2 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h10_onucus/s3 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h10_onucus/s4 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h02c_onucus/s0 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h02c_onucus/s1 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h02c_onucus/s2 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h02c_onucus/s3 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h02c_onucus/s4 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h10_c1_onucus/s0 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h10_c1_onucus/s1 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h10_c1_onucus/s2 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h10_c1_onucus/s3 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h10_c1_onucus/s4 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h02_c1_onucus/s0 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h02_c1_onucus/s1 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h02_c1_onucus/s2 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h02_c1_onucus/s3 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h02_c1_onucus/s4 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h08c_onucus/s0 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h08c_onucus/s1 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h08c_onucus/s2 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h08c_onucus/s3 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h08c_onucus/s4 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h08_c1_onucus/s0 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h08_c1_onucus/s1 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h08_c1_onucus/s2 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h08_c1_onucus/s3 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h08_c1_onucus/s4 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h14_c1_onucus/s0 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h14_c1_onucus/s1 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h14_c1_onucus/s2 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h14_c1_onucus/s3 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h14_c1_onucus/s4 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h11_onucus/s0 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h11_onucus/s1 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h11_onucus/s2 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h11_onucus/s3 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h11_onucus/s4 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h01c_onucus/s0 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h01c_onucus/s1 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h01c_onucus/s2 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h01c_onucus/s3 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h01c_onucus/s4 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h12_onucus/s0 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h12_onucus/s1 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h12_onucus/s2 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h12_onucus/s3 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h12_onucus/s4 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h01_c1_onucus/s0 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h01_c1_onucus/s1 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h01_c1_onucus/s2 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h01_c1_onucus/s3 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h01_c1_onucus/s4 ozdeslik_gecti=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h10_onucus/s0 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h10_onucus/s1 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h10_onucus/s2 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h10_onucus/s3 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h10_onucus/s4 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h02c_onucus/s0 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h02c_onucus/s1 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h02c_onucus/s2 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h02c_onucus/s3 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h02c_onucus/s4 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h10_c1_onucus/s0 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h10_c1_onucus/s1 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h10_c1_onucus/s2 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h10_c1_onucus/s3 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h10_c1_onucus/s4 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h02_c1_onucus/s0 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h02_c1_onucus/s1 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h02_c1_onucus/s2 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h02_c1_onucus/s3 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h02_c1_onucus/s4 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h08c_onucus/s0 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h08c_onucus/s1 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h08c_onucus/s2 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h08c_onucus/s3 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h08c_onucus/s4 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h08_c1_onucus/s0 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h08_c1_onucus/s1 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h08_c1_onucus/s2 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h08_c1_onucus/s3 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h08_c1_onucus/s4 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h14_c1_onucus/s0 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h14_c1_onucus/s1 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h14_c1_onucus/s2 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h14_c1_onucus/s3 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h14_c1_onucus/s4 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h11_onucus/s0 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h11_onucus/s1 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h11_onucus/s2 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h11_onucus/s3 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h11_onucus/s4 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h01c_onucus/s0 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h01c_onucus/s1 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h01c_onucus/s2 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h01c_onucus/s3 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h01c_onucus/s4 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h12_onucus/s0 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h12_onucus/s1 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h12_onucus/s2 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h12_onucus/s3 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h12_onucus/s4 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h01_c1_onucus/s0 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h01_c1_onucus/s1 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h01_c1_onucus/s2 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h01_c1_onucus/s3 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h01_c1_onucus/s4 ozdeslik_argmax_oran=1 ∈ [0.995,1.0] |
| 🟢 OK | ikincil_aralik | e2_h10_onucus/s0 ozdeslik_p_maks_fark=0 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h10_onucus/s1 ozdeslik_p_maks_fark=0 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h10_onucus/s2 ozdeslik_p_maks_fark=0 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h10_onucus/s3 ozdeslik_p_maks_fark=0 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h10_onucus/s4 ozdeslik_p_maks_fark=0 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h02c_onucus/s0 ozdeslik_p_maks_fark=5.215e-06 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h02c_onucus/s1 ozdeslik_p_maks_fark=6.616e-06 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h02c_onucus/s2 ozdeslik_p_maks_fark=9.596e-06 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h02c_onucus/s3 ozdeslik_p_maks_fark=5.186e-06 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h02c_onucus/s4 ozdeslik_p_maks_fark=1.24e-05 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h10_c1_onucus/s0 ozdeslik_p_maks_fark=0 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h10_c1_onucus/s1 ozdeslik_p_maks_fark=0 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h10_c1_onucus/s2 ozdeslik_p_maks_fark=0 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h10_c1_onucus/s3 ozdeslik_p_maks_fark=0 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h10_c1_onucus/s4 ozdeslik_p_maks_fark=0 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h02_c1_onucus/s0 ozdeslik_p_maks_fark=6.527e-06 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h02_c1_onucus/s1 ozdeslik_p_maks_fark=2.772e-06 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h02_c1_onucus/s2 ozdeslik_p_maks_fark=3.129e-06 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h02_c1_onucus/s3 ozdeslik_p_maks_fark=4.828e-06 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h02_c1_onucus/s4 ozdeslik_p_maks_fark=5.841e-06 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h08c_onucus/s0 ozdeslik_p_maks_fark=5.007e-06 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h08c_onucus/s1 ozdeslik_p_maks_fark=5.603e-06 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h08c_onucus/s2 ozdeslik_p_maks_fark=5.245e-06 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h08c_onucus/s3 ozdeslik_p_maks_fark=5.186e-06 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h08c_onucus/s4 ozdeslik_p_maks_fark=4.888e-06 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h08_c1_onucus/s0 ozdeslik_p_maks_fark=6.139e-06 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h08_c1_onucus/s1 ozdeslik_p_maks_fark=8.166e-06 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h08_c1_onucus/s2 ozdeslik_p_maks_fark=5.305e-06 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h08_c1_onucus/s3 ozdeslik_p_maks_fark=4.798e-06 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h08_c1_onucus/s4 ozdeslik_p_maks_fark=4.292e-06 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h14_c1_onucus/s0 ozdeslik_p_maks_fark=0 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h14_c1_onucus/s1 ozdeslik_p_maks_fark=0 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h14_c1_onucus/s2 ozdeslik_p_maks_fark=0 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h14_c1_onucus/s3 ozdeslik_p_maks_fark=0 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h14_c1_onucus/s4 ozdeslik_p_maks_fark=0 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h11_onucus/s0 ozdeslik_p_maks_fark=0 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h11_onucus/s1 ozdeslik_p_maks_fark=0 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h11_onucus/s2 ozdeslik_p_maks_fark=0 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h11_onucus/s3 ozdeslik_p_maks_fark=0 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h11_onucus/s4 ozdeslik_p_maks_fark=0 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h01c_onucus/s0 ozdeslik_p_maks_fark=0 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h01c_onucus/s1 ozdeslik_p_maks_fark=0 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h01c_onucus/s2 ozdeslik_p_maks_fark=0 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h01c_onucus/s3 ozdeslik_p_maks_fark=0 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h01c_onucus/s4 ozdeslik_p_maks_fark=0 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h12_onucus/s0 ozdeslik_p_maks_fark=0 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h12_onucus/s1 ozdeslik_p_maks_fark=0 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h12_onucus/s2 ozdeslik_p_maks_fark=0 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h12_onucus/s3 ozdeslik_p_maks_fark=0 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h12_onucus/s4 ozdeslik_p_maks_fark=0 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h01_c1_onucus/s0 ozdeslik_p_maks_fark=0 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h01_c1_onucus/s1 ozdeslik_p_maks_fark=0 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h01_c1_onucus/s2 ozdeslik_p_maks_fark=0 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h01_c1_onucus/s3 ozdeslik_p_maks_fark=0 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h01_c1_onucus/s4 ozdeslik_p_maks_fark=0 ∈ [0.0,0.001] |
| 🟢 OK | ikincil_aralik | e2_h10_onucus/s0 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h10_onucus/s1 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h10_onucus/s2 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h10_onucus/s3 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h10_onucus/s4 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h02c_onucus/s0 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h02c_onucus/s1 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h02c_onucus/s2 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h02c_onucus/s3 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h02c_onucus/s4 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h10_c1_onucus/s0 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h10_c1_onucus/s1 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h10_c1_onucus/s2 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h10_c1_onucus/s3 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h10_c1_onucus/s4 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h02_c1_onucus/s0 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h02_c1_onucus/s1 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h02_c1_onucus/s2 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h02_c1_onucus/s3 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h02_c1_onucus/s4 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h08c_onucus/s0 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h08c_onucus/s1 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h08c_onucus/s2 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h08c_onucus/s3 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h08c_onucus/s4 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h08_c1_onucus/s0 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h08_c1_onucus/s1 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h08_c1_onucus/s2 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h08_c1_onucus/s3 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h08_c1_onucus/s4 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h14_c1_onucus/s0 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h14_c1_onucus/s1 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h14_c1_onucus/s2 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h14_c1_onucus/s3 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h14_c1_onucus/s4 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h11_onucus/s0 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h11_onucus/s1 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h11_onucus/s2 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h11_onucus/s3 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h11_onucus/s4 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h01c_onucus/s0 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h01c_onucus/s1 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h01c_onucus/s2 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h01c_onucus/s3 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h01c_onucus/s4 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h12_onucus/s0 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h12_onucus/s1 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h12_onucus/s2 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h12_onucus/s3 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h12_onucus/s4 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h01_c1_onucus/s0 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h01_c1_onucus/s1 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h01_c1_onucus/s2 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h01_c1_onucus/s3 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h01_c1_onucus/s4 val_kumesi_dogru=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | e2_h10_onucus/s0 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h10_onucus/s1 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h10_onucus/s2 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h10_onucus/s3 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h10_onucus/s4 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h02c_onucus/s0 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h02c_onucus/s1 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h02c_onucus/s2 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h02c_onucus/s3 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h02c_onucus/s4 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h10_c1_onucus/s0 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h10_c1_onucus/s1 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h10_c1_onucus/s2 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h10_c1_onucus/s3 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h10_c1_onucus/s4 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h02_c1_onucus/s0 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h02_c1_onucus/s1 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h02_c1_onucus/s2 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h02_c1_onucus/s3 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h02_c1_onucus/s4 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h08c_onucus/s0 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h08c_onucus/s1 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h08c_onucus/s2 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h08c_onucus/s3 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h08c_onucus/s4 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h08_c1_onucus/s0 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h08_c1_onucus/s1 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h08_c1_onucus/s2 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h08_c1_onucus/s3 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h08_c1_onucus/s4 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h14_c1_onucus/s0 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h14_c1_onucus/s1 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h14_c1_onucus/s2 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h14_c1_onucus/s3 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h14_c1_onucus/s4 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h11_onucus/s0 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h11_onucus/s1 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h11_onucus/s2 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h11_onucus/s3 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h11_onucus/s4 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h01c_onucus/s0 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h01c_onucus/s1 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h01c_onucus/s2 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h01c_onucus/s3 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h01c_onucus/s4 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h12_onucus/s0 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h12_onucus/s1 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h12_onucus/s2 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h12_onucus/s3 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h12_onucus/s4 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h01_c1_onucus/s0 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h01_c1_onucus/s1 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h01_c1_onucus/s2 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h01_c1_onucus/s3 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h01_c1_onucus/s4 bcn_etiket_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h10_onucus/s0 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h10_onucus/s1 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h10_onucus/s2 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h10_onucus/s3 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h10_onucus/s4 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h02c_onucus/s0 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h02c_onucus/s1 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h02c_onucus/s2 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h02c_onucus/s3 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h02c_onucus/s4 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h10_c1_onucus/s0 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h10_c1_onucus/s1 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h10_c1_onucus/s2 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h10_c1_onucus/s3 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h10_c1_onucus/s4 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h02_c1_onucus/s0 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h02_c1_onucus/s1 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h02_c1_onucus/s2 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h02_c1_onucus/s3 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h02_c1_onucus/s4 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h08c_onucus/s0 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h08c_onucus/s1 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h08c_onucus/s2 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h08c_onucus/s3 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h08c_onucus/s4 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h08_c1_onucus/s0 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h08_c1_onucus/s1 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h08_c1_onucus/s2 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h08_c1_onucus/s3 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h08_c1_onucus/s4 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h14_c1_onucus/s0 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h14_c1_onucus/s1 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h14_c1_onucus/s2 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h14_c1_onucus/s3 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h14_c1_onucus/s4 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h11_onucus/s0 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h11_onucus/s1 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h11_onucus/s2 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h11_onucus/s3 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h11_onucus/s4 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h01c_onucus/s0 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h01c_onucus/s1 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h01c_onucus/s2 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h01c_onucus/s3 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h01c_onucus/s4 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h12_onucus/s0 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h12_onucus/s1 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h12_onucus/s2 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h12_onucus/s3 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h12_onucus/s4 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h01_c1_onucus/s0 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h01_c1_onucus/s1 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h01_c1_onucus/s2 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h01_c1_onucus/s3 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h01_c1_onucus/s4 resmi_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h10_onucus/s0 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h10_onucus/s1 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h10_onucus/s2 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h10_onucus/s3 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h10_onucus/s4 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h02c_onucus/s0 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h02c_onucus/s1 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h02c_onucus/s2 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h02c_onucus/s3 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h02c_onucus/s4 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h10_c1_onucus/s0 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h10_c1_onucus/s1 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h10_c1_onucus/s2 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h10_c1_onucus/s3 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h10_c1_onucus/s4 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h02_c1_onucus/s0 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h02_c1_onucus/s1 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h02_c1_onucus/s2 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h02_c1_onucus/s3 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h02_c1_onucus/s4 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h08c_onucus/s0 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h08c_onucus/s1 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h08c_onucus/s2 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h08c_onucus/s3 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h08c_onucus/s4 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h08_c1_onucus/s0 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h08_c1_onucus/s1 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h08_c1_onucus/s2 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h08_c1_onucus/s3 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h08_c1_onucus/s4 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h14_c1_onucus/s0 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h14_c1_onucus/s1 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h14_c1_onucus/s2 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h14_c1_onucus/s3 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h14_c1_onucus/s4 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h11_onucus/s0 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h11_onucus/s1 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h11_onucus/s2 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h11_onucus/s3 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h11_onucus/s4 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h01c_onucus/s0 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h01c_onucus/s1 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h01c_onucus/s2 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h01c_onucus/s3 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h01c_onucus/s4 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h12_onucus/s0 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h12_onucus/s1 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h12_onucus/s2 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h12_onucus/s3 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h12_onucus/s4 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h01_c1_onucus/s0 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h01_c1_onucus/s1 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h01_c1_onucus/s2 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h01_c1_onucus/s3 c_test_okundu=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | e2_h01_c1_onucus/s4 c_test_okundu=0 ∈ [0,0] |
