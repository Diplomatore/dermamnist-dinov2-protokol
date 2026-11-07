# Beklenti Kapısı — kuyruk `e1-h14` aşama 2 → **GEÇTİ**
*2026-11-07 16:35* · metrik `val_acc` (max) · beklenti: `e1-h14.beklenti.json` · 2 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h14_c1_onucus | 2 | 0.9241 | 0.925, 0.9232 | 299.6 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 2 başarılı koşum (en az 2) |
| 🟢 OK | farkli_seedler_farkli | h14_c1_onucus: seed'ler farklı |
| 🟢 OK | aralik | h14_c1_onucus: ort 0.9241 ∈ [0.74, 0.96] (k5-c1-B a2 ile aynı) |
| 🟢 OK | ikincil_aralik | h14_c1_onucus/s0 val_auc=0.9836 ∈ [0.8,0.995] |
| 🟢 OK | ikincil_aralik | h14_c1_onucus/s1 val_auc=0.9863 ∈ [0.8,0.995] |
| 🟢 OK | ikincil_aralik | h14_c1_onucus/s0 epoch_sure_ort_sn=56.37 ∈ [30,110] |
| 🟢 OK | ikincil_aralik | h14_c1_onucus/s1 epoch_sure_ort_sn=56.3 ∈ [30,110] |
| 🟢 OK | ikincil_aralik | h14_c1_onucus/s0 augment_frekans_maks_sapma=0.002572 ∈ [0.0,0.01] |
| 🟢 OK | ikincil_aralik | h14_c1_onucus/s1 augment_frekans_maks_sapma=0.00332 ∈ [0.0,0.01] |
| 🟢 OK | ikincil_aralik | h14_c1_onucus/s0 augment_toplam_ornek=4.108e+04 ∈ [41075,41075] |
| 🟢 OK | ikincil_aralik | h14_c1_onucus/s1 augment_toplam_ornek=4.108e+04 ∈ [41075,41075] |
| 🟢 OK | ikincil_aralik | h14_c1_onucus/s0 onbellek_gorunum_tutarlilik_maks_fark=0.002195 ∈ [0.0,0.005] |
| 🟢 OK | ikincil_aralik | h14_c1_onucus/s1 onbellek_gorunum_tutarlilik_maks_fark=0.002195 ∈ [0.0,0.005] |
| 🟢 OK | ikincil_aralik | h14_c1_onucus/s0 param_egitilen=2.836e+07 ∈ [28364551,28364551] |
| 🟢 OK | ikincil_aralik | h14_c1_onucus/s1 param_egitilen=2.836e+07 ∈ [28364551,28364551] |
| 🟢 OK | ikincil_aralik | h14_c1_onucus/s0 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h14_c1_onucus/s1 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h14_c1_onucus/s0 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h14_c1_onucus/s1 val_n=573 ∈ [573,573] |
