# Beklenti Kapısı — kuyruk `kc2-tur4-h10` aşama 2 → **GEÇTİ**
*2026-10-11 16:02* · metrik `val_acc` (max) · beklenti: `kc2-tur4-h10.beklenti.json` · 2 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h10_dinov2b_k4_ft224_onucus | 2 | 0.8789 | 0.8764, 0.8814 | 250.2 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 2 başarılı koşum (en az 2) |
| 🟢 OK | farkli_seedler_farkli | h10_dinov2b_k4_ft224_onucus: seed'ler farklı |
| 🟢 OK | aralik | h10_dinov2b_k4_ft224_onucus: ort 0.8789 ∈ [0.72, 0.93] (H10.md uygulama notu a2) |
| 🟢 OK | ikincil_aralik | h10_dinov2b_k4_ft224_onucus/s0 epoch_sure_ort_sn=46.48 ∈ [13,260] |
| 🟢 OK | ikincil_aralik | h10_dinov2b_k4_ft224_onucus/s1 epoch_sure_ort_sn=46.71 ∈ [13,260] |
| 🟢 OK | ikincil_aralik | h10_dinov2b_k4_ft224_onucus/s0 val_auc=0.978 ∈ [0.88,0.995] |
| 🟢 OK | ikincil_aralik | h10_dinov2b_k4_ft224_onucus/s1 val_auc=0.9782 ∈ [0.88,0.995] |
| 🟢 OK | ikincil_aralik | h10_dinov2b_k4_ft224_onucus/s0 param_egitilen=2.836e+07 ∈ [28364551,28364551] |
| 🟢 OK | ikincil_aralik | h10_dinov2b_k4_ft224_onucus/s1 param_egitilen=2.836e+07 ∈ [28364551,28364551] |
| 🟢 OK | ikincil_aralik | h10_dinov2b_k4_ft224_onucus/s0 onbellek_tutarlilik_maks_fark=0.002426 ∈ [0.0,0.005] |
| 🟢 OK | ikincil_aralik | h10_dinov2b_k4_ft224_onucus/s1 onbellek_tutarlilik_maks_fark=0.002426 ∈ [0.0,0.005] |
| 🟢 OK | ikincil_aralik | h10_dinov2b_k4_ft224_onucus/s0 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h10_dinov2b_k4_ft224_onucus/s1 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h10_dinov2b_k4_ft224_onucus/s0 tepe_gpu_bellek_mb=2432 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h10_dinov2b_k4_ft224_onucus/s1 tepe_gpu_bellek_mb=2432 ∈ [1,22500] |
