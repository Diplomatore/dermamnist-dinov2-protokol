# Beklenti Kapısı — kuyruk `kc2-tur4-h11` aşama 2 → **GEÇTİ**
*2026-10-11 16:01* · metrik `val_acc` (max) · beklenti: `kc2-tur4-h11.beklenti.json` · 2 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h11_dinov2s_ft224_onucus | 2 | 0.8639 | 0.8604, 0.8674 | 243.6 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 2 başarılı koşum (en az 2) |
| 🟢 OK | farkli_seedler_farkli | h11_dinov2s_ft224_onucus: seed'ler farklı |
| 🟢 OK | aralik | h11_dinov2s_ft224_onucus: ort 0.8639 ∈ [0.72, 0.93] (H11.md uygulama notu a2) |
| 🟢 OK | ikincil_aralik | h11_dinov2s_ft224_onucus/s0 epoch_sure_ort_sn=44.49 ∈ [10.8,216] |
| 🟢 OK | ikincil_aralik | h11_dinov2s_ft224_onucus/s1 epoch_sure_ort_sn=44.45 ∈ [10.8,216] |
| 🟢 OK | ikincil_aralik | h11_dinov2s_ft224_onucus/s0 val_auc=0.975 ∈ [0.88,0.995] |
| 🟢 OK | ikincil_aralik | h11_dinov2s_ft224_onucus/s1 val_auc=0.9795 ∈ [0.88,0.995] |
| 🟢 OK | ikincil_aralik | h11_dinov2s_ft224_onucus/s0 param_egitilen=2.163e+07 ∈ [21631495,21631495] |
| 🟢 OK | ikincil_aralik | h11_dinov2s_ft224_onucus/s1 param_egitilen=2.163e+07 ∈ [21631495,21631495] |
| 🟢 OK | ikincil_aralik | h11_dinov2s_ft224_onucus/s0 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h11_dinov2s_ft224_onucus/s1 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h11_dinov2s_ft224_onucus/s0 tepe_gpu_bellek_mb=1761 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h11_dinov2s_ft224_onucus/s1 tepe_gpu_bellek_mb=1761 ∈ [1,22500] |
