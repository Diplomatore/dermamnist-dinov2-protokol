# Beklenti Kapısı — kuyruk `tur2-h05` aşama 2 → **GEÇTİ**
*2026-10-11 01:38* · metrik `val_acc` (max) · beklenti: `tur2-h05.beklenti.json` · 2 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h05_r18in_ft224_cw05_onucus | 2 | 0.8594 | 0.8604, 0.8584 | 257.1 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 2 başarılı koşum (en az 2) |
| 🟢 OK | farkli_seedler_farkli | h05_r18in_ft224_cw05_onucus: seed'ler farklı |
| 🟢 OK | aralik | h05_r18in_ft224_cw05_onucus: ort 0.8594 ∈ [0.72, 0.9] (H05 kartı a2 (H02 a2 ile aynı: alt ≈ rakip testi − 1,5; üst = tavan 0,90)) |
| 🟢 OK | ikincil_aralik | h05_r18in_ft224_cw05_onucus/s0 epoch_sure_ort_sn=46.96 ∈ [10,90] |
| 🟢 OK | ikincil_aralik | h05_r18in_ft224_cw05_onucus/s1 epoch_sure_ort_sn=48.12 ∈ [10,90] |
| 🟢 OK | ikincil_aralik | h05_r18in_ft224_cw05_onucus/s0 val_auc=0.9703 ∈ [0.88,0.99] |
| 🟢 OK | ikincil_aralik | h05_r18in_ft224_cw05_onucus/s1 val_auc=0.9701 ∈ [0.88,0.99] |
| 🟢 OK | ikincil_aralik | h05_r18in_ft224_cw05_onucus/s0 val_tahmin_nv_payi=0.6859 ∈ [0.55,0.7059] |
| 🟢 OK | ikincil_aralik | h05_r18in_ft224_cw05_onucus/s1 val_tahmin_nv_payi=0.68 ∈ [0.55,0.7059] |
| 🟢 OK | ikincil_aralik | h05_r18in_ft224_cw05_onucus/s0 val_dengeli_acc=0.724 ∈ [0.6,0.7712] |
| 🟢 OK | ikincil_aralik | h05_r18in_ft224_cw05_onucus/s1 val_dengeli_acc=0.7315 ∈ [0.6,0.7712] |
| 🟢 OK | ikincil_aralik | h05_r18in_ft224_cw05_onucus/s0 ilk_adim_loss=2.04 ∈ [1.5,2.5] |
| 🟢 OK | ikincil_aralik | h05_r18in_ft224_cw05_onucus/s1 ilk_adim_loss=2.117 ∈ [1.5,2.5] |
| 🟢 OK | ikincil_aralik | h05_r18in_ft224_cw05_onucus/s0 sinif_agirlik_df=4.424 ∈ [4.37,4.47] |
| 🟢 OK | ikincil_aralik | h05_r18in_ft224_cw05_onucus/s1 sinif_agirlik_df=4.424 ∈ [4.37,4.47] |
| 🟢 OK | ikincil_aralik | h05_r18in_ft224_cw05_onucus/s0 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h05_r18in_ft224_cw05_onucus/s1 n_train=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h05_r18in_ft224_cw05_onucus/s0 tepe_gpu_bellek_mb=1742 ∈ [1,3900] |
| 🟢 OK | ikincil_aralik | h05_r18in_ft224_cw05_onucus/s1 tepe_gpu_bellek_mb=1742 ∈ [1,3900] |
