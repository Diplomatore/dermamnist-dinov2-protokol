# Beklenti Kapısı — kuyruk `kc2-tur4-h11` aşama 1 → **GEÇTİ**
*2026-10-11 15:53* · metrik `val_acc` (max) · beklenti: `kc2-tur4-h11.beklenti.json` · 1 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h11_dinov2s_ft224_duman | 1 | 0.7797 | 0.7797 | 64.4 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 1 başarılı koşum (en az 1) |
| 🟢 OK | aralik | h11_dinov2s_ft224_duman: ort 0.7797 ∈ [0.62, 0.93] (H11.md uygulama notu a1) |
| 🟢 OK | ikincil_aralik | h11_dinov2s_ft224_duman/s0 param_sayisi=2.163e+07 ∈ [21631495,21631495] |
| 🟢 OK | ikincil_aralik | h11_dinov2s_ft224_duman/s0 param_egitilen=2.163e+07 ∈ [21631495,21631495] |
| 🟢 OK | ikincil_aralik | h11_dinov2s_ft224_duman/s0 ilk_adim_loss=2.577 ∈ [2.5756281485557557,2.5776281485557555] |
| 🟢 OK | ikincil_aralik | h11_dinov2s_ft224_duman/s0 egitim_sure_sn=43.19 ∈ [10,600] |
| 🟢 OK | ikincil_aralik | h11_dinov2s_ft224_duman/s0 tepe_gpu_bellek_mb=1761 ∈ [1,22500] |
