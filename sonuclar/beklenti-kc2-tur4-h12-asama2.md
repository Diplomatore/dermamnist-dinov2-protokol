# Beklenti Kapısı — kuyruk `kc2-tur4-h12` aşama 2 → **GEÇTİ**
*2026-10-11 17:45* · metrik `val_acc` (max) · beklenti: `kc2-tur4-h12.beklenti.json` · 2 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h12_dinov2s_lp_r224_onucus | 2 | 0.819 | 0.8195, 0.8185 | 135.4 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 2 başarılı koşum (en az 2) |
| 🟢 OK | aralik | h12_dinov2s_lp_r224_onucus: ort 0.819 ∈ [0.74, 0.88] (H12.md uygulama notu a2/a3: alt ≈ kart alt ucu 0,787 − 0,047 (arıza alarmı; SIRALAMA 'Beklenti kapıları' H12 [0,74; 0,88]), üst = LP 224 tavanı 0,88 (literatur-cipa §4(b))) |
| 🟢 OK | ikincil_aralik | h12_dinov2s_lp_r224_onucus/s0 val_auc=0.9538 ∈ [0.9,0.99] |
| 🟢 OK | ikincil_aralik | h12_dinov2s_lp_r224_onucus/s1 val_auc=0.9537 ∈ [0.9,0.99] |
| 🟢 OK | ikincil_aralik | h12_dinov2s_lp_r224_onucus/s0 lbfgs_sgd_fark=0.0004985 ∈ [-0.005,0.005] |
| 🟢 OK | ikincil_aralik | h12_dinov2s_lp_r224_onucus/s1 lbfgs_sgd_fark=0.0004985 ∈ [-0.005,0.005] |
| 🟢 OK | ikincil_aralik | h12_dinov2s_lp_r224_onucus/s0 oznitelik_img_s=497.5 ∈ [10,2000] |
| 🟢 OK | ikincil_aralik | h12_dinov2s_lp_r224_onucus/s1 oznitelik_img_s=497.5 ∈ [10,2000] |
| 🟢 OK | ikincil_aralik | h12_dinov2s_lp_r224_onucus/s0 param_sayisi=2695 ∈ [2695,2695] |
| 🟢 OK | ikincil_aralik | h12_dinov2s_lp_r224_onucus/s1 param_sayisi=2695 ∈ [2695,2695] |
| 🟢 OK | ikincil_aralik | h12_dinov2s_lp_r224_onucus/s0 oznitelik_boyut=384 ∈ [384,384] |
| 🟢 OK | ikincil_aralik | h12_dinov2s_lp_r224_onucus/s1 oznitelik_boyut=384 ∈ [384,384] |
| 🟢 OK | ikincil_aralik | h12_dinov2s_lp_r224_onucus/s0 oznitelik_determinizm_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h12_dinov2s_lp_r224_onucus/s1 oznitelik_determinizm_maks_fark=0 ∈ [0.0,1e-05] |
