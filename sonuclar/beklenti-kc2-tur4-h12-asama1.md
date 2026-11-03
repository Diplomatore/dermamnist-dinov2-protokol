# Beklenti Kapısı — kuyruk `kc2-tur4-h12` aşama 1 → **GEÇTİ**
*2026-10-11 17:40* · metrik `val_acc` (max) · beklenti: `kc2-tur4-h12.beklenti.json` · 1 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h12_dinov2s_lp_r224_duman | 1 | 0.6094 | 0.6094 | 28.8 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 1 başarılı koşum (en az 1) |
| 🟢 OK | aralik | h12_dinov2s_lp_r224_duman: ort 0.6094 ∈ [0.55, 0.95] (H12.md uygulama notu a1 (H01 a1 ile aynı gerekçe: alt küme, tek C)) |
| 🟢 OK | ikincil_aralik | h12_dinov2s_lp_r224_duman/s0 param_sayisi=2695 ∈ [2695,2695] |
| 🟢 OK | ikincil_aralik | h12_dinov2s_lp_r224_duman/s0 oznitelik_boyut=384 ∈ [384,384] |
| 🟢 OK | ikincil_aralik | h12_dinov2s_lp_r224_duman/s0 oznitelik_determinizm_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h12_dinov2s_lp_r224_duman/s0 egitim_sure_sn=25.8 ∈ [1,600] |
