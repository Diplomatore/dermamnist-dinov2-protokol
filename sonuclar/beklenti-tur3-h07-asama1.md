# Beklenti Kapısı — kuyruk `tur3-h07` aşama 1 → **GEÇTİ**
*2026-10-11 04:22* · metrik `val_acc` (max) · beklenti: `tur3-h07.beklenti.json` · 1 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h07_h02_tta8_duman | 1 | 0.8614 | 0.8614 | 24.8 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | ikincil_aralik | h07_h02_tta8_duman/s0 ozdeslik_kimlik_maks_fark=0 ∈ [0.0,0.0001] |
| 🟢 OK | ikincil_aralik | h07_h02_tta8_duman/s0 ozdeslik_kimlik_acc_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h07_h02_tta8_duman/s0 gorunum_arasi_maks_fark=0.998 ∈ [0.001,1.0] |
| 🟢 OK | ikincil_aralik | h07_h02_tta8_duman/s0 param_sayisi=1.118e+07 ∈ [11180103,11180103] |
| 🟢 OK | ikincil_aralik | h07_h02_tta8_duman/s0 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h07_h02_tta8_duman/s0 gorunum_sayisi=8 ∈ [8,8] |
| 🟢 OK | ikincil_aralik | h07_h02_tta8_duman/s0 val_ece=0.04082 ∈ [0.01,1.0] |
| 🟢 OK | ikincil_aralik | h07_h02_tta8_duman/s0 egitim_sure_sn=12.88 ∈ [1,300] |
| 🟢 OK | min_kosum | 1 başarılı koşum (en az 1) |
| 🟢 OK | aralik | h07_h02_tta8_duman: ort 0.8614 ∈ [0.8, 0.9] (H07 uygulama notu a1: üst 0,90 = H02 tavanı, alt 0,80 (Δ ≈ −6,7) bozuk görünüm alarmı) |
