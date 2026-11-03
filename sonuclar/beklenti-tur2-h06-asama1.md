# Beklenti Kapısı — kuyruk `tur2-h06` aşama 1 → **GEÇTİ**
*2026-10-11 01:17* · metrik `val_acc` (max) · beklenti: `tur2-h06.beklenti.json` · 1 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h06_h01_logitayar_duman | 1 | 0.8245 | 0.8245 | 4.4 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 1 başarılı koşum (en az 1) |
| 🟢 OK | aralik | h06_h01_logitayar_duman: ort 0.8245 ∈ [0.8123, 0.8423] (H06 kartı: alt = kısıt (H01 0,8223 − 0,010), üst = görülmemiş ACC Δ > +2,0) |
| 🟢 OK | ikincil_aralik | h06_h01_logitayar_duman/s0 tau0_ozdeslik_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h06_h01_logitayar_duman/s0 pi_sira_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h06_h01_logitayar_duman/s0 kat_sapma=0 ∈ [0,0] |
