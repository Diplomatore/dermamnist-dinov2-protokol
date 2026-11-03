# Beklenti Kapısı — kuyruk `tur2-h04` aşama 1 → **GEÇTİ**
*2026-10-11 01:14* · metrik `val_acc` (max) · beklenti: `tur2-h04.beklenti.json` · 1 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h04_h02_logitayar_duman | 1 | 0.8684 | 0.8684 | 4.4 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 1 başarılı koşum (en az 1) |
| 🟢 OK | aralik | h04_h02_logitayar_duman: ort 0.8684 ∈ [0.8572, 0.8872] (H04 kartı: alt = kısıt (H02 0,8672 − 0,010), üst = görülmemiş OOF ACC Δ > +2,0 (literatur-cipa §4c)) |
| 🟢 OK | ikincil_aralik | h04_h02_logitayar_duman/s0 tau0_ozdeslik_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h04_h02_logitayar_duman/s0 pi_sira_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h04_h02_logitayar_duman/s0 kat_sapma=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | h04_h02_logitayar_duman/s0 val_n=1003 ∈ [1003,1003] |
