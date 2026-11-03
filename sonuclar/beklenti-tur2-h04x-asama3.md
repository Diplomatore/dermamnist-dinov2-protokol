# Beklenti Kapısı — kuyruk `tur2-h04x` aşama 3 → **GEÇTİ**
*2026-10-11 01:17* · metrik `val_acc` (max) · beklenti: `tur2-h04x.beklenti.json` · 10 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h04x_h02aucsecim_logitayar | 5 | 0.8491 | 0.8445, 0.8465, 0.8544, 0.8564, 0.8435 | 9.4 |
| h04x_baseline_logitayar | 5 | 0.734 | 0.7218, 0.6999, 0.7507, 0.7617, 0.7358 | 9.4 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 10 başarılı koşum (en az 10) |
| 🟢 OK | aralik | h04x_h02aucsecim_logitayar: ort 0.8491 ∈ [0.8488, 0.8788] (H04 ikincil (10): H02 aucsecim ACC 0,8588 (tahlil §1b) − 0,010 / + 0,020) |
| 🟢 OK | aralik | h04x_baseline_logitayar: ort 0.734 ∈ [0.7296, 0.7596] (H04 ikincil (11) / SIRALAMA M3: kısıt B − 0,010 (B 0,7396); üst B + 0,020) |
| 🟢 OK | kollar_farkli | kollar birbirinden farklı |
| 🟢 OK | ikincil_aralik | h04x_h02aucsecim_logitayar/s0 tau0_ozdeslik_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h04x_h02aucsecim_logitayar/s1 tau0_ozdeslik_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h04x_h02aucsecim_logitayar/s2 tau0_ozdeslik_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h04x_h02aucsecim_logitayar/s3 tau0_ozdeslik_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h04x_h02aucsecim_logitayar/s4 tau0_ozdeslik_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h04x_baseline_logitayar/s0 tau0_ozdeslik_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h04x_baseline_logitayar/s1 tau0_ozdeslik_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h04x_baseline_logitayar/s2 tau0_ozdeslik_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h04x_baseline_logitayar/s3 tau0_ozdeslik_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h04x_baseline_logitayar/s4 tau0_ozdeslik_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h04x_h02aucsecim_logitayar/s0 pi_sira_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h04x_h02aucsecim_logitayar/s1 pi_sira_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h04x_h02aucsecim_logitayar/s2 pi_sira_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h04x_h02aucsecim_logitayar/s3 pi_sira_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h04x_h02aucsecim_logitayar/s4 pi_sira_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h04x_baseline_logitayar/s0 pi_sira_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h04x_baseline_logitayar/s1 pi_sira_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h04x_baseline_logitayar/s2 pi_sira_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h04x_baseline_logitayar/s3 pi_sira_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h04x_baseline_logitayar/s4 pi_sira_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h04x_h02aucsecim_logitayar/s0 kat_sapma=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | h04x_h02aucsecim_logitayar/s1 kat_sapma=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | h04x_h02aucsecim_logitayar/s2 kat_sapma=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | h04x_h02aucsecim_logitayar/s3 kat_sapma=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | h04x_h02aucsecim_logitayar/s4 kat_sapma=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | h04x_baseline_logitayar/s0 kat_sapma=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | h04x_baseline_logitayar/s1 kat_sapma=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | h04x_baseline_logitayar/s2 kat_sapma=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | h04x_baseline_logitayar/s3 kat_sapma=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | h04x_baseline_logitayar/s4 kat_sapma=0 ∈ [0,0] |
