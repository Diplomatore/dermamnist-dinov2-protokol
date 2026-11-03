# Beklenti Kapısı — kuyruk `tur3-h09k` aşama 3 → **GEÇTİ**
*2026-10-11 09:14* · metrik `val_acc` (max) · beklenti: `tur3-h09k.beklenti.json` · 5 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h09k_h09_logitayar | 5 | 0.8604 | 0.8654, 0.8544, 0.8594, 0.8584, 0.8644 | 9.4 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 5 başarılı koşum (en az 5) |
| 🟢 OK | farkli_seedler_farkli | h09k_h09_logitayar: seed'ler farklı |
| 🟢 OK | aralik | h09k_h09_logitayar: ort 0.8604 ∈ [0.8, 0.9] (H09 beklenti kapısı bandı (0,80 bozuk alarmı, 0,90 H02 tavanı); OOF ACC kısıtı kaynak − 0,010) |
| 🟢 OK | ikincil_aralik | h09k_h09_logitayar/s0 tau0_ozdeslik_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h09k_h09_logitayar/s1 tau0_ozdeslik_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h09k_h09_logitayar/s2 tau0_ozdeslik_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h09k_h09_logitayar/s3 tau0_ozdeslik_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h09k_h09_logitayar/s4 tau0_ozdeslik_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h09k_h09_logitayar/s0 pi_sira_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h09k_h09_logitayar/s1 pi_sira_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h09k_h09_logitayar/s2 pi_sira_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h09k_h09_logitayar/s3 pi_sira_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h09k_h09_logitayar/s4 pi_sira_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h09k_h09_logitayar/s0 kat_sapma=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | h09k_h09_logitayar/s1 kat_sapma=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | h09k_h09_logitayar/s2 kat_sapma=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | h09k_h09_logitayar/s3 kat_sapma=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | h09k_h09_logitayar/s4 kat_sapma=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | h09k_h09_logitayar/s0 val_dengeli_acc=0.719 ∈ [0.6,0.85] |
| 🟢 OK | ikincil_aralik | h09k_h09_logitayar/s1 val_dengeli_acc=0.7384 ∈ [0.6,0.85] |
| 🟢 OK | ikincil_aralik | h09k_h09_logitayar/s2 val_dengeli_acc=0.7322 ∈ [0.6,0.85] |
| 🟢 OK | ikincil_aralik | h09k_h09_logitayar/s3 val_dengeli_acc=0.7648 ∈ [0.6,0.85] |
| 🟢 OK | ikincil_aralik | h09k_h09_logitayar/s4 val_dengeli_acc=0.7465 ∈ [0.6,0.85] |
| 🟢 OK | ikincil_aralik | h09k_h09_logitayar/s0 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h09k_h09_logitayar/s1 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h09k_h09_logitayar/s2 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h09k_h09_logitayar/s3 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h09k_h09_logitayar/s4 val_n=1003 ∈ [1003,1003] |
