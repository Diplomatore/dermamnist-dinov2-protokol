# Beklenti Kapısı — kuyruk `tur3-h08k` aşama 3 → **GEÇTİ**
*2026-10-11 09:14* · metrik `val_acc` (max) · beklenti: `tur3-h08k.beklenti.json` · 5 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h08k_h08_logitayar | 5 | 0.88 | 0.8774, 0.8744, 0.8833, 0.8814, 0.8833 | 9.8 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 5 başarılı koşum (en az 5) |
| 🟢 OK | farkli_seedler_farkli | h08k_h08_logitayar: seed'ler farklı |
| 🟢 OK | aralik | h08k_h08_logitayar: ort 0.88 ∈ [0.8, 0.9] (H08 beklenti kapısı bandı (0,80 bozuk alarmı, 0,90 H02 tavanı); OOF ACC kısıtı kaynak − 0,010) |
| 🟢 OK | ikincil_aralik | h08k_h08_logitayar/s0 tau0_ozdeslik_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h08k_h08_logitayar/s1 tau0_ozdeslik_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h08k_h08_logitayar/s2 tau0_ozdeslik_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h08k_h08_logitayar/s3 tau0_ozdeslik_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h08k_h08_logitayar/s4 tau0_ozdeslik_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h08k_h08_logitayar/s0 pi_sira_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h08k_h08_logitayar/s1 pi_sira_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h08k_h08_logitayar/s2 pi_sira_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h08k_h08_logitayar/s3 pi_sira_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h08k_h08_logitayar/s4 pi_sira_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h08k_h08_logitayar/s0 kat_sapma=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | h08k_h08_logitayar/s1 kat_sapma=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | h08k_h08_logitayar/s2 kat_sapma=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | h08k_h08_logitayar/s3 kat_sapma=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | h08k_h08_logitayar/s4 kat_sapma=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | h08k_h08_logitayar/s0 val_dengeli_acc=0.7854 ∈ [0.6,0.85] |
| 🟢 OK | ikincil_aralik | h08k_h08_logitayar/s1 val_dengeli_acc=0.7815 ∈ [0.6,0.85] |
| 🟢 OK | ikincil_aralik | h08k_h08_logitayar/s2 val_dengeli_acc=0.8244 ∈ [0.6,0.85] |
| 🟢 OK | ikincil_aralik | h08k_h08_logitayar/s3 val_dengeli_acc=0.7788 ∈ [0.6,0.85] |
| 🟢 OK | ikincil_aralik | h08k_h08_logitayar/s4 val_dengeli_acc=0.8182 ∈ [0.6,0.85] |
| 🟢 OK | ikincil_aralik | h08k_h08_logitayar/s0 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h08k_h08_logitayar/s1 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h08k_h08_logitayar/s2 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h08k_h08_logitayar/s3 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h08k_h08_logitayar/s4 val_n=1003 ∈ [1003,1003] |
