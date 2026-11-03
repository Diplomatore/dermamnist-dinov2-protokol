# Beklenti Kapısı — kuyruk `tur5-h13k` aşama 3 → **GEÇTİ**
*2026-10-11 18:06* · metrik `val_acc` (max) · beklenti: `tur5-h13k.beklenti.json` · 5 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h13k_h13_logitayar | 5 | 0.8953 | 0.9023, 0.8943, 0.8993, 0.8903, 0.8903 | 9.7 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 5 başarılı koşum (en az 5) |
| 🟢 OK | farkli_seedler_farkli | h13k_h13_logitayar: seed'ler farklı |
| 🟢 OK | aralik | h13k_h13_logitayar: ort 0.8953 ∈ [0.85, 0.93] (H13 a3 bandı; OOF ACC kısıtı kaynak − 0,010 (H10 + H04 OOF ACC 0,8855)) |
| 🟢 OK | ikincil_aralik | h13k_h13_logitayar/s0 tau0_ozdeslik_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h13k_h13_logitayar/s1 tau0_ozdeslik_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h13k_h13_logitayar/s2 tau0_ozdeslik_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h13k_h13_logitayar/s3 tau0_ozdeslik_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h13k_h13_logitayar/s4 tau0_ozdeslik_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h13k_h13_logitayar/s0 pi_sira_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13k_h13_logitayar/s1 pi_sira_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13k_h13_logitayar/s2 pi_sira_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13k_h13_logitayar/s3 pi_sira_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13k_h13_logitayar/s4 pi_sira_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13k_h13_logitayar/s0 kat_sapma=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | h13k_h13_logitayar/s1 kat_sapma=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | h13k_h13_logitayar/s2 kat_sapma=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | h13k_h13_logitayar/s3 kat_sapma=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | h13k_h13_logitayar/s4 kat_sapma=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | h13k_h13_logitayar/s0 val_dengeli_acc=0.8642 ∈ [0.7,0.9] |
| 🟢 OK | ikincil_aralik | h13k_h13_logitayar/s1 val_dengeli_acc=0.8371 ∈ [0.7,0.9] |
| 🟢 OK | ikincil_aralik | h13k_h13_logitayar/s2 val_dengeli_acc=0.8353 ∈ [0.7,0.9] |
| 🟢 OK | ikincil_aralik | h13k_h13_logitayar/s3 val_dengeli_acc=0.8719 ∈ [0.7,0.9] |
| 🟢 OK | ikincil_aralik | h13k_h13_logitayar/s4 val_dengeli_acc=0.84 ∈ [0.7,0.9] |
| 🟢 OK | ikincil_aralik | h13k_h13_logitayar/s0 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h13k_h13_logitayar/s1 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h13k_h13_logitayar/s2 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h13k_h13_logitayar/s3 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h13k_h13_logitayar/s4 val_n=1003 ∈ [1003,1003] |
