# Beklenti Kapısı — kuyruk `tur2-h04` aşama 3 → **İHLAL — BÜYÜK KOŞUMA GEÇİLMEZ**
*2026-10-11 01:15* · metrik `val_acc` (max) · beklenti: `tur2-h04.beklenti.json` · 5 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h04_h02_logitayar | 5 | 0.8576 | 0.8514, 0.8534, 0.8654, 0.8564, 0.8614 | 9.4 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 5 başarılı koşum (en az 5) |
| 🟢 OK | aralik | h04_h02_logitayar: ort 0.8576 ∈ [0.8572, 0.8872] (H04 kartı uygulama notu a3: alt = kısıt, üst = görülmemiş ACC) |
| 🟢 OK | farkli_seedler_farkli | h04_h02_logitayar: seed'ler farklı |
| 🟢 OK | ikincil_aralik | h04_h02_logitayar/s0 val_dengeli_acc=0.7186 ∈ [0.6,0.7662] |
| 🟢 OK | ikincil_aralik | h04_h02_logitayar/s1 val_dengeli_acc=0.7506 ∈ [0.6,0.7662] |
| 🟢 OK | ikincil_aralik | h04_h02_logitayar/s2 val_dengeli_acc=0.7102 ∈ [0.6,0.7662] |
| 🔴 İHLAL | ikincil_aralik | h04_h02_logitayar/s3 val_dengeli_acc=0.8026 ∉ [0.6,0.7662] |
| 🟢 OK | ikincil_aralik | h04_h02_logitayar/s4 val_dengeli_acc=0.7307 ∈ [0.6,0.7662] |
| 🟢 OK | ikincil_aralik | h04_h02_logitayar/s0 val_auc=0.9643 ∈ [0.955,0.98] |
| 🟢 OK | ikincil_aralik | h04_h02_logitayar/s1 val_auc=0.9704 ∈ [0.955,0.98] |
| 🟢 OK | ikincil_aralik | h04_h02_logitayar/s2 val_auc=0.9684 ∈ [0.955,0.98] |
| 🟢 OK | ikincil_aralik | h04_h02_logitayar/s3 val_auc=0.9708 ∈ [0.955,0.98] |
| 🟢 OK | ikincil_aralik | h04_h02_logitayar/s4 val_auc=0.9637 ∈ [0.955,0.98] |
| 🟢 OK | ikincil_aralik | h04_h02_logitayar/s0 tau0_ozdeslik_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h04_h02_logitayar/s1 tau0_ozdeslik_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h04_h02_logitayar/s2 tau0_ozdeslik_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h04_h02_logitayar/s3 tau0_ozdeslik_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h04_h02_logitayar/s4 tau0_ozdeslik_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h04_h02_logitayar/s0 pi_sira_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h04_h02_logitayar/s1 pi_sira_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h04_h02_logitayar/s2 pi_sira_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h04_h02_logitayar/s3 pi_sira_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h04_h02_logitayar/s4 pi_sira_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h04_h02_logitayar/s0 kat_sapma=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | h04_h02_logitayar/s1 kat_sapma=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | h04_h02_logitayar/s2 kat_sapma=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | h04_h02_logitayar/s3 kat_sapma=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | h04_h02_logitayar/s4 kat_sapma=0 ∈ [0,0] |
| 🟢 OK | ikincil_aralik | h04_h02_logitayar/s0 val_dengeli_acc_delta=0.008772 ∈ [-1.0,0.06] |
| 🟢 OK | ikincil_aralik | h04_h02_logitayar/s1 val_dengeli_acc_delta=0.04557 ∈ [-1.0,0.06] |
| 🟢 OK | ikincil_aralik | h04_h02_logitayar/s2 val_dengeli_acc_delta=0.02795 ∈ [-1.0,0.06] |
| 🔴 İHLAL | ikincil_aralik | h04_h02_logitayar/s3 val_dengeli_acc_delta=0.07623 ∉ [-1.0,0.06] |
| 🟢 OK | ikincil_aralik | h04_h02_logitayar/s4 val_dengeli_acc_delta=0.02292 ∈ [-1.0,0.06] |
| 🟢 OK | ikincil_aralik | h04_h02_logitayar/s0 val_acc_delta=-0.01695 ∈ [-1.0,0.02] |
| 🟢 OK | ikincil_aralik | h04_h02_logitayar/s1 val_acc_delta=-0.01496 ∈ [-1.0,0.02] |
| 🟢 OK | ikincil_aralik | h04_h02_logitayar/s2 val_acc_delta=-0.004985 ∈ [-1.0,0.02] |
| 🟢 OK | ikincil_aralik | h04_h02_logitayar/s3 val_acc_delta=-0.008973 ∈ [-1.0,0.02] |
| 🟢 OK | ikincil_aralik | h04_h02_logitayar/s4 val_acc_delta=-0.001994 ∈ [-1.0,0.02] |

## Yapılacak
1. Koşum kuyruğu durdu; sıradaki aşama başlatılmadı.
2. Her İHLAL için hata hipotezi kur (beklenti-ihlali.md): kurulum/veri yolu/seed/metrik hesabı.
3. Düzeltme sonra duman aşamasından yeniden; aynı kuyruk adı tekrar koşulmaz (`<ad>-v2`).
4. Beklenti yanlışsa: kaynağıyla düzelt (eşiği sonucu geçirmek için gevşetme yasak).
