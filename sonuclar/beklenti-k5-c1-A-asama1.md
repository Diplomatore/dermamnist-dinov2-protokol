# Beklenti Kapısı — kuyruk `k5-c1-A` aşama 1 → **İHLAL — BÜYÜK KOŞUMA GEÇİLMEZ**
*2026-10-11 21:52* · metrik `val_acc` (max) · beklenti: `k5-c1-A.beklenti.json` · 4 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h02_c1_duman | 1 | 0.9058 | 0.9058 | 30.9 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🔴 İHLAL | mekanik | 2 koşum rc≠0: 20261011-214829_h01_c1_duman_s0(rc=1), 20261011-214933_baseline_c1_duman_s0(rc=1) |
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟡 UYARI | min_kosum | 1 başarılı koşum (en az 3) |
| 🟢 OK | kollar_farkli | kollar birbirinden farklı |
| 🟢 OK | aralik | h02_c1_duman: ort 0.9058 ∈ [0.68, 0.96] (C-val çoğunluk 0,782 − 0,10 (1 epoch / 256 örnek sonda); üst §4.2 0,96) |
| 🟢 OK | ikincil_aralik | h02_c1_duman/s0 n_train=8215 ∈ [8215,8215] |
| 🟢 OK | ikincil_aralik | h02_c1_duman/s0 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h02_c1_duman/s0 tepe_gpu_bellek_mb=1669 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h02_c1_duman/s0 ilk_adim_loss=2.028 ∈ [1.5,2.6] |

## Yapılacak
1. Koşum kuyruğu durdu; sıradaki aşama başlatılmadı.
2. Her İHLAL için hata hipotezi kur (beklenti-ihlali.md): kurulum/veri yolu/seed/metrik hesabı.
3. Düzeltme sonra duman aşamasından yeniden; aynı kuyruk adı tekrar koşulmaz (`<ad>-v2`).
4. Beklenti yanlışsa: kaynağıyla düzelt (eşiği sonucu geçirmek için gevşetme yasak).
