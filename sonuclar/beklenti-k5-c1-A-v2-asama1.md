# Beklenti Kapısı — kuyruk `k5-c1-A-v2` aşama 1 → **İHLAL — BÜYÜK KOŞUMA GEÇİLMEZ**
*2026-10-11 22:56* · metrik `val_acc` (max) · beklenti: `k5-c1-A-v2.beklenti.json` · 4 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h01_c1_duman | 1 | 0.7305 | 0.7305 | 30.6 |
| h02_c1_duman | 1 | 0.9058 | 0.9058 | 30.9 |
| baseline_c1_duman | 1 | 0.808 | 0.808 | 33.8 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 3 başarılı koşum (en az 3) |
| 🟢 OK | kollar_farkli | kollar birbirinden farklı |
| 🟢 OK | aralik | h01_c1_duman: ort 0.7305 ∈ [0.68, 0.96] (C-val çoğunluk 0,782 − 0,10 (1 epoch / 256 örnek sonda); üst §4.2 0,96) |
| 🟢 OK | aralik | h02_c1_duman: ort 0.9058 ∈ [0.68, 0.96] (C-val çoğunluk 0,782 − 0,10 (1 epoch / 256 örnek sonda); üst §4.2 0,96) |
| 🟢 OK | aralik | baseline_c1_duman: ort 0.808 ∈ [0.68, 0.9] (C-val çoğunluk tabanı 0,782 (448/573) − 0,10 duman payı (3 epoch); üst = §4.2 baseline üst 0,90) |
| 🔴 İHLAL | ikincil_aralik | h01_c1_duman/s0 n_train=256 ∉ [8215,8215] |
| 🟢 OK | ikincil_aralik | h02_c1_duman/s0 n_train=8215 ∈ [8215,8215] |
| 🔴 İHLAL | ikincil_aralik | h01_c1_duman/s0 val_n=256 ∉ [573,573] |
| 🟢 OK | ikincil_aralik | h02_c1_duman/s0 val_n=573 ∈ [573,573] |
| 🟢 OK | ikincil_aralik | h01_c1_duman/s0 tepe_gpu_bellek_mb=667.8 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | h02_c1_duman/s0 tepe_gpu_bellek_mb=1669 ∈ [1,22500] |
| 🟢 OK | ikincil_aralik | baseline_c1_duman/s0 secim_val_auc_farki=0 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h02_c1_duman/s0 ilk_adim_loss=2.028 ∈ [1.5,2.6] |
| 🟢 OK | ikincil_aralik | h01_c1_duman/s0 param_omurga=8.572e+07 ∈ [85724928,85724928] |
| 🟢 OK | ikincil_aralik | h01_c1_duman/s0 oznitelik_determinizm_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h01_c1_duman/s0 ayni_seed_tahmin_maks_fark=0 ∈ [0.0,0.0] |

## Yapılacak
1. Koşum kuyruğu durdu; sıradaki aşama başlatılmadı.
2. Her İHLAL için hata hipotezi kur (beklenti-ihlali.md): kurulum/veri yolu/seed/metrik hesabı.
3. Düzeltme sonra duman aşamasından yeniden; aynı kuyruk adı tekrar koşulmaz (`<ad>-v2`).
4. Beklenti yanlışsa: kaynağıyla düzelt (eşiği sonucu geçirmek için gevşetme yasak).
