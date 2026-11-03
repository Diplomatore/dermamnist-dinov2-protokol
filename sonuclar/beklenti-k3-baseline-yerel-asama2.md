# Beklenti Kapısı — kuyruk `k3-baseline-yerel` aşama 2 → **GEÇTİ**
*2026-10-10 09:43* · metrik `val_acc` (max) · beklenti: `k3-baseline-yerel.beklenti.json` · 2 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| baseline_onucus | 2 | 0.7817 | 0.7866, 0.7767 | 428.1 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 2 başarılı koşum (en az 2) |
| 🟢 OK | farkli_seedler_farkli | baseline_onucus: seed'ler farklı |
| 🟢 OK | aralik | baseline_onucus: ort 0.7817 ∈ [0.669, 0.8] (Literatürde VAL sayısı yok (yalnız test: Yang 2023 T3 0.735; Doerrich 2025 S4 73.38±0.61; Medeiros 2026 T2 0.742). 20 epoch kısaltılmış eğitim için gevşetilmiş: alt = val çoğunluk tabanı 0.669 (manifest; 'hep nv' tahmini — altı öğrenmeme demek); üst = 28 px tavanı ~0.78 + marj (literatur-cipa.md §4e). Üstü sızıntı/protokol şüphesi) |
| 🟢 OK | ikincil_aralik | baseline_onucus/s0 val_auc=0.9305 ∈ [0.85,0.96] |
| 🟢 OK | ikincil_aralik | baseline_onucus/s1 val_auc=0.9313 ∈ [0.85,0.96] |
| 🟢 OK | ikincil_aralik | baseline_onucus/s0 en_iyi_epoch=11 ∈ [2,20] |
| 🟢 OK | ikincil_aralik | baseline_onucus/s1 en_iyi_epoch=13 ∈ [2,20] |
| 🟢 OK | ikincil_aralik | baseline_onucus/s0 param_sayisi=1.117e+07 ∈ [11172423,11172423] |
| 🟢 OK | ikincil_aralik | baseline_onucus/s1 param_sayisi=1.117e+07 ∈ [11172423,11172423] |
| 🟢 OK | ikincil_aralik | baseline_onucus/s0 secim_val_auc_farki=0 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | baseline_onucus/s1 secim_val_auc_farki=0 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | baseline_onucus/s0 egitim_sure_sn=377.8 ∈ [10,1800] |
| 🟢 OK | ikincil_aralik | baseline_onucus/s1 egitim_sure_sn=461.5 ∈ [10,1800] |
