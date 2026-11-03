# Beklenti Kapısı — kuyruk `k3-baseline-yerel` aşama 1 → **GEÇTİ**
*2026-10-10 09:29* · metrik `val_acc` (max) · beklenti: `k3-baseline-yerel.beklenti.json` · 1 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| baseline_duman | 1 | 0.7089 | 0.7089 | 66.1 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 1 başarılı koşum (en az 1) |
| 🟢 OK | aralik | baseline_duman: ort 0.7089 ∈ [0.619, 0.8] (alt: val çoğunluk tabanı 671/1003=0.669 (data/data_manifest.md) − 0.05 gevşetme (3 epoch); üst: 28 px bilgiyle bilinen tavan ~0.78 (literatur-cipa.md §4e: DINOv2 LP-28 77.37, MedViT-S 0.780) + marj. Yerel GTX 1650 dumanı (seed 0, 3 ep) val_acc 0.7089 — ölçüm değil, kod doğruluğu) |
| 🟢 OK | ikincil_aralik | baseline_duman/s0 val_auc=0.9067 ∈ [0.6,0.96] |
| 🟢 OK | ikincil_aralik | baseline_duman/s0 param_sayisi=1.117e+07 ∈ [11172423,11172423] |
| 🟢 OK | ikincil_aralik | baseline_duman/s0 secim_val_auc_farki=0 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | baseline_duman/s0 egitim_sure_sn=58.9 ∈ [1,900] |
