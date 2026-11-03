# Beklenti Kapısı — kuyruk `tur2-h05-on` aşama 1 → **GEÇTİ**
*2026-10-11 01:24* · metrik `val_acc` (max) · beklenti: `tur2-h05.beklenti.json` · 2 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h05_r18in_ft224_cw00_duman | 1 | 0.8116 | 0.8116 | 64.5 |
| h05_r18in_ft224_cw05_duman | 1 | 0.7866 | 0.7866 | 64.7 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 2 başarılı koşum (en az 2) |
| 🟢 OK | aralik | h05_r18in_ft224_cw00_duman: ort 0.8116 ∈ [0.62, 0.9] (H05 kartı a1: alt = çoğunluk tabanı 0,669 − 0,05 (H02 a1 ile aynı), üst = H02 tavanı 0,90) |
| 🟢 OK | aralik | h05_r18in_ft224_cw05_duman: ort 0.7866 ∈ [0.62, 0.9] (H05 kartı a1: alt = çoğunluk tabanı 0,669 − 0,05 (H02 a1 ile aynı), üst = H02 tavanı 0,90) |
| 🟢 OK | kollar_farkli | kollar birbirinden farklı |
| 🟢 OK | ikincil_aralik | h05_r18in_ft224_cw00_duman/s0 ozdeslik_ilk_adim_loss_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h05_r18in_ft224_cw00_duman/s0 ozdeslik_ilk_epoch_loss_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h05_r18in_ft224_cw00_duman/s0 ozdeslik_val_acc_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h05_r18in_ft224_cw00_duman/s0 param_sayisi=1.118e+07 ∈ [11180103,11180103] |
| 🟢 OK | ikincil_aralik | h05_r18in_ft224_cw05_duman/s0 param_sayisi=1.118e+07 ∈ [11180103,11180103] |
| 🟢 OK | ikincil_aralik | h05_r18in_ft224_cw00_duman/s0 egitim_sure_sn=45.7 ∈ [10,300] |
| 🟢 OK | ikincil_aralik | h05_r18in_ft224_cw05_duman/s0 egitim_sure_sn=46.02 ∈ [10,300] |
| 🟢 OK | ikincil_aralik | h05_r18in_ft224_cw00_duman/s0 epoch_sure_ort_sn=45.7 ∈ [10,90] |
| 🟢 OK | ikincil_aralik | h05_r18in_ft224_cw05_duman/s0 epoch_sure_ort_sn=46.02 ∈ [10,90] |
| 🟢 OK | ikincil_aralik | h05_r18in_ft224_cw00_duman/s0 ilk_adim_loss=1.852 ∈ [1.5,2.5] |
| 🟢 OK | ikincil_aralik | h05_r18in_ft224_cw05_duman/s0 ilk_adim_loss=2.04 ∈ [1.5,2.5] |
| 🟢 OK | ikincil_aralik | h05_r18in_ft224_cw05_duman/s0 sinif_agirlik_df=4.424 ∈ [4.37,4.47] |
| 🟢 OK | ikincil_aralik | h05_r18in_ft224_cw05_duman/s0 sinif_agirlik_nv=0.5776 ∈ [0.57,0.59] |
| 🟢 OK | ikincil_aralik | h05_r18in_ft224_cw05_duman/s0 sinif_agirlik_akiec=2.62 ∈ [2.57,2.67] |
| 🟢 OK | ikincil_aralik | h05_r18in_ft224_cw00_duman/s0 tepe_gpu_bellek_mb=1742 ∈ [1,3900] |
| 🟢 OK | ikincil_aralik | h05_r18in_ft224_cw05_duman/s0 tepe_gpu_bellek_mb=1742 ∈ [1,3900] |
