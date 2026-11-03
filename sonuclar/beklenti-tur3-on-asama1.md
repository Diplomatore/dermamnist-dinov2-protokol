# Beklenti Kapısı — kuyruk `tur3-on` aşama 1 → **GEÇTİ**
*2026-10-11 04:33* · metrik `val_acc` (max) · beklenti: `tur3-on.beklenti.json` · 2 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h08_r18in_ft224_d4kapali_duman | 1 | 0.8116 | 0.8116 | 64.5 |
| h09_r18in_ft224_renk02kapali_duman | 1 | 0.8116 | 0.8116 | 64.7 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 2 başarılı koşum (en az 2) |
| 🟢 OK | aralik | h08_r18in_ft224_d4kapali_duman: ort 0.8116 ∈ [0.62, 0.9] (H08/H09 a1: alt = çoğunluk tabanı 0,669 − 0,05, üst = H02 tavanı) |
| 🟢 OK | aralik | h09_r18in_ft224_renk02kapali_duman: ort 0.8116 ∈ [0.62, 0.9] (H08/H09 a1: alt = çoğunluk tabanı 0,669 − 0,05, üst = H02 tavanı) |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4kapali_duman/s0 ozdeslik_ilk_adim_loss_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02kapali_duman/s0 ozdeslik_ilk_adim_loss_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4kapali_duman/s0 ozdeslik_ilk_epoch_loss_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02kapali_duman/s0 ozdeslik_ilk_epoch_loss_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4kapali_duman/s0 ozdeslik_val_acc_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02kapali_duman/s0 ozdeslik_val_acc_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4kapali_duman/s0 param_sayisi=1.118e+07 ∈ [11180103,11180103] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02kapali_duman/s0 param_sayisi=1.118e+07 ∈ [11180103,11180103] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4kapali_duman/s0 egitim_sure_sn=45.86 ∈ [10,300] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02kapali_duman/s0 egitim_sure_sn=46.14 ∈ [10,300] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4kapali_duman/s0 epoch_sure_ort_sn=45.86 ∈ [10,90] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02kapali_duman/s0 epoch_sure_ort_sn=46.14 ∈ [10,90] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4kapali_duman/s0 tepe_gpu_bellek_mb=1742 ∈ [1,3900] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02kapali_duman/s0 tepe_gpu_bellek_mb=1742 ∈ [1,3900] |
