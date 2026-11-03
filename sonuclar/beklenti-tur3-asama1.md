# Beklenti Kapısı — kuyruk `tur3` aşama 1 → **GEÇTİ**
*2026-10-11 04:41* · metrik `val_acc` (max) · beklenti: `tur3.beklenti.json` · 2 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h08_r18in_ft224_d4_duman | 1 | 0.8096 | 0.8096 | 65.7 |
| h09_r18in_ft224_renk02_duman | 1 | 0.8046 | 0.8046 | 65.8 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 2 başarılı koşum (en az 2) |
| 🟢 OK | aralik | h08_r18in_ft224_d4_duman: ort 0.8096 ∈ [0.62, 0.9] (H08/H09 a1: alt = çoğunluk tabanı − 0,05, üst = H02 tavanı) |
| 🟢 OK | aralik | h09_r18in_ft224_renk02_duman: ort 0.8046 ∈ [0.62, 0.9] (H08/H09 a1: alt = çoğunluk tabanı − 0,05, üst = H02 tavanı) |
| 🟢 OK | kollar_farkli | kollar birbirinden farklı |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_duman/s0 ozdeslik_ilk_adim_loss_fark=0.007363 ∈ [1e-07,10.0] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02_duman/s0 ozdeslik_ilk_adim_loss_fark=0.03662 ∈ [1e-07,10.0] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_duman/s0 augment_d4=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_duman/s0 augment_toplam_ornek=7007 ∈ [7007,7007] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02_duman/s0 augment_renk=0.2 ∈ [0.2,0.2] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_duman/s0 param_sayisi=1.118e+07 ∈ [11180103,11180103] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02_duman/s0 param_sayisi=1.118e+07 ∈ [11180103,11180103] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_duman/s0 egitim_sure_sn=46.72 ∈ [10,300] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02_duman/s0 egitim_sure_sn=47.31 ∈ [10,300] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_duman/s0 epoch_sure_ort_sn=46.72 ∈ [10,90] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02_duman/s0 epoch_sure_ort_sn=47.31 ∈ [10,90] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_duman/s0 ilk_adim_loss=1.845 ∈ [1.5,2.6] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02_duman/s0 ilk_adim_loss=1.816 ∈ [1.5,2.6] |
| 🟢 OK | ikincil_aralik | h08_r18in_ft224_d4_duman/s0 tepe_gpu_bellek_mb=1742 ∈ [1,3900] |
| 🟢 OK | ikincil_aralik | h09_r18in_ft224_renk02_duman/s0 tepe_gpu_bellek_mb=1742 ∈ [1,3900] |
