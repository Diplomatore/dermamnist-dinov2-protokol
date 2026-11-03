# Beklenti Kapısı — kuyruk `tur1` aşama 1 → **GEÇTİ**
*2026-10-10 20:41* · metrik `val_acc` (max) · beklenti: `tur1.beklenti.json` · 3 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h03_dinov2b_lp_r28_duman | 1 | 0.6445 | 0.6445 | 37.1 |
| h01_dinov2b_lp_r224_duman | 1 | 0.6523 | 0.6523 | 58.2 |
| h02_r18in_ft224_duman | 1 | 0.8116 | 0.8116 | 71.0 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 3 başarılı koşum (en az 3) |
| 🟢 OK | aralik | h03_dinov2b_lp_r28_duman: ort 0.6445 ∈ [0.55, 0.95] (varyantlar/KOMUTLAR.md §5 h01/h03 a1 taslağı (20:10, duman öncesi): 256 train, tek C, yakınsamamış sonda → alt ≈ çoğunluk tabanı 0,669 − 0,12; üst gevşek (alt küme, ölçüm değil)) |
| 🟢 OK | aralik | h01_dinov2b_lp_r224_duman: ort 0.6523 ∈ [0.55, 0.95] (varyantlar/KOMUTLAR.md §5 h01 a1 taslağı (aynı gerekçe)) |
| 🟢 OK | aralik | h02_r18in_ft224_duman: ort 0.8116 ∈ [0.62, 0.9] (varyantlar/KOMUTLAR.md §5 h02 a1 (20:10): alt = çoğunluk tabanı 0,669 − 0,05 (rakip duman gevşetmesiyle aynı); üst = H02 görülmemiş eşiği 0,90 (literatur-cipa §4a)) |
| 🟢 OK | kollar_farkli | kollar birbirinden farklı |
| 🟢 OK | ikincil_aralik | h03_dinov2b_lp_r28_duman/s0 ayni_seed_tahmin_maks_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_duman/s0 ayni_seed_tahmin_maks_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h03_dinov2b_lp_r28_duman/s0 seed_tahmin_en_kucuk_fark=0.135 ∈ [1e-06,1.0] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_duman/s0 seed_tahmin_en_kucuk_fark=0.1389 ∈ [1e-06,1.0] |
| 🟢 OK | ikincil_aralik | h03_dinov2b_lp_r28_duman/s0 param_omurga=8.572e+07 ∈ [85724928,85724928] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_duman/s0 param_omurga=8.572e+07 ∈ [85724928,85724928] |
| 🟢 OK | ikincil_aralik | h03_dinov2b_lp_r28_duman/s0 oznitelik_boyut=768 ∈ [768,768] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_duman/s0 oznitelik_boyut=768 ∈ [768,768] |
| 🟢 OK | ikincil_aralik | h03_dinov2b_lp_r28_duman/s0 oznitelik_determinizm_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_duman/s0 oznitelik_determinizm_maks_fark=0 ∈ [0.0,1e-05] |
| 🟢 OK | ikincil_aralik | h03_dinov2b_lp_r28_duman/s0 oznitelik_img_s=23.2 ∈ [10,1000] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_duman/s0 oznitelik_img_s=23.33 ∈ [10,1000] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_duman/s0 epoch_sure_ort_sn=48.22 ∈ [10,90] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_duman/s0 ilk_adim_loss=1.852 ∈ [1.5,2.5] |
| 🟢 OK | ikincil_aralik | h03_dinov2b_lp_r28_duman/s0 egitim_sure_sn=33.34 ∈ [1,600] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_duman/s0 egitim_sure_sn=53.88 ∈ [1,600] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_duman/s0 egitim_sure_sn=48.22 ∈ [1,600] |
| 🟢 OK | ikincil_aralik | h03_dinov2b_lp_r28_duman/s0 tepe_gpu_bellek_mb=642.9 ∈ [1,3900] |
| 🟢 OK | ikincil_aralik | h01_dinov2b_lp_r224_duman/s0 tepe_gpu_bellek_mb=642.9 ∈ [1,3900] |
| 🟢 OK | ikincil_aralik | h02_r18in_ft224_duman/s0 tepe_gpu_bellek_mb=1742 ∈ [1,3900] |
