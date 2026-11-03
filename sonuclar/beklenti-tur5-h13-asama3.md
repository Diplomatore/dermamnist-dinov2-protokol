# Beklenti Kapısı — kuyruk `tur5-h13` aşama 3 → **GEÇTİ**
*2026-10-11 18:04* · metrik `val_acc` (max) · beklenti: `tur5-h13.beklenti.json` · 5 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| h13_h02c_h10_lpo | 5 | 0.8961 | 0.8973, 0.8993, 0.8913, 0.9003, 0.8923 | 5.1 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 5 başarılı koşum (en az 5) |
| 🟢 OK | farkli_seedler_farkli | h13_h02c_h10_lpo: seed'ler farklı |
| 🟢 OK | aralik | h13_h02c_h10_lpo: ort 0.8961 ∈ [0.86, 0.93] (H13 uygulama notu a3: aralik [0.86, 0.93] (üst = görülmemiş ACC)) |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s0 val_auc=0.9863 ∈ [0.96,0.995] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s1 val_auc=0.9859 ∈ [0.96,0.995] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s2 val_auc=0.9877 ∈ [0.96,0.995] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s3 val_auc=0.989 ∈ [0.96,0.995] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s4 val_auc=0.9853 ∈ [0.96,0.995] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s0 ozdeslik_tek_uye_maks_fark=1.641e-07 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s1 ozdeslik_tek_uye_maks_fark=1.524e-07 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s2 ozdeslik_tek_uye_maks_fark=1.754e-07 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s3 ozdeslik_tek_uye_maks_fark=1.564e-07 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s4 ozdeslik_tek_uye_maks_fark=1.598e-07 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s0 ozdeslik_tek_uye_argmax_ayni_oran=1 ∈ [1.0,1.0] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s1 ozdeslik_tek_uye_argmax_ayni_oran=1 ∈ [1.0,1.0] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s2 ozdeslik_tek_uye_argmax_ayni_oran=1 ∈ [1.0,1.0] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s3 ozdeslik_tek_uye_argmax_ayni_oran=1 ∈ [1.0,1.0] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s4 ozdeslik_tek_uye_argmax_ayni_oran=1 ∈ [1.0,1.0] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s0 ozdeslik_tek_uye_acc_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s1 ozdeslik_tek_uye_acc_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s2 ozdeslik_tek_uye_acc_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s3 ozdeslik_tek_uye_acc_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s4 ozdeslik_tek_uye_acc_fark=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s0 etiket_hizasi=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s1 etiket_hizasi=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s2 etiket_hizasi=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s3 etiket_hizasi=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s4 etiket_hizasi=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s0 uye_ebeveyn_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s1 uye_ebeveyn_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s2 uye_ebeveyn_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s3 uye_ebeveyn_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s4 uye_ebeveyn_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s0 seed_denet_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s1 seed_denet_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s2 seed_denet_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s3 seed_denet_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s4 seed_denet_ok=1 ∈ [1,1] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s0 satir_toplam_maks_sapma=4.315e-08 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s1 satir_toplam_maks_sapma=3.868e-08 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s2 satir_toplam_maks_sapma=4.424e-08 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s3 satir_toplam_maks_sapma=3.67e-08 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s4 satir_toplam_maks_sapma=4.258e-08 ∈ [0.0,1e-06] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s0 kirp=1e-12 ∈ [1e-12,1e-12] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s1 kirp=1e-12 ∈ [1e-12,1e-12] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s2 kirp=1e-12 ∈ [1e-12,1e-12] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s3 kirp=1e-12 ∈ [1e-12,1e-12] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s4 kirp=1e-12 ∈ [1e-12,1e-12] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s0 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s1 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s2 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s3 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s4 val_n=1003 ∈ [1003,1003] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s0 uye_sayisi=2 ∈ [2,2] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s1 uye_sayisi=2 ∈ [2,2] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s2 uye_sayisi=2 ∈ [2,2] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s3 uye_sayisi=2 ∈ [2,2] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s4 uye_sayisi=2 ∈ [2,2] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s0 agirlik_uye1=0.5 ∈ [0.5,0.5] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s1 agirlik_uye1=0.5 ∈ [0.5,0.5] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s2 agirlik_uye1=0.5 ∈ [0.5,0.5] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s3 agirlik_uye1=0.5 ∈ [0.5,0.5] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s4 agirlik_uye1=0.5 ∈ [0.5,0.5] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s0 param_sayisi=9.691e+07 ∈ [96910414,96910414] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s1 param_sayisi=9.691e+07 ∈ [96910414,96910414] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s2 param_sayisi=9.691e+07 ∈ [96910414,96910414] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s3 param_sayisi=9.691e+07 ∈ [96910414,96910414] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s4 param_sayisi=9.691e+07 ∈ [96910414,96910414] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s0 egitim_sure_sn=4.22 ∈ [0.0,300] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s1 egitim_sure_sn=4.89 ∈ [0.0,300] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s2 egitim_sure_sn=4.38 ∈ [0.0,300] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s3 egitim_sure_sn=4.293 ∈ [0.0,300] |
| 🟢 OK | ikincil_aralik | h13_h02c_h10_lpo/s4 egitim_sure_sn=4.522 ∈ [0.0,300] |
