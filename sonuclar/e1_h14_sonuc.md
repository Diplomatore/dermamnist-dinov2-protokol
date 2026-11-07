# E1 / H14 — test sonrası keşifsel analiz
*2026-11-07T18:09:59* · betik `c38cba390fc05ba7…` · kart `1991c2441e2ea8f1…` · k5_test `13a71cf1375fc98a…`

**Etiketlerin hepsi:** (test sonrası keşifsel, düzeltmesiz). K5 Holm ailelerine girmez; K5 kararı değişmez.

## 1) Kuru koşu özdeşliği (k5_c1_sonuc.json yeniden üretimi) → **GEÇTİ** (tolerans 0.005 puan)
| kıyas | Δ (bu) | GA (bu) | Δ (k5_c1) | GA (k5_c1) | maks fark |
|---|---|---|---|---|---|
| h10-h08 | -0.9617 | [-2.1027; +0.1793] | -0.9617 | [-2.1027; +0.1793] | 0.00e+00 |
| h08-h02 | +2.0864 | [+1.2388; +2.9666] | +2.0864 | [+1.2388; +2.9666] | 0.00e+00 |
| h10-h02 | +1.1247 | [-0.0493; +2.2983] | +1.1247 | [-0.0493; +2.2983] | 0.00e+00 |

## 2) Kural (f) Durum: **A** · K1 karşılaştırıcısı `H10`
- {"kontrol_ilk_adim_loss_fark": 0.0, "kontrol_argmax_oran": 1.0, "kontrol_p_maks_fark": 1.341104507446289e-06, "kontrol_val_acc_fark": 0.0, "kontrol_surum_fark": "cudnn: 92700 → 91900", "kontrol_durum": "A", "kontrol_run": "20261107-161257_h14_c1_d4kapali_duman_s0", "d4_duman_ilk_adim_loss_fark": 0.03211057186126709}

## 3) Kollar (C-test, seed ortalaması)
| kol | n | ACC | dengeli | dengeli-5 | mel duy. | makro-F1 | ECE-15 | NLL | nv payı | mel→nv | ACC − 0,7726 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H14 | 5 | 0.9198 ± 0.0039 | 0.8035 | 0.7535 | 0.6457 | 0.8074 | 0.0475 | 0.3025 | 0.7788 | 14.4 | +0.1472 |
| H10 | 5 | 0.9161 ± 0.0040 | 0.7622 | 0.7426 | 0.6686 | 0.7804 | 0.0487 | 0.3297 | 0.7848 | 14.6 | +0.1435 |
| H08c | 5 | 0.9257 ± 0.0059 | 0.7963 | 0.7757 | 0.6629 | 0.8063 | 0.0407 | 0.2681 | 0.7791 | 14.4 | +0.1531 |
| H02c | 5 | 0.9048 ± 0.0099 | 0.7047 | 0.7228 | 0.5971 | 0.7333 | 0.0420 | 0.3313 | 0.7844 | 17.6 | +0.1322 |

## 4) Kıyaslar (B − A; görüntü bootstrap'ı B = 2.000, rng 0; düzeltmesiz)
| kıyas | metrik | Δ (puan) | %95 GA | p(Δ*≤0) / p(Δ*≥0) | etiket |
|---|---|---|---|---|---|
| K1: H14 − H10 | acc | +0.37 | [-0.20; +0.98] | 0.093 / 0.909 | ayırt edilemedi (test sonrası keşifsel, düzeltmesiz) |
| K1: H14 − H10 | dengeli_acc | +4.14 | [+0.92; +7.51] | — | H14 üstün (test sonrası keşifsel, düzeltmesiz) |
| K1: H14 − H10 | dengeli_acc_5sinif | +1.09 | [-0.90; +3.07] | — | ayırt edilemedi (test sonrası keşifsel, düzeltmesiz) |
| K1: H14 − H10 | mel_duyarlilik | -2.29 | [-6.47; +2.19] | 0.86 / 0.155 | ayırt edilemedi (test sonrası keşifsel, düzeltmesiz) |
| K1: H14 − H10 | seed t (ACC, iki yönlü, bilgi) | +0.37 | — | p = 0.154 | — |
| K2: H14 − H08c | acc | -0.59 | [-1.68; +0.51] | 0.86 / 0.141 | ayırt edilemedi (test sonrası keşifsel, düzeltmesiz) |
| K2: H14 − H08c | dengeli_acc | +0.73 | [-3.79; +6.33] | — | ayırt edilemedi (test sonrası keşifsel, düzeltmesiz) |
| K2: H14 − H08c | dengeli_acc_5sinif | -2.22 | [-6.40; +2.00] | — | ayırt edilemedi (test sonrası keşifsel, düzeltmesiz) |
| K2: H14 − H08c | mel_duyarlilik | -1.71 | [-10.36; +6.08] | 0.679 / 0.334 | ayırt edilemedi (test sonrası keşifsel, düzeltmesiz) |
| K2: H14 − H08c | seed t (ACC, iki yönlü, bilgi) | -0.59 | — | p = 0.155 | — |
| betimsel: H14 − H02c | acc | +1.50 | [+0.34; +2.69] | 0.005 / 0.997 | H14 üstün (test sonrası keşifsel, düzeltmesiz) |
| betimsel: H14 − H02c | dengeli_acc | +9.88 | [+5.17; +14.71] | — | H14 üstün (test sonrası keşifsel, düzeltmesiz) |
| betimsel: H14 − H02c | dengeli_acc_5sinif | +3.07 | [-1.17; +7.14] | — | ayırt edilemedi (test sonrası keşifsel, düzeltmesiz) |
| betimsel: H14 − H02c | mel_duyarlilik | +4.86 | [-3.82; +13.44] | 0.143 / 0.862 | ayırt edilemedi (test sonrası keşifsel, düzeltmesiz) |
| betimsel: H14 − H02c | seed t (ACC, iki yönlü, bilgi) | +1.50 | — | p = 0.0363 | — |

## 5) Görülmemiş (İNCELE), manipülasyon ve kural notları
- Görülmemiş: {"(i) K1 Δ > +4,2": false, "(ii) H14 C-test ACC > 0,9457": false, "(iii) H14 C-val ACC > 0,96": false, "(iv) dengeli Δ(K1) > +16,0": false, "INCELE": false}
- Manipülasyon/kural notları: {"butce_E1_L4sa": 1.597, "manipulasyon_s0_ep5_loss": 0.13742105013991882, "manipulasyon_esik_>0,069": true, "kural_j_notu": false, "secilen_epoch_ort": 8.0, "secilen_epoch_>7,0": true, "cval_acc_ort": 0.9277486910994763, "epoch_sure_ort_sn": 54.256533333333344, "kural_e_az_egitim": false, "ece_H14_<=_K1ref": true, "nv_yanliligi_notu": false}

## 6) Önceden yazılmış cümle (mekanik seçim)
- **K2_etiketi:** ayırt edilemedi
- **K2_cumle:** Augmentasyon simetrik hâle getirildiğinde de (H14 = H10 + D4) DINOv2-B ile D4'lü ResNet-18 lezyon-ayrık C1 bölmesinde ayırt edilemedi (Δ −0,59 puan [−1,68; +0,51]). Bu, asimetrik kıyasın ana bulguyu değiştirmediği yönünde keşifsel bir kanıttır. GA genişliği yaklaşık ±1,09 puanlık farkları dışlamaz. Analiz test sonrası ve düzeltmesizdir.
- **K1_etiketi:** ayırt edilemedi
- **K1_eslik_cumle:** D4 augmentasyonu DINOv2-B'de C1'de ayırt edilebilir bir değişiklik yaratmadı (Δ +0,37 [−0,20; +0,98]). Karşılaştırma için aynı augmentasyonun ResNet-18'e kazancı +2,09 [+1,24; +2,97] idi.
- **dengeli_mel_cumle:** K1 (H14 − H10): dengeli doğruluk Δ +4,14 puan [+0,92; +7,51] (H14 üstün); mel duyarlılığı Δ −2,29 puan [−6,47; +2,19] (ayırt edilemedi; n_mel = 70); K2 (H14 − H08c): dengeli doğruluk Δ +0,73 puan [−3,79; +6,33] (ayırt edilemedi); mel duyarlılığı Δ −1,71 puan [−10,36; +6,08] (ayırt edilemedi; n_mel = 70) (test sonrası keşifsel, düzeltmesiz).
- **n_seed:** 5
- **sonuctan_bagimsiz_oneri:** Özet ve sonuç bölümündeki H08c ↔ H10 cümlesine '(ResNet-18'e D4 eklendi, DINOv2-B'ye eklenmedi; simetrik kıyas için bkz. Bölüm [E1])' kaydı (yazım kararı makale-yazıcı/kullanıcıda).
