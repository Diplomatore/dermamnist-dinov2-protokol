# E1 / H14 — test sonrası keşifsel analiz — KURU KOŞU (H14 yerine arşiv H10)
*2026-11-03T23:13:13* · betik `c38cba390fc05ba7…` · kart `1991c2441e2ea8f1…` · k5_test `13a71cf1375fc98a…`

**Etiketlerin hepsi:** (test sonrası keşifsel, düzeltmesiz). K5 Holm ailelerine girmez; K5 kararı değişmez.

> **KURU KOŞU:** H14 kolu = arşiv `h10_c1-test` (aynı tahminler). K1 = 0 beklenir; K2 = H10 − H08 (C1 (i) satırı). Yeni C-test açılmadı; yalnız K5 C1'de açılmış arşiv tahminleri okundu.

## 1) Kuru koşu özdeşliği (k5_c1_sonuc.json yeniden üretimi) → **GEÇTİ** (tolerans 0.005 puan)
| kıyas | Δ (bu) | GA (bu) | Δ (k5_c1) | GA (k5_c1) | maks fark |
|---|---|---|---|---|---|
| h10-h08 | -0.9617 | [-2.1027; +0.1793] | -0.9617 | [-2.1027; +0.1793] | 0.00e+00 |
| h08-h02 | +2.0864 | [+1.2388; +2.9666] | +2.0864 | [+1.2388; +2.9666] | 0.00e+00 |
| h10-h02 | +1.1247 | [-0.0493; +2.2983] | +1.1247 | [-0.0493; +2.2983] | 0.00e+00 |

## 2) Kural (f) Durum: **KURU (atlandı)** · K1 karşılaştırıcısı `h10_c1 (arşiv)`
- "kuru koşu: kontrol kolu yok; Durum A/B/C gerçek analizde kontrol_* metriklerinden"

## 3) Kollar (C-test, seed ortalaması)
| kol | n | ACC | dengeli | dengeli-5 | mel duy. | makro-F1 | ECE-15 | NLL | nv payı | mel→nv | ACC − 0,7726 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H14 | 5 | 0.9161 ± 0.0040 | 0.7622 | 0.7426 | 0.6686 | 0.7804 | 0.0487 | 0.3297 | 0.7848 | 14.6 | +0.1435 |
| H10 | 5 | 0.9161 ± 0.0040 | 0.7622 | 0.7426 | 0.6686 | 0.7804 | 0.0487 | 0.3297 | 0.7848 | 14.6 | +0.1435 |
| H08c | 5 | 0.9257 ± 0.0059 | 0.7963 | 0.7757 | 0.6629 | 0.8063 | 0.0407 | 0.2681 | 0.7791 | 14.4 | +0.1531 |
| H02c | 5 | 0.9048 ± 0.0099 | 0.7047 | 0.7228 | 0.5971 | 0.7333 | 0.0420 | 0.3313 | 0.7844 | 17.6 | +0.1322 |

## 4) Kıyaslar (B − A; görüntü bootstrap'ı B = 2.000, rng 0; düzeltmesiz)
| kıyas | metrik | Δ (puan) | %95 GA | p(Δ*≤0) / p(Δ*≥0) | etiket |
|---|---|---|---|---|---|
| K1: H14 − H10 | acc | +0.00 | [+0.00; +0.00] | 1 / 1 | ayırt edilemedi (test sonrası keşifsel, düzeltmesiz) |
| K1: H14 − H10 | dengeli_acc | +0.00 | [+0.00; +0.00] | — | ayırt edilemedi (test sonrası keşifsel, düzeltmesiz) |
| K1: H14 − H10 | dengeli_acc_5sinif | +0.00 | [+0.00; +0.00] | — | ayırt edilemedi (test sonrası keşifsel, düzeltmesiz) |
| K1: H14 − H10 | mel_duyarlilik | +0.00 | [+0.00; +0.00] | 1 / 1 | ayırt edilemedi (test sonrası keşifsel, düzeltmesiz) |
| K1: H14 − H10 | seed t (ACC, iki yönlü, bilgi) | +0.00 | — | p = 1 | — |
| K2: H14 − H08c | acc | -0.96 | [-2.10; +0.18] | 0.951 / 0.0495 | ayırt edilemedi (test sonrası keşifsel, düzeltmesiz) |
| K2: H14 − H08c | dengeli_acc | -3.41 | [-8.30; +1.42] | — | ayırt edilemedi (test sonrası keşifsel, düzeltmesiz) |
| K2: H14 − H08c | dengeli_acc_5sinif | -3.30 | [-7.57; +0.97] | — | ayırt edilemedi (test sonrası keşifsel, düzeltmesiz) |
| K2: H14 − H08c | mel_duyarlilik | +0.57 | [-8.26; +9.03] | 0.46 / 0.552 | ayırt edilemedi (test sonrası keşifsel, düzeltmesiz) |
| K2: H14 − H08c | seed t (ACC, iki yönlü, bilgi) | -0.96 | — | p = 0.0758 | — |
| betimsel: H14 − H02c | acc | +1.12 | [-0.05; +2.30] | 0.031 / 0.971 | ayırt edilemedi (test sonrası keşifsel, düzeltmesiz) |
| betimsel: H14 − H02c | dengeli_acc | +5.74 | [+1.21; +10.17] | — | H14 üstün (test sonrası keşifsel, düzeltmesiz) |
| betimsel: H14 − H02c | dengeli_acc_5sinif | +1.98 | [-2.05; +6.11] | — | ayırt edilemedi (test sonrası keşifsel, düzeltmesiz) |
| betimsel: H14 − H02c | mel_duyarlilik | +7.14 | [-2.11; +16.13] | 0.0645 / 0.941 | ayırt edilemedi (test sonrası keşifsel, düzeltmesiz) |
| betimsel: H14 − H02c | seed t (ACC, iki yönlü, bilgi) | +1.12 | — | p = 0.0181 | — |

## 5) Görülmemiş (İNCELE), manipülasyon ve kural notları
- Görülmemiş: {"(i) K1 Δ > +4,2": false, "(ii) H14 C-test ACC > 0,9457": false, "(iii) H14 C-val ACC > 0,96": false, "(iv) dengeli Δ(K1) > +16,0": false, "INCELE": false}
- Manipülasyon/kural notları: {"manipulasyon_s0_ep5_loss": 0.03447580994472443, "manipulasyon_esik_>0,069": false, "kural_j_notu": true, "secilen_epoch_ort": 7.0, "secilen_epoch_>7,0": false, "cval_acc_ort": 0.9260034904013962, "epoch_sure_ort_sn": 53.0012, "kural_e_az_egitim": false, "ece_H14_<=_K1ref": true, "nv_yanliligi_notu": false, "kural_j": "D4 devrede ama ezberi geciktirmedi"}

## 6) Önceden yazılmış cümle (mekanik seçim)
- **K2_etiketi:** ayırt edilemedi
- **K2_cumle:** Augmentasyon simetrik hâle getirildiğinde de (H14 = H10 + D4) DINOv2-B ile D4'lü ResNet-18 lezyon-ayrık C1 bölmesinde ayırt edilemedi (Δ −0,96 puan [−2,10; +0,18]). Bu, asimetrik kıyasın ana bulguyu değiştirmediği yönünde keşifsel bir kanıttır. GA genişliği yaklaşık ±1,14 puanlık farkları dışlamaz. Analiz test sonrası ve düzeltmesizdir.
- **K1_etiketi:** ayırt edilemedi
- **K1_eslik_cumle:** D4 augmentasyonu DINOv2-B'de C1'de ayırt edilebilir bir değişiklik yaratmadı (Δ +0,00 [+0,00; +0,00]). Karşılaştırma için aynı augmentasyonun ResNet-18'e kazancı +2,09 [+1,24; +2,97] idi.
- **dengeli_mel_cumle:** K1 (H14 − H10): dengeli doğruluk Δ +0,00 puan [+0,00; +0,00] (ayırt edilemedi); mel duyarlılığı Δ +0,00 puan [+0,00; +0,00] (ayırt edilemedi; n_mel = 70); K2 (H14 − H08c): dengeli doğruluk Δ −3,41 puan [−8,30; +1,42] (ayırt edilemedi); mel duyarlılığı Δ +0,57 puan [−8,26; +9,03] (ayırt edilemedi; n_mel = 70) (test sonrası keşifsel, düzeltmesiz).
- **n_seed:** 5
- **sonuctan_bagimsiz_oneri:** Özet ve sonuç bölümündeki H08c ↔ H10 cümlesine '(ResNet-18'e D4 eklendi, DINOv2-B'ye eklenmedi; simetrik kıyas için bkz. Bölüm [E1])' kaydı (yazım kararı makale-yazıcı/kullanıcıda).
