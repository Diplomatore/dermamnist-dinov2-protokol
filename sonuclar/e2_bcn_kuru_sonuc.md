# E2 / H15 — BCN20000 dış merkez analizi — KURU KOŞU (VEKİL dış veri = C-test; BCN AÇILMADI)
*2026-11-07T20:53:39* · betik `cdd056f46e2710b1…` · kart `eeef871456c78ee3…` · k5_test `13a71cf1375fc98a…`

**Etiketler:** B1/B2 Holm (m = 2, iki yönlü); ikinciller (keşifsel, düzeltmesiz). K5 kararı ve Holm aileleri değişmez.

> **KURU KOŞU:** dış küme = C-test (1.227): resmî kollar → arşiv resmî test tahminlerinin C-test satırları (C0), C1 kolları → arşiv C-test tahminleri. 'SCC', kopya ve lezyon birleştirmesi SAHTEDİR (kod yolu sınaması). Sayılar BCN sonucu DEĞİLDİR.

## 1) K-d kuru koşu özdeşliği → **GEÇTİ** (tolerans 0.005 puan)
| satır | bu (Δ; GA) | donmuş JSON | kart | maks fark (donmuş / kart) |
|---|---|---|---|---|
| R_acc | +2,26; [+1,15; +3,30] | +2,2643 | +2,26 | 0.0e+00 / 0.0043 |
| R_deng | +6,03; [+2,86; +9,37] | +6,0315 | +6,03 | 0.0e+00 / 0.0033 |
| C1_acc | +1,12; [−0,05; +2,30] | +1,1247 | +1,12 | 0.0e+00 / 0.0047 |
| C1_deng | +5,74; [+1,21; +10,17] | +5,7427 | +5,74 | 0.0e+00 / 0.0037 |
| C0_acc | +0,28; [−0,83; +1,47] | +0,2771 | +0,28 | 0.0e+00 / 0.0029 |
| C0_deng | +4,45; [−0,79; +10,10] | +4,4542 | +4,45 | 0.0e+00 / 0.0042 |
| R_acc_h08 | −0,36; [−1,40; +0,70] | −0,3591 | −0,36 | 0.0e+00 / 0.0009 |
| C1_acc_h08 | −0,96; [−2,10; +0,18] | −0,9617 | −0,96 | 0.0e+00 / 0.0017 |

## 1b) Vekil kontrol yapılandırması (E2 eşleme + ham küme = C-test) → **GEÇTİ**
| denetim | bu | beklenen | fark |
|---|---|---|---|
| dis_R_dengeli = C0 dengeli | +4.4541627929 | +4.4541627929 | 1.2e-14 |
| dis_R_accpiC = C0 ACC | +0.2770986145 | +0.2770986145 | 1.8e-14 |
| dis_C1_accpiC = C1 ACC | +1.1246943765 | +1.1246943765 | 1.6e-14 |
| dis_C1_dengeli = C1 dengeli | +5.7427440481 | +5.7427440481 | 1.1e-14 |
| e_C1_pi = 0 | +0.0000000000 | +0.0000000000 | 1.1e-14 |
| e_C0_piC = 0 | +0.0000000000 | +0.0000000000 | 1.1e-14 |

## 2) Ana yapılandırma: E1 eşleme, kopyasız küme, merkez_kare — n = 1213, lezyon 1136 (bootstrap kümesi 1135), sınıf görüntü [23, 44, 105, 8, 69, 947, 17]
π_R = [0.0329, 0.0514, 0.1097, 0.0115, 0.1112, 0.6688, 0.0145] · π_C = [0.0285, 0.0359, 0.0856, 0.0065, 0.057, 0.7726, 0.0139] (iç test önselleri; etiketlerden)

| kol | rol | BCN dengeli [GA] | ACC_π_R | ACC_π_C | ham ACC | makro-AUC | mel duy. | iç lezyon-ayrık dengeli | düşüş (puan) |
|---|---|---|---|---|---|---|---|---|---|
| H10 | birincil | 0.7679 [0.7117; 0.8172] | 0.8848 | 0.9133 | 0.9171 | 0.9855 | 0.6667 | 0.7654 | +0,25 |
| H02c | birincil | 0.7184 [0.6505; 0.7808] | 0.8793 | 0.9095 | 0.9134 | 0.9775 | 0.6406 | 0.7209 | −0,25 |
| h10_c1 | birincil | 0.7701 [0.7124; 0.8209] | 0.8882 | 0.9175 | 0.9204 | 0.9867 | 0.6638 | 0.7622 | +0,80 |
| h02_c1 | birincil | 0.7012 [0.6353; 0.7616] | 0.8709 | 0.9039 | 0.9082 | 0.9774 | 0.5913 | 0.7047 | −0,35 |
| H08c | ikincil | 0.7975 [0.7410; 0.8471] | 0.8992 | 0.9238 | 0.9273 | 0.9848 | 0.6812 | 0.8035 | −0,60 |
| h08_c1 | ikincil | 0.7953 [0.7390; 0.8446] | 0.8987 | 0.9253 | 0.9281 | 0.9861 | 0.6580 | 0.7963 | −0,10 |
| h14_c1 | ikincil | 0.8098 [0.7649; 0.8516] | 0.8935 | 0.9209 | 0.9240 | 0.9870 | 0.6406 | 0.8035 | +0,63 |
| H11 | betimsel | 0.7778 [0.7211; 0.8288] | 0.8893 | 0.9179 | 0.9212 | 0.9874 | 0.6348 | 0.7777 | +0,01 |
| H01c | betimsel | 0.6283 [0.5565; 0.6984] | 0.8320 | 0.8721 | 0.8770 | 0.9734 | 0.5797 | 0.6345 | −0,62 |
| H12 | betimsel | 0.6710 [0.5995; 0.7380] | 0.8380 | 0.8783 | 0.8824 | 0.9670 | 0.5507 | 0.6710 | +0,01 |
| h01_c1 | betimsel | 0.5983 [0.5201; 0.6746] | 0.8323 | 0.8750 | 0.8790 | 0.9767 | 0.5739 | 0.5998 | −0,15 |

## 3) Birincil aile (Holm m = 2, iki yönlü; B = 10.000 iki düzeyli bootstrap)
| | nokta (puan) | %95 GA | p | Holm p | etiket |
|---|---|---|---|---|---|
| **B1** Δ_dış,R dengeli (H10 − H02c) | +4,95 | [−1,23; +11,40] | 0,12 | 0,24 | **ayırt edilemedi** |
| **B2** D_π = \|e_R\| − \|e_C1\| | +1,48 | [−1,55; +3,22] | 0,437 | 0,437 | **protokoller ayırt edilemedi** |

| bileşen | nokta | %95 GA |
|---|---|---|
| ic_R_acc | +2,26 | [+1,06; +3,48] |
| dis_R_accpiR | +0,55 | [−1,33; +2,45] |
| e_R_pi | +1,72 | [−0,45; +3,90] |
| ic_C1_acc | +1,12 | [−0,20; +2,51] |
| dis_C1_accpiC | +1,36 | [−0,04; +2,81] |
| e_C1_pi | −0,23 | [−2,03; +1,56] |
| ic_R_dengeli | +6,03 | [+2,32; +9,90] |
| ic_C1_dengeli | +5,74 | [+0,45; +11,56] |
| ic_C0_acc | +0,28 | [−1,09; +1,66] |
| ic_C0_dengeli | +4,45 | [−1,79; +11,11] |
| dis_R_accpiC | +0,38 | [−1,03; +1,83] |
- Duyarlılık (seed sabit, yalnız veri bootstrap'ı): B1 +4,95 [−0,54; +10,65] · B2 +1,48 [−1,39; +3,00]
- Beklenti aralıkları (kart, bilgi): {"B1_[-4;+12]": true, "B2_dis_accpiR_[-4;+6]": true, "D_pi_[-1,5;+1,5]": true}

## 4) İkinciller (keşifsel, düzeltmesiz)
| | nokta (puan) | %95 GA | etiket |
|---|---|---|---|
| S1 Δ_dış,C1 dengeli (h10_c1 − h02_c1) | +6,89 | [+1,64; +12,72] | GA alt > 0 |
| S1 Δ_dış,C1 ACC_π_C | +1,36 | [−0,04; +2,81] | ayırt edilemedi |
| S2 D'_π = \|e_R\| − \|e_C0\| (π_C hedefi) | +1,62 | [−1,51; +2,84] | ayırt edilemedi |
| S2 D'_dengeli | +0,58 | [−5,38; +4,28] | ayırt edilemedi |
| S2 e_C0 (π_R hedefi, bilgi) | −0,27 | [−2,46; +1,91] | ayırt edilemedi |
| S3 D_dengeli (bilgilendirici değil) | −0,07 | [−6,26; +6,13] | ayırt edilemedi |
| S4 Δ_dış,R dengeli (H10 − H08c) | −2,96 | [−9,45; +3,24] | ayırt edilemedi |
| S4 D_π (H10 − H08 çifti) | +0,90 | [−1,46; +2,66] | ayırt edilemedi |
| S7_E2_esleme B1 | +4,62 | [−1,46; +11,05] | ayırt edilemedi |
| S7_E2_esleme B2 (D_π) | +1,76 | [−1,32; +3,20] | ayırt edilemedi |
| S8_ham_kume B1 | +4,70 | [−1,48; +11,36] | ayırt edilemedi |
| S9_cerceve B1 | +4,95 | [−1,32; +11,57] | ayırt edilemedi |
| S9_cerceve B2 (D_π) | +1,48 | [−1,51; +3,22] | ayırt edilemedi |
| S10_lezyon_agirlikli B1 | +5,14 | [−1,12; +11,64] | ayırt edilemedi |
| H14 h14_c1-h08_c1 dengeli | +1,45 | [−3,98; +7,44] | ayırt edilemedi |
| H14 h14_c1-h08_c1 ACC_π_C | −0,44 | [−1,83; +0,94] | ayırt edilemedi |
| H14 h14_c1-h10_c1 dengeli | +3,97 | [−0,07; +9,09] | ayırt edilemedi |
| H14 h14_c1-h10_c1 ACC_π_C | +0,34 | [−0,50; +1,17] | ayırt edilemedi |
| T4 (H08c⊕H10) dengeli 0.8159 · ACC_π_R 0.9108 | | | |
| T4 − H10 dengeli | +4,80 | [+0,38; +8,91] | GA alt > 0 |

**S5 kapsama:** e_R_pi +1,72 [−0,45; +3,90] → uyumlu · e_C1_pi −0,23 [−2,03; +1,56] → uyumlu · e_C0_piC −0,10 [−1,94; +1,74] → uyumlu · e_R_dengeli +1,08 [−6,32; +8,27] → uyumlu · e_C1_dengeli −1,15 [−8,28; +5,92] → uyumlu · e_C0_dengeli −0,50 [−8,86; +8,19] → uyumlu

**S6 Kendall τ_b (betimsel):** resmî k = 6 (iç = resmî test) +0.867 · (iç = C0) +1.000 · C1 k = 5 +1.000 · saçılım: `e2_bcn_kuru_sonuc.svg`

**S11 (kol başına):**
| kol | ECE-15 | NLL | tahmin nv payı | mel→nv | nv→mel | seed std dış / iç (dengeli) | seed çifti anlaşmazlığı |
|---|---|---|---|---|---|---|---|
| H10 | 0.0551 | 0.3361 | 0.7886 | 14.0 | 11.4 | 0.0091 / 0.0133 | 0.0499 |
| H02c | 0.0491 | 0.3235 | 0.7901 | 15.2 | 9.6 | 0.0235 / 0.0092 | 0.0564 |
| h10_c1 | 0.0458 | 0.3051 | 0.7918 | 14.6 | 9.4 | 0.0174 / 0.0192 | 0.0475 |
| h02_c1 | 0.0412 | 0.3213 | 0.7923 | 17.6 | 13.0 | 0.0335 / 0.0332 | 0.0704 |
| H08c | 0.0422 | 0.2723 | 0.7858 | 14.2 | 12.0 | 0.0109 / 0.0098 | 0.0500 |
| h08_c1 | 0.0393 | 0.2601 | 0.7871 | 14.4 | 10.6 | 0.0175 / 0.0182 | 0.0563 |
| h14_c1 | 0.0442 | 0.2806 | 0.7860 | 14.4 | 11.6 | 0.0198 / 0.0208 | 0.0501 |
| H11 | 0.0537 | 0.3411 | 0.7893 | 15.0 | 13.2 | 0.0349 / 0.0215 | 0.0663 |
| H01c | 0.0249 | 0.3423 | 0.7993 | 18.0 | 15.8 | 0.0017 / 0.0024 | 0.0038 |
| H12 | 0.0203 | 0.3557 | 0.8018 | 19.0 | 14.4 | 0.0018 / 0.0031 | 0.0023 |
| h01_c1 | 0.0334 | 0.3233 | 0.8099 | 16.8 | 8.8 | 0.0059 / 0.0054 | 0.0087 |

**S12 SCC görüntüleri (n = 11) tahmin dağılımı (seed ort.):**
- H10: akiec 0.527, bcc 0.036, bkl 0.309, df 0.000, mel 0.018, nv 0.109, vasc 0.000
- H02c: akiec 0.582, bcc 0.055, bkl 0.182, df 0.000, mel 0.073, nv 0.109, vasc 0.000
- h10_c1: akiec 0.509, bcc 0.000, bkl 0.309, df 0.000, mel 0.036, nv 0.145, vasc 0.000
- h02_c1: akiec 0.582, bcc 0.018, bkl 0.236, df 0.000, mel 0.127, nv 0.036, vasc 0.000
- H08c: akiec 0.673, bcc 0.000, bkl 0.109, df 0.000, mel 0.145, nv 0.073, vasc 0.000
- h08_c1: akiec 0.655, bcc 0.018, bkl 0.200, df 0.036, mel 0.073, nv 0.018, vasc 0.000
- h14_c1: akiec 0.527, bcc 0.018, bkl 0.327, df 0.000, mel 0.018, nv 0.109, vasc 0.000
- H11: akiec 0.600, bcc 0.018, bkl 0.200, df 0.018, mel 0.036, nv 0.127, vasc 0.000
- H01c: akiec 0.545, bcc 0.182, bkl 0.091, df 0.000, mel 0.000, nv 0.182, vasc 0.000
- H12: akiec 0.545, bcc 0.182, bkl 0.273, df 0.000, mel 0.000, nv 0.000, vasc 0.000
- h01_c1: akiec 0.545, bcc 0.091, bkl 0.145, df 0.000, mel 0.036, nv 0.182, vasc 0.000

## 5) Görülmemiş (İNCELE) → **AÇIK — sayı yazılmaz, cümle seçilmez**
- açık bayraklar: (i) h10: BCN dengeli ≥ iç lezyon-ayrık, (i) h10_c1: BCN dengeli ≥ iç lezyon-ayrık, (i) h14_c1: BCN dengeli ≥ iç lezyon-ayrık, (i) h11: BCN dengeli ≥ iç lezyon-ayrık, (i) h12: BCN dengeli ≥ iç lezyon-ayrık, (ii) h10: BCN dengeli > 0,70, (ii) h02c: BCN dengeli > 0,70, (ii) h10_c1: BCN dengeli > 0,70, (ii) h02_c1: BCN dengeli > 0,70, (ii) h08c: BCN dengeli > 0,70, (ii) h08_c1: BCN dengeli > 0,70, (ii) h14_c1: BCN dengeli > 0,70, (ii) h11: BCN dengeli > 0,70

## 6) Önceden yazılmış cümleler (mekanik seçim)
- **not:** Görülmemiş (İNCELE) bayrağı açık → soruşturma kapanana dek sayı yazılmaz, cümle seçilmez (kart).

### 6b) KURU — cümle doldurma yolu sınaması (İNCELE yok sayılarak; vekil sayılar, makaleye GİRMEZ)
- giris: Resmî bölmede ve lezyon-ayrık C1 bölmesinde eğitilmiş kollar, yeniden eğitilmeden, eşik ya da kalibrasyon ayarı yapılmadan, farklı bir merkezin dermoskopi verisinde (BCN20000, Barselona; 1.213 görüntü, 1.136 lezyon; SCC dışlandı) bir kez değerlendirildi. Dengeli doğruluk iç lezyon-ayrık değerlere göre 0,01–0,80 puan düştü.
- B1_etiketi: ayırt edilemedi
- B1: Dış merkezde DINOv2-B ile ImageNet-ResNet-18 dengeli doğrulukta ayırt edilemedi: Δ +4,95 puan [−1,23; +11,40]. İki iç protokol de yaklaşık +6 puanlık bir üstünlük öngörmüştü. GA, ±6,32 puandan küçük farkları dışlamaz. Sonuç ‘fark yok’ değil, ‘bu dış veriyle belirlenemedi’ diye okunmalıdır.
- B2_etiketi: protokoller ayırt edilemedi
- B2: İki iç protokolün öngörü hataları ayırt edilemedi (e_R +1,72, e_C1 −0,23; D +1,48 puan [−1,55; +3,22]). Bu kıyasın çözünürlüğü iki iç kestirim arasındaki farkla (1,1 puan) sınırlıdır. Bu nedenle sonuç ‘iki protokol eşit iyi öngördü’ diye değil, ‘bu veriyle hangisinin daha iyi öngördüğü belirlenemedi’ diye okunmalıdır.
- kapsama_S5: Dış fark, resmî bölme kestiriminin %95 aralığıyla uyumlu (e_R +1,72 [−0,45; +3,90]) ve lezyon-ayrık kestirimin aralığıyla uyumlu (e_C1 −0,23 [−2,03; +1,56]) çıktı (keşifsel, düzeltmesiz).
- sinirliliklar: Dış küme BCN20000'in yalnız etiketli eğitim bölümüdür ve bu görüntüler modellerimizin eğitiminde kullanılmadı; ISIC test etiketleri kullanılmadı; SCC dışlandı (duyarlılıkta akiec'e eşlendi); hasta düzeyinde ayrıklık ölçülemedi, merkezler coğrafi olarak ayrık; görüntü düzeyi metrik lezyon-kümeli bootstrap ile verildi; rakip 28 px kolları ağırlık dosyası olmadığı için değerlendirilemedi; klinik bir iddia taşınmaz.
