# H16 / E3 — ön-işleme hattı denetimi (KURU VEKİL — SONUÇ DEĞİL)
*2026-11-08T15:44:54* · `scripts/e3_onisleme_analiz.py` (SHA256 `fdd35b9f43b15ef0…`) · kart SHA256 `f3c6998196e06248…` (onaylı) · keşifsel, sonradan · bootstrap B = 10000, default_rng(0)

**K-d kuru koşu:** GEÇTİ · kaynak arşiv tahmin_val.npy (runs_colab; yerel kol vekili)

| kol | yerel dengeli (bu) | v3 orijinal | fark (puan) |
|---|---|---|---|
| h10 | 79.770 | 79.770 | +0.0000 |
| h02c | 70.999 | 70.999 | +0.0000 |
| h08c | 77.627 | 77.627 | +0.0000 |
| h11 | 75.760 | 75.760 | +0.0000 |

Δ_yerel(H10 − H02c) = +8.772 (v3 +8.772; tol ±0.05) · öz-sınama hat := yerel → Δ_hat +0.00e+00, GA [0.0, 0.0] · eşikler m = 2.4813 (sabit 2,48), G = 10.3053 (sabit 10,31) → tuttu

## Birincil: Δ_hat = +1,46 puan [−0,49; +3,87] (p 0,144) → **KÜÇÜK-BELİRSİZ**
n = 1001 görüntü, 969 lezyon kümesi · Δ_deng(H10 − H02c): yerel +8,78 [+3,68; +14,10], BCN hattı +10,24 [+5,15; +15,37]

## S1–S2 (kol başına; keşifsel, düzeltmesiz)

| kol | D_k (puan) [GA] | dengeli yerel → hat | ACC Δ | nv payı Δ | uyum ort (min) | maks\|Δp\| | dönme |
|---|---|---|---|---|---|---|---|
| h10 | −0,30 [−0,81; +0,00] | 79.73 → 79.43 | −0,08 | +0.0022 | 0.9950 (0.9830) | 0.267 | 5.0 |
| h02c | −1,76 [−4,29; +0,25] | 70.95 → 69.19 | −0,48 | -0.0034 | 0.9670 (0.9391) | 0.831 | 33.0 |
| h08c | −0,00 [−1,56; +1,74] | 77.58 → 77.58 | −0,32 | -0.0048 | 0.9800 (0.9680) | 0.672 | 20.0 |
| h11 | −0,12 [−0,48; +0,22] | 75.72 → 75.60 | −0,14 | +0.0000 | 0.9964 (0.9920) | 0.400 | 3.6 |

## S3 / S4 / S5–S7 / S8

- S3 h10-h08c: Δ_hat −0,30 [−2,05; +1,18]
- S3 h11-h02c: Δ_hat +1,64 [−0,42; +4,20]
- S4 N (654 görüntü, 638 lezyon; boş sınıf []): Δ_hat +1,56 [−0,69; +4,15] → KÜÇÜK-BELİRSİZ
- S4 S (347 görüntü, 331 lezyon; boş sınıf []): Δ_hat +1,16 [−1,46; +4,55] → KÜÇÜK-BELİRSİZ
- jpeg: Δ +1,87 [−0,10; +4,18] (aralık etiketi KÜÇÜK-BELİRSİZ) · D_h10 −0,28, D_h02c −2,15, D_h08c −1,75, D_h11 −0,50
- z125: Δ +1,46 [−0,49; +3,87] (aralık etiketi KÜÇÜK-BELİRSİZ) · D_h10 −0,30, D_h02c −1,76, D_h08c −0,00, D_h11 −0,12
- z080: Δ +1,87 [−0,10; +4,18] (aralık etiketi KÜÇÜK-BELİRSİZ) · D_h10 −0,28, D_h02c −2,15, D_h08c −1,75, D_h11 −0,50
- S8 seed sabit: +1,46 [+0,28; +2,69]
- S10: R18 dönme val 2,65% / kalibrasyon 3/160 (1,88%) → oran 1.41

## İNCELE: kapalı · {'i_delta_hat_gt_8': False, 'ii_delta_hat_lt_-6': False, 'iii_vit_D_lt_-5': False, 'iv_r18_uyum_lt_0_90': False, 'INCELE': False}

## Beklenti içinde mi (betimsel)

| nicelik | ölçülen | aralık | içinde |
|---|---|---|---|
| RMSE medyanı (hat ↔ yerel) | None | [1.3, 2.5] | None |
| argmax uyumu h02c | 0.967 | [0.94, 0.995] | True |
| maks|Δp| h02c | 0.8309 | [0.4, 0.99] | True |
| D_k h02c | -1.7589 | [-4.0, 1.0] | True |
| nv payı değişimi h02c | -0.0034 | [-0.01, 0.03] | True |
| argmax uyumu h08c | 0.98 | [0.94, 0.995] | True |
| maks|Δp| h08c | 0.6721 | [0.4, 0.99] | True |
| D_k h08c | -0.0017 | [-4.0, 1.0] | True |
| nv payı değişimi h08c | -0.0048 | [-0.01, 0.03] | True |
| argmax uyumu h10 | 0.995 | [0.985, 1.0] | True |
| maks|Δp| h10 | 0.2668 | [0.03, 0.5] | True |
| D_k h10 | -0.2994 | [-1.5, 0.5] | True |
| nv payı değişimi h10 | 0.0022 | [-0.01, 0.01] | True |
| argmax uyumu h11 | 0.9964 | [0.98, 1.0] | True |
| maks|Δp| h11 | 0.4003 | [0.03, 0.5] | True |
| D_k h11 | -0.1167 | [-2.0, 0.5] | True |
| nv payı değişimi h11 | 0.0 | [-0.01, 0.01] | True |
| Δ_hat (birincil) | 1.4595 | [-1.5, 4.0] | True |
| Δ_hat GA yarı genişliği | 2.179 | [1.0, 2.5] | True |
| S4 Δ_hat (N) | 1.5603 | [-2.0, 5.0] | True |
| S5 Δ (JPEG) | 1.8693 | [-1.5, 4.0] | True |
| S6/S7 Δ (z125) | 1.4595 | [-3.0, 8.0] | True |
| S6/S7 Δ (z080) | 1.8693 | [-3.0, 8.0] | True |
| S10 dönme oranı (görülmemiş / eğitim) | 1.4119 | [1.0, 4.0] | True |

## Önceden yazılmış cümleler (mekanik)

- **giris:** Dış farkın bir bölümünün BCN20000 görüntülerini hazırlarken kullandığımız ön-işleme hattından gelip gelmediğini sınamak için, resmî doğrulama kümesinin 1.001 görüntüsünü HAM10000 özgün dosyalarından aynı kodla yeniden ürettik. Aynı kolları, yeniden eğitmeden, bu girdide ve MedMNIST+ girdisinde karşılaştırdık. Analiz hakem önerisiyle, E2 sonuçları görüldükten sonra tasarlandı; test kümesi ve BCN etiketleri açılmadı. Piksel farkı küçüktü (görüntü başına RMSE medyanı — gri düzey).
- **etiket:** KÜÇÜK-BELİRSİZ
- **birincil:** Ön-işleme hattının farka etkisi nokta kestirimde küçüktü: +1,46 puan [−0,49; +3,87]. Ancak aralık ±2,48 puanlık eşdeğerlik bandını aştığından, hattın dış farkın birkaç puanını açıklaması bu veriyle dışlanamadı. Dış farkın büyük bölümü hat etkisinin aralığının dışında kalır.
- **S1_S2:** ResNet-18 kollarının tahminleri görüntülerin %2,6'ünde değişti (DINOv2-B'de %0,5); dengeli doğruluk değişimleri H02c −1,76, H10 −0,30 puandı.
- **S10:** Bu oran, daha önce yalnız eğitim görüntülerinde yapılan sınamadakinin (%1,9) 1,4 katıdır.
- **S4_N_etiketi:** KÜÇÜK-BELİRSİZ
- **sinirlilik:** Ön-işleme duyarlılığı doğrulama kümesinde ölçüldü. Bu küme epoch seçiminde kullanıldığı için tamamen bağımsız değildir. Ölçülen, yazılım hattının etkisidir; dermatoskop, büyütme ve görüş alanı gibi edinim farkları merkez kaymasının parçası sayıldı ve burada ayrıştırılmadı.
