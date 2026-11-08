# H16 / E3 — ön-işleme hattı denetimi
*2026-11-08T16:27:15* · `scripts/e3_onisleme_analiz.py` (SHA256 `fdd35b9f43b15ef0…`) · kart SHA256 `f3c6998196e06248…` (onaylı) · keşifsel, sonradan · bootstrap B = 10000, default_rng(0)

**K-d kuru koşu:** GEÇTİ · kaynak a3 yerel kol (tahmin_val_yerel.npy)

| kol | yerel dengeli (bu) | v3 orijinal | fark (puan) |
|---|---|---|---|
| h10 | 79.770 | 79.770 | +0.0000 |
| h02c | 70.999 | 70.999 | +0.0000 |
| h08c | 77.627 | 77.627 | +0.0000 |
| h11 | 75.760 | 75.760 | +0.0000 |

Δ_yerel(H10 − H02c) = +8.772 (v3 +8.772; tol ±0.05) · öz-sınama hat := yerel → Δ_hat +0.00e+00, GA [0.0, 0.0] · eşikler m = 2.4813 (sabit 2,48), G = 10.3053 (sabit 10,31) → tuttu

## Birincil: Δ_hat = +1,64 puan [−1,90; +4,89] (p 0,332) → **KÜÇÜK-BELİRSİZ**
n = 1003 görüntü, 970 lezyon kümesi · Δ_deng(H10 − H02c): yerel +8,77 [+3,69; +14,07], BCN hattı +10,41 [+5,59; +15,28]

## S1–S2 (kol başına; keşifsel, düzeltmesiz)

| kol | D_k (puan) [GA] | dengeli yerel → hat | ACC Δ | nv payı Δ | uyum ort (min) | maks\|Δp\| | dönme |
|---|---|---|---|---|---|---|---|
| h10 | −1,04 [−3,32; +0,93] | 79.77 → 78.73 | −0,76 | -0.0038 | 0.9663 (0.9641) | 0.817 | 33.8 |
| h02c | −2,68 [−4,98; −0,39] | 71.00 → 68.32 | −0,64 | +0.0142 | 0.9529 (0.9352) | 0.837 | 47.2 |
| h08c | −2,95 [−6,01; −0,63] | 77.63 → 74.68 | −1,14 | +0.0136 | 0.9545 (0.9432) | 0.894 | 45.6 |
| h11 | −0,77 [−1,93; +0,61] | 75.76 → 74.99 | −0,54 | +0.0098 | 0.9753 (0.9681) | 0.703 | 24.8 |

## S3 / S4 / S5–S7 / S8

- S3 h10-h08c: Δ_hat +1,90 [−1,28; +5,15]
- S3 h11-h02c: Δ_hat +1,91 [−0,63; +4,39]
- S4 N (655 görüntü, 638 lezyon; boş sınıf []): Δ_hat +1,05 [−2,56; +4,13] → KÜÇÜK-BELİRSİZ
- S4 S (348 görüntü, 332 lezyon; boş sınıf []): Δ_hat +2,51 [−4,16; +8,93] → KISMEN
- jpeg: Δ +1,63 [−2,20; +4,97] (aralık etiketi KÜÇÜK-BELİRSİZ) · D_h10 −1,05, D_h02c −2,68, D_h08c −3,65, D_h11 −0,95
- z125: Δ +7,44 [+2,91; +12,34] (aralık etiketi KISMEN) · D_h10 +0,83, D_h02c −6,61, D_h08c −6,23, D_h11 −1,71
- z080: Δ −13,61 [−20,06; −7,44] (aralık etiketi TERS) · D_h10 −20,36, D_h02c −6,75, D_h08c −4,76, D_h11 −4,64
- S8 seed sabit: +1,64 [−0,58; +3,81]
- S10: R18 dönme val 4,63% / kalibrasyon 3/160 (1,88%) → oran 2.47

## İNCELE: kapalı · {'i_delta_hat_gt_8': False, 'ii_delta_hat_lt_-6': False, 'iii_vit_D_lt_-5': False, 'iv_r18_uyum_lt_0_90': False, 'INCELE': False}

## Beklenti içinde mi (betimsel)

| nicelik | ölçülen | aralık | içinde |
|---|---|---|---|
| RMSE medyanı (hat ↔ yerel) | 1.6954 | [1.3, 2.5] | True |
| argmax uyumu h02c | 0.9529 | [0.94, 0.995] | True |
| maks|Δp| h02c | 0.8367 | [0.4, 0.99] | True |
| D_k h02c | -2.6822 | [-4.0, 1.0] | True |
| nv payı değişimi h02c | 0.0142 | [-0.01, 0.03] | True |
| argmax uyumu h08c | 0.9545 | [0.94, 0.995] | True |
| maks|Δp| h08c | 0.8942 | [0.4, 0.99] | True |
| D_k h08c | -2.9466 | [-4.0, 1.0] | True |
| nv payı değişimi h08c | 0.0136 | [-0.01, 0.03] | True |
| argmax uyumu h10 | 0.9663 | [0.985, 1.0] | False |
| maks|Δp| h10 | 0.8174 | [0.03, 0.5] | False |
| D_k h10 | -1.0449 | [-1.5, 0.5] | True |
| nv payı değişimi h10 | -0.0038 | [-0.01, 0.01] | True |
| argmax uyumu h11 | 0.9753 | [0.98, 1.0] | False |
| maks|Δp| h11 | 0.7027 | [0.03, 0.5] | False |
| D_k h11 | -0.7682 | [-2.0, 0.5] | True |
| nv payı değişimi h11 | 0.0098 | [-0.01, 0.01] | True |
| Δ_hat (birincil) | 1.6373 | [-1.5, 4.0] | True |
| Δ_hat GA yarı genişliği | 3.393 | [1.0, 2.5] | False |
| S4 Δ_hat (N) | 1.0517 | [-2.0, 5.0] | True |
| S5 Δ (JPEG) | 1.6333 | [-1.5, 4.0] | True |
| S6/S7 Δ (z125) | 7.4401 | [-3.0, 8.0] | True |
| S6/S7 Δ (z080) | -13.6091 | [-3.0, 8.0] | False |
| S10 dönme oranı (görülmemiş / eğitim) | 2.4673 | [1.0, 4.0] | True |

## Önceden yazılmış cümleler (mekanik)

- **giris:** Dış farkın bir bölümünün BCN20000 görüntülerini hazırlarken kullandığımız ön-işleme hattından gelip gelmediğini sınamak için, resmî doğrulama kümesinin 1.003 görüntüsünü HAM10000 özgün dosyalarından aynı kodla yeniden ürettik. Aynı kolları, yeniden eğitmeden, bu girdide ve MedMNIST+ girdisinde karşılaştırdık. Analiz hakem önerisiyle, E2 sonuçları görüldükten sonra tasarlandı; test kümesi ve BCN etiketleri açılmadı. Piksel farkı küçüktü (görüntü başına RMSE medyanı 1,70 gri düzey).
- **etiket:** KÜÇÜK-BELİRSİZ
- **birincil:** Ön-işleme hattının farka etkisi nokta kestirimde küçüktü: +1,64 puan [−1,90; +4,89]. Ancak aralık ±2,48 puanlık eşdeğerlik bandını aştığından, hattın dış farkın birkaç puanını açıklaması bu veriyle dışlanamadı. Dış farkın büyük bölümü hat etkisinin aralığının dışında kalır.
- **S1_S2:** ResNet-18 kollarının tahminleri görüntülerin %4,6'ünde değişti (DINOv2-B'de %3,4); dengeli doğruluk değişimleri H02c −2,68, H10 −1,04 puandı.
- **S10:** Bu oran, daha önce yalnız eğitim görüntülerinde yapılan sınamadakinin (%1,9) 2,5 katıdır.
- **S4_N_etiketi:** KÜÇÜK-BELİRSİZ
- **duyarlilik_z125:** ×1,25 ölçek (yakınlaştırma) tek başına Δ_hat'i +7,44 puana çıkarıyor; keşifsel.
- **sinirlilik:** Ön-işleme duyarlılığı doğrulama kümesinde ölçüldü. Bu küme epoch seçiminde kullanıldığı için tamamen bağımsız değildir. Ölçülen, yazılım hattının etkisidir; dermatoskop, büyütme ve görüş alanı gibi edinim farkları merkez kaymasının parçası sayıldı ve burada ayrıştırılmadı.
