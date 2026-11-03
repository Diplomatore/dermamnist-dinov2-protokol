# Türetilmiş değerler (otomatik — `evidence/scripts/build_derivations.py`)

*Kaynak: `evidence/evidence.json`. Formüldeki her terim bir kanıt id'sidir. "(n)" = GA'sız nokta farkı; bu değerler için anlamlılık ifadesi kullanılmaz. Puan = ×100.*

## 1. Nokta farkları (resmî test, (n))

| Kod | İfade | Formül | Değer | Kaynak id | Not |
|---|---|---|---|---|---|
| D01 | H10 − H01c test ACC | `100·(T-h10-acc − T-h01-acc)` | **+6,79 puan (n)** | T-h10-acc, T-h01-acc | tam değerden; karar.md §4 yuvarlanmış ACC'lerden hesapladığı için ±0,01 fark edebilir (ablasyon.csv ile aynı) |
| D02 | H02c − H01c test ACC | `100·(T-h02-acc − T-h01-acc)` | **+4,53 puan (n)** | T-h02-acc, T-h01-acc | tam değerden; karar.md §4 yuvarlanmış ACC'lerden hesapladığı için ±0,01 fark edebilir (ablasyon.csv ile aynı) |
| D03 | H11 − H12 test ACC | `100·(T-h11-acc − T-h12-acc)` | **+6,77 puan (n)** | T-h11-acc, T-h12-acc | tam değerden; karar.md §4 yuvarlanmış ACC'lerden hesapladığı için ±0,01 fark edebilir (ablasyon.csv ile aynı) |

## 2. Val → test kayması (karar.md §1.2)

| Kod | İfade | Formül | Değer | Kaynak id | Not |
|---|---|---|---|---|---|
| D04.1 | B test − val ACC | `100·(T-baseline-acc − V-baseline-acc)` | **−0,63 puan (n)** | T-baseline-acc, V-baseline-acc |  |
| D04.2 | H01c test − val ACC | `100·(T-h01-acc − V-h01-acc)` | **+0,37 puan (n)** | T-h01-acc, V-h01-acc |  |
| D04.3 | H02c test − val ACC | `100·(T-h02-acc − V-h02-acc)` | **+0,57 puan (n)** | T-h02-acc, V-h02-acc |  |
| D04.4 | H08c test − val ACC | `100·(T-h08-acc − V-h08-acc)` | **+1,10 puan (n)** | T-h08-acc, V-h08-acc |  |
| D04.5 | H10 test − val ACC | `100·(T-h10-acc − V-h10-acc)` | **−0,08 puan (n)** | T-h10-acc, V-h10-acc | W = H10; karar.md §1.2 yuvarlanmış değerlerden −0,07 yazar — makalede tam değerden türetilen bu değer ya da "≈ −0,1" kullanılmalı |
| D04.6 | H11 test − val ACC | `100·(T-h11-acc − V-h11-acc)` | **+0,37 puan (n)** | T-h11-acc, V-h11-acc |  |
| D04.7 | H12 test − val ACC | `100·(T-h12-acc − V-h12-acc)` | **−0,28 puan (n)** | T-h12-acc, V-h12-acc |  |
| D04.8 | T4 test − val ACC | `100·(T-t4-acc − V-t4-acc)` | **+0,74 puan (n)** | T-t4-acc, V-t4-acc |  |

## 3. Sınıf duyarlılığı farkları (resmî test; [T] `kollar.*.ort.duyarlilik_*`; GA yok)

| Kod | İfade | Formül | Değer | Kaynak id | Not |
|---|---|---|---|---|---|
| D05.h08h02.akiec | H08c − H02c duyarlılık akiec | `100·(T-h08-duyarlilik_akiec − T-h02-duyarlilik_akiec)` | **+8,48 puan (n)** | T-h08-duyarlilik_akiec, T-h02-duyarlilik_akiec |  |
| D05.h08h02.bcc | H08c − H02c duyarlılık bcc | `100·(T-h08-duyarlilik_bcc − T-h02-duyarlilik_bcc)` | **+3,88 puan (n)** | T-h08-duyarlilik_bcc, T-h02-duyarlilik_bcc |  |
| D05.h08h02.bkl | H08c − H02c duyarlılık bkl | `100·(T-h08-duyarlilik_bkl − T-h02-duyarlilik_bkl)` | **+5,27 puan (n)** | T-h08-duyarlilik_bkl, T-h02-duyarlilik_bkl |  |
| D05.h08h02.df | H08c − H02c duyarlılık df | `100·(T-h08-duyarlilik_df − T-h02-duyarlilik_df)` | **+21,74 puan (n)** | T-h08-duyarlilik_df, T-h02-duyarlilik_df |  |
| D05.h08h02.mel | H08c − H02c duyarlılık mel | `100·(T-h08-duyarlilik_mel − T-h02-duyarlilik_mel)` | **+8,07 puan (n)** | T-h08-duyarlilik_mel, T-h02-duyarlilik_mel |  |
| D05.h08h02.nv | H08c − H02c duyarlılık nv | `100·(T-h08-duyarlilik_nv − T-h02-duyarlilik_nv)` | **+0,39 puan (n)** | T-h08-duyarlilik_nv, T-h02-duyarlilik_nv |  |
| D05.h08h02.vasc | H08c − H02c duyarlılık vasc | `100·(T-h08-duyarlilik_vasc − T-h02-duyarlilik_vasc)` | **+11,03 puan (n)** | T-h08-duyarlilik_vasc, T-h02-duyarlilik_vasc |  |
| D05.h08h02.sağlama | Σ Δduyarlılık / 7 = Δdengeli (sağlama) | `Σ_s D05.s / 7` | **+8,41 (Δdengeli +8,41)** | T-h08-dengeli_acc, T-h02-dengeli_acc | eşit olmalı |
| D06 | H08c − H02c dengeli Δ'sına df + vasc katkısı | `(D05.h08h02.df + D05.h08h02.vasc)/7` | **+4,68 puan** | T-h08-duyarlilik_df, T-h02-duyarlilik_df, T-h08-duyarlilik_vasc, T-h02-duyarlilik_vasc |  |
| D07 | df + vasc payı (dengeli Δ içinde) | `D06 / Δdengeli(H08c−H02c)` | **%56** | T-h08-dengeli_acc, T-h02-dengeli_acc | brief madde 2: %56 |
| D08 | seed başına ek doğru görüntü (df, vasc) | `Δdf·n_df/100 , Δvasc·n_vasc/100` | **5,0 + 3,2 ≈ 8** | DM-test-df, DM-test-vasc | brief madde 2: ≈ 8 |
| D05.h10h02.akiec | H10 − H02c duyarlılık akiec | `100·(T-h10-duyarlilik_akiec − T-h02-duyarlilik_akiec)` | **−1,21 puan (n)** | T-h10-duyarlilik_akiec, T-h02-duyarlilik_akiec |  |
| D05.h10h02.bcc | H10 − H02c duyarlılık bcc | `100·(T-h10-duyarlilik_bcc − T-h02-duyarlilik_bcc)` | **+0,78 puan (n)** | T-h10-duyarlilik_bcc, T-h02-duyarlilik_bcc |  |
| D05.h10h02.bkl | H10 − H02c duyarlılık bkl | `100·(T-h10-duyarlilik_bkl − T-h02-duyarlilik_bkl)` | **+2,00 puan (n)** | T-h10-duyarlilik_bkl, T-h02-duyarlilik_bkl |  |
| D05.h10h02.df | H10 − H02c duyarlılık df | `100·(T-h10-duyarlilik_df − T-h02-duyarlilik_df)` | **+19,13 puan (n)** | T-h10-duyarlilik_df, T-h02-duyarlilik_df |  |
| D05.h10h02.mel | H10 − H02c duyarlılık mel | `100·(T-h10-duyarlilik_mel − T-h02-duyarlilik_mel)` | **+9,60 puan (n)** | T-h10-duyarlilik_mel, T-h02-duyarlilik_mel |  |
| D05.h10h02.nv | H10 − H02c duyarlılık nv | `100·(T-h10-duyarlilik_nv − T-h02-duyarlilik_nv)` | **+0,89 puan (n)** | T-h10-duyarlilik_nv, T-h02-duyarlilik_nv |  |
| D05.h10h02.vasc | H10 − H02c duyarlılık vasc | `100·(T-h10-duyarlilik_vasc − T-h02-duyarlilik_vasc)` | **+11,03 puan (n)** | T-h10-duyarlilik_vasc, T-h02-duyarlilik_vasc |  |
| D05.h10h02.sağlama | Σ Δduyarlılık / 7 = Δdengeli (sağlama) | `Σ_s D05.s / 7` | **+6,03 (Δdengeli +6,03)** | T-h10-dengeli_acc, T-h02-dengeli_acc | eşit olmalı |
| D05.h10h08.akiec | H10 − H08c duyarlılık akiec | `100·(T-h10-duyarlilik_akiec − T-h08-duyarlilik_akiec)` | **−9,70 puan (n)** | T-h10-duyarlilik_akiec, T-h08-duyarlilik_akiec |  |
| D05.h10h08.bcc | H10 − H08c duyarlılık bcc | `100·(T-h10-duyarlilik_bcc − T-h08-duyarlilik_bcc)` | **−3,11 puan (n)** | T-h10-duyarlilik_bcc, T-h08-duyarlilik_bcc |  |
| D05.h10h08.bkl | H10 − H08c duyarlılık bkl | `100·(T-h10-duyarlilik_bkl − T-h08-duyarlilik_bkl)` | **−3,27 puan (n)** | T-h10-duyarlilik_bkl, T-h08-duyarlilik_bkl |  |
| D05.h10h08.df | H10 − H08c duyarlılık df | `100·(T-h10-duyarlilik_df − T-h08-duyarlilik_df)` | **−2,61 puan (n)** | T-h10-duyarlilik_df, T-h08-duyarlilik_df |  |
| D05.h10h08.mel | H10 − H08c duyarlılık mel | `100·(T-h10-duyarlilik_mel − T-h08-duyarlilik_mel)` | **+1,52 puan (n)** | T-h10-duyarlilik_mel, T-h08-duyarlilik_mel |  |
| D05.h10h08.nv | H10 − H08c duyarlılık nv | `100·(T-h10-duyarlilik_nv − T-h08-duyarlilik_nv)` | **+0,51 puan (n)** | T-h10-duyarlilik_nv, T-h08-duyarlilik_nv |  |
| D05.h10h08.vasc | H10 − H08c duyarlılık vasc | `100·(T-h10-duyarlilik_vasc − T-h08-duyarlilik_vasc)` | **+0,00 puan (n)** | T-h10-duyarlilik_vasc, T-h08-duyarlilik_vasc |  |
| D05.h10h08.sağlama | Σ Δduyarlılık / 7 = Δdengeli (sağlama) | `Σ_s D05.s / 7` | **−2,38 (Δdengeli −2,38)** | T-h10-dengeli_acc, T-h08-dengeli_acc | eşit olmalı |

## 4. Çoğunluk tabanına göre resmî test ACC (C tablolarındaki ACC − 0,7726 karşılığı)

| Kod | İfade | Formül | Değer | Kaynak id | Not |
|---|---|---|---|---|---|
| D09.0 | resmî test çoğunluk tabanı (tam) | `DM-test-nv / T-meta-n` | **0,6688** | DM-test-nv, T-meta-n | veri manifesti 0,669 ile aynı |
| D09.1 | B resmî test ACC − taban | `100·(T-baseline-acc − D09.0)` | **+7,06 puan** | T-baseline-acc, D09.0 |  |
| D09.2 | H01c resmî test ACC − taban | `100·(T-h01-acc − D09.0)` | **+15,72 puan** | T-h01-acc, D09.0 |  |
| D09.3 | H02c resmî test ACC − taban | `100·(T-h02-acc − D09.0)` | **+20,25 puan** | T-h02-acc, D09.0 |  |
| D09.4 | H08c resmî test ACC − taban | `100·(T-h08-acc − D09.0)` | **+22,87 puan** | T-h08-acc, D09.0 |  |
| D09.5 | H10 resmî test ACC − taban | `100·(T-h10-acc − D09.0)` | **+22,51 puan** | T-h10-acc, D09.0 |  |
| D09.6 | H11 resmî test ACC − taban | `100·(T-h11-acc − D09.0)` | **+21,49 puan** | T-h11-acc, D09.0 |  |
| D09.7 | H12 resmî test ACC − taban | `100·(T-h12-acc − D09.0)` | **+14,71 puan** | T-h12-acc, D09.0 |  |

## 5. Bütçe

| Kod | İfade | Formül | Değer | Kaynak id | Not |
|---|---|---|---|---|---|
| D10 | Tur-4 bütçesi (tek kalem verilmemiş) | `BUT-tur4 − BUT-tur3` | **6,8 sa** | BUT-tur4, BUT-tur3 | yaklaşık; Tur-4 satırı oturum tabanı, Tur-3 GPU-sa — karışık taban, yalnız kaba sıra |

## 6. Literatür çıpası (bağlam; aynı koşulda üretilmedi — review 5.1, 2.1)

| Kod | İfade | Formül | Değer | Kaynak id | Not |
|---|---|---|---|---|---|
| D11.1 | H10 resmî test ACC − Wu & Papież uçtan uca DINOv2-B ACC | `100·T-h10-acc − LIT-wu2025-e2e-acc` | **−2,62 puan** | T-h10-acc, LIT-wu2025-e2e-acc | hakemsiz ön baskı; farklı koşul (bölme/seçim/augmentasyon farklı ya da bildirilmemiş); sayısal yöntem kıyası değil, yalnız konumlandırma |
| D11.2 | H01c resmî test ACC − Wu & Papież doğrusal sonda ACC | `100·T-h01-acc − LIT-wu2025-lp-acc` | **−1,37 puan** | T-h01-acc, LIT-wu2025-lp-acc | farklı koşul (bölme/seçim/augmentasyon farklı ya da bildirilmemiş); sayısal yöntem kıyası değil, yalnız konumlandırma |
| D11.3 | H02c resmî test ACC − MedMNIST+ ImageNet R18 E2E 224 px ACC | `100·T-h02-acc − LIT-doerrich2025-e2e224-acc` | **+4,80 puan** | T-h02-acc, LIT-doerrich2025-e2e224-acc | farklı koşul (bölme/seçim/augmentasyon farklı ya da bildirilmemiş); sayısal yöntem kıyası değil, yalnız konumlandırma |
| D11.4 | Zhang & Li Vanilla R18-224 ACC − H02c resmî test ACC | `100·(LIT-zhang2026-acc − T-h02-acc)` | **+1,07 puan** | LIT-zhang2026-acc, T-h02-acc | tur-1'de doğrulandı (PMC13413053 Tablo 1(d)); tarif farkı: wd 0,05, ısınma, batch 128, etiket yumuşatma 0,1 — farklı koşul (bölme/seçim/augmentasyon farklı ya da bildirilmemiş); sayısal yöntem kıyası değil, yalnız konumlandırma |

## 7. Çoğunluk tabanı farkı (ek satırlar; review 7.5)

| Kod | İfade | Formül | Değer | Kaynak id | Not |
|---|---|---|---|---|---|
| D12 | T4 C0 ACC − C-test çoğunluk tabanı | `100·(T-C0-t4-acc − T-C0-cogunluk)` | **+16,09 puan** | T-C0-t4-acc, T-C0-cogunluk | betimsel ek satır |

## 8. Eğitim kaybı oranları (val dönemi, K4 yerel koşumları; review B4)

| Kod | İfade | Formül | Değer | Kaynak id | Not |
|---|---|---|---|---|---|
| D13.1 | epoch 5 eğitim kaybı oranı, D4 (H08) / H02 | `EGK-H08-ep5 / EGK-H02-ep5 (seed-eşli: s_i/s_i)` | **18,5× (5 seed ortalaması); s0 21,2×; seed-eşli aralık 16,0–22,3×** | EGK-H08-ep5, EGK-H02-ep5 | tahlil-tur3 '≈21×' / '≈2,5×' değerleri s0 oranıdır (EGK-tahlil-satiri) |
| D13.2 | epoch 5 eğitim kaybı oranı, renk oynaması (H09) / H02 | `EGK-H09-ep5 / EGK-H02-ep5 (seed-eşli: s_i/s_i)` | **2,3× (5 seed ortalaması); s0 2,5×; seed-eşli aralık 2,0–2,7×** | EGK-H09-ep5, EGK-H02-ep5 | tahlil-tur3 '≈21×' / '≈2,5×' değerleri s0 oranıdır (EGK-tahlil-satiri) |
| D13.3 | epoch 30 eğitim kaybı oranı, D4 (H08) / H02 | `EGK-H08-ep30 / EGK-H02-ep30` | **20,5×** | EGK-H08-ep30, EGK-H02-ep30 |  |
| D13.4 | epoch 30 eğitim kaybı oranı, renk (H09) / H02 | `EGK-H09-ep30 / EGK-H02-ep30` | **1,3×** | EGK-H09-ep30, EGK-H02-ep30 |  |

## 9. Test içi lezyon çokluğu (yalnız meta veri; review 9.1)

| Kod | İfade | Formül | Değer | Kaynak id | Not |
|---|---|---|---|---|---|
| D14.1 | resmî test: birden çok görüntülü lezyonlara ait görüntü payı | `LES-test-coklu-goruntu / T-meta-n` | **%13,3** | LES-test-coklu-goruntu, T-meta-n |  |
| D14.2 | S alt kümesi: birden çok görüntülü lezyonlara ait görüntü payı | `LES-S-coklu-goruntu / T-meta-nS` | **%14,9** | LES-S-coklu-goruntu, T-meta-nS |  |
| D14.3 | N alt kümesi: birden çok görüntülü lezyonlara ait görüntü payı | `LES-N-coklu-goruntu / T-meta-nN` | **%12,5** | LES-N-coklu-goruntu, T-meta-nN |  |
| D14.4 | C-test: birden çok görüntülü lezyonlara ait görüntü payı | `LES-Ctest-coklu-goruntu / T-C0-n` | **%12,2** | LES-Ctest-coklu-goruntu, T-C0-n |  |

## 10. Bootstrap p çözünürlüğü (review 7.6)

| Kod | İfade | Formül | Değer | Kaynak id | Not |
|---|---|---|---|---|---|
| D15.1 | en küçük bootstrap p | `1 / (ON-boot-B + 1)` | **0,000500** | ON-boot-B |  |
| D15.2 | Holm sonrası alt sınır (4 karşılaştırmalı Aile 1; 2 karşılaştırmalı Aile 2) | `4/(B+1) ; 2/(B+1)` | **0,0020 ; 0,0010** | ON-boot-B | tablolardaki 0,002 ve 0,001 |

## 11. Parametre oranları (review 4.8/4.17)

| Kod | İfade | Formül | Değer | Kaynak id | Not |
|---|---|---|---|---|---|
| D16.1 | H10 eğitilen / toplam parametre | `PRM-H10-egitilebilir / PRM-H10-toplam` | **%33,1** | PRM-H10-egitilebilir, PRM-H10-toplam |  |
| D16.2 | H10 toplam / H02c toplam parametre | `PRM-H10-toplam / PRM-H02` | **7,7×** | PRM-H10-toplam, PRM-H02 |  |
