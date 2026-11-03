# K5 analiz — kip `kuru-val` (2026-10-11T18:54:46)

n = 1003 (S 348 / N 655) · tabaka 19 · betik SHA256 `13a71cf1375fc98a…` · ön-kayıt `ac8215ead0d6e47c…`

> **KURU KOŞU (P5):** 'test' = resmî VAL, 'C-test' = C-val ∩ val. Test dosyası açılmadı. C1/C2 satırları vekildir.

## Kollar (ort ± std, 5 seed)
| kol | ACC | AUC | dengeli | dengeli-5 | makro-F1 | ECE | NLL | H04 τ* → ACC / dengeli | R-T ECE |
|---|---|---|---|---|---|---|---|---|---|
| baseline_c | 0.7458 ± 0.0074 | 0.9223 | 0.4333 | 0.5237 | 0.4444 | 0.0399 | 0.6892 | τ*=0.2 → 0.7370 / 0.4634 | 0.0332 |
| H01c | 0.8223 ± 0.0013 | 0.9563 | 0.6494 | 0.6768 | 0.6777 | 0.0309 | 0.5179 | τ*=0.2 → 0.8136 / 0.6639 | 0.0192 |
| H02c | 0.8656 ± 0.0019 | 0.9697 | 0.7100 | 0.7549 | 0.7463 | 0.0762 | 0.5188 | τ*=0.7 → 0.8558 / 0.7498 | 0.0328 |
| H08c | 0.8865 ± 0.0033 | 0.9782 | 0.7763 | 0.7939 | 0.7947 | 0.0654 | 0.4212 | τ*=0.7 → 0.8774 / 0.8077 | 0.0225 |
| H10 | 0.8947 ± 0.0066 | 0.9814 | 0.7977 | 0.8039 | 0.8229 | 0.0601 | 0.4166 | τ*=0.7 → 0.8855 / 0.8140 | 0.0176 |
| H11 | 0.8800 ± 0.0086 | 0.9788 | 0.7576 | 0.7811 | 0.7753 | 0.0798 | 0.5258 | τ*=1.0 → 0.8712 / 0.8024 | 0.0222 |
| H12 | 0.8187 ± 0.0008 | 0.9537 | 0.6195 | 0.6577 | 0.6593 | 0.0326 | 0.5159 | τ*=0.2 → 0.8164 / 0.6691 | 0.0298 |
| baseline_c (son epoch) | 0.7378 ± 0.0060 | 0.9028 | 0.4844 | 0.5129 | 0.5084 | 0.2136 | 1.8309 | — | — |
| T4 (H08c⊕H10) | 0.9095 ± 0.0060 | 0.9895 | 0.8038 | 0.8291 | 0.8348 | 0.0456 | 0.2882 | — | — |

## Aile 1 (rakibe karşı; marj ≥ +2,0 puan ∧ GA alt > 0) — Holm
| kıyas | Δ ACC (puan) | %95 GA | p_t | p_boot | Holm t | Holm boot | kural | ifade |
|---|---|---|---|---|---|---|---|---|
| h01-baseline | +7.66 | [+5.14; +10.25] | 7.13e-06 | 0.0005 | 7.13e-06 | 0.002 | GEÇTİ | anlamlı |
| h02-baseline | +11.98 | [+9.91; +14.00] | 8.14e-07 | 0.0005 | 3.26e-06 | 0.002 | GEÇTİ | anlamlı |
| h08-baseline | +14.08 | [+12.00; +16.09] | 8.35e-07 | 0.0005 | 3.26e-06 | 0.002 | GEÇTİ | anlamlı |
| h10-baseline | +14.90 | [+12.54; +17.21] | 1.17e-06 | 0.0005 | 3.26e-06 | 0.002 | GEÇTİ | anlamlı |

## Aile 2 (H02c'ye karşı; ≥ +0,5 ∧ GA alt > 0) — Holm
| kıyas | Δ ACC (puan) | %95 GA | p_t | p_boot | Holm t | Holm boot | kural | ifade |
|---|---|---|---|---|---|---|---|---|
| h08-h02 | +2.09 | [+1.08; +3.17] | 0.000124 | 0.0005 | 0.000248 | 0.001 | GEÇTİ | anlamlı |
| h10-h02 | +2.91 | [+1.34; +4.61] | 0.000335 | 0.001 | 0.000335 | 0.001 | GEÇTİ | anlamlı |

## Doğrulama kuralları ve görülmemiş bayrakları (§2.1–2.2)
- **H08 (§2.1):** Δ +2.09 [+1.08; +3.17] → DOĞRULANDI · bayraklar {'dACC>3': False, 'ACC>0,90': False, 'AUC>0,99': False, 'dengeli_delta>6,5': True} · R18 satırı: H08
- **H10 (§2.2):** Δ +2.91 [+1.34; +4.61] → DOĞRULANDI · bayraklar {'ACC>0,93': False, 'AUC>0,99': False, 'H10-H02c>+6,0': False} · beklenti [+1,0;+4,5] içinde: True

## S/N ayrışımı (§2.6; kural = HAM Δ_S − Δ_N, K5-ONKAYIT-b §1; tabaka-std ikincil, etiket vermez)
| kıyas | hipotez | ham Δ_S / Δ_N | **ham Δ_S − Δ_N [GA]** | **kural** | ikincil: tab Δ_S / Δ_N | ikincil: tab Δ_S − Δ_N [GA] |
|---|---|---|---|---|---|---|
| h10-h02 | T-SN1: Δ_S > Δ_N | +6.26 / +1.13 | +5.13 [+1.76; +8.93] | desteklendi (GA alt > 0) | +5.38 / +1.46 | +3.92 [-2.25; +10.16] |
| h08-h02 | T-SN2: Δ_S ≈ Δ_N | +2.87 / +1.68 | +1.19 [-1.07; +3.43] | ayırt edilemedi (GA 0'ı kapsıyor → 'dengeli dağılım' ile uyumlu) | +2.74 / +3.28 | -0.54 [-4.07; +3.19] |
| h10-h08 | betimsel | +3.39 / -0.55 | +3.94 [+0.30; +7.68] | desteklendi (GA alt > 0) | +2.64 / -1.82 | +4.46 [-1.40; +10.44] |

## C0 (§3) — alt küme n = 573, çoğunluk tabanı 0.7818
| kol | ACC | dengeli | dengeli-5 | ACC − 0,7726 |
|---|---|---|---|---|
| baseline_c | 0.8387 ± 0.0070 | 0.4479 | 0.5643 | +0.0661 |
| H01c | 0.8914 ± 0.0008 | 0.6657 | 0.6920 | +0.1188 |
| H02c | 0.9127 ± 0.0037 | 0.6696 | 0.7523 | +0.1401 |
| H08c | 0.9302 ± 0.0041 | 0.7530 | 0.7879 | +0.1576 |
| H10 | 0.9232 ± 0.0033 | 0.7496 | 0.7637 | +0.1506 |
| H11 | 0.9222 ± 0.0088 | 0.7219 | 0.7684 | +0.1496 |
| H12 | 0.8813 ± 0.0000 | 0.5861 | 0.6376 | +0.1087 |
| T4 (H08c⊕H10) | 0.9382 ± 0.0053 | 0.7689 | 0.8022 | +0.1656 |
- C0-1 h10-h02: Δ +1.05 [-0.66; +2.79] → ayırt edilemedi · dengeli Δ +8.00 [+0.10; +16.75]
- C0-2 h08-h02: Δ +1.75 [+0.59; +3.04] → desteklendi (GA alt > 0) · dengeli Δ +8.34 [+2.79; +14.39]
- betimsel h10-h08: Δ -0.70 [-2.41; +0.87] → ayırt edilemedi · dengeli Δ -0.34 [-7.20; +6.31]
- betimsel h10-baseline: Δ +8.45 [+5.86; +11.13] → desteklendi (GA alt > 0) · dengeli Δ +30.17 [+19.75; +39.36]
- **KÇ0:** K0_H10 +1.86, K0_H08 +0.35, fark +1.52 [+0.02; +3.09] → desteklendi (GA alt > 0) (beklenen [+0,5; +2,5])

## C1 (§4) — KURU VEKİL
| kol | ACC | dengeli | dengeli-5 | ACC − 0,7726 |
|---|---|---|---|---|
| baseline | 0.8387 | 0.4479 | 0.5643 | +0.0661 |
| h01 | 0.8914 | 0.6657 | 0.6920 | +0.1188 |
| h02 | 0.9127 | 0.6696 | 0.7523 | +0.1401 |
| h08 | 0.9302 | 0.7530 | 0.7879 | +0.1576 |
| h10 | 0.9232 | 0.7496 | 0.7637 | +0.1506 |
- (i) h10-h02: Δ +1.05 [-0.66; +2.79] → ayırt edilemedi · dengeli yönü desteklendi (GA alt > 0)
- (i) h08-h02: Δ +1.75 [+0.59; +3.04] → desteklendi (GA alt > 0) · dengeli yönü desteklendi (GA alt > 0)
- (i) h10-h08: Δ -0.70 [-2.41; +0.87] → ayırt edilemedi · dengeli yönü ayırt edilemedi
- (ii) KÇ: K_H10 +1.86, K_H08 +0.35, fark +1.52 [+0.02; +3.09] → desteklendi (GA alt > 0); ifade eşiği True
- (iii) baseline C1 ACC 0.8387; G3 True; L False

## C2 (§5) — KURU VEKİL
- (a) baseline C2 0.8387 − 0,825 = +1.37 puan → yayımlanmış C sonucuyla uyumlu
- (b) baseline C2 − C1 +0.00 [+0.00; +0.00]
- (b) h08 C2 − C1 +0.00 [+0.00; +0.00]
- (b) h10 C2 − C1 +0.00 [+0.00; +0.00]
- (c) H10 − H08 (C2) -0.70 [-2.41; +0.87]

## §7 ölçütleri (mekanik; karar tahlilcide/kullanıcıda): {"W": "H10 (H13 Tur-5'te ÖLDÜR)", "G1": true, "G2_bayraklar": {"H08 (§2.1)": {"dACC>3": false, "ACC>0,90": false, "AUC>0,99": false, "dengeli_delta>6,5": true}, "H10 (§2.2)": {"ACC>0,93": false, "AUC>0,99": false, "H10-H02c>+6,0": false}}, "V": true, "G3": true, "L": false}
