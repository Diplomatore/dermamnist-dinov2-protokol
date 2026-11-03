# Leaderboard — metrik `val_acc` (yüksek iyi), baseline `baseline_colab`

| # | varyant | n seed | ort ± std | min / max | Δ vs baseline | p (eşleştirilmiş) | durum |
|---|---|---|---|---|---|---|---|
| 1 | h08_r18in_ft224_d4_colab | 5 | 0.8865 ± 0.00334 | 0.8824 / 0.8903 | +0.1408 | p=1.67e-06 (n=5) | OK |
| 2 | h08_r18in_ft224_d4_colab_onucus | 2 | 0.8709 ± 0.00493 | 0.8674 / 0.8744 | +0.1251 | p=0.0321 (n=2) | YETERSİZ SEED (<3) |
| 3 | h02_r18in_ft224_colab | 5 | 0.8656 ± 0.00192 | 0.8634 / 0.8684 | +0.1198 | p=1.63e-06 (n=5) | OK |
| 4 | h02_r18in_ft224_colab_onucus | 2 | 0.8599 ± 0.00352 | 0.8574 / 0.8624 | +0.1142 | p=0 (n=2) | YETERSİZ SEED (<3) |
| 5 | h01_dinov2b_lp_r224_colab_onucus | 2 | 0.823 ± 0.00211 | 0.8215 / 0.8245 | +0.07727 | p=0.00897 (n=2) | YETERSİZ SEED (<3) |
| 6 | h01_dinov2b_lp_r224_colab | 5 | 0.8223 ± 0.0013 | 0.8215 / 0.8245 | +0.07657 | p=1.43e-05 (n=5) | OK |
| 7 | h08_r18in_ft224_d4_colab_duman | 1 | 0.8116 ± 0 | 0.8116 / 0.8116 | +0.0658 | p=— (n=1) | YETERSİZ SEED (<3) |
| 8 | h02_r18in_ft224_colab_duman | 1 | 0.8026 ± 0 | 0.8026 / 0.8026 | +0.05683 | p=— (n=1) | YETERSİZ SEED (<3) |
| 9 | baseline_colab_onucus | 2 | 0.7797 ± 0.00564 | 0.7757 / 0.7836 | +0.0339 | p=0.0347 (n=2) | YETERSİZ SEED (<3) |
| 10 | baseline_colab | 5 | 0.7458 ± 0.00736 | 0.7368 / 0.7547 | +0 | p=— (n=5) | BASELINE |
| 11 | baseline_colab_duman | 1 | 0.6869 ± 0 | 0.6869 / 0.6869 | -0.05882 | p=— (n=1) | YETERSİZ SEED (<3) |
| 12 | h01_dinov2b_lp_r224_colab_duman | 1 | 0.6523 ± 0 | 0.6523 / 0.6523 | -0.09342 | p=— (n=1) | YETERSİZ SEED (<3) |

_Yalnız damgalı (rc=0, metrics var) koşumlar sayıldı. n<3 varyant makaleye giremez._
