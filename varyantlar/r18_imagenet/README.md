# r18_imagenet — H02 (`h02_r18in_ft224`)
Ebeveynden fark: rakibin 28 px sıfırdan ResNet-18'i yerine ImageNet önağırlıklı torchvision ResNet-18'in native 224 px
uçtan uca ince ayarı (AdamW 1e-4, kosinüs, 30 ep, bs 64, augment yok, epoch seçimi val ACC).
Koşum: bkz. `../KOMUTLAR.md` §3. Çıktılar: metrics.json (val_*), progress.json, gecmis.json, tahmin_val.npy, best_model.pth.
Ek (2026-10-10, onay sonrası): `tahmin_val_aucsecim.npy` + `val_*_aucsecim` ikincilleri (val-AUC seçimli epoch; kart b'').
Veri hattı düzeltmesi: `hazirla()` içinde `permute(...).contiguous()` (NHWC adımlı girdi deterministik cuDNN'de 232 s/epoch → 48,7 s; matematik aynı).
Kuyruk: `kuyruk/tur1.jsonl` (KOMUTLAR §T1).

## H05 eki (Tur-2, 2026-10-11) — `h05_r18in_ft224_cw05`
Ebeveynden (H02) fark: kayıp `F.cross_entropy(weight=w)`, w_c = π_c^−β / Σ_j π_j^(1−β), β = 0,5 (`--sinif-agirlik-us 0.5`;
π yalnız train). Varsayılan β = 0 → H02 yolu değişmez: GPU özdeşlik dumanı `20261011-012209_h05_r18in_ft224_cw00_duman_s0`
ilk_adim_loss / ilk_epoch_loss / val_acc / val_dengeli_acc H02 dumanıyla (`20261010-204010_…_duman_s0`) bit bit aynı (fark 0,0).
Yeni yalnız-ölçüm alanları: gecmis/progress `val_dengeli_acc` (epoch başına), metrics `sinif_agirlik_*`,
`val_dengeli_acc_eniyi_epoch`, `val_dengeli_acc_son`, `--ozdeslik-ref` ile `ozdeslik_*_fark`. Kuyruk: `kuyruk/tur2-h05.jsonl` (KOMUTLAR §T2).

## H08 / H09 ekleri (Tur-3, 2026-10-11) — `h08_r18in_ft224_d4`, `h09_r18in_ft224_renk02`
- H08 ebeveynden (H02) fark: train mini-batch'inde her örneğe D4'ün 8 dönüşümünden biri düzgün rastgele (`--augment-d4`;
  GPU, normalizasyondan sonra, `ortak/dihedral.py`; ayrı üreteç `g_aug` = seed + 100003 → batch sırası H02 ile aynı; `.contiguous()`).
- H09 ebeveynden (H02) fark: train'de örnek başına parlaklık → kontrast → doygunluk çarpanları U[0,8; 1,2], ton yok
  (`--augment-renk 0.2`; [0,1] ölçeğinde, normalizasyondan önce, her adımdan sonra kırpma; ayrı üreteç `g_renk` = seed + 200003).
- Kart yazımındaki bayrak adları (`--augment-d4`, `--augment-renk G`) kullanıldı; ikisi birlikte verilemez (M2). Varsayılan kapalı →
  H02 yolu bit bit aynı: GPU özdeşlik dumanları `20261011-043048_h08_r18in_ft224_d4kapali_duman_s0` ve
  `20261011-043202_h09_r18in_ft224_renk02kapali_duman_s0` ilk_adim_loss / ilk_epoch_loss / val_acc farkı 0 (kapı `tur3-on` GEÇTİ);
  CPU hata ayıklama ayarında eski ↔ yeni kod tahminleri bit bit aynı.
- Yeni yalnız-ölçüm alanları: metrics `train_loss_son`; H08 `augment_frekans_{0..7}`, `augment_frekans_maks_sapma`,
  `augment_toplam_ornek`, gecmis `augment_sayim`; H09 `renk_carpan_{b,c,s}_{ort,min,max}`, `augment_renk`.
- GPU ek yükü (40 adımlık hata ayıklama ölçümü): kapalı 16,40 s, D4 16,65 s (+%1,5), renk 16,86 s (+%2,8) → ≈ 49–50 s/epoch.
- Kuyruk: `kuyruk/tur3.jsonl` (KOMUTLAR §T3; şef başlatır).
- Düzeltme (04:50, hiçbir H09 açık koşumundan önce): renk çarpanları float32 u'dan **float64**'te hesaplanır (float32'de 0,8 + 0,4·u
  en üst uçta 1,2f = 1,20000005'e yuvarlanıp `renk_carpan_*_max ≤ 1,2` kapısında sahte İHLAL verebilirdi; dağılım ve üreteç
  tüketimi aynı). Kapalı yol dokunulmadı: CPU hata ayıklama ayarında eski kodla tahminler yine bit bit aynı.
