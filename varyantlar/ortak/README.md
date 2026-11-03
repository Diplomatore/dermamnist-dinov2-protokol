# ortak — H01/H02/H03 ortak kod (rakip koduna dokunmaz)
- `veri.py`: dermamnist{,_64,_128,_224}.npz yükleyici (kök `data/medmnist`, resmî MD5 doğrulaması), boyutlar arası
  etiket tutarlılığı (train/val), val lezyon sızıntı maskesi (S = lezyonu train'de olan). **Test bölümü `test_izin=True`
  olmadan okunamaz (M4).**
- `degerlendir.py`: ACC/AUC = `medmnist.evaluator.getACC/getAUC` (birebir), dengeli ACC, makro-F1, ECE-15, NLL,
  sınıf duyarlılığı `alt_grup`, val bootstrap %95 GA (2.000, RNG 0). K3.5 ekleri (`scripts/hata_kirilim.py` tanımlarıyla
  aynı; baseline 5 koşumunda sayısal eşleşme doğrulandı): `val_tahmin_nv_payi`, `val_acc_molemax`, `val_acc_molemax_disi`,
  `val_dengeli_acc_molemax_disi`, **`val_sizinti_tabakali_SN`** (sınıf × dx_type × kaynak tabakalı, N-ağırlıklı S−N;
  raporlanan sızıntı ölçüsü) + `val_sizinti_tabakali_kapsam`.
  Sınıf başına OvR AUC `val_auc_{akiec,bcc,bkl,df,mel,nv,vasc}` (kart: mel/bkl; baseline'da hata_kirilim ile fark 0).
  **`*_ham` sonekli sızıntı anahtarları (`val_acc_sizan_ham`, `val_sizinti_farki_ham`, `..._Nagirlikli_ham` vb.)
  RAPORLANMAZ:** görüntü kaynağı/dx_type karıştırıcısı yüzünden yanıltıcıdır (Simpson paradoksu; baseline ham −18,8,
  tabakalı +2,8 puan — `sonuclar/hata-analizi.md` §4). Yalnız denetim için yazılır.
- `eslestirilmis_bootstrap.py`: iki koşum kümesinin `tahmin_val.npy`'sinden aynı val görüntüleri üzerinde Δ val ACC
  %95 GA (2.000, RNG 0; görüntü başına seed-ortalamalı doğruluk; isteğe bağlı sınıf-tabakalı). `--a-dosya/--b-dosya` (H02:
  `tahmin_val_aucsecim.npy`); çıktıda koşumlar arası seed tahmin farkı `seed_tahmin_fark_{a,b}`. K4c leaderboard eki.
- `kayit.py`: metrics.json (düz; `test_` anahtarı yazılamaz), progress.json, tahmin_val.npy/etiket_val.npy, config.json eki,
  seed bağlama (rakip köprüsüyle aynı + TF32 kapalı), cihaz seçimi (CUDA yoksa rc=97; CPU yalnız `--cpu-izin`).
