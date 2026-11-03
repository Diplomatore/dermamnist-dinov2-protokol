# dinov2_sonda — H01 (`h01_dinov2b_lp_r224`) ve H03 (`h03_dinov2b_lp_r{28,64,128}`)
Ebeveynden fark (H01): 28 px sıfırdan ResNet-18 yerine native 224 px donuk DINOv2 ViT-B/14 CLS + L2'li softmax-LR sondası
(C yalnız val'de). H03: aynı hat, girdi r∈{28,64,128} bicubic 224'e büyütülmüş (`--r`).
Koşum: bkz. `../KOMUTLAR.md` §1–2. Çıktılar: metrics.json (val_*), tahmin_val.npy, c_izgara.json, sonda_agirlik.npz.
Ek (2026-10-10, onay sonrası): `seed_tahmin_maks_fark` / `seed_tahmin_en_kucuk_fark` (tek koşumda, seçim seed'lerinin C*
tahminlerinden), `ayni_seed_tahmin_maks_fark` (C*'da aynı seed tekrarı), `yakinsama_fark` = |lbfgs − SGD ort| (öldürme e),
`val_auc_{mel,bkl,...}`. lbfgs basamak başına bir kez (a3 s0; diğer seed'ler `--lbfgs-yok`). Kuyruk: `kuyruk/tur1.jsonl` (KOMUTLAR §T1).
