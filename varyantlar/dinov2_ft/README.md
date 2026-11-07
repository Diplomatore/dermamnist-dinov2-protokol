# dinov2_ft — DINOv2 (timm) native 224 ince ayarı (Tur-4 ön-ölçüm aracı; KART YOK, 2026-10-11)
Ebeveynden fark (aday H10/H11; kart yazılınca kesinleşir): omurga R18-IN → DINOv2 ViT-B/14 (son K blok, kısmi) ya da
ViT-S/14 (tam). Tarif ebeveyni `h02_r18in_ft224` (augment yok, FP32, AdamW, kosinüs, etkin bs 64, val ACC seçimi).

- `ince_ayar_vit.py --mod kismi --son-k K [--model b]`: ilk 12 − K blok donuk. Çıkış token'ları bir kez
  `data/oznitelik/dinov2_vitb14_r224_blok{12−K}_tok_{train,val}.npy` (fp16, memmap, 3,16 GB) + `.json` meta'ya yazılır.
  Önbellek her koşumda tam ağa karşı denetlenir (`onbellek_tutarlilik_maks_fark` ≈ fp16 payı 2,4e-3).
  `--sadece-onbellek`: yalnız önbelleği yaz (ilk kuyruk satırı ya da ayrı ölçüm).
- `--mod tam [--model s|b]`: bütün ağ; `--grad-ckpt` (tam'da varsayılan 1), mikro-batch + gradyan birikimi.
- metrics anahtarları H02 ile aynı (`ortak/degerlendir.py`), ek: ölçüm alanları (tepe GPU/RAM, img/s, önbellek).
  Test okunmaz (M4). `best_model.pth` TAM state_dict + `best_model_meta.json` (yükleme tarifi; K5 için).
- Varsayılanlar kart değildir: lr baş 1e-3 / gövde 1e-5 (Wu & Papież 2025), wd 0,01 (1-B ve pos/cls muaf), adım
  başına kosinüs, ısınma 0, LLRD kapalı, drop-path 0, `--epoch 15`.

GTX 1650 notu: GPU belleği 4 GB'ı aşınca WDDM paylaşımlı belleğe düşüyor → 10× yavaş (S tam ckpt0 mikro 64: 3,8 img/s,
rc=98). Güvenli: kısmi mikro ≤ 32, S tam ckpt0 mikro ≤ 16 (2,2 GB) veya ckpt1, B tam ckpt1 mikro 16.
`kosum.py --gpu-zorunlu` venv python'u ile çağrılmalı (uv python'unda CUDA'lı torch yok → rc=97).
Ölçüm tablosu ve komutlar: `../KOMUTLAR.md` §T4.

Koşum (WS kökünden):
  env/.venv/Scripts/python.exe varyantlar/dinov2_ft/ince_ayar_vit.py --mod kismi --son-k 2 --seed 0 --epoch 15 --out {RUN_DIR}/metrics.json
  env/.venv/Scripts/python.exe varyantlar/dinov2_ft/ince_ayar_vit.py --mod tam --model s --grad-ckpt 0 --mikro-bs 16 --seed 0 --epoch 15 --out {RUN_DIR}/metrics.json

## E1 / H14 bayrakları (2026-11-03; `hipotezler/H14.md`, `hipotezler/H14-UYGULAMA-NOTU.md`)
Ebeveynden (`h10_c1`) fark (tek cümle): eğitimde her örneğe D4'ün 8 dönüşümünden biri düzgün dağılımla atanır
(`g_aug = seed + 100_003`). Dönüşüm donuk 8 bloktan ÖNCE görüntüye uygulanır ve 8 görünümlü fp16 önbellekten okunur.
Bayrakların varsayılanı KAPALIDIR. Kapalıyken H10 yolu bit bit değişmez; yerel CPU sınaması H14-UYGULAMA-NOTU §1'de.
- `--onbellek-gorunum 8`: g0 = H10'un `..._blok8_tok_train.npy` dosyası; g1..g7 = `..._blok8_tok_d4g{k}_train.npy`; meta `..._tok_d4.json`.
  - Toplam ≈ 26,2 GB; Colab yerel diskine yazılır.
  - Yazımdan önce boş disk < `--disk-min-gb` (40) → DUR.
  - Görünüm tutarlılığı > 5e-3 ya da özdeş görünüm > 0 → DUR.
- `--augment-d4`: örnek başına görünüm seçimi. Sayaçlar H08 ile aynı adlarla tutulur (`augment_*`).
- `--ozdeslik-ref R`: kontrol kolu.
  - `kontrol_*` anahtarlarını yazar ve kural (f) Durum A/B/C'yi belirler. Durum C → rc=5.
  - R = run_id | yol | `@<kuyruk_onek>/<varyant>`.

Koşum (Colab; kuyruk satırları `colab/hat/e1_kuyruk_yaz.py`):
  python colab/fp32_kos.py varyantlar/k5/c_bolme_sar.py --veri c1 -- varyantlar/dinov2_ft/ince_ayar_vit.py --mod kismi --model b --son-k 4 --seed S --epoch 15 --lr-bas 1e-3 --lr-govde 1e-5 --wd 0.01 --katman-azalma 1 --isinma-epoch 0 --drop-path 0 --mikro-bs 32 --onbellek-gorunum 8 --augment-d4 --out {RUN_DIR}/metrics.json
