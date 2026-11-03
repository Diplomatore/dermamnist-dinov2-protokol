# Rakip (otoriter baseline)

- Resmî kod: <https://github.com/MedMNIST/experiments> @ `70b6b3a7ad7afddff1df2a3b735235830fbdb142`
  (`MedMNIST2D/train_and_eval_pytorch.py`, `MedMNIST2D/models.py`). Lisans: Apache-2.0. **Bu depoya kopyalanmadı.**
- Getirme: `bash rakip/getir.sh` (→ `rakip/experiments/`, `.gitignore`'da).
- Bizim köprülerimiz (rakip koduna dokunmaz):
  - `scripts/rakip_kos.py`: resmî `main()`'i resmî varsayılanlarla çağırır. Seed, veri kökü ve kayıt sarmalayıcısı
    ekler. Rakip dosyalarının git blob SHA1 değerlerini her koşumda doğrular.
  - `varyantlar/k5/c_rakip_kos.py`: DermaMNIST-C (C1) için yalnız veri yolu uyarlaması yapar
    (`hipotezler/K5-C1-UYGULAMA-NOTU.md`).
- Protokol eşlemesi, sapmalar ve kısaltılmış koşumlar: `rakip/KURULUM.md`. Yamalanan dosya yok.
