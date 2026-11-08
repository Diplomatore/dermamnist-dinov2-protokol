# Üçüncü taraf kod — BCN20000 siyah çerçeve kırpıcısı

- Dosya: `image_cropper.py` (bayt bayt aynı kopya; DEĞİŞTİRİLMEZ).
- Kaynak: GitHub `imatge-upc/BCN20000` @ `7ea35f5bf524b66f2ee98eef9428bba74f44547b`, yol `preprocessing/image_cropper.py`.
- İndirme: `https://raw.githubusercontent.com/imatge-upc/BCN20000/7ea35f5bf524b66f2ee98eef9428bba74f44547b/preprocessing/image_cropper.py`
  (2026-11-07, deney-yürütücü).
- SHA256: `103d0a3df048bb37473b2fac2b574380fb3e8e05c0a4e60d6e836fa3bbaf74d6` (3.740 B).
- Lisans: depo kökündeki `LICENSE` = Creative Commons Attribution 4.0 International (CC BY 4.0). Atıf: Hernández-Pérez vd.,
  *BCN20000: Dermoscopic Lesions in the Wild*, Sci Data 11, 641 (2024).
- Kullanım (H15 S9 duyarlılığı): `varyantlar/e2/cerceve.py` bu modülün `obtain_bb` işlevini **olduğu gibi** çağırır.
  `intensity_ratio` yerine aynı formülün vektörel eşdeğeri kullanılır (gerekçe ve eşdeğerlik sınaması: `cerceve.py`
  belge dizisi ve `hipotezler/H15-UYGULAMA-NOTU.md`).
