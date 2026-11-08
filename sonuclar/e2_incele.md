# E2 / H15 — İNCELE soruşturması: görülmemiş eşiği (iii)

*Tahlilci, 2026-11-08. Tetikleyen bayrak: `sonuclar/e2_bcn_sonuc.md` §5, (iii) |B1 Δ_dış,R dengeli (H10 − H02c)| = +16,34
[+13,86; +18,99] > 12,0. Bayraklar (i), (ii) ve (iv) kapalı. Soruşturma kartın satır 45'teki sırasını izler (`hipotezler/H15.md:45`).*

**Bu soruşturmada yapılmayanlar:**
- Hiçbir model BCN görüntüsü üzerinde çalıştırılmadı. Yeni bir BCN tahmini üretilmedi, dolayısıyla K-f'deki "ikinci BCN tahmini" sayılmaz.
- BCN üzerinde eşik, sıcaklık, önsel ya da kol seçimi yapılmadı.
- `e2_bcn_analiz.py` yeniden çalıştırılmadı. Kart, KİLİT, kod ve `e2_bcn_sonuc.*` dosyalarına dokunulmadı.

**Yapılanlar:** Kod ve dosya denetimi yapıldı. Drive'daki 55 a3 koşumunun tahmin dosyalarıyla bağımsız yeniden hesap yapıldı;
etiketler okundu, ama sonuç olarak değil, doğrulama amacıyla. Modeller yerelde yalnız BCN dışı görüntülerde çalıştırıldı: 8 HAM
kalibrasyon görüntüsü ve resmî val. Betikler ve çıktılar `sonuclar/e2_incele_ek/` altındadır.
**Mekanizma bölümü (§8) KEŞİFSEL ve SONRADANDIR.** Hiçbir karar kuralını değiştirmez.

---

## 1) Eşleme tablosu (BCN kodu → DermaMNIST indeksi) → **GEÇTİ**

| BCN | indeks | DermaMNIST |
|---|---|---|
| AK | 0 | akiec |
| BCC | 1 | bcc |
| BKL | 2 | bkl |
| DF | 3 | df |
| MEL | 4 | mel |
| NV | 5 | nv |
| VASC | 6 | vasc |
| SCC | dışlandı (E1) | — |

**Kanıt:**
- **Kodda tanım:** `scripts/e2_bcn_hazirla.py:43-45` (`DM` sırası ve `ESLEME`). Donmuş değer `kuyruk/e2_bcn_beklenen.json → esleme.E1`
  ve `sonuclar/e2_bcn.KILIT → esleme.E1` içindedir; ikisi aynı. a0 birim sınaması `esleme_sinama.gecti = true`.
- **Proje kodundan bağımsız kaynak:** `medmnist` 3.0.2 `INFO['dermamnist']['label']` şunu verir: 0 actinic keratoses…, 1 basal cell,
  2 benign keratosis, 3 dermatofibroma, 4 melanoma, 5 melanocytic nevi, 6 vascular. Bu sıra `ortak/veri.py:23` `SINIF_KISA` ile aynı.
- **ISIC'ten dış doğrulama:** 8 kalibrasyon görüntüsünün ISIC API tanısı ile `dermamnist_224.npz` train etiketi 8/8 tutarlı:
  "Solar or actinic keratosis" → 0, "Nevus" ×4 → 5, "Melanoma, NOS" ×3 → 4 (`data/bcn20000/kalibrasyon_ham/*.json`).
  Bu sınama akiec, nv ve mel indekslerini doğrular. Diğer dört sınıfın dayanağı medmnist INFO ve sınıf oranlarıdır.
- **Sınıf oranları:** DermaMNIST train payları [0,0325; 0,0512; 0,1097; 0,0114; 0,1112; 0,6698; 0,0141]. Sıralama HAM10000
  sınıf büyüklükleriyle aynı: nv ≫ mel ≈ bkl > bcc > akiec > vasc > df.
- **Kapsam:** BCN CSV'sindeki 8 tanı kodunun hepsi ele alınmış. Kodlar ve sayıları: MEL 2857, NV 4206, BCC 2809, BKL 1138,
  AK 737, DF 124, VASC 111 ve SCC 431 (dışlandı). Eşlenmeden kalan kod yok.

## 2) Kol başına normalizasyon, RGB sırası ve giriş çözünürlüğü (28 px mi, 224 px mi?) → **GEÇTİ**

**ResNet-18 eğitimi (H02c, H08c, h02_c1, h08_c1):**
- `varyantlar/r18_imagenet/ince_ayar.py:80`: `--size` varsayılanı 224. Satır 116-118: `V.npz_dogrula(a.size)` / `V.yukle(a.size)`.
- Satır 145-152, `hazirla`: uint8 NHWC → `permute(0,3,1,2)` → `/255` → `(x − IMAGENET_ORT) / IMAGENET_STD`. Sabitler satır 67-68'de.
- 20 R18 kaynak koşumunun `config.json` kayıtları: `protokol.size = 224`; `veri.npz = dermamnist_224.npz` (C1 kollarında
  `dermamnist_c_resmipiksel_224.npz`); `normalizasyon = "ImageNet (GPU'da)"`. `metrics.json` kaydı: `giris_cozunurluk = 224`.
- Bu kollarda 28 px'e ait hiçbir iz yok.

**DINOv2 eğitimi:**
- Tam ve kısmi ince ayar: `varyantlar/dinov2_ft/ince_ayar_vit.py:562-567`, aynı `hazirla` formülü.
- Sondalar: `varyantlar/dinov2_ozn/ozn_cikar.py:112-120`, aynı (/255, ImageNet ort/std). `sonda_agirlik.npz → kaynak_px = 224`.

**E2 çıkarımı:**
- `varyantlar/e2/e2_cikarim.py:51`: sabitler `k5_cikarim`'den alınıyor.
- Satır 218-223: `hazirla`, eğitimdeki formülün aynısı.
- Satır 225-227: val girdisi `V.yukle(224)`.
- Satır 239-254: r18 yolu. torchvision resnet18 + fc(512,7), `strict=True` ile yükleniyor; satır 244'te `.eval()` var, yani BN
  koşu istatistikleri kullanılıyor.
- Satır 438-453: BCN dizisi, val ile **aynı** `olasilik_dizi` işlevine giriyor. Kol türüne özgü ayrı bir BCN yolu yok.

**Girdi biçimi:**
- BCN dosyası `bcn_e2_merkez_kare_224.npz → images`: (12.413, 224, 224, 3) uint8.
- DermaMNIST val: (1.003, 224, 224, 3) uint8. Düzen (NHWC, uint8, 0–255) aynı.
- BCN 224 dizisi doğrudan 1024² özgün JPEG'den üretiliyor: `scripts/indir_bcn20000.py:646-649` (`convert("RGB")` →
  `geometri(merkez_kare)` → `PIL_BILINEAR(224)`), geometri tanımı satır 357-367'de.
- 28 px dosyası (`bcn20000_train_merkez_kare_28.npz`) ayrıdır ve yalnız `nn_tam` için kullanılır. **28 → 224 büyütme yok.**
- **Yanıt:** ResNet kolları eğitimde de BCN'de de 224 px'ten, aynı kod yolundan besleniyor. Δ'yı yapay şişirecek bir çözünürlük
  uyuşmazlığı bulunmadı.

**RGB sırası (ölçüldü):**
- Kanal ortalamaları. BCN (1.500 rastgele görüntü): [148,0; 128,7; 123,8]. DermaMNIST train: [195,5; 136,7; 142,8].
  İkisinde de R en büyük kanal. BGR takası olsaydı BCN'de 1. kanal en küçük olurdu.
- 8 HAM görüntüsü BCN yolundan geçirilip npz'deki aynı görüntüyle karşılaştırıldığında RMSE 1,76 gri düzey çıkıyor
  (`kalibrasyon_onisleme.json`). Kanal takası RMSE'yi onlarca düzeye çıkarırdı.

**Ampirik özdeşlik:**
- a3'te 55/55 kol-seed K-c'yi geçti: 54'ü Durum A, 1'i Durum B. B olan h02c s4: argmax 1003/1003, maks|Δp| 1,24e-5,
  cuDNN 92700 → 91900 (`runs/20261108-002423_e2_h02c-bcn_s4/metrics.json`). Bu, eğitim zamanında arşivlenen val tahminlerinin E2
  kodu ve E2 ön-işlemesiyle yeniden üretildiği anlamına gelir.
- Bu soruşturmada model yükleyicisi yeniden yazılıp yerelde çalıştırıldı (GTX 1650, torch 2.5.1; `v3_model.py`). Altı resmî kolun
  (H10, H02c, H08c, H11, H01c, H12) 30 kol-seed'inde resmî val argmax uyumu arşivle %100, maks|Δp| ≤ 1,1e-4 (§8d tablosu,
  "orijinal" sütunu).

## 3) 8 HAM kalibrasyon görüntüsünün argmax uyumu → **GEÇTİ**

- **a1 (Colab, H10 s0):** `kal_argmax_uyum = 8/8`, `kal_p_maks_fark = 0,0128` (`runs/20261107-233322_e2_h10_duman_s0/metrics.json`).
- **Bu soruşturmada genişletildi:** Ölçüm 55 kol-seed'in tümüne yayıldı (yerel GPU, BCN görüntüsü yok). 8 görüntünün hepsi hem resmî
  train'de hem C-train'de (`data/splits/dermamnist_c_esleme.csv`). Bu yüzden C1 kollarının bu görüntüler üzerinde çalıştırılması
  H14 kural (i)'ye takılmaz.

**Kalibrasyon sonuçları (55 kol-seed, `v3_model.py`):**
- Uyum ölçütü: aynı HAM görüntüsü iki yoldan modele verildi: BCN ön-işleme yolu (`bcn_e2_kal_224.npz`) ve `dermamnist_224.npz` train.
  İki yoldaki argmax sayıldı. Kart eşiği ≥ 7/8.
- "Doğru" sütunu bilgi amaçlıdır: argmax'ın HAM etiketine eşit olduğu görüntü sayısı, önce BCN yolu sonra npz yolu.

| kol | tür | argmax uyumu, s0…s4 | en düşük | maks\|Δp\| (5 seed) | doğru (BCN yolu / npz), s0…s4 |
|---|---|---|---|---|---|
| H10 | vit_kismi | 8 8 8 8 8 | 8/8 | 0,016 | 8/8 ×5 |
| H02c | r18 | 8 8 8 8 8 | 8/8 | 0,231 | 8/8 ×5 |
| h10_c1 | vit_kismi | 8 8 8 8 8 | 8/8 | 0,057 | 8/8 ×5 |
| h02_c1 | r18 | 8 7 7 8 8 | 7/8 | 0,437 | 8/8 · 7/7 · 7/8 · 8/8 · 8/8 |
| H08c | r18 | 8 8 8 8 8 | 8/8 | 0,194 | 8/8 ×5 |
| h08_c1 | r18 | 8 8 8 7 8 | 7/8 | 0,554 | 8/8 · 8/8 · 8/8 · 7/8 · 8/8 |
| h14_c1 | vit_kismi | 8 8 8 8 8 | 8/8 | 0,040 | 8/8 ×5 |
| H11 | vit_tam | 8 8 8 8 8 | 8/8 | 0,006 | 8/8 ×5 |
| H01c | sonda | 8 8 8 8 8 | 8/8 | 0,181 | 8/8 ×5 |
| H12 | sonda | 7 7 7 7 7 | 7/8 | 0,117 | 6/6 ×5 |
| h01_c1 | sonda | 7 7 7 8 7 | 7/8 | 0,174 | 6/7 · 6/7 · 7/8 · 7/7 · 6/7 |

- **55/55 kol-seed eşiği (≥ 7/8) geçti.**
- ViT kolları (vit_kismi + vit_tam) 160/160 uyumlu, R18 kolları 157/160 (%98,1), sondalar 111/120.
- Sondalar ezberlemiyor (doğrusal sonda), bu yüzden sınırdaki görüntülerde 7/8 çıkması doğal. H12'de iki yol aynı 6 görüntüyü doğru
  bildi.
- **Not (sınırlılık):** BCN ön-işlemesi MedMNIST'inkiyle birebir aynı değil; kalibrasyonda en iyi aday RMSE 1,76'da kalmıştı.
  R18 bu küçük farka DINOv2'den belirgin biçimde daha duyarlı:
  - maks|Δp|: R18 en çok 0,55, kısmen donuk DINOv2-B en çok 0,06.
  - argmax: R18'de 3/160 dönme var, ViT'te 0/160.
- Bu fark tek başına 16 puanlık Δ'yı açıklayamaz: %2'lik argmax dönmesi karşısında ölçülen düşüş 30–50 puan. Bununla birlikte
  R18'in genel kırılganlığının bir parçasıdır ve makalede sınırlılık olarak yazılmalıdır.

- **Sınamanın sınırı:** Kalibrasyon görüntüleri eğitim görüntüleridir ve ince ayarlı kollarca ezberlenmiş olabilir. Örneğin H02c'nin
  eğitim kaybı 7e-4 (`runs/20261011-130855_…_s0/stdout.log`). Bu yüzden argmax uyumu zayıf bir sınamadır; maks|Δp| daha
  bilgilendiricidir.
- **Gizli uygulama kusuru (sonuçları ETKİLEMEDİ):**
  - `e2_cikarim.py:417` kalibrasyon referansını `V.npz_yolu(224)`'ten okuyor. C1 kollarında bu işlev `c_bolme` tarafından
    yamalandığından (`e2_cikarim.py:134-136`) referans C npz'sinin `train_images[kidx]`'i olurdu, yani başka görüntüler.
  - `--kal` yalnız a1'de, resmî kol H10'da çalıştı. a3 kal kullanmıyor. Kusur tetiklenmedi.
  - Kayda geçmesi için yazıldı.

## 4) `tahmin_bcn` satır sırası ↔ `bcn_filename` hizası → **GEÇTİ**

**Girdi dosyaları:**
- Etiketsiz e2 girdisi `bcn_e2_merkez_kare_224.npz` yalnız `images` ve `bcn_filename` anahtarlarını taşıyor.
- Etiketli kaynakla (`bcn20000_train_merkez_kare_224.npz`) dosya adı sırası 12.413/12.413 aynı.
- Görüntü baytları aynı (`np.array_equal` = True).
- `sira_sha256 = 66904b9b…` beklenenle aynı; içerik SHA256'sı beklenenle aynı.
- Sıra sıralı ve tekil. Kaynak npz'deki `diagnosis` alanı CSV ile 12.413/12.413 aynı.

**Çıkarım ve analiz:**
- Çıkarım satır sırasını koruyor: parçalar sırayla birleştiriliyor (`e2_cikarim.py:448-453`).
- 55 a3 koşumunun hepsinde `bcn_sira_sha256` beklenenle aynı ve tahmin şekli (12.413, 7). Analiz bunu denetliyor
  (`e2_bcn_analiz.py:772-780`).
- Etiketler dosya adıyla birleştiriliyor (`e2_bcn_analiz.py:798-810`).

**Bağımsız yeniden hesap (`v2_tahmin.py`):** Drive'daki `tahmin_bcn_merkez_kare.npy` dosyaları, CSV ve kopyasız liste kullanıldı;
analiz kodu kullanılmadı.

| kol | dengeli |
|---|---|
| H10 | 0,4501 |
| H02c | 0,2867 |
| h10_c1 | 0,4510 |
| h02_c1 | 0,2912 |

- **Δ = +16,34.** Bu sonuç `e2_bcn_sonuc.md` ile 4 hanede aynı. 11 kolun hepsi aynı.
- Seed bazında H10 0,4385–0,4599, H02c 0,2680–0,3055. Seed aralıkları birbirine değmiyor.
- Hiza bozuk olsaydı bütün kollar şansa (≈ 0,14 dengeli, AUC ≈ 0,5) inerdi. Ölçülen makro-AUC değerleri H10 0,861, H02c 0,750.

## 5) Kopyasız liste ve lezyon kümeleri → **GEÇTİ** (bir açıklama notuyla)

**Kopyasız liste:**
- Zip CRC32 + boyutla bağımsız olarak yeniden türetildi (`v1_veri.py`): 47 kopya grubu ve 47 fazla görüntü.
- 2 çelişkili grup var: BCN_0000014439/14930 (MEL↔NV) ve BCN_0000015995/18062 (NV↔MEL). Toplam 49 çıkarma.
- Yeniden türetilen küme `kopyasiz_liste.json → dislanan` ile birebir aynı.

**Sayılar:**
- 11.933 görüntü / 3.451 lezyon.
- Sınıf başına görüntü: [729, 2802, 1124, 124, 2847, 4196, 111]. Bu değerler beklenen dosyayla ve sonuçla aynı.

**Lezyon kümeleri:**
- BKL çifti (BCN_0001604 ↔ BCN_0004816): iki lezyon da kopyasız kümede, birleşti. bkl küme sayısı 352 = 353 − 1.
- NV çifti (BCN_0002049 ↔ BCN_0002919): BCN_0002919'un 5 görüntüsünün 5'i de kopya fazlası olarak çıkarıldı, yani bu lezyon kopyasız
  kümede yok. Birleştirme bu yüzden etkisiz kaldı.
- Sonuçta yazan "lezyon 3451 (bootstrap kümesi 3450)" bunun sonucudur; 3.449 beklemek yanlış olurdu. Hata değil.
  Makalede "iki çiftten biri kopya temizliğiyle zaten tek kimliğe indi" diye yazılabilir.

**Duyarlılık:** Kopyalar dahil ham küme (S8) için B1 = +16,34, ana sonuçla aynı. Kopya kararı farkı üretmiyor.

## 6) `nn_tam` (sızıntı) → **GEÇTİ**

- Kaynak: `data/bcn20000/bcn20000_nn_tam.json`. 12.413 BCN görüntüsü × 10.015 DermaMNIST görüntüsü (28 px).
- RMSE ≤ 5 olan çift sayısı **0** (K-a kapalı). RMSE ≤ 10: 20; en küçük RMSE 7,67; medyan 28,83. Colab veri kapısında da aynı:
  `kapi_nn_esik5_alti = 0` (`runs/20261107-233010_e2_veri_kapisi_s0`).
- RMSE ≤ 10 olan 20 BCN görüntüsünün 7'si aynı referansa (train:1853) gidiyor ve bu görüntülerin tanıları farklı
  (BCC/BKL/NV/AK/SCC). Bu, içeriksiz ve düz görüntüler arasındaki benzerliktir, kopya değildir.
- Bu 20 görüntü çıkarıldığında (`v4_nn.py`; n = 11.915): H10 0,4503, H02c 0,2868, **Δ = +16,35**. Sızıntı yolu Δ'yı açıklamıyor.

## 7) Kol ↔ run_id ↔ `config.json` eşlemesi → **GEÇTİ**

55/55 kol-seed için şunlar denetlendi (`v1_veri.py`), hata listesi boş:
- a3 `config.e2.kaynak_run` = `kollar.py` run_id = KİLİT `kollar` listesi = `config.ebeveyn`.
- Komuttaki `--kol` değeri kolla aynı.
- Seed a3 ile kaynakta aynı.
- Kaynak `config.varyant` = `kollar.py` varyantı.
- Kaynak `config.json` SHA256'sı a3'te kaydedilen değerle aynı.
- Yeniden çıkarılan `val_acc` kaynak `val_acc`'ye bit bit eşit.
- Her kolda 5 seed'in ağırlık SHA256'ları birbirinden farklı (11/11 kol).

Tür karışması da mümkün değil:
- R18 yolu `strict=True` ile yüklüyor ve bir ViT state_dict'i R18'e yüklenemez.
- Kaydedilen parametre sayıları türleri doğruluyor: R18 11.180.103, ViT-B 85.730.311.

---

## 8) Mekanizma — **KEŞİFSEL, SONRADAN** (karar vermez; makalede bu etiketle)

**a) ResNet kolları nv'ye kayıyor, ama sorun yalnız önsel değil, sıralama da bozuluyor.**
- Tahminlerde nv payı: R18 kolları 0,66–0,74, H10 0,44. Gerçek pay 0,352.
- Seed ortalamasıyla H02c → H10 karşılaştırması:

| geçiş | H02c | H10 |
|---|---|---|
| mel→nv | 1.628/2.847 | 812 |
| bcc→nv | 1.319/2.802 | 457 |
| bkl→nv | 617/1.124 | 278 |

- H10, nv dışındaki altı sınıfın hepsinde daha duyarlı (H02c → H10):

| sınıf | H02c | H10 |
|---|---|---|
| akiec | 0,06 | 0,08 |
| bcc | 0,19 | 0,44 |
| bkl | 0,33 | 0,55 |
| df | 0,05 | 0,31 |
| mel | 0,22 | 0,51 |
| vasc | 0,23 | 0,40 |

- Makro-AUC değerleri H02c 0,750, H10 0,861. Fark eşikten bağımsız bir sıralama farkıdır; bir önsel ya da eşik ayarı onu kapatmaz.
  Böyle bir ayar zaten yapılmadı.

**b) Piksel düzeyinde kayma büyük (etiketsiz ölçüm):**
- Kanal ortalaması: BCN [148; 129; 124], DermaMNIST [195; 137; 143]. R kanalı 47 düzey düşük.
- Kanal std: BCN [65; 62; 63], DermaMNIST [35; 40; 45]. Kontrast 1,4–1,9 kat.
- Koyu köşe oranı (köşe gri ortalaması < 40): BCN %46, DermaMNIST %0,04 (`nn_tam` 28 px ölçümü: 0,42 / 0,0004).

**c) Fark alt gruplarda homojen; tek bir artefakttan gelmiyor** (BCN, mevcut tahminler, `v2_tahmin.py`):

| alt grup | Δ (H10 − H02c), puan |
|---|---|
| yazar kırpıcısı tetiklendi (n 2.111) | +17,5 |
| yazar kırpıcısı tetiklenmedi (n 9.822) | +16,0 |
| koyu köşe | +16,0 |
| koyu köşe yok | +16,0 |
| R kanalı çeyrek 1 | +17,1 |
| R kanalı çeyrek 2 | +20,0 |
| R kanalı çeyrek 3 | +13,6 |
| R kanalı çeyrek 4 | +14,7 |

- H02c'nin nv payı bütün alt gruplarda 0,64–0,70.
- Çerçeve kırpma duyarlılığı S9 (+16,40) da aynı yönde.

**d) Sentetik kayma, BCN olmadan da aynı deseni üretiyor** (resmî val, 5 seed ortalaması; yalnız resmî kollar; `v3_model.py`):

| kol (5 seed) | orijinal | renk_ort | renk_ort_std | cerceve | renk+cerceve |
|---|---|---|---|---|---|
| H10 | 0,798 (nv 0,69) | 0,755 (0,67) | 0,708 (0,64) | 0,736 (0,70) | 0,662 (0,67) |
| H02c | 0,710 (nv 0,70) | **0,393 (0,79)** | 0,443 (0,70) | 0,573 (0,72) | **0,380** (0,66) |
| H08c | 0,776 (nv 0,69) | **0,422 (0,81)** | 0,433 (0,74) | 0,617 (0,64) | **0,308** (0,67) |
| H11 | 0,758 (nv 0,68) | 0,561 (0,72) | 0,551 (0,70) | 0,661 (0,68) | 0,498 (0,74) |
| H01c | 0,649 (nv 0,70) | 0,572 (0,66) | 0,550 (0,61) | 0,537 (0,62) | 0,486 (0,51) |
| H12 | 0,619 (nv 0,70) | 0,615 (0,67) | 0,561 (0,65) | 0,540 (0,63) | 0,522 (0,57) |
| **Δ H10 − H02c (puan)** | **+8,8** | **+36,2** | +26,5 | +16,2 | +28,3 |

- Hücrelerde val dengeli doğruluk ve parantez içinde tahmin nv payı var.
- "orijinal" sütununun arşivle uyumu: 6 kolun 30 kol-seed'inde argmax %100, maks|Δp| ≤ 1,1e-4. Yerel yükleyici bağımsız olarak
  yeniden yazıldı; bu satır §2'deki özdeşliği destekler.
- Yalnız ortalama renk kaymasında (`renk_ort`) düşüş H02c'de −31,7 puan, H10'da −4,3 puan. R18 kolları nv'ye kayıyor (nv payı 0,79–0,81).
- Kolların sentetik kaymadaki sıralaması BCN'deki sıralamayla aynı: H10 > H11 ≈ sondalar > R18.

- Bozulmaların tanımı:
  - `renk_ort`: kanal başına ortalama DermaMNIST'ten BCN'ye kaydırıldı.
  - `renk_ort_std`: ortalama ve std birlikte eşlendi.
  - `cerceve`: daire dışı siyah.
  - `renk+cerceve`: son iki işlemin birleşimi.
- Bu sınama BCN etiketine ya da BCN görüntüsüne dokunmaz. Yalnız iki veri kümesinin global kanal istatistiklerini kullanır.

**e) Model ailesi deseni (BCN dengeli):**
- Kısmen donuk DINOv2-B (H10, h10_c1, h14_c1): 0,45–0,46.
- Tam ince ayarlı DINOv2-S (H11): 0,37.
- Donuk sondalar: 0,35–0,38.
- R18'in bütün kolları 0,28–0,29; D4 augmentasyonlu H08c ve h08_c1 de dahil.
- R18'ler 30 epoch boyunca renk augmentasyonu olmadan eğitildi (H02 `augment: yok`, H08 yalnız D4) ve eğitim kaybı ≈ 1e-3'e indi.
- Gözlem, Taori vd. 2020'nin (kart kaynağı) yönüyle uyumlu: çeşitli ve büyük ön-eğitim doğal kaymaya karşı dayanıklılık sağlıyor.

**Sınanmamış ek hipotez:** R18'in BatchNorm katmanları eval modunda kaynak alanın (HAM) koşu istatistiklerini kullanır. Bu, CNN'lerin
kovaryat kaymasına kırılganlığının bilinen bir kaynağıdır (Li vd. 2016, "Revisiting Batch Normalization for Practical Domain
Adaptation", arXiv:1603.04779; Schneider vd. 2020, "Improving robustness against common corruptions by covariate shift adaptation",
NeurIPS). ViT ise örnek başına LayerNorm kullanır. Bu hipotez **yön kaynağıdır, burada sınanmadı.** Sınamak BCN üzerinde bir
uyarlama (BN yeniden kestirimi) gerektirir; bu E2'nin "ayarsız" tanımının dışındadır ve yeni kart ister (kart ablasyon iii eşi).

**Eşiğin okunuşu:**
- (iii) eşiği literatürden değil, iç Δ'dan türetilmiş bir orandır (2 × 6,03).
- Mutlak eşikler (i) ve (ii) kapalı: H10'un BCN dengelisi 0,450, lezyon-ayrık iç değeri 0,765'in 31,5 puan altında ve 0,70'in altında.
- Yani "görülmemiş kadar iyi" olan DINOv2 değil. Farkı açan, ResNet-18'in kart beklentisinden ([−35; −8]) daha büyük düşüşü
  (−41 … −52). Bu kötü yönlü bir beklenti ihlalidir ve arıza alarmıdır.
- Bu alarm §1–§7'deki R18'e özgü bütün hata yollarıyla sınandı: çözünürlük, normalizasyon, RGB, BN eval, yükleme, hiza, eşleme.
  Hepsi kapalı.

---

## 9) Karar → **KAPANDI** (hata yok; Δ ölçüm hattı açısından gerçek)

- **Kart sırasındaki yedi madde: 7/7 GEÇTİ.** Kalibrasyon sınaması a1'deki tek kol-seed'den 55/55 kol-seed'e genişletildi. Ek olarak bağımsız yeniden hesap (+16,34) ve sızıntı dışlama (+16,35) Δ'yı değiştirmedi.
- **Δ'yı yapay şişirebilecek kol-özgü bir ön-işleme uyuşmazlığı bulunmadı.** ResNet kolları eğitimde de BCN'de de 224 px, aynı
  `hazirla` ve aynı ImageNet normalizasyonuyla besleniyor. Val özdeşliği 55/55.
- **Mekanizma (keşifsel):** ImageNet-ResNet-18 kolları BCN'nin renk ve kontrast kaymasında nv sınıfına çöküyor; DINOv2-B (kısmen
  donuk) çökmüyor.
  - Aynı desen BCN olmadan, resmî val'e uygulanan BCN-benzeri renk kaymasıyla yeniden üretiliyor (§8d). Örneğin yalnız ortalama
    renk kayması H10 − H02c farkını val'de +8,8'den +36,2 puana çıkarıyor.
  - Fark çerçeve, parlaklık ve renk alt gruplarında homojen (§8c).
- **K-f:** Teknik nedenle yeniden açılış **gerekmez.** Yeni BCN çıkarımı ya da ikinci analiz yok, "ikinci açılış" beyanı yok.
- **Sonraki adım (karar kullanıcının):** Kart kuralına göre soruşturma kapanınca sayılar yazılabilir ve mekanik cümle seçilebilir.
  B1 etiketi "DINOv2 üstün" (Holm p 0,0004), B2 etiketi "protokoller ayırt edilemedi". Önerilen ek beyanlar:
  - B1 kartın ön-kayıtlı aralığının ([−4; +12]) dışında ve (iii) bayrağı açılıp soruşturuldu. Bu, makalede gizlenmeden yazılmalı
    (M7).
  - Fark büyük ölçüde ResNet-18'in nv'ye çöküşünden geliyor. Bu bulgu "DINOv2 dışta daha iyi genelleştirir" diye genellenmemeli; bu
    eğitim tarifinin (renk augmentasyonu yok) ResNet-18'i için geçerlidir. Renk augmentasyonlu R18 (H09 yolu) E2'de yoktu;
    sınırlılık olarak yazılmalı.
  - §8'deki bütün mekanizma ifadeleri "keşifsel, sonradan" etiketiyle verilmeli.

**Günlük arıza tablosu için satır önerisi:**
`2026-11-08 | E2 a3 analizi | Görülmemiş (iii): B1 +16,34 > 12,0 | Kart satır 45 sırasıyla 7 madde + bağımsız yeniden hesap; hata yok |
KAPANDI. Mekanizma (keşifsel): R18 nv çöküşü / renk-kontrast kayması; val sentetik kaymayla yeniden üretildi | ikinci açılış yok;
e2_cikarim.py:417 C1-kal gizli kusuru not edildi (tetiklenmedi)`

**Betikler ve ham çıktılar** (yeniden üretilebilir; mevcut kod ve sonuç dosyalarına dokunmaz): `sonuclar/e2_incele_ek/`
- `v1_veri.py` → `v1.json`: eşleme, hiza, kopyasız liste, `nn_tam`, kol eşlemesi.
- `v2_tahmin.py` → `v2.json`: bağımsız yeniden hesap ve alt gruplar.
- `v3_model.py` → `v3.json`, `v3.log`: kalibrasyon ve val sentetik kayma. Model BCN görüntüsü görmez. İlk koşum Drive okuma hatasıyla
  (OSError 22) h14_c1 s2'de kesildi; betik kaldığı yerden sürdürüldü, tamamlanan satırlar yeniden koşulmadı.
- `v4_nn.py`: `nn_tam` RMSE ≤ 10 olan görüntüler hariç tutularak Δ.
- Çalıştırma: `env/.venv/Scripts/python.exe sonuclar/e2_incele_ek/<betik> <çıktı.json>` (v4 argümansız).
