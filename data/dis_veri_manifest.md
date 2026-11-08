# Dış Veri Manifesti — BCN20000 (E2 kilitli dış test) (3.1 / med-1)

> **GÜNCELLEME 2026-11-08 — TAM VERİ İNDİRİLDİ VE KULLANILDI.** Aşağıdaki "META VERİ TAM · GÖRÜNTÜ KISMİ" durumu 2026-11-07 19:27 anına aittir ve tarihsel olarak korunur.
> - Kuyruk `e2-veri` 3/3 bitti (2026-11-07 21:51; 8.058 s): tam zip indirildi → ön-işleme → piksel-NN (`data/bcn20000/bcn20000_nn_tam.json`).
> - K-a (sızıntı): RMSE ≤ 5 çifti **0** (min 7,67) → GEÇTİ. K-b: CSV MD5 eşleşti; E1 eşleme (AK→akiec, SCC dışla) kopyasız ana küme **11.933 görüntü / 3.451 lezyon** (çıkarma 49). Çerçeve kırpıcısı 2.210/12.413 (%17,8; cv2 4.12.0). Kalibrasyon RMSE 1,759. Ayrıntı ve npz SHA256: `data/bcn20000/e2_a0_rapor.md`.
> - Drive kopyası `yz/3-1/veri_dis/bcn20000/e2_girdi/` (SHA256 eşleşti). E2 değerlendirmesi 114/114 (2026-11-08 01:29); analiz bir kez (`sonuclar/e2_bcn.KILIT`).
> - Ortam eki: `opencv-python-headless==4.12.0.88` (`env/requirements.lock`).
> - Gizli BCN test etiketleri (ISIC Arşivi) KULLANILMADI (şef kararı: yazar gizlemesi + CC BY-NC).

Rol: E2 = DermaMNIST (resmî) ve DermaMNIST-C (lezyon-ayrık) üzerinde eğitilmiş mevcut kolların **yeniden eğitilmeden**
bir kez değerlendirileceği dış veri (KULLANICI KARARI 13, 2026-11-07). Bu belge yalnız veriyi kapsar. Koşum, model
ve test değerlendirmesi yapılmadı. Eşleme ve protokol kararları ön-kayıt kartına bırakıldı.
Sayıların tamamı `data/dis_veri_manifest.json` dosyasında. Ham ölçümler: `data/bcn20000/bcn20000_analiz.json`,
`data/bcn20000/indirme_kaydi_bcn.json`, `data/bcn20000/kalibrasyon_onisleme.json`.

Betik: `scripts/indir_bcn20000.py` (WS kökünden; GPU kullanmaz; tüm çıktılar `data/bcn20000/` altında)
```
PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I scripts/indir_bcn20000.py --asama meta         # figshare API+CSV (MD5), ISIC 2019 CSV, ISIC koleksiyon 249/212, Cassidy listesi
PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I scripts/indir_bcn20000.py --asama zipliste     # zip merkezi dizini HTTP Range ile (zip indirilmez)
PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I scripts/indir_bcn20000.py --asama ornek --n-ornek 40 --n-ham 8 --seed 0
PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I scripts/indir_bcn20000.py --asama kalibrasyon  # MedMNIST ön-işlemesini ölçerek belirleme
PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I scripts/indir_bcn20000.py --asama analiz
PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I scripts/indir_bcn20000.py --asama onisleme --kaynak ornek
# TAM VERİ (uzun, kuyrukta, BAŞLATILMADI): --asama indir · --asama onisleme --kaynak tam --isci 4 · --asama nn_tam
```
Tarih: 2026-11-07 (başlangıç 19:10:48+03:00, son analiz 19:27) · Hedef makine: A, yerel (GTX 1650; GPU kullanılmadı) ·
Ortam: `env/.venv` (Python 3.11.16, numpy 2.2.6, Pillow 12.3.0, scikit-image 0.26.0). Günlükler `data/bcn20000/*.log`.

> **DURUM: META VERİ TAM · GÖRÜNTÜ KISMİ.** Meta veri, zip içerik listeleri, 40 BCN örnek görüntüsü ve 8 HAM kalibrasyon
> görüntüsü indirildi ve denetlendi. **Tam görüntü arşivi (`BCN_20k_train.zip`, 6.368.901.290 B) İNDİRİLMEDİ.**
> Ölçülen figshare hızı **1,03 MB/s** (30 MB Range isteği 30,6 sn'de, 1.027.482 B/s). Bu hızla süre ≈ 103 dk, yani 5 dk
> eşiğini aşıyor. Bu nedenle `kuyruk/e2-veri.jsonl`'a 3 satır yazıldı (indir → ön-işleme → piksel-NN). **Başlatılmadı, şef başlatır.**
> Öneri için bkz. §İndirme yeri.

## Kaynak
| | değer |
|---|---|
| Makale | Hernández-Pérez C., Combalia M., Podlipnik S., … *BCN20000: Dermoscopic Lesions in the Wild.* **Sci Data 11, 641 (2024)**, doi:10.1038/s41597-024-03387-w (ilk sürüm arXiv:1908.02288, 2019) |
| Kalıcı depo (makalenin "Data Records" bölümü) | **figshare 10.6084/m9.figshare.24140028.v1** (yayım 2024-09-19). API yanıtı: `data/bcn20000/meta/figshare_24140028.json` |
| Merkez | Hospital Clínic de Barcelona. Dermoskopi, 1024×1024 JPEG |
| Resmî kod | GitHub `imatge-upc/BCN20000` @ `7ea35f5bf524b66f2ee98eef9428bba74f44547b` (CC-BY-4.0). Siyah çerçeve kırpıcı `preprocessing/image_cropper.py` (cv2) burada. `master_split_file.csv` train içi 5 katlı CV bölmesidir (62.065 satır), bizim kullanımımızı etkilemez |
| Çapraz denetim | ISIC Archive API v2: koleksiyon **249 "BCN20000"** (18.946 görüntü) ve **212 "HAM10000"** (11.720). ISIC 2019 eğitim CSV'leri (`isic-archive.s3.amazonaws.com/challenges/2019/`). Cassidy vd. 2022 kopya listesi (GitHub `mmu-dermatology-research/isic_duplicate_removal_strategy` @ `b785b75`) |

## Bütünlük (V2)
| dosya | durum | bayt (ölçülen = resmî) | MD5 ölçülen = figshare `computed_md5` | SHA256 (ölçülen) |
|---|---|---|---|---|
| `meta/bcn_20k_train.csv` | indirildi | 945.681 ✓ | `5fd3522a69512dcf5b3c225dc1664ec6` **eşleşti** | `0f219633f49c07b0a1c844c4faa0fa1fc26690743f2c44dc7337a7a4135bc40b` |
| `meta/bcn_20k_test.csv` | indirildi | 446.244 ✓ | `4917fd7e99a90a273cb4a853ce7fca2f` **eşleşti** | `703f7f81c09f9e1dcb870e8cd92929993ebe1d66116d8b4029e461c75c4a04f9` |
| `BCN_20k_train.zip` | **İNDİRİLMEDİ** (kuyrukta) | (resmî 6.368.901.290) | resmî `2293c9c654a3f76d6506a57ad17cddd1` | — |
| `BCN_20k_test.zip` | **indirilmeyecek** (etiketsiz, bkz. Split) | (resmî 3.327.235.230) | resmî `eeae0d7d675f1ce05250496815c94466` | — |

Figshare kaydının toplam boyutu 9.697.528.445 B. Yardımcı meta dosyaların (ISIC 2019 CSV'leri, ISIC koleksiyon JSONL'leri,
Cassidy listesi) URL'leri ve SHA256 değerleri `indirme_kaydi_bcn.json` içinde. Bu dosyaların resmî hash'i yok.

**Zip içeriği (zip indirilmeden, merkezi dizin HTTP Range ile okundu: train 5 istek / 1,31 MB, test 3 istek / 0,66 MB):**
| zip | dosya | uzantı | açık boyut toplamı | CSV ile karşılaştırma |
|---|---|---|---|---|
| train | 12.413 | .jpg 12.413 | 6.413.593.124 B (ort. 516.684 B/görüntü) | CSV 12.413 = zip 12.413, ortak 12.413 ✓ |
| test | 6.501 | .jpg 6.501 | 3.350.990.217 B | **CSV 6.533, zip 6.501 → 32 test satırının görüntüsü zip'te YOK** (liste `bcn20000_analiz.json` → `zip_csv.test.eksik_ornek`) |

Dosya bazında CRC32 listesi: `meta/zip_icerik_{train,test}.csv`. Tam indirmede zipfile her dosyanın CRC'sini doğrular,
zip'in tamamı ise figshare MD5'iyle doğrulanır.

## Split ve etiket durumu (V3, V4)
| bölüm | görüntü | lezyon | etiket | kaynak |
|---|---|---|---|---|
| train | **12.413** | 3.576 | **VAR** (8 sınıf; `diagnosis` boş satır 0) | resmî (figshare `split=train`) |
| test | 6.533 (zip'te 6.501) | 1.864 | **YOK** (figshare CSV'de `diagnosis` 6.533/6.533 boş) | resmî (figshare `split=test`) |

- **Kullanılabilir etiketli kısım yalnız figshare train'dir: 12.413 görüntü.** Makale de tanının *"exclusively provided for
  the training set"* olduğunu söylüyor. E2'de BCN "train" adı yanıltıcı olmamalı. Bu görüntüler bizim modellerin eğitiminde
  **hiç kullanılmadı**, E2 için tamamı dış test kümesidir.
- **V4 / kilit:** E2 yeniden eğitimsiz olduğu için BCN'den model seçimi ya da kalibrasyon yapılmamalı. Eşik, sıcaklık veya
  önsel düzeltmesi gibi bir ayar gerekiyorsa kart bunu önceden yazmalı ve ayarı yalnız DermaMNIST val'den almalı.
  BCN içinde val ayrılmadı, `data/splits/` altına BCN split'i yazılmadı.
- **ÖNEMLİ BULGU: ISIC Arşivi test tanılarını içeriyor.** Koleksiyon 249'daki 18.946 kaydın 12.413'ü ISIC 2019 eğitim
  BCN satırlarıyla aynı `isic_id`'ye sahip. Kalan **6.533** kayıt (lezyon 1.864, figshare test ile aynı sayılar) için ISIC
  Arşivi **5.377** görüntüde tanı veriyor (1.156'sında tanı alanı boş). Dağılım: Nevus 1.441 · Melanoma NOS 1.146 ·
  BCC 867 · AK 351 · Seboreik keratoz 339 + Solar lentigo 74 (=BKL 413) · SCC 128 · DF 44 · Vasküler 40 (8 sınıf toplamı
  **4.430**). Dağılım dışı (OOD): Melanoma metastazı 633 · Skar 314 · tanısız 1.156 (toplam **2.103**). Bu kısmın görüntü
  kaynağı ISIC'tir (figshare test zip'inin dosya adları ISIC kimliğine eşlenmiyor). Kullanılıp kullanılmayacağı
  **protokol kararıdır** (kullanıcı/kart): yazarlar test etiketlerini bilerek gizlemiş, ISIC lisansı CC BY-NC.
  Ayrıca bkz. Riskler.

## Dağılım (figshare train, etiketli 12.413)
| BCN sınıfı | görüntü | lezyon | ISIC Arşivi alt tanısı (aynı görüntüler) |
|---|---|---|---|
| NV | 4.206 | 1.281 | Nevus 4.206 |
| MEL | 2.857 | 524 | Melanoma, NOS 2.857 |
| BCC | 2.809 | 983 | Basal cell carcinoma 2.809 |
| BKL | 1.138 | 353 | Seborrheic keratosis 929 + Solar lentigo 209 |
| AK | 737 | 236 | Solar or actinic keratosis 737 |
| SCC | 431 | 122 | Squamous cell carcinoma, NOS 431 |
| DF | 124 | 40 | Dermatofibroma 124 |
| VASC | 111 | 37 | Benign soft tissue proliferations – Vascular 111 |
| dengesizlik (max/min) | 37,89 | — | — |

- Birden çok tanılı lezyon: 0. "Diğer/unknown" sınıfı train'de yok (yalnız test'te var).
- **Lezyon başına görüntü (train):** ortalama 3,47, en çok 31. Dağılım: 1 görüntü 488 lezyon, 2 görüntü 1.069, 3 görüntü 815,
  4 görüntü 433, 5 görüntü 243, 6 görüntü 194, 7 görüntü 112, 8 görüntü 59, 9 görüntü 54, ≥10 görüntü 109 lezyon.
  Görüntüler bağımsız değil. Güven aralığı için lezyon kümeli bootstrap gerekir (HAM10000'de ortalama 1,34 idi).
- Tarih (`capture_date`): train 2011-01-01 – 2016-12-22, test aynı aralık (boş 0/0). Cinsiyet: erkek 6.499, kadın 5.840, boş 74.
  Bölge: gövde ön 5.086, baş/boyun 3.223, alt ekstremite 2.173, üst ekstremite 1.319, avuç/taban 380, oral/genital 59, boş 173.
- Görüntü boyutu: ISIC meta verisinde 18.946/18.946 görüntü 1024×1024 ve hepsi `dermoscopic`. 40 örnekte ölçülen 40/40 1024×1024.
  Figshare JPEG'leri ortalama 516.684 B. Aynı görüntülerin ISIC kopyaları ortalama 61.047 B (yeniden sıkıştırılmış).
- **Önsel kayması:** DermaMNIST'te nv payı %66,9 (test), BCN train'de %33,9–37,4 (eşleme seçeneğine göre, bkz. aşağıdaki tablo).
  "Hep nv" ACC tabanı DermaMNIST'te 0,669, BCN'de 0,339–0,374. ACC iç/dış arasında doğrudan kıyaslanamaz (bkz. Riskler).

## Sınıf eşlemesi — SEÇENEKLER (karar ön-kayıt kartının)
HAM10000/DermaMNIST tanımları: `akiec` = *Actinic keratoses and intraepithelial carcinoma (Bowen's disease)*, yani in situ.
`bkl` = seboreik keratoz + solar lentigo + liken planus benzeri keratoz.

| BCN | DermaMNIST | tür | görüntü / lezyon | not |
|---|---|---|---|---|
| NV | nv (5) | birebir | 4.206 / 1.281 | |
| MEL | mel (4) | birebir | 2.857 / 524 | |
| BCC | bcc (1) | birebir | 2.809 / 983 | |
| BKL | bkl (2) | birebir (alt tür kapsamı hafif farklı) | 1.138 / 353 | BCN'de LPLK yok (SK 929 + SL 209) |
| DF | df (3) | birebir | 124 / 40 | |
| VASC | vasc (6) | birebir | 111 / 37 | |
| AK | akiec (0) | yakın (akiec'in AK bileşeni) | 737 / 236 | |
| SCC | akiec (0) **ya da dışla** | **belirsiz** | 431 / 122 | BCN SCC "NOS", invaziv SCC'yi de kapsar. HAM akiec yalnız in situ (Bowen). |

Seçeneklerin sayıları (`bcn20000_analiz.json` → `esleme_secenekleri_figshare_train`):
| seçenek | görüntü | lezyon | dışlanan (eşlenemeyen) | akiec | nv payı | kopyasız sürüm* |
|---|---|---|---|---|---|---|
| **E1** AK→akiec, SCC dışla (7 sınıf) | **11.982** | 3.454 | 431 | 737 | 0,3510 | 11.933 / 3.451 |
| **E2** AK+SCC→akiec (7 sınıf) | **12.413** | 3.576 | 0 | 1.168 | 0,3388 | 12.364 / 3.573 |
| **E3** AK ve SCC dışla (6 sınıf; akiec boş) | **11.245** | 3.218 | 1.168 | 0 | 0,3740 | 11.204 / 3.215 |

\*Kopyasız: 47 ikili-özdeş kopya fazlası ve 4 çelişkili etiketli görüntü çıkarılmış hâli (bkz. Sızıntı). E3'te akiec sınıfı
boş kalır, bu yüzden 7 sınıflı makro AUC/dengeli ACC tanımı değişir. Kart, metriğin hangi sınıf kümesinde hesaplanacağını yazmalı.
ISIC test kısmı kullanılırsa (bkz. Split) eşlenebilir 4.430 görüntü (SCC 128 dahil), eşlenemeyen OOD 2.103 (melanoma
metastazı 633 → mel'e eşlenmesi de ayrıca bir karardır).

## Ön-işleme önerisi (MedMNIST ile uyumlu; ÖLÇÜLEREK belirlendi)
**Kalibrasyon** (`--asama kalibrasyon`): DermaMNIST **train** bölümünden seed 0 ile seçilen 8 HAM görüntüsü ISIC API'den
orijinal hâliyle (600×450) indirildi. Aday ön-işlemelerle 28/224 px üretilip resmî `dermamnist.npz` / `dermamnist_224.npz`
(MD5 `8974907d…` = resmî) train görüntüsüyle karşılaştırıldı (RMSE, 0–255).

| boyut | en iyi | RMSE ort. (maks) | 2. sıradaki | gerdirme (600×450→kare) en iyisi |
|---|---|---|---|---|
| 28 | **merkez kare kırpma 450×450 + `skimage.transform.resize(order=1, anti_aliasing=True)`** | **4,40** (6,05) | aynı kırpma + PIL BILINEAR 4,58 | 14,37 |
| 224 | **merkez kare kırpma 450×450 + PIL `BILINEAR`** | **1,76** (2,12) | skimage o1 aa 1,89 | 17,18 |

- **MedMNIST görüntüyü gerdirmiyor, ortadaki kareyi kırpıyor.** 224 px'te kırpma ofseti tarandı (yatay sol kenar 70/73/75/77/80
  → RMSE 6,79/3,85/**1,76**/3,84/6,78; oturum scratchpad betiği, sonuç bu satırda): tam merkez (75) açıkça en iyisi.
  Kalan RMSE muhtemelen ISIC JPEG'i ile MedMNIST'in kullandığı kaynak dosya arasındaki çözücü/kodlama farkından geliyor. Bu doğrulanmadı.
  4,40'lık değer, ana manifestteki "RMSE ≤5 = aynı görüntü" eşiğinin altında kalıyor.
- **BCN için öneri (birincil, MedMNIST'e sadık):** 1024×1024 görüntüde merkez kare kırpma özdeşliktir. Yani doğrudan
  28 px'e skimage o1 aa, 224 px'e PIL BILINEAR ile küçültülür. Ardından her kolun eğitimdeki normalizasyonu (ImageNet/DINOv2
  ortalama-std, ya da resmî 28 px kolunun dönüşümü) değerlendirme kodunda aynen uygulanır. Bu adım veri dosyasında değil, koddadır.
  Uygulama: `--asama onisleme` (varsayılan `--geometriler merkez_kare`).
- **İkincil seçenek (duyarlılık analizi, kart karar verir):** yazarların siyah çerçeve kırpıcısı (`image_cropper.py`
  @7ea35f5; cv2 gerekir, uygulanmadı). 40 örnekte 28 px köşe ortalaması <40 olan görüntü oranı **0,40**. DermaMNIST train'de
  aynı ölçü **0,0004**. Örnek sınıf-tabakalı olduğu için temsilî değil. Makale ~%20 diyor, tam veri oranını `nn_tam` aşaması ölçer.
  Görsel denetimde (yalnız train görüntüleri) BCN'de dermatoskop açıklığının oluşturduğu dairesel siyah çerçeve ve daha küçük
  görünen lezyonlar belirgin. HAM'de bu yok. Bu, ön-işlemeden bağımsız bir alan kaymasıdır.
- Doğrulama örneği yazıldı (40 görüntü, sınıf başına 5, seed 0):
  `data/bcn20000/ornek/bcn20000_ornek_merkez_kare_28.npz` (40×28×28×3; SHA256 `67838684ee608e8aea5ba3f8eb0cd25219e7237e33c28a3b50e2aab6e54aa86d`),
  `…_224.npz` (40×224×224×3; SHA256 `76b73bdebcecffdf8bfe6bfbaa044d321a35368efc14303745be24b01c044fcc`). npz anahtarları:
  `images` uint8, `bcn_filename`, `diagnosis` (ham BCN kodu; eşleme değerlendirme kodunda uygulanır), `lesion_id`.
  Paralel tam ön-işleme yolu 8 görüntülük duman zip'iyle sınandı: seri çıktıyla bit düzeyinde aynı (28 ve 224).

## Sızıntı / çakışma denetimi (V5) — BCN20000 ↔ HAM10000/DermaMNIST
| denetim | sonuç |
|---|---|
| Kaynak merkez | HAM10000 `dataset` alanı: vidir_modern 3.363, vidir_molemax 3.954, vienna_dias 439, rosendahl 2.259 (Viyana + Queensland). ISIC 249 atfı: **"Hospital Clínic de Barcelona" 18.946/18.946**. ISIC 212 atfı: "MILK study team" 11.720/11.720 (ISIC'in kayıtlı değeri) |
| ISIC görüntü kimliği: koleksiyon 249 ∩ 212 | **0** |
| ISIC görüntü kimliği: 249 ∩ DermaMNIST 10.015 `image_id` | **0** (DermaMNIST'in 10.015/10.015'i koleksiyon 212'de) |
| ISIC lezyon kimliği (ortak `IL_` ad alanı): 249 ∩ 212 | **0** (BCN 5.440 lezyon) |
| Hasta kimliği | **denetlenemedi**: `patient_id` alanı 249'da 0/18.946, 212'de 0/11.720 dolu. Figshare CSV'de de yok. Coğrafi ayrık merkezler hasta çakışmasını olanaksıza yakın kılıyor ama bu ölçülmedi |
| ISIC 2019 eğitim CSV'si (25.331 satır) | `lesion_id` öneki: BCN_ 12.413 · HAM_ 10.015 · MSK4 819 · boş 2.084. BCN_ ∩ HAM_ lezyon **0**, görüntü **0**. DermaMNIST 10.015 görüntüsü: HAM_ satırlarında 10.015, **BCN_ satırlarında 0**, BCN_ lezyonlarında 0 |
| Figshare train = ISIC 2019 BCN kısmı mı? | Evet. 12.413 = 12.413. Lezyon kimlikleri 3.576/3.576 aynı, sınıf sayıları birebir aynı. **ISIC 2019'dan HAM satırları dışlanmış durumda, çünkü kaynak doğrudan figshare BCN** |
| İkili-özdeş (bayt) kopya HAM↔BCN | Yapısal olarak olanaksız: HAM 600×450, BCN 1024×1024 |
| Piksel-NN (28 px, kalibre ön-işleme, DermaMNIST 10.015 görüntüye karşı) | **40 örnek: RMSE ≤5 → 0, ≤10 → 0, ≤15 → 3**. NN RMSE en küçük 13,05, medyan 26,49 (ana manifestte train→train NN medyanı 16,01). **Tam 12.413 görüntü için `nn_tam` bekliyor (kuyrukta)** |

**Sonuç: HAM10000/DermaMNIST ile çakışan görüntü 0, çakışan lezyon 0 (ISIC kimlikleri). Hasta düzeyi denetlenemedi.
Piksel düzeyi 0/40, tam veri bekliyor.** Not: ana manifestteki bulgu, aynı lezyonun farklı çekimlerinin 28 px piksel-NN'de
çoğunlukla görünmediğiydi. Piksel denetimi kopya yakalar, lezyon tekrarını yakalamaz. BCN↔HAM lezyon ayrıklığının kanıtı
ISIC kimlikleri ve merkez ayrımıdır.

**BCN içi (dış test kümesinin kendi temizliği):**
- Zip CRC32+boyut ile ikili-özdeş kopya (görüntü indirmeden): **train'de 47 grup / 47 fazla görüntü**. Bunların 37'si aynı
  lezyon, **10'u farklı lezyon kimliği**. Farklı kimlikli gruplar 3 lezyon çiftinden geliyor: BCN_0002049↔BCN_0002919 (NV, 5 grup),
  BCN_0004816↔BCN_0001604 (BKL, 3 grup), **BCN_0003560 (MEL) ↔ BCN_0000237 (NV), 2 grup: aynı görüntüye çelişkili etiket**
  (BCN_0000014439/14930, BCN_0000015995/18062). Test'te 0 grup. train∩test CRC 0.
- Bağımsız teyit: Cassidy vd. 2022'nin (Med Image Anal 75:102305) "all train duplicates" listesinde ISIC 2019 BCN
  görüntülerinden silinen **47**. Sayı birebir aynı, görüntü-görüntü eşlemesi yapılmadı.
- figshare train∩test: dosya adı 0, `lesion_id` 0 (biçimler farklı: `BCN_0003884` / `BCN4657`; rakam normalize edilince de 0),
  ISIC `IL_` lezyon 0. Yazarların bölmesi lezyon-ayrık.
- Bölme türü: lezyon bazlı (yazarların). Hasta bazlı olup olmadığı bilinmiyor.

## Lisans ve etik (V6)
- **Figshare (kullanılacak kaynak): CC BY 4.0** (API `license.name = "CC BY 4.0"`, makale aynı). Ticari kısıt yok.
  **Atıf zorunlu:** Hernández-Pérez vd., Sci Data 11:641 (2024) + figshare DOI 10.6084/m9.figshare.24140028. Değişiklik
  (yeniden boyutlandırma, sınıf eşleme) yapıldığı belirtilmeli. Önerilen ek atıf: Combalia vd. arXiv:1908.02288 (2019).
- **ISIC Arşivi kopyası: "CC-BY-NC"** (18.946/18.946 kayıt). Görünür lisans çelişkisi var. ISIC kopyası (ör. test tanıları)
  kullanılırsa NC geçerli olur ve atıf "Hospital Clínic de Barcelona / ISIC Archive" biçiminde yapılır. Çalışma zaten
  DermaMNIST/HAM10000 (CC BY-NC 4.0) nedeniyle ticari değil, yani pratikte ek kısıt doğmuyor.
- Resmî kod CC-BY-4.0. Cassidy listesi yalnız kimlik denetimi için okundu.
- **Klinik iddia yok:** E2 bir araştırma genellenebilirlik ölçümüdür. Sonuçlar tanısal veya klinik performans iddiası taşımaz.
- Kişisel veri: anonim. Meta veride yaş (5 yıllık aralık), cinsiyet, bölge ve çekim tarihi var, isim veya kimlik yok.
  Kullanılan alanlar `bcn_filename`, `diagnosis`, `lesion_id`.

## İndirme yeri — öneri
| seçenek | süre (ölçülen/tahmini) | artı | eksi |
|---|---|---|---|
| **A. Colab (önerilen)**: `/content`'e indir, MD5 doğrula, `onisleme` + `nn_tam`, npz'leri Drive'a yaz | indirme **ölçülmedi** (R2/Cloudflare, büyük olasılıkla dakikalar). Ön-işleme 28 px skimage ≈0,42 sn/görüntü/çekirdek (yerelde ölçüldü) | Ağırlıklar ve E2 koşumu zaten Colab/Drive'da. 1,9 GB'lık 224 npz'nin yerelden yüklenmesi gerekmez | Bağlantı ilk hücrede ölçülmeli. Betik WS göreli yollar kullanır, Colab paketine `scripts/indir_bcn20000.py` + `data/bcn20000/meta/` + `kalibrasyon_onisleme.json` + `dermamnist.npz` girmeli |
| **B. Yerel kuyruk `e2-veri`** (hazır, 3 satır) | indirme ≈ 103 dk (1,03 MB/s) + ön-işleme ≈ 25–30 dk (4 süreç) + NN < 1 dk ≈ **2,2 sa**, yalnız CPU | Hazır, gözetimsiz koşar, GPU'yu meşgul etmez | Sonra 224 npz'nin (≈1,9 GB) Drive'a yüklenmesi gerekir. Disk: zip 6,4 GB + npz ≈ 2 GB (C: 83 GB boş) |
| C. ISIC Arşivi (görüntü başına API) | train kısmı 757.823.281 B, 12.413 istek (ardışık istek 0,7–2,5 sn) | Küçük dosyalar, test tanıları da burada | CC BY-NC. Yeniden sıkıştırılmış JPEG (figshare'den farklı pikseller). Kalıcı depo değil |

Öneri: **A**. Colab'ın ilk hücresi 30 MB Range ile hız ölçsün, <10 MB/s ise B'ye dönülsün. Kuyruk satırları
(`kuyruk/e2-veri.jsonl`): `veri_bcn_indir` (sınır 240 dk) → `veri_bcn_onisleme` (`--isci 4`, 120 dk) → `veri_bcn_nn_tam` (20 dk).
Hepsi `--asama 3`, GPU'suz, `{RUN_DIR}/metrics.json` yazar. Beklenti dosyası yazılmadı (veri işi). `--kademeli`siz koşulmalı
ya da şef beklenti dosyası eklemeli.

## Literatürle uyum
- Görüntü sayısı 18.946 (makale) = CSV 12.413 + 6.533 = ISIC 249 kaydı 18.946 ✓. Train 12.413 / test 6.533 ✓.
- Train sınıf sayıları (NV 4.206 … VASC 111) makale Tablo 1 ile **birebir** ✓ ve ISIC 2019 BCN kısmıyla birebir ✓.
- **Lezyon sayısı uyumsuz:** makale 5.583, ölçülen **5.440** (figshare 3.576 + 1.864. ISIC `IL_` kimlikleri de 5.440). Fark 143, açıklanamadı.
- **Tarih:** makale "2010–2016", CSV 2011-01-01 – 2016-12-22. 2011-01-01 bir yer tutucu değer olabilir, denetlenmedi.
- **Test OOD:** makaledeki OOD alt grupları (metastatik/primer olmayan melanom 632, benign neoplazm 422, skar 395, enflamatuvar
  375, normal varyant 116, enfeksiyöz 68, diğer 32; toplamı 2.040, toplama bizim) ile ISIC'teki ölçüm (metastaz 633, skar 314,
  tanısız 1.156 = 2.103) tam örtüşmüyor. Etiketsiz test kullanılmayacaksa E2'yi etkilemez.
- **Test zip'i CSV'den 32 görüntü eksik** (6.501 / 6.533). Makalede bundan söz edilmiyor.
- Siyah çerçeve oranı: makale ~%20. Tam veri ölçümü bekliyor (örnek temsilî değil).

## PAD-UFES-20 (yedek aday, yalnız değerlendirme)
Mendeley Data doi:10.17632/zr7vgbcyr2.1 (2020), **CC BY 4.0**. 2.298 **akıllı telefon klinik fotoğrafı** (PNG, farklı boyutlar;
**dermoskopi DEĞİL**), 1.373 hasta, 1.641 lezyon. Görüntüler 3 zip'te toplam 3.592.396.763 B. `metadata.csv` (316.209 B,
SHA256 `14d14523…` Mendeley değeriyle eşleşti; oturum scratchpad'ine indirildi, WS'ye yazılmadı). Sınıflar: BCC 845 ·
ACK 730 · NEV 244 · SEK 235 · SCC 192 (Bowen dahil) · MEL 52. DermaMNIST ile örtüşme: bcc, mel, nv, bkl (SEK) ve akiec
(ACK; SCC+Bowen belirsiz). **df ve vasc yok**, yani 7 sınıfın 5'i kapsanıyor. Artısı `patient_id` olması (hasta düzeyi
denetim mümkün). Eksileri: modalite farkı (klinik ↔ dermoskopi) dermoskopiyle eğitilmiş modeller için çok sert bir kayma
yaratır, MEL 52 görüntüyle melanom ölçütlerinin güven aralığı çok geniş olur. **İkincil/yedek olarak uygun, birincil dış test için önerilmez.**

## Riskler ve belirsizlikler
1. **Tam görüntüler indirilmedi.** Tam veri piksel-NN çakışma denetimi (`nn_tam`), tam veri siyah çerçeve oranı ve
   12.413 görüntünün boyut denetimi kuyruğa bağlı. E2 koşumundan ÖNCE bu sonuç manifeste eklenmeli (kapı).
2. **ISIC'teki test tanıları:** 5.377 ek etiketli görüntü (4.430'u eşlenebilir) dış testi büyütür. Ancak yazar niyetine
   (gizli test) ve ISIC 2019 değerlendirmesine aykırı olabilir, ayrıca CC BY-NC'dir. Kullanım kararı kullanıcıya ait.
   Görüntü kaynağı ISIC kopyası olur (farklı JPEG kodlaması).
3. **SCC eşlemesi** (E1/E2/E3) ACC ve dengeli ACC'yi doğrudan değiştirir. akiec'in 431 invaziv SCC ile şişirilmesi (E2)
   etiket kayması yaratır. E3 metriğin sınıf kümesini değiştirir.
4. **Önsel kayması** (nv %67 → %34–37): ACC iç/dış arasında kıyaslanamaz. Kıyas için dengeli ACC ve makro AUC daha
   anlamlı. Seçim kartın işi.
5. **Lezyon başına çok görüntü** (ort. 3,47, en çok 31): görüntü bazlı güven aralıkları dar çıkar. Lezyon kümeli bootstrap
   ya da lezyon düzeyi toplama kartta tanımlanmalı.
6. **Etiket gürültüsü:** 47 ikili kopya, 4 görüntüde çelişkili etiket (MEL/NV). "Ham" ve "kopyasız" sürümlerden hangisinin
   ana olacağı kartta yazılmalı (fark 49–51 görüntü).
7. **Alan kayması:** siyah dairesel çerçeve (örnekte %40, DermaMNIST'te %0,04), 1024² kare ile 600×450 kırpması arasındaki
   ölçek farkı, farklı cihazlar. Bu kayma E2'nin ölçmek istediği şeydir. Yine de çerçeve kırpma duyarlılık analizi kartta önceden yazılmalı.
8. Kalibrasyonda kalan RMSE (28 px 4,40, 224 px 1,76) sıfır değil. MedMNIST'in tam kodu (kaynak dosya, çözücü) elde yok.
   Seçilen yöntem adaylar arasında açık ara en iyisi (gerdirme 14–17).
9. Lisans çelişkisi (figshare CC BY 4.0 ↔ ISIC CC BY-NC): figshare kopyası kullanıldığında CC BY 4.0 geçerli.
   Çalışma zaten NC.
10. Hasta düzeyi BCN↔HAM ayrıklığı hiçbir kaynakta ölçülemiyor (hasta kimliği yok). Merkezler coğrafi olarak ayrık.
