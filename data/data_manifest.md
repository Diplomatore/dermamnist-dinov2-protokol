# Veri Manifesti — DermaMNIST (MedMNIST v3.0, Zenodo 10.5281/zenodo.10519652) (3.1 / med-1)

İndirme komutu (WS kökünden):
`PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe scripts/veri_hazirla.py --asama indir --boyutlar 28`
(içeride: `medmnist.DermaMNIST(split="train", size=28, root="data/medmnist", download=True)` → torchvision
`download_url` + resmî MD5 kontrolü; metadata CSV'leri `urllib` ile sabit URL'lerden)
Analiz komutu: `PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe scripts/veri_hazirla.py --asama analiz --boyutlar 28`
(süre 31,6 sn; çıktı `data/data_manifest.json`, günlük `data/analiz.log`)
Tarih: 2026-10-10 (indirme 07:08:42+03:00 başladı; 28 px 193,0 sn) · Hedef makine: A — yerel (GTX 1650; GPU kullanılmadı)
Ortam: `env/.venv` — Python 3.11.16, medmnist 3.0.2, numpy 2.2.6, torchvision 0.20.1+cu121, pandas 3.0.6, Pillow 12.3.0

> **DURUM: KISMİ.** 28 px indirildi ve tam denetlendi. **64/128/224 px İNDİRİLMEDİ.** Zenodo hızı ölçümde
> 28 px için ortalama 102 KB/s idi (19.725.078 B / 193,0 sn; anlık 23–530 KB/s). Bu hızla tahmini süreler:
> 64 px ≈ 16 dk, 128 px ≈ 61 dk, 224 px ≈ 178 dk; toplam ≈ 4,3 saat. Talimattaki "≫5 dk → dur ve bildir"
> kuralı gereği 64 px indirmesi başladıktan sonra durduruldu ve 262.144 baytlık yarım dosya silindi
> (`data/indirme_kaydi.json`). Devam etmek için tek komut yeterli (arka planda; mevcut 28 px MD5 ile atlanır):
> `PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe scripts/veri_hazirla.py --asama hepsi --boyutlar 28 64 128 224`
> Bu komut boyutlar arası sıra/etiket tutarlılığını da ölçer (büyük boyut → 28'e bicubic küçültme, aynı indeks RMSE).
> `download_url` kaldığı yerden devam edemez; kesinti olursa dosya baştan iner.

## Bütünlük
| dosya | durum | bayt | SHA256 (ölçülen) | MD5 ölçülen = resmî (`medmnist.INFO` ve Zenodo API) |
|---|---|---|---|---|
| `data/medmnist/dermamnist.npz` (28 px) | indirildi | 19.725.078 | `1a309fec2e33bb6aba88e7d078e5ccbb9736c84a0b415ac197eb3c8fa331e050` | `0744692d530f8e62ec473284d019b0c7` — **eşleşti** |
| `dermamnist_64.npz` | **indirilmedi** | (Zenodo: 100.130.190) | — | resmî `b70a2f5635c6199aeaa28c31d7202e1f` |
| `dermamnist_128.npz` | **indirilmedi** | (Zenodo: 372.579.947) | — | resmî `2defd784463fa5243564e855ed717de1` |
| `dermamnist_224.npz` | **indirilmedi** | (Zenodo: 1.091.112.502) | — | resmî `8974907d8e169bef5f5b96bc506ae45d` |

Zenodo boyut ve MD5 değerleri `curl https://zenodo.org/api/records/10519652` çıktısından alındı.

Yardımcı metadata (yalnız sızıntı denetimi için; görüntü içermez):
| dosya | kaynak | bayt | SHA256 |
|---|---|---|---|
| `data/meta/HAM10000_metadata.csv` | Harvard Dataverse doi:10.7910/DVN/DBW86T, v4.0, dosya id 4338392 (`/api/access/datafile/4338392?format=original`) | 690.218 | `de17ec44cb25ea9c3f6378ba18d69224c740afd62e86ec2edc6a894d6aa663d3` |
| `data/meta/dermamnist_split_info.csv` | Abhishek, Jain, Hamarneh (2025), GitHub `kakumarabhishek/Corrected-Skin-Image-Datasets` @ `ac655ac5a5b93264c57d69d6d520778d915bfc89`. İçerik: DermaMNIST `split,index` → HAM10000 `image_id`; dosya adları MedMNIST yazarlarından alınmıştır | 243.056 | `0c9b1719bf04fa9ad3cdee2a852067a8b848795624e88ea794df40fd13c099f0` |

Eşlemenin doğrulanması: 10.015/10.015 `image_id` HAM10000 metadata'sıyla eşleşti (eşleşmeyen 0). HAM10000 `dx`
kodundan türetilen etiket (akiec=0, bcc=1, bkl=2, df=3, mel=4, nv=5, vasc=6) ile npz etiketi **10.015/10.015**
örnekte aynı. Birleştirilmiş tablo: `data/splits/dermamnist_lezyon_esleme.csv` (split, index, image_id,
lesion_id, dx; 10.015 satır). Bu dosya **yeni bir split değildir**, yalnız resmî split'in lezyon etiketlemesidir.

npz içeriği: `train/val/test_images` uint8 (N, 28, 28, 3), piksel aralığı [0, 255]; `*_labels` uint8 (N, 1).

## Split
| bölüm | örnek | beklenen (`INFO['dermamnist']['n_samples']`) | kaynak |
|---|---|---|---|
| train | 7.007 | 7.007 ✓ | resmî (MedMNIST, 7:1:2) |
| val | 1.003 | 1.003 ✓ | resmî |
| test | 2.005 | 2.005 ✓ | resmî |
| toplam | 10.015 | = HAM10000 görüntü sayısı | — |

- **V3:** Resmî split kullanılır, dosyaya yeniden bölme yapılmadı ve `data/splits/` altında seed'li split **yok**.
  MedMNIST+ boyutları (64/128/224) aynı split'i paylaşır (MedMNIST+ beyanı). İndirildiğinde betik bunu indeks bazında doğrulayacak.
- **V4:** Resmî validation kümesi var (1.003). **Ayrıca ayırma yapılmadı.** Model ve hiperparametre seçimi
  yalnız val ACC (eşitlikte val AUC) ile yapılır. Test yalnız K5'te bir kez açılır (konu.md).
- Resmî açıklama (`medmnist.INFO`): *"We split the images into training, validation and test set with a ratio of
  7:1:2. The source images of 3×600×450 are resized into 3×28×28."* Lezyon/hasta bazlı bölmeden söz edilmiyor.

## Dağılım (28 px, sınıf sayıları)
| # | sınıf (MedMNIST adı, HAM `dx`) | train | val | test | toplam |
|---|---|---|---|---|---|
| 0 | actinic keratoses and intraepithelial carcinoma (akiec) | 228 | 33 | 66 | 327 |
| 1 | basal cell carcinoma (bcc) | 359 | 52 | 103 | 514 |
| 2 | benign keratosis-like lesions (bkl) | 769 | 110 | 220 | 1.099 |
| 3 | dermatofibroma (df) | 80 | 12 | 23 | 115 |
| 4 | melanoma (mel) | 779 | 111 | 223 | 1.113 |
| 5 | melanocytic nevi (nv) | 4.693 | 671 | 1.341 | 6.705 |
| 6 | vascular lesions (vasc) | 99 | 14 | 29 | 142 |
| | dengesizlik oranı (max/min) | 58,66 | 55,92 | 58,30 | — |

- Çoğunluk sınıfı (nv) payı: train %66,98, test %66,88 (1.341/2.005). Yani "hep nv" tahmini test ACC'si 0,669.
  ACC çıpası 0,735 bu tabanla birlikte okunmalı. Sınıf oranları üç bölümde de neredeyse aynı, yani bölme tabakalı görünüyor.
- HAM10000 lezyon başına görüntü sayısı: 1 görüntü 5.514 lezyon, 2 görüntü 1.423, 3 görüntü 490, 4 görüntü 34,
  5 görüntü 5, 6 görüntü 4 (toplam 7.470 lezyon, 10.015 görüntü). Birden çok görüntüsü olan 1.956 lezyon toplam 4.501 görüntüye sahip (10.015 − 5.514).

## Sızıntı denetimi (V5) — resmî split DEĞİŞTİRİLMEDİ, yalnız ölçüldü
**(a) Tam-piksel kopya (28 px, görüntü baytlarının SHA1 özeti):**
- train∩val: **0** · train∩test: **0** · val∩test: **0** (çakışan örnek: 0)
- Split içi kopya: train'de 1 çift (idx 2815 ↔ 6521; ISIC_0024366 ↔ ISIC_0029861, aynı lezyon HAM_0002300, etiket nv);
  val'de 1 çift (idx 542 ↔ 696; ISIC_0024777 ↔ ISIC_0029938, aynı lezyon HAM_0001895, nv); test'te 0.

**(b) Lezyon kimliği (`lesion_id`) çakışması — ASIL BULGU: MedMNIST bölmesi lezyon bazlı DEĞİL**
| çift | ortak lezyon | iki bölümdeki toplam görüntü | sızan görüntü (ikinci bölümde) |
|---|---|---|---|
| train∩val | 332 | 769 | val'de 348 |
| train∩test | 641 | 1.518 | test'te 693 |
| val∩test | 113 | 241 | test'te 120 |
| üç bölümde birden | 40 lezyon / 130 görüntü | — | — |

- Birden fazla bölümde bulunan lezyon: **1.006 / 7.470 (%13,47)**.
- **Test görüntülerinin %34,56'sının (693/2.005) lezyonu train'de de var.** Val için oran %34,70 (348/1.003).
  Val de aynı biçimde sızıntılı olduğundan val ile yapılan model seçimi de lezyon ezberini ödüllendirebilir.

**(c) Yakın-kopya (28 px; en yakın komşu):** Ham piksel RMSE (0–255 ölçeği, 2.352 boyut) ve 64-bit dHash
(gri, 9×8, Hamming). Referans olarak train→train (kendisi hariç) dağılımı da verildi.
| sorgu→ref | RMSE NN medyan (p1 / min) | RMSE ≤5 | RMSE ≤10 (aynı lezyon) | RMSE ≤15 | dHash =0 (aynı lezyon) | dHash ≤2 (aynı lezyon) | NN aynı lezyon |
|---|---|---|---|---|---|---|---|
| test→train | 16,09 (8,00 / 2,54) | 7 (7'si aynı lezyon) | 90 (32) | 822 | 5 (2) | 76 (10) | 95 |
| val→train | 15,80 (7,18 / 4,39) | 4 (4'ü aynı lezyon) | 48 (19) | 434 | 6 (2) | 36 (6) | 49 |
| test→val | 18,36 (9,92 / 3,90) | 1 (1) | 22 (7) | 465 | 0 (0) | 29 (0) | 19 |
| train→train (ref.) | 16,01 (7,93 / 0,00) | 15 | 301 | 2.875 | 15 | 240 | 350 |

- **Piksel düzeyinde yakın-kopya denetimi lezyon sızıntısının çoğunu GÖRMÜYOR.** Lezyonu train'de olan 693 test
  görüntüsünün yalnız %13,71'inde piksel-NN aynı lezyonu buluyor (val→train %14,08, test→val %15,83).
  Sızan ve sızmayan test görüntülerinin NN-RMSE medyanı neredeyse aynı (16,81 ve 15,78). Aynı lezyonun farklı
  çekimleri (yakınlaştırma, döndürme, aydınlatma) 28 px'te ayrık görünüyor. Sızıntının doğru ölçüsü (b)'deki lezyon kimliğidir.
- RMSE ≤5 eşiği burada kesin bir kopya göstergesi: bu eşiğin altındaki 12 çiftin 12'si de aynı lezyon. dHash 28 px'te ayırt
  edici değil: dHash=0 olan 11 test/val sorgusunun yalnız 4'ü aynı lezyon. Eşik için **RMSE ≤5** önerilir, dHash önerilmez.
- Görsel kontrol yalnız val↔train için yapıldı (test görüntülerine bakılmadı): `data/raporlar/en_yakin_16_val_train.png`
  ve `.csv`. En yakın 16 çiftin 13'ü aynı lezyon. Kalan 3 çift (val 809↔train 2574, 76↔6858, 11↔981) görsel olarak
  çok benzer ama metadata'ya göre farklı lezyon. Bunlar Abhishek vd.'nin 18 "yeni kopya" listesinde de yok ve
  birinin etiketleri farklı (nv/bkl). Bkz. Belirsizlikler.

**Kaynak (araştırma):** Abhishek, Jain, Hamarneh, *"Investigating the Quality of DermaMNIST and Fitzpatrick17k
Dermatological Image Datasets"*, Scientific Data 12, 196 (2025), doi:10.1038/s41597-025-04382-5 (arXiv:2401.14497).
Makaleye göre lezyon bazlı bölmenin DermaMNIST hazırlanırken *"not accounted for"* olduğu belirtiliyor. Yazarlar
dosya adlarını MedMNIST yazarlarından alıp HAM10000 metadata'sıyla birleştirmiş ve düzeltilmiş **DermaMNIST-C**
(8.215 / 573 / 1.227) ile **DermaMNIST-E** sürümlerini yayımlamış (Zenodo 10.5281/zenodo.12739457; 28 ve 224 px).
**Protokol kararı (resmî split ile mi, ek olarak DermaMNIST-C ile mi raporlanacağı) kullanıcıya aittir.**
Bu manifest resmî split'i değiştirmez.

## Lisans ve etik (V6)
- **DermaMNIST: CC BY-NC 4.0 (ticari kullanım YASAK).** medmnist.com: *"The MedMNIST dataset is licensed under
  CC BY 4.0 … except DermaMNIST under CC BY-NC 4.0"*. Paket içi `INFO['dermamnist']['license']` = `"CC BY-NC 4.0"`.
  Not: Zenodo kaydı 10519652 bütün olarak `cc-by-4.0` gösteriyor (API). DermaMNIST için belirleyici olan
  medmnist.com/INFO istisnasıdır ve kaynak verinin NC lisansından gelir.
- **Kaynak HAM10000: CC BY-NC 4.0.** Harvard Dataverse (doi:10.7910/DVN/DBW86T, v4.0) kullanım şartları:
  "Creative Commons Attribution-NonCommercial 4.0 International Public License".
- MedMNIST kodu: Apache-2.0. Abhishek vd. metadata CSV'leri CC BY 4.0, kodları Apache-2.0.
- **Atıf gereği:** MedMNIST v2 (Yang vd., Scientific Data 2023) + MedMNIST Decathlon (Yang vd., ISBI 2021) +
  kaynak veri HAM10000 (Tschandl vd., Scientific Data 2018). medmnist.com: *"Please also cite the corresponding paper(s)
  of source data"*. Sızıntı analizi kullanılırsa Abhishek vd. 2025 de anılmalı.
- **Klinik iddia yok:** Bu çalışma bir araştırma benchmark'ıdır. Sonuçlar tanısal veya klinik performans iddiası
  taşımaz, klinik karar desteği olarak yorumlanamaz.
- Kişisel veri: HAM10000 anonimleştirilmiş dermoskopi görüntüleridir. Metadata'da yaş, cinsiyet ve lokalizasyon var ama
  isim veya kimlik yok. Bu çalışmada yalnız `lesion_id`, `image_id` ve `dx` kullanıldı.

## Literatürle uyum
- Split sayıları 7.007 / 1.003 / 2.005: `INFO` ve MedMNIST v2 ile **birebir uyumlu**. Toplam 10.015 = HAM10000.
- Sınıf toplamları (327 / 514 / 1.099 / 115 / 1.113 / 6.705 / 142) HAM10000 `dx` dağılımıyla **uyumlu**
  (etiket↔dx 10.015/10.015).
- Lezyon sızıntısı, Abhishek vd. 2025 ile **lezyon sayılarında birebir uyumlu**: train-test 641, train-val 332,
  val-test 113, üç bölüm 40, toplam 1.006 / 7.470 (%13,47).
  **Görüntü sayılarında tanım farkı var:** makale "train-test: 886 images", "train-valid: 440", "valid-test: 128",
  "train-valid-test: 51" diyor. Bizim ölçülerimiz test'teki sızan görüntü 693, iki bölümdeki toplam 1.518; val'deki
  sızan 348, toplam 769; val-test 120 / 241; üç bölümde 130 görüntü. Makalenin görüntü sayma tanımı (ör. lezyon başına
  fazlalık görüntü) doğrulanamadı. Lezyon sayıları aynı olduğundan eşleme tutarlı.
- ACC 0,735 / AUC 0,917 çıpası (ResNet-18, 28 px) bu adımda doğrulanmadı (K3'ün işi). DINOv2 için raporlanan %92,02,
  test görüntülerinin %34,56'sında lezyonun train'de bulunduğu bu split üzerinde ölçülmüştür. Güçlü ön-eğitimli
  özellikler lezyon tanımadan orantısız yararlanabilir. Bu bir yorum notudur, ölçülmedi.

## Kullanım notu (veri yükleme)
```python
import medmnist
ROOT = r"<WS>\data\medmnist"
tr = medmnist.DermaMNIST(split="train", size=28, root=ROOT, download=False)   # 7007
va = medmnist.DermaMNIST(split="val",   size=28, root=ROOT, download=False)   # 1003
te = medmnist.DermaMNIST(split="test",  size=28, root=ROOT, download=False)   # 2005 — yalnız K5
# 224 px (indirildikten sonra): size=224, mmap_mode="r" önerilir (dosya ~1,09 GB)
# Değerlendirme: medmnist.Evaluator("dermamnist", "test", size=28, root=ROOT)  (ACC + AUC)
# Lezyon etiketleri (yalnız analiz için): data/splits/dermamnist_lezyon_esleme.csv  (split,index sırası npz ile aynı)
```
- Artırma ve normalizasyon istatistikleri yalnız train'den hesaplanır. Test yalnız K5'te açılır.

## Belirsizlikler
1. **64/128/224 px indirilmedi** (≈4,3 saat tahmini, Zenodo ≈100 KB/s). Bu dosyaların SHA256 değerleri ve boyutlar
   arası indeks tutarlılığı henüz ölçülmedi. 224 px manipülasyonu (K4) bu indirmeye bağlı.
2. Val↔train'de görsel olarak neredeyse özdeş ama metadata'ya göre farklı lezyon olan 3 çift var. Bunlar yayımlanmış
   kopya listesinde yok. 28 px'te ayırt edilemeyen benzer lezyonlar mı yoksa metadata'da yer almayan kopyalar mı
   olduğu ancak tam çözünürlükte (HAM10000 görüntüleri, ~2,8 GB, CC BY-NC) doğrulanabilir. Bu doğrulama yapılmadı.
3. `dermamnist_split_info.csv` üçüncü taraf bir dosyadır (Abhishek vd.; MedMNIST yazarlarından alındığı beyan ediliyor).
   Doğruluğu dolaylı olarak teyit edildi: 10.015/10.015 etiket uyumu ve yayımlanmış lezyon sayılarıyla birebir eşleşme.
4. Abhishek vd.'nin görüntü bazlı sızıntı sayıları (886/440/128/51) bizim tanımlarımızla yeniden üretilemedi
   (lezyon sayıları ise birebir aynı).
5. Zenodo kayıt lisansı (CC BY 4.0) ile DermaMNIST istisnası (CC BY-NC 4.0) arasında görünür çelişki var. Daha kısıtlayıcı
   olan CC BY-NC 4.0 esas alındı.

## KULLANICI KARARLARI (2026-10-10, şef kaydı)
- **Protokol:** resmî MedMNIST split'i ANA (replikasyon, tüm K4 seçimleri val'de). **DermaMNIST-C** (Abhishek vd. 2025,
  Zenodo 10.5281/zenodo.12739457; lezyon-ayrık 8.215/573/1.227) EK değerlendirme: K5'te baseline + kazanan(lar) aynı
  seed kümesiyle C üzerinde de eğitilip test edilir → "kazanç sızıntısız bölmede kalıyor mu" analizi.
- **Lisans:** CC BY-NC 4.0 koşulları kabul edildi (akademik, ticari değil; MedMNIST v2 + ISBI 2021 + HAM10000 atıfı zorunlu; klinik iddia yok).
- **64/128/224 px ve DermaMNIST-C dosyaları:** yerelde İNDİRİLMEZ (Zenodo ~100 KB/s); Colab defteri indirir ve MD5'i
  bu manifestteki/`medmnist.INFO` değerleriyle doğrular, SHA256'yı `runs/<id>/config.json`'a yazar.

## DermaMNIST-C (ek değerlendirme)
> **Hesaplama hedefi notu (2026-10-10):** Yukarıdaki "KULLANICI KARARLARI"nda yer alan *"yerelde İNDİRİLMEZ; Colab
> defteri indirir"* notu **GEÇERSİZDİR**. Hesaplama hedefi = **yerel makine** (GTX 1650). DermaMNIST-C dosyaları
> yerelde `data/dermamnist_c/` altına indirilir ve burada doğrulanır. Resmî split ANA, DermaMNIST-C EK değerlendirme
> kararı değişmedi.

Kaynak: Abhishek, Jain, Hamarneh, *Sci Data* 12, 196 (2025), doi:10.1038/s41597-025-04382-5. Zenodo
10.5281/zenodo.12739457, sürüm v1.1, yayım tarihi 2024-06-11 (kavram kaydı 11101337). Kod GitHub
`kakumarabhishek/Corrected-Skin-Image-Datasets`. Zenodo API yanıtı `data/dermamnist_c/zenodo_12739457.json` dosyasında saklı.

İndirme komutu (WS kökünden; HTTP Range ile kaldığı yerden devam eder, MD5 uyuşmazsa dosyayı siler):
`PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe scripts/indir_dermamnist_c.py --asama indir --boyutlar 28`
(2026-10-10 09:32:11+03:00; günlük `data/dermamnist_c/indirme_c_28.log`, kayıt `data/dermamnist_c/indirme_kaydi_c.json`)
Analiz komutu: `PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I scripts/indir_dermamnist_c.py --asama analiz --boyutlar 28`
(09:38:26; süre ~1 sn; çıktı `data/dermamnist_c/dermamnist_c_analiz.json`, günlük `data/dermamnist_c/analiz_c.log`,
eşleme `data/splits/dermamnist_c_esleme.csv`). Hedef makine: A, yerel. GPU kullanılmadı.
Betik düzeltmeleri (2026-10-10, devralınan sürüme göre): npz dizileri bir kez belleğe alınıyor (önceki sürüm `NpzFile`'ı
her örnekte yeniden açtığı için analizi kesilmişti). Piksel RMSE int16 taşması int32 ile giderildi. 224 px için
boyutlar arası sıra denetimi (`boyut_sira_denetimi`, yalnız train/val) eklendi.

### Bütünlük
| dosya | durum | bayt (ölçülen = Zenodo) | MD5 ölçülen = Zenodo | SHA256 (ölçülen) |
|---|---|---|---|---|
| `DermaMNIST-C.csv` | indirildi (3,0 sn) | 747.941 ✓ | `1c54bc9f483e97a9d7dbbde076a45900` **eşleşti** | `c27194d338986463966e1694d5b49fda933222368f04b316c702ac2e8a0d4f5c` |
| `dermamnist_corrected_28.npz` | indirildi (28,5 sn, 710 KB/s) | 20.698.236 ✓ | `96e2862980877d45d9cacb5c25de6950` **eşleşti** | `c3d9cbd9008a0dcb76943d5cf38681bdc5166effc79f9e215657b859b9226aa8` |
| `dermamnist_corrected_224.npz` | **İNDİRİLMEDİ** | (Zenodo: 1.131.258.316) | resmî `84920fb70c83b234c295b6f0d4ae2bc0` | — |

Devam oturumunda (09:37) iki dosyanın MD5 ve SHA256 değerleri diskten yeniden hesaplandı ve yukarıdaki değerlerle aynı çıktı.
Bu yüzden yeniden indirme yapılmadı. DermaMNIST-E dosyaları (`dermamnist_extended_*.npz`, `DermaMNIST-E.csv`) bilerek indirilmedi.

**224 px süre tahmini** (1.131.258.316 B): 102 KB/s (MedMNIST 28 px ortalaması) ile ≈ 180 dk; şu anki paylaşımlı
~500 KB/s ile (MedMNIST 128 px indirmesi sürerken ölçülen) ≈ 37 dk; 710 KB/s (C-28 ölçümü) ile ≈ 26 dk. Süre 8 dk eşiğinin
üzerinde olduğu için indirilmedi. Arka planda başlatılacak komut (indirme + 28/224 analizi; 28 px MD5 ile atlanır):
`cd <WS> && nohup env PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I scripts/indir_dermamnist_c.py --asama hepsi --boyutlar 28 224 > data/dermamnist_c/indirme_c_224.log 2>&1 &`
Disk: C: 99 GB boş. Analiz 224 px diziyi belleğe alır (10.015×224×224×3 uint8 ≈ 1,51 GB; RAM 15,9 GB).

### Yapı ve split
npz anahtarları: `train/val/test_images` uint8 (N, 28, 28, 3), piksel aralığı [0, 255] (üç bölümde de min 0, max 255);
`train/val/test_labels` uint8 (N, 1). Etiket kodlaması resmî DermaMNIST ile aynı (akiec=0 … vasc=6).
CSV sütunları: `lesion_id, image_id, dx, dx_type, age, sex, localization, dataset, split` (10.015 satır; **`index` sütunu yok**).

| bölüm | örnek | beklenen | lezyon | kaynak |
|---|---|---|---|---|
| train | 8.215 | 8.215 ✓ | 5.758 | DermaMNIST-C resmî (Abhishek vd.) |
| val | 573 | 573 ✓ | 563 | DermaMNIST-C resmî |
| test | 1.227 | 1.227 ✓ | 1.149 | DermaMNIST-C resmî |
| toplam | 10.015 | = HAM10000 | 7.470 | — |

- CSV split sayıları = npz sayıları. **CSV satır sırası npz sırasıdır:** CSV sırasındaki `dx` ile npz etiketi train
  8.215/8.215, val 573/573, test 1.227/1.227 aynı. Bağımsız kanıt olarak en yakın komşu denetimi yapıldı (seed 0
  örneklem; yalnız train/val; resmî 28 px train+val havuzu). C görüntüsünün resmî 28 px'teki en yakın komşusu CSV
  sırasının gösterdiği `image_id`: train 285/356, val 337/400. CSV eşleşmesinin RMSE medyanı NN medyanına çok yakın
  (train 14,10 / 14,04; val 13,02 / 12,93). Betik: oturum scratchpad'i `sira.py` (WS dışında; sonuç bu satırda).
- Yeni seed'li split **üretilmedi**. `data/splits/dermamnist_c_esleme.csv` C'nin resmî bölmesinin kaydıdır
  (`split_C, image_id, lesion_id, dx, split_resmi, index_resmi`).
- V4: C'nin kendi val kümesi (573) var. EK değerlendirmede model seçimi C-val ile, test bir kez (K5).

### Dağılım (C, 28 px etiketleri; HAM `dx` ile aynı)
| # | sınıf | train | val | test | toplam |
|---|---|---|---|---|---|
| 0 | akiec | 273 | 19 | 35 | 327 |
| 1 | bcc | 448 | 22 | 44 | 514 |
| 2 | bkl | 944 | 50 | 105 | 1.099 |
| 3 | df | 102 | 5 | 8 | 115 |
| 4 | mel | 1.021 | 22 | 70 | 1.113 |
| 5 | nv | 5.309 | 448 | 948 | 6.705 |
| 6 | vasc | 118 | 7 | 17 | 142 |
| | dengesizlik (max/min) | 52,05 | 89,60 | 118,50 | — |

- Sınıf toplamları resmî DermaMNIST ile birebir aynı. Görüntüler yeniden dağıtılmış, çıkarılan ya da eklenen görüntü yok.
- **Bölme artık tabakalı değil:** nv payı train %64,63, val %78,18, test %77,26 (resmîde ~%67). "Hep nv" test
  ACC tabanı C'de **0,7726** (resmîde 0,669). C test ACC'si resmî ACC ile doğrudan kıyaslanamaz. Ayrıca val'de df 5
  ve vasc 7, test'te df 8 örnek var, dolayısıyla sınıf bazlı ölçütlerin varyansı yüksek.

### Resmî MedMNIST split'iyle ilişki (ISIC kimliği üzerinden)
- C'deki 10.015 `image_id`'nin tamamı resmî split'te ve HAM10000 metadata'sında var (eşleşmeyen 0). CSV `lesion_id` ile HAM eşleşmesi 10.015/10.015.
- Geçiş matrisi (satır resmî → sütun C):

| resmî \ C | train | val | test |
|---|---|---|---|
| train (7.007) | 7.007 | 0 | 0 |
| val (1.003) | 430 | 573 | 0 |
| test (2.005) | 778 | 0 | 1.227 |

- Yani **C-test ⊂ resmî test** (1.227/2.005), **C-val ⊂ resmî val** (573/1.003), C-train = resmî train + 430 val + 778 test.
  Resmî train'den hiçbir görüntü val/test'e taşınmamış.
- Taşıma kuralı ölçüldü: lezyonu başka bir resmî bölümde de bulunan resmî val (428) ve test (773) görüntülerinin
  **tamamı** C-train'e taşınmış (kural sağlanıp yerinde kalan: 0). Bunlara ek olarak lezyon çakışması OLMAYAN
  7 görüntü daha taşınmış: val'den 2 (ISIC_0024736, ISIC_0033481) ve test'ten 5 (ISIC_0027820, ISIC_0024770,
  ISIC_0029625, ISIC_0033556, ISIC_0033417). Bunların yazarların metadata dışı kopya listesinden gelmesi muhtemel, doğrulanmadı.
  Bu 7 görüntü, ana manifestteki görsel olarak benzer 3 val↔train çiftinden (val 809/76/11) farklı.
- **Piksel farkı:** C-28 görüntüleri resmî 28 px ile aynı `image_id`'de piksel-özdeş **değil** (özdeş 0/10.015;
  RMSE medyan 14,05, maks 85,63; 0–255 ölçeği). C görüntüleri kaynak HAM10000'den farklı bir küçültme/kırpma ile yeniden
  üretilmiş görünüyor (sıra eşleşmesi yukarıda doğrulandı). Bu nedenle C üzerinde eğitilen model resmî
  `dermamnist.npz` görüntüleriyle değil, `dermamnist_corrected_*.npz` ile beslenmeli. Ön işleme farkı, resmî↔C sonuç
  farkına sızıntıdan bağımsız bir katkı ekler (bkz. Belirsizlikler).

### Sızıntı denetimi (V5) — C
- **Lezyon çakışması: train∩val 0 · train∩test 0 · val∩test 0 · üç bölümde 0** (lezyon-ayrık ✓).
- `image_id` çakışması: 0 / 0 / 0.
- Tam-piksel (SHA1, 28 px) bölümler arası çakışma: train∩val 0 · train∩test 0 · val∩test 0. Split içi kopya:
  train 2, val 0, test 0 (train içi; sızıntı değil).
- Bölme türü: lezyon (hasta değil) bazlı. HAM10000'de hasta kimliği yok, bu yüzden hasta düzeyinde ayrıklık denetlenemez.

### Lisans ve etik (V6) — C
- **npz dosyaları (`dermamnist_corrected_28/224.npz`): CC BY-NC 4.0 (ticari kullanım YASAK).** Metadata CSV'leri
  (`DermaMNIST-C.csv`): CC BY 4.0. Kod: Apache-2.0. (Kaynak: Zenodo kaydı açıklamasındaki "License" bölümü. Kayıt
  düzeyindeki `license.id` alanı yalnız `cc-by-4.0` gösteriyor; dosya bazındaki NC ifadesi esas alındı.)
- Atıf: Abhishek, Jain, Hamarneh (Sci Data 2025) + kaynak HAM10000 (Tschandl vd. 2018) + MedMNIST v2 / ISBI 2021.
- Klinik iddia yok (ana bölümdeki kayıt geçerli).

### Literatürle uyum — C
- Split 8.215 / 573 / 1.227: Abhishek vd. ve konu kaydıyla **birebir uyumlu**.
- Lezyon-ayrıklığı (0/0/0) makalenin beyanıyla **uyumlu**.
- C-train'e taşınan 1.208 görüntünün 1.201'i lezyon kuralıyla açıklanıyor. Kalan 7 görüntünün gerekçesi bu
  oturumda makaleden doğrulanmadı.

### Belirsizlikler — C
1. **224 px indirilmedi.** SHA256 değeri, 224↔28 sıra tutarlılığı ve 224 px sınıf sayıları yukarıdaki komutla ölçülecek.
2. C-28 ile resmî 28 px arasındaki ön işleme farkının (RMSE medyan 14) nedeni (interpolasyon, kırpma, renk) belirlenmedi.
   Resmî↔C karşılaştırmasında "split etkisi" ile "ön işleme etkisi" karışabilir. Ayrıştırmak için resmî görüntüler C
   bölmesine göre yeniden düzenlenerek ek bir kontrol koşumu yapılabilir (`dermamnist_c_esleme.csv` içindeki
   `split_resmi/index_resmi` sütunları bunu sağlar). Bu kontrolü yapıp yapmama kararı şefe aittir.
3. Lezyon kuralı dışında taşınan 7 görüntünün gerekçesi doğrulanmadı.
4. Val (573) ve test (1.227) resmîden küçük ve nv ağırlıklı. Bu yüzden C üzerindeki farkların güven aralıkları geniş olacak.

### KULLANICI KARARI (2026-10-10) — C-değerlendirme protokolü
- **Birincil:** resmî MedMNIST(+) görüntüleri (28/64/128/224, MD5 doğrulanmış) + DermaMNIST-C lezyon-ayrık bölmesi
  (`data/splits/dermamnist_c_esleme.csv`) → bölme dışındaki her şey sabit; ölçülen fark yalnız sızıntı/bölme etkisi.
- **İkincil:** DermaMNIST-C'nin yayımlanmış npz'si (kendi küçültmesi) — literatürle kıyas için.
