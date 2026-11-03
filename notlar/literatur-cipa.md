# Literatür Çıpası — Konu 3.1 (DermaMNIST · ResNet-18 28 px → DINOv2)

*K1.5 çıpa modu · hazırlanma: 2026-10-10 · erişim tarihi (tüm URL'ler): 2026-10-10*
*Künyeler: `notlar/references-cekirdek.bib`. Bu belgedeki her sayının yanında kaynağı ve tablo numarası var. "Bulunamadı" yazan değerler için literatürde kaynak bulunamadı.*

---

## 0. Kısa sonuç

| Kalem | Değer |
|---|---|
| Rakip | ResNet-18 (28), sıfırdan eğitim — `yang_2023_medmnistv2`, **Tablo 3**: DermaMNIST **ACC 0.735 / AUC 0.917** (katalogdaki değer doğru) |
| Resmî kod | `MedMNIST/experiments` @ **70b6b3a7ad7afddff1df2a3b735235830fbdb142** (2024-07-11; tag yok) + `MedMNIST/MedMNIST` @ **805b74237fb5bd2db376fd94e50646b6a8905d9d** (2025-01-11; tag yok; PyPI `medmnist==3.0.2`) |
| Bağımsız reprodüksiyon | `doerrich_2025_medmnistplus` Ek Tablo S4: ResNet-18 E2E 28 px **73.38 ± 0.61 ACC** / 88.35 ± 0.28 AUC (3 seed) |
| Kapı (birincil) | `--metrik test_acc --yon max --lit 0.7338 --lit-ust 0.742 --tolerans 0.012` |
| Katalogdaki %92.02 | **Kısmen doğru.** Değer gerçekten var, ancak koşul farklı: `wu_2025_foundationmedmnist` (hakemsiz arXiv ön baskısı), DINOv2 ViT-B/14, **uçtan uca ince ayar (fine-tune)**, **224 px**, metrik **ACC** (AUC 98.92). **Linear probe değil**: aynı makalede linear probe sonucu **83.97 ± 0.08**. Katalogdaki "(MedNet)" atfı ikincil; MedNet (`ferdous_2025_mednet`) bu sayının kaynağı olarak Wu & Papież'i gösteriyor |
| Lisans | DermaMNIST **CC BY-NC 4.0** (diğer MedMNIST alt kümeleri CC BY 4.0); HAM10000 **CC BY-NC 4.0**; MedMNIST kodu Apache-2.0; DINOv2 kodu ve ağırlıkları Apache-2.0 |

---

## 1. Rakibin orijinal makalesi ve resmî kod

### 1.1 Künye
- **Yang, J., Shi, R., Wei, D., Liu, Z., Zhao, L., Ke, B., Pfister, H., Ni, B. (2023).** MedMNIST v2 — A large-scale lightweight benchmark for 2D and 3D biomedical image classification. *Scientific Data* 10(1), 41. DOI: 10.1038/s41597-022-01721-8 · arXiv:2110.14795 · PMC9852451.
  - URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC9852451/ (erişim 2026-10-10). Künye Crossref ile doğrulandı.
- Öncül sürüm: Yang, Shi, Ni (2021), MedMNIST Classification Decathlon, IEEE ISBI 2021, s. 191–195, DOI 10.1109/ISBI48211.2021.9434062.

### 1.2 Tablo 3 — DermaMNIST satırları
Tablo başlığı: "Benchmark on each dataset of MedMNIST2D in metrics of AUC and ACC." Standart sapma verilmemiş. Makale "her yöntem ve veri seti için en az 3 denemenin ortalamasını" bildiriyor.

| Yöntem | AUC | ACC |
|---|---|---|
| ResNet-18 (28) | **0.917** | **0.735** |
| ResNet-18 (224) | 0.920 | 0.754 |
| ResNet-50 (28) | 0.913 | 0.735 |
| ResNet-50 (224) | 0.912 | 0.731 |
| auto-sklearn | 0.902 | 0.719 |
| AutoKeras | 0.915 | 0.749 |
| Google AutoML Vision | 0.914 | 0.768 |

Çapraz doğrulama: Aynı yedi satır `manzari_2023_medvit` Tablo 3'te birebir aynı kopyalanmış (https://arxiv.org/html/2302.09462v1). PDF sayfa numarası doğrulanamadı; PDF metni çıkarılamadı, bu yüzden atıfta "Tablo 3" kullanılmalı.

> Not (alan-protokolleri.md ile çapraz): Katalogdaki "MedMNIST ortalaması ResNet-18(224) 0.821 …" değerleri 12 veri setinin ortalamasıdır, DermaMNIST'e özgü değildir. DermaMNIST'e özgü değerler yukarıdaki tablodadır. **Google AutoML Vision koşulamaz** (Temmuz 2021 deneyi, Edge modeli). Tabloda `kaynak: literatur` etiketiyle yer alır.

### 1.3 Resmî kod depoları ve pin
| Depo | Commit (pin) | Tarih / mesaj | Tag |
|---|---|---|---|
| https://github.com/MedMNIST/experiments | `70b6b3a7ad7afddff1df2a3b735235830fbdb142` | 2024-07-11 "update pytorch training scripts" (yalnız `--size` argümanını ekliyor; lr, epoch, seçim ve dönüşümler değişmemiş) | yok |
| https://github.com/MedMNIST/MedMNIST | `805b74237fb5bd2db376fd94e50646b6a8905d9d` | 2025-01-11 "minor" | yok |
| PyPI `medmnist` | 3.0.2 | 2024-09-16 ("Bug fixed in Evaluator (`size_flag`)") | — |

Kaynak: GitHub REST API `/commits` ve `/tags` (2026-10-10). Her iki depoda tag listesi boş (`[]`).
Veri dosyaları (`medmnist/info.py` @805b742): Zenodo kaydı 10519652.
- `dermamnist.npz` MD5 `0744692d530f8e62ec473284d019b0c7`
- `dermamnist_224.npz` MD5 `8974907d8e169bef5f5b96bc506ae45d`
- `_64` MD5 `b70a2f5635c6199aeaa28c31d7202e1f`, `_128` MD5 `2defd784463fa5243564e855ed717de1`

### 1.4 Resmî eğitim protokolü (makale + `MedMNIST2D/train_and_eval_pytorch.py` @70b6b3a)
| Öğe | Değer | Kaynak |
|---|---|---|
| Epoch | 100 (`--num_epochs` varsayılanı) | makale + kod |
| Optimizer | Adam, lr = 0.001 (kodda sabit) | makale + kod |
| LR takvimi | MultiStepLR, milestones = [0.5·E, 0.75·E] = [50, 75], gamma = 0.1 | makale ("delaying the learning rate by 0.1 after 50 and 75 epochs") + kod |
| Batch | 128 | makale + kod |
| Kayıp | `nn.CrossEntropyLoss()` (çok sınıflı) | makale + kod |
| Augmentasyon | **Yok**: yalnız `ToTensor()` + `Normalize(mean=[.5], std=[.5])`; train loader `shuffle=True` | kod |
| Normalizasyon | mean 0.5 / std 0.5 (ImageNet istatistiği değil) | kod |
| 28 px mimari | Yerel `models.ResNet18`: **conv1 = 3×3, stride 1, padding 1, bias=False; maxpool yok**; [2,2,2,2] BasicBlock; layer2–4 stride 2; AdaptiveAvgPool → FC(512→7) | `models.py` @70b6b3a |
| 224 px sürümü (makale) | `--resize`: `transforms.Resize((224,224), interpolation=NEAREST)` ile **28 px'ten büyütme**; model `torchvision.resnet18(pretrained=False)` (standart 7×7 stem + maxpool, **ön-eğitimsiz**) | makale ("224 (resized from 28)") + kod |
| Model seçimi | Her epoch'ta **val AUC**; `cur_auc > best_auc` ise best güncellenir. ACC seçimde kullanılmaz | kod; bakımcı yanıtı, experiments issue #5 |
| Değerlendirme | Eğitim sonunda `best_model` train/val/test üzerinde değerlendirilir; çok sınıflı çıktıda **softmax** uygulanıp Evaluator'a verilir | kod |
| Seed | **Kodda seed ayarı yok** (`torch.manual_seed` yok). Makalede "en az 3 deneme ortalaması" var, seed değerleri yok | kod + makale |
| `as_rgb` | DermaMNIST zaten 3 kanallı (`n_channels: 3`) | `info.py` |

**KRİTİK protokol bulgusu (en iyi epoch ile son epoch farkı):** `8b0f553` commit'inden (2023-05-18, "update best model save") önceki kodda `best_model = model` yazıyordu. Bu atama yalnızca referans kopyalıyordu, bu yüzden "best" model aslında **son epoch ağırlıklarıydı**. Kaydedilen state_dict ve test metriği de son epoch'a aitti. Düzeltme `deepcopy(model)` ile yapıldı. Kanıtlar: commit diff'i (GitHub API) ve `12e9f40` sürümündeki kod. Experiments issue #5'te bir kullanıcı PathMNIST ResNet-18(28) için "son epoch AUC 0.985 makaleye yakın, en iyi epoch AUC 0.994" bildirmiş. **Çıkarım (doğrulanmadı):** 2021 tablo değerleri büyük olasılıkla son epoch ağırlıklarıyla üretildi. Pinlediğimiz 70b6b3a ise gerçekten en iyi val-AUC epoch'unu seçiyor. Bu nedenle `rakip_kos.py` iki test metriğini de yazmalı: `test_acc` (best-val-AUC) ve `test_acc_son` (100. epoch). Kapı birincil olarak `test_acc` üzerinde çalışır. SARI çıkarsa ikinci değer protokol farkını açıklamak için kullanılır.

**Bellek tuzağı (4 GB GPU):** `--size 224` verilip `--resize` verilmezse kod **yerel 3×3 stem'li, maxpool'suz ResNet18'i** 224 px girişe uygular. Bu durumda 64 kanallı 224×224 öznitelik haritaları oluşur ve bellek çok ağırlaşır. Makaledeki "ResNet-18 (224)" ise `--resize` ile torchvision ResNet-18'dir. Doerrich ve ark.'ın (MedMNIST+) ve bizim 224 deneylerimizin standart torchvision/timm stem'ini kullanması gerekir.

---

## 2. Literatür değeri, aralığı ve replikasyon kapısı

### 2.1 ResNet-18'in 28 px bilgiyle DermaMNIST sonuçları
| Kaynak | Protokol | Test ACC | Test AUC | n / std |
|---|---|---|---|---|
| `yang_2023_medmnistv2` Tablo 3 | Resmî: sıfırdan, 28 px, Adam 1e-3, 100 ep, bs128 | **0.735** | **0.917** | ≥3 deneme, std yok |
| `doerrich_2025_medmnistplus` Ek Tablo S4 | ImageNet ön-eğitimli ResNet-18, uçtan uca; 28 px görüntü **224 tuvale sıfır dolgulu**; AdamW 1e-4, kosinüs, ≤100 ep, bs64, val'de erken durdurma | **73.38 ± 0.61** | 88.35 ± 0.28 | 3 seed |
| `medeiros_2026_tta` Tablo 2 | Sıfırdan ResNet-18, 28 px, Adam 1e-3, kosinüs, 25–30 ep, bs256, val ACC ile erken durdurma | 0.742 (TTA'sız) | — | seed sayısı yok |
| `abhishek_2025_dermamnistkalite` Tablo 1 | "Original-28" satırı | 0.735 | 0.917 | std yok; MedMNIST v2 ile aynı sayılar, yeniden koşulduğu belirtilmemiş (bağımsız sayılmamalı) |
| İkincil (doğrulanmadı) | Kadric ve ark. arXiv:2507.12961 Tablo 3'ün aktardığı değerler: Xu ve ark. 2022 TMI = 0.741; Ahmed ve ark. 2022 = 0.7367 | 0.741 / 0.7367 | — | birincil tablo açılamadı → **kapıda kullanılmadı** |

Gözlem: 28 px bilgiyle elde edilen ACC değerleri, protokol farklılıklarına rağmen dar bir bantta toplanıyor (0.7338–0.742). AUC'de ise Doerrich (88.35), sıfır dolgu ve ön-eğitim nedeniyle resmî değerin (91.7) belirgin altında. Bu yüzden AUC için bağımsız bir çıpa **yok**.

### 2.2 Replikasyon kapısı argümanları (HAZIR)

**Birincil (karar verici):**
```
python SK/replikasyon_kapisi.py --kok WS --metrik test_acc --yon max \
  --lit 0.7338 --lit-ust 0.742 --tolerans 0.012 \
  --kaynak "Yang ve ark. 2023 Sci Data 10:41 Tablo 3 (0.735); Doerrich ve ark. 2025 Sci Rep 15:7669 Ek Tablo S4 (73.38±0.61, 3 seed); Medeiros 2026 arXiv:2604.09697 Tablo 2 (0.742)"
```
- Kabul aralığı (YEŞİL): [0.7218, 0.754]. SARI bandı: [0.7098, 0.766].
- **Tolerans gerekçesi:** Doerrich'in 3 seed üzerinden bildirdiği std 0.61 puan. Bunun yaklaşık 2 katı alınarak 0.0122 ≈ **0.012** belirlendi. Aralık uçları birbirinden bağımsız iki ölçümden geliyor: alt sınır Doerrich 0.7338, üst sınır Medeiros 0.742. Resmî değer 0.735 bu aralığın içinde. Üst kabul sınırı 0.754, makaledeki "ResNet-18 (224, 28'den büyütülmüş)" değerine eşit. Sıfırdan 28 px baseline'ın bunun üstüne çıkması, protokol veya sızıntı şüphesi doğurmalı (KIRMIZI bölgesi).

**İkincil (yalnız bilgi amaçlı, karar vermez):**
```
python SK/replikasyon_kapisi.py --kok WS --metrik test_auc --yon max \
  --lit 0.917 --tolerans 0.01 --kaynak "Yang ve ark. 2023 Sci Data 10:41 Tablo 3"
```
Gerekçe: AUC için tek kaynak var (std bildirilmemiş). 28 px ResNet-18 AUC std'leri Doerrich S4'te 0.28 puan. Tek kaynağa bağlı olduğumuz için 2×std yerine daha geniş bir değer, 0.01, seçildi.

### 2.3 Val mi, test mi? Net öneri
- Literatürde **val sayısı yok**. Tüm kaynaklar resmî test (n = 2005) değerini bildiriyor. Resmî betik de test metriğini eğitim sonunda kendisi hesaplıyor.
- **Öneri:** Replikasyon kapısı **yalnız rakip (baseline) varyantı için, test_acc üzerinde ve tek seferlik** çalıştırılsın. Gerekçeler:
  1. Rakibi literatürle karşılaştırmak, **kendi yöntemimiz hakkında karar vermek değildir**. Test kilidinin amacı, manipülasyon seçimlerinin teste sızmasını önlemek.
  2. Bağımsız tek çıpa test sayısı. Val üzerinde bir kapı kurulamaz, çünkü karşılaştırılacak literatür değeri yok.
  3. Baseline'ın test sayıları bu koşumdan **dondurulur** ve K5 tablosunda aynen kullanılır. K5'te baseline yeniden koşulmaz.
- **Yan koşullar:**
  - (a) `rakip_kos.py` val_acc, val_auc, test_acc, test_auc ve test_acc_son değerlerini yazar.
  - (b) Manipülasyon varyantlarının koşum sarmalayıcısı test metriği **yazmaz**, ya da yazdığı test metriği K5'e kadar mühürlü ayrı bir dosyada kalır.
  - (c) K4'teki karşılaştırma ve durma kuralı yalnız **val_acc** ile yapılır. Baseline'ın 5 seed val_acc ortalaması iç çıpa olarak `sonuclar/replikasyon.md`'ye yazılır.
  - (d) Sarmalayıcı seed'i rakip koduna dokunmadan ayarlamalı (`torch.manual_seed`, `numpy`, `random`, `cudnn.deterministic`). Resmî kodda seed yok; seed ayarlanmazsa 5 koşum tekrarlanamaz.

---

## 3. "DINOv2 ile %92.02" iddiasının denetimi

- **Kaynak:** Wu, F., Papież, B. W. (2025). *Rethinking Foundation Models for Medical Image Classification through a Benchmark Study on MedMNIST.* arXiv:2501.14685 (v1, 2025-01-24; yorum alanı "submitted to MIDL2025"; journal-ref yok). URL: https://arxiv.org/html/2501.14685v1
- **Değer:** Ana sonuç tablosu (`tab:results1`, bölüm 3.2.2), DermaMNIST sütunu, **uçtan uca (END-TO-END)** satırı: DINOv2 ViT-B/14 (`vit_base_patch14_dinov2.lvd142m`) için **ACC 92.02 ± 0.32 / AUC 98.92 ± 0.11**, 3 koşu.
- **Koşullar:** 224×224 görüntü ("primarily use images of 224×224"). Görüntülerin MedMNIST+ native 224 sürümünden mi alındığı, yoksa 28'den mi büyütüldüğü belirtilmemiş. AdamW; sınıflandırıcı lr 1e-3, her 200 iterasyonda ×0.9; kodlayıcı lr {1e-4, 1e-5, 1e-6} ızgarasından seçilmiş; 15 000 iterasyon. **Split, model seçimi, batch boyutu ve augmentasyon belirtilmemiş.** Tablo "en iyi uçtan uca sonuçlar" olarak tanımlanıyor; lr seçiminin val ile mi test ile mi yapıldığı belirsiz.
- **Aynı makalede DINOv2 ViT-B/14 linear probe (224):** **83.97 ± 0.08 ACC / 96.62 ± 0.02 AUC.**
- **Hüküm:** Sayı doğru aktarılmış, ancak katalogdaki ifade yanıltıcı: (i) bir **ince ayar** sonucu, öznitelik + linear probe sonucu değil; (ii) **hakemsiz ön baskı**; (iii) protokolde belirsizlik var (split ve seçim yazılmamış). "(MedNet)" atfı ikincil: `ferdous_2025_mednet` (Sci Rep 15:41936) kaynakçasında Wu & Papież [20] var, DINOv2 adı geçmiyor. Bizim kapsamımızdaki "DINOv2 öznitelik + linear probe" için doğru çıpa **≈ 84 (224 px)** ve **≈ 77.4 (28 px)**. %92 bu kurulum için **tavan değil, ulaşılmaz bir hedef** olarak ele alınmalı.

---

## 4. Manipülasyonların literatürdeki etki büyüklükleri ve "görülmemiş kadar iyi" eşikleri

Tüm değerler DermaMNIST resmî **test** sonuçlarıdır (aksi belirtilmedikçe). "Δ" = puan (yüzde puanı).

### (a) Çözünürlük 224 + ImageNet ön-ağırlığı
| Kaynak | Kurulum | 28 px → 224 px ACC | Δ |
|---|---|---|---|
| `yang_2023_medmnistv2` T3 | Sıfırdan, **28'den NN ile büyütme** (yeni bilgi yok) | 0.735 → 0.754 | +1.9 |
| `doerrich_2025_medmnistplus` S4 | ImageNet ön-eğitimli ResNet-18 E2E, MedMNIST+ native çözünürlük | 73.38 → 82.33 (64: 76.36, 128: 79.19) | +8.95 (AUC 88.35 → 95.27) |
| `wu_2025_foundationmedmnist` | ResNet-18 E2E 224 | 86.05 ± 0.78 | — |
| `zhang_2026_weightedsampling` Tablo 1d | ImageNet ResNet-18, DermaMNIST-224, AdamW 1e-4, 30 ep, 3 seed | 0.882 ± 0.002 (AUC 0.973) | — |
| Aynı kaynak, Tablo 1a | 96 px (28'den büyütme) | 0.779 ± 0.002 | — |
| `doerrich_2025_medmnistplus` S4 | En iyi DermaMNIST ACC: DenseNet-121 E2E 224 | 84.74 ± 0.51 | — |

- **Beklenen etki:** Native 224 + ImageNet ile **+8 ile +15 puan** (literatür 82.3–88.2 ACC). Yalnız 28'den büyütme ile **+2 ile +4 puan**.
- **Görülmemiş eşiği:** Native 224 ResNet-18 için **> 0.90**. 28'den büyütülmüş girdi için **> 0.80** (28 px bilgiyle bilinen tavan için bkz. (e)).
- **Protokol uyarısı:** Ön-eğitimli modellerde normalizasyon ImageNet istatistiğine geçirilmeli. Resmî kod 0.5/0.5 kullanıyor; bu değişiklik ayrıca belgelenmeli. Adil kıyas için (alan-protokolleri M3) ImageNet ön-eğitimli ResNet-18 de "rakip + aynı ön-eğitim" satırı olarak koşulmalı.

### (b) DINOv2 öznitelikleri + linear probe
| Kaynak | Model / çözünürlük | ACC | AUC |
|---|---|---|---|
| `wu_2025_foundationmedmnist` ana tablo | DINOv2 ViT-B/14, LP, 224 | **83.97 ± 0.08** | 96.62 ± 0.02 |
| Aynı kaynak, Ek D | DINOv2 ViT-B/14 LP, 28 / 64 / 128 | 77.37 ± 0.25 / 79.55 ± 0.08 / 81.88 ± 0.50 | 92.91 / 94.41 / 95.34 |
| Aynı kaynak, Ek D | DINOv2 ViT-B/14 E2E, 28 / 64 / 128 | 76.79 ± 0.72 / 81.68 ± 0.76 / 87.35 ± 0.06 | 92.13 / 94.96 / 97.60 |
| `doerrich_2025_medmnistplus` S4 | DINO (v1) ViT-B/16 LP, 28 / 64 / 128 / 224 | 75.78 / 79.88 / 81.65 / **84.42 ± 0.23** | 91.87 / 95.40 / 95.95 / **96.83** |
| Aynı kaynak | CLIP ViT-B/16 LP 224 | 81.93 ± 0.31 | 95.92 |
| `ren_2025_medalmighty` Tablo 2 | Donuk DINOv2 + doğrusal kafa, 28'den 224'e büyütme, augmentasyon yok | ViT-S/14 0.727 ± 0.037; ViT-B/14 0.725 ± 0.034; ViT-L/14 0.732 ± 0.042 | 0.903 / 0.905 / 0.901 |

- **ViT-S/14 için DermaMNIST'te güvenilir bir LP sayısı bulunamadı.** Resmî DINOv2 deposuna göre ImageNet LP başarımı ViT-S/14'te 81.1, ViT-B/14'te 84.5; S, B'den yaklaşık 3.4 puan geride. DermaMNIST'te de S < B beklenir, ancak bu bir **çıkarımdır, sayı değil**.
- `ren_2025_medalmighty`'nin büyük std değerleri (±3–4 puan) ve 28'den büyütme işlemi, 28 px bilgiyle donuk özniteliklerin ~0.73'te kaldığını düşündürüyor. Kaynak orta güvenilirlikte.
- **Beklenen etki:** 28 px bilgiyle baseline'a göre +2 ile +4 puan (~0.75–0.77). Native 224 ile ViT-B/14 için +10 puan civarı (~0.84).
- **Görülmemiş eşiği:** LP 224 için **> 0.88**; LP 28 için **> 0.80**.

### (c) Focal loss ve sınıf ağırlığı / dengeli örnekleme
| Kaynak | Kurulum | CE | Focal (γ = 2) | Ağırlıklı örnekleme (WS) |
|---|---|---|---|---|
| `zhang_2026_weightedsampling` Tablo 1a (96 px, ImageNet ResNet-18, 3 seed) | ACC / makro-ACC / AUC | 0.779 / 0.579 / 0.920 | 0.775 / 0.565 / 0.922 | 0.764 / 0.621 / 0.917 |
| Aynı kaynak, Tablo 1d (224) | ACC / makro-ACC / AUC | 0.882 / 0.758 / 0.973 | 0.877 / 0.753 / 0.978 | 0.885 / 0.781 / 0.975 |

- **Etki:** Focal loss ACC'de **−0.4 / −0.5 puan**, AUC'de **+0.2 / +0.5 puan**. WS ACC'de −1.5 ile +0.3 puan arasında; **makro-ACC'de +2.3 ile +4.2 puan**.
- **Beklenen etki:** Ana metrik ACC'de **≈ 0 (±1 puan)**. Kazanç, dengeli/makro metriklerde görülür. Bu nedenle hipotez kartlarına ikincil metrik olarak makro-ACC veya dengeli doğruluk (balanced accuracy) eklenmesi önerilir. ISIC 2018 T3'ün resmî metriği de dengeli doğruluktur (alan-protokolleri.md).
- **Görülmemiş eşiği:** Focal veya ağırlıklandırmayla ACC'de **> +2 puan** kazanç.
- Çoğunluk sınıfı tabanı: Hem Doerrich S4 hem Wu & Papież'de SAM ViT-B/16 LP tüm çözünürlüklerde sabit **66.88** ACC veriyor. Bu değer büyük olasılıkla test kümesindeki "melanocytic nevi" oranıdır (yaklaşık 1341/2005). **Çıkarımdır**; K2 veri manifestinde ölçülmeli.

### (d) Augmentasyon ve TTA
- `medeiros_2026_tta` Tablo 2 (hakemsiz ön baskı): DermaMNIST'te sıfırdan eğitilmiş ResNet-18 (28 px) için TTA ile (N = 50 görünüm, flip + ±15° döndürme + RRC 0.8–1.0 + renk/bulanıklık, olasılık ortalaması) **0.742 → 0.758 (+1.6 puan)**. 12 kombinasyon içinde **tek iyileşme** bu. Diğer veri setlerinde TTA zarar veriyor (en kötü −31.6 puan, PathMNIST). ECE 0.196'dan 0.039'a düşüyor. Seed sayısı bildirilmemiş.
- Eğitim augmentasyonunun DermaMNIST'teki izole etkisi: **bulunamadı** (resmî baseline ve Doerrich augmentasyon kullanmıyor veya bildirmiyor).
- **Beklenen etki:** TTA için 0 ile +2 puan; eğitim augmentasyonu için kaynak yok. Ön-kayıtta belirsiz olarak işaretlenmeli.
- **Görülmemiş eşiği:** TTA ile **> +3 puan**.

### (e) Ek yöntemler: kNN, tam ince ayar, CLIP/BiomedCLIP, MedMNIST SOTA bildirimleri
| Yöntem | DermaMNIST test ACC | Kaynak |
|---|---|---|
| DINO ViT-B/16 kNN (k = 11, kosinüs), 28 / 224 | 73.97 / 78.35 (AUC yok) | Doerrich S4 |
| DINOv2 ViT-B/14 tam ince ayar 224 | 92.02 ± 0.32 | Wu & Papież (bkz. §3) |
| DINO ViT-B/16 E2E 224 | 81.31 ± 1.05 (Doerrich) / 89.39 ± 0.12 (Wu) | protokol farkı ~8 puan |
| CLIP ViT-B/16 LP 224 | 81.93 (Doerrich) / 82.36 (Wu) | — |
| BiomedCLIP | **güvenilir sayı bulunamadı** | — |
| MedViT-S (224, MedMNIST v2 verisi, 100 ep, AdamW 1e-3) | 0.780 (AUC 0.937); T: 0.768, L: 0.773 | `manzari_2023_medvit` Tablo 3 |
| Google AutoML Vision | 0.768 | Yang 2023 T3 |

- **28 px bilgiyle bilinen tavan:** Yaklaşık **0.77–0.78**. Kaynaklar: DINOv2 LP 28 px 77.37; VGG16 E2E 28 px 76.58 (Doerrich); MedViT-S 0.780. MedViT, MedMNIST+'ten önce yayımlandı (2023), dolayısıyla 28 px veriden büyütme kullanmış olması kuvvetle muhtemel; bu bir çıkarımdır. **28 px kaynaklı bir yöntemle > 0.80 ACC, katalog hedefi olsa da literatürde görülmemiş düzeydir.** Hedefe ancak native 64/128/224 MedMNIST+ ile ulaşılması gerçekçi (kapsam içi).
- **Native 224 tavanı:** Hakemli kaynaklarda 84.7 (Doerrich) ile 88.5 (WS, Zhang & Li) arası. Hakemsiz kaynakta 92.02 (Wu & Papież). **Görülmemiş eşiği: > 0.93 ACC** veya **> 0.99 AUC** → sızıntı denetimi (bkz. §5).
- Not: Doerrich ile Wu arasında aynı model için 3–8 puan fark var (ör. ResNet-18 E2E 224: 82.33 ile 86.05; DINO E2E: 81.31 ile 89.39). Bu fark protokolden kaynaklanıyor (lr ızgarası, iterasyon sayısı, seçim). Kendi sayılarımız bu iki kaynak arasında yorumlanmalı.

---

## 5. Değerlendirme protokolü ve bilinen tuzaklar

- **Split:** Resmî 7 007 / 1 003 / 2 005 (7:1:2), MedMNIST v2 Tablo 2 ve `info.py`. MedMNIST+ (64/128/224) "aynı split ve aynı örnek indisleri"ni koruyor (`on_medmnist_plus.md`). 28 px görüntüler 3×600×450 kaynaktan küçültülmüş. Daha büyük boyutlar genel kural olarak aynı ilk kırpmadan yeniden boyutlandırılmış; DermaMNIST'e özel bir not yok.
- **Lezyon sızıntısı (kritik):** `abhishek_2025_dermamnistkalite`'ye göre MedMNIST, HAM10000'i **lesion_id'yi dikkate almadan** bölmüş. 10 015 görüntü 7 470 lezyona ait ve bunların **1 006'sı (%13.47) birden fazla bölmede** yer alıyor. Örtüşmeler: train–test **886 görüntü / 641 lezyon**; train–val 440 / 332; val–test 128 / 113; üç bölmede birden 51 / 40. Ayrıca metadata'da yer almayan 18 kopya çift var, 7'si bölmeler arasında. Düzeltilmiş sürümler: DermaMNIST-C (8 215 / 573 / 1 227) ve DermaMNIST-E. Bu sürümlerde ResNet-18 28 px sonucu **0.825 ± 0.006 ACC / 0.946 AUC** (Tablo 1, 3 koşu).
  - **Sonuç:** Rakiple kıyaslanabilirlik için resmî split kullanılır. Sızıntı **sınırlılıklar** bölümünde açıkça yazılır. Sızıntı, ezberleyen yöntemleri (kNN, yüksek kapasiteli ince ayar) kayırabilir; kNN ve E2E sonuçları bu uyarıyla yorumlanır. DermaMNIST-C kapsam dışı; yalnız tartışmada anılır.
- **ACC tanımı (Evaluator, `getACC`, çok sınıflı):** `accuracy_score(y_true, argmax(y_score))`. Eşik kullanılmaz.
- **AUC tanımı (`getAUC`, çok sınıflı):** Her sınıf için ayrı ikili `roc_auc_score` hesaplanır (bir-e-karşı-hepsi, OvR) ve **ağırlıksız ortalaması** alınır (makro OvR). Softmax uygulanmaz, bu yüzden skorlar olasılık olmalı. Resmî betik softmax uyguluyor. Kendi kodumuz da softmax olasılıklarını vermeli; logit verilirse sıralama korunduğu için sınıf başına AUC değişmez, ancak tutarlılık için olasılık kullanılmalı.
- **`medmnist.Evaluator(..., size=size)`:** v3.0.2 sürümünde `size_flag` hatası düzeltildi. 3.0.2 kullanılmalı.
- **Seed:** Resmî kodda yok. Sarmalayıcı seed ayarlamalı (bkz. §2.3).
- **Klinik iddia yasak** (alan-protokolleri V6). HAM10000 araştırma amaçlı bir benchmark'tır.

---

## 6. Lisans

| Varlık | Lisans | Kaynak |
|---|---|---|
| DermaMNIST (tüm boyutlar) | **CC BY-NC 4.0** | medmnist.com; README @805b742 ("except DermaMNIST under Creative Commons Attribution-NonCommercial 4.0 International"); `info.py` `license: CC BY-NC 4.0` |
| Diğer MedMNIST alt kümeleri | CC BY 4.0 | aynı |
| HAM10000 | **CC BY-NC 4.0** | Harvard Dataverse doi:10.7910/DVN/DBW86T, sürüm 4 (API) |
| MedMNIST kodu (her iki depo) | Apache-2.0 | medmnist.com / README |
| DINOv2 kodu ve ağırlıkları | Apache-2.0 | facebookresearch/dinov2 README |

→ Ticari olmayan akademik makale için uygun. Makalede atıf ve "NonCommercial" kaydı yer almalı.

---

## 7. Türkçe terim önerileri
doğrusal sonda (linear probe) · uçtan uca ince ayar (end-to-end fine-tuning) · test-zamanı artırma (test-time augmentation, TTA) · odak kaybı (focal loss) · ağırlıklı örnekleme (weighted sampling) · dengeli doğruluk (balanced accuracy) · makro OvR AUC (macro one-vs-rest AUC) · öz-denetimli öğrenme (self-supervised learning) · temel model (foundation model) · sıfır dolgu (zero padding) · lezyon düzeyinde bölme (lesion-level split) · veri sızıntısı (data leakage).

---

## 8. Kaynak listesi (URL · erişim 2026-10-10)
1. Yang ve ark. 2023, MedMNIST v2 — https://pmc.ncbi.nlm.nih.gov/articles/PMC9852451/ · https://doi.org/10.1038/s41597-022-01721-8 · Crossref API
2. Yang, Shi, Ni 2021, ISBI — https://arxiv.org/abs/2010.14925 · https://doi.org/10.1109/ISBI48211.2021.9434062
3. MedMNIST/experiments — https://github.com/MedMNIST/experiments (commits 70b6b3a, 8b0f553, 12e9f40; issues #5, #12, #14); ham dosyalar `MedMNIST2D/train_and_eval_pytorch.py`, `MedMNIST2D/models.py`
4. MedMNIST/MedMNIST — https://github.com/MedMNIST/MedMNIST (commit 805b742; `medmnist/evaluator.py`, `medmnist/info.py`, `README.md`, `on_medmnist_plus.md`; issue #34) · https://pypi.org/project/medmnist/ · https://medmnist.com/
5. Doerrich ve ark. 2025 — https://arxiv.org/html/2404.15786v3 (Ek Tablo S4) · https://doi.org/10.1038/s41598-025-92156-9 · kod: https://github.com/sdoerrich97/rethinking-model-prototyping-MedMNISTPlus (son commit b67b772, 2025-03-05)
6. Wu & Papież 2025 — https://arxiv.org/abs/2501.14685 · https://arxiv.org/html/2501.14685v1
7. Ferdous ve ark. 2025 (MedNet) — https://doi.org/10.1038/s41598-025-25857-w (Crossref kaynakça listesi)
8. Abhishek, Jain, Hamarneh 2025 — https://arxiv.org/html/2401.14497v2 · https://doi.org/10.1038/s41597-025-04382-5
9. Manzari ve ark. 2023 (MedViT) — https://arxiv.org/html/2302.09462v1 · https://doi.org/10.1016/j.compbiomed.2023.106791
10. Ren, Gu, Liu 2025 (MedAlmighty) — https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1527980/full
11. Zhang & Li 2026 — https://pmc.ncbi.nlm.nih.gov/articles/PMC13413053/ · https://doi.org/10.3390/jimaging12070293
12. Medeiros 2026 (TTA) — https://arxiv.org/abs/2604.09697 · https://arxiv.org/html/2604.09697v1
13. Oquab ve ark. 2024 (DINOv2) — https://arxiv.org/abs/2304.07193 · https://mlanthology.org/tmlr/2024/oquab2024tmlr-dinov2 · https://github.com/facebookresearch/dinov2
14. Tschandl ve ark. 2018 (HAM10000) — https://doi.org/10.1038/sdata.2018.161 · lisans: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
15. Codella ve ark. 2019 (ISIC 2018) — https://arxiv.org/abs/1902.03368
16. He ve ark. 2016; Lin ve ark. 2017; Caron ve ark. 2021; Radford ve ark. 2021 — Crossref / PMLR sayfaları (künyeler .bib'de)
17. Düşük güven, atıf önerilmez: Kadric ve ark. 2025 arXiv:2507.12961 (ikincil tablo); Gokmen & Bumgardner 2025 arXiv:2511.01610 (DINO-MX; DermaMNIST'te DINOv2-small LP 0.66 / kNN 0.76, çözünürlük ve split belirtilmemiş).
