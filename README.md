# DermaMNIST'te bölme protokolü: resmî bölme ve lezyon-ayrık DermaMNIST-C (ResNet-18 ve DINOv2-B)

Bu depo, aşağıdaki makalenin kodunu, ön-kayıtlarını ve dondurulmuş sonuçlarını içerir:

> `<MAKALE_KÜNYESİ — yazarlar, başlık, dergi, yıl, DOI (kabul sonrası doldurulacak)>`

**English summary.** This repository accompanies a pre-registered comparison of a ResNet-18 baseline (the official
MedMNIST 28-px benchmark code, pinned) and DINOv2-B variants on DermaMNIST. We ask whether the foundation-model
advantage measured on the official image-level split still holds under the lesion-disjoint DermaMNIST-C split
(Abhishek et al., 2025), where images of the same lesion never cross train/val/test. All hypotheses, decision rules
and the test-time analysis script were frozen (SHA-256) before the test set was opened. The official test set was
opened once under a lock file, and each arm was evaluated on the C-test once. The repository holds code, hypothesis cards, expectation files,
run metadata and the frozen result files with their SHA-256 sidecars. Data and model weights are not included.
Run `python tools/dogrula.py` to verify the pre-registration chain.

## Çalışmanın amacı
- **Soru:** DermaMNIST'te DINOv2-B'nin ResNet-18'e üstünlüğü bölme protokolüne bağlı mı?
  Karşılaştırılan iki bölme: resmî MedMNIST bölmesi (görüntü düzeyi, aynı lezyonun görüntüleri bölümler arasında
  dağılabilir) ve lezyon-ayrık **DermaMNIST-C** bölmesi (8.215 / 573 / 1.227).
- **Kollar:** resmî ResNet-18 (28 px, sıfırdan; `MedMNIST/experiments@70b6b3a`), ImageNet önağırlıklı ResNet-18
  (224 px ince ayar), DINOv2-B doğrusal sonda ve kısmi ince ayar, DINOv2-S (Pareto satırları).
- **Ön-kayıt ve test kilidi:** her hipotez koşumdan önce kart olarak yazıldı (`hipotezler/H*.md`). Model ve
  hiperparametre seçimi yalnız val üzerinde yapıldı. K5 test protokolü (`hipotezler/K5-ONKAYIT.md`, `-b.md`) ve
  analiz betiği (`scripts/k5_test.py`) test açılmadan SHA256 ile donduruldu (`sonuclar/k5_dondurma.json`). Resmî
  test bir kez açıldı (`sonuclar/k5_test.KILIT`, `ikinci_acilis: null`). C-test her kol için bir kez açıldı.
- **İstatistik:** eşleştirilmiş görüntü-bootstrap (B=2000, rng 0) ve seed-eşli tek yönlü t, Holm düzeltmesiyle
  (`varyantlar/ortak/eslestirilmis_bootstrap.py`, `scripts/k5_test.py`).

## Dizin yapısı
| yol | içerik |
|---|---|
| `varyantlar/` | Varyant kodu: `ortak/` (veri, değerlendirme, bootstrap, kayıt), `r18_imagenet/`, `dinov2_sonda/`, `dinov2_ozn/`, `dinov2_ft/`, `k5/` (C bölmesi, C1 köprüleri), `logit_ayari/`, `tta/`, `topluluk/`; `KOMUTLAR.md` |
| `scripts/` | Deney betikleri: veri indirme/analiz (`veri_hazirla.py`, `indir_dermamnist_c.py`, `c_bolme_npz.py`), rakip köprüsü (`rakip_kos.py`), K5 analizi (`k5_test.py`, `k5_dondurma.py`, `k5_c_denetim.py`), tur tahlilleri, Colab arşiv/izleme |
| `tools/` | Orkestra betikleri (`kosum.py`, `kuyruk.py`, `beklenti_kontrol.py`, `leaderboard.py`, `replikasyon_kapisi.py`) ve `dogrula.py` (SHA256 doğrulaması) |
| `rakip/` | Rakip kurulum notu ve pinli resmî kodu getirme betiği. Resmî kod depoda yok |
| `colab/` | Colab L4 hattı: defterler (`hat/3-1-colab-*.ipynb`), defter üreticileri, `fp32_kos.py` (TF32 kapatma), `veri_listesi.json` (URL, MD5, SHA256) |
| `hipotezler/` | Hipotez kartları H01–H13, sıralamalar, K5 ön-kaydı (`K5-ONKAYIT.md`, `-b.md`), ön koşul kontrolü, C1 uygulama notu |
| `kuyruk/` | Kuyruk tanımları (`*.jsonl`), beklenti dosyaları (`*.beklenti.json`), gruplar, C bölmesi beklenen özeti |
| `sonuclar/` | Leaderboard'lar, replikasyon kapısı, tur tahlilleri, `beklenti-*.md` kapı raporları, `karar.md`, dondurulmuş K5 sonuçları ve `.sha256` dosyaları |
| `runs_meta/` | Her koşumun JSON damgaları (`config.json`, `metrics.json`, `varyant_bilgi.json`, `fp32_durum.json` …): `runs/` yerel, `runs_colab/` Colab. Arşiv manifesti `runs_colab/ARSIV_MANIFEST.json` |
| `data/` | `data_manifest.md/.json`, bölme ve eşleme dosyaları (`splits/*.csv`, yalnız indeks/kimlik/etiket), indirme kayıtları. Görüntü yok |
| `env/` | Sürüm pinleri (`requirements.lock`), kurulum notu, Colab ölçüm ortamı (`ortam-colab-olcum.md`) |
| `gunluk/` | Deney günlüğü |
| `makale/` | Makale kanıt defteri (`evidence/evidence.json`), tablo/figür üreticileri, üretilmiş tablolar ve figür verileri |
| `orkestrasyon_durumu.md`, `konu.md` | Kullanıcı kararlarının kaydı, konu kartı |
| `HIJYEN_DEGISIKLIKLERI.json` | Depo kopyasında kişisel yolların yer tutuculara çevrildiği dosyalar (önceki/sonraki SHA256) |

**Yer tutucular:** `<WS>` deney çalışma alanının kökü, yani bu deponun köküdür. `<MAKALE_WS>` = `makale/`.
`<DRIVE_KOK>` Google Drive'da proje klasörünün üst klasörüdür (Colab'da örneğin `/content/drive/MyDrive/<klasör>`).
`<PYTHON311>` bir Python 3.11 yorumlayıcısıdır. Bu yer tutucular kayıt ve damga dosyalarında yalnız bilgi amaçlıdır.
Kod çalıştırılırken yolu argüman ya da ortam değişkeni olarak verin.

## Kurulum
İki ortam kullanıldı. Ayrıntı: `env/KURULUM.md`, `env/ortam-colab-olcum.md`.

| amaç | ortam |
|---|---|
| Yerel (duman testleri, K3 replikasyonu, analiz) | Python 3.11.16; `torch==2.5.1+cu121`, `torchvision==0.20.1+cu121`, `timm==1.0.30`, `medmnist==3.0.2`, `scikit-learn==1.9.1`, `numpy==2.2.6`; tam liste `env/requirements.lock` (46 paket) |
| Colab ölçüm koşumları (K4/K5, raporlanan tüm test/C1 sayıları) | NVIDIA L4 24 GB; Python 3.13.15, `torch==2.11.0+cu130`, `torchvision==0.26.0+cu130`, `timm==1.0.30`, `medmnist==3.0.2`, `scikit-learn==1.6.1`, `numpy==2.1.3`. Her koşumun `runs_meta/runs_colab/<id>/config.json` → `donanim` alanında kayıtlı |

```bash
python3.11 -m venv env/.venv
env/.venv/bin/pip install -r env/requirements.lock \
  --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple
```

## Veri (depoda yok)
DermaMNIST (MedMNIST v3, Zenodo 10.5281/zenodo.10519652) ve DermaMNIST-C (Zenodo 10.5281/zenodo.12739457) programatik
olarak indirilir. Betikler MD5'i resmî değerle karşılaştırır. URL, MD5, SHA256 ve bayt değerleri
`colab/hat/veri_listesi.json` ve `data/data_manifest.md` içindedir.
```bash
python scripts/veri_hazirla.py --asama hepsi --boyutlar 28 224          # MedMNIST DermaMNIST → data/medmnist/
python scripts/indir_dermamnist_c.py --asama hepsi --boyutlar 28 224   # DermaMNIST-C → data/dermamnist_c/
python scripts/c_bolme_npz.py --kip c1 --out <dizin>/metrics.json  # C1 verisi: resmî pikseller C bölmesine göre dizilir
                                                                    # → data/dermamnist_c_resmipiksel_{28,224}.npz; içerik SHA256 = kuyruk/k5_c_bolme_beklenen.json
```
| dosya | MD5 (resmî) | SHA256 |
|---|---|---|
| `dermamnist.npz` (28 px) | `0744692d530f8e62ec473284d019b0c7` | `1a309fec2e33bb6aba88e7d078e5ccbb9736c84a0b415ac197eb3c8fa331e050` |
| `dermamnist_224.npz` | `8974907d8e169bef5f5b96bc506ae45d` | `0143c9ae425867916d9d2a7769ed1a9b7b8c5eb9324a4bb384be6ef4d42da22d` |
| `dermamnist_corrected_28.npz` | `96e2862980877d45d9cacb5c25de6950` | `c3d9cbd9008a0dcb76943d5cf38681bdc5166effc79f9e215657b859b9226aa8` |
| `dermamnist_corrected_224.npz` | `84920fb70c83b234c295b6f0d4ae2bc0` | `f8d80f4fac5a0a1f53256e0c337456a8ab8020828a94a9eb8a1eb8ef5d748486` |
| `DermaMNIST-C.csv` | `1c54bc9f483e97a9d7dbbde076a45900` | `c27194d338986463966e1694d5b49fda933222368f04b316c702ac2e8a0d4f5c` |

C1 npz'nin beklenen içerik özeti `kuyruk/k5_c_bolme_beklenen.json` dosyasındadır. Lezyon eşlemeleri
`data/splits/dermamnist_lezyon_esleme.csv` (resmî bölme ve HAM10000 `lesion_id`) ile `data/splits/dermamnist_c_esleme.csv`
dosyalarındadır. Önağırlıkların SHA256 değerleri `colab/hat/veri_listesi.json` → `agirliklar` altındadır:
`timm/vit_base_patch14_dinov2.lvd142m`, `vit_small_patch14_dinov2.lvd142m`, `torchvision resnet18-f37072fd`.

## Koşumların yeniden üretimi
- **Donanım ve sayısal ayarlar:** Colab Pro, NVIDIA L4. Her komut `colab/hat/fp32_kos.py` ile sarılır. Bu sarmalayıcı
  TF32'yi kapatır (`allow_tf32=False`, `fp32_precision="ieee"`, `NVIDIA_TF32_OVERRIDE=0`), cuDNN'i deterministik
  yapar (`benchmark=False`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`) ve AMP kullanılmaz. Durum her koşumda `fp32_durum.json`'a yazılır.
- **Seed'ler:** her kol için 0–4 (5 seed).
- **Akış:** kuyruk satırları `kuyruk/*.jsonl` dosyalarındadır. `tools/kuyruk.py` her satırı `tools/kosum.py`
  sarmalayıcısıyla koşar. Aşama kapıları (duman → ön-uçuş → tam) `*.beklenti.json` kurallarıyla
  `tools/beklenti_kontrol.py` tarafından denetlenir. Raporlar `sonuclar/beklenti-*.md` dosyalarındadır.
- **Colab:** `colab/hat/README.md`. Defterin ilk hücresinde `SEKME` ve `KUYRUK` seçilir. `YZ = pathlib.Path('<DRIVE_KOK>')`
  satırı kendi Drive klasörünüze göre doldurulur. Kod paketi `colab/hat/drive_paketle.py --drive <DRIVE_KOK>` ile hazırlanır.
- **K5 analizi** (test kilidi dosyası yoksa betik testi açmaz):
  ```bash
  python -I scripts/k5_test.py --kip test --runs <runs_dizini> --test-kilidi-acik
  python -I scripts/k5_test.py --kip c1   --runs <runs_dizini> --test-kilidi-acik
  ```
- **Makale tabloları/figürleri:** `makale/README.md`.

## Ön-kayıt zinciri ve SHA256 doğrulaması
Zincirin sırası:
1. Hipotez kartları (`hipotezler/H*.md`).
2. Val'de seçim (`sonuclar/tahlil-tur*.md`, `SIRALAMA*.md`).
3. K5 ön-kaydı ve dondurma: `sonuclar/k5_dondurma.json`, 2026-10-11 18:54:44. Ön-kayıt, analiz betiği, C bölme
   betikleri, k5 kuyruk ve beklenti dosyaları ile val tahminlerinin SHA256 değerlerini içerir.
4. Test açılışı: `sonuclar/k5_test.KILIT`, 21:12:03.
5. `k5_test_sonuc.json` ve `.sha256`; `k5_dondurma_ek_c1.json`; ardından C1 (`k5_c1_sonuc.json` ve `.sha256`).

Doğrulama: ayrıntılar `DOGRULA.md` dosyasında.
```bash
python tools/dogrula.py            # çıkış kodu 0 = geçti
```
Dondurmadan sonra yapılan C1 uygulama düzeltmeleri `hipotezler/K5-C1-UYGULAMA-NOTU.md` dosyasında belgelenmiştir:
`c_bolme.py`, `c_rakip_kos.py` ve `c_bolme_sar.py`. Ayrıca `gruplar.json`'a v2/v3 kuyrukları eklendi. Doğrulayıcı bu
dört dosyayı "BELGELİ SAPMA" olarak raporlar ve düzeltme sonrası özetleri nottaki tabloya karşı denetler.

## Rakip kodu
Resmî ResNet-18 eğitim betiği bu depoya **kopyalanmamıştır**. Kaynak: <https://github.com/MedMNIST/experiments>,
commit `70b6b3a7ad7afddff1df2a3b735235830fbdb142` (`MedMNIST2D/train_and_eval_pytorch.py`, `models.py`). Kodu getirmek
için `bash rakip/getir.sh` kullanılır. Rakip koduna dokunmadan çalışan köprüler: `scripts/rakip_kos.py` (resmî bölme)
ve `varyantlar/k5/c_rakip_kos.py` (DermaMNIST-C; yalnız veri yolu uyarlaması). Köprü, rakip dosyalarının git blob
SHA1 değerlerini her koşumda doğrular. Ayrıntı: `rakip/KURULUM.md`.

## Lisans
- **Kod:** MIT (`LICENSE`).
- **Veri:** DermaMNIST/MedMNIST ve DermaMNIST-C görüntüleri **CC BY-NC 4.0** lisanslıdır (ticari kullanım yasak) ve
  bu depoda **yer almaz**. Depoda yalnız indirme betikleri, MD5/SHA256 değerleri ve indeks/kimlik düzeyinde bölme
  eşlemeleri bulunur. Eşlemelerde HAM10000 `image_id`, `lesion_id` ve `dx` alanları vardır; bunlar HAM10000
  metadata'sından türetilmiştir (Tschandl vd., 2018, CC BY-NC 4.0). Bu dosyaları kullanırken özgün veri
  kümelerine atıf yapın.
- Resmî rakip kodu Apache-2.0 lisanslıdır ve depoda yer almaz.

## Atıf
```bibtex
@article{<anahtar>,
  title   = {<MAKALE_BAŞLIĞI>},
  author  = {<YAZARLAR>},
  journal = {<DERGİ>},
  year    = {<YIL>},
  doi     = {<DOI>}
}
```
