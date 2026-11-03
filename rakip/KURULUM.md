# Rakip Kurulumu (K3): MedMNIST resmî ResNet-18 (28 px, sıfırdan) · DermaMNIST

*rakip-kurucu · 2026-10-10 · hedef D (Colab Pro, NVIDIA L4) · yerel GTX 1650 yalnız duman*

## 1. Pin (R1)
| bileşen | depo | pin | doğrulama |
|---|---|---|---|
| Eğitim betiği + model | https://github.com/MedMNIST/experiments | `70b6b3a7ad7afddff1df2a3b735235830fbdb142` (2024-07-11 "update pytorch training scripts"; tag yok) | `git log -1` = 70b6b3a; `MedMNIST2D/train_and_eval_pytorch.py` blob `09157bd8d6032be12cbf1eea0cfa8ccbd39914cb`, `MedMNIST2D/models.py` blob `24579d6612441aa1e1af63f3d246d4e61db822fd` |
| Veri sınıfı + Evaluator | https://github.com/MedMNIST/MedMNIST | `805b74237fb5bd2db376fd94e50646b6a8905d9d` = PyPI `medmnist==3.0.2` | venv'de `medmnist.__version__ == "3.0.2"` |

`scripts/rakip_kos.py` her koşumda iki rakip dosyasının git blob SHA1'ini ham bayt üzerinden hesaplar.
Yukarı akış dosyaları CRLF satır sonu içerir ve blob da CRLF'lidir. Uyuşmazlıkta koşum `HATA (R2)` ile durur.
Bu yüzden `.git` olmadan da (Colab paketi) commit'e bağlılık doğrulanır.

```bash
# WS kökünden (Git Bash)
cd rakip && git clone https://github.com/MedMNIST/experiments.git experiments
cd experiments && git checkout 70b6b3a7ad7afddff1df2a3b735235830fbdb142 && git log -1 --format='%H %ci %s'
```

### Bağımlılıklar
Deney ortamı `env/.venv` (bkz. `env/KURULUM.md`). Resmî betik `from tensorboardX import SummaryWriter` içe aktarır. Bu paket venv'de yoktu, eklendi:
```bash
UV=uv
$UV pip install --python env/.venv/Scripts/python.exe tensorboardX      # → tensorboardx 2.6.5 (+ protobuf 7.36.2)
$UV pip freeze --python env/.venv/Scripts/python.exe > env/requirements.lock   # 46 paket
```
Colab'da defter hücre 3 aynı çekirdek pinleri kurar: `torch==2.5.1`, `torchvision==0.20.1` (cu121 index),
`medmnist==3.0.2`, `timm==1.0.30`, `scikit-learn==1.9.1`, `numpy==2.2.6`, `tensorboardX==2.6.5`. Pinler `assert` ile doğrulanır.
`pip freeze` çıktısı `env/requirements-colab.lock`'a yazılır.

**Yama: YOK.** Rakip kodunda hiçbir değişiklik gerekmedi, `rakip/YAMALAR.md` bu yüzden yok. Sürüm uyumsuzluğu çıkmadı.
`torch 2.5.1` ile resmî betik değiştirilmeden çalışıyor. `pretrained=False` uyarısı yalnız `--resize` yolunda çıkar, o yol kullanılmıyor.

## 2. Köprü: `scripts/rakip_kos.py` (rakibe dokunmadan)
Köprü resmî modülü import eder ve resmî `main()`'i resmî CLI varsayılanlarıyla çağırır. Yaptığı müdahaleler ve gerekçeleri:
| müdahale | nasıl | neden |
|---|---|---|
| Seed | `random`/`numpy`/`torch`/`torch.cuda` seed + `cudnn.deterministic=True`, `benchmark=False` | Resmî kodda seed yok (R5). DataLoader `num_workers=0`, shuffle global torch RNG'sini kullanıyor |
| Veri kökü | `medmnist.DermaMNIST` ve `medmnist.Evaluator` yerine yalnız `root=--veri-kok` enjekte eden alt sınıflar | Resmî kod `root` vermiyor (varsayılan `~/.medmnist`). `download=False`, MD5 köprüde doğrulanıyor |
| Kayıt | Resmî `train()`/`test()` sarmalanıyor (argümanlar ve dönüş aynen), Evaluator'a giden softmax skorları kopyalanıyor | Epoch geçmişi, `progress.json`, son-epoch metrikleri, tahmin dosyaları için |
| CPU düşüşü | CUDA yoksa rc=97 ile çıkılıyor (`--cpu-izin` yalnız hata ayıklama içindir) | Sessiz CPU düşüşü yasak |

**Çıktılar (`{RUN_DIR}`):**
- `metrics.json`: düz sözlük, üç grup anahtar.
  - Seçilen epoch: `val_acc`, `val_auc`, `val_loss`, `val_dengeli_acc`, `test_acc`, `test_auc`, `test_dengeli_acc`, `train_acc`, `train_auc`.
  - Son epoch: `val_acc_son`, `val_auc_son`, `test_acc_son`, `test_auc_son`.
  - Koşum bilgisi: `en_iyi_epoch` (1 tabanlı), `epoch_sayisi`, `secim_val_auc_farki`, `egitim_sure_sn` = `sure_sn`, `param_sayisi`, `seed`, `cihaz`, `alt_grup`. `alt_grup`, seçilen modelin val'deki sınıf bazlı doğruluğudur (`sinif_0..6`).
- `progress.json` (her epoch): `adim`, `loss`, `val_acc`, `val_auc`, `val_loss`, `val_acc_eniyi`, `gecen_sn`. **Test değeri yazılmaz.**
- `gecmis.json`: epoch başına train/val/test.
- `tahmin_val.npy` ve `tahmin_test.npy`: seçilen epoch, softmax, float32, (N, 7).
- `tahmin_*_son.npy`, `etiket_val.npy`, `etiket_test.npy`.
- `rakip_bilgi.json`: commit, blob, protokol, veri MD5/SHA256, donanım, sürümler. Aynı alanlar `config.json`'a da eklenir.
- `resmi_cikti/`: resmî betiğin kendi çıktısı (`best_model.pth`, standart değerlendirme CSV'leri, Tensorboard).

**Test anahtarları** yalnız bu rakip köprüsünde yer alır. Bu şef kararıdır (`orkestrasyon_durumu.md`): replikasyon test'te bir kez yapılır ve baseline test sayıları dondurulur. K4 varyant köprüleri test yazmaz (M4).

ACC/AUC tanımı `medmnist.Evaluator` (3.0.2) ile aynıdır: argmax doğruluğu ve makro OvR AUC. Köprü val ACC'yi argmax ile bağımsız olarak yeniden hesaplayıp karşılaştırır.

## 3. Protokol eşleme (R3)
| öğe | resmî (makale + 70b6b3a) | bizim (aşama 3, `baseline`) | sapma ve etkisi |
|---|---|---|---|
| Epoch | 100 | 100 | yok |
| Optimizer | Adam, lr 0.001 (kodda sabit) | aynı (resmî kod) | yok |
| LR takvimi | MultiStepLR milestones `[0.5E, 0.75E]` = [50, 75], γ = 0.1 | aynı | yok |
| Batch | 128, train `shuffle=True`, `drop_last=False` | aynı | yok |
| Kayıp | `CrossEntropyLoss` | aynı | yok |
| Augmentasyon | yok: `ToTensor` + `Normalize(0.5, 0.5)` | aynı | yok. Augmentasyonun kapalı olması baseline'ı zayıflatma değildir, resmî protokol budur |
| Model | `models.ResNet18`: 3×3 stem, maxpool yok, 11.172.423 parametre | aynı | yok (`param_sayisi` beklenti kuralında sabitlendi) |
| Girdi | 28 px, `--resize` yok, `as_rgb=False` (DermaMNIST zaten 3 kanal) | aynı | yok |
| Model seçimi | her epoch val AUC, `cur_auc > best_auc` → `deepcopy` | aynı (resmî kod) | yok. Ek olarak son epoch (`*_son`) yazılıyor: 2023-05 öncesi `best_model = model` hatası yüzünden makale sayıları muhtemelen son epoch'a ait (literatur-cipa.md §1.4). Kapı birincil olarak `test_acc` kullanır |
| Değerlendirme | `medmnist.Evaluator`, softmax olasılığı | medmnist 3.0.2 (`size_flag` düzeltmeli) | yok |
| Seed | yok; makale "en az 3 deneme" | {0,1,2,3,4} + cuDNN deterministik | Tekrar üretilebilirlik için gerekli. Maliyet: GTX 1650'de eğitim adımı 14,35 s/epoch (deterministik) vs 11,76 (değil), yani +%22. Toplam epoch süresine etkisi yaklaşık +%11. Beklenen doğruluğu değiştirmez |
| Seed sayısı | ≥3 | 5 | Küçük veri için M5 (≥5); ortalama daha kararlı |
| Veri kökü / indirme | `~/.medmnist`, `--download` | `data/medmnist`, `download=False`, MD5 = `0744692d530f8e62ec473284d019b0c7` | Yok (aynı dosya, resmî MD5) |
| Yazılım | Python 3.6, PyTorch 1.3.1 (README) | Python 3.11 (yerel) / Colab Python, PyTorch 2.5.1+cu121 | Sürüm farkı. Adam, BN ve CE varsayılanları aynı. Küçük sayısal farklar beklenir, sistematik etki beklenmez. SARI çıkarsa ilk teşhis adayı budur |
| Donanım | belirtilmemiş | NVIDIA L4 24 GB (sabit sınıf); conv'larda TF32 (PyTorch varsayılanı `cudnn.allow_tf32=True`) | TF32 conv sayısal hassasiyeti FP32'den düşüktür, sıfırdan CNN eğitiminde doğruluk etkisi bilinmiyor/küçük. Kapatılmadı, çünkü resmî kod da varsayılanı kullanıyor |

**Kısaltılmış koşumlar (ölçüm değildir, replikasyona girmez):**
| aşama | varyant adı | seed | epoch | milestones (resmî kod `0.5E/0.75E`) | amaç |
|---|---|---|---|---|---|
| 1 duman | `baseline_duman` | 0 | 3 | 1.5 / 2.25 | Colab'da kod/pin/veri yolu |
| 2 ön-uçuş | `baseline_onucus` | 0, 1 | 20 | 10 / 15 | seed bağlılığı, öğrenme, süre |

- **Not (şefin talimatına göre düzeltme):** Resmî kod milestone'ları sabit 50/75 tutmaz, `0.5·num_epochs` ve `0.75·num_epochs` olarak hesaplar.
  20 epoch'luk ön-uçuşta takvim bu yüzden 10/15'e ölçeklenir. Bu resmî kodun kendi davranışı. Köprü bunu değiştirmez. Ön-uçuş tam protokolden yalnız epoch sayısında sapar.
- **Ön-uçuşta 2 seed (0, 1):** Talimat yalnız seed 0 diyordu. `farkli_seedler_farkli` kuralı ancak ≥2 seed ile sınanabildiği için seed 1 eklendi (beklenti şablonu: ön-uçuş ≥2 seed). Ek maliyet L4'te yaklaşık 3–4 dk.
- **Varyant adları neden ayrı:** `replikasyon_kapisi.py` ve `leaderboard.py` kuyruk ve aşama ayırmaz. `varyant == "baseline"` ve `rc == 0` olan TÜM koşumları alırlar. Duman ve ön-uçuş `baseline` adını taşısaydı 3 ve 20 epoch'luk sonuçlar replikasyon ortalamasına karışırdı.

## 4. Yerel duman (GTX 1650, FP32; kod doğruluğu, ÖLÇÜM DEĞİL)
Kök `runs_yerel_duman/`. Bu klasör pakete girmez, `runs/`'a yazılmadı.
```bash
WS=<WS>; PY=$WS/env/.venv/Scripts/python.exe
SK=<WS>/tools
cd $WS && PYTHONIOENCODING=utf-8 $PY $SK/kosum.py --kok $WS/runs_yerel_duman --varyant baseline_duman --seed 0 --asama 1 --gpu-zorunlu \
  --kural '{"nan_kes":true,"ilerleme_zorunlu_sn":300,"sabit_sn":300,"min_adim":30,"alt_sinir":{"val_acc_eniyi":0.60}}' \
  -- $PY $WS/scripts/rakip_kos.py --seed 0 --epoch 3 --veri-kok data/medmnist --out "{RUN_DIR}/metrics.json"
```
| koşum | rc | süre | ep1 train_loss | ep1 val_acc / val_auc | son val_acc / val_auc | not |
|---|---|---|---|---|---|---|
| `baseline_duman` s0, 3 ep | 0 | 93,3 s (eğitim 80,7) | 0.98551540 | 0.6919 / 0.8701 | 0.7089 / 0.9067 (ep 3 seçildi) | tüm anahtarlar sonlu; `secim_val_auc_farki` = 0; param 11.172.423 |
| `baseline_duman_1ep` s1, 1 ep | 0 | 50,4 s | 1.00536053 | 0.6969 / 0.8445 | — | seed 0'dan **farklı** (R5 ✓) |
| `baseline_duman_1ep` s0, 1 ep | 0 | 45,2 s | 0.98551540 | 0.6919 / 0.8701 | — | 3 ep koşumunun 1. epoch'uyla **bit-bit aynı** (determinizm ✓) |
| `kural_testi` s0 (kasıtlı) | 95 | 45,1 s | — | — | — | `alt_sinir {"val_acc_eniyi":0.99}` 1. epoch'tan sonra öldürdü. Canlı erken durdurma anahtarı doğru ✓ |

Tüm çıktı dosyaları (`metrics.json`, `progress.json`, `gecmis.json`, `tahmin_*.npy`, `etiket_*.npy`, `rakip_bilgi.json`, `resmi_cikti/`) oluştu. `config.json`'a `donanim`/`veri`/`rakip`/`protokol` eklendi.

## 5. Süre tahmini (L4) ve gerekçesi
GTX 1650'de bir epoch 23,4 s sürdü. Bileşen ölçümüne göre (scratchpad `profil.py`):
- GPU eğitimi 14,4 s (7.007 örnek, deterministik).
- GPU ileri geçiş yaklaşık 5,2 s (train+val+test değerlendirmesi, 10.015 örnek; resmî kod her epoch üç bölümü de değerlendirir).
- CPU veri hattı yaklaşık 5,7 s (17.022 örnek; PIL→`ToTensor`→`Normalize`, `num_workers=0`, tek iş parçacığı).

L4'ün FP32 gücü yaklaşık 10 kat (30 vs 3 TFLOPS) ve conv'larda TF32 var. GPU kısmı bu yüzden yaklaşık 2–3 s/epoch olur. CPU veri hattı hızlanmaz, Colab Xeon'larında tek çekirdek daha yavaş olabilir: 6–8 s. Sonuç olarak:
- epoch ≈ 8–12 s → **100 epoch ≈ 15–20 dk/seed**
- aşama 3 (5 seed) ≈ 75–100 dk
- aşama 1 ≈ 1 dk, aşama 2 ≈ 2 × 3–4 dk
- kurulum + veri ≈ 5 dk
- **Toplam ≈ 1,5–2 saat** (L4-saat bütçesi 24'ün yaklaşık %8'i)

Zaman sınırları (kuyruk satırı başına): a1 15 dk, a2 30 dk, a3 75 dk.

## 6. Colab akışı (hedef D)
1. Yerelde paketi üret: `PYTHONIOENCODING=utf-8 <uv-python> colab/paket_hazirla.py` → `colab/3-1-paket.zip`.
2. Kullanıcı `3-1-paket.zip` ve `colab/k3_baseline.ipynb` dosyalarını Drive `MyDrive/yz/` altına yükler, L4 seçer, *Tümünü çalıştır* (`colab/OKU-BENI.md`). Defter adımları:
   1. Drive'ı bağlar, paketi açar.
   2. L4 değilse `assert` ile durur.
   3. Pinli kurulumu yapar ve doğrular.
   4. DermaMNIST 28'i indirir, MD5 ve SHA256'yı `assert` ile doğrular.
   5. `python orkestra/kuyruk.py --kok . kos k3-baseline --kademeli --paralel 1` komutunu koşar.
   6. 5b: oturum koptuysa devam eder.
   7. `kosum.py durum` ve aşama-3 kural raporunu alır, `k3_baseline_sonuc.zip`'i hazırlar (`.pth` hariç).
3. Zip `colab/gelen/` altına gelir. Şef açar ve kendisi koşar: `kosum.py durum`, `replikasyon_kapisi.py` (birincil `test_acc`, ikincil `test_auc`), `leaderboard.py`.

**Devam (resume):** `kuyruk.py` bitmiş satırları **atlamaz**. Aynı ad yeniden koşulursa tamamlanmış seed'ler tekrarlanır, `baseline` seed'leri de çift sayılır. `kuyruk.py` değiştirilmedi. Defterin 5b hücresi bunu dışarıdan çözer:
- Ölü koşumları `kosum.py durum --yetim-damgala` ile rc=96 damgalar (replikasyon rc≠0'ı almaz).
- Kalan satırlarla `k3-baseline-devamN` kuyruğunu yazar. Beklenti dosyası bayt bayt aynı kopyalanır.
- Aşama 2 daha önce geçtiyse yalnız bitmemiş aşama-3 seed'lerini koşar, geçmediyse hepsini baştan koşar.

**Toplam süre (R6):** Colab koşumundan sonra `runs/kuyruklar/*.KUYRUK.done` → `sure_sn` buraya yazılacak.
