# H14 / E1 uygulama notu (deney-yürütücü, 2026-11-03; hiçbir e1 koşumundan ÖNCE)

*Kapsam: yalnız uygulama.*
- Kartın (`H14.md`, SHA256 `1991c244…5d2a`, mtime 22:39:10) üst kısmı, beklenti taslağı ve kapsamı DEĞİŞMEDİ.
- `SIRALAMA.md` (SHA256 `e017899a…0209`, mtime 2026-10-11 17:53:06) ve eski kartlar okunmadı/yazılmadı.
- Rakip koduna dokunulmadı.
- C-test açılmadı. Kuru analiz yalnız K5 C1'de zaten açılmış arşiv tahminlerini okudu.
- Kartta yazılı olmayan ya da kartın ifadesinden ayrılan her uygulama kararı aşağıda numaralı olarak listelenir.

## §1 Kod: varsayılanı kapalı bayraklar ve bit-eşitlik kanıtı
`varyantlar/dinov2_ft/ince_ayar_vit.py`
- Eski SHA256: `4bccfa72…f87f9f`. Bu, K5 C1 paketindeki dosyayla aynıdır.
- Yeni SHA256: SHA256 tablosunda (aşağıda).
- Yeni bayraklar: `--onbellek-gorunum {1,8}`, `--augment-d4`, `--ozdeslik-ref`, `--disk-min-gb`.
- Eski satırlardan yalnız 3'ü değişti, geri kalanı eklemedir:
  - `girdi` imzasına isteğe bağlı `k` eklendi.
  - Eğitim çağrısı `girdi("train", j)` biçiminde kaldı; yalnız D4 açıkken `k` geçer.
  - Config'teki `augment` alanı D4 kapalıyken yine `"yok"` yazar.

KANIT_YER_TUTUCU

## §2 Sapma: `h10r_c1_duman` satırına `--ozdeslik-ref @e1-h14/h14_c1_d4kapali_duman` eklendi
- Kart, e1-h10r satırları için "`c_kol(..., "h10", ...)` ile aynı komut (yeni bayrak yok)" diyor. Aynı kartın h10r beklentisi ise a1'de `kontrol_*` eşini, yani |Δ ilk_adim_loss| [0, 1e-6] kuralını istiyor. Bu anahtarı yazmanın tek yolu, satırın kontrol koluyla kendi kendini karşılaştırması.
- Kontrol kolunun run_id'si Colab'da doğacak; şu an bilinmiyor. Bu yüzden sembolik başvuru kullanıldı: aynı seed, rc ∈ {0, 5}, tek koşum.
- Bayrak yalnız eğitim **sonrası** dosya okur. Eğitim yolu H10 ile aynıdır: `--onbellek-gorunum` varsayılanı 1'dir, `girdi` H10 satırıdır. §1'deki "yeni varsayılan ↔ eski" bit-eşitliği bunu kapsar.
- Diğer h10r satırları (a2, a3, C-test) k5-c1-B h10 satırlarıyla aynı. Tek farklar varyant adı ve bütçe sarmalayıcısı (§5).

## §3 Kural (f) Durum C → kontrol kolu rc = 5
- Kartın beklenti notu, Durum A/B/C sınıflamasının analizde yapıldığını ve kapının yalnız kaba ihlalleri yakaladığını söyler.
- Kuyruk ise a1 → a2 → a3 diye kesintisiz ilerler. Tolerans içi bir Durum C (sürümler aynı ama bit-bit değil) kapıdan geçerdi ve a3 ile C-test açılırdı.
- Bunu önlemek için kontrol kolu Durum C'de `metrics.json`'u yazar ve **5 koduyla çıkar**. Kapı rc ≠ 0'ı mekanik İHLAL sayar, kuyruk a1'de durur. Bu, kural (f)'deki "DUR, teşhis, -v2" hükmünün mekanik karşılığıdır.
- Beklenti dosyası değişmedi.
- Sınıflama:
  - Kodda: `kontrol_hesapla`.
  - Analizde aynı eşiklerle yeniden: `e1_h14_analiz.durum_sinifla`.
  - Sürüm karşılaştırması: torch, CUDA, cuDNN ve timm, referansın `config.json` → `donanim` alanına karşı.

## §4 Kural (b) ve (g) kod içinde de DUR
- **(b)** 8 görünüm yazılacaksa, yazımdan önce `shutil.disk_usage(OZN_DIZIN).free` ölçülür. Değer < 40 GB ise `SystemExit` (DUR) verilir.
  - Değer `disk_bos_gb_once` olarak yazılır, ama yalnız önbelleği yazan satırda.
  - Defter hücre 8 aynı denetimi kuyruktan önce yapar.
- **(g)** Her 8-görünüm koşumunda, ilk 8 train görüntüsü için iki ölçüm yapılır:
  - Her k için tam ağ (fp32, T_k(görüntü)) ↔ önbellek g_k farkı: normlanmış CLS maks |fark|. 5e-3 eşiği aşılırsa DUR.
  - Görünüm çifti × görüntü bit-özdeşlik sayısı (28 çift × 8 = 224 sınama). 0'dan büyükse DUR.
  - Böylece a2/a3'te de (kapının yalnız bilgi verdiği a3 dahil) bozuk önbellekle eğitim yapılmaz.
  - İlk 8 train görüntüsü önbellek varken npz'den zip akışıyla okunur (`npz_ilk_satirlar`); tüm dizi açılmaz.
- **(g) D4 dumanı ↔ kontrol:** kapıdaki `kollar_farkli` val_acc'yi karşılaştırır. `ilk_adim_loss` eşitliği (|Δ| ≤ 1e-6) analizde DUR koşuludur.

## §5 Kural (h) bütçe: `colab/erteleme_kapisi.py --kuyruk-onek e1-`
- `--kuyruk-onek` ve `--kural` seçenekleri eklendi.
  - Varsayılan davranış değişmedi. Kendi-testi genişletildi ve geçti.
- e1-h14 ve e1-h10r'nin a2, a3 ve C-test satırları şu önekle sarıldı: `python colab/erteleme_kapisi.py --esik 5.0 --baslangic-sa 0 --kuyruk-onek e1- -- …`
  - E1 kuyruklarının birleşik koşum süresi 5,0 L4-sa'i aşarsa satır başlamaz (rc = 94). Kuyruk durur, kullanıcıya rapor edilir.
  - Ekim koşumları sayılmaz.
- a1 satırları sarılmadı, çünkü ilk aşamadır.

## §6 e1-h10r beklentisi (karttaki tanımın uygulanması)
- `k5_kuyruk_yaz.beklenti_c("e1-h10r", ["h10"], "c1")` ile üretildi. Bunlar k5-c1-B'nin h10 kurallarıdır, varyant adları `h10r_*` olarak değiştirildi.
- `min_kosum` değerleri: a1 1; a2 1 (kart: "yalnız s0"); a3 10 (5 eğitim + 5 C-test).
- Eklenen kurallar:
  - a1 `kontrol_ilk_adim_loss_fark` [0, 1e-6] (kart).
  - a1 `kontrol_p_maks_fark` [0, 1e-5]. Kural (f) Durum B metninden alındı. Kartın h10r beklenti cümlesi yalnız ilk_adim_loss'u anıyor; bu kural daha sıkıdır, gevşetme değildir.
  - a3 `veri_c` [1, 1] (kart).
- İlk satır `c_bolme_veri_c1`'dir. Ayrı Colab oturumunda C1 verisi yerel diskte yoktur.

## §7 Defter (`colab/e1-h14.ipynb`, üretici `colab/hat/defter_uret_e1.py`)
1. **torch sabitleme** (kart "Ortam"):
   - `torch==2.11.0+cu130` ve `torchvision==0.26.0+cu130` PyTorch cu130 dizininden kurulmaya çalışılır.
   - Kurulan torch CUDA/L4 görmezse Colab sürümüne geri dönülür.
   - Sonuç `env/ortam-colab-<oturum>.json` → `torch_sabitleme` ve `ekim_farki` alanlarına yazılır.
   - Not: cuDNN sürümü pip bağımlılığına göre Ekim'deki 92700'den farklı olabilir. Kontrol kolu bunu Durum B olarak yakalar.
2. **Sürüklenme denetimi** E1'in ilk oturumuna göre yapılır (`env/ortam-colab-e1-ilk.json`):
   - Herhangi bir `e1-*` kuyruğu başlamışsa ve çekirdek sürüm farklıysa → DUR.
   - Böylece H10r ile H14 aynı ortamda koşar.
   - Hattın Ekim `ortam-colab-ilk.json` denetimi E1'de kullanılmaz. Durum B'de sürüm farkı beklenen bir durumdur, kart onu H10r ile çözer.
3. **Oturum devamı:** devam kuyruğu a1 kapısını geçmişse, `c_bolme_veri_c1` satırı yeniden koşmaz.
   - Yeni oturumda C1 npz yerel diskte olmadığından, defter hücre 8 aynı betiği (`scripts/c_bolme_npz.py --kip c1`, içerik SHA256 denetimli) kuyruk dışında koşar.
   - Kayıt: `durum/c_bolme_<oturum>/`.
   - 8 görünümlü önbellek ilk eğitim satırında kod tarafından yeniden yazılır (≈ 6 dk).
4. **Durum B dalı:** e1-h14 kuyruğu tek parçadır. Durum B'de H14 a2/a3, `h10r_c1_duman`'dan ÖNCE koşar; ikinci sekmede paralel başlatılırsa birlikte koşar.
   - h10r kontrolü tutmazsa `e1_h14_analiz.py` DUR verir ve K1 yazılmaz.
   - H14 C-test bu durumda yine bir kez açılmış olur (kural i ihlali değildir).
   - Açık soru 1'e bakınız.

## §8 Analiz (`scripts/e1_h14_analiz.py`)
- `k5_test.py` içe aktarılır. SHA256'sı `k5_dondurma.json` ve `k5_c1_sonuc.json` ile aynı olmalıdır (13a71cf1…); değilse DUR.
- Kart dışı hesap yalnız iki kalemdir:
  - **mel duyarlılığı bootstrap'ı**: aynı W ile hesaplanır, (W·doğru·1[y=mel]) / (W·1[y=mel]).
  - **iki yönlü seed t**: `scipy.stats.ttest_rel` varsayılanı. `k5_test.kiyas` tek yönlüdür; kart iki yönlü istiyor.
- Kural (a) eksik seed durumunda bütün kollar H14'ün seed kesişimine indirilir ve "n = k, eksik" notu düşülür.
- Var olan sonuç yalnız `--yeniden "<neden>"` ile ezilir. `--cikti` seçeneği yalnız kod sınaması içindir.

## SHA256 (yazım anı)
| dosya | SHA256 |
|---|---|
| `varyantlar/dinov2_ft/ince_ayar_vit.py` | `60353f342a7a…` (eski `4bccfa72f423…`) |
| `varyantlar/dinov2_ft/README.md` | `5d8391e9f733…` |
| `colab/hat/erteleme_kapisi.py` | `b80366e8882a…` (eski `8a228149d91c…`) |
| `colab/hat/drive_paketle.py` | `3b31765e5f1a…` (eski `df1911dd8f73…`) |
| `colab/hat/e1_kuyruk_yaz.py` | `70b1cee884ec…` |
| `colab/hat/defter_uret_e1.py` | `be654d746041…` |
| `colab/e1-h14.ipynb` | `a6b7218f51ae…` |
| `colab/E1-OKU-BENI.md` | `b8cfd085d23d…` |
| `scripts/e1_h14_analiz.py` | `c38cba390fc0…` |
| `kuyruk/e1-h14.beklenti.json` | `148644bac196…` (= kart taslağı, ayrıştırılmış JSON eşit) |
| `kuyruk/e1-h14.jsonl` | `f1e89900e166…` (15 satır) |
| `kuyruk/e1-h10r.beklenti.json` | `424df8065e17…` |
| `kuyruk/e1-h10r.jsonl` | `5a9852a50e56…` (13 satır) |

Beklenti dosyaları 23:01:09'da, satırlar 23:01:11–13'te yazıldı. Kartın mtime'ı 22:39:10'dur; henüz hiçbir e1 koşumu yok.
Drive paketi henüz YAZILMADI. Şef `colab/hat/drive_paketle.py`'yi koşunca paket sürümü değişir.
