# H15 / E2 uygulama notu (deney-yürütücü, 2026-11-07; hiçbir e2_* koşumundan ÖNCE)

*Kapsam: yalnız uygulama. Karar, eşik, aralık ya da cümle DEĞİŞMEDİ.*
- Kartın (`H15.md`) SHA256'sı `eeef871456c78ee3686a99958ce6a89d967be6e1ed03ed73489a6cff8ab51b7b`, mtime 2026-11-07 19:46:24. Bu
  değer KULLANICI ONAYI H15'teki değerle aynı; kart okunup yazılmadı. Analiz ve KİLİT betikleri bu değeri sabit olarak denetler.
- `SIRALAMA*.md` ve eski kartlar okunmadı, yazılmadı. Rakip koduna dokunulmadı.
- Koşan `e2-veri` kuyruğuna, `kuyruk/e2-veri.jsonl`'a ve `scripts/indir_bcn20000.py`'ye dokunulmadı. Bu betik yalnız içe aktarıldı
  (`geometri`, `adaylar`).
- Hiçbir modelin hiçbir BCN görüntüsü üzerindeki tahmini üretilmedi. Yerel sınamalar BCN yerine DermaMNIST val görüntüleriyle
  (vekil) yapıldı. 40 BCN örnek görüntüsü yalnız piksel işlemesiyle (çerçeve kırpıcı) sınandı, modele verilmedi.
- İç test kümeleri açılmadı. Kuru analiz yalnız K5/E1'de zaten açılmış arşiv tahminlerini okudu.
- Drive'a yazılmadı. Paketleme ve veri kopyası şefte.
- Aşağıda kartta yazılı olmayan ya da kartın ifadesinden ayrılan her uygulama kararı numaralı olarak listelenir.

## §1 Dosya yerleşimi
- Kartın adlandırması izlendi: `varyantlar/e2/e2_cikarim.py`. Orkestratörün örnek adı `varyantlar/dis_deger/` kullanılmadı.
- Yeni dosyalar: `varyantlar/e2/` (`kollar.py`, `e2_cikarim.py`, `bcn_veri.py`, `cerceve.py`, `ucuncu/`, `README.md`) ·
  `scripts/e2_bcn_hazirla.py`, `scripts/e2_kilit.py`, `scripts/e2_bcn_analiz.py` · `colab/hat/e2_kuyruk_yaz.py`,
  `defter_uret_e2.py`, `e2_veri_kopya.py` · `colab/e2-h15.ipynb`, `colab/E2-OKU-BENI.md`.
- Değişen tek mevcut dosya `colab/hat/drive_paketle.py` (§13).
- `varyantlar/e2/kollar.py` 11 kol × 5 seed için kaynak run_id listesinin TEK kaynağıdır. 55 run_id'nin hepsi `runs_colab/`
  arşivinde ve Drive `yz/3-1/runs/` altında `best_model.pth` / `sonda_agirlik.npz` ile birlikte denetlendi (2026-11-07).

## §2 Sapma: `k5_cikarim.py` yükleyicileri "içe aktarılır" yerine satır satır aktarıldı
- Kart, `TURLER` yükleyicilerinin içe aktarılmasını istiyor. Bu yükleyiciler `k5_cikarim.main()` içinde kapanış olarak
  tanımlı, bu yüzden içe aktarılamıyor. `k5_cikarim.py` değiştirilmedi.
- Yalnız sabitler ve SHA256 işlevi içe aktarıldı (`IMAGENET_ORT/STD`, `sha256`). r18, vit_kismi, vit_tam ve sonda yolları
  satır satır aktarıldı. Toplu iş boyları, fp16 token yolu, `set_fused_attn` ve sonda scaler aynı.
- **Bit-eşitlik kanıtı** (yerel CPU, 2026-11-07; aynı kaynak ve aynı argümanlarla `k5_cikarim.py` ↔ `e2_cikarim.py --kip onucus`,
  `tahmin_val_yeniden.npy` bayt bayt karşılaştırıldı):
  - r18: yerel H02 s0, val ilk 32 görüntü → **aynı**.
  - sonda: yerel H01 s0, val 1.003 görüntü → **aynı**.
  - vit_kismi ve vit_tam: rastgele başlı sahte ViT-S kaynakları, val ilk 16 görüntü → **aynı**.
  - C1 yaması (r18, C-val 573) → **aynı**.
- Colab'da her kol-seed ayrıca arşivdeki `tahmin_val.npy` ile K-c denetiminden geçer.

## §3 Sapma: çerçeve kırpma `indir_bcn20000.py`'ye değil, yeni dosyalara eklendi
- Kart, `scripts/indir_bcn20000.py`'ye `cerceve_kirp` geometrisinin eklenmesini söylüyor. Bu betik koşan `e2-veri` kuyruğunda
  ve orkestratör dokunmayı yasakladı. Bu yüzden S9 yolu `varyantlar/e2/cerceve.py` + `scripts/e2_bcn_hazirla.py --asama cerceve`
  ile kuruldu.
- **Yazarın kodu:** `image_cropper.py` @7ea35f5 bayt bayt kopyalandı (`varyantlar/e2/ucuncu/`, SHA256 `103d0a3d…`, CC BY 4.0).
  `obtain_bb` olduğu gibi çağrılır. Eşikler yazarınki: gri ≥ 60, ≥ 100 px kenarlı son dış kontur, oran > 1,1 → kırp. Kontur yoksa
  yazarın `except` dalı uygulanır, yani kırpma yapılmaz.
- **`intensity_ratio` yerine vektörel eşdeğeri kullanıldı.** Formül ve paydalar aynı: Σ / (yükseklik + genişlik).
  - Gerekçe 1: özgün işlev 1024² görüntüde iki Python döngüsüyle çalışır, 12.413 görüntüde saatler sürer.
  - Gerekçe 2: numpy ≥ 2'de (NEP 50) `0 + np.uint8` uint8 kalır ve toplam taşar. Özgün işlev numpy 2.2.6'da yanlış oran verir.
    25/25 rastgele dizide uint8 girdiyle farklı sonuç verdiği ölçüldü.
  - Eşdeğerlik: özgün işlev int64 girdiyle (numpy 1.x anlamı) çağrıldığında 25/25 dizide sonuç birebir aynı.
- **Pikseller ana yolla aynı çözücüden gelir** (PIL `.convert("RGB")`). Kırpma kutusu yazarın cv2 çözümlemesinden alınır.
- Yazarın `cv2.imwrite` (JPEG yeniden kodlama) adımı uygulanmadı, çünkü S9 yalnız çerçevenin etkisini ayırmalı.
- Kırpılan dikdörtgene ana 224 px yolu uygulanır: `geometri(.., "merkez_kare")` + `PIL_BILINEAR`.
- **EXIF yön koruması:** cv2 yön etiketini uygular, PIL uygulamaz. Boyut ya da yön farklıysa kutu PIL pikselinden hesaplanır ve
  `yon_uyusmaz` olarak sayılır (a0 raporunda; 0 beklenir).
- Yalnız kırpılan satırlar saklanır (`bcn_e2_cerceve_fark_224.npz`). Kırpılmayan satırlar ana merkez_kare ile aynıdır. a0 bunu
  her 50. görüntüde bayt bayt sınar.
- **Sınama (40 BCN örneği, yalnız piksel):** 6/40 görüntü kırpıldı. Ana yolun yeniden üretimi 40/40 görüntüde e2-veri örnek
  npz'siyle bayt bayt aynı. Hız ≈ 0,056 sn/görüntü/çekirdek; 4 süreçle tam veri ≈ 3–4 dk.
- **cv2:** `opencv-python-headless==4.12.0.88`, `--no-deps` ile `env/.venv`'e kuruldu (yalnız a0 için; numpy değişmedi).
  `env/requirements.lock` güncellenmedi (açık soru 3).

## §4 Uygulama: etiketsiz model girdisi ve Drive'a taşıma
- e2-veri'nin 224 px npz'si `diagnosis` ve `lesion_id` anahtarlarını da taşıyor. Çıkarım tarafına etiket hiç gitmesin diye a0
  yalnız `images` + `bcn_filename` içeren türev dosyalar yazar:
  - `bcn_e2_merkez_kare_224.npz` (≈ 1,2 GB)
  - `bcn_e2_cerceve_fark_224.npz`
  - `bcn_e2_kal_224.npz` (8 HAM kalibrasyon görüntüsü, BCN yoluyla)
  `bcn_veri.py` etiket anahtarı taşıyan bir girdiyi reddeder. Böylece `bcn_etiket_okundu = 0` yapı gereği sağlanır.
- **İçerik SHA256'sı** = sha256(dtype | şekil | C-sıralı baytlar). Zip zaman damgasından bağımsızdır. Dosya SHA256'sı ile birlikte
  `kuyruk/e2_bcn_beklenen.json`'da donar.
- **Drive'a taşıma:** `colab/hat/e2_veri_kopya.py` dosyaları `yz/3-1/veri_dis/bcn20000/e2_girdi/` altına kopyalar ve geri okuyarak
  SHA256'yı doğrular. `drive_paketle.py`'ye büyük dosya eklenmedi.
  - **Önemli:** `yz/3-1/data/` drive_paketle'nin YÖNETTİĞİ bir dizin. Pakette olmayan dosyalar orada her paketlemede silinir.
    npz'ler bu yüzden `data/` altına değil, `veri_dis/` altına konur.
- **Colab'da açma:** defter hücre 8 dosyaları `/content/veri` altına kopyalar ve SHA256'yı denetler. Sonra `bcn_veri.py ac` npz'yi
  düz `.npy` olarak açar ve içerik SHA256'sını doğrular. Her satır bellek eşlemeli okur ve içerik SHA256'sını yeniden doğrular
  (`bcn_npz_sha_eslesti`).
- **Seçilmeyen seçenek:** BCN zip'inin Colab'da yeniden ön-işlenmesi. Pillow/libjpeg sürüm farkı pikselleri değiştirebilir;
  içerik SHA256'sı tutmazsa K-b DUR verir. Bu yol yalnız yedektir ve ancak içerik SHA256'sı a0 ile birebir aynıysa kullanılabilir.

## §5 a0 (yerel CPU, `scripts/e2_bcn_hazirla.py --asama hepsi`)
- **K-a:** `bcn20000_nn_tam.json` → RMSE ≤ 5 çifti > 0 ise DUR (rc 2).
- **K-b:**
  - CSV MD5'i figshare değeriyle aynı olmalı.
  - Kopyasız liste zip CRC32 + boyutla yeniden türetilir: 47 kopya fazlası ∪ çelişkili grupların 4 üyesi = 49 çıkarma.
  - E1/E2 × ham/kopyasız sayıları (sınıf başına görüntü ve lezyon) `bcn20000_analiz.json` ile ve kartın 11.933 / 3.451 değeriyle
    aynı olmalı.
  - Lezyon birleştirmeleri programla türetilir ve kartın iki çiftine eşit olmalı.
  - Eşleme birim sınaması `ortak.veri.SINIF_KISA` sırasına karşı yapılır.
- Kopyasız liste ve lezyon kümeleri dosyaları tanı içermez, yalnız dosya adı ve lezyon kimliği içerir.
- **Kartta yazılı "manifest güncellenir" adımı:** betik `data/bcn20000/e2_a0_rapor.{md,json}` yazar. Bu rapor tam veri çerçeve
  oranını, `nn_tam` özetini ve boyut sayımını içerir. `data/dis_veri_manifest.md`'nin güncellenmesi şefe bırakıldı.
- **Sınama (scratchpad):** 40 örnek görüntü, sahte `nn_tam` (eşik 0) ve örnek npz ile dört adım uçtan uca geçti. Kapı gerçek CSV
  ve zip listesiyle 11.933 / 3.451 / 49 / 2 çifti yeniden üretti. Kalibrasyon girdisi RMSE ort. 1,759 (kalibrasyonda 1,76).

## §6 a1 duman: tahmin YAZILMAZ, KİLİT gerekmez
- Kart a1, KİLİT öncesinde 48 BCN görüntüsüne (etiketsiz) izin veriyor. Bu tahminler yalnız bellekte denetlenir (şekil,
  sonluluk, toplam, tek sınıf payı) ve diske yazılmaz.
- K-f'deki "aynı kol-seed için ikinci BCN tahmini" sayımı yalnız a3'ün tam dosyalarına bakar.
- Kalibrasyon denetiminde 8 HAM görüntüsünün BCN yolu tahmini, aynı görüntülerin `dermamnist_224.npz` TRAIN tahminiyle
  karşılaştırılır (`kal_argmax_uyum`, beklenen ≥ 7).

## §7 a1'e eklenen CPU satırları
- `e2_veri_kapisi` (`bcn_veri.py kapi`): a0'ın donmuş K-a/K-b sonucu ve Colab kopyasının dosya + içerik SHA256'sı. Metrik adları
  `kapi_*` olduğu için duman kurallarına çarpmaz.
- `c_bolme_veri_c1`: C1 kollarının C-val'i için, E1 ile aynı satır.
- Bu satırlar `val_acc` yazmaz, dolayısıyla kapının "başarılı koşum" sayımına girmez. rc ≠ 0 mekanik İHLAL'dir.

## §8 KİLİT: `scripts/e2_kilit.py`, a3'ün ilk satırı
- KİLİT yazılmadan önce şunlar sınanır: kart SHA256'sı = onaylı; K-d kuru koşu; girdi SHA256'ları; 55 a2 ön-uçuşu
  (birincil kolda eksik → DUR).
- İçerik kartın listesiyle aynı. Ayrıca `k5_test.py` SHA256'sı ve a2 özeti yazılır.
- Oturum devamında aynı içerikle yeniden açılmaz. İçerik farklıysa DUR.
- **Colab'da K-d:**
  - Drive `sonuclar/`'da `k5_test_sonuc.json` var ama `k5_c1_sonuc.json` yok. `e2_veri_kopya.py` onu da kopyalar, yalnız yoksa.
  - Yoksa K-d kart sayılarıyla yapılır; kartın ölçütü de budur.
  - Yerelde iki karşılaştırma da yapılır (donmuş JSON: fark 0; kart: ≤ 0,0047, kart değerleri iki haneye yuvarlak).
- Kart Colab paketinde yok, çünkü `hipotezler/` paket dizini değil. Onaylı kart bayt bayt `kuyruk/e2_H15_kart.md` olarak paketlenir
  (§13). Analiz ve KİLİT, `hipotezler/H15.md` yoksa bu kopyayı okur.

## §9 K-f "ikinci BCN tahmini"nin mekanik tanımı
- a3 satırı tahminleri en sonda atomik olarak yazar (geçici dosya + yeniden adlandırma).
- Aynı varyant + seed için `runs/` altında `tahmin_bcn_*.npy` taşıyan başka bir koşum varsa satır çıkarım yapmadan DUR verir
  (rc 8).
- `--ikinci-acilis "<neden>"` kuyruk satırlarında YOK. Yalnız kullanıcı kararıyla yeni bir satırda kullanılır; bu durumda iki
  sonuç da yazılır.
- Analiz de iki kez çalışmaz. KİLİT'teki `analiz_zamani` ya da var olan sonuç dosyası → DUR (yalnız `--ikinci-acilis`).

## §10 Durum A/B/C kod içinde (K-c)
- A: argmax N/N ∧ maks |Δp| ≤ 1e-5.
- B: argmax ≥ 0,995 ∧ maks |Δp| ≤ 1e-3 ∧ torch/CUDA/cuDNN/timm'den biri kaynak `config.json → donanim` değerinden farklı.
- C: diğer her durum.
- `ozdeslik_gecti = A ∨ B`; durum kodu metriklere yazılır. Durum C ya da yanlış val kümesi → rc 4. a3'te bu durumda BCN açılmaz.
- Birincil kolda C → analiz E2'yi tümüyle DURDURUR.
- Not: h14_c1'in kaynak ortamı E1'dir (cuDNN 91900); Ekim kolları 92700. Aynı Colab ortamında bazı kollar A, bazıları B olabilir.

## §11 Analiz (`scripts/e2_bcn_analiz.py`)
- **Nerede:** kuyrukta değil, Colab bitince şefte, yerelde, bir kez. Kart "son adımda bir kez" diyor; Colab'da etiket yok.
- **Sıra:** k5_test SHA256 → KİLİT SHA256'ları → K-d → a3 denetimleri (55 koşum, Durum, KİLİT, ikinci çıkarım, sıra SHA256,
  K-g bütçe, K-e) → ancak bundan sonra etiketler (CSV MD5). KİLİT'e `analiz_zamani` eklenir.
- **Bootstrap:**
  - `k5_test.Bootstrap` B = 2.000'e sabitli. 10.000'lik iç çekiliş bu yüzden aynı şemayla yeniden yazıldı:
    `integers(0, N, (B, N))` → çokluk.
  - Tek `default_rng(0)` sırası: seed (B × 5) → iç resmî test (B × 2.005) → dış tabakalı lezyon kümeleri. Dış çekilişler
    yapılandırma sırasıyla alınır: ana, S7, S8, S9; S10 ana çekilişi kullanır.
  - **İç tarafta ANA YOL kullanıldı** (C0/C1 aynı çekilişin C-test çokluklarıyla); yedek yol gerekmedi.
  - Boş sınıflı yinelemede dengeli doğruluk tanımsızdır ve düşer; `n_gecerli` yazılır.
  - Seed sabit duyarlılığı aynı dış çekilişle yapılır.
  - K-d (B = 2.000) doğrudan `k5_test.kiyas` / `alt_kiyas` ile yapılır.
- **S2 hedefleri:**
  - `e_C0^π` C0'ın kendi önseliyle (π_C) hedeflenir; π_R hedefi de bilgi olarak yazılır.
  - D'_π = |e_R^π| − |e_C0^π(π_C)|.
- **S6:**
  - Resmî kollar için iki τ_b yazılır: iç = resmî test, iç = C0.
  - Saçılım ve İNCELE (i) lezyon-ayrık iç değeri kullanır: resmî kollar için C0, C1 kolları için C-test.
  - Saçılım bağımlılıksız SVG'dir, çünkü matplotlib kurulu değil.
- **S12:** kopyasız SCC görüntüleri. **T4:** dış tarafta kol olarak (½;½ log-olasılık).
- **Kuru kip (`--kuru`):**
  - Vekil dış küme = C-test (1.227). Resmî kollar için arşiv test tahminlerinin C-test satırları (C0), C1 kolları için arşiv C-test
    tahminleri kullanılır.
  - Akiec lezyonlarının üçte biri "SCC" olarak etiketlenir. 3 sahte kopya ve 1 sahte birleştirme eklenir.
  - Kontrol yapılandırması (E2 eşleme + ham küme) iç değerleri BİREBİR vermelidir.
  - Cümle doldurma yolu ayrıca sınanır (6b; vekil sayılarla, makaleye girmez).
  - BCN dosyası açılmaz.

## §12 Kuyruk `e2-degerlendirme` ve beklenti
- **Beklenti:** kart taslağından kesinleştirildi (`colab/hat/e2_kuyruk_yaz.py`). Taslak SHA256 `2bd91eb1…56c9` (19:43:53).
  - Aralıklar ve kurallar DEĞİŞMEDİ.
  - `_durum` satırı kaldırıldı; `_aciklama`'ya kesinleştirme kaydı ve 4 not eklendi.
  - a1'e daha sıkı bir kural eklendi: `kal_girdi_sha_eslesti` [1, 1]. Gevşetme yok.
  - Kesin dosya 20:49:12'de, satırlar 20:49:27'de yazıldı. Kart 19:46:24'te; henüz hiçbir e2 koşumu yok.
- **Satırlar:**
  - a1: 3 satır (veri kapısı, C1 verisi, `e2_h10_duman`).
  - a2: 55 satır.
  - a3: 56 satır (`e2_kilit_ac` + 55).
  - Sıra: birincil kollar önce (H10, H02c, h10_c1, h02_c1), sonra ikincil, sonra betimsel. Sıra sonucu etkilemez.
- **Bütçe kapısı:** a2/a3 GPU satırları `erteleme_kapisi.py --esik 3.0 --kuyruk-onek e2-degerlendirme` ile sarılır (K-g).
  Kurulum süresi sayılmaz; yalnız koşum süreleri sayılır (E1 ile aynı tanım).
- **Kural ve zaman sınırları:**
  - Çıkarım satırlarında `--kural {"nan_kes", "ilerleme_zorunlu_sn": 600, "sabit_sn": 600}` yalnız takılmayı yakalar.
  - BCN çıkarımı 1.024'lük parçalarla ilerleme yazar. Parçalar 128/64/32'nin katı olduğu için sonuç tüm-dizi yoluyla bit bit
    aynıdır; vit_kismi için yerelde sınandı.
  - Zaman sınırları: duman 25 dk, ön-uçuş 15 dk, BCN 40 dk, KİLİT 30 dk, C1 verisi 70 dk.

## §13 `colab/hat/drive_paketle.py` (varsayılan davranış korunarak)
- Eski SHA256 `3b31765e5f1a…`, yeni `c079bfc0e7e1…`.
- Eklenenler: `kuyruk/e2-degerlendirme.jsonl` varsa E2 kuyruğu + beklenti + `kuyruk/e2_bcn_beklenen.json` + kart kopyası
  (`kuyruk/e2_H15_kart.md`) + `yz/e2-h15.ipynb`.
- a0 yapılmadıysa HATA verir, `--e2-haric` ile atlanır.
- Beklenen dosyası ve kart kopyası e2-degerlendirme başladıktan sonra ezilmez.
- **Kanıt:** `--kuru --e2-haric` listesi eski dosyanın `--kuru` listesiyle aynı (diff boş).

## §14 Defter (`colab/e2-h15.ipynb`, üretici `colab/hat/defter_uret_e2.py`)
- Hücreler hat ve E1 ile aynı. Hücre 4 E1'in torch sabitleme hücresidir. Ekim farkına `numpy`, E1 ilk oturumuna göre farka
  `PIL` eklendi. Sürüklenme denetimi `env/ortam-colab-e2-ilk.json`'a göre yapılır.
- Hücre 8 E2 hazırlığıdır (§4).
- BCN doğruluğu hiçbir hücrede hesaplanmaz ve gösterilmez.

## Açık sorular (şef / kullanıcı)
1. **Drive yükleme süresi:** ≈ 2,5 GB (merkez_kare ≈ 1,2 GB + çerçeve farkı + kalibrasyon) Drive for desktop ile yüklenecek.
   Hız ölçülmedi.
2. **H15 S9 duyarlılığı yazar kırpıcısının oranına bağlı:** 40 örnekte 6/40 (%15). Makale ~%20 diyor. Tam oran a0 raporunda olacak.
3. **`env/requirements.lock`:** opencv eklemesi kilit dosyasına yazılsın mı? Şu an yalnız bu notta kayıtlı.
4. **Analizin Drive KİLİT'i:** analiz `analiz_zamani`'yi `--kilit` ile verilen dosyaya yazar. Önerilen yol yerel kopyadır
   (`sonuclar/e2_bcn.KILIT`); Drive kopyası değişmez.

## SHA256 (yazım anı, ilk 12 hane)
| dosya | SHA256 |
|---|---|
| `hipotezler/H15.md` (değişmedi) | `eeef871456c7…` |
| `varyantlar/e2/kollar.py` | `0ae040181760…` |
| `varyantlar/e2/e2_cikarim.py` | `e4d00edab6a0…` |
| `varyantlar/e2/bcn_veri.py` | `df0c8830f1fe…` |
| `varyantlar/e2/cerceve.py` | `6fa5b1f086de…` |
| `varyantlar/e2/ucuncu/image_cropper.py` | `103d0a3df048…` |
| `varyantlar/e2/README.md` | `2d412dd8a5aa…` |
| `scripts/e2_bcn_hazirla.py` | `16ae1fc7d39e…` |
| `scripts/e2_kilit.py` | `46d12657efa0…` |
| `scripts/e2_bcn_analiz.py` | `cdd056f46e27…` |
| `colab/hat/e2_kuyruk_yaz.py` | `9125c4e65e0e…` |
| `colab/hat/defter_uret_e2.py` | `7c291f8ad25f…` |
| `colab/hat/e2_veri_kopya.py` | `647e6484c617…` |
| `colab/hat/drive_paketle.py` | `c079bfc0e7e1…` (eski `3b31765e5f1a…`) |
| `colab/e2-h15.ipynb` | `fd22be691483…` |
| `colab/E2-OKU-BENI.md` | `8b6f6674c475…` |
| `kuyruk/e2-degerlendirme.beklenti.json` | `f6ef0c28c130…` (taslak `2bd91eb178d2…`) |
| `kuyruk/e2-degerlendirme.jsonl` | `182490986b58…` (114 satır) |
| `sonuclar/e2_bcn_kuru_sonuc.json` | `9d506c4a82c6…` |

`e2_cikarim.py`, `e2_bcn_analiz.py`, beklenti dosyası ve beklenen dosyası a3'te KİLİT'e yazılır. Bunlardan biri KİLİT'ten sonra
değişirse analiz DUR verir.
