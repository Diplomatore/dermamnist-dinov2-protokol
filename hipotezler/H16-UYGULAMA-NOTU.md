# H16 / E3 uygulama notu (deney-yürütücü, 2026-11-08; hiçbir `e3_*` koşumundan ÖNCE)

*Kapsam: yalnız uygulama. Karar, eşik, aralık ya da cümle DEĞİŞMEDİ.*
- Kart `hipotezler/H16.md`: SHA256 `f3c6998196e0624801979b01da5226747c8f876502bd8e9c785bd5c6d2157109`, mtime
  2026-11-08 15:20:08. Kart okundu, yazılmadı. Analiz betiği bu SHA256'yı sabit olarak denetler.
- Şef kararları (açık iki madde kapandı): (1) duyarlılık kolları S5–S7 DAHİL; (2) q_BCN BCN zip'indeki sıralı ilk 500 dosyanın
  JPEG başlığından kestirilir.
- `runs/` altında `e3` önekli koşum sayısı 0. `data/ham_val_isic/` henüz yok; resmî val'in hiçbir ISIC dosyası indirilmedi.
- Resmî test, C-test, BCN etiket CSV'si açılmadı. `k5_test.KILIT` ve `e2_bcn.KILIT`'e dokunulmadı. Rakip koduna, `SIRALAMA*.md`'ye,
  `scripts/indir_bcn20000.py`'ye, `scripts/e2_bcn_analiz.py`'ye ve `sonuclar/e2_incele_ek/v3_model.py`'ye dokunulmadı (yalnız içe
  aktarıldı ya da satırları taşındı). Drive'a yazılmadı.

## §1 Dosyalar ve SHA256

| dosya | iş | SHA256 |
|---|---|---|
| `scripts/e3_ham_val_hazirla.py` | a0 veri kapısı + etiketsiz hat girdisi | `a602b627716a8ae1a36a13c3e405dd0e7e1c2e94568e9646166dac5e374cbc54` |
| `varyantlar/e3/e3_cikarim.py` | çıkarım (duman / onucus / tam) | `ba6ed609049b868af1b5de78c919d95cc64e4f517ef5ea215c42003fb629363d` |
| `scripts/e3_onisleme_analiz.py` | `--kuru` / `--kilit-ac` / analiz (bir kez) | `fdd35b9f43b15ef05c919a5f43f4853583f8384618c7c7b2def6bf82aaf09c45` |
| `varyantlar/e3/e3_kuyruk_yaz.py` | `e3-onisleme` satırlarını a0 sonrası yazar | `e1ebba08f1068a56ad63e0ca6c3c833e4eb422f15349404bd17f36ae06388679` |
| `varyantlar/e3/README.md` | ebeveynden fark + nasıl koşulur | `b65ff4a92b7316d71d2b09839a4650d262c78c52f582ad7eef1f641449cd7746` |
| `kuyruk/e3-onisleme.beklenti.json` | ön-kayıt beklentisi (kart taslağı, kesinleşmiş; 15:40:31) | `44f6b052fe2fc241b7517a34bae70d84c40c69ef80c15f6080b8c5045609179b` |
| `kuyruk/e3-veri.jsonl` | a0 kuyruğu (1 satır, CPU; 15:44:20) | `dab0f7b61718db49589306d17c9b0d84efbdfd1c32ff7de59d8d686a7ac21b8f` |
| `sonuclar/e3_onisleme_kuru_sonuc.json` (+ `.md`, `.sha256`) | kuru koşu (§4) | `843f758d9047bef8a08d988522dbd886f3d398e3061eb1c403097d813db290f0` |

İçe aktarılan / taşınan kaynaklar (değişmedi): `scripts/indir_bcn20000.py` `41575214…` (geometri, adaylar, http_get, ISIC_API,
HttpRangeDosya, figshare_kaydi, fig_dosya, ZIPDIR) · `scripts/e2_bcn_analiz.py` `cdd056f4…` (multinom, ozet, dis_kol, tr, ga_tr,
p_tr, jsonla, sha) · `sonuclar/e2_incele_ek/v3_model.py` `3ba5d836…` (satır 19–21 `hazirla`, 45–80 `yerel` + `kol_yukle` r18 /
vit_kismi / vit_tam dalları) · `varyantlar/e2/kollar.py` `0ae04018…` · `data/bcn20000/kalibrasyon_onisleme.json` `670e0534…`
(224 en_iyi = `merkez_kare|PIL_BILINEAR`).

`kuyruk/e3_onisleme_beklenen.json` ve `kuyruk/e3-onisleme.jsonl` HENÜZ YOK. İlki a0 (e3-veri) tarafından donar; ikincisi
ancak ondan sonra `e3_kuyruk_yaz.py` ile yazılır (§2-1).

## §2 Kartın ifadesinden ayrılan ya da kartta yazmayan uygulama kararları
1. **Sıra: a0 ayrı kuyrukta, a1–a3 satırları a0'dan sonra.** Ardışık ölçüm (ISIC API, bir eğitim görüntüsü): meta 0,7 s + dosya
   0,8 s → 1.003 görüntü ardışık ≈ 25 dk, 6 iş parçacığıyla ≈ 4–6 dk. 5 dakikayı aşabildiği için foreground'da koşulmadı:
   `kuyruk/e3-veri`. "Beklenen ve beklenti dosyası satırlardan önce" kuralı yüzünden `e3-onisleme` satırları şimdi yazılmadı.
   Satırları `varyantlar/e3/e3_kuyruk_yaz.py` yazar. Bu betik beklenen dosyası yoksa, K-b geçmemişse, 4 npz'nin SHA256'sı
   tutmuyorsa, beklenti dosyası beklenen dosyasından sonra değişmişse ya da jsonl zaten varsa yazmaz (rc 2). `--kuru` ile
   26 komutun dökümü alındı (a1 2 + a2 4 + a3 20; hepsi `--gpu-zorunlu`, kural `{"nan_kes":true,"ilerleme_zorunlu_sn":600,
   "sabit_sn":600}`, zaman sınırı H10 30 / H11 20 / R18 15 dk; a3 sırası h10, h02c, h08c, h11).
2. **`test_okundu` → `resmi_test_okundu`.** `ortak/kayit.metrics_yaz` `test_` önekli anahtarı M4 gereği reddediyor (yerel sınamada
   RuntimeError). E2'deki ad kullanıldı. Beklenti dosyasında üç kuralda (a1/a2/a3) yalnız anahtar adı değişti; aralık aynı.
   Bu, beklenti taslağından tek farktır (dosyanın `_notlar` alanında yazılı).
3. **S5 JPEG kolu sırası:** HAM JPEG çözülür → `geometri(·, "merkez_kare")` (450²) → Pillow JPEG (`quality = q_BCN`,
   `subsampling` = BCN çoğunluğu) → çözülür → birincil hat (`geometri` kare girdide özdeşliktir, sonra `adaylar()[yon](·, 224)`).
   Kartın "sıkıştırma 450² ölçeğinde" ifadesi bu sırayla sağlandı. Tam 600×450'yi sıkıştırıp sonra kırpmak 8 piksellik blok
   ızgarasını kaydırırdı (75 px ofset).
4. **S6/S7 geometri:** kenar = round(450 / 1,25) = 360 ve int(450 / 0,8) = 562 (kartta yazılı). Merkezden kırpılır: z125 ofseti
   (120, 45); z080'de yatayda 19 px kırpılır, dikeyde 56 + 56 px yansıtmalı dolgu yapılır (`numpy.pad`, `mode="reflect"`, kenar
   pikseli tekrarlanmaz). Ardından `adaylar()[yon](·, 224)` uygulanır. Boyutu 600×450 olmayan görüntüde aynı oranlar kısa kenardan
   hesaplanır ve boyut dağılımı rapora yazılır.
5. **q_BCN kestirimi:** Pillow'un `quality = 1..100` için yazdığı nicemleme tabloları referans alındı (dosyadan okunan tablolarla
   aynı sıra). Her BCN dosyası için luma + kroma tablosunda L1 uzaklığı en küçük q seçildi; sonuç medyan (yarım → yukarı
   yuvarlanır). Alt-örnekleme `JpegImagePlugin.get_sampling` çoğunluğundan alındı. Yerel zip mevcut (`data/bcn20000/zip/
   BCN_20k_train.zip`, 6.368.901.290 B). Üye başına yalnız ilk 64 KB okunur (başlık kesikse 256 KB, en son tam üye) ve piksel
   çözülmez. Yedek yol figshare HTTP Range'dir (`IB.HttpRangeDosya`; dosya başına tek Range isteği); iki yol da sınandı.
   `bcn_20k_train.csv` açılmaz.
6. **Eksik görüntü varsa:** hat npz'leri n_hat (≥ 993) satırlıdır. `tahmin_val_<hat>.npy` n_hat × 7 boyutunda yazılır, yanına
   `hat_index.npy` konur. `tahmin_val_yerel.npy` her zaman 1.003 × 7'dir ve analiz iki kolu da `hat_index` üzerinden eşler.
   Eksik yoksa kartın "1.003 × 7, index sırasıyla" tanımıyla birebir aynıdır.
7. **ISIC tanı ↔ val_labels taksonomi tablosu** (`e3_ham_val_hazirla.py` `TANI3`, `TANI2`): önce `diagnosis_3`, sonra
   `diagnosis_2`; eşlenemeyen uyuşmaz sayılır. Tablo, yerel ISIC koleksiyon 212 (HAM10000) meta sözlüğündeki 8 (d2, d3)
   değerinin hepsini kapsıyor; eşlenemeyen 0. Bu denetimde görüntü ile etiket arasında bağ kurulmadı.
8. **Taşınan yükleyici:** v3'ün toplu iş boyları korundu (r18 64, ViT 16, vit_kismi fp16 token). Bunlar `e2_cikarim.py`'ninkinden
   (128 / 32–64) farklıdır; kart v3 yolunu istiyor. Ek olarak `ortak.kayit.seed_bagla` çağrılır (cudnn deterministik,
   benchmark kapalı, TF32 kapalı; kartın "Ortam" satırı). `HF_HUB_OFFLINE=1`.
9. **Arşiv referansı (K-c):** `tahmin_val.npy` / `etiket_val.npy` Drive'daki aynı run dizininden geçici kopyayla okunur. Dört
   örnek kol-seed'de Drive ve `runs_colab` kopyaları bayt bayt aynı (SHA256 denetimi). `val_labels` ≠ `etiket_val.npy` → DUR.
10. **K-f bütçe kapısı (kip tam):** model yüklenmeden önce hesaplanır: `e3-onisleme` kuyruğunda harcanan süre (`.done sure_sn`
    toplamı) + kalan a3 kol-seed'lerinin a2 süresiyle tahmini. Tahmin > 2,0 GPU-sa ise ya da a2 süresi yoksa rc 10 döner ve
    model yüklenmez. Kartın "a2'den sonra a3 süresi tahmin edilir; tahmin tavanı aşıyorsa a3 başlamaz" kuralı böyle uygulandı.
11. **Analiz kilidi (E2 deseni):** `--kilit-ac` a3 denetimlerini (20 koşum, rc 0, kaynak run_id, Durum A/B, girdi/sıra SHA, 4
    geçiş, K-e, K-g sayaçları, bütçe) ve gerçek yerel kollarla K-d'yi çalıştırır, sonra `sonuclar/e3_onisleme.KILIT`'i yazar
    (kart, analiz, beklenti, beklenen, e3_cikarim, hazırlık, e2_bcn_analiz, kollar SHA256'ları + koşum listesi). Analiz kilit
    olmadan, SHA256 uyuşmazlığında ya da `analiz_zamani` doluysa çalışmaz. İkinci çalıştırma yalnız `--ikinci-acilis "<neden>"`
    ile olur; iki sonuç da yazılır ve "E3 analizi ikinci kez çalıştı; neden: …" beyanı eklenir. Bu yollar sentetik a3
    koşumlarıyla sınandı (Durum C'li H02c → DUR dahil).
12. **Eşik denetimi:** 2,48 ve 10,31 betikte sabittir. `e2_bcn_sonuc.json`'dan m = 16,3369 − 13,8556 = **2,4813** ve G = 16,3369
    − 6,0315 = **10,3053** okundu; ikisi de ±0,005 içinde. KISMEN oranında kullanılan B1 = 16,34 de ±0,005 ile denetlenir.
13. **S10 kalibrasyon payı:** v3.json `kal` alanından, R18 türündeki 4 kol × 5 seed × 8 görüntü = 160 karşılaştırmadan 3 dönme
    (kartın %1,9'u). Görülmemiş val dönme oranı H02c ve H08c'nin kol-seed ortalamasıdır.
14. **N/S:** val'de 970 lezyon var: N 638 (655 görüntü), S 332. N alt kümesi W'nin N küme sütunlarıdır.
15. **Lisans (K-h):** `data/ham_val_isic/` WS `.gitignore` (`data/`) ile dışarıda. `makale_workspace/.../repo_esitleme/kopyala.py`
    izin listesiyle çalışır ve bu yolu içermez; değiştirilmedi. Görüntü başına `copyright_license` a0 raporuna sayılır.
16. **Kuyruk Python'u:** `kosum.py --gpu-zorunlu` torch'u kendi yorumlayıcısında içe aktarır. `kuyruk.py kos e3-onisleme`
    bu yüzden `env/.venv/Scripts/python.exe` ile başlatılmalı.

## §3 Yerel sınamalar (ölçüm değil; val hat girdisi yok)
- **"Aynı işlev" kanıtı:** `e3_ham_val_hazirla.py --sinama` 8 HAM kalibrasyon görüntüsünü (DermaMNIST TRAIN) çalıştırdı. Birincil
  hat çıktısı E2'nin `bcn_e2_kal_224.npz` görüntüleriyle **bayt bayt aynı** (içerik SHA256 = `e2_bcn_beklenen.json → kal_images`).
  RMSE ort 1,7587 (kalibrasyon 1,76), maks 2,1147. z125 ve z080 RMSE'si 16–19 gri düzey; geometrik ölçek değişiminde beklenen
  büyüklük.
- **q_BCN yolu:** yerel zip ve Range yolu ilk 4–6 dosyada aynı sonucu verdi: q = 100, alt-örnekleme 2 (4:2:0), tablolar tam
  eşleşme, 1024×1024. Kesin değer a0'da 500 dosyadan dondurulur.
- **İndirme:** `_indir_bir` bir TRAIN görüntüsünü (ISIC_0024688) indirdi; dosya kalibrasyon kopyasıyla bayt bayt aynı. 404 dönen
  kimlik 5 denemeden sonra "eksik" sayıldı.
- **`liste`:** val 1.003, index {0..1002}, image_id tekil, test kesişimi 0.
- **Çıkarım (CPU, `--alt-kume 20`, vekil hat = val görüntüleri):** 4 kolda Durum A (maks|Δp| ≤ 7,8e-6). Vekil hat := yerel → uyum
  1,0, |Δp| = 0 (aynı toplu iş). `h10_c1` → rc 6 (K-g). Bozuk beklenen → rc 9.

## §4 Kuru koşu (K-d) — `sonuclar/e3_onisleme_kuru_sonuc.{json,md}` — **GEÇTİ**
Yerel kol vekili, arşiv `tahmin_val.npy`'dir (runs_colab; v3'te 20 kol-seed argmax %100, maks|Δp| ≤ 1,1e-4).

| kol | yerel dengeli (puan) | v3 `orijinal` | fark |
|---|---|---|---|
| H10 | 79,770 | 79,770 | 0,0000 |
| H02c | 70,999 | 70,999 | 0,0000 |
| H08c | 77,627 | 77,627 | 0,0000 |
| H11 | 75,760 | 75,760 | 0,0000 |

- Δ_yerel(H10 − H02c) = **+8,772** (v3 +8,772; tol ± 0,05).
- Öz-sınama, hat := yerel: Δ_hat = **0** ve GA **[0; 0]** (tam eşitlik).
- Eşikler: m 2,4813 ↔ 2,48 ve G 10,3053 ↔ 10,31 (± 0,005) tuttu.
- **Uçtan uca kod sınaması: VEKİL, SONUÇ DEĞİL.** Hat kolları yerine arşivin son-epoch (`tahmin_val_son`) ve AUC-seçim tahminleri
  kondu; 2 görüntü yapay olarak "eksik" bırakıldı. Bütün yollar çalıştı: birincil, S1–S10, N/S, İNCELE ve cümle seçimi. Bu
  sayılar ön-işleme hattı hakkında bilgi taşımaz.

## §5 Şefin başlatacağı komutlar (sıra; WS kökü)
`PY = env/.venv/Scripts/python.exe`, `KQ = <WS>/tools/kuyruk.py`
1. **a0 (CPU + ağ; GPU yok):** `PY -I KQ --kok <WS> kos e3-veri` (çift kanal: arka plan + `izle`). Tahmini 8–15 dk (indirme
   ≈ 4–6 dk; q_BCN ≈ 10 sn; 4 hat ≈ 2–3 dk; npz yazımı + SHA ≈ 1–2 dk). Kartın tahmini 0,3–0,6 sa. rc 2 = K-a/K-b DUR →
   `data/ham_val_isic/e3_a0_ara.json`, rapor ve teşhis. Çıktı: `kuyruk/e3_onisleme_beklenen.json` (donar) +
   `data/ham_val_isic/e3_a0_rapor.md`.
2. **Satırlar:** `PY -I varyantlar/e3/e3_kuyruk_yaz.py` → `kuyruk/e3-onisleme.jsonl` (26 satır). Kapı tutmazsa rc 2, satır yok.
3. **a1 → a2 → a3 (GPU, yerel GTX 1650, paralel 1):** `PY -I KQ --kok <WS> kos e3-onisleme --kademeli`. Tahmini a1 ≈ 5 dk,
   a2 ≈ 10–12 dk, a3 ≈ 45–60 dk; toplam ≈ 1,0–1,3 sa duvar saati (kart 0,72 GPU-sa, tavan 2,0). Her aşama sonunda
   `beklenti_kontrol.py` kapısı çalışır; ihlalde sonraki aşama başlamaz (`beklenti-ihlali.md`). a3 satırları kendi K-f bütçe
   tahminini yapar (rc 10).
4. **Analiz (CPU, ≈ 1 dk):** `PY -I scripts/e3_onisleme_analiz.py --kilit-ac`, ardından `PY -I scripts/e3_onisleme_analiz.py`
   (BİR KEZ) → `sonuclar/e3_onisleme_sonuc.{json,md}` + `.sha256`. Kartın "Sonuç" bölümünü kuyruk sonrası deney-yürütücü
   doldurur.
