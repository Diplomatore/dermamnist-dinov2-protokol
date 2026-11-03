# K5 ÖN-KAYDI — test kilidi, doğrulama kuralları, DermaMNIST-C (C0 / C1 / C2), karar çıkışları

*Yazım: 2026-10-11 (Tur-4 tahlilinden ve H13 kartından sonra). Bu dosya **test kilidi açılmadan önce** yazıldı. Bağlayıcıdır.*
- **Test geçişi başladıktan sonra hiçbir satırı değiştirilmez.** Test öncesindeki düzeltmeler yalnız `K5-ONKAYIT-b.md` ile yapılır ve gerekçe yazılır.
- Kesin damga: bu dosyanın mtime'ı ve `sonuclar/k5_dondurma.json` içindeki SHA256. Her ikisi de test geçişinin ilk `*_test` koşumundan önce olmalıdır.

**Durum: TASLAK. Bağlayıcı olması ve test kilidinin açılması için KULLANICI ONAYI GEREKİR (K5 kritik kapı).** Bu
dosya protokol düzeyinde kurallar içerir: test kolları, C bölmesi ile yeniden eğitim, karar çıkışları. Otomatik koşmaz.

**Dayanaklar:**
- KULLANICI KARARLARI 3, 5, 7, 8, 9 (`orkestrasyon_durumu.md`):
  - C-değerlendirmede birincil = resmî pikseller + C bölmesi, ikincil = C npz.
  - H08 yol A.
  - Colab tekrar kapısı.
  - 3 paralel L4.
  - C1 kolları = baseline, H02, H08, H10 (+ H01); C0 eklenir; M3 (rakibe D4) HAYIR → sınırlılık.
- SIRALAMA "K5 ÖN-KAYDI — H08 TEST DOĞRULAMA KURALI" (2026-10-11 09:42; **aynen geçerli**, burada değiştirilmedi).
- SIRALAMA "COLAB TEKRAR KAPISI": K5 testi **yalnız Colab kollarıyla**, bir kez.
- `tahlil-tur4.md` §8.
- `dongu-pipeline.md` K5.
- `deney-sozlesmesi.md` D1–D5.

---

## §0 Önkoşullar — test açılmadan önce hepsi tamam olmalı (şef kontrol listesi)

| # | Önkoşul | Kanıt |
|---|---|---|
| P1 | **H12** koşuldu ve tahlil edildi. H11'in görülmemiş 3. maddesi kapandı. H12 C\* ve H06-protokolü τ\* belirlendi. | tahlil eki |
| P2 | **H13** (Tur-5) kararı verildi (TUT / ÖLDÜR / İNCELE). TUT ise tam-val τ\* ve T hesaplandı. | tahlil-tur5 |
| P3 | **Tur-5 → K5 geçişi kullanıcı kararıyla onaylandı.** Durma sayacı 0/2 veya 1/2 olduğundan K5'i kendiliğinden tetiklemez; bkz. SIRALAMA-TUR5 §5. | günlük "KULLANICI KARARI" |
| P4 | **Donmuş parametre dosyası** `sonuclar/k5_dondurma.json` yazıldı. İçeriği aşağıda. | dosya + mtime |
| P5 | **Analiz betiği** `scripts/k5_test.py` yazıldı ve **val üzerinde kuru koşuldu** (test yerine val verilir, tüm tablolar üretilir). Kuru koşu val sayıları tahlil-tur4 sayılarını yeniden üretmeli: H10 − H02c +2,91 [+1,34; +4,61]. Betik SHA256'sı P4'e yazıldı. | kuru koşu çıktısı |
| P6 | **Özdeşlik (yükleme doğruluğu):** her kolun `best_model.pth` dosyası (sondada kayıtlı ağırlık ya da aynı seed + C\* ile deterministik yeniden uydurma) val'de yeniden çıkarılır. Kayıtlı `tahmin_val.npy` ile koşullar: argmax 1.003/1.003 aynı, olasılık maks \|Δ\| ≤ 1e-5. H10/H11 için **val'de kullanılan kod yolu aynen** kullanılır: H10'da blok-8 fp16 önbellek yolu, test görüntüleri için de aynı biçimde önbelleğe alınır. Herhangi bir ihlal → DUR, teşhis. | `k5_ozdeslik.json` |
| P7 | **C-test lezyon ayrıklığı:** lezyon(C-test 1.227) ∩ lezyon(resmî train ∪ resmî val) = ∅ olmalı. Beklenen 0/1.227; train için tahlil-tur4 denetledi. Ayrıca C-test ⊂ resmî test olmalı ve indeks eşlemesi `data/splits/dermamnist_c_esleme.csv` ile 1.227/1.227 tutmalı. ≠ ∅ → DUR. | `k5_c_denetim.json` |
| P8 | Colab `runs/` yerel arşive kopyalandı (D5). Günlük bütçe sayacı güncel. | günlük |

**`k5_dondurma.json` içeriği:**
- Kol listesi ve run_id'ler (§1).
- τ\* değerleri (tam-val, H04/H06 protokolü):
  - H02c 0,7; H08c 0,7; H10 0,7; H11 1,0 (ızgara ucu notu); H01c 0,2.
  - H12 ve H13: P1/P2'den.
  - baseline_c: **Colab baseline val'inde aynı yöntemle test öncesinde hesaplanır.**
- R-T: seed başına tam-val T, tüm kollar için.
- H01c/H12 sonda C\* değerleri.
- `katlar.json` SHA256 `0db03edb…e0193`.
- RNG 0, B = 2.000.
- `k5_test.py` SHA256 ve bu dosyanın SHA256'sı.

Bunlardan hiçbiri test'te seçilmez, aranmaz ya da ayarlanmaz.

---

## §1 Test kolları ve sırası (resmî test, n = 2.005, **bir kez**)

Tüm çıkarımlar tek bir Colab L4 geçişinde (`k5-test` kuyruğu, FP32, TF32 kapalı) yapılır. Ardından `k5_test.py` **bir kez** çağrılır. Epoch, C\*, τ\* ve T seçimleri val'de yapılmış ve donmuştur.

| Sıra | Kol | Run'lar (s0…s4) | Rol | Katmanlar (donmuş) | Holm ailesi |
|---|---|---|---|---|---|
| 1 | **baseline_c** `baseline_colab` | `20261011-142338/143613/144846/150122/151359` | rakip; test sayıları kc1'de **dondu** (0,7395 ± 0,0053). **Yeniden çıkarım YOK**: mevcut `tahmin_test.npy` (resmî val-AUC seçimi) kullanılır. Özdeşlik: dosyadan hesaplanan ACC = metrics `test_acc` (±1e-4). `tahmin_test_son.npy` ikincil satırdır. | H04 (τ\* P4), R-T | 1 (referans) |
| 2 | **H01c** `h01_dinov2b_lp_r224_colab` | `20261011-125349/125744/130031/130320/130608` | sonda satırı; test öznitelikleri (DINOv2-B r224, 2.005) bu geçişte çıkarılır | **H06 τ\* 0,2**, R-T T 1,19 | 1 |
| 3 | **H02c** `h02_r18in_ft224_colab` | `20261011-130855/131620/132346/133112/133837` | R18 ince ayar; adil kıyas (rakibe önağırlık hakkı) | H04 τ\* 0,7, R-T | 1; 2 (referans) |
| 4 | **H08c** `h08_r18in_ft224_d4_colab` | `20261011-134604/135335/140107/140837/141608` | K5 doğrulama adayı (H08 kuralı aynen) | H04 τ\* 0,7, R-T | 1; 2 |
| 5 | **H10** `h10_dinov2b_k4_ft224` | `20261011-160211/161411/162611/163811/165010` | **Tur-4 kazananı (val'de seçildi; test sonucu kazananı DEĞİŞTİRMEZ)** | H04 τ\* 0,7, R-T | 1; 2 |
| 6 | **H11** `h11_dinov2s_ft224` | `20261011-160117/161247/162419/163552/164724` | Pareto satırı (TUT, kazanan değil) | H04 τ\* 1,0 (ızgara ucu notu), R-T | — (betimsel) |
| 7 | **H12** `h12_dinov2s_lp_r224` | P1'den | Pareto satırı (analiz kartı) | H06 protokolü τ\*, R-T | — (betimsel) |
| 8 | **H13** (yalnız Tur-5'te **TUT** ise) | 3 ve 5'in test tahminlerinden CPU'da (½ ; ½ log-olasılık, kırpma 1e-12) | Tur-5 kazananı → **K5 kazananı** | H04 τ\*, R-T (P2) | 1; 2 (H13 − H10) |
| (8') | T4 koşullu: H08c_s ⊕ H10_s | 4 ve 5'ten CPU | Her durumda hesaplanır ve **ek tabloda yazılır** (D3). **İddia taşıması yalnız H08 doğrulanırsa** (§2.1). H13 ÖLDÜR olsa da T4 kazanan olamaz. | — | — |
| (8'') | M3 topluluk hakkı (yalnız H13 TUT ise) | baseline_c homojen 5-seed; baseline_c_s ⊕ H02c_s | betimsel | — | — |

- **Test'e gitmeyenler:** H03, H05, H07, H09. H13 ÖLDÜR/İNCELE ise H13 de gitmez (öldürülen kart test'e gitmez; H05/H07/H09 ile tutarlı).
- **Kazanan:** W = H13 (Tur-5'te TUT ise), aksi H10. W val'de belirlenir. Test sonucu W'yi, H08 ↔ H02 R18 satırı kuralı dışında hiçbir satırı değiştirmez.

---

## §2 Doğrulama kuralları ve istatistik (resmî test)

### 2.1 H08 kuralı — AYNEN (SIRALAMA 09:42)
H08c − H02c eşleştirilmiş. **Doğrulandı ⇔ ΔACC ≥ +0,5 puan VE görüntü bootstrap %95 GA alt > 0.**
- H08 doğrulanırsa makalenin R18 kazanan satırı H08, doğrulanmazsa H02.
- H08 görülmemiş eşikleri: ΔACC > 3, ACC > 0,90, AUC > 0,99, dengeli Δ > 6,5 → rapor bayrağı. Dengeli ayrıca df/vasc hariç 5 sınıf.
- **M3 HAYIR** (KARAR 9): rakibe D4 hakkı verilmedi. H08 satırı makaleye girerse "rakibe augmentasyon hakkı verilmedi" **sınırlılık** cümlesi zorunludur. Adil kıyas satırı augmentasyonsuz H02c'dir.

### 2.2 H10 kuralı (H08 kuralının birebir aynısı)
H10 − H02c eşleştirilmiş, seed {0..4}. **Doğrulandı ⇔ ΔACC ≥ +0,5 puan VE görüntü bootstrap %95 GA alt > 0.**
- **Görülmemiş (rapor bayrağı + sızıntı denetimi; karar yazılmadan önce İNCELE kapanır):** H10 test ACC > 0,93, AUC > 0,99 ya da H10 − H02c > +6,0 puan (Wu & Papież 2025 DINOv2-B **uçtan uca** − R18 uçtan uca = +5,97; kısmi ince ayar bunu aşmamalı).
- **Beklenti (val'den; M1: val görüldü):** H10 − H02c test Δ ∈ [+1,0; +4,5].
  - Val'in S oranı (%34,7) test'te de aynı: %34,6, yani 693/2.005. Bu nedenle val'deki S-yoğun kazancın test'te tekrarlanması beklenir (tahlil-tur4 §4.4).

### 2.3 H13 kuralı (yalnız Tur-5'te TUT ise)
H13 − H10 eşleştirilmiş. **Doğrulandı ⇔ ΔACC ≥ +0,5 VE GA alt > 0.**
- Doğrulanmazsa da W = H13 kalır (val seçimi), ama makale cümlesi "topluluk test'te tek modeli ayırt edilebilir biçimde geçmedi" olur.

### 2.4 İstatistik (tüm resmî test kıyasları)
- **Seed düzeyi:** kesin tek yönlü seed-eşleştirilmiş t (scipy), n = 5, Δ_s = ACC_B(s) − ACC_A(s).
- **Görüntü bootstrap'ı:**
  - B = 2.000, `numpy.random.default_rng(0)`, aynı indeksler iki kola.
  - Δ\* = ort_s ACC_B − ort_s ACC_A.
  - Yüzdelik %95 GA.
  - Tek yönlü bootstrap p = (1 + #{Δ\* ≤ 0}) / (B + 1).
- **Tanımlar ve satırlar:** ACC, AUC (makro OvR, softmax olasılıkları) ve medmnist `Evaluator` tanımları kullanılır. Dengeli ACC ve df/vasc hariç 5 sınıflı dengeli ACC, makro-F1 ve ECE-15/NLL **ikincil**dir. Katmanlı satırlar (H04 τ\*, R-T T) ikinci ve üçüncü satırlardır ve **karar vermez**.
- **Rakip satırı:** baseline_c'nin birincil test tahmini resmî seçim kuralıyla (val-AUC) alınır. Son-epoch satırı ikincildir. Bu rakip protokolüdür ve değiştirilmez.

### 2.5 Çoklu karşılaştırma — Holm (α = 0,05, tek yönlü)
- **Aile 1 (rakibe karşı):** {H01c, H02c, H08c, H10, (+ H13 TUT ise)} − baseline_c. Boyut 4 ya da 5.
- **Aile 2 (eşit önağırlık hakkına karşı):** {H08c, H10} − H02c (+ H13 − H10 TUT ise). Boyut 2 ya da 3.
- Holm, **t p'lerine** ve **bootstrap p'lerine** ayrı ayrı uygulanır.
- **Doğrulama kuralları (2.1–2.3) Holm'dan bağımsız, ön-kayıtlı haliyle bağlayıcıdır.** Holm yalnız "anlamlı" kelimesini yönetir:
  - **"Anlamlı" ⇔ ilgili doğrulama kuralı geçti ∧ Holm-düzeltilmiş t p < 0,05 ∧ Holm-düzeltilmiş bootstrap p < 0,05.**
  - Kural geçer ama Holm geçmezse: "doğrulandı (düzeltilmemiş); çoklu karşılaştırma sonrası anlamlılık korunmadı".
- **Pratik marj:**
  - Aile 1: ΔACC ≥ **+2,0 puan**. Gerekçe: replikasyon kabul aralığı [0,7218; 0,754] 3,2 puan genişliğinde; marj bunun yarısından büyük, yani rakibin protokol oynaklığının dışında.
  - Aile 2: +0,5 puan (= doğrulama kuralları).

### 2.6 Resmî test S/N ayrışımı (ön-kayıtlı; **kural**, keşifsel değil)
- **Tanımlar:**
  - S_test = lezyonu resmî train'de olan test görüntüsü (693); N_test = kalan (1.312).
  - Tabakalı S − N, val ile aynı tanımla: sınıf × dx_type × kaynak, N-ağırlıklı, `tahlil_tur4_ezber.py`.
  - Ham ve tabaka-std Δ_S / Δ_N değerleri GA'larıyla raporlanır.
- **Hipotez T-SN1 (H10 − H02c):** Δ_S > Δ_N.
  - Beklenen Δ_S − Δ_N ∈ [+1; +9] (val +5,1).
  - **Kural:** GA(Δ_S − Δ_N) alt > 0 → **"H10'un R18'e kazancı lezyon tekrarlı görüntülerde yoğun" desteklenir**. GA üst < 0 → tersi. Aksi → "ayırt edilemedi".
- **Hipotez T-SN2 (H08c − H02c):** Δ_S ≈ Δ_N. Beklenen [−3; +4]. Aynı kural; GA 0'ı kapsarsa "dengeli dağılım" ile **uyumlu**.
- **Betimsel satırlar:** H10 − H08c ve H13 − H10 (TUT ise; Tur-5 yönü Δ_N > Δ_S).

---

## §3 C0 — yeniden eğitimsiz lezyon-ayrık değerlendirme (KARAR 9; 0 ek maliyet)

- **Ne:** §1'deki resmî modellerin test tahminleri, **C-test alt kümesinde** (1.227 görüntü; C-test ⊂ resmî test, P7) yeniden puanlanır.
  - **Ayrı bir test açılışı değildir.** Aynı tek geçişin bir alt kümesidir.
  - Görüntüler resmî 224 px'tir (28 px baseline için resmî 28 px). Bu görüntülerin lezyonları ne resmî train'de ne resmî val'de; ne eğitimde ne model seçiminde görüldü.
- **Kural C0-1:** H10 − H02c için **GA alt > 0 (C-test bootstrap'ı, 1.227 indeks) → "kazanç lezyon-ayrık görüntülerde de var"**. Aksi → "lezyon-ayrık görüntülerde ayırt edilemedi".
  - Beklenen Δ ∈ [−1,0; +3,0], merkez +1,0. Kaynak: C-val +1,05, N +1,13; görüldü, M1.
  - GA yarı genişliği tahmini ±1,2–1,5 → P(GA alt > 0) ≈ 0,4.
- **Kural C0-2:** H08c − H02c için aynı kural (= H08 kuralı madde 3'ün yeniden eğitimsiz eşi).
  - Beklenen [0; +3,5], merkez +1,7 (C-val +1,75).
- **Betimsel:** H10 − H08c (val C-val −0,70); tüm kolların C-test ACC'si ve dengeli ACC'si.
  - C-test'te çoğunluk tabanı 0,7726'dır (resmî test 0,669). C-test ACC değerleri resmî test ACC değerleriyle **doğrudan kıyaslanmaz**; yalnız kol farkları ve dengeli ACC yorumlanır.
- **Kazanç çöküşü — değerlendirme bileşeni (aynı modeller, en güçlü test):**
  - Tanım: K0_X = Δ_X(resmî test) − Δ_X(C-test). X ∈ {H10 − H02c, H08c − H02c}.
  - **Hipotez KÇ0:** K0_H10 > K0_H08. Beklenen K0_H10 − K0_H08 ∈ [+0,5; +2,5], merkez +1,5. Val karşılığı: (2,91 − 1,05) − (2,09 − 1,75) = +1,52; görüldü, M1.
  - **Kural:** ortak bootstrap (2.005 resmî test indeksi yeniden örneklenir; C-test Δ'sı örneklemdeki C-test indekslerinden hesaplanır). GA alt > 0 → **"H10'un kazancı lezyon tekrarına H08'inkinden daha duyarlı" desteklenir**. GA üst < 0 → tersi. Aksi → "ayırt edilemedi".
  - C-test resmî testin %61'i olduğu için iki Δ pozitif ilişkilidir ve GA daralır.

---

## §4 C1 — KARAR 3 birincil: resmî pikseller + C bölmesi, yeniden eğitim

### 4.1 Veri (ön-kayıtlı yapım ve denetim)
- **Yapım:** `scripts/c_bolme_npz.py` (yeni) → `data/dermamnist_c_resmipiksel_{28,224}.npz`. Resmî MedMNIST(+) pikselleri `dermamnist_c_esleme.csv` (`split_C`, `split_resmi`, `index_resmi`) ile yeniden dizilir.
- **Denetimler (DUR koşulları):**
  - Sayılar 8.215 / 573 / 1.227.
  - Etiketler = C etiketi.
  - Lezyon ayrıklığı 0/0/0.
  - **C-test görüntüleri resmî test'teki aynı indeksin bayt-özdeş kopyası** (C0 ile aynı pikseller → ortak bootstrap mümkün).
  - SHA256 değerleri `k5_dondurma.json`'a eklenir. Bu, test geçişinden sonra yapılır; yapım kuralı burada dondu.
- **DİKKAT:** C-train, resmî test'in **778** görüntüsünü ve resmî val'in 430 görüntüsünü içerir.
  - **C1 modelleri resmî test'te, S_test'te ya da resmî val'de ASLA değerlendirilmez.**
  - C1 kuyruğu ancak resmî test geçişi bitip çıktıları SHA256 ile dondurulduktan **sonra** ve ayrı kuyrukta (`k5-c1`) başlar.

### 4.2 Kollar ve tarifler (AYNEN; KARAR 9)
- **Kollar:** baseline, H01, H02, H08, H10. Seed 0–4. Colab L4, FP32, TF32 kapalı (tekrar kapısı ortamı).
- **Tarif:** val tarifleriyle bit düzeyinde aynı komut satırları. Tek fark veri dosyası ve bölmedir.

| Kol | Tarif | Epoch / C seçimi (C-val, 573) |
|---|---|---|
| baseline | rakip kodu (dokunulmaz; yalnız veri yolu override), 28 px resmî piksel, 100 ep | rakibin resmî kuralı: val-AUC |
| H01 | DINOv2-B r224 donuk CLS + softmax-LR sonda; öznitelikler C-train/C-val için (resmî piksel) | C\* = argmax 5-seed ort. C-val ACC (aynı 8 değerlik ızgara) |
| H02 | R18-IN, 224, tam ince ayar, 30 ep | C-val ACC (eşitlikte AUC) |
| H08 | H02 + eğitimde D4, 30 ep | aynı |
| H10 | DINOv2-B son 4 blok, 15 ep, baş 1e-3 / gövde 1e-5; blok-8 fp16 önbelleği C-train/C-val için yeniden yazılır (resmî piksel) | aynı |

- **M8:** her kol a1 duman → a2 ön-uçuş → a3. Beklenti dosyaları `kuyruk/k5-c1-*.beklenti.json` koşumdan önce yazılır. Aralıklar arıza alarmıdır: baseline C-val ACC [0,74; 0,90], diğerleri [0,78; 0,96]. Gerekçe: C-val nv payı %78; ezber azaldığından alt uç geniş.
- **C1'de H04/R-T katmanı YOK.** C-val için lezyon-gruplu kat dosyası ön-kayıtlı değildir; yeni bir seçim yüzeyi açılmaz.
- H13 Tur-5'te TUT ise **H13-C1** = H02_C1_s ⊕ H10_C1_s, CPU'da, aynı kural.
- **Değerlendirme:** C-test (1.227) yalnız **bir kez**. İstatistik §2.4 ile aynı (indeksler C-test).
  - Holm aileleri C1 içinde aynı yapıdadır: aile C1-1 {H01, H02, H08, H10 (+ H13)} − baseline; aile C1-2 {H08, H10} − H02.

### 4.3 Ön-kayıtlı sorular ve kurallar
- **(i) Lezyon-ayrık bölmede kazanç:**
  - **H10 − H02 (C1): GA alt > 0 → "DINOv2 ince ayarının R18'e kazancı lezyon-ayrık eğitim/test bölmesinde de var"**. Aksi → "ayırt edilemedi". Beklenen Δ ∈ [−1,0; +3,0], merkez +1,0.
  - **H08 − H02 (C1):** aynı kural (= H08 kuralı madde 3). Beklenen [0; +3,5], merkez +1,7.
  - H10 − H08 (C1) betimseldir.
- **(ii) Kazanç çöküşü (HİPOTEZ; yön ve eşik):**
  - Tanım: K_X = Δ_X(resmî test, resmî modeller) − Δ_X(C1, C-test). Bu fark iki bileşene ayrılır ve ikisi de raporlanır:
    - **Değerlendirme bileşeni** K0_X = Δ_X(resmî) − Δ_X(C0) (§3).
    - **Eğitim-bölmesi bileşeni** K1_X = Δ_X(C0) − Δ_X(C1). Bu bileşen için yön ön-kaydı yoktur, keşifseldir.
  - **Hipotez KÇ (ön-kayıtlı yön):** **K_H10 > K_H08**, yani H10'un kazanç düşüşü H08'inkinden büyüktür. Dayanak: tahlil-tur4 ezber bulgusu (kazancın %75'i S; H08 dengeli).
  - **Beklenen büyüklük:** K_H10 − K_H08 ∈ [+0,5; +2,5], merkez +1,5. Ayrıca K_H10 ∈ [+0,5; +3,5] (merkez +1,9), K_H08 ∈ [−1,5; +1,5] (merkez +0,3).
  - **Kural:** ortak bootstrap. 2.005 resmî test indeksi yeniden örneklenir; resmî Δ tüm örneklemden, C1 Δ'sı örneklemdeki C-test indekslerinden hesaplanır. Seed ortalamaları kullanılır.
    - GA(K_H10 − K_H08) alt > 0 → **desteklendi**.
    - GA üst < 0 → **çürütüldü**.
    - Aksi → **"ayırt edilemedi"**.
  - **Güç beyanı:** GA yarı genişliği tahmini ±2,5–3,0, yani MDE ≈ 3 puan. Beklenen merkez (+1,5) altında olduğundan "ayırt edilemedi" olasılığı yüksektir ve bu önceden beyan edilir. §3'teki KÇ0 aynı modelleri kullandığı için daha güçlüdür.
  - **İfade eşiği:** K_H10 / Δ_H10(resmî) ≥ 0,5 (nokta) **ve** KÇ desteklendi → makale cümlesi: "H10'un R18'e resmî bölmedeki kazancının yarısından fazlası lezyon-ayrık bölmede kayboluyor."
- **(iii) Rakibin C1 düzeyi:**
  - baseline C1 C-test ACC beklentisi **[0,79; 0,85]**. Çıpa: Abhishek vd. 2025 Tablo 1, ResNet-18 28 px, C npz: **0,825 ± 0,006** (3 koşu; `literatur-cipa.md` §5). Bizde resmî piksel kullanıldığı için piksel farkı karıştırıcıdır.
  - **Görülmemiş:** baseline C1 ACC > 0,87 → İNCELE (veri yapımı / sızıntı).
  - **Çoğunluk tabanı 0,7726:** her kolun ACC − 0,7726 farkı yazılır. baseline < 0,7726 + 0,01 ise "çoğunluk tabanının üstüne çıkamadı" notu düşülür.
  - **Dengeli ACC C1'in birincil ikincilidir.** Test nv payı %77,3 olduğu için ACC kazançları nv tarafından sıkıştırılır. Her kolun dengeli ACC'si ve 5 sınıflı dengeli ACC'si (C-test'te df 8, vasc 17, akiec 35 → df/vasc hariç satır zorunlu) GA'larıyla raporlanır. **Dengeli için yön ön-kaydı:** H10 − H02 > 0 ve H08 − H02 > 0. Kural (i) ile aynı biçimde etiketlenir, karar vermez.

---

## §5 C2 — ikincil: C'nin yayımlanmış npz'si (KARAR 3: ikincil EVET)

- **Kollar (öneri; teknik karar, kullanıcı aksini demedikçe):** **baseline (C npz 28 px) + H10 + H08 (C npz 224)**. Seed 0–4. Tarifler ve seçim C1 ile aynı.
  - H02 dahil edilmez. C2'nin amacı yayımlanmış hâlle (Abhishek 0,825) kıyas ve piksel/görüntü-hazırlama etkisidir. "Vs H02" kararları C1'de verilir.
- **Kurallar (betimsel; karar vermez):**
  - **(a)** baseline C2 ACC ile 0,825 arasındaki fark |Δ| ≤ 2,0 puan ise "yayımlanmış C sonucuyla uyumlu", değilse fark ve olası nedenler yazılır.
  - **(b)** Kol başına C2 − C1 farkı "görüntü hazırlama etkisi" olarak GA'lı raporlanır (aynı C-test görüntülerinin farklı küçültmesi; eşleştirilmiş görüntü bootstrap'ı).
  - **(c)** H10 − H08 (C2).
- **Maliyet:** baseline ≈ 1,2 + H10 ≈ 1,25 (C 224 pikselden yeni blok-8 önbelleği dahil) + H08 ≈ 0,75 → **≈ 3,2 L4-sa** (üst).
- **Bütçe kapısı (ön-kayıt; sonuca bakmaz):** C2, C1 bittiğinde birleşik defter **≤ 28,8** ise başlar (28,8 + 3,2 ≤ 32). Aksi halde C2 koşulmaz ve sınırlılık olarak yazılır.

---

## §6 Raporlama ve dürüstlük (D1–D4)

1. **Test bir kez.** Resmî test geçişi tek seferdir. C0 bu geçişin alt kümesidir. C1 ve C2'de C-test, kol başına bir kezdir.
2. **Tüm sonuçlar ne çıkarsa çıksın yazılır:** her kol, her katman satırı, T4, M3 satırları, başarısız doğrulamalar. Hiçbir kol "kötü çıktı" diye tablodan çıkarılmaz.
3. **Test sonrası yasaklar:**
   - τ/T/C/epoch yeniden seçimi yok.
   - Kazanan değiştirilmez.
   - Yeni seed, yeni kol ya da yeni alt küme kuralı eklenmez.
   - Holm aileleri değiştirilmez.
   - Test sonrası yapılan her ek analiz "**test sonrası keşifsel**" etiketiyle ve ayrı bölümde verilir.
4. **Hata bulunursa** (yükleme, hizalama, betik): düzeltme yapılır, geçiş yeniden koşulur, **iki sonuç da** yazılır ve makalede "test ikinci kez açıldı; neden: …" beyanı yer alır (`dongu-pipeline.md` döngü sınırları).
5. **Başarısız denemeler tablosu (D4):** H05, H07, H09 ve (ÖLDÜR ise) H13 val sonuçlarıyla, mekanizma cümleleriyle yer alır. Doğrulanamayan kurallar (H08, H10, H13) da bu tabloya "test'te doğrulanamadı" olarak girer.
6. **Sınırlılıklar (zorunlu):**
   - M3 HAYIR (rakibe D4 hakkı yok).
   - Resmî bölmede lezyon tekrarı (%34,6), resmî sonuçların S/N ayrışımıyla birlikte verilir.
   - C-test'te nadir sınıf sayıları küçük.
   - Paralel Colab süreleri yalnız maliyet bilgisidir (R6).
   - Wu & Papież 92,02 sonucu da resmî bölmeye ait.

---

## §7 K5 karar çıkışları — ön-kayıtlı ölçütler (`dongu-pipeline.md` K5; tahlilci önerir, kullanıcı seçer)

W = Tur-5 sonrası kazanan (H13 TUT ise H13, aksi H10).

| Ölçüt | Tanım |
|---|---|
| **G1** (resmî test, rakibe karşı) | W − baseline_c: ΔACC ≥ +2,0 puan ∧ Holm-düzeltilmiş (aile 1) t p < 0,05 ∧ GA alt > 0 |
| **G2** (görülmemiş yok) | W'nin ve doğrulanan kolların görülmemiş eşikleri aşılmamış, ya da aşılmış ama İNCELE "arıza yok" diye kapanmış |
| **G3** (lezyon-ayrık bölmede rakibe karşı) | C1: W_C1 − baseline_C1 (C-test): ΔACC ≥ +2,0 ∧ GA alt > 0 |
| **V** (DINOv2 ↔ eşit önağırlıklı R18) | 2.2 H10 kuralı (resmî) |
| **L** (DINOv2 üstünlüğü lezyon-ayrık) | C1 kuralı (i) H10 − H02 GA alt > 0 |

- **A. GEÇTİ ⇔ G1 ∧ G2 ∧ G3** → iyileştirme makalesi. Tez kademesi mekanik olarak belirlenir:
  - **A-güçlü ⇔ V ∧ L:** "DINOv2 ince ayarı, aynı önağırlık hakkı verilmiş ResNet-18'i hem resmî hem lezyon-ayrık bölmede geçer."
  - **A-kayıtlı ⇔ V ∧ ¬L:** "Rakip her iki bölmede geçildi. DINOv2'nin R18 ince ayarına üstünlüğü resmî (görüntü düzeyi) bölmede doğrulandı, lezyon-ayrık bölmede ayırt edilemedi." Bu cümle S/N (2.6) ve KÇ (§3, §4.3-ii) sonuçlarıyla birlikte verilir.
  - **A-paket ⇔ ¬V:** "Kazanç 224 px + önağırlık + ince ayar paketinden geliyor. DINOv2 ↔ R18 omurga farkı ayırt edilemedi." Varsa H08 R18 satırı (2.1) bu kademede belirtilir.
- **B. GEÇEMEDİ, ANLAŞILDI ⇔ G2 ∧ (¬G1 ∨ ¬G3)** ve mekanizma açıklaması `karar.md`'de yazılı.
  - **¬G1:** W, resmî test'te rakibi pratik marjla geçemedi. Val ↔ test kayması analizi yapılır.
  - **G1 ∧ ¬G3:** "Resmî bölmedeki kazanç lezyon-ayrık bölmede kayboluyor; rakibin geçilmesi bölme protokolüne duyarlı". Bu bir analiz / protokol makalesidir; dayanak S/N ve KÇ'dir.
- **C. TEKRARLANAMADI:** Replikasyon yerelde ve Colab'da YEŞİL (iki kez). Bu yüzden C yalnız şu durumda kullanılır: test geçişinde baseline özdeşliği (dondurulmuş test ACC'si `tahmin_test.npy`'den ±1e-4 ile) **veya** P6 özdeşliği sağlanamaz **ve** teşhis edilemez. Teşhis edilirse düzeltme + §6.4 beyanı yapılır, C çıkışı kullanılmaz.
- **¬G2:** önce İNCELE (sızıntı, yükleme, hizalama). İNCELE kapanmadan çıkış yazılmaz.
- **Seçim:** `deney-tahlilci` (mod: karar) `sonuclar/karar.md` taslağında yukarıdaki ölçütleri mekanik olarak işaretler ve önerisini belirtir. **Kullanıcı seçer**; seçim `karar.md`'ye ve `konu.md`'ye yazılır.

---

## §8 Bütçe ve sıra (birleşik defter; L4-sa ≈ GPU-sa)

| Kalem | Maliyet | Defter (önce → sonra) |
|---|---|---|
| Şimdiki (yerel 15,3 + Colab ≈ 6,2) + H12 (koşuyor, ≈ 0,15) | — | ≈ 21,7 |
| H13 / Tur-5 (CPU) | 0 GPU, ≈ 0,3 sa uygulama | ≈ 21,7 |
| Resmî test geçişi (çıkarım + H01/H12 test öznitelikleri) + C0 | ≈ 0,2 | → ≈ 21,9 |
| C1 (baseline 1,2 · H02 0,75 · H08 0,75 · H10 1,25 · H01 0,15) | ≈ 4,1 (3 paralel oturumda ≈ 1,5 sa duvar) | → ≈ 26,0 |
| C2 (bütçe kapısına bağlı) | ≤ 3,2 | → ≤ 29,2 / 32 |

**Sıra:**
1. P1–P8.
2. Kullanıcı onayı.
3. Resmî test geçişi + C0 + `k5_test.py`, tek çağrı.
4. Resmî sonuç raporu SHA256 ile donar.
5. C1 (`k5-c1`).
6. C2 (kapı).
7. `karar.md`.

---

## §9 M1 beyanı (bu ön-kaydın yazıldığı anda görülmüş bilgi)

- **Görüldü:**
  - Tüm val sonuçları (Tur-1…4).
  - Tek modellerin val S/N ve **C-val alt kümesi** ayrışımları (tahlil-tur4 §4).
  - **baseline'ın resmî test sayıları** (yerel 0,7296 ± 0,0176; Colab 0,7395 ± 0,0053; replikasyon kapısı, izinli).
- **Görülmedi:**
  - baseline dışındaki **hiçbir kolun test çıktısı**. Varyant `metrics.json`'larında `test*` anahtarı 0 (tahlil-tur4 §0).
  - C-test üzerindeki hiçbir model sonucu.
  - Aileler arası topluluk satırları (H13 koşulmadı).
- Bu dosyadaki beklentiler val sonuçlarından türetildi ve öyle beyan edildi.
- Kurallar val sonuçlarına göre ayarlanmadı:
  - H10 kuralı, 09:42'de (H10 koşulmadan önce) yazılmış H08 kuralının birebir aynısıdır.
  - C0/C1 "GA alt > 0" kuralı, H08 kuralı madde 3 ve tahlil-tur4 §8.4 ile aynıdır.

## §10 Kullanıcı kararı gerektiren maddeler
1. **Bu ön-kaydın onayı ve test kilidinin açılması** (K5 kritik kapı).
2. **Tur-5 → K5 geçişi:** durma sayacı K5'i tetiklemez (SIRALAMA-TUR5 §5). Öneri: K5'e geç.

Diğer her şey burada ön-kayıtlıdır ve teknik varsayılanla yürür: C2 kolları, bütçe kapısı, Holm aileleri, katman parametreleri.

> **EK b (2026-10-11, test açılmadan):** `hipotezler/K5-ONKAYIT-b.md` — §2.6 T-SN1/T-SN2 kuralı HAM Δ_S − Δ_N ile (tabaka-std ikincil);
> rakibin C-test epoch değerlendirmesi protokol notu; K5 sonrası test dosyalarının arşive kopyalanması.
