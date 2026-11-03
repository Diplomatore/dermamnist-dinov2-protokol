# Hata Analizi (K3.5) — 3.1 DermaMNIST · rakip ResNet-18 (28 px) · YALNIZ VALIDATION

*Tahlilci, Mod 1 · 2026-10-10 · test tahminleri açılmadı (tahmin_test*.npy, etiket_test.npy okunmadı; gecmis.json'dan test_* anahtarları atıldı).*

| Girdi | Değer |
|---|---|
| Koşumlar (varyant=`baseline`, `.done` rc=0, kuyruk k3-baseline-yerel-v2 a3) | `20261010-171037_baseline_s0`, `-174311_s1`, `-181653_s2`, `-184922_s3`, `-192128_s4` (seed 0–4). rc=95 olan 094349/104956 **kullanılmadı** |
| Tahmin | `tahmin_val.npy` = resmî seçim (val-AUC en iyi epoch) softmax; `tahmin_val_son.npy` = 100. epoch |
| Ek veri | `data/splits/dermamnist_lezyon_esleme.csv` (S/N), `data/meta/HAM10000_metadata.csv` (dx_type, kaynak), `data/splits/dermamnist_c_esleme.csv` (C-val alt kümesi) |
| Betik (tekrar üretilebilir; K5'te kazanan için aynısı) | `scripts/hata_kirilim.py` → `sonuclar/hata-kirilim.csv` (321 satır, uzun biçim) + `sonuclar/hata_kirilim_ozet.json` |
| Komut (WS kökü, CPU) | `PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I scripts/hata_kirilim.py --varyant baseline` (K5: `--varyant <kazanan> --etiket <ad>` veya `--kosumlar <dizinler>`) |
| Bütünlük denetimleri (S-grubu) | 5/5 koşumda `etiket_val.npy` = npz `val_labels` (1.003/1.003); lezyon CSV `dx` = npz etiketi (1.003/1.003); S/N maskesinin hizası bağımsız olarak doğrulanmış (manifest: RMSE≤5 olan 4/4 val→train yakın-kopya aynı lezyon). **S-ihlali yok.** |

Notasyon: "ort ± std" = 5 seed (ddof=1); "puan" = yüzde puanı; val n = 1.003; çoğunluk (nv) tabanı = 671/1.003 = **0,669**.

---

## 0. Özet tablo (K4'ün iç çıpası)

| metrik (val) | seçilen epoch (resmî, val-AUC) | son epoch (100) |
|---|---|---|
| ACC | **0,7396 ± 0,0205** | 0,7386 ± 0,0069 |
| AUC (makro OvR) | **0,9228 ± 0,0018** | 0,9049 ± 0,0057 |
| dengeli ACC | **0,4621 ± 0,0413** | 0,4802 ± 0,0275 |
| makro-F1 | 0,4602 ± 0,0451 | 0,5063 ± 0,0272 |
| ECE-15 | 0,0365 ± 0,0099 | 0,2099 ± 0,0051 |
| NLL | 0,695 ± 0,019 | 1,772 ± 0,024 |
| tahminlerin nv payı | 0,672 ± 0,056 | 0,696 ± 0,008 |
| en iyi epoch | 14, 11, 10, 11, 8 (ort 10,8) | — |

D-temel çizgi: val ACC 0,7396 literatürün 28 px bandında (0,7338–0,742; `literatur-cipa.md` §2.1). "Zayıf baseline" itirazı yok.

---

## 1. Sınıf bazlı kırılım ve çoğunluk sınıfı yanlılığı

| sınıf (val destek) | duyarlılık | kesinlik | F1 | OvR AUC | tahmin sayısı (gerçek) | hata kütlesi payı |
|---|---|---|---|---|---|---|
| akiec (33) | 0,382 ± 0,172 | 0,464 ± 0,124 | 0,382 ± 0,113 | 0,940 ± 0,011 | 30,8 ± 20,1 | %7,9 |
| bcc (52) | 0,500 ± 0,198 | 0,568 ± 0,121 | 0,491 ± 0,108 | 0,942 ± 0,002 | 51,0 ± 31,0 | %9,8 |
| bkl (110) | 0,511 ± 0,059 | 0,513 ± 0,070 | 0,506 ± 0,022 | **0,882** ± 0,007 | 112,6 ± 29,5 | %20,8 |
| df (12) | **0,017 ± 0,037** | 0,20 ± 0,45 | 0,031 ± 0,069 | 0,949 ± 0,008 | **0,2 ± 0,4** | %4,5 |
| mel (111) | 0,495 ± 0,135 | 0,493 ± 0,127 | 0,468 ± 0,025 | **0,874** ± 0,003 | 122,0 ± 53,5 | %21,7 |
| nv (671) | 0,872 ± 0,049 | 0,870 ± 0,024 | 0,870 ± 0,013 | 0,909 ± 0,003 | 674,0 ± 56,0 | %32,3 |
| vasc (14) | 0,457 ± 0,164 | 0,546 ± 0,101 | 0,474 ± 0,104 | 0,964 ± 0,018 | 12,4 ± 5,9 | %2,9 |

Karışıklık matrisi (sayım, 5-seed ortalaması; satır = gerçek, sütun = tahmin; std'ler CSV `karisiklik` bölümünde):

| gerçek \ tahmin | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|---|
| akiec | **12,6** | 5,8 | 8,0 | 0,0 | 3,2 | 3,2 | 0,2 |
| bcc | 6,2 | **26,0** | 10,2 | 0,0 | 2,6 | 6,4 | 0,6 |
| bkl | 6,0 | 7,0 | **56,2** | 0,0 | 10,6 | 30,0 | 0,2 |
| df | 2,2 | 5,8 | 2,6 | **0,2** | 0,0 | 1,2 | 0,0 |
| mel | 2,8 | 0,8 | 8,6 | 0,0 | **55,0** | 43,8 | 0,0 |
| nv | 0,8 | 4,0 | 26,4 | 0,0 | 49,4 | **585,4** | 5,0 |
| vasc | 0,2 | 1,6 | 0,6 | 0,0 | 1,2 | 4,0 | **6,4** |

Ölçülen gözlemler:
- **Toplam hata ≈ 261,2 / 1.003.** nv↔mel çifti 93,2 hata (%35,7), nv↔bkl 56,4 (%21,6), mel↔bkl 19,2 (%7,4). **nv–mel–bkl üçgeni hataların %64,6'sı.**
- **Çoğunluk yanlılığı tek yönlü değil, simetrik bir sınır karmaşası.** Tahminlerin nv payı 0,672 ± 0,056, gerçek pay 0,669. nv-dışı 332 örneğin 88,6 ± 23,7'si (%26,7) nv'ye gidiyor (hataların %34,6'sı "nv tahmini"). Buna karşılık 85,6 ± 32,6 nv de başka sınıfa (çoğu mel 49,4 ve bkl 26,4) kaçıyor. H01/H02 kartlarındaki "çoğunluk sınıfına çöken kararlar" ifadesi ölçümle **kısmen** doğrulanıyor: mel→nv (%39,5) ve bkl→nv (%27,3) güçlü, ama ters yön de aynı büyüklükte.
- **Nadir sınıflar (df, vasc, akiec; 59 örnek):** havuz hata oranı **%67,5 ± 9,5**, makro duyarlılık 0,285 ± 0,073. df'yi model neredeyse hiç tahmin etmiyor (seed başına 0,2 tahmin; duyarlılık 1/60). df'nin hataları çoğunlukla bcc'ye (5,8/12) gidiyor. Bu üç sınıf toplam hatanın %15,2'si.
- **Sıralama bilgisi var, karar sınırı yanlış (nadir sınıflar):** df/vasc/akiec/bcc'nin OvR AUC'si 0,94–0,96 ama duyarlılıkları 0,02–0,50. **Sıralama bilgisinin kendisi zayıf (temsil sorunu):** en düşük AUC mel (0,874) ve bkl (0,882). Bu iki farklı arıza iki farklı manipülasyon ailesine gidiyor (§7).
- Sınıf duyarlılıklarının seed std'si büyük (akiec ±17, bcc ±20, mel ±13 puan). Tek seed'lik sınıf kırılımı yorumlanamaz.

## 2. Dengeli ACC (0,462) ve ACC (0,740): dengesizliğin etkisi

- ACC = Σ π_c · duyarlılık_c. **nv'nin katkısı 0,669 × 0,872 = 0,584.** Kalan 6 sınıfın toplam katkısı yalnız 0,156, ortalama duyarlılıkları 0,394.
- Çoğunluk tabanına göre kazanç: ACC +7,1 puan (0,669 → 0,740), dengeli ACC +31,9 puan (1/7 = 0,143 → 0,462).
- **Seed'ler arası ACC'yi belirleyen şey nv'ye ne kadar tahmin verildiği:** corr(val ACC, tahmin nv payı) = **+0,88** (5 seed). corr(dengeli ACC, nv payı) = −0,08. Örnek: seed 0/1'in nv payı 0,616/0,619, ACC'si 0,732/0,708; seed 2/3'ün nv payı 0,722/0,736, ACC'si 0,753/0,760. En yüksek dengeli ACC ise seed 0'da (0,502).
  → **I-grubu riski:** Ana metrik ACC olduğu sürece, nv payını artıran her değişiklik dengeli ACC'de kazanç olmadan ACC "kazanır". Tur-2+ kartlarına ön-beyanlı ikincil olarak `tahmin_nv_payi` eklenmeli. ACC artışı nv payındaki artışla birlikte geliyorsa bu, leaderboard'da ödünleşim notu olarak yazılmalı.
- Teşhis (val'de ayar YAPILMADAN, sabit τ=1): train önseliyle düzeltme p(c|x)/π_train(c) (Menon vd. 2021 logit ayarı) dengeli ACC'yi **0,462 → 0,525 ± 0,025 (+6,3 puan)** çıkarıyor, ama ACC'yi **0,740 → 0,583 ± 0,028 (−15,7 puan)** düşürüyor. nv payı 0,42'ye iniyor. Bu bir sonuç değil, Tur-2 kartının yön ve büyüklük ön-kaydı için hammadde. τ=1 ACC için fazla agresif. ACC-nötr bir dengesizlik müdahalesi ancak ılımlı bir ayarla mümkün (Zhang & Li 2026 T1a: WS ile ACC −1,5 puan, makro-ACC +4,2; `literatur-cipa.md` §4(c)).

## 3. Kalibrasyon

| | seçilen epoch | son epoch |
|---|---|---|
| ECE-15 | **0,0365 ± 0,0099** | **0,2099 ± 0,0051** |
| ort. güven − ACC (aşırı güven) | +1,5 ± 1,3 puan | +20,9 puan (0,948 − 0,739) |
| NLL | 0,695 | 1,772 |

Güven histogramı (seçilen epoch, 15 eşit kutu, 5-seed ortalaması; ilk 3 kutu boş):

| kutu | (0,20–0,27] | (0,27–0,33] | (0,33–0,40] | (0,40–0,47] | (0,47–0,53] | (0,53–0,60] | (0,60–0,67] | (0,67–0,73] | (0,73–0,80] | (0,80–0,87] | (0,87–0,93] | (0,93–1] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| örnek | 2,2 | 18,8 | 41,6 | 59,6 | 80,4 | 98,2 | 70,2 | 66,4 | 74,0 | 80,6 | 75,8 | 335,2 |
| ort. güven | 0,255 | 0,306 | 0,370 | 0,435 | 0,501 | 0,567 | 0,633 | 0,701 | 0,766 | 0,832 | 0,901 | 0,987 |
| doğruluk | 0,000 | 0,334 | 0,357 | 0,446 | 0,495 | 0,568 | 0,644 | 0,681 | 0,722 | 0,758 | 0,859 | 0,982 |

- Seçilen epoch **iyi kalibre**. Aşırı güven yalnız 0,73–0,93 bandında görülüyor (4–7 puan). Güveni ≥ 0,9 olan 374,6 örneğin doğruluğu 0,976, yalnız 9,2'si yanlış. Yanlış nv tahminlerinin ortalama güveni 0,634, yani hatalar çoğunlukla "kararsız" hatalar.
- Son epoch ağır biçimde aşırı güvenli (train ACC = 1,000, ortalama güven 0,948). Literatürle uyum (D): Medeiros 2026 T2 sıfırdan R18-28 test ECE **0,196** (`literatur-cipa.md` §4(d)). Bizim son epoch val ECE'miz 0,210 bu değere yakın. Protokolün seçtiği erken epoch ise 0,037.
- **Kart revizyonu (H01):** H01 "ECE baseline'dan düşük" yönünü Medeiros'un 0,196 değerine dayandırıyor. Oysa protokol baseline'ının ölçülen val ECE'si **0,0365**. LR sondası için yön artık belirsiz. Kıyas değeri 0,0365 olarak yazılmalı ve yön "keşifsel" olmalı. Kart TASLAK olduğu için bu düzeltme meşru.

## 4. Lezyon sızıntısı kırılımı: S (lezyonu train'de, 348) ve N (655)

| ölçü | seçilen epoch | son epoch |
|---|---|---|
| ACC_S | 0,617 ± 0,038 | 0,627 ± 0,012 |
| ACC_N | 0,805 ± 0,013 | 0,798 ± 0,006 |
| **ham S − N** | **−18,8 ± 2,9 puan** | −17,1 ± 1,0 |
| sınıf-düzeltmeli S − N (S'nin sınıf-koşullu ACC'si, N sınıf oranlarıyla; `degerlendir.py` tanımı) | −11,4 ± 5,6 (sınıf içi perm. iki yönlü p = 1e-4) | — |
| **tabakalı S − N (sınıf × dx_type × kaynak; N-ağırlıklı)** | **+2,8 ± 4,1 puan** (4/5 seed > 0; görüntü düzeyi perm. tek yönlü p ≈ 0,20) | **+5,1 ± 2,4 puan** (5/5 seed > 0) |

**Beklentinin tersi → arıza alarmı → teşhis edildi (D-grubu, açıklandı):**
- Ham fark ezber beklentisinin (S > N) **tersi**. Önce hizalama hatası arandı (S-grubu): etiket ve dx eşleşmesi 1.003/1.003, manifestteki bağımsız yakın-kopya denetimi aynı lezyonu buluyor. Hata yok.
- **Neden: Simpson paradoksu, karıştırıcı = görüntü kaynağı / tanı türü.** HAM10000 metadata'sına göre val'deki 357 `dx_type=follow_up` nv görüntüsünün (tamamı `dataset=vidir_molemax`) **hepsi tek görüntülü lezyon**, yani hepsi N'de (S'de 0). Bu alt grup neredeyse kusursuz sınıflanıyor (ACC **0,985 ± 0,007**). S ise %85 `histo` (296/348) ve histo görüntüleri zor (ACC 0,596). Sınıf içinde de (nv) S 0,765 / N 0,911 farkı bu kompozisyondan geliyor: nv-histo'da S 0,768 / N 0,699, nv-consensus'ta 0,753 / 0,809.
- **Tabakalı tahmin ezber yönünde ve eğitim uyumuyla büyüyor:** seçilen erken epoch'ta (train ACC 0,75–0,79) +2,8 puan (anlamlı değil). Son epoch'ta (train ACC 1,00) +5,1 puan ve 5/5 seed pozitif. Not: seed'ler aynı val görüntülerini paylaştığı için 5/5 bağımsız kanıt değildir. Tabakalar N'nin yalnız %41,8'ini kapsıyor, çünkü follow_up tabakasında S yok.
- **I-grubu sonucu (kart revizyonu, ZORUNLU):** H01/H02/H03'ün ön-kayıtlı "her model için S − N > 0" yönü **ham ölçüyle baseline'da zaten yanlışlanıyor**. Sınıf-düzeltmeli ölçü (−11,4) de karıştırıcıyı kaldırmıyor. Sızıntı ölçüsü **sınıf × dx_type × kaynak tabakalı S − N** olarak yeniden tanımlanmalı (`varyantlar/ortak/degerlendir.py`'ye `<on>_sizinti_farki_tabakali` eklenmeli; tanım `scripts/hata_kirilim.py` içinde). Kart karşılaştırmaları (H02 > baseline, H02 > H01) bu ölçüyle yapılmalı. Baseline referansı: **+2,8 (seçilen) / +5,1 (son epoch) puan**.
- **DermaMNIST-C ile ilişkisi (keşifsel, K5 yorumu için):** Resmî val'in C-val'de kalan 573 görüntüsünde (nv payı 0,782 = çoğunluk tabanı) resmî train ile eğitilmiş baseline **0,826 ± 0,017** ACC alıyor. C-train'e taşınan 430 görüntüde ise 0,625 ± 0,028. Abhishek vd. 2025 (Tablo 1; `literatur-cipa.md` §5) C üzerinde yeniden eğitilmiş R18-28 için 0,825 bildiriyor. Kıyas doğrudan değil (farklı train kümesi; onların sayısı C-test, bizimki C-val alt kümesi). Yine de C'deki yüksek ACC'nin büyük ölçüde **bileşimden** (zor, çok görüntülü histo lezyonlarının çıkması ve nv/molemax payının artması) geldiğini düşündürüyor; "sızıntı kalktı, sonuç düştü" çerçevesi beklenmemeli. K5'teki C analizi bu bileşim etkisini ayrıştırmalı (aynı tabakalama ile).

## 5. Seed varyansı ve val-AUC seçiminin oynaklığı

| ölçü | değer |
|---|---|
| val ACC seed std (seçilen) | **2,05 puan**. Binom örneklem SE'si (√(p(1−p)/1003)) 1,39 puan, yani seed varyansı örneklem gürültüsünün ~1,5 katı |
| val ACC seed std (son epoch) | 0,69 puan |
| seçilen epoch | 8–14 (ort 10,8). val kaybının en düşük olduğu epoch 8–12 (ort 10,0): val-AUC seçimi pratikte "val kaybı minimumu" |
| seçilen epoch'ta train ACC | 0,752–0,786 (yetersiz uyum noktası). Epoch 30'da 0,987, epoch 51'den itibaren 1,000 |
| seçilen epoch ± 2 içinde val ACC aralığı | **3,9 ± 1,0 puan**. Epoch 5–20 penceresinde epoch'tan epoch'a std 1,6 puan |
| 100 epoch içindeki en yüksek val ACC | 0,7593 ± 0,0025 (epoch 11–15) → **val-ACC ile seçim yapılsaydı +1,97 puan iyimserlik** |
| val AUC eğrisi (seed ort.) | ep10 0,918 → ep20 0,905 → ep50 0,900 → ep51 (LR düşüşü) 0,905 → ep100 0,905 |
| seed çiftleri arası argmax anlaşmazlığı | **%19,0 ± 3,4** (10 çift) |
| 5-seed olasılık ortalaması (teşhis) | ACC **0,771** (+3,1), AUC 0,935, dengeli ACC 0,506, ECE 0,049 |

Yorum ve sonuçlar:
- Seçim yetersiz uyumlu, gürültülü bir platoya düşüyor. Platonun hangi noktasının seçildiği nv sınırını kaydırıyor (§2: ACC ↔ nv payı r = 0,88). Son-epoch modeli daha kararlı (std 0,69) ama aşırı güvenli ve AUC'si 1,8 puan düşük.
- **İstatistik gücü (K4 kararları için kritik):** 5 seed'li eşleştirilmiş t-testte (tek yönlü α = 0,05, güç %80) en küçük saptanabilir etki: iki kol da baseline kadar oynaksa **≈ 3,9 puan**; varyant kolu sonda gibi neredeyse sıfır varyanslıysa **≈ 2,7 puan**. H01/H02'nin beklenen +8–12'si rahatça saptanır. **Tur-2/Tur-3'ün ±1–2 puanlık etkileri R18 baseline'ına karşı saptanamaz.** Bu turlar düşük varyanslı ebeveyne (H01 sondası) karşı kıyaslanmalı ve **aynı val görüntüleri üzerinde eşleştirilmiş bootstrap / McNemar** ile desteklenmeli.
- **H03 T3 kuralı** ("Δ(r28 − baseline) ≥ +2,0 ve p < 0,05") güç açısından sınırda: Δ = +2,0 puanda beklenen t ≈ 2,18 (sd 2,05, n = 5), p ≈ 0,047. Kartta bootstrap GA'nın neden eşlik ettiği bununla gerekçelendirilmeli. Eşik değişmez.
- **I-grubu riski (H02):** H02 epoch'u val ACC ile seçiyor, baseline val AUC ile. Seçim farkının baseline'daki ölçülen iyimserliği **+1,97 puan**. H02'nin TUT/ÖLDÜR kararı ve baseline'a karşı Δ, aynı seçim kuralıyla da (`val_acc_aucsecim`) raporlanmalı. Kart bu anahtarı zaten yazıyor, ama kararda hangisinin kullanılacağı ön-kayıtta açıkça belirtilmeli.

## 6. Zor örnekler (5 seed'in hepsinde yanlış)

| sınıf | destek | hep yanlış | S içinde | N içinde | hep doğru | hep-yanlışların çoğunluk oyu |
|---|---|---|---|---|---|---|
| akiec | 33 | 11 (%33) | 3 | 8 | 3 | bkl 6, mel 3, bcc 1, nv 1 |
| bcc | 52 | 13 (%25) | 8 | 5 | 7 | bkl 5, nv 4, akiec 2, mel 2 |
| bkl | 110 | 33 (%30) | 15 | 18 | 35 | **nv 20**, mel 6, bcc 4, akiec 3 |
| df | 12 | **11 (%92)** | 6 | 5 | 0 | **bcc 7**, bkl 2, akiec 1, nv 1 |
| mel | 111 | 30 (%27) | 22 | 8 | 28 | **nv 25**, bkl 3, akiec 2 |
| nv | 671 | 24 (%3,6) | 9 | 15 | 505 | bkl 13, mel 8, bcc 2, vasc 1 |
| vasc | 14 | 5 (%36) | 1 | 4 | 2 | nv 3, bcc 1, mel 1 |
| **toplam** | 1.003 | **127 (%12,7)** | 64 | 63 | **580 (%57,8)** | kararsız (1–4 seed yanlış): **296 (%29,5)** |

- Hep-yanlış oranı S'de %18,4, N'de %9,6. Bu fark §4'teki kaynak karıştırıcısıyla aynı yönde ve tek başına ezber kanıtı değil.
- Hep-yanlışların ortalama güveni 0,618. Sistematik ama "emin" olmayan hatalar.
- Tablonun gösterdiği iki kütle: (a) **mel→nv ve bkl→nv** (45 kalıcı hata; temsil/çözünürlük), (b) **df→bcc** ve nadir sınıfların kalıcı kaybı (karar sınırı/önsel). Ayrıca **296 kararsız örnek** var: varyans kütlesi; topluluk (A9) ve seçim kuralı bunu hedefler.
- İndeks listesi: `hata_kirilim_ozet.json` → `zor.hep_yanlis_indeks` (K5'te "önce/sonra" için).

## 6b. Ek kırılım: görüntü kaynağı / tanı türü (HAM10000 metadata)

| alt grup | n | val ACC | not |
|---|---|---|---|
| kaynak = vidir_molemax | 384 | **0,949 ± 0,006** | 362'si nv. dx_type = follow_up 357 (ACC 0,985) |
| **molemax dışı** | **619** | **0,609 ± 0,033** | dengeli ACC 0,457 |
| dx_type = histo | 541 | 0,596 ± 0,033 | S'nin %85'i |
| dx_type = consensus | 98 | 0,631 ± 0,036 | |
| kaynak = rosendahl | 228 | 0,531 ± 0,025 | en zor kaynak |
| kaynak = vidir_modern | 346 | 0,649 ± 0,035 | |

- Val'in %38'i (molemax) neredeyse kusursuz. Hataların yalnız ≈ 19,4'ü (%7,4) bu gruptan geliyor. **Hata kütlesinin %92,6'sı 619 molemax-dışı görüntüde.** ACC'deki her iyileşme pratikte bu alt kümeden gelmek zorunda. Bu yüzden **`val_acc_molemax_disi`** (ve tabakalı sızıntı farkı) bütün kartlara ön-beyanlı ikincil olarak eklenmeli. Bu alt küme aynı zamanda "kaynak kısayolu"na en az duyarlı ölçü.
- Kaynak kısayolu yorumu (follow_up nv'nin cihaz/çekim izinden tanınması) **ölçülmedi, bir hipotez**. Tur-3 renk augmentasyonu bu ölçüyle sınanabilir (§7 Z4).

---

## 7. Manipülasyon hammaddesi (ölçülmüş zayıflık → aile/kart)

| # | Zayıflık (ölçülen sayı) | Hedefleyen aile (manipulasyon-kutuphanesi) | Kart / tur | Ön-kayda girmesi gereken ölçü |
|---|---|---|---|---|
| **Z1** | **nv–mel–bkl üçgeni hataların %64,6'sı.** En düşük OvR AUC mel 0,874 ve bkl 0,882, yani sıralama bilgisi de zayıf (temsil/çözünürlük sorunu). mel→nv 43,8/111, bkl→nv 30,0/110. Molemax-dışı ACC 0,609 | **A4** ön-eğitim/omurga (tipik 1–10 puan, MedMNIST'te büyük) | **H01** (paket), **H03** (çözünürlük payı / temsil payı), **H02** (CNN eşi) | mel ve bkl OvR AUC, mel/bkl duyarlılık, `val_acc_molemax_disi`. Keşifsel yön: H03'te r büyüdükçe mel/bkl AUC artar |
| **Z2** | **Nadir sınıflarda bilgi var ama karar yok:** df duyarlılık 0,017 (AUC 0,949), vasc 0,457 (0,964), akiec 0,382 (0,940). Havuz hata oranı %67,5. Dengeli ACC 0,462. Teşhis: τ=1 önsel düzeltme dengeli ACC'de +6,3, ACC'de −15,7 puan | **A5** kayıp (sınıf ağırlığı, focal, logit ayarı; tipik 0,3–2 puan alt-gruplarda) + **A8** karar eşiği | **Tur-2** (H01 ebeveynli, önbellekli öznitelik) | Birincil etki dengeli ACC ve makro-F1 (beklenti +2…+4, Zhang & Li T1). ACC'de beklenti 0 ± 1 (τ ılımlıysa), τ = 1'de belirgin düşüş. `tahmin_nv_payi`, df/vasc/akiec duyarlılığı. Görülmemiş: ACC'de > +2 puan (literatur-cipa §4(c)) |
| **Z3** | **Seed/seçim varyansı:** val ACC std 2,05 puan (binom SE'nin 1,5 katı), ACC ↔ nv payı r = 0,88, seed çiftleri arası %19 anlaşmazlık, 296 kararsız örnek, ± 2 epoch içinde 3,9 puan oynaklık. Teşhis: 5-seed ortalaması +3,1 ACC / +4,4 dengeli ACC | **A9** topluluk (tipik 0,3–1 puan; burada varyans yüksek olduğu için daha büyük olabilir) + **A11** seçim kuralı/EMA | **Tur-5** son itme (rakibe de topluluk hakkı, M3). Kart değil, tasarım kuralı: Tur-2/3 kıyasları H01 ebeveynine karşı + eşleştirilmiş görüntü bootstrap'ı | A9 kartında "+3,1" bir **R18 baseline teşhisi**dir, H01 sondası için beklenti değildir (dışbükey sondanın seed varyansı çok küçük, Wu ±0,08) |
| **Z4** | **Aşırı uyum + kaynak kısayolu:** train ACC epoch 51'den itibaren 1,000. Val AUC 0,918'den (ep10) 0,905'e iniyor. Son epoch ECE 0,210. Molemax 0,949 ile molemax-dışı 0,609 arasında 34 puan fark | **A3** augmentasyon (tipik 0,3–1,5) + **A8** TTA (Medeiros +1,6 puan, ECE 0,196 → 0,039; 28 px) | **Tur-3** (her biri ayrı kart) | `val_acc_molemax_disi` (augmentasyonun asıl hedefi). Kalibrasyon kıyası **doğru referansla** yapılmalı: TTA'nın ECE kazancı yakınsamış modele (≈ 0,21) göre büyük, seçilmiş erken epoch'a (0,037) göre yok. A3 için DermaMNIST'e özgü literatür etkisi **yok** (literatur-cipa §4(d)), bu yüzden beklenti "belirsiz" yazılmalı |
| **Z5** | **Sızıntı ölçüsü karıştırıcılı:** ham S − N −18,8, sınıf-düzeltmeli −11,4, tabakalı +2,8 / +5,1 (son epoch) | A12 tadında (ölçüm) | H01–H03 sızıntı ön-beyanı + K5 C analizi | Tabakalı S − N, bütün kartlarda aynı tanımla |

## 8. Kart revizyonu için sayılar (orkestrasyon_durumu.md "Varyant kodu hazır" madde 3–4)

**Baseline iç çıpası (val, 5 seed, resmî seçim):** ACC **B = 0,7396** (std 0,0205) · AUC **0,9228** (std 0,0018) · dengeli ACC **0,4621** (std 0,0413) · makro-F1 0,4602 · ECE 0,0365. Çoğunluk tabanı 0,669. Val ACC binom SE ≈ 1,39 puan (p ≈ 0,74) / ≈ 1,16 puan (p ≈ 0,84).

**Madde 3 çakışması çözülüyor:** Ön-uçuştaki ~0,78 yerine tam val ortalaması 0,7396 kullanılınca, "+8–12 puan" mutlak olarak **[0,820; 0,860]** ediyor. Bu, görülmemiş eşiği 0,88'in (= B + 14,0 puan) 2 puan altında kalıyor. Çakışma yok. Literatür değerleri test kümesine ait. Val ile test aynı sınıf oranlarına (nv %66,9) ve aynı sızıntı oranına (%34,7 / %34,6; manifest) sahip olduğu için val/test kayması 0 varsayılıyor (test bu analizde kullanılmadı).

| Kart | Kaynaklı mutlak beklenti (val ACC) | Δ = mutlak − B | Görülmemiş (değişmez; literatur-cipa) | Öldürme / T eşikleri mutlak karşılığı | AUC beklentisi (mutlak) | Öneri |
|---|---|---|---|---|---|---|
| **H01** DINOv2-B LP r224 | **[0,820; 0,860]**: Wu 2025 83,97 ± 0,08 ve Doerrich 2025 S4 84,42 ± 0,23 (test) ± ~2 binom SE (val, p ≈ 0,84 → ± 2,3 puan) | **+8,0 … +12,0** (merkez +10,5) | val_acc > **0,88** (B + 14,0) · AUC > 0,99 | Δ < +0,020 ⇔ val_acc < **0,7596**; alt sınır 0,690 aynen | [0,955; 0,980] (Wu 96,62, Doerrich 96,83; Δ +3,2 … +5,7) | Kartın "+8…+12" aralığı korunur. Yanına mutlak [0,820; 0,860] yazılır. ECE kıyas değeri 0,0365, yön keşifsel (§3) |
| **H02** R18-IN FT r224 | **[0,820; 0,885]**: Doerrich 82,33, Wu 86,05, Zhang 2026 T1d 0,882 ± 0,002 | **+8,0 … +14,5** | val_acc > **0,90** (B + 16,0) · AUC > 0,99 | Δ < +0,020 ⇔ < 0,7596 | [0,950; 0,975] (Doerrich 95,27, Zhang 97,3) | Kartta yazan "+8…+15" B + 15 = 0,890 ediyor ve 0,90 eşiğinin yalnız 1,0 puan (≈ 1 SE) altında kalıyor. **Üst ucu +14,5'e (0,885 ≈ Zhang 0,882) çekme önerisi.** Eşik 0,90 gevşetilmez. Karar Δ'sı ayrıca `val_acc_aucsecim` ile (seçim iyimserliği +1,97 puan; §5) |
| **H03** r28 | **[0,730; 0,780]**: Ren 2025 0,725 ± 0,034 … Wu 77,37 (Δ'lar rakip 0,735'e göre −1 … +4) | **−1,0 … +4,0** | r28 > **0,80** (B + 6,0); r64 > 0,85; r128 > 0,87; Δ(224 − 28) > +13 | T3 "≥ +2,0" ⇔ r28 ≥ **0,7596**; "≤ 0" ⇔ ≤ 0,7396 | r28 [0,915; 0,935] (Wu 92,91; baseline 0,9228) | T3'ün gücü sınırda (§5): eşik değişmez, bootstrap GA eşlik eder |
| **H03** r64 / r128 | r64 ≈ [0,775; 0,825] (Wu 79,55/80,45, Doerrich DINO 79,88), r128 ≈ [0,795; 0,845] (Wu 81,88/82,61, Doerrich 81,65) | r64 +3,5 … +8,5; r128 +5,5 … +10,5 | yukarıdaki gibi | — | Wu 94,41 / 95,34 | T1 ve T2 göreli olduğu için B'den bağımsız |
| **Tur-2** dengesizlik (H01 ebeveynli) | ACC: ebeveyn ± 1 puan (Zhang & Li T1a/T1d; focal −0,4/−0,5, WS −1,5/+0,3) | — | ACC'de > +2 puan | — | — | **Birincil etki dengeli ACC / makro-ACC: +2,3 … +4,2** (Zhang & Li). Baseline teşhisi: τ=1 → +6,3 dengeli / −15,7 ACC. `tahmin_nv_payi` zorunlu ikincil |
| **Tur-3** augment / TTA | TTA: 0 … +2 (Medeiros 2026 28 px +1,6); augment: kaynak yok | — | TTA > +3 puan | — | — | Ebeveyn H01 ise R18 tabanlı Medeiros sayısı yalnız yön içindir. ECE kazancı referansı açıkça yazılmalı |

**Madde 4** (`yon_vs`/tavan aynı kuyrukta baseline ister): Önerilen mutlak aralık biçimi **[B + 0,02; tavan] = [0,7596; 0,88]** (H01), **[0,7596; 0,90]** (H02). Beklenti dosyasına sabit sayı olarak yazılabilir (B bu dosyadan; baseline kuyruğu yeniden koşulmaz).

**Beklenti dosyası için ek ön-beyan önerileri (K4 ilk kart onayından ÖNCE; kartlar TASLAK):**
1. Sızıntı ölçüsü = tabakalı S − N (§4). Ham ve sınıf-düzeltmeli ölçüler yalnız bilgi olarak.
2. `val_acc_molemax_disi`, `tahmin_nv_payi`, mel/bkl OvR AUC ön-beyanlı ikincil.
3. H01 ECE kıyas değeri 0,0365 (Medeiros 0,196 değil).
4. H02 kararında seçim kuralı eşitliği (`val_acc_aucsecim`).
5. Tur-2+ kıyasları için eşleştirilmiş görüntü bootstrap'ı (5 seed'in gücü < 3 puan etkiye yetmiyor).

---

## Tahlil defteri (S/D/I)

| grup | bulgu | durum |
|---|---|---|
| S | Etiket/indeks/lezyon hizası, olasılık biçimi | geçti (ihlal 0) |
| D | Ham S − N ezber beklentisinin tersi (−18,8) | **açıklandı**: kaynak/dx_type karıştırıcısı (Simpson). Tabakalı ölçü beklenen yönde (+2,8 / +5,1) |
| D | Son epoch ECE 0,210 ≈ Medeiros 0,196 | uyumlu |
| D | Baseline val ACC literatür bandında | uyumlu |
| I | `degerlendir.py` sızıntı ölçüleri karıştırıcılı | düzeltme önerildi (§4) |
| I | ACC ↔ nv payı r = 0,88: ACC kazancı nv yanlılığından gelebilir | ikincil ölçü önerildi (§2) |
| I | H02 val-ACC seçimi ile baseline val-AUC seçimi asimetrisi (+1,97 puan iyimserlik) | önerildi (§5) |
| I | H01 ECE yönünün yanlış kıyas değerine dayanması | önerildi (§3) |

**Sınırlılıklar:** 5 seed aynı val görüntülerini paylaşıyor (seed'ler görüntü örneklemine göre bağımsız değil). df (12) ve vasc (14) duyarlılıkları örnek başına 7–8 puanlık adımlarla değişiyor ve yorumlanamaz. Kaynak kısayolu bir hipotez, ölçülmedi. Önsel düzeltme ve topluluk satırları teşhis amaçlı, kart sonucu değil. C-val alt küme ACC'si C üzerinde eğitilmiş bir modelin sonucu değil.

---

## 9. K5 — kazanan H10 için "önce/sonra" kırılımı (YALNIZ VALIDATION; D5, karar.md §8)

*deney-yürütücü · 2026-10-12 · CPU · test tahminleri açılmadı (betik `tahmin_test*`/`etiket_test` okumaz; `gecmis.json` test_* anahtarları atılır). Yeni koşum yok: K4'te (Colab L4) üretilmiş val tahminleri yeniden puanlandı.*

| Girdi | Değer |
|---|---|
| Kollar (hepsi Colab L4, `runs_colab/`, `.done` rc=0, seed 0–4; K5 kuru-val kollarıyla aynı koşumlar) | **baseline_colab** `20261011-142338…151359` · **H02c** `130855…133837` · **H08c** `134604…141608` · **H10 = W** `20261011-160211`, `-161411`, `-162611`, `-163811`, `-165010` |
| Komut (WS kökü) | `PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I scripts/hata_kirilim.py --kosumlar <5 dizin> --etiket h10_dinov2b_k4_ft224 --cikti-csv sonuclar/hata-kirilim-h10_dinov2b_k4_ft224.csv --cikti-json sonuclar/hata_kirilim_ozet_h10_dinov2b_k4_ft224.json` (aynısı `baseline_colab`, `h02_r18in_ft224_colab`, `h08_r18in_ft224_d4_colab` için). Yan yana tablo: `scripts/hata_kirilim_karsilastir.py` → `sonuclar/hata-kirilim-h10-karsilastirma.csv`, `sonuclar/hata-kirilim-h10-zor-gecis.json` |
| Bütünlük | 20/20 koşumda `etiket_val.npy` = npz `val_labels`. Genel ACC'ler K5 kuru-val ile aynı (baseline_c 0,7458 · H02c 0,8656 · H08c 0,8865 · H10 0,8947; `k5_kuru_val_sonuc.json`). **S-ihlali yok.** |
| Seçim kuralı notu | CSV'deki "secilen epoch (val-AUC)" notu betiğin sabit metnidir. baseline_c val-AUC ile, H02c/H08c/H10 val-ACC ile seçer (seed `val_acc_epoch_maks` = genel ACC). Kural farkının etkisi test'te ölçüldü: −0,31 [−1,28; +0,73] (karar.md §1.3) |
| Kıyas tabanı | Önce = **baseline_colab** (K5'teki baseline_c, aynı L4 ortamı). §0–§8'deki K3.5 sayıları yerel GTX 1650 baseline'ına aittir (ACC 0,7396). Fark +0,62 puan, tekrar kapısı içinde (orkestrasyon_durumu.md) |

"ort ± std" = 5 seed (ddof = 1). "(n)" = iki ortalamanın farkı, GA'sız nokta değer. Val n = 1.003, çoğunluk tabanı 0,669.

### 9.1 Özet (val)

| metrik | baseline_c | H02c | H08c | **H10 (W)** | H10 − baseline_c (n) |
|---|---|---|---|---|---|
| ACC | 0,7458 ± 0,0074 | 0,8656 ± 0,0019 | 0,8865 ± 0,0033 | **0,8947 ± 0,0066** | +14,89 |
| AUC (makro OvR) | 0,9223 ± 0,0023 | 0,9697 ± 0,0042 | 0,9782 ± 0,0024 | **0,9814 ± 0,0013** | +5,91 |
| dengeli ACC | 0,4333 ± 0,0352 | 0,7100 ± 0,0203 | 0,7763 ± 0,0121 | **0,7977 ± 0,0133** | +36,44 |
| makro-F1 | 0,4444 ± 0,0217 | 0,7463 ± 0,0136 | 0,7946 ± 0,0097 | **0,8229 ± 0,0154** | +37,85 |
| ECE-15 | 0,0399 ± 0,0180 | 0,0762 ± 0,0127 | 0,0654 ± 0,0047 | 0,0601 ± 0,0062 | +2,02 (kötüleşme) |
| NLL | 0,689 | 0,519 | 0,421 | **0,417** | −0,272 |
| ort. güven | 0,761 | 0,937 | 0,947 | 0,952 | +19,2 puan |
| tahmin nv payı | 0,717 ± 0,026 | 0,702 | 0,686 | 0,690 ± 0,008 | −2,8 puan |
| seed çiftleri argmax anlaşmazlığı | %16,6 | %9,2 | %7,4 | **%6,4** | −10,2 puan |
| toplam hata (Σ sınıf hata sayısı, 5 seed ort.) | 255,0 | 134,8 | 113,8 | **105,6** | −149,4 |

### 9.2 Sınıf bazlı: hata kütlesi nereden silindi?

| sınıf (destek) | duyarlılık baseline_c → H10 | OvR AUC baseline_c → H10 | hata sayısı baseline_c → H10 | H02c / H08c duyarlılık |
|---|---|---|---|---|
| akiec (33) | 0,339 → **0,721** | 0,938 → 0,984 | 21,8 → 9,2 | 0,606 / 0,648 |
| bcc (52) | 0,496 → 0,873 | 0,941 → 0,997 | 26,2 → 6,6 | 0,896 / **0,931** |
| bkl (110) | 0,476 → 0,749 | 0,881 → 0,976 | 57,6 → 27,6 | 0,705 / **0,756** |
| df (12) | **0,000** → 0,650 | 0,953 → 0,983 | 12,0 → 4,2 | 0,567 / 0,650 |
| mel (111) | 0,400 → **0,714** | **0,871** → 0,953 | 66,6 → 31,8 | 0,613 / 0,678 |
| nv (671) | 0,907 → 0,963 | 0,908 → 0,978 | 62,6 → 25,0 | 0,954 / 0,957 |
| vasc (14) | 0,414 → **0,914** | 0,965 → 0,999 | 8,2 → 1,2 | 0,629 / 0,814 |

Karışıklık hücreleri (sayım, 5 seed ort.):

| hücre | baseline_c | H02c | H08c | H10 |
|---|---|---|---|---|
| mel → nv | 53,4 | 37,8 | 30,2 | **24,6** |
| nv → mel | 27,0 | 14,6 | 14,6 | 13,2 |
| bkl → nv | 36,6 | 17,0 | **9,4** | 14,0 |
| nv → bkl | 24,0 | 9,4 | 9,2 | 8,6 |
| mel → bkl / bkl → mel | 10,8 / 10,0 | 3,0 / 9,6 | 3,4 / 10,2 | 5,0 / 6,6 |
| df → bcc | 6,4 | 0,6 | 1,2 | 1,2 |

Ölçülen gözlemler:
- **Z1 (nv–mel–bkl üçgeni) kısmen çözüldü, ama hâlâ baskın.**
  - Üçgen hataları 161,8 → 72,0 (−%55).
  - Yine de H10'un kalan hatasının **%68,2'si** bu üçgende (baseline_c %63,5).
  - mel ve bkl'nin sıralama bilgisi büyük ölçüde düzeldi (OvR AUC 0,871 / 0,881 → 0,953 / 0,976).
  - Kalan en büyük hücre **mel → nv 24,6**.
- **Z2 (nadir sınıflar) en büyük göreli kazancı aldı.**
  - df/vasc/akiec makro duyarlılık **0,251 → 0,762**; havuz hata oranı %71,2 → %24,8.
  - baseline_c df'yi hiç doğru tahmin etmiyor (0/12); H10'da duyarlılık 0,650.
  - vasc'ta H10 (0,914) H08c'yi (0,814) de geçiyor.
  - Önsel düzeltme teşhisi (τ = 1, ayarsız): H10'da dengeli ACC yalnız +2,8 puan artıyor (0,7977 → 0,8259; ACC −1,6). baseline_c'de artış +8,8 (ACC −14,0). Yani H10'da karar sınırı sorunu büyük ölçüde kalkmış.
- **H08c ile ayrışma sınıf düzeyinde:**
  - H08c önde: bcc, bkl ve bkl → nv hücresi (9,4 ↔ 14,0).
  - H10 önde: mel, akiec, vasc ve mel → nv hücresi.
  - İki kolun ACC farkı val'de +0,82 [−0,76; +2,43], ayırt edilemedi (tahlil-tur4 §5).

### 9.3 Lezyon tekrarı (S 348 / N 655) — K5'in birincil sorusunun val kırılımı

| ölçü | baseline_c | H02c | H08c | **H10** |
|---|---|---|---|---|
| ACC_S | 0,619 ± 0,013 | 0,816 ± 0,006 | 0,844 ± 0,009 | **0,878 ± 0,017** |
| ACC_N | 0,813 ± 0,009 | 0,892 ± 0,003 | **0,909 ± 0,002** | 0,904 ± 0,001 |
| ham S − N (puan) | −19,4 | −7,7 | −6,5 | **−2,5** |
| **tabakalı S − N** (sınıf × dx_type × kaynak, N-ağırlıklı; puan) | +0,9 ± 1,4 | +6,2 ± 1,6 | +5,6 ± 1,3 | **+10,1 ± 2,0** (perm. p_tek 0,0002) |
| dengeli ACC S / N | 0,417 / 0,450 | 0,762 / 0,660 | 0,811 / **0,743** | **0,881** / 0,726 |
| hep-yanlış oranı S / N | %21,3 / %10,2 | %8,6 / %4,9 | %8,3 / %4,3 | **%5,2 / %6,1** |
| C-val alt kümesi (573, lezyon-ayrık; resmî train modeli) ACC | 0,839 ± 0,007 | 0,913 ± 0,004 | **0,930 ± 0,004** | 0,923 ± 0,003 |
| C-train'e taşınan val (430) ACC | 0,622 | 0,803 | 0,828 | **0,857** |

- H10'un baseline_c'ye kazancı S'de **+25,9**, N'de **+9,0** (n). H02c'ye kazancı S'de **+6,3**, N'de **+1,1** (n) (tahlil-tur4 §4: Δ_N +1,13 [−0,71; +3,16]).
- **H10, hep-yanlış oranının S'de N'den düşük olduğu tek kol.**
  - Diğer bütün kollarda S görüntüleri daha zor. Karıştırıcı: histo ve çok görüntülü lezyonlar (§4).
  - H10'da bu ilişki tersine dönüyor ve tabakalı S − N +10,1'e çıkıyor.
  - Bu, lezyonu eğitimde görülmüş görüntülerin tanınmasıyla tutarlı bir imza. Mekanizma (ezber ya da lezyona özgü öznitelik) ayrıştırılmadı.
- Lezyon-ayrık alt kümede (C-val 573) sıralama H08c > H10 > H02c. N'de ve C-val'de H10'un R18 + D4'e üstünlüğü yok.
- K5 test bu val imzasını doğruladı (karar.md §5): T-SN1 ham Δ_S − Δ_N +4,83 [+2,22; +7,18]; C0-1'de ve C1'de H10 − H02 ayırt edilemedi. Bu bölüm o sonucu val'de **önceden** gösteren kırılımdır; test tahminleri burada kullanılmadı.

### 9.4 Kaynak / tanı türü (§6b'nin önce/sonrası)

| alt grup (n) | baseline_c | H02c | H08c | **H10** |
|---|---|---|---|---|
| molemax (384) | 0,947 | 0,975 | 0,981 | **0,982** |
| **molemax dışı (619)** | 0,621 ± 0,011 | 0,798 | 0,828 | **0,841 ± 0,011** |
| molemax dışı dengeli | 0,430 | 0,717 | 0,770 | **0,805** |
| dx_type = histo (541) | 0,601 | 0,777 | 0,808 | **0,819** |
| kaynak = rosendahl (228) | 0,536 | 0,726 | **0,779** | 0,744 |

- Kazanç §6b'nin öngördüğü yerde, molemax dışında (+22,0 (n)). molemax zaten tavanda (0,947 → 0,982).
- **rosendahl** en zor kaynak olmaya devam ediyor; burada H08c, H10'dan 3,5 puan önde (n). D4'ün kaynak/çekim değişkenliğine karşı belirgin yarar sağladığı tek alt grup bu (keşifsel; GA hesaplanmadı).

### 9.5 Kalibrasyon ve kararlılık

- **Ödünleşim (kalibrasyon):**
  - H10 ECE 0,060 > baseline_c 0,040. Ortalama güven 0,761 → 0,952; aşırı güven (güven − ACC) +5,8 puan.
  - NLL ve Brier ise belirgin iyileşiyor: 0,689 → 0,417 ve 0,350 → 0,172.
  - Test'te sıcaklık ölçekleme ECE'yi 0,0621 → 0,0179'a indiriyor (karar.md §7.1 #12; katman satırı, karar vermez).
- **Seçim oynaklığı (Z3) kalktı:**
  - Seçilen epoch ± 2 içinde val ACC aralığı 4,6 → **0,4** puan.
  - Epoch 5–20 penceresinde epoch'tan epoch'a std 2,1 → 0,6 puan.
  - Seed çifti anlaşmazlığı %16,6 → %6,4.
  - 5-seed olasılık ortalaması (teşhis) H10'a yalnız +0,4 puan ekliyor (0,8983); baseline_c'ye +2,7. Bu, H13'ün (topluluk) neden ÖLDÜR aldığıyla tutarlı (`basarisiz-onkayitlar.md` §A).

### 9.6 Zor örnekler (5 seed'in hepsinde yanlış) — önce/sonra

| | baseline_c | H02c | H08c | **H10** |
|---|---|---|---|---|
| hep yanlış | 141 | 62 | 57 | **58** |
| hep doğru | 612 | 787 | 815 | **840** |

- baseline_c'nin 141 kalıcı hatasından **113'ünü H10 düzeltiyor**; 28'i ortak kalıyor. H10'da **30 yeni** kalıcı hata var (`hata-kirilim-h10-zor-gecis.json`).
- H10'un 58 kalıcı hatasının 22'si H02c ile, 20'si H08c ile ortak; **14'ü üç kolda da ortak**. Bu 14 görüntü, üç farklı tarifin hiçbirinin çözemediği çekirdektir. İndeksler `hata_kirilim_ozet_*.json → zor.hep_yanlis_indeks` alanında.
- H10'un kalıcı hataları: mel 20 (16'sı → nv), bkl 15 (7'si → nv), nv 10, akiec 7, df 4. 58'in 40'ı N'de, 18'i S'de.

### 9.7 Tahlil defteri (K5 eki)

| grup | bulgu | durum |
|---|---|---|
| S | 20/20 etiket hizası; genel ACC'ler K5 kuru-val ile aynı | geçti |
| D | H10 tabakalı S − N +10,1 (baseline_c +0,9); hep-yanlış oranı S < N yalnız H10'da | K5 test'te doğrulandı (T-SN1); "lezyon tekrarına bağlı kazanç" makalede A2 vurgusunun dayanağı |
| D | rosendahl'da H08c > H10 (+3,5 (n)); C-val'de H08c > H10 | keşifsel; GA yok |
| I | H10 ECE 0,060 (aşırı güven +5,8) | ödünleşim; R-T katmanı test'te düzeltiyor |

**Sınırlılıklar:**
- Bütün sayılar val'den (n = 1.003). Seed'ler aynı val görüntülerini paylaşır.
- "(n)" farkları GA'sızdır.
- df (12) ve vasc (14) duyarlılıkları görüntü başına 7–8 puanlık adımlarla değişir.
- Test kırılımı yapılmadı (KULLANICI KARARI 12: test'e yeniden bakılmaz).
