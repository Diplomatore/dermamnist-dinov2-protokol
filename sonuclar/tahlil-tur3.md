# Tur-3 Tahlili (K4c) — 3.1 DermaMNIST · H07 / H08 / H09 · YALNIZ VALIDATION

*Deney-tahlilci, Mod 2 + İNCELE teşhisi · 2026-10-11 · GPU kullanılmadı. Test tahmini ya da etiketi açılmadı: H07/H08/H09 koşumlarının `metrics.json` dosyalarında `"test` anahtarı 0 (grep).*

| Girdi | Değer |
|---|---|
| Koşumlar (rc=0, seed 0–4, ebeveyn aynı seed H02) | H02 `20261010-222930/225428/231909/234351/20261011-000829` · **H07** `20261011-042306…042456` (+ x/a/k/rt kolları) · **H08** `20261011-045930/052423/054917/061415/063910_h08_r18in_ft224_d4_s0…s4` (+ `h08k` 091137…091217, `rt_h08` 091226…091247) · **H09** `20261011-070422/072946/075502/082014/084527_h09_r18in_ft224_renk02_s0…s4` (+ `h09k`, `rt_h09`). Duman ve ön-uçuş koşumları hiçbir karar hesabına girmedi. |
| Ölçüm kaynakları | `sonuclar/tur3_h0{7,8,9}_olcum.json` + `_bootstrap.csv` (deney-yürütücü, `scripts/tur3_olcum.py`), `tur3_h08h09_ek_olcum.json`, `hata-kirilim-h08_*/h09_*`, kapılar `beklenti-tur3-*.md` (hepsi GEÇTİ), `leaderboard*.md`. |
| Soruşturma (bu tahlil) | `scripts/inceleme_h08.py` → `sonuclar/inceleme_h08.json`, ayrıntılı rapor `sonuclar/inceleme-h08.md`. Aynı W: dengeli GA [+4,97; +11,79] birebir yeniden üretildi. |
| Notasyon | "puan" = yüzde puanı. ort ± std = 5 seed. GA = eşleştirilmiş görüntü bootstrap'ı, %95 (2.000, RNG 0). p_t = kesin tek yönlü seed-eşleştirilmiş t (scipy 1.17.1). P* = bootstrap olasılığı. |

---

## 1. Ölçüm okuması (val, n = 5, H02'ye karşı seed eşleşmeli)

### 1a. Karar istatistikleri

| Kart | val ACC (ort ± std) | ΔACC [GA] | p_t (B>A) | sd_Δ seed | Δ dengeli [GA] | Karar değişen görüntü/seed |
|---|---|---|---|---|---|---|
| H02 (ebeveyn) | 0,8672 ± 0,0028 | — | — | — | (0,7062) | — |
| **H07** TTA-8 | 0,8628 ± 0,0019 | **−0,44 [−1,48; +0,60]** | 0,973 | 0,36 | −2,79 [−6,73; +0,51] | 73,2 |
| **H08** D4 eğitim | **0,8893 ± 0,0042** | **+2,21 [+1,18; +3,27]** | **2,9·10⁻⁵** | 0,28 | **+8,06 [+4,97; +11,79]** | 99,4 |
| **H09** renk eğitim | 0,8704 ± 0,0037 | +0,32 [−0,58; +1,26] | 0,033 | 0,29 | +0,88 [−0,98; +3,05] | 88,2 |

### 1b. İkincil metrikler (karar vermez)

| | H02 | H07 | H08 | H09 |
|---|---|---|---|---|
| molemax-dışı ACC (Z4 hedefi) | 0,800 | 0,792 (−0,81 [−2,49; +0,84]) | **0,833 (+3,23 [+1,62; +4,90])** | 0,806 (+0,55 [−0,89; +2,07]) |
| AUC (mel) | 0,9697 (0,938) | 0,9766 (0,943) | **0,9785 (0,952)** | 0,9697 (0,938) |
| ECE / NLL | 0,0745 / 0,518 | **0,0401** / 0,404 | 0,0618 / 0,416 | 0,0777 / 0,519 |
| nv payı | 0,701 | 0,711 (+0,98 [+0,16; +1,85]) | 0,681 (−2,05 [−3,03; −1,08]) | 0,699 |
| mel duyarlılığı Δ | — | **−7,39 [−12,46; −2,38]** | +7,39 [+3,31; +11,68] | +3,24 [−0,88; +7,65] |
| Tabakalı S − N | +6,65 | +2,42 (Δ −4,23 [−8,91; +0,59]) | +5,90 (Δ −0,75 [−4,40; +2,78]) | +6,19 (Δ −0,46) |
| Seed argmax anlaşmazlığı | %9,0 | %6,4 | %7,95 | %9,2 |
| Train kaybı ep 5 (s0) | 0,012 | — | **0,257 (≈ 21×)** | 0,031 (≈ 2,5×) |
| **2. satır: H04 katmanı OOF dengeli / ACC** | 0,7425 / 0,8576 | 0,7534 / 0,8524 | **0,7976 / 0,8800** | 0,7402 / 0,8604 |
| **3. satır: R-T OOF ECE** | 0,0310 | 0,0244 | **0,0212** | 0,0259 |
| GPU | — | ≈ 0,05 sa | 2,24 sa | 2,27 sa |

---

## 2. Kırmızı bayrak taraması

| Kart | Bayrak | Değerlendirme |
|---|---|---|
| H07 | Görülmemiş eşikleri (Δ > 3, ACC > 0,90, AUC > 0,99, ECE < 0,010) | aşılmadı |
| H07 | Ebeveynle bit bit aynı mı? | Hayır: kimlik görünümü bit bit aynı (doğru), görünümler arası max\|Δp\| 0,98–1,00 (devrede) |
| H07 | Beklentinin tersi mi? | Δ −0,44, ön-kayıtlı [−0,5; +1,5] aralığının alt ucunda, yani kartın öngördüğü negatif risk. Mekanizma ölçüldü: dönüşmüş görünümler −2,63 puan dağılım dışı ve D2 − 90° = +0,94 [+0,08; +1,77]. **İNCELE gerekmez.** |
| H07 | Ödünleşim | **mel duyarlılığı −7,39 (GA 0'ın altında)**, nv payı +0,98. TTA olasılık kütlesini çoğunluk sınıfına kaydırıyor. Not olarak raporlanır, karar etkilenmez (kart zaten ÖLDÜR). |
| H07 | H04 katmanı kısıt açığı (OOF ACC 0,8524 < 0,8528, fark 0,04 puan) | **Bayrak değil.** Kısıt seçim katlarında uygulanıyor, OOF'ta garanti edilmiyor (H04'te de aynı yapı vardı, pay 0,04). Katman karar vermiyor. |
| **H08** | **Dengeli Δ +8,06 > kart eşiği +6,5** | **İNCELE (M6). Soruşturma §3.3 ve `inceleme-h08.md`'de: arıza yok.** |
| H08 | Ön-kayıtlı beklenti aralığının üstü (ΔACC +2,21 > +1,5; AUC Δ +0,88 > +0,8; molemax-dışı +3,23 > +2,4) | Görülmemiş eşiklerin altında kalıyor. D grubu "beklenenden iyi" sonucu manipülasyon denetimiyle açıklanıyor (train kaybı 21×). Bayrak değil, not. |
| H08 | nv payı \|Δ\| 0,0205 > 0,02 | Sınırda bir ikincil. Yön nv'den uzaklaşma, yani Tur-1 nv kuralının tersi. Not. |
| **H09** | **(b') t ile bootstrap çelişkisi** (p_t 0,033 anlamlı, GA [−0,58; +1,26] 0'ı kapsıyor) | **Açıklandı, kararı etkilemiyor (§3.2).** |
| H09 | ECE 0,0777 > 0,075; mel duyarlılığı +3,24 > +3; ep 5 kaybı 0,0305 > 0,030 | Ön-kayıtlı aralıkların hemen üstünde kalan ikinciller. Not. |

**Kırmızı bayrak sayısı: 2** (H08 görülmemiş eşiği → soruşturuldu, arıza yok; H09 (b') çelişkisi → açıklandı, karar etkisiz). H07: 0.

---

## 3. Kart kararları

### 3.1 H07 — TTA (8 dihedral görünüm, H02) → **ÖLDÜR (kural b)**

- **Kural (b)** iki koldan da tetikleniyor: 5-seed ort. ΔACC −0,44 < +0,5 (mutlak 0,8628 < 0,8722) **ve** bağlayıcı bootstrap GA alt −1,48 ≤ 0. (b') uygulanmıyor, çünkü (b) geçmedi; t de aynı yönde (p(B>A) 0,973). (c) özdeşlik: kimlik max\|Δp\| = 0, ACC farkı 0. (d) devrede. Görülmemiş eşiği aşılmadı.
- **Başarısız denemeler satırı (D4, mekanizmayla):** "Augmentasyonsuz eğitilmiş R18'de 8 görünümlü TTA ACC'yi artırmadı: Δ −0,44 [−1,48; +0,60]. Ön-kayıt yön etiketi 'zarar verdi' (Δ < 0), ama GA 0'ı kapsadığı için anlamlı bir zarar iddiası kurulmaz. Dönüşmüş görünümler kimlikten 2,63 puan kötü (dağılım dışı). Eksen değiştiren 90° görünümleri D2'den 0,94 [+0,08; +1,77] puan daha zararlı; bu, anizotropik ölçekleme hipoteziyle uyumlu. TTA kalibrasyonu düzeltiyor (ECE 0,0745 → 0,040, NLL 0,52 → 0,40) ve AUC'yi +0,7 artırıyor. Buna karşılık mel duyarlılığını −7,4 [−12,5; −2,4] düşürüyor. Aynı TTA rakibe de kazandırmıyor (−0,40)." Kartın "saptanamadı (MDE ≈ 0,8)" kalıbı burada tam oturmuyor: GA üst sınırı +0,60, yani +0,6 puandan büyük bir ACC kazancı %95 güvenle dışlanıyor. Bu yüzden doğru ifade "saptanamadı" değil, "kazandırmadı".
- **H04 katmanı notu:** OOF ACC'nin kısıtın 0,04 puan altında kalması katmanın tasarımından geliyor (§2). Rapor satırında "kısıt OOF'ta 0,04 puan ihlal" diye açıkça yazılır.
- **Sonuçları:** TTA, Tur-4+ için değerlendirme katmanı **olmaz** (SIRALAMA Tur-4 ebeveyn kuralı). Durma sayacının ACC ölçütü ham satırdan okunur. H07 × H08 ve H07 × H09 birleşimleri açılmaz (M2: H07 TUT almadı). K5'te TTA test'e götürülmez.

### 3.2 H09 — renk oynaması (B/C/S U[0,8; 1,2]) → **ÖLDÜR (kural b; (b') çelişkisi karar etkisiz)**

**Kartın metni:** "(b) ÖLDÜR: 5-seed ort. ΔACC < +0,5 puan (mutlak < 0,8722) **VEYA** kesin tek yönlü t p ≥ 0,05. (b') İNCELE: t ile ΔACC'nin bootstrap'ı çelişirse … **TUT verilmez.**" SIRALAMA'daki ortak ön-kayıt da aynı şeyi söylüyor: "Çelişki → İNCELE, TUT verilmez."

- (b)'nin **ilk koşulu kendi başına sağlanıyor**: ΔACC +0,32 < +0,5, mutlak 0,8704 < 0,8722. Kural "VEYA" ile bağlı olduğu için ÖLDÜR tetikleniyor.
- (b') TUT'u engelleyen bir kural, ÖLDÜR'ü geri alan bir kural değil. Çelişki hangi yönde çözülürse çözülsün ACC pratik marjın altında kalıyor, dolayısıyla İNCELE açmanın karar değeri yok. Hiçbir soruşturma sonucu TUT üretemez.
- **Çelişki hesap hatası değil (S denetimi):** ACC GA'ları referans betikle 3/3 aynı. İki test farklı belirsizlikleri ölçüyor. t yalnız seed varyansına bakıyor ve bu varyans çok küçük (sd_Δ 0,29), çünkü batch sırası ve başlatma bit bit aynı, kollar güçlü eşleşik. Bootstrap ise 1.003 görüntünün örneklemine bakıyor (yarı genişlik ≈ ±0,9). Δ +0,32, t'nin MDE'sinin (≈ 0,40) hemen altında, bootstrap MDE'sinin (≈ 1,0) ise çok altında. Böyle bir çelişki **küçük ama seed'ler arasında tutarlı bir etkinin** beklenen imzası (5 seed'in 4'ünde +0,3…+0,7, birinde −0,1).
- **Başarısız denemeler satırı:** "Parlaklık/kontrast/doygunluk ±%20 oynaması ACC'yi pratik marjın (+0,5) altında değiştirdi: +0,32 [−0,58; +1,26], p_t 0,033. Bu, n = 5 ve 1.003 val görüntüsüyle saptanabilecek etkinin altında (bootstrap MDE ≈ 1,0 puan). Kaynak kısayolu imzası ön-kayıtlı yönlerde: molemax-dışı +0,55, molemax −0,05, D_kk +0,60 [−0,89; +2,16]. Ancak ayırt edilemiyor, yani **kısayol hipotezi bu genlikte sınanamadı**. Renk oynaması zayıf bir düzenleyici (train kaybı ≈ 2,5×; D4'te ≈ 21×). Renk tanısal bilgisi kaybolmadı (mel AUC Δ +0,01)." Ablasyon (iii) koşulu ("kaybederse ama imza pozitifse") GA alt > 0 şartını sağlamıyor, dolayısıyla makalede "kısayol var" denmez. Doğru ifade "ölçülemedi"dir.
- **Sonuçları:** H08 + H09 birleşimi açılmaz (M2). Tek bileşen ablasyonları (i) açılmaz. Rakibe renk augmentasyonu hakkı (M3) gündeme gelmez.

### 3.3 H08 — eğitimde örnek başına D4 → **İNCELE → teşhis temiz (arıza yok); M6 gereği bu kartla TUT VERİLMEDİ**

**Kural girdileri (mekanik):** (a) n = 5. (b) **sağlanmıyor**: ΔACC +2,21 ≥ +0,5, p_t 2,9·10⁻⁵. (b') çelişki yok (GA alt +1,18 > 0). (c) ve (d) temiz. (e) seçilen epoch 30 yalnız 1 seed'de. (f) özdeşlik bit bit aynı. (g) devrede (frekans sapması ≤ 0,0016). ACC tarafında **TUT'un tüm koşulları karşılanıyor.** **Engel yalnız M6:** dengeli Δ +8,06 > +6,5.

**Soruşturma özeti** (ayrıntı ve tablolar `sonuclar/inceleme-h08.md`'de):

| Madde | Test | Sonuç |
|---|---|---|
| a. Train/val karışması | Kod: ayrı npz anahtarları, augment yalnız eğitim döngüsünde (satır 213–224), val `hazirla()` augmentasyonsuz, `xtr[j]` kopya, etiket hizası korunuyor, epoch başına 7.007 = n_train. **Piksel taraması** (28 px, 1.003 × 7.007 × 8 dönüşüm): D4 (kimlik dışı) altında RMSE ≤ 5 olan val görüntüsü **0**. Kimlik altında 4 tane var (H02'yle ortak). Aynı lezyon çiftlerinin medyan RMSE'si ≈ 30 (kopya değil, farklı çekim) | **temiz** |
| b. Seçim iyimserliği | **Son epoch** kolunda Δdengeli **+8,59** [+5,44; +12,03], val-AUC seçiminde +7,84, ACC +2,43 / +2,07. H08 ep 21–30 ortalaması 0,776, H02 son epoch 0,695 (+8,2). Seçim iyimserliği iki kolda simetrik (≈ 1 puan) | **temiz**, seçimden bağımsız |
| c. Kazancın kaynağı | Δdengeli'nin **4,35 puanı (%54) df + vasc'tan** geliyor: 26 görüntü, seed başına ≈ 4 ek doğru. df/vasc hariç 5 sınıflı Δ **+5,19 [+2,91; +7,65]**. **En etkili 2 görüntü (1 df, 1 vasc) çıkarılınca Δ = 6,50.** Seed dengeli Δ +7,5 / +7,9 / +13,2 / +1,6 / +10,1; df + vasc katkısı +4,4 / +4,3 / +8,8 / −1,4 / +5,6. Buna karşın kazanç **tüm azınlık sınıflarında aynı yönde** (mel +7,4, bkl +6,2, bcc +5,4; GA'lar > 0; çoğunluk oyu dönüşleri 37'ye 16). S ≈ N (molemax-dışı S +3,07 / N +3,43) | gerçek kazanç; **eşik aşımı 2 nadir görüntüye bağlı** |
| d. Eşik | +6,5'in kaynağı **Z2 yeniden ağırlıklandırma** literatürü (Cui +4,46, Zhang +4,2 × 1,5; H05 kartı) ve τ zarfı. H08 ise A3 (augmentasyon). **H08 noktası (0,8893; 0,7868), H02'nin seçimsiz τ eğrisinin tamamen dışında** (eğrinin en yüksek ACC'si 0,8672, en yüksek dengeli değeri 0,7605). Kazanç önsel kayması değil, sıralama iyileşmesi (AUC 6/7 sınıfta ↑). Δdengeli bootstrap sd 1,72, seed SE 1,90. **P*(Δ ≤ 6,5) = 0,185.** Aileye uygun çıpa adayı: **Tupper & Gagné 2024** (arXiv:2403.09828, hakemsiz; R18 ince ayarı, meme US, 5×2 CV): tekil döndürme ±30° dengeli **+5,4 / +5,1 / +5,3**, çevirme +0,7…+2,4. Sonuçtan **sonra** bulundu (M1) | eşik bu manipülasyon için yanlış hedeflenmiş bir çıpaya dayanıyor. **Gevşetilmedi.** |
| e. Mekanizma | Train kaybı ep 5'te ≈ 21× yüksek (ezber gecikiyor). Nadir sınıflarda 8× farklı görünüm (df 80 → 640). Duyarlılık artışı sınıf büyüklüğüyle ters orantılı (vasc > df > akiec ≈ mel > bkl > bcc > nv). Perez 2018 (yön), Tupper & Gagné 2024 (büyüklük, komşu alan). DermaMNIST'e özgü kaynak yok | makul |

**Sonuç (i):** *Teşhis temiz. Kazanç gerçek; ACC'de seçimden, seed'den ve lezyon ezberinden bağımsız. Dengeli eşiğin aşılması (1,56 puan) nadir sınıf örnekleme oynaklığından (iki görüntü) ve başka bir manipülasyon ailesinden aktarılmış eşik çıpasından geliyor.* Hata bulunmadı, ama M6 açıkça şunu söylüyor: "eşik aşılırsa TUT verilmez". Kartın üst tablosu değiştirilemez (düzeltme = H08b). **Bu nedenle tahlilci H08'e TUT veremez.**

**Seçenekler (karar şef ve kullanıcının; tahlilci karar vermez):**

| | Yol | H08 statüsü | Tur-4 tarif ebeveyni (SIRALAMA kuralı) | Durma sayacı | Maliyet | Artı / eksi |
|---|---|---|---|---|---|---|
| **A (önerilen; ön-kayda en sadık; şef yürütebilir)** | H08 "İNCELE kapandı: arıza yok; TUT verilmedi (M6)" olarak kalır. **K5'te ön-kayıtlı test doğrulama adayı** olur (aşağıdaki ölçütler test açılmadan önce SIRALAMA'ya yazılır) | aday | **H02** (mekanik: TUT alan yok) | **1/2** | 0 GPU (K5'te çıkarım ≈ 3 dk) | + M6 ve ön-kayıt bozulmaz. − Kazananın belirlenmesi test'e ertelenir. Makale bunu beyan etmeli (D1: "H08'in kazanan sayılması, test açılmadan önce yazılmış bir doğrulama kuralına bağlandı"). Tur-4 augmentasyonsuz sürer; bu, omurga değişimi için tek değişken bakımından temizdir |
| **B (yalnız kullanıcı kararı)** | "Beklenti yanlış kaynaklı" düzeltmesi (`sonuc-tahlili.md` §2(b)). Dengeli eşiğin çıpası aileye uygun kaynağa (Tupper & Gagné 2024) taşınır. H08, **soruşturma temiz + kullanıcı kararı** ile TUT alır. Günlükte "M6 istisnası", makalede beyan | TUT (istisna) | **H08** (D4) | **0/2** (0,8893 ≥ 0,8722; H04 katmanı 0,7976 ≥ 0,7475) | 0 GPU | + Karar val'de kalır, test seçimi yok. − Eşik çıpası sonuç görüldükten sonra değişmiş olur; hakem bunu "sonradan uyarlama" diye okuyabilir. Çarpan seçimi (×1,5 → 8,1, ×2 → 10,8) keyfi. Tahlilci bu yolu **önermez ama meşru sayar**, yeter ki açıkça beyan edilsin |
| C (önerilmez) | H08b: aynı tarif, seed {5..9} | yeniden ölçüm | — | — | ≈ 2,2 GPU-sa (H02 s0–4'e karşı eşleşmesiz) / ≈ 4,5 (H02 {5..9} ile eşleşmeli) | Aynı 1.003 val görüntüsü kullanıldığı için df/vasc oynaklığı çözülmez. Eşik aynı kalırsa ≈ %80 olasılıkla yine İNCELE çıkar. Tek yan faydası Tur-5 için ayrık seed |

**A yolu için K5 doğrulama ölçütü taslağı** (test açılmadan önce şefin SIRALAMA/K5 planına yazması gerekir; tahlilci önerisi, karar değil):
1. Test (n = 2.005), seed {0..4}, H08 − H02 eşleştirilmiş. **Doğrulandı ⇔ ΔACC ≥ +0,5 puan ve görüntü bootstrap GA alt > 0.** Aynı test görünümünde iki aday raporlanır; H08 doğrulanırsa makaledeki R18 kazanan satırı H08 olur, doğrulanmazsa H02 olur. Kural bundan önce yazılmış olur.
2. Görülmemiş denetimi test'te de aynı eşiklerle yapılır (ΔACC > 3, ACC > 0,90, AUC > 0,99, dengeli Δ > 6,5 → rapor bayrağı). Dengeli Δ ayrıca 5 sınıflı (df/vasc hariç) olarak raporlanır.
3. **DermaMNIST-C (lezyon-ayrık; kullanıcı kararı 3, birincil protokol)**: H08 − H02 ΔACC GA alt > 0 ise "kazanç lezyon ezberinden değil" desteklenir. Bu, val'in çözemediği sorunun (S − N) kesin testidir.
4. M3: H08 makale kazananı olursa rakibe aynı augmentasyon hakkı verilir (baseline 28 px + D4, ≈ 2,7 GPU-sa). Bu K5 öncesi şef/kullanıcı kararıdır. Adil kıyas satırı yine augmentasyonsuz H02'dir.

**Öğrenilen (H08):** Örnek başına D4 augmentasyonu, aynı 30 epoch'ta ezberi geciktiriyor (train kaybı ≈ 21×) ve augmentasyonsuz ince ayara göre val ACC'yi her seed'de +1,8…+2,5 puan artırıyor. Kazanç seçim kuralından bağımsız. Lezyonu train'de görülmüş görüntülerde yoğunlaşmıyor, molemax-dışı kaynaklarda toplanıyor. Bu bir önsel kayması değil, sıralama iyileşmesi: H02'nin hiçbir logit ayarı bu noktaya erişemiyor ve kazanç H04 ile toplanabiliyor. Dengeli doğruluk kazancının yarısından fazlası 26 df/vasc görüntüsünden geliyor. Bu yüzden val'de "dengeli Δ" tek başına görülmemiş eşiğine karşı güvenilir bir ölçü değil.

---

## 4. Durma kuralı sayacı (konu.md 2026-10-10 20:15; SIRALAMA Tur-3 okuma kuralı)

Kurala göre Tur-3'ün en iyisi **yalnız TUT alan H07/H08/H09 arasından** okunur. İNCELE'deki kart TUT değildir, sayaca girmez. H07 ve H09 ÖLDÜR aldı.

| | Tur-3 öncesi en iyi | Tur-3 en iyi (TUT) | Eşik | ≥ 0,5 puan? |
|---|---|---|---|---|
| val ACC (ham) | 0,8672 (H02) | — (TUT yok) | ≥ 0,8722 | hayır |
| val dengeli (H04 katmanı OOF) | 0,7425 (H04) | — | ≥ 0,7475 | hayır |

**Sayaç (mekanik okuma): 1 / 2 — H08 kararına bağlı, askıda.** A yolunda 1/2 kalır. Tur-4 de iki ölçütte iyileşme getirmezse 2/2 olur ve K5'e geçilir. B yolunda (H08 TUT) sayaç **0/2** olur, çünkü 0,8893 ≥ 0,8722 ve 0,7976 ≥ 0,7475. H08'in ham ACC'si (0,8893) eşiğin çok üstünde olsa da sayılmaz, çünkü kural TUT şartı koyuyor. Bunu şefin gözden kaçırmaması gerekir: **1/2'de bile Tur-4 koşar.** Sayacın farkı yalnız Tur-4 başarısız olursa ortaya çıkar.

---

## 5. Tur-4 önerisi (kart DEĞİL; sıradaki boş id **H10**)

**Tarif ebeveyni:** A yolunda `h02_r18in_ft224` (augmentasyonsuz), B yolunda `h08_r18in_ft224_d4`. H04 katmanı ve R-T ikinci ve üçüncü satırda sürer. TTA katmanı yok (H07 öldü).

**Önce ölçüm (SIRALAMA Tur-4 notu; karar değil):** Kart yazılmadan (a) DINOv2-B son k blok kısmi ince ayar (önbellekli) ve (b) DINOv2-S/14 tam ince ayar için 1 epoch'luk yerel GPU dumanı yapılır. s/epoch, tepe bellek ve önbellek yazma süresi ölçülür (≈ 15 dk GPU). **Bu ölçüm koşumunu şef koşar.** Önerilen komut biçimi (KOMUTLAR'daki yerel ölçüm kalıbıyla):
`env/.venv/Scripts/python.exe varyantlar/<dinov2_ft>/ince_ayar.py --seed 0 --epoch 1 --out runs_yerel_duman/<ad>/metrics.json` (betik henüz yok; deney-yürütücü yazar).

| Öncelik | Öneri | Aile / Z | Değişen tek şey | Beklenti adayı (kaynak) | Görülmemiş | Maliyet / tetik |
|---|---|---|---|---|---|---|
| **1** | **H10 — DINOv2-B son k blok kısmi ince ayar, r224** | A4 / Z1 (mel→nv 28–36, mel AUC 0,938–0,952) | Omurga: R18 → DINOv2-B (ilk 12 − k blok donuk, çıkış token'ları önbellekte) | Wu & Papież 2025 (hakemsiz): DINOv2-B uçtan uca 224 **92,02** ↔ R18 uçtan uca 86,05 (Δ ≈ +6, test, seçim belirsiz). Kısmi ince ayar bunun altında kalabilir. Ebeveyne göre +0…+4 aralığında bir aday | val ACC > 0,93 **veya** AUC > 0,99 (`literatur-cipa.md` §4(e)) | 1 epoch ölçümünden sonra. **Önbellek tek görünümlüdür:** B yolunda D4 için 8 × 2,8 ≈ 22 GB disk gerekir ya da augmentasyon katmanı yeniden tasarlanır. Bu, A yolunu Tur-4 için pratik olarak da kolaylaştırıyor |
| 2 | H11 — DINOv2-S/14 tam ince ayar, r224 | A4 / Z1 | Omurga: R18 → DINOv2-S | Kaynak: Wu & Papież (B için). S için doğrudan sayı yok, kart yazımında aranmalı | aynı | 5-seed tahmini > **6 GPU-sa** ya da epoch > 300 s ise **Colab önerisi → KULLANICI ONAYI GEREKİR** (konu.md 09:25: Colab yalnız tıkanınca, eşleştirilmiş, rakiple aynı defterde; ebeveyn H02/H08 aynı donanımda yeniden koşulur ya da `donanim` etiketiyle ayrılır) |
| (3) | O1/O2 sondaları (ViT-L/14, ViT-S/14; ≈ 0,25 / 0,05 GPU-sa) | Pareto satırı | — | — | — | Kart sınırı (≤ 3) içinde, ucuz |

**Neden bu sıra:** Kalan hata kütlesi temsil düzeyinde. H08'den sonra bile mel→nv 28,4, mel duyarlılığı 0,69, mel AUC 0,952. Z2 (nadir sınıf) H04 ile kapatıldı, Z4 (ezber) H08 ile kısmen ele alındı. Tur-4 ACC hedefli bir omurga değişimidir. **H08 ablasyon (i) (yalnız D2 ile eğitim, ≈ 2,2 GPU-sa)** yalnız B yolunda ve H08 kazanan zincirine girerse K5 öncesi ablasyon olarak değerlendirilir. Bunu destekleyen keşifsel sinyal: H07'de D2 − 90° +0,94.

---

## 6. K5 hazırlığı

### 6a. Ablasyon zinciri (val) ve kazanan zincir

| Basamak | Varyant | val ACC | val dengeli | Δ (önceki basamağa) | Durum |
|---|---|---|---|---|---|
| 0 | Rakip: R18, 28 px, sıfırdan | 0,7396 | 0,4621 | — | replikasyon YEŞİL (test 0,7296) |
| 1a | + ImageNet önağırlık + 224 + ince ayar (**H02**) | 0,8672 | 0,7062 | +12,76 / +24,42 | TUT (paket, ayrıştırılmadı) |
| 1b | DINOv2-B sonda r224 (H01) | 0,8223 | 0,6494 | +8,28 / +18,74 | TUT |
| 2a | H02 + logit ayarı (H04, OOF) | 0,8576 | 0,7425 | −0,96 / +3,63 | TUT (ödünleşim) |
| 3a | H02 + TTA-8 (H07) | 0,8628 | 0,6783 | −0,44 / −2,79 | ÖLDÜR → başarısız tablosu |
| **3b** | **H02 + eğitimde D4 (H08)** | **0,8893** | **0,7868** | **+2,21 / +8,06** | **İNCELE-temiz; TUT verilmedi (A) / kullanıcıya bağlı (B)** |
| 3b + 2a | H08 + H04 katmanı (OOF) | 0,8800 | **0,7976** | −0,93 / +1,08 (H08'e göre) | katman |
| 3c | H02 + renk (H09) | 0,8704 | 0,7151 | +0,32 / +0,88 | ÖLDÜR → başarısız tablosu |
| + | R-T | = | = | ECE: H02 0,031, H08 0,021 | rapor satırı |

**Kazanan zincir (bugün):**
- A yolunda: val kazananı **B → H02** (ACC), dengeli için **B → H02 → H04 → R-T**. **H08 K5 doğrulama adayıdır**; doğrulanırsa zincir B → H08 → H04 → R-T olur.
- B yolunda: doğrudan **B → H08 → H04 katmanı → R-T** (ACC 0,8893; dengeli OOF 0,7976; ECE 0,021).

### 6b. K5'te test'e bir kez (ön-kayıtla)
Baseline (donduruldu). H02 × 5 `best_model.pth`. **H08 × 5 `best_model.pth`** (A: doğrulama adayı; B: kazanan). H04 τ\* (H02: 0,7; H08: tam-val τ\* 0,5) aynı olasılıklara. R-T tam-val T. H01 + H06 (τ\* 0,2). DermaMNIST-C birincil ve ikincil protokol: baseline + H02 + H08 (+ H01). H05, H07 ve H09 test'e götürülmez.

### 6c. D5 kanıt paketi — eksikler ve sorumlular

| # | Eksik | Kim / ne zaman |
|---|---|---|
| 1 | `gunluk/deney-gunlugu.md`: Tur-3 satırları (§8), bütçe sayacı (≈ 9,8'de kalmış, güncel ≈ 14,7), H08 arıza/İNCELE satırı, hata_kirilim üzerine yazma olayı | **şef**, Tur-4 öncesi |
| 2 | H04/H06 yalnız-nv ablasyonu (ii) — Tur-2'den beri eksik | deney-yürütücü, CPU, K5 öncesi |
| 3 | H08 ablasyon (i) D2 — yalnız H08 kazanan zincirindeyse | şef/kullanıcı kararı (≈ 2,2 GPU-sa) |
| 4 | M3: rakibe D4 hakkı (baseline 28 px + D4, ≈ 2,7 GPU-sa) — yalnız H08 makale kazananıysa | **K5 öncesi şef/kullanıcı kararı** |
| 5 | Başarısız denemeler tablosu: H05, H07, H09 (+ Tur-1 r28 nv notu) | tahlilci, K5'te (satır metinleri §3.1–3.2'de hazır) |
| 6 | H02 paketinin ayrıştırılmaması (önağırlık / 224 / ince ayar) | sınırlılıklar bölümü |
| 7 | 28 px'te train↔val kimlik altında 4 yakın kopya (RMSE ≤ 5) | sınırlılıklar notu (tüm kolları etkiler) |

---

## 7. Bütçe görüşü

- **Tur-3 GPU:** H07 ≈ 0,05 + H08 2,24 + H09 2,27 ≈ **4,6 GPU-sa** (tahmin 4,9; üst 5,6). Sayaç ≈ 9,8 + 4,6 → **≈ 14,4–14,7 / 32 GPU-sa** (şefin değeri 14,7), **tur 3 / 8**. Kalan ≈ 17,3 GPU-sa, 5 tur.
- **Rezervler:** K5 ≈ 1,5 GPU-sa (test çıkarımları, DermaMNIST-C 224, H01 test öznitelikleri). H08 kazanan zincirine girerse ek **M3 ≈ 2,7** ve **D2 ablasyonu ≈ 2,2**. Bu durumda Tur-4…8 için kalan ≈ 10,9. A yolunda ve doğrulanmazsa ≈ 15,8 kalır.
- **En iyi kullanım:** Tur-4'ün 1 epoch'luk ölçümü (≈ 0,25) kesin önceliktir. Kısmi ince ayar 5 seed'de ≤ 6 GPU-sa tutarsa yerelde koşar. Tutmazsa Colab önerisi kullanıcıya gider. **H08b tekrarına (C) bütçe harcanması önerilmez.**
- **Getiri gözlemi:** Tur-3'te 4,6 GPU-sa'nın 2,24'ü (H08) projenin en büyük ACC kazancını verdi (+2,21). H09'un 2,27'si yalnız mekanizma bilgisi üretti (ön-kayıtta bu bekleniyordu, P(TUT) 0,15). H07 0,05 GPU-sa'e mal oldu.

---

## 8. Günlük için satır önerileri (`gunluk/deney-gunlugu.md`)

**Tur tablosu (Tur-3):**

| Tur | Tarih | Hipotez | Ebeveyn → Varyant | Koşum id'leri | Ana metrik (ort±std, n) | Δ vs ebeveyn | Karar | Öğrenilen |
|---|---|---|---|---|---|---|---|---|
| 3 | 2026-10-11 | H07 | h02 → h07_h02_tta8 | 20261011-042306…042456 (5) | val_acc 0,8628±0,0019 | −0,44 [−1,48; +0,60] | ÖLDÜR (b) | Görünümler −2,6 puan dağılım dışı; 90° görünümleri D2'den daha zararlı (+0,94 [+0,08; +1,77]); ECE 0,0745 → 0,040 ama mel duyarlılığı −7,4 |
| 3 | 2026-10-11 | H08 | h02 → h08_r18in_ft224_d4 | 20261011-045930…063910 (5) | val_acc 0,8893±0,0042; dengeli 0,7868 | ACC +2,21 [+1,18; +3,27] p_t 2,9e-5; dengeli +8,06 [+4,97; +11,79] | **İNCELE → arıza yok; M6 gereği TUT verilmedi** (yol A/B kullanıcıda) | Train kaybı ≈ 21×; kazanç seçimden ve lezyon ezberinden bağımsız; dengeli eşik aşımı df/vasc'taki 2 görüntüye bağlı (5 sınıf +5,19) |
| 3 | 2026-10-11 | H09 | h02 → h09_r18in_ft224_renk02 | 20261011-070422…084527 (5) | val_acc 0,8704±0,0037 | +0,32 [−0,58; +1,26] p_t 0,033 | ÖLDÜR (b; (b') çelişkisi karar etkisiz) | Kaynak kısayolu imzası doğru yönde ama ayırt edilemiyor (D_kk +0,60 [−0,89; +2,16]); zayıf düzenleyici (kayıp 2,5×) |

**Arıza / beklenti ihlali tablosu:**

| Tarih | Belirti | Hipotez | Teşhis yöntemi | Kök neden | Düzeltme | Etkilenen koşumlar |
|---|---|---|---|---|---|---|
| 2026-10-11 ~09:40 | H08 dengeli Δ +8,06 > kart "görülmemiş" eşiği +6,5 (M6 İNCELE) | augment val'e sızması / train-val karışması / D4 ile döndürülmüş kopya sızıntısı / etiket indeksleme / epoch uyumu / seçim iyimserliği / eşik çıpası | Kod okuması; 28 px piksel taraması (1.003 × 7.007 × 8 dönüşüm; D4 RMSE ≤ 5: 0); aynı lezyon çiftleri (medyan RMSE 30); son epoch ve AUC seçimli kollar (+8,59 / +7,84); sınıf ayrışımı (df+vasc 4,35/8,06); en etkili k görüntü (k = 2 → 6,50); H02 τ eğrisi; P*(Δ ≤ 6,5) 0,185 (`scripts/inceleme_h08.py`) | **Arıza yok.** Aşım nadir sınıf örnekleme oynaklığından ve Z2 literatüründen aktarılmış eşik çıpasından geliyor | Kod düzeltmesi yok. Kart TUT almadı (M6). Karar yolu A/B kullanıcıda. Gelecek kartlarda A3 dengeli eşiği aileye uygun çıpayla (Tupper & Gagné 2024) ve df/vasc hariç 5 sınıflı ek satırla yazılsın | Yok (10 koşum geçerli) |
| 2026-10-11 | H09 (b') t ↔ bootstrap çelişkisi | hesap hatası? | ACC GA'ları referans betikle 3/3 aynı; seed sd_Δ 0,29 ↔ bootstrap ±0,9 | Arıza yok: iki test farklı belirsizliği ölçüyor; küçük ama tutarlı etkinin imzası | Yok; (b) ilk koşulu ÖLDÜR'ü bağımsız tetikliyor | Yok |

**Bütçe sayacı satırı:** ≈ 14,7 / 32 GPU-sa (K3 ≈ 2,7 + kayıplar ≈ 1 + tur1 ≈ 3,7 + tur2 ≈ 2,3 + tur3 ≈ 4,6 + tur3 yükleme/duman payı). Tur 3 / 8. Durma sayacı **1/2 (askıda: H08 yolu A ise 1/2, B ise 0/2)**.

---

## 9. Tahlil defteri (S/D/I)

| grup | bulgu | durum |
|---|---|---|
| S | Etiket hizası (H02/H08 10 koşum = npz), npy = metrics = gecmis (5/5), özdeşlik dumanları bit bit aynı, augment sayaçları, D4 öz-sınaması, H07 kimlik görünümü max\|Δp\| = 0, H09 çarpan sınırları, bootstrap GA'larının yeniden üretimi | geçti |
| S | D4 ile döndürülmüş kopya sızıntısı (piksel taraması) | yok |
| D | H08 dengeli Δ > eşik | açıklandı (df/vasc ayrıklığı + yanlış hedefli çıpa; τ eğrisinin dışında → sıralama kazancı) |
| D | H08 beklenenden iyi (ΔACC +2,21 > +1,5) | açıklandı (manipülasyon denetimi: kayıp 21×, ECE ↓, seçilen epoch daha geç) |
| D | H07 mel duyarlılığı −7,4 (beklenti yazılmamıştı) | açıklandı (TTA olasılık kütlesini nv'ye kaydırıyor, nv payı +0,98) |
| D | H08 D4 hizalı aynı-lezyon alt kümesi +10,4 (p 0,018, eşik 3) | eşiğe duyarlı (0/1/5/8 → +4,5/+3,9/+4,6/+4,0) → bulgu sayılmaz (keşifsel) |
| I | "H08 dengeli doğruluğu 8 puan artırır" cümlesi tek başına yazılamaz. Yarısından fazlası 26 df/vasc görüntüsünden geliyor; 5 sınıflı +5,19 ile birlikte verilmeli | makale cümlesi önerisi |
| I | H07 "zarar verdi" etiketi: GA 0'ı kapsıyor → "kazandırmadı (yön negatif)" | başarısız tablosu ifadesi |
| I | H09 "kısayol var" denemez; "bu genlikte ölçülemedi" | başarısız tablosu ifadesi |
| I | A yolu seçilirse H08'in kazanan sayılması test'teki doğrulamaya bağlanır → D1 beyanı | K5 notu |

**Bu tahlilde yeni görülen val bilgisi (M1 beyanı; gelecekteki kartlar için):** H02'nin seçimsiz τ eğrisi (0,05 adım, tam val; Tur-2'de 0,1 adımla görülmüştü). H08'in son epoch ve AUC seçimli kolları. Sınıf ve görüntü düzeyinde katkılar. S/N ve D4 hizalı alt küme kırılımları. 28 px train↔val en yakın komşu RMSE dağılımı. Tupper & Gagné 2024 çıpası H08 sonucundan **sonra** bulundu.

**Sınırlılıklar:** 5 seed aynı 1.003 val görüntüsünü paylaşıyor. df (12) ve vasc (14) dengeli ACC'yi tek görüntüyle 1,2 / 1,0 puan oynatıyor. Piksel taraması 28 px'te yapıldı; 224 px'teki yakın kopyalar 28 px'e göre daha seyrek ya da eşit olmalı, ama doğrulanmadı. Lezyon ezberinin kesin testi val'de değil, K5 DermaMNIST-C'dedir.

---

## 10. Kart başına hazır sonuç paragrafları (kart dosyalarına şef/yürütücü ekler; bu tahlilde kartlara dokunulmadı)

### H07 (Karar + Öğrenilen)
> - Karar: **ÖLDÜR** — kural (b): ΔACC −0,44 < +0,5 (mutlak 0,8628 < 0,8722) **ve** bağlayıcı bootstrap GA alt −1,48 ≤ 0. (b') uygulanmaz (t aynı yönde, p(B>A) 0,973). (c) özdeşlik 0, (d) devrede. Görülmemiş eşiği yok. Kırmızı bayrak yok. Ödünleşim notu: mel duyarlılığı −7,39 [−12,46; −2,38], nv payı +0,98. H04 katmanı OOF ACC'si 0,8524, kendi kısıtının 0,04 puan altında (kısıt seçim katlarında uygulanıyor; karar vermez). Başarısız denemeler tablosuna şöyle girer: "ACC kazandırmadı: Δ −0,44 [−1,48; +0,60]. Yön negatif, anlamlı zarar iddiası yok. Mekanizma: dönüşmüş görünümler dağılım dışı (−2,63) ve 90° görünümleri daha zararlı." TTA katmanı Tur-4+'a taşınmaz. H07 × H08/H09 açılmaz.
> - Öğrenilen: Augmentasyonsuz ince ayarlı R18 yönelime duyarlı (görünüm başına −2,1…−3,3 puan). Görünüm ortalaması iyi bir kalibratör (ECE ÷ 1,9), ama karar sınırını çoğunluk sınıfına itiyor (mel −7,4). Anizotropik ölçekleme hipoteziyle uyumlu olarak eksen değiştiren görünümler daha kötü (D2 − 90° = +0,94 [+0,08; +1,77]). Değişmezlik test zamanında eklenemiyor, eğitimde öğretilmesi gerekiyor (bkz. H08).

### H08 (Karar + Öğrenilen)
> - Karar: **İNCELE → teşhis temiz (arıza yok); M6 gereği bu kartla TUT verilmedi.** ACC tarafında TUT koşullarının hepsi karşılanıyor: (b) ΔACC +2,21 [+1,18; +3,27], p_t 2,9·10⁻⁵; (b') çelişki yok; (c)–(g) temiz. Engel, dengeli Δ +8,06'nın kartın +6,5 eşiğini aşması. Soruşturma (`sonuclar/inceleme-h08.md`, `scripts/inceleme_h08.py`) şunları buldu: train/val karışması ve D4 kopya sızıntısı yok; kazanç seçimden bağımsız (son epoch +8,59); S ≈ N. Aşımın 4,35 puanı df/vasc'tan (26 görüntü) geliyor; en etkili 2 görüntü çıkarılınca Δ 6,50. 5 sınıflı Δ +5,19 [+2,91; +7,65]. P*(Δ ≤ 6,5) 0,185. Eşik Z2 literatüründen aktarılmış ve H08 noktası H02'nin τ eğrisinin dışında. Yol (şef/kullanıcı): **A** K5 doğrulama adayı (önerilen) / **B** kullanıcı kararıyla çıpa düzeltmesi + TUT / C seed {5..9} tekrarı (önerilmez).
> - Öğrenilen: Eğitimde örnek başına D4, aynı epoch bütçesinde ezberi geciktiriyor (kayıp ≈ 21×). Val ACC'yi her seed'de +1,8…+2,5 puan, molemax-dışı ACC'yi +3,2 puan artırıyor. Kalibrasyon (ECE 0,062; R-T ile 0,021) ve AUC (+0,9) iyileşiyor. Kazanç bir önsel kayması değil, sıralama iyileşmesi; H04 ile toplanabiliyor (H08 + H04 OOF dengeli 0,798). Tabakalı S − N (ezber imzası) ölçülebilir biçimde azalmadı. Dengeli ACC kazancının yarısından fazlası en küçük iki sınıftan geliyor, bu yüzden val'de bu ölçü görülmemiş eşiğine karşı oynak.

### H09 (Karar + Öğrenilen)
> - Karar: **ÖLDÜR** — kural (b)'nin ilk koşulu: ΔACC +0,32 < +0,5 (mutlak 0,8704 < 0,8722). (b') t ↔ bootstrap çelişkisi kayıtlı (p_t 0,033, GA [−0,58; +1,26]). (b') yalnız TUT'u engeller ve (b)'nin pratik marj koşulu çelişkiden bağımsız olarak tetiklendiği için İNCELE'nin karar değeri yok. Çelişki hesap hatası değil (GA'lar referans betikle aynı; seed sd_Δ 0,29 ↔ görüntü bootstrap ±0,9). Görülmemiş eşiği yok. (h) notları uygulanmaz. Başarısız denemeler tablosuna "n = 5'te saptanamadı (MDE ≈ 0,9–1,0 puan); ölçülen +0,32" olarak girer.
> - Öğrenilen: ±%20 fotometrik oynama bu ince ayar tarifinde zayıf bir düzenleyici (kayıp ≈ 2,5×, ECE değişmedi). Renk tanısal bilgisini bozmuyor (mel AUC Δ +0,01). Kaynak kısayolu imzası ön-kayıtlı yönde (D_kk +0,60), ama 1.003 val görüntüsüyle ayırt edilemiyor. Kısayol hipotezi bu genlikte sınanamadı, daha güçlü bir sınama (ör. sabit fotometrik kaydırmaya duyarlılık teşhisi) ancak yeni bir kartla açılabilir.

---
*Kaynaklar (bu tahlilde yeni):* Tupper, A. & Gagné, C. (2024). Analyzing Data Augmentation for Medical Images: A Case Study in Ultrasound Images. arXiv:2403.09828 (Tablo 1; erişim 2026-10-11). Balestriero, R., Bottou, L. & LeCun, Y. (2022). The Effects of Regularization and Data Augmentation are Class Dependent. NeurIPS 2022, arXiv:2204.03632. Diğerleri kartlardaki künyelerle (Perez vd. 2018 arXiv:1809.01442; Medeiros 2026 arXiv:2604.09697; Wu & Papież 2025).
