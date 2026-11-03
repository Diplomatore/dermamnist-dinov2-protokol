# Tur-1 Tahlili (K4c) — 3.1 DermaMNIST · H01 / H02 / H03 · YALNIZ VALIDATION

*Deney-tahlilci, Mod 2 · 2026-10-11 · test tahmini/etiketi açılmadı (varyant `metrics.json` ve `gecmis.json`'larında `test` anahtarı 0, varyant koşum dizinlerinde `*test*` dosyası 0, öznitelik önbelleklerinde yalnız `train_*`/`val_*` dizileri — üçü de bu tahlilde ayrıca denetlendi).*

| Girdi | Değer |
|---|---|
| Koşumlar (rc=0, seed 0–4, a3) | baseline `20261010-171037/174311/181653/184922/192128` · H03 r28 `205746/210137/210519/210903/211247` · r64 `211625/213824/214206/214540/214915` · r128 `215253/215633/220012/220350/220730` · H01 `221110/221455/221838/222218/222554` · H02 `222930/225428/231909/234351/20261011-000829`. **Yetim `20261010-212001_h03_dinov2b_lp_r64_s1` (rc=96, metrics yok) hiçbir hesaba girmedi**; r64 s1 = yeniden koşum `213824`. |
| Betik (tekrar üretilebilir, CPU, RNG 0) | `scripts/tahlil_tur1.py` → `sonuclar/tahlil_tur1_bootstrap.{json,csv}`, `sonuclar/tahlil_tur1_ikincil.csv`, `sonuclar/tahlil_tur1_teshis.json`. ACC bootstrap'ı `varyantlar/ortak/eslestirilmis_bootstrap.py` ile **birebir aynı** (12/12 kıyasta GA eşitliği betikte doğrulandı); aynı yeniden örnekleme indeksleri dengeli ACC, molemax-dışı ACC, N/S alt kümeleri ve tabakalı S − N için de kullanıldı. |
| Kırılım (kazananlar) | `scripts/hata_kirilim.py --varyant h02_r18in_ft224` → `sonuclar/hata-kirilim-h02_r18in_ft224.csv`, `hata_kirilim_ozet_h02_r18in_ft224.json`; aynısı H01 (`…h01_dinov2b_lp_r224…`). Baseline dosyaları (`hata-kirilim.csv`, `hata_kirilim_ozet.json`) **üzerine yazılmadı**. |
| Notasyon | "puan" = yüzde puanı; ort ± std = 5 seed (ddof 1); GA = eşleştirilmiş görüntü bootstrap'ı %95 yüzdelik (2.000, RNG 0); p_t = seed-eşleştirilmiş **tek yönlü** t-testi (scipy, kesin); B = baseline iç çıpası. |

---

## 0. Şef özetinin doğrulaması ve bir I-grubu düzeltmesi

| Şef özeti | Doğrulama |
|---|---|
| H01 0,8223 ± 0,0013, Δ +8,3 | ✓ (Δ +8,28) |
| H02 0,8672 ± 0,0028, Δ +12,8; aucsecim 0,8588 | ✓ (Δ +12,76; aucsecim 0,8588 ± 0,0070) |
| H03 r28 0,7739 / r64 0,8098 / r128 0,8195 | ✓ |
| varyant `metrics.json`'da `test_` yok | ✓ (ek olarak `gecmis.json`, dosya adları, önbellek dizileri temiz) |
| **p değerleri (0,0115 / 0,0095 / 0,037)** | **Ön-kayıtlı test değil.** `leaderboard.py` scipy'sız Python ile koşmuş; `eslesmis_test` normal yaklaşımlı **iki yönlü** p'ye düşmüş (df = 4'te kaba). Kartların ön-kayıtlı testi **tek yönlü eşleştirilmiş t**: H01 **p_t = 0,00044**, H02 **9·10⁻⁵**, r28 **0,0115** (iki yönlü kesin: 0,0009 / 0,0002 / 0,023). Hiçbir karar değişmiyor (hepsi < 0,05), ama makaleye yaklaşık p girmemeli. **Düzeltme:** leaderboard'u `env/.venv/Scripts/python.exe` (scipy 1.17.1) ile yeniden üret. Günlük arıza satırı önerisi §9'da. |

---

## 1. Ölçüm okuması (val, n = 5, seed {0..4} eşleşmiş)

### 1a. Ana metrik ve ön-kayıtlı karar istatistikleri

| Kıyas | ΔACC (puan) | GA %95 | p_t (tek y.) | sd(Δ_seed) | Δ dengeli ACC [GA] | Δ molemax-dışı ACC [GA] |
|---|---|---|---|---|---|---|
| **H01 − B** | **+8,28** | [+5,66; +10,83] | 0,00044 | 2,08 | +18,74 [+12,92; +25,15] | +12,05 [+8,11; +15,98] |
| **H02 − B** | **+12,76** | [+10,67; +14,86] | 9·10⁻⁵ | 2,15 | +24,42 [+18,73; +30,34] | +19,10 [+15,97; +22,21] |
| H02(aucsecim) − B [kural b''] | +11,92 | [+9,89; +13,92] | 0,0002 | 2,44 | +22,35 [+16,57; +28,23] | +17,74 [+14,69; +20,71] |
| **r28 − B [T3]** | **+3,43** | [+1,16; +5,74] | 0,0115 | 2,14 | **+0,13 [−5,85; +6,53]** | +4,78 [+1,34; +8,34] |
| r64 − B | +7,02 | [+4,59; +9,37] | 0,0009 | 2,13 | +8,51 [+1,91; +15,45] | +10,15 [+6,39; +13,88] |
| r128 − B | +8,00 | [+5,66; +10,43] | 0,0005 | 2,09 | +10,00 [+3,78; +16,69] | +11,76 [+8,05; +15,40] |
| r64 − r28 [T1] | +3,59 | [+1,28; +6,04] | <10⁻⁵ | 0,19 | +8,38 [+2,18; +15,03] | +5,36 [+1,74; +9,21] |
| r128 − r64 [T1] | +0,98 | **[−0,86; +2,95]** | 0,0001 | 0,18 | +1,50 [−4,09; +6,99] | +1,62 [−1,26; +4,62] |
| r224 − r128 [T1] | +0,28 | **[−1,42; +1,99]** | 0,051 | 0,29 | +8,73 [+3,93; +13,68] (C\* etkisi, §3c) | +0,29 [−2,39; +2,95] |
| **r224 − r28 [T2]** | **+4,85** | [+2,29; +7,56] | <10⁻⁶ | 0,21 | +18,61 [+11,00; +26,54] | +7,27 [+3,12; +11,52] |
| **H02 − H01 [ebeveyn kuralı]** | **+4,49** | [+2,31; +6,62] | <10⁻⁵ | 0,31 | +5,68 [−0,09; +11,04] | +7,04 [+3,52; +10,35] |
| H02(aucsecim) − H01 | +3,65 | [+1,42; +5,68] | 0,0002 | 0,74 | +3,61 [−2,19; +9,02] | +5,69 [+2,16; +8,93] |

Not: sonda–sonda kıyaslarında sd(Δ_seed) ≈ 0,2 puan olduğu için t-testi neredeyse her farkı "anlamlı" bulur (r128 − r64 p = 0,0001, GA 0'ı kapsıyor). H03 kartının ön-kaydı bu yüzden basamaklar arası **birincil ölçü olarak bootstrap'ı** belirlemişti. Bu tabloda da karar bootstrap'tan okunur.

### 1b. İkincil metrikler (tahminlerden tek tip hesap; baseline dahil)

| metrik (val) | B | r28 | r64 | r128 | H01 (r224) | H02 | H02 aucsecim |
|---|---|---|---|---|---|---|---|
| ACC | 0,7396 ± 0,0205 | 0,7739 ± 0,0015 | 0,8098 ± 0,0015 | 0,8195 ± 0,0019 | 0,8223 ± 0,0013 | **0,8672 ± 0,0028** | 0,8588 ± 0,0070 |
| AUC (makro OvR) | 0,9228 | 0,9321 | 0,9503 | 0,9565 | 0,9563 | **0,9697** | 0,9718 |
| dengeli ACC | 0,4621 | 0,4634 | 0,5472 | 0,5621 | 0,6494 | **0,7062** ± 0,016 | 0,6856 |
| makro-F1 | 0,4602 | 0,5094 | 0,5950 | 0,6230 | 0,6777 | **0,7415** | 0,7222 |
| ECE-15 | 0,0365 | 0,0243 | 0,0270 | 0,0295 | 0,0309 | **0,0745** ± 0,008 | 0,0722 |
| ort. güven − ACC | +1,5 | −0,8 | −1,9 | −2,5 | +2,5 | **+7,0** | +7,1 |
| NLL | 0,695 | 0,615 | 0,533 | 0,512 | 0,518 | 0,518 | 0,484 |
| nv payı (gerçek 0,669) | 0,672 ± 0,056 | **0,739** | 0,726 | 0,723 | 0,698 | 0,701 | 0,694 |
| molemax-dışı ACC (n = 619) | 0,609 | 0,657 | 0,711 | 0,727 | 0,730 | **0,800** | 0,787 |
| mel / bkl OvR AUC | 0,874 / 0,882 | 0,878 / 0,889 | 0,892 / 0,927 | **0,906 / 0,937** | 0,894 / 0,931 | **0,938 / 0,958** | 0,943 / 0,960 |
| tabakalı S − N (puan) | +2,8 | +0,2 | +4,1 | +1,7 | +4,0 | **+6,65** | +5,25 |
| seed çifti argmax anlaşmazlığı | %19,0 | %0,9 | %0,5 | %0,6 | %0,9 | %9,0 | %9,3 |
| eğitim süresi / seed | 1.942 s (100 ep, 28 px) | 212 s | 209 s | 212 s | 213 s | 1.458 s (30 ep) | — |
| öğrenilen param | 11,17 M | 5.383 | 5.383 | 5.383 | 5.383 | 11,18 M | — |
| tepe GPU belleği | — | 43 MB | 43 MB | 43 MB | 43 MB | 1.742 MB | — |

Sonda süresi 8 C × 5 seed ızgarasını içerir. Omurga çıkarımı (≈ 5,3 dk/basamak, `on-ozn`) buna dahil değildir.

### 1c. Sınıf bazlı duyarlılık (val destek: akiec 33, bcc 52, bkl 110, df 12, mel 111, nv 671, vasc 14)

| sınıf | B | r28 | r128 | H01 | H02 | sınıf AUC (H02) |
|---|---|---|---|---|---|---|
| akiec | 0,382 | 0,242 | 0,455 | **0,600** | 0,582 | 0,959 |
| bcc | 0,500 | 0,546 | 0,631 | 0,746 | **0,904** | 0,993 |
| bkl | 0,511 | 0,469 | 0,653 | 0,569 | **0,707** | 0,958 |
| df | 0,017 | 0,167 | 0,267 | 0,333 | **0,533** | 0,983 |
| mel | 0,495 | **0,418** | 0,478 | 0,532 | **0,618** | 0,938 |
| nv | 0,872 | 0,944 | 0,953 | 0,937 | 0,957 | 0,964 |
| vasc | 0,457 | 0,457 | 0,500 | **0,829** | 0,643 | 0,994 |

df (12) ve vasc (14) satırları örnek başına 7–8 puanlık adımlarla değişir. Ön-kayda göre yorumlanmaz.

---

## 2. Kırmızı bayrak taraması

| # | Bayrak | H01 | H02 | H03 |
|---|---|---|---|---|
| 1 | Tavan ("görülmemiş") | 0,8223 < 0,88; AUC 0,956 < 0,99 → **yok** | 0,8672 < 0,90; AUC 0,970 < 0,99 → **yok** | r28 0,774 < 0,80; r64 0,810 < 0,85; r128 0,820 < 0,87; Δ(224 − 28) +4,85 < +13 → **yok** |
| 2 | Ebeveynle bit-bit aynı / değişiklik devrede değil | Tahminler B'den 400 görüntüde farklı doğruluk → yok | 405 görüntü → yok | `kollar_farkli`: r28 ↔ r64 160 görüntü, r128 ↔ r224 91 görüntü → yok |
| 3 | n < 5 | n = 5 | n = 5 | her basamak n = 5 |
| 4 | Seed bağlılığı (R5) | Koşumlar arası max\|Δp\| 0,041–0,065 (> 0) ve aynı seed tekrarı 0,0 → bağlı + deterministik. **a2 ↔ a3 aynı seed: max\|Δp\| = 0,0, argmax %100 aynı** (s0, s1) → tekrar üretilebilir | Koşumlar arası max\|Δp\| 0,92–0,99, ilk adım kaybı 2,07 ± 0,19, seed çifti anlaşmazlığı %9 → bağlı | r28/r64/r128: 0,023–0,041 (> 0), aynı seed 0,0, r28 a2 ↔ a3 bit-bit aynı → bağlı |
| 5 | Sonda yakınsaması (kural e) | \|SGD − lbfgs\| = **0,22** puan ≤ 0,5 ✓. Ancak eğitim amacı göreli farkı %0,91 (r28–r128'de %0,04). Daha zayıf düzenlileştirmede (C = 0,1) 100 epoch SGD optimuma daha uzak kalıyor. Tur-2'de C büyürse bu artar (§6 notu) | — | r28 **0,42** (sınıra 0,08 puan; a2'de 0,4985 idi), r64 0,02, r128 0,20 ✓ |
| 6 | Epoch seçimi sonda yakın + hâlâ artış (H02 kural e) | — | Seçilen epoch'lar 29, 7, 29, 6, 16 → 30 yok. s0 ve s2'de 29 ama eğri ep ≈ 13'ten beri 0,86–0,87 platosunda (son epoch 0,862 / 0,869) → **tetiklenmedi** | — |
| 7 | **İkincil metrikte bedel (M7, I-bedel)** | ECE 0,031 < B 0,0365; nv payı +2,6 puan ama dengeli +18,7 → **bedel yok** | **ECE 0,0745 ≈ B × 2,0**, ort. güven − ACC **+7,0 puan**. Güveni ≥ 0,9 olan 827 tahminin 60'ı yanlış (B'de 375'te 9). → **ödünleşim notu** (aşağıda) | r28: **nv payı +6,7 puan, dengeli ACC +0,13 [−5,9; +6,5]** → ödünleşim kuralı harfiyen tetiklenmedi ama sınırda (aşağıda) |
| 8 | Beklentinin tersi | Hepsi kaynaklı aralıklarda (§4). Tek ikincil sapma: molemax-dışı 0,730 < türetilmiş [0,74; 0,80] → açıklandı (§3d) | Yok | Keşifsel "r ↑ → mel/bkl AUC ↑" 224 basamağında tersine döndü (r128 > r224) → D-gözlemi (§3c), arıza değil |
| 9 | Sızıntı (keşifsel, karar vermez) | +4,0; sınıf içi perm. p_tek 0,15 | **+6,65; perm. p_tek 0,009 (p_iki 0,021)** → **SARI** (§3b) | r ile monoton değil (+0,2 / +4,1 / +1,7 / +4,0) |

**Kırmızı bayrak sayısı: 0** (TUT'u engelleyen yok). **Sarı/ödünleşim notu: 3** (H02 kalibrasyonu; H02 tabakalı sızıntı sinyali; r28 ACC kazancının nv sınırı payı). **Araç (I-grubu) bulgusu: 1** (leaderboard p'leri yaklaşık, §0).

---

## 3. Bayrakların yorumu

### 3a. H02 kalibrasyonu (ECE 0,0745)
- Ölçülen: H02 aşırı güvenli. Ortalama güven 0,937, ACC 0,867. Son epoch'ta ECE 0,083, aucsecim kolunda 0,072. Yani seçim kuralından bağımsız. Baseline'ın seçilen erken epoch'u (0,0365) ve sondalar (0,024–0,031) iyi kalibre.
- Mekanizma (D, literatürle uyumlu): uçtan uca ince ayar, train kaybını sıfıra iterken softmax'ı keskinleştirir (Guo vd. 2017). Bu, H02 kartının ön-beyanında "val-ACC seçimiyle orta epoch'ta aşırı güven görülebilir" diye öngörülmüştü.
- Teşhis (**val'de uydurulan sıcaklık, iyimser, karar değil**): T = 1,48–1,86 ile ECE **0,0745 → 0,0315**, yani baseline düzeyine iniyor. H01'de T = 1,19 ile 0,031 → 0,019. Argmax ve ACC değişmez. **Bedel ölçeklemeyle giderilebilir görünüyor.** Bu bir ödünleşim notudur, TUT'u engellemez. Makale H02 satırında ECE'yi ölçeklemesiz yazar, ölçeklenmiş değeri ayrıca verir (§6 H07 önerisi).

### 3b. Tabakalı sızıntı: ince ayarın ezber payı artıyor mu?
- Nokta tahminleri ön-kayıtlı keşifsel yönde: **B +2,8 < H01 +4,0 < H02 +6,65.** Kart H02 > B ve H02 > H01 bekliyordu.
- **Kol içi:** yalnız H02'de tabaka içi permütasyon anlamlı (p_tek 0,009). B'de 0,20, H01'de 0,15. Kaynak kırılımı da aynı yönde. histo'da H02 S 0,807 / N 0,748 (+6,0), H01 0,709 / 0,701 (+0,8). rosendahl'da H02 +6,7, H01 +0,1. vidir_modern'de H02 +3,3, H01 −2,1.
- **Kollar arası fark anlamlı değil:** H02 − B +3,87 [−3,65; +11,24], H02 − H01 +2,63 [−5,25; +10,82]. Val'de tabakalar N'nin yalnız %41,8'ini kapsıyor (19 tabaka) ve GA genişliği ±8 puan. **Val bu soruyu çözemez.**
- **H02 − H01 kazancı sızıntıyla açıklanabilir mi? En kötü durum sınırı:** H02'nin H01'i aşan tabakalı farkının (+2,63 puan) tamamı S'deki (348/1.003 = %34,7) ezberden gelse, H02'nin ACC'si ≈ 0,9 puan düşer. H02'nin tüm S üstünlüğü (+6,65) sıfırlansa ≈ 2,3 puan düşer, ama aynı düzeltme H01'i de ≈ 1,4 puan düşürür. Her iki senaryoda H02 − H01 ≥ ≈ +3,6 puan kalır. Bu, ebeveyn kuralının 1,0 puan eşiğinin çok üstünde. Ayrıca lezyonu train'de **olmayan** N alt kümesinde de H02 önde: +2,32 [−0,03; +4,60]. Molemax-dışı N'de (n = 274, tavan etkisi yok) +5,04 [−0,22; +10,29].
- **Sonuç:** "İnce ayar lezyon kimliğini sondadan daha çok kullanıyor" hipotezi **destekleniyor ama kanıtlanmadı** (kol içi anlamlı, kollar arası değil). Bu sinyal H02'nin H01'e üstünlüğünü açıklamaya yetmiyor. Test'in S oranı aynı (%34,6). Literatür sayıları da aynı resmî split'te olduğundan rakiplere karşı kıyas adil. Ancak makalede "H02 > H01" iddiası **DermaMNIST-C (K5, kullanıcı kararı 3) sonucuna bağlanmalı.** K5'te aynı tabakalı ölçü C-val/C-test üzerinde ve H01/H02 için birlikte koşulmalı.

### 3c. H03: 224 basamağı ve C\* karıştırıcısı
- ACC'de r128 → r224 adımı **+0,28 [−1,42; +1,99]**, ayırt edilemiyor. Literatürde aynı adım daha büyük: Wu 81,88 → 83,97 (+2,1), Doerrich DINO 81,65 → 84,42 (+2,8). Bizim GA'nın üst ucu (+2,0) literatür değerine değiyor. **D-gözlemi, ihlal değil.** r28/r64/r128 Wu Tablo A ile 0,3–1,3 puan içinde (77,37 / 79,55 / 81,88), r224 ise Wu'nun 1,7 puan altında (val binom SE ≈ 1,2). K5 test'te yeniden bakılacak.
- **Dengeli ACC'deki r128 → r224 sıçraması (+8,7) büyük ölçüde C\* farkından geliyor.** r28–r128'de seçilen C = 0,01, r224'te C = 0,1. Teşhis (lbfgs, aynı ölçekleme, val'de seçim yok):

  | | C = 0,01 | C = 0,1 |
  |---|---|---|
  | r128 ACC / dengeli | 0,8175 / 0,558 | 0,8146 / 0,611 |
  | r224 ACC / dengeli | 0,8215 / 0,567 | 0,8245 / 0,652 |

  Sabit C'de çözünürlük payı dengeli ACC'de +0,9 (C = 0,01) ile +4,1 (C = 0,1) arasında. Kalan +4–8 puan, ACC'de ≈ eşit iki C arasındaki seçimden geliyor (r224 ızgarası: C = 0,01 → 0,8211, C = 0,1 → 0,8223). **Bulgu (Tur-2 hammaddesi):** ACC ile C seçmek, ACC'yi 0,1–0,3 puan oynatırken dengeli ACC'yi 5–9 puan oynatıyor. Daha güçlü L2 nadir sınıfları bastırıyor (Z2'nin sonda karşılığı). Makaledeki çözünürlük figürü dengeli ACC'yi **sabit C'de de** göstermeli, aksi halde eğri yanıltır (I-grubu).
- Keşifsel ön-beyan "r ↑ → mel/bkl AUC ↑" 28 → 128'de tutuyor (mel 0,878 → 0,906, bkl 0,889 → 0,937), 224'te hafifçe geri dönüyor (0,894 / 0,931). C\* farkıyla da uyumlu, ayrıştırılmadı.
- Keşifsel ön-beyan "çözünürlükten en çok mel, bkl, bcc yararlanır": aynı C'de (r28 → r128) bkl +18,4, akiec +21,2, bcc +8,5, mel +6,0 puan. **Kısmen** destekleniyor: bkl ve akiec en çok, mel en az.

### 3d. H01 molemax-dışı 0,730 < türetilmiş [0,74; 0,80]
Türetilmiş aralık "molemax alt kümesi 0,949'da sabit kalır" varsayımının aritmetiğiydi. Kartta literatür değeri değil, aritmetik olarak yazılmıştı. Ölçülen molemax ACC 0,971 (+2,2). Aynı toplam ACC bu yüzden daha düşük bir molemax-dışı ACC ile elde ediliyor. Varsayım bozuldu, sonuç değil. **D: açıklandı.**

### 3e. r28: ACC kazancının nv sınırı payı (T3'ün yorumu için kritik)
- Ödünleşim kuralı (Δacc > 0 **ve** Δdengeli ≤ 0 **ve** Δnv payı > +0,02): Δacc +3,43, **Δdengeli +0,13** (> 0, kural harfiyen tetiklenmedi), Δnv payı **+6,7 puan**. Dengeli fark GA'sı [−5,85; +6,53], yani sıfırdan ayırt edilemiyor. **Kural sınırda.**
- Mekanizma: r28 sondası nv duyarlılığını 0,872 → 0,944'e çıkarıyor. Bunun bedelini mel (0,495 → 0,418) ve bkl (0,511 → 0,469) ödüyor. hata-analizi §2'deki ACC ↔ nv payı r = +0,88 tam olarak bunu öngörmüştü.
- Teşhis (**val'de ayarlı, iyimser, karar değil**): her kolun nv log-olasılığı, nv payı gerçek paya (0,669) eşitlenecek kadar kaydırıldığında ACC B 0,7428, r28 0,7579 (**+1,5**), dengeli ACC 0,467 ve 0,492. Yani r28'in +3,4 puanının **yaklaşık yarısı nv karar sınırının kaymasından**, yarısı temsilden geliyor. Eşik bağımsız göstergeler de küçük ama pozitif bir temsil kazancı gösteriyor: AUC +0,93, NLL 0,695 → 0,615, makro-F1 +4,9 puan.
- Makale için: T3 ön-kayıtlı kurala göre "kazandırır" çıkıyor (§4), ama cümle **"28 px bilgiyle donuk DINOv2 sondası ACC'de +3,4 puan kazandırır; kazancın yaklaşık yarısı nv sınırının kaymasından gelir; sınıf-dengeli doğrulukta fark yoktur"** biçiminde kurulmalı (I-bedel).

---

## 4. Kart başına karar ve hazır sonuç paragrafları

> Kart dosyalarının "Sonuç" bölümleri doldurulmadı (deney-yürütücü yapacak). Aşağıdaki paragraflar o bölüme konmak üzere hazır.

### H01 — DINOv2-B donuk sonda @224 → **TUT**
Öldürme kuralları: (a) n = 5 ✓. (b) 0,8223 ≥ 0,7596 ve p_t = 0,00044 < 0,05 → geçti. (b') GA [+5,66; +10,83] > 0, t-testiyle tutarlı → çelişki yok. (c) ≤ 0,690 değil. (d) 25,4 img/s ≥ 10. (e) \|SGD − lbfgs\| 0,22 puan ≤ 0,5. Tavan yok. Ödünleşim kuralı tetiklenmedi.

> **Sonuç (H01):**
> - Koşumlar: `20261010-221110_h01_dinov2b_lp_r224_s0`, `-221455_s1`, `-221838_s2`, `-222218_s3`, `-222554_s4` (C\* = 0,1; lbfgs denetimi 0,8245).
> - Ölçülen: val ACC **0,8223 ± 0,0013** (n = 5), Δ vs B **+8,28 puan**, seed-eşleştirilmiş tek yönlü p = 0,00044, eşleştirilmiş görüntü bootstrap %95 GA **[+5,66; +10,83]**. AUC 0,9563 (Δ +3,35). Dengeli ACC 0,649 (Δ +18,7 [+12,9; +25,2]). Makro-F1 0,678. ECE 0,031 (B 0,0365). Molemax-dışı ACC 0,730 (Δ +12,1). mel/bkl AUC 0,894/0,931. Tabakalı S − N +4,0 (perm. p 0,15). Beklenti [0,820; 0,860] içinde (alt uçta). Literatür (test): Wu 83,97, Doerrich DINO-v1 84,42.
> - Karar: **TUT**. Öldürme kuralının (b), (b') ve (e) koşulları geçti, tavan yok, ödünleşim yok. Kazancın çoğu nv↔mel karışıklığının azalmasından geliyor (93,2 → 60,4 hata). Nadir sınıf duyarlılığı 0,285 → 0,587.
> - Öğrenilen: Donuk SSL özniteliği üzerinde 5.383 parametrelik bir doğrusal sonda, sıfırdan eğitilen 28 px ResNet-18'i val'de 8,3 puan geçiyor ve seed varyansını 2,05'ten 0,13 puana indiriyor. Sonda ızgarasında ACC ile C seçimi dengeli ACC'yi 5–9 puan oynatabiliyor (C = 0,01 ↔ 0,1).

### H02 — ResNet-18 ImageNet, native 224, uçtan uca ince ayar → **TUT (ödünleşim notuyla)**
Öldürme kuralları: (a) ✓. (b) 0,8672 ≥ 0,7596, p_t = 9·10⁻⁵ → geçti. (b') GA [+10,67; +14,86] > 0, çelişki yok. (b'') `val_acc_aucsecim` 0,8588 ≥ 0,7596 → geçti (GA [+9,89; +13,92]). (c) yok. (d) epoch 48,6 s ≤ 90. (e) seçilen epoch 30 yok, eğri platoda. Tavan yok. Ödünleşim kuralı tetiklenmedi (Δdengeli +24,4). Seçim kuralı farkı val_acc − aucsecim = +0,84 puan ∈ [0; +2] ✓.

> **Sonuç (H02):**
> - Koşumlar: `20261010-222930_h02_r18in_ft224_s0`, `-225428_s1`, `-231909_s2`, `-234351_s3`, `20261011-000829_h02_r18in_ft224_s4`.
> - Ölçülen: val ACC **0,8672 ± 0,0028** (n = 5; seçilen epoch 29/7/29/6/16), Δ vs B **+12,76 puan**, tek yönlü p = 9·10⁻⁵, bootstrap GA **[+10,67; +14,86]**. Baseline'ın seçim kuralıyla (val-AUC) `val_acc_aucsecim` **0,8588 ± 0,0070**, Δ +11,92 [+9,89; +13,92]. AUC 0,9697. Dengeli ACC 0,706 (Δ +24,4 [+18,7; +30,3]; beklenti [0,70; 0,80]). Makro-F1 0,742. Molemax-dışı 0,800 (beklenti [0,74; 0,84]). mel/bkl AUC 0,938/0,958. **ECE 0,0745 (B 0,0365; aşırı güven +7,0 puan).** Tabakalı S − N +6,65 (perm. p 0,009). Beklenti [0,820; 0,885] içinde. Literatür (test): Doerrich 82,33, Wu 86,05, Zhang 0,882 (augmentasyonlu).
> - Karar: **TUT**. Öldürme kurallarının hepsi geçti, tavan yok. **Ödünleşim:** kalibrasyon bozuluyor (ECE × 2). Val'de uydurulan sıcaklıkla 0,032'ye iniyor (teşhis). Lezyon tanıma işareti kol içinde anlamlı (+6,65), ama H01'e ve B'ye karşı farkı GA içinde. Makale ikisini de H02 satırında yazar.
> - Öğrenilen: "Aynı önağırlık hakkı rakibe" verildiğinde, ImageNet ResNet-18 ince ayarı native 224'te donuk DINOv2-B sondasını val'de 4,5 puan [+2,3; +6,6] geçiyor. Bu, "DINOv2 saman adamı yendi" itirazının ölçülmüş karşılığı. Bedeli 7× süre (sonda ızgarasına göre), aşırı güven ve olası lezyon ezberi.

### H03 — çözünürlük merdiveni (analiz kartı) → **T1 reddedilmedi · T2 kabul · T3 "28 px'te de kazandırır" (nv notuyla)**
Operasyonel DUR koşulları: val_acc ≤ 0,690 yok; yakınsama r28 0,42 / r64 0,02 / r128 0,20 puan ≤ 0,5; boyutlar arası etiket hizası 5/5 koşumda npz ile aynı; `kollar_farkli` ✓.

> **Sonuç (H03):**
> - Koşumlar: r28 `20261010-205746/210137/210519/210903/211247`; r64 `211625/213824/214206/214540/214915` (yetim `212001` rc=96 dışlandı); r128 `215253/215633/220012/220350/220730`; r224 = H01.
> - Ölçülen (val ACC, n = 5): r28 **0,7739 ± 0,0015**, r64 **0,8098 ± 0,0015**, r128 **0,8195 ± 0,0019**, r224 **0,8223 ± 0,0013**. AUC 0,932 / 0,950 / 0,957 / 0,956. Dengeli ACC 0,463 / 0,547 / 0,562 / 0,649 (224'te C\* değişti, aşağıda). Komşu basamak farkları (bootstrap): +3,59 [+1,28; +6,04], +0,98 [−0,86; +2,95], +0,28 [−1,42; +1,99]. **Δ(r224 − r28) = +4,85 [+2,29; +7,56].** **r28 − B = +3,43**, tek yönlü p = 0,0115, GA [+1,16; +5,74]. Literatür Wu Tablo A 77,37/79,55/81,88/83,97 (r28–r128 için 0,3–1,3 puan içinde).
> - Karar: **T1** reddedilmedi (hiçbir komşu fark < −1,0). Eğri 28 → 64'te belirgin, **64 → 128 → 224 adımları ayırt edilemiyor** (plato). **T2** kabul: +4,85 ∈ ön-kayıtlı [+4; +10] ve > +2,0, tavan +13'ün altında. **T3: "SSL temsil 28 px'te de kazandırır"** (ortalama 0,7739 ≥ 0,7596, p < 0,05, GA alt > 0; Wu ile uyumlu, Ren 0,725 ile çelişiyor). **Nitelik:** kazancın yaklaşık yarısı nv karar sınırının kaymasından geliyor (nv payı 0,672 → 0,739; dengeli ACC farkı +0,13 [−5,9; +6,5]; nv payı eşitlenince teşhis Δ ≈ +1,5).
> - Öğrenilen: Donuk DINOv2'nin 28 px'teki üstünlüğü küçük ve büyük ölçüde nv sınırında. Çözünürlük payı (+4,85) temsil payından (+3,43) büyük. Asıl sıçrama 28 → 64'te. 128 px'ten sonra val'de ek kazanç ölçülemiyor (literatürde +2,1…+2,8). Dengeli ACC eğrisi C seçimine çok duyarlı (sabit C'de 128 → 224 farkı +0,9…+4,1).

---

## 5. Tur-2 ebeveyni (SIRALAMA "Tur-2 ebeveyn kuralı")

Kural: H01 ve H02 arasından 5-seed ortalama val_acc'si en yüksek **TUT** varyant ebeveyn olur. Fark < 1,0 puan **ve** (p ≥ 0,05 **veya** GA 0'ı kapsıyor) ise H01 seçilir.

- İkisi de TUT. Fark H02 − H01 = **+4,49 puan** (≥ 1,0). Tek yönlü p_t < 10⁻⁵, GA [+2,31; +6,62] 0'ı kapsamıyor. Baseline'ın seçim kuralıyla (aucsecim) da +3,65 [+1,42; +5,68]. En kötü durum sızıntı düzeltmesiyle de ≥ ≈ +3,6 (§3b).
- **→ Tur-2 ebeveyni: `h02_r18in_ft224`.** Bu, kuralın mekanik çıktısı. Kural yorumlanmadan uygulandı.
- SIRALAMA'daki "büyük olasılıkla H01" tahmini, H01'in düşük varyansına dayanıyordu (Z3). H02'de bu kaygı da büyük ölçüde kalkıyor: seed std'si 0,28 puan, B'de 2,05. Ebeveyne karşı seed-eşleştirilmiş t-testinin saptanabilir en küçük etkisi (iki kol aynı oynaklıkta, sd_Δ ≈ √2 × 0,28 ≈ 0,40; %80 güç, tek yönlü α 0,05) **≈ 0,5 puan**; gözlenen H02 − H01 sd_Δ'sıyla (0,31) ≈ 0,4. Bağlayıcı belirsizlik ise görüntü örneklemi: eşleştirilmiş bootstrap GA yarı genişliği ≈ ±2 puan (H02 − H01'de [+2,31; +6,62]). ±1–2 puanlık Tur-2 etkileri t-testiyle 'anlamlı', bootstrap'la sınırda çıkabilir. Ön-kayıtlı kural gereği ikisi çelişirse TUT verilmez, bu yüzden ikisi birlikte raporlanmalı.
- Pratik sonuç: H02'nin `tahmin_val.npy` ve `best_model.pth` dosyaları saklı. **Eğitimsiz (post-hoc) Z2 müdahaleleri dakikalar sürer** (logit ayarı, eşik). Yeniden eğitim isteyen müdahalelerin (kayıp ağırlığı) her biri ≈ 2,1 GPU-sa.

---

## 6. Tur-2 hammaddesi (H02 kırılımı, `hata-kirilim-h02_r18in_ft224.csv`) ve hipotez önerileri

### 6a. Hata kütlesi nerede (H02, val)
| Ölçü | B | H01 | **H02** |
|---|---|---|---|
| toplam hata (5-seed ort.) | 261,2 | 178,2 | **133,2** |
| nv–mel–bkl üçgeni payı | %64,6 | %67,0 | **%66,7** (nv↔mel 48,8, nv↔bkl 26,0, mel↔bkl 14,0) |
| mel → nv (/111) | 43,8 | 38,6 | **36,0** (en büyük tek hücre) |
| nv dışının nv'ye gitmesi (/332) | 88,6 | 71,0 | 61,4 |
| tahmin sayısı nv (gerçek 671) / mel (111) | 674 / 122 | 700 / 103 | **703 / 97** → nv fazla, mel eksik tahmin |
| nadir üçlü (df+vasc+akiec, 59) duyarlılık | 0,285 | 0,587 | 0,586; hata oranı %41,4 |
| nadir sınıf AUC ↔ duyarlılık (H02) | — | — | df 0,983 ↔ 0,533 · vasc 0,994 ↔ 0,643 · akiec 0,959 ↔ 0,582 → **"bilgi var, karar yok" (Z2) H02'de de sürüyor** |
| tahmin sayısı df / vasc / akiec (gerçek 12/14/33) | 0,2 / 12,4 / 30,8 | — | 8,0 / 10,4 / 29,2 → nadir sınıflar eksik tahmin |
| hep-yanlış (5/5 seed) | 127 | 174 | **62** (mel→nv 21, bkl→nv 9, bkl→mel 5, nv→bkl 5) |
| kararsız (1–4 seed yanlış) | 296 | — | 783 hep-doğru, ≈ 158 kararsız |
| **teşhis (val'de ayar yok): önsel düzeltme τ = 1** (p/π_train) | ACC 0,583 (−15,7), dengeli 0,525 (+6,3) | ACC 0,720 (−10,3), dengeli 0,732 (+8,2) | **ACC 0,845 (−2,3), dengeli 0,761 (+5,4), nv payı 0,648** |
| teşhis: 5-seed olasılık ortalaması (A9) | 0,771 (+3,1) | 0,824 (+0,1) | **0,878 (+1,1)**, dengeli 0,721, ECE 0,037 |
| teşhis: DermaMNIST-C val alt kümesi (573) / C-train'e taşınan (430) | 0,826 / 0,625 | 0,891 / 0,730 | 0,911 / 0,808 (K5 için ön bilgi; C üzerinde eğitilmiş değil) |

**Okuma:** H02'de Z2 hâlâ geçerli ve en ucuz hedef. Nadir sınıflarda AUC 0,96–0,99, duyarlılık 0,53–0,64. nv fazla, mel/nadir sınıflar eksik tahmin ediliyor. τ = 1 önsel düzeltmesinin bedeli H02'de yalnız −2,3 ACC (B'de −15,7, H01'de −10,3). Bu, ılımlı bir τ'nun ACC-nötr kalıp dengeli ACC'yi +2…+5 puan artırabileceğini düşündürüyor. Z1'in kalanı (mel → nv 36/111) Tur-3 (augmentasyon/TTA) ve Tur-4'ün işi.

### 6b. Hipotez önerileri (kart DEĞİL; kartlar ön-kayıtla yazılır; sıradaki boş id H04)

| Öncelik | Öneri | Aile / Z | Ebeveyn | Değişen tek şey | Ön-kayıt adayı beklenti (kaynak) | Görülmemiş | Maliyet |
|---|---|---|---|---|---|---|---|
| **1** | **H04 — post-hoc logit ayarı** | A8 / **Z2** | H02 | Tahminde log p(c\|x) − τ·log π_train(c) (Menon vd. 2021). τ ∈ {0; 0,1; …; 1,0}, seçim **val dengeli ACC** ile, **val ACC ≥ ebeveyn − 1,0** kısıtı altında (SIRALAMA D1 seçim kuralı) | **Dengeli ACC +2…+4 puan** (Zhang & Li 2026 T1d WS makro-ACC +2,3, T1a +4,2; literatur-cipa §4(c)). Üst zarf ölçülmüş: τ = 1'de +5,4. **ACC −1…0** (kısıt). nv payı ↓, mel/df/vasc/akiec duyarlılığı ↑ | ACC'de > +2 puan (literatur-cipa §4(c)). Dengeli ACC'de > +6 puan → İNCELE (τ = 1 zarfının üstü; kendi ölçümümüze dayalı, literatür değil, öyle yazılmalı) | Eğitimsiz, CPU < 5 dk. K5'te test logitleri `best_model.pth`'den bir kez |
| **2** | **H05 — sınıf-ağırlıklı CE** | A5 / **Z2** | H02 | Kayıpta w_c ∝ π_c^(−β), **tek β = 0,5 (arama yok)**. Tarif başka hiçbir yerde değişmez. Epoch seçimi proje kuralıyla (val ACC); dengeli ACC ikincil olarak ayrıca raporlanır | **Dengeli ACC +2,3…+4,2**, **ACC −1,5…+0,3** (Zhang & Li 2026 T1d/T1a: aynı model ailesi, 224, AdamW 1e-4, 30 ep; literatur-cipa §4(c)) | ACC'de > +2 puan | 5 seed × 30 ep ≈ **2,1 GPU-sa** (+ duman/ön-uçuş ≈ 0,15) |
| 3 | H06 — aynı logit ayarı H01 sondasında | A8 / Z2 | H01 | H04'ün birebir aynısı, sonda ebeveyninde | Dengeli ACC artışı, ACC −1…0. τ = 1 teşhisi: dengeli +8,2, ACC −10,3 → ılımlı τ'da ACC kısıtı bağlayıcı olur | aynı | CPU dakikalar. **Değeri:** etkinin temsil türünden bağımsızlığı (iki ebeveynde aynı yön). Makale genellemesi için ucuz kanıt |
| 4 | H07 — sıcaklık ölçekleme (kalibrasyon) | A8 / H02 ödünleşimi | H02 | Tek T, val NLL ile **2 katlı çapraz uydurma** (val içinde, sınıf-tabakalı; iyimserliği önler) | ECE 0,0745 → ≈ 0,03–0,04 (teşhis 0,032, iyimser). ACC/dengeli **değişmez** (argmax değişmez) | ECE < 0,01 → İNCELE | CPU saniyeler. **Durma sayacına katkısı yok.** Tur içinde ayrı satır, ödünleşimin makaledeki çözümü |

Notlar:
- **M2:** H04 ile H07 birleştirilmez. T, logit ayarının argmax'ını etkiler (log p/T − τ log π). H04, H07'siz ham olasılıklar üzerinde ön-kayıtlanır.
- **H04 iyimserliği:** τ val'de seçilip val'de ölçülürse iyimserlik oluşur. Önerilen ön-kayıt, val içinde 2 katlı sınıf-tabakalı çapraz uydurmayla bir "yansız val" sayısı **ve** tüm val'de seçilmiş τ. Tarafsız sayı K5 test'tir.
- **H06 ve H01 sondası:** C büyürse SGD yakınsama farkı artar (§2 #5). Sonda tabanlı yeni kartlarda lbfgs denetimi zorunlu kalmalı. H06 eğitim gerektirmez, bu yüzden etkilenmez.
- Tur başına ≤ 3 kart: **önerilen paket H04 + H05 (+ H06)**. H07 ACC-nötr bir ölçüm satırı olarak eklenebilir. Focal loss önerilmiyor: literatürde 224'te ACC −0,5, makro-ACC −0,5 (Zhang & Li T1d). H04/H05'ten ayırt edilemeyecek kadar küçük.

---

## 7. Durma kuralı sayacı (konu.md 2026-10-10 20:15: iki ölçüt)

| | Tur-1 öncesi en iyi (B) | Tur-1 en iyi (TUT) | İyileşme | ≥ 0,5 puan? |
|---|---|---|---|---|
| val ACC | 0,7396 | **0,8672** (H02) | **+12,76** | evet |
| val dengeli ACC | 0,4621 | **0,7062** (H02) | **+24,42** | evet |

**Sayaç: 0 / 2** (ardışık iyileşmesiz tur yok). Tur-2'nin sayacı sıfırda tutması için en iyi val ACC ≥ **0,8722** ya da en iyi val dengeli ACC ≥ **0,7112** olmalı. Tur-2 ön-kayıtlı olarak ACC-nötr. Sayacı dengeli ACC ölçütü belirler: H04/H05'in beklenen +2…+4 puanı eşiğin (+0,5) rahatça üstünde. H03 analiz kartı sayaçta yer almaz.

## 8. Bütçe görüşü

- Tur-1 GPU (leaderboard `sure_sn`, duman + ön-uçuş + tam): **≈ 3,5 sa** (H02 2,18; sondalar 1,30). Ek olarak yetim r64 s1 (birkaç dk) ve Tur-1 öncesi `on-ozn` (≈ 0,4 sa). Leaderboard'daki kümülatif toplam 6,43 sa. rc=95 ilk baseline denemeleri ve `on-ozn` ile birlikte **≈ 7–7,5 / 32 GPU-sa**, **1 / 8 tur**. Kesin sayaç şef günlüğünde. **`gunluk/deney-gunlugu.md` bütçe sayacı hâlâ "0 / 24 sa · 0 / 8 tur" gösteriyor:** hem güncel değil hem eski üst sınırı (24) taşıyor → güncellenmeli (32 GPU-sa, 40 sa duvar).
- Kalan ≈ 24,5 GPU-sa ve 7 tur. Önerilen Tur-2 paketi (H04 + H05 + H06 + H07) ≈ **2,3 GPU-sa**. Bütçe Tur-3 (augmentasyon H02'de yeniden eğitim ≈ 2,1 sa/kart; TTA eğitimsiz) ve Tur-4'e (kısmi DINOv2 ince ayarı, yerelde pahalı → Colab tetiği) bol pay bırakıyor. Durma kuralına yaklaşılmadı.
- **Kalan bütçenin en iyi kullanımı:** H02 ebeveyninde eğitimsiz müdahaleler önce (H04/H07, dakikalar), sonra tek yeniden eğitim (H05). Tur-5'teki topluluk (A9) teşhisi H02'de +1,1 puan veriyor (sonda 0). Rakibe de topluluk hakkı verilmeli (M3).

## 9. Günlük arıza tablosu için satır önerileri (`gunluk/deney-gunlugu.md`)

| Tarih | Belirti | Hipotez | Teşhis yöntemi | Kök neden | Düzeltme | Etkilenen koşumlar |
|---|---|---|---|---|---|---|
| 2026-10-11 | Leaderboard p'leri (H01 0,0115, r28 0,037) kesin tek yönlü t-testinden 3–100 kat büyük | p hesap yolu farklı | `leaderboard.py` `eslesmis_test` okundu. scipy ile yeniden hesap: tek yönlü 0,00044 / 0,0115 | Orkestra Python'unda scipy yok → normal yaklaşımlı iki yönlü yedek yol | Leaderboard deney venv'iyle (scipy 1.17.1) yeniden üretilmeli. Makale kesin tek yönlü p'yi kullanır | Hiçbiri geçersiz değil (kararlar aynı). Yalnız raporlanan p'ler |
| 2026-10-11 | `hata_kirilim.py --varyant h02…` KeyError `val_loss` | H02 `gecmis.json` şeması farklı | Betik okundu, `gecmis.json` anahtarları: epoch, train_loss, val_acc, val_auc, egitim_sn, epoch_sn, lr | `ince_ayar.py` val_loss / train_acc yazmıyor | Betikte `val_loss_min_epoch` koşullu yapıldı. Not etiketleri "100. epoch" → "son epoch" | Yok |
| 2026-10-11 | Baseline kırılımı yeniden koşulunca tabakalı perm. p 0,1984 → 0,1998 | Rastgelelik kaynağı sabit değil | Çıktı JSON'ları karşılaştırıldı. Fark yalnız permütasyon p'lerinde ve 1e-17 düzeyi kayan nokta | `for k in set(tab)` sırası PYTHONHASHSEED'e bağlı → RNG akışı değişiyor | `sorted(set(tab))` (iki yer). Baseline'ın kayıtlı p'si 0,198 ± 0,002 belirsizliğiyle geçerli | Yok |

## 10. Tahlil defteri (S/D/I)

| grup | bulgu | durum |
|---|---|---|
| S | Etiket hizası (5/5 koşum × 6 kol = npz), test yokluğu, yetim koşum dışlama, bootstrap betik ↔ tahlil eşitliği (12/12), a2 ↔ a3 bit-bit tekrar | geçti |
| D | B, H01, H02 ve H03 r28–r128 literatür bantlarında. r224 Wu'nun 1,7 puan altında, 128 → 224 adımı literatürden küçük | uyumlu / gözlem (K5 test'te yeniden) |
| D | H02 aşırı güven (ECE × 2) | açıklandı (ince ayar keskinleşmesi; ölçeklemeyle düzeliyor, teşhis) |
| D | r224 dengeli ACC sıçraması | açıklandı (C\* karıştırıcısı, lbfgs teşhisi) |
| D | H01 molemax-dışı türetilmiş aralığın altında | açıklandı (molemax 0,949 varsayımı bozuldu: 0,971) |
| I | Leaderboard p'leri yaklaşık | düzeltme önerildi (§0, §9) |
| I | r28 ACC kazancının yarısı nv sınırında (ödünleşim kuralı sınırda) | makale cümlesi önerildi (§3e) |
| I | H02 tabakalı sızıntı sinyali kol içinde anlamlı | sınırlılık + K5 DermaMNIST-C'ye bağlandı (§3b) |
| I | Çözünürlük figüründe dengeli ACC, C seçimine duyarlı | sabit C eğrisi önerildi (§3c) |

**Sınırlılıklar:** 5 seed aynı 1.003 val görüntüsünü paylaşıyor. Bootstrap görüntü örneklemini yansıtır, seed eşleştirmesi başlatmayı. nv eşitleme, sıcaklık ve lbfgs-C satırları val'de ayarlı teşhislerdir, kart sonucu değildir. df/vasc duyarlılıkları yorumlanmaz. Tabakalı sızıntı ölçüsü N'nin %41,8'ini kapsıyor. H02 `gecmis.json`'unda train ACC yok, aşırı uyum yalnız güven/ECE üzerinden okunabildi.
