# H08 İNCELE Soruşturması — dengeli ACC Δ kartın "görülmemiş" eşiğini aştı (YALNIZ VALIDATION)

*Deney-tahlilci, Mod 2 (K4c) + İNCELE teşhisi · 2026-10-11 · GPU kullanılmadı, test açılmadı (H07/H08/H09 `metrics.json`'larında `"test` anahtarı 0).*
*Betik (tekrar üretilebilir, CPU, RNG 0): `scripts/inceleme_h08.py` → `sonuclar/inceleme_h08.json`. Bootstrap W, `tur3_olcum.py` ile aynı (2.000, `default_rng(0)`, görüntü yeniden örneklemesi). Birincil kolun dengeli GA'sı bu betikte [+4,97; +11,79] olarak yeniden üretildi; kartla birebir aynı.*

## 0. Tetikleyici ve kartın kuralı

| | Değer | Kart eşiği (H08 "Literatürde görülmemiş") | Durum |
|---|---|---|---|
| val ACC Δ (5-seed ort.) | +2,21 [+1,18; +3,27] | > +3,0 | aşılmadı |
| ort. val ACC | 0,8893 (seed maks. 0,8933) | > 0,90 | aşılmadı |
| val AUC | 0,9785 | > 0,99 | aşılmadı |
| **dengeli ACC Δ** | **+8,06 [+4,97; +11,79]** | **> +6,5** | **AŞILDI → İNCELE, TUT verilmez (M6)** |

Kartın soruşturma sırası şöyle: augmentasyonun val'e sızması, train/val dizisinin karışması, dönüşümün etiketle birlikte yanlış indekslenmesi, `best_model.pth` ↔ `tahmin_val.npy` epoch uyumu. Görev buna dört madde ekledi: seçim iyimserliği, kazancın kaynağı, eşiğin kendisi ve mekanizma.

---

## (a) Train/val karışması ve augmentasyon sızıntısı — **TEMİZ**

**Kod okuması (`varyantlar/r18_imagenet/ince_ayar.py`, `ortak/veri.py`, `ortak/dihedral.py`):**

| Denetim | Kanıt | Sonuç |
|---|---|---|
| Train ve val ayrı diziler mi? | `V.yukle(224, ("train","val"))`. Diziler npz'nin ayrı anahtarlarından okunuyor (`train_images` / `val_images`, satır 118–120). `alt_kume` a3'te `None` | ✓ |
| Augment yalnız train'de mi? | D4 yalnız eğitim döngüsünde uygulanıyor (satır 213–224, `if a.augment_d4`). `val_olasilik()` (190–197) ise `hazirla(xva[...])` çağırıyor; bu yol augmentasyonsuz, `model.eval()` ve `no_grad` altında | ✓ |
| Augment kaynak diziyi değiştiriyor mu? | `xtr[j]` gelişmiş indeksleme olduğu için kopya üretir. Dönüşüm bu kopyanın GPU'daki hâline uygulanır, `xtr`/`xva` değişmez | ✓ |
| Etiket hizası | Dönüşüm satırların yerinde yapılıyor (`xb[m_] = D4.donustur(xb[m_], t_)`). `yb = ytr[j]` aynı `j` sırasında kalıyor, gruplar ayrık. `oz_sinama()` 8 dönüşümün ayrık olduğunu ve grubun kapalı olduğunu doğruluyor | ✓ |
| Epoch başına örnek sayısı | `augment_sayim` epoch toplamı 7.007 = `n_train` (5/5 seed, 30/30 epoch). Frekans sapması en çok 0,0016 | ✓ |
| `best_model.pth` ↔ `tahmin_val.npy` ↔ `metrics` | `tahmin_val.npy` argmax ACC = `metrics.val_acc` = `gecmis.json`'daki seçilen epoch (5/5). Seçilen epoch'un `gecmis` dengeli değeri `metrics.val_dengeli_acc` ile bit bit aynı (5/5; `inceleme_h08.json` → `h08_epoch_dengeli`) | ✓ |
| Etiket dosyaları | 10 koşumun `etiket_val.npy` dosyası, npz `val_labels` ile aynı (betikte assert) | ✓ |

**Piksel düzeyi tarama.** D4'e özgü bir sızıntı yolu düşünülebilir: bir val görüntüsü bir train görüntüsünün döndürülmüş ya da çevrilmiş kopyasıysa, H08 eğitimde o val görüntüsünü fiilen görür. Bunu sınamak için 28 px resmî npz'de 1.003 val × 7.007 train × 8 dönüşüm üzerinden en yakın komşu RMSE'si hesaplandı (0–255 ölçeği):

| Ölçü | Kimlik | D4 (kimlik dışı 7 dönüşüm) |
|---|---|---|
| En yakın komşu RMSE: en küçük / %1 / %5 / medyan | 4,39 / 7,18 / 10,1 / 15,8 | 7,18 / 8,65 / 10,3 / 15,0 |
| RMSE ≤ 3 olan val görüntüsü | 0 | 0 |
| RMSE ≤ 5 | 4 (H02'yle ortak; D4'e özgü değil) | **0** |
| RMSE ≤ 8 ve **yalnız D4 ile** yakın | — | 4 (1'i S'de). RMSE ≤ 8 ve kimlikten en az 1 birim daha yakın olan 3 nv görüntüsünde kazanç −13,3 puan |

Sonuç: D4 ile ortaya çıkan bir yakın kopya yok. 28 px'te kimlik altında görülen 4 yakın çift H02'de de vardı ve tüm kolları aynı biçimde etkiler. Bu bir veri seti özelliğidir, H08 hatası değildir. K5 sınırlılıklarına not düşülmeli.

**Aynı lezyon çiftleri (S kümesi).** 348 val görüntüsü için train'de aynı lezyona ait 440 çift var. Medyan RMSE kimlikte 30,0, en iyi D4 dönüşümünde 29,3. Bu değerler en yakın komşu medyanının (15,8) iki katı, yani bu çiftler farklı çekimler, kopya değil. Çiftlerin 252'sinde en iyi hizalama kimlikle, 188'inde başka bir dönüşümle oluyor.

Keşifsel bir alt küme de denendi: başka bir D4 dönüşümünün kimlikten en az 3 RMSE birimi daha iyi hizaladığı 25 val görüntüsü. Bu kümede ΔACC +10,4 puan, sınıf ve molemax tabakalı permütasyon sıfırı +2,9, p = 0,018. Ancak sinyal **eşiğe duyarlı**: eşik 0 / 1 / 5 / 8 alınınca kazanç +4,5 / +3,9 / +4,6 / +4,0 oluyor, p 0,37 ve 0,54'e çıkıyor (eşik 0 ve 1'de tabakalı permütasyon kurulamadı, çünkü havuz yetmedi). Sonuçtan sonra seçilmiş bir eşiğe bağlı tek bir p değeri bulgu sayılmaz. Bu 25 görüntünün içinde df ya da vasc da **yok**, dolayısıyla dengeli eşik aşımıyla ilgisi yok. ACC'ye olası katkısı en çok ≈ 0,19 puan ((10,4 − 2,9) × 25 / 1.003).

**S/N dağılımı.** ΔACC, lezyonu train'de olan val görüntülerinde (S) +3,22, olmayanlarda (N) +1,68 puan. Ancak molemax-dışı alt kümede S +3,07, N +3,43 puan, yani fark molemax bileşiminden geliyor. Tabakalı S − N +6,65'ten +5,90'a indi (Δ −0,75 [−4,40; +2,78]). **Kazanç, lezyonu train'de görülmüş val görüntülerinde yoğunlaşmıyor.** Lezyon ezberinin büyümesi beklenen imzaydı, görülmedi.

## (b) Seçim iyimserliği — **TEMİZ (seçim kuralından bağımsız)**

| Kol (H08 − H02, ikisi de aynı kural) | ΔACC [GA] | Δdengeli [GA] | P*(Δdengeli ≤ 6,5) |
|---|---|---|---|
| Seçilen epoch (val ACC; birincil) | +2,21 [+1,18; +3,27] | **+8,06 [+4,97; +11,79]** | 0,185 |
| **Son epoch (30; seçimsiz)** | +2,43 [+1,38; +3,55] | **+8,59 [+5,44; +12,03]** | 0,107 |
| Val-AUC seçimi (rakip kuralı) | +2,07 [+1,00; +3,17] | +7,84 [+4,55; +11,44] | 0,221 |

- H08'in epoch başına dengeli ACC'si (`gecmis.json`) şöyle: seçilen epoch'ta 0,787, ep 21–30 ortalaması 0,776, son epoch 0,781. Seçilen epoch'un dengeli değeri seed içinde %83–100'lük dilimde. Bunun nedeni ACC'nin dengeli ACC ile ilişkili olması. Seçimin kazandırdığı ≈ +0,6–1,0 puan. H02'de aynı fark (seçilen 0,706 − son 0,695) +1,2 puan. İyimserlik iki kolda simetrik ve fark üzerinde etkisiz.
- Seçimsiz ölçüde de durum değişmiyor: H08'in ep 21–30 ortalaması (0,776) ile H02'nin son epoch'u (0,695) arasındaki fark +8,2 puan. **Eşik aşımı seçim kuralından gelmiyor.** Son epoch kolunda aşım daha da büyük.

## (c) Kazancın kaynağı — **nadir sınıf ayrıklığı (df/vasc), ama kazanç tüm azınlık sınıflarında aynı yönde**

**Sınıf ayrışımı** (Δdengeli = (1/7) Σ Δduyarlılık; görüntü birimi = seed başına ortalama ek doğru):

| Sınıf | val desteği | train desteği | H02 doğru (seed) | H08 doğru (seed) | Δ görüntü | Δ duyarlılık (puan) | **Δdengeli'ye katkı (puan)** |
|---|---|---|---|---|---|---|---|
| akiec | 33 | 228 | 21/19/19/17/20 | 23/18/23/21/23 | +2,4 | +7,27 | 1,04 |
| bcc | 52 | 359 | 47/45/49/47/47 | 48/50/51/49/51 | +2,8 | +5,38 | 0,77 |
| bkl | 110 | 769 | 75/76/79/84/75 | 86/83/88/83/83 | +6,8 | +6,18 | 0,88 |
| **df** | **12** | 80 | 6/7/5/7/7 | 8/8/9/5/10 | **+1,6** | +13,33 | **1,90** |
| mel | 111 | 779 | 72/68/65/74/64 | 76/83/73/79/73 | +8,2 | +7,39 | 1,06 |
| nv | 671 | 4.693 | 641/647/648/629/644 | 640/642/640/638/639 | −2,0 | −0,30 | −0,04 |
| **vasc** | **14** | 99 | 9/9/8/10/9 | 11/12/12/11/11 | **+2,4** | +17,14 | **2,45** |
| **Toplam** | 1.003 | 7.007 | | | | | **8,06** |

- **df + vasc katkısı 4,35 puan (%54).** Bu katkı 26 val görüntüsünden (val'in %2,6'sı), seed başına ortalama **4,0 ek doğru görüntüden** geliyor.
- **df/vasc hariç 5 sınıflı dengeli Δ: +5,19 [+2,91; +7,65]**, P*(≤ 6,5) = 0,86. Karar vermez, yalnız ayrıştırma amaçlı.
- **En etkili k görüntü çıkarılınca** (seed ortalamalı görüntü katkısına göre): k = 1 (df) 7,19; **k = 2 (df + vasc) 6,50, yani eşiğin tam kendisi**; k = 3 5,93; k = 5 4,79; k = 10 3,03. Kazancın yarısını 7 görüntü açıklıyor. 121 görüntü pozitif, 67 negatif katkı veriyor.
- **Seed tutarlılığı.** Seed başına dengeli Δ +7,48 / +7,93 / +13,15 / +1,63 / +10,10 (sd 4,24). Bunun df + vasc katkısı +4,42 / +4,25 / +8,84 / −1,36 / +5,61. **Seed oynaklığının çoğu df/vasc'tan geliyor.** s3'te (Δ +1,63) df'de 7 → 5 görüntüye düşüş var. ACC Δ ise 5/5 seed'de +1,79…+2,49 aralığında (sd_Δ 0,28) ve kararlı.
- **Çoğunluk oyu (≥ 3/5 seed) dönüşleri.** H02'de yanlış olup H08'de doğru olanlar 37 görüntü: akiec 4, bcc 3, bkl 9, df 1, mel 9, nv 8, vasc 3. Ters yönde 16 görüntü (12'si nv). Kazanç tek bir sınıfta değil, **tüm azınlık sınıflarında aynı yönde.** mel, bcc ve bkl'de duyarlılık GA'sı 0'ın üstünde (kart Sonuç bölümü).

## (d) Eşiğin kendisi — kaynağı farklı bir manipülasyon ailesine ait; val'de yüksek varyans (YORUM; eşik GEVŞETİLMEDİ)

1. **Kaynak zinciri.** H08 kartında eşik şöyle gerekçelendirilmiş: "+6,5 (H05 ile aynı sınır; eğitim değişikliğiyle Z2'de τ = 1 zarfının (+5,4) üstü beklenmez)". H05 kartında ise kaynak "Cui'nin iyimser +4,46'sı ve Zhang'ın +4,2'si × ≈ 1,5". İki kaynak da **sınıf yeniden ağırlıklandırma** literatürü (Z2). H08 ise geometrik augmentasyon (A3, Z4). Yani eşik, başka bir manipülasyon ailesinin büyüklüğünden aktarılmış.
2. **τ zarfı argümanı H08'e uygulanamıyor.** τ zarfı yalnız sabit skorlarla yapılan önsel kaydırmanın sınırını verir. H02'nin seçimsiz τ eğrisi (tam val, τ 0…1, 0,05 adım) en çok dengeli **0,7605** (τ = 1, ACC 0,8445) ve en çok ACC **0,8672** (τ = 0) veriyor. **H08 noktası (ACC 0,8893, dengeli 0,7868) bu eğrinin tamamen dışında.** Eğrinin hiçbir noktası H08'in ACC'sine ulaşmıyor. Kazanç bir önsel kayması değil; skorların sıralaması iyileşmiş. Bire-karşı-hepsi AUC 6/7 sınıfta yükseliyor: akiec +1,56, mel +1,37, bkl +1,28, df +1,25, nv +0,95, bcc +0,47 puan (vasc −0,72, n = 14). nv payı yalnız −2,05 puan değişiyor. H04 katmanı H08'in üstüne dengeliyi ayrıca **+5,51 [+2,46; +8,84]** artırıyor. Bu da iki etkinin toplanabilir olduğunu, yani aynı mekanizma olmadığını gösteriyor.
3. **Val'de varyans.** Δdengeli'nin görüntü bootstrap sd'si 1,72 puan. Seed bileşeninin standart hatası 4,24 / √5 = 1,90 puan. Eşik ile nokta tahmini arasındaki fark (1,56 puan) bu iki belirsizliğin her birinden küçük. **P*(Δdengeli ≤ 6,5) = 0,185.** Kartın kuralı nokta tahminine bakıyor, dolayısıyla İNCELE mekanik olarak doğru. Ama "eşik aşıldı" ifadesi istatistiksel olarak kurulmuş bir aşım anlamına gelmiyor. Aşımın kendisi 2 görüntüye bağlı (k = 2 → 6,50).
4. **Aileye uygun bir çıpa adayı (bulundu, eşiği DEĞİŞTİRMEZ).** Tupper & Gagné 2024, "Analyzing Data Augmentation for Medical Images: A Case Study in Ultrasound Images", arXiv:2403.09828 (hakemsiz; erişim 2026-10-11, Tablo 1). Kurulum: Torchvision ResNet-18 ince ayarı, 5×2 çapraz doğrulama, metrik dengeli doğruluk. Tekil augmentasyonun augmentasyonsuz ayara göre etkisi: **döndürme ±30°: +5,4 (BUSI), +5,1 (BUS-BRA patoloji), +5,3 (BUS-BRA BI-RADS)**; yatay çevirme +2,0 / +1,7 / +2,4; dikey çevirme +1,3 / +0,7 / +2,0 puan. Bu kaynak manipülasyon ailesi (geometrik augmentasyon), metrik (dengeli doğruluk) ve model (R18 ince ayarı) bakımından Cui/Zhang'dan daha uygun. Farklı modalitedir (meme ultrasonu) ve **H08 sonucu görüldükten sonra bulunmuştur (M1 beyanı).** Bu yüzden H08'in eşiğine geriye dönük uygulanmaz. Hangi çarpanla kullanılacağı da (×1,5 → 8,1; ×2 → 10,8) keyfi bir seçimdir. Çıpa ilkesine (`sonuc-tahlili.md` §2(b)) göre bu kaynak **yeni kartların** dengeli eşiği için kullanılabilir. H08 için kullanılması ancak kullanıcı kararıyla olur (Seçenek B).

## (e) Mekanizma makul mü — **evet, iç ölçümlerle tutarlı; literatür yön kaynağı var**

- **Manipülasyon denetimi.** Train kaybı s0'da ep 5'te 0,012'den 0,257'ye çıkıyor (≈ 21×; beş seed 0,253–0,284), ep 30'da 0,00056'dan 0,0110'a. Ezber gecikiyor. Seçilen epoch'lar daha geç (ort. 24,8, H02'de 17,4). ECE 0,0745'ten 0,0618'e, NLL 0,518'den 0,416'ya iniyor.
- **Nadir sınıf aritmetiği.** Train'de df 80, vasc 99, akiec 228 görüntü var. D4, bu görüntülerin her birinin 8 elemanlı yörüngesini sunuyor (df için 640 farklı görünüm). Ezberin en hızlı olduğu yer en az örnekli sınıflar olduğundan, ezberin gecikmesinin en çok buralara yaraması beklenir. Ölçülen sıralama bununla uyumlu: duyarlılık artışı vasc +17, df +13, akiec +7, mel +7, bkl +6, bcc +5, nv −0,3. Ancak df/vasc GA'ları geniş (df [0; +29], vasc [+4; +33]).
- **Literatür.** Perez vd. 2018 (arXiv:1809.01442; ISIC 2017) yön kaynağı: geometrik ve birleşik augmentasyon dermoskopide ImageNet CNN'lerini iyileştiriyor. Tupper & Gagné 2024 (yukarıda) tekil geometrik augmentasyonun R18 ince ayarında dengeli doğruluğu +2…+5 puan artırdığını gösteriyor. Balestriero, Bottou & LeCun 2022 ("The Effects of Regularization and Data Augmentation are Class Dependent", NeurIPS 2022, arXiv:2204.03632) augmentasyon etkisinin sınıfa göre çok farklı olabileceğini gösteriyor. Bu, ortalamanın sınıf düzeyindeki etkiyi gizleyebileceği anlamına geliyor. Bizde sınıf etkileri aynı yönde (nv hariç ≈ 0). **DermaMNIST'e özgü izole bir D4 etkisi literatürde bulunamadı** (`literatur-cipa.md` §4(d) hâlâ geçerli).
- **H07 ile çapraz tutarlılık.** Augmentasyonsuz eğitilmiş H02'de D4 görünümleri ≈ 2,6 puan dağılım dışı kalıyor ve TTA kazandırmıyor (H07). Aynı dönüşümlerle eğitim ise kazandırıyor (H08). Yani model döndürülmüş görünümleri öğrenebiliyor; anizotropik ölçekleme riski (H07'de D2 − 90° = +0,94) eğitimde kazancı engellemedi. Ablasyon (i), yani yalnız D2 ile eğitim, hâlâ açık bir sorudur.

---

## Sonuç: **(i) Teşhis temiz.** Arıza bulunmadı, kazanç gerçek. Eşik aşımı, nadir sınıf örnekleme oynaklığından ve başka bir aileden aktarılmış eşik çıpasından geliyor.

| Hipotez (Mod 1b listesi) | Test | Sonuç |
|---|---|---|
| Kurulum/sürüm | Özdeşlik dumanı: `ilk_adim_loss` H02 ile bit bit aynı (fark 0). Açık duman fark 0,0074 | yok |
| Veri yolu/split | Ayrı npz anahtarları; augment yalnız train'de; n_train 7.007; D4 yakın kopya 0 | yok |
| Seed bağlanmamış | 5 seed farklı (ACC 0,8833–0,8933); `g_aug` ayrı üreteç, batch sırası H02 ile aynı | yok |
| Metrik hesabı | npy = metrics = gecmis (5/5); dengeli GA bağımsız betikte birebir | yok |
| Normalizasyon sızıntısı | Val `hazirla()` yolu değişmemiş; BN istatistikleri yalnız train'den | yok |
| Seçim iyimserliği | Son epoch kolu +8,59 (daha büyük); seçim etkisi simetrik | yok |
| Lezyon ezberi / D4 hizalama | S ≈ N (molemax-dışı), tabakalı S − N ↓; D4 hizalı alt küme sinyali eşiğe duyarlı ve df/vasc içermiyor | kanıt yok (keşifsel not) |
| Beklenti yanlış kaynaklı | Eşik Z2 (yeniden ağırlıklandırma) literatüründen aktarılmış. Aileye uygun çıpa adayı: Tupper & Gagné 2024 (+5,1…+5,4, tekil geometrik) | **kısmen evet**, ama çıpa sonuçtan sonra bulundu → geriye dönük uygulanmaz |

**Kartın kuralı (M6) sonucu değiştirmiyor: bu kartla TUT verilemez.** Önerilen yollar ve maliyetleri `tahlil-tur3.md` §3.3'te. Kısa özet:
- **A (önerilen, ön-kayda en sadık, 0 GPU).** H08 "İNCELE kapandı: arıza yok; M6 gereği TUT verilmedi" olarak kalır ve **K5'te ön-kayıtlı test doğrulama adayı** olur. Tur-4 ebeveyni mekanik olarak H02 olur. Durma sayacı 1/2.
- **B (yalnız kullanıcı kararı, 0 GPU).** "Beklenti yanlış kaynaklı" düzeltmesi: dengeli eşiğin çıpası Tupper & Gagné 2024'e taşınır. H08, soruşturma temiz çıktığı ve kullanıcı karar verdiği için TUT alır (günlükte M6 istisnası olarak, makalede beyanla). Tur-4 ebeveyni H08 (D4), sayaç 0/2.
- **C (önerilmez; ≈ 2,2 GPU-sa eşleşmesiz, ≈ 4,5 eşleşmeli).** H08b: seed {5..9} tekrarı. Aynı 1.003 val görüntüsünü kullandığı için df/vasc örnekleme oynaklığını çözmez. Eşik aynı kalırsa yeni 5 seed'in ortalaması ≈ %80 olasılıkla yine > 6,5 çıkar ve kart yine İNCELE'ye düşer. Karar değeri düşük. Tek yan faydası Tur-5 için ayrık seed kazandırması.
