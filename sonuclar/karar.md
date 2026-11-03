# K5 KARAR TASLAĞI — 3.1 DermaMNIST · ResNet-18 (28 px) → DINOv2

*deney-tahlilci, Mod 3 (K5) · 2026-10-12 · **TASLAKTIR: çıkışı kullanıcı seçer** (K5-ONKAYIT §7 "Seçim").*
*Yeni koşum yapılmadı. Test yeniden açılmadı. Bütün sayılar dondurulmuş çıktılardan okundu. "(türetilmiş)" etiketli değerler aynı dosyalardaki sayılardan elle hesaplandı; GA'ları yoktur.*

**Kaynak kısaltmaları**

| Kısaltma | Dosya |
|---|---|
| [T] | `sonuclar/k5_test_sonuc.md` + `.json` (kip test, 2026-10-11 21:12, SHA256 `e7ae79c3…`; `k5_test.KILIT` `ikinci_acilis: null`) |
| [C1] | `sonuclar/k5_c1_sonuc.md` + `.json` (kip c1, 2026-10-12 02:11) |
| [KV] | `sonuclar/k5_kuru_val_sonuc.md` (Colab val, test açılmadan) |
| [ON] / [ONb] | `hipotezler/K5-ONKAYIT.md` / `K5-ONKAYIT-b.md` |
| [SIR] | `hipotezler/SIRALAMA.md` "K5 ÖN-KAYDI — H08 TEST DOĞRULAMA KURALI" (09:42) |
| [İH] | `sonuclar/inceleme-h08.md` (val İNCELE) |
| [T4h] | `sonuclar/tahlil-tur4.md` |
| [LC] | `notlar/literatur-cipa.md` |
| [DM] | `data/data_manifest.md` |
| [R] / [RC] | `sonuclar/replikasyon.md` / `replikasyon-colab.md` |
| [G] | `gunluk/deney-gunlugu.md` |
| [OD] | `orkestrasyon_durumu.md` |
| [P6] | `<DRIVE_KOK>\3-1\sonuclar\beklenti-k5-test-asama1.md` ve `-asama3.md` (yerel WS'de yok, bkz. §8) |

---

## 0. Özet (tek ekran)

| Ölçüt (§7 [ON]) | Değer | Kaynak | Durum |
|---|---|---|---|
| W | H10. H13 Tur-5'te ÖLDÜR aldığı için W = H10 (val'de seçildi; test W'yi değiştirmez) | [T] §7, [G] Tur-5 | — |
| **G1**: W − baseline_c, resmî test | **+15,45 [+13,84; +17,01]**; Holm t 1,29e-06, Holm boot 0,002; marj ≥ +2,0 | [T] Aile 1 | ✔ |
| **G2**: görülmemiş bayrak yok ya da İNCELE "arıza yok" | H10: 3/3 bayrak FALSE. H08: `dengeli_delta>6,5` = **TRUE** (+8,41). Teşhis §2'de: **arıza yok, açıklanabilir** | [T] §2.1–2.2 | ✔ (kullanıcı teşhisi onaylarsa) |
| **G3**: W_C1 − baseline_C1, C-test | **+8,26 [+6,63; +9,94]**; Holm t 3e-05; marj ≥ +2,0 | [C1] aile_C1_1 | ✔ |
| **V**: H10 − H02c, resmî test | **+2,26 [+1,15; +3,30]**; Holm t 1,07e-04, Holm boot 0,001 → "anlamlı" | [T] Aile 2 | ✔ |
| **L**: H10 − H02, C1 | **+1,12 [−0,05; +2,30]** → ayırt edilemedi | [C1] (i) | ✘ |

→ **Mekanik çıkış: A. GEÇTİ, kademe A-kayıtlı (V ∧ ¬L).** B'nin ön-kayıtlı koşulu (¬G1 ∨ ¬G3) sağlanmıyor. C'nin koşulu da sağlanmıyor: P6 özdeşliği ve baseline özdeşliği geçti, replikasyon iki kez YEŞİL.

**Kırmızı bayrak sayısı: 2.**
1. H08 dengeli Δ > 6,5. Ön-kayıtlı bayrak; teşhis edildi, arıza yok.
2. T4 AUC 0,9904 > 0,99. Ön-kayıtta T4 için tanımlı değil; bunu [LC] §4(e) genel eşiğine göre tahlilci işaretledi. T4 iddia taşırsa denetim gerekir.

---

## 1. İstatistik (D2) — resmî test, val → test, bütünlük

### 1.1 Resmî test (n = 2.005, 5 seed, bir kez açıldı) [T]

| Kol | ACC (ort ± std) | AUC | dengeli | dengeli-5 | ECE | NLL |
|---|---|---|---|---|---|---|
| baseline_c (R18, 28 px, sıfırdan; val-AUC seçimi) | 0,7395 ± 0,0053 | 0,9148 | 0,4353 | 0,4919 | 0,0380 | 0,7027 |
| H01c (DINOv2-B donuk + LR sonda, 224) | 0,8260 ± 0,0004 | 0,9584 | 0,6382 | 0,6658 | 0,0265 | 0,4889 |
| H02c (R18-ImageNet, 224, tam ince ayar) | 0,8713 ± 0,0041 | 0,9736 | 0,7200 | 0,7527 | 0,0693 | 0,4616 |
| **H08c** (H02 + eğitimde D4) | **0,8976 ± 0,0027** | 0,9830 | 0,8041 | 0,8049 | 0,0530 | 0,3660 |
| **H10 = W** (DINOv2-B son 4 blok, 224) | **0,8940 ± 0,0033** | 0,9828 | 0,7803 | 0,7768 | 0,0621 | 0,4114 |
| H11 (DINOv2-S tam ince ayar) | 0,8837 ± 0,0083 | 0,9807 | 0,7703 | 0,7767 | 0,0758 | 0,4906 |
| H12 (DINOv2-S sonda) | 0,8160 ± 0,0005 | 0,9501 | 0,6456 | 0,6339 | 0,0196 | 0,5297 |
| baseline_c (son epoch, ikincil) | 0,7364 ± 0,0060 | 0,9024 | 0,4823 | 0,5086 | 0,2146 | 1,8606 |
| T4 = H08c ⊕ H10 (ek satır, betimsel) | 0,9169 ± 0,0027 | 0,9904 | 0,8296 | 0,8278 | 0,0419 | 0,2780 |

**Eşleştirilmiş testler (Holm, tek yönlü; GA = görüntü bootstrap'ı %95, B = 2.000):**

| Kıyas | ΔACC (puan) [GA] | Holm t | Holm boot | Kural |
|---|---|---|---|---|
| H10 − baseline_c | +15,45 [+13,84; +17,01] | 1,29e-06 | 0,002 | GEÇTİ (aile 1) |
| H08c − baseline_c | +15,81 [+14,21; +17,29] | 1,09e-06 | 0,002 | GEÇTİ |
| H02c − baseline_c | +13,19 [+11,72; +14,67] | 2,89e-06 | 0,002 | GEÇTİ |
| H01c − baseline_c | +8,66 [+6,96; +10,42] | 2,89e-06 | 0,002 | GEÇTİ |
| H10 − H02c | +2,26 [+1,15; +3,30] | 1,07e-04 | 0,001 | DOĞRULANDI, anlamlı (aile 2) |
| H08c − H02c | +2,62 [+1,85; +3,36] | 1,91e-05 | 0,001 | DOĞRULANDI, anlamlı (aile 2) |
| H10 − H08c (betimsel, Holm dışı) | −0,36 [−1,40; +0,70] | — (p_t 0,948) | — | ayırt edilemedi |

### 1.2 Val → test kayması (aşırı ayar işareti) — [KV] Colab val ↔ [T] test (türetilmiş)

| Kol | val ACC | test ACC | test − val (puan) |
|---|---|---|---|
| baseline_c | 0,7458 | 0,7395 | −0,63 |
| H01c | 0,8223 | 0,8260 | +0,37 |
| H02c | 0,8656 | 0,8713 | +0,57 |
| H08c | 0,8865 | 0,8976 | +1,11 |
| **H10 (val'de seçilen W)** | **0,8947** | **0,8940** | **−0,07** |
| H11 | 0,8800 | 0,8837 | +0,37 |
| H12 | 0,8187 | 0,8160 | −0,27 |
| T4 | 0,9095 | 0,9169 | +0,74 |
| Δ H10 − H02c | +2,91 | +2,26 | −0,65 (ön-kayıtlı beklenti [+1,0; +4,5] içinde [T]) |
| Δ H08c − H02c | +2,09 | +2,62 | +0,53 |

**Okuma:** W val'den test'e 0,07 puan düşüyor. Bu, seed std'sinin (0,33) altında. Kazanan seçiminde iyimserlik (winner's curse) ölçülebilir değil. Val'de "H10 > H08c" sıralaması (+0,82) test'te −0,36'ya dönüyor; iki fark da GA içinde. Bu dönüş bir aşırı ayar işareti değil, sıralamanın kararsız olduğunu gösteriyor.

### 1.3 Bütünlük (C çıkışını dışlayan kanıt)
- **P6 özdeşliği:** 7 kol × 5 seed. a1'de (val) ve a3'te (test geçişi) `ozdeslik_argmax_oran = 1`, `ozdeslik_p_maks_fark = 0` [P6]. Bu iki dosya yalnız Drive'da; yerel arşive kopyalanmalı (§8).
- **Baseline özdeşliği:** test ACC 0,7395 ± 0,0053 [T]. Seed değerleri (0,7436 / 0,7317 / 0,7411 / 0,7441 / 0,7367) kc1'de dondurulan değerlerle aynı [RC].
  - Not: yerel son-epoch değeri de 0,7395 ± 0,0076 [R]. Ortalamalar tesadüfen aynı; seed değerleri ve std'ler farklı. Bu bir kopyalama hatası değildir.
- **Test kilidi:** tek açılış 21:12; `ikinci_acilis: null`. C1 C-test kol başına bir kez açıldı [OD].
- **Rakibin seçim kuralı:** son-epoch satırı ile resmî val-AUC satırı arasındaki fark −0,31 [−1,28; +0,73] [T]. Rakip seçim kuralı ACC sonucunu değiştirmiyor.

---

## 2. G2 — kırmızı bayrak teşhisleri

### 2.1 H08 `dengeli_delta>6,5` = TRUE → **"görülmemiş kadar iyi" değil; açıklanabilir, arıza yok**

**Belirti.** Resmî test'te H08c − H02c dengeli ACC farkı **+8,41 [+5,85; +11,22]** [T]. Ön-kayıtlı eşik +6,5 ([SIR] madde 2, [ON] §2.1). ACC, ACC > 0,90 ve AUC bayrakları FALSE: ΔACC +2,62 < 3; ACC 0,8976; AUC 0,9830.

| Hipotez (Mod 1b) | Ölçülebilir test (dondurulmuş çıktılardan) | Sonuç |
|---|---|---|
| **Test'e özgü sızıntı / hizalama** | Val → test tutarlılığı. Val İNCELE'de +8,06 [+4,97; +11,79] [İH], Colab val'de +6,63 (türetilmiş: 0,7763 − 0,7100 [KV]), test'te +8,41. Test'e özgü bir arıza test'te bir sıçrama üretirdi; sıçrama yok. P6 özdeşliği 35/35 geçti [P6]. | yok |
| **Lezyon / D4 yakın-kopya sızıntısı** | Fark testi: lezyon-ayrık alt kümede bayrak sürüyor mu? Yeniden eğitimsiz C0'da dengeli Δ **+8,26 [+4,38; +12,78]** [T], yeniden eğitimli C1'de **+9,15 [+5,64; +12,78]** [C1]. Eğitimde lezyonu bulunmayan görüntülerde de aynı büyüklük ölçülüyor, dolayısıyla sızıntı açıklaması dışlanıyor. Val'de 28 px D4 piksel taramasında RMSE ≤ 5 olan D4'e özgü kopya sayısı 0 [İH](a). | yok |
| **Seçim iyimserliği** | Val'de son epoch kolu +8,59; val-AUC seçimi +7,84 [İH](b). Seed başına test dengeli Δ: 8,08 / 8,55 / 8,65 / 8,06 / 8,72 [T json `t_dengeli`]. Beş seed'in hepsi eşiğin üstünde ve dar bir bantta. | yok |
| **Metrik hesabı** | Δdengeli = (1/7) Σ Δduyarlılık (türetilmiş, [T json `duyarlilik_*`]): akiec +8,48, bcc +3,88, bkl +5,27, **df +21,74**, mel +8,07, nv +0,39, **vasc +11,03**. Toplam 58,86 / 7 = 8,41, kayıtlı değerle aynı. | yok |
| **Nadir sınıf örnekleme oynaklığı** | Test'te df 23, vasc 29 görüntü var [DM]. df + vasc katkısı (21,74 + 11,03) / 7 = **4,68 puan, toplamın %56'sı** (val'de %54 [İH]). Bu katkı seed başına ≈ 5,0 (df) + 3,2 (vasc) = **≈ 8 ek doğru görüntüden** geliyor (türetilmiş). **df/vasc hariç 5 sınıflı dengeli Δ +5,22 [+3,46; +6,88]**, nokta değeri eşiğin altında (val'de +5,19 [İH]). Kazanç nv dışındaki bütün sınıflarda aynı yönde. | **ana açıklama** |
| **Beklenti yanlış kaynaklı** | Eşik, Z2 (sınıf yeniden ağırlıklandırma) literatüründen aktarıldı [İH](d). Aileye uygun çıpa (Tupper & Gagné 2024, tekil geometrik augmentasyon, dengeli doğruluk +5,1…+5,4) sonuç görüldükten sonra bulundu. Bu yüzden **eşik gevşetilmez** (çıpa ilkesi). Literatür karşılaştırması: Zhang & Li 2026 Tablo 1d'de R18-ImageNet-224 makro-ACC değerleri CE 0,758 / WS 0,781 [LC] §4(c). H08c'nin 0,8041'i bunun 2,3 puan üstünde; bu "görülmemiş" düzeyde değil. | kısmen (eşik aile dışı) |
| **Mekanizma** | Val'de D4 ile eğitim kaybı ≈ 21 kat arttı, yani ezber gecikti. Örnek sayısı en az olan sınıflar en çok kazandı [İH](e). ECE / NLL test'te de iyileşiyor: 0,0693 → 0,0530 ve 0,4616 → 0,3660 [T]. | tutarlı |

**Teşhis kararı (öneri):** Bayrak **mekanik olarak doğru** ve makalede beyan edilmeli. **Arıza yok.** Aşım iki etkenden geliyor: df ve vasc'ta seed başına ≈ 8 görüntülük nadir sınıf kazancı ve aile dışından aktarılmış bir eşik. Bayrak val'de (M1) görülmüş, ön-kayıtta "rapor bayrağı" olarak bırakılmıştı. Test değeri val ile tutarlı. → **G2: "aşıldı, İNCELE 'arıza yok' diye kapandı" koşulu sağlanıyor** (val İNCELE: [İH]; test teşhisi: bu bölüm). **Kullanıcı onayı gerekir.**

**Makaleye zorunlu cümle (taslak):** "H08'in dengeli doğruluk kazancı (+8,4 puan) ön-kayıtlı +6,5 puanlık 'beklenmedik büyüklük' eşiğini aştı. Kazancın %56'sı test'te 23 df ve 29 vasc görüntüsünden geliyor (seed başına ≈ 8 ek doğru). Bu iki sınıf hariç tutulduğunda fark +5,2 puandır. Kazanç lezyon-ayrık değerlendirmelerde de sürüyor (C0 +8,3; C1 +9,2)."

**Açık kalan, düşük değerli denetim:** train × test × 8 D4 piksel yakın-kopya taraması yalnız val için yapıldı. C0/C1 fark testi bu açıklamayı zaten dışlıyor. İstenirse CPU'da ve tahmin dosyası açmadan yapılabilir (yalnız piksel verisi).

**Yan not (bayrak değil):** H08c seed 0 test ACC değeri 0,9007 > 0,90 [T json]. Ön-kayıtlı kural 5 seed ortalamasına bakıyor (0,8976), dolayısıyla bayrak FALSE. [LC] §4(a)'da "Native 224 ResNet-18 > 0,90" görülmemiş eşiğine 0,24 puan uzaklıkta.

### 2.2 T4 AUC 0,9904 > 0,99 → tahlilci bayrağı (ön-kayıtta T4 için tanımlı değil)
- **Belirti:** T4 test AUC 0,9904; seed başına 0,9893 / 0,9907 / 0,9911 / 0,9902 / 0,9904, yani 4/5 seed > 0,99 [T json]. [LC] §4(e)'ye göre native 224'te "> 0,99 AUC → sızıntı denetimi".
- **Açıklamayı destekleyenler:**
  - Bileşenler (H08c 0,9830, H10 0,9828) P6 özdeşliğinden geçti.
  - Topluluk kuralı test'ten önce donmuştu: ½ ; ½ log-olasılık ([ON] §1 (8'); H13 kartı).
  - Val'de T4 AUC zaten 0,9895'ti [KV]. Test'te sıçrama yok (+0,09).
  - Literatürdeki tek model DINOv2-B uçtan uca: 98,92 ± 0,11 [LC] §3. T4 bundan 0,12 puan yukarıda ve iki farklı ailenin topluluğu.
- **Açık:** T4'ün lezyon-ayrık AUC'si hesaplanmadı. C0 tablosunda yalnız ACC/dengeli var: C0 ACC 0,9335 [T]. T4 C1'de yok (ön-kayıtlı değil).
- **Öneri:** T4 **ek tabloda betimsel** kalsın, tez veya özette iddia taşımasın. Kullanıcı T4'e iddia yüklemek isterse önce şu yapılmalı: T4 AUC'si **C0 alt kümesinde ve N_test'te**, dondurulmuş tahminlerden ve CPU'da, "test sonrası keşifsel" etiketiyle hesaplanır ([ON] §6.3). Bu denetim yeni koşum değildir, ama test tahminlerine yeniden bakmaktır. Bu yüzden kullanıcı kararı gerekir.

---

## 3. Kazanan / iddia edilebilir kol: H10 mı, H08 mi, ikisi mi?

| Kanıt | H10 − H08c (puan) [GA] | Kaynak |
|---|---|---|
| Val (Colab) | +0,82 [−0,76; +2,43] ayırt edilemedi | [T4h] §5 |
| Resmî test ACC | **−0,36 [−1,40; +0,70]** ayırt edilemedi | [T] |
| Resmî test dengeli / dengeli-5 | −2,38 [−5,50; +0,58] / **−2,81 [−5,45; −0,23]** | [T json] |
| Resmî test S / N | S +1,15 [−0,85; +3,26]; **N −1,16 [−2,33; −0,02]** | [T json] |
| C0 (C-test, yeniden eğitimsiz) | **−1,22 [−2,36; −0,08] → tersi** (betimsel) | [T] |
| C1 (C bölmesi, yeniden eğitim) | −0,96 [−2,10; +0,18] ayırt edilemedi | [C1] |
| Kalibrasyon (ham / R-T) | ECE 0,0621 ↔ 0,0530; R-T 0,0179 ↔ 0,0168; NLL 0,4114 ↔ 0,3660 | [T] |

**Tahlil:**
- **W = H10 değişmez.** "Test sonucu kazananı DEĞİŞTİRMEZ" ([ON] §1 ve §6.3). Makale H10'u val'de seçilmiş kazanan olarak raporlar.
- **H08c, makalenin doğrulanmış R18 satırıdır.** [SIR] madde 1 şöyle: "H08 doğrulanırsa makaledeki R18 kazanan satırı H08". Madde 3 de geçti: C1'de +2,09 [+1,24; +2,97], yani "kazanç lezyon ezberinden değil" desteklendi.
- **İddia edilebilir yapı "ikisi birlikte, ayrı rollerle" olmalı:**
  - H10: DINOv2 ince ayarı. Rakibe karşı G1/G3, eşit önağırlıklı R18'e karşı V (yalnız resmî bölmede).
  - H08c: güçlü R18 temel çizgisi. Her üç değerlendirmede H02c'yi geçiyor.
  - **H10 ile H08c arasında üstünlük iddiası yok.** Resmî test'te ve C1'de ayırt edilemiyorlar. Lezyon-ayrık alt kümede (C0) ve N_test'te H08c önde. Dengeli-5'te de H08c önde.
- **Karıştırıcı:** H10 augmentasyonsuz eğitildi (önbellek tek görünüm [T4h] §3 madde 9 ve §8.6 madde 5). H10 ↔ H08c kıyası "omurga" ile "augmentasyon"u karıştırır. H10 + D4 koşulmadı. Test sonrası yeni kol eklenemeyeceği için ([ON] §6.3) bu bir sınırlılıktır.

**T4 (H08c ⊕ H10):**
- **Ön-kayıtlı mı?** Evet, ek satır olarak: "her durumda hesaplanır ve ek tabloda yazılır (D3). İddia taşıması yalnız H08 doğrulanırsa. H13 ÖLDÜR olsa da T4 kazanan olamaz" ([ON] §1 (8')). Val değeri (0,9095) test'ten önce görülmüştü [G] Tur-5.
- **İddiaya girebilir mi?** H08 doğrulandığı için koşul sağlandı, ama yalnız **betimsel** olarak:
  - T4 için ön-kayıtlı bir doğrulama kuralı ya da Holm ailesi yok.
  - C1 karşılığı yok.
  - Çıkarım maliyeti iki model kadar.
  - AUC bayrağı açık (§2.2).
- İzin verilen cümle (betimsel): "İki ailenin seed eşli topluluğu 0,9169 ± 0,0027 test ACC'ye ulaştı (T4 − H10 +2,29 [+1,61; +2,93]; T4 − H08c +1,94 [+1,21; +2,65]; çoklu karşılaştırma düzeltmesi yok; ön-kayıtlı ek satır) [T]."
- Yasak cümleler: "T4 önerilen yöntemdir" / "en iyi modelimiz 0,917".

---

## 4. Ablasyon zinciri B → H01 → H02 → H10 / H08

*Δ sütunları bitişik basamağa göredir. GA'lı değerler [T] / [C1] `kiyas` alanlarından alındı. "(n)" işareti, GA'sı olmayan nokta farkını (türetilmiş) gösterir.*

| Basamak | Ne değişti | Resmî test ACC | Δ (puan) [GA] | dengeli (test) | C0 ACC | C1 ACC | C1 Δ [GA] |
|---|---|---|---|---|---|---|---|
| B baseline_c | R18, 28 px, sıfırdan (rakip) | 0,7395 | — | 0,4353 | 0,8289 | 0,8334 ± 0,0054 | — |
| H01c | + 224 px, + DINOv2-B önağırlık, **donuk** sonda | 0,8260 | vs B +8,66 [+6,96; +10,42] | 0,6382 | 0,8735 | 0,8755 | vs B +4,21 [+2,41; +6,03] |
| H02c | R18-ImageNet, 224, **tam ince ayar** (eşit önağırlık hakkı) | 0,8713 | vs B +13,19 [+11,72; +14,67]; vs H01c +4,53 (n) | 0,7200 | 0,9102 | 0,9048 | vs B +7,14 [+5,75; +8,57] |
| **H10** | DINOv2-B, son 4 blok ince ayar | 0,8940 | **vs H02c +2,26 [+1,15; +3,30]**; vs H01c +6,80 (n) (val GA'lı +7,24 [+5,48; +9,11] [T4h] §5) | 0,7803 | 0,9130 (vs H02c +0,28 [−0,83; +1,47]) | 0,9161 | **vs H02 +1,12 [−0,05; +2,30]** |
| **H08c** | H02 + eğitimde D4 | 0,8976 | **vs H02c +2,62 [+1,85; +3,36]** | 0,8041 | 0,9252 (vs H02c +1,50 [+0,70; +2,28]) | 0,9257 | **vs H02 +2,09 [+1,24; +2,97]** |
| H10 ↔ H08c | omurga + augmentasyon (karışık) | — | −0,36 [−1,40; +0,70] | — | −1,22 [−2,36; −0,08] | — | −0,96 [−2,10; +0,18] |
| (Pareto) H12 | DINOv2-S donuk sonda | 0,8160 | vs H01c −1,01 [−2,60; +0,63] | 0,6456 | 0,8781 | — | — |
| (Pareto) H11 | DINOv2-S tam ince ayar | 0,8837 | vs H02c +1,24 [+0,42; +2,07]; vs H12 +6,77 (n) | 0,7703 | 0,9180 | — | — |
| (ek) T4 | H08c ⊕ H10 | 0,9169 | vs H10 +2,29 [+1,61; +2,93] | 0,8296 | 0,9335 | — | — |

**Okuma:**
1. **Kazancın büyük kısmı ilk basamakta: 224 px + önağırlık + ince ayar paketi** (H02c − B +13,19).
2. **Sonda → ince ayar adımı büyük** (H10 − H01c +6,8 (n); H11 − H12 +6,8 (n)). Val'de N alt kümesinde de var: H10 − H01c N +3,51 [+1,69; +5,38] [T4h] §4.
3. **Omurga adımı (R18 → DINOv2, ikisi de ince ayarlı) küçük ve bölmeye duyarlı:** resmî test +2,26 ✔, C0 +0,28, C1 +1,12, yani lezyon-ayrık bölmede ayırt edilemiyor.
4. **D4 adımı her üç değerlendirmede GA > 0:** +2,62 / +1,50 / +2,09.
5. **C1'deki göreli kazançlar resmî bölmedekilerden küçük.** Bunun bir nedeni C-test çoğunluk tabanının 0,7726 olması (resmî test 0,669 [DM]). C-test ACC değerleri resmî test ACC'leriyle doğrudan kıyaslanmaz ([ON] §3).

**Eksik ablasyonlar** (test sonrası eklenemez, sınırlılık):
- H10 + D4.
- Rakip + D4 (M3).
- H10 K = 2 ("kaç blok yeter").
- Yalnız D2 ile eğitim (H08 ablasyon i).
- H04 yalnız-nv ablasyonu.

---

## 5. Lezyon tekrarı: S/N ve kazanç çöküşü (A-kayıtlı tezin zorunlu eşlikçileri, [ON] §7)

| Ön-kayıtlı soru | Sonuç [GA] | Etiket | Kaynak |
|---|---|---|---|
| **T-SN1** H10 − H02c: Δ_S > Δ_N | Δ_S **+5,43 [+3,23; +7,47]**, Δ_N **+0,59 [−0,58; +1,76]**, ham Δ_S − Δ_N **+4,83 [+2,22; +7,18]**; net kazancın S payı **%83** | **desteklendi** | [T] |
| **T-SN2** H08c − H02c: Δ_S ≈ Δ_N | Δ_S +4,27 [+2,56; +5,89], Δ_N **+1,75 [+0,91; +2,60]**, ham fark +2,52 [+0,52; +4,32] (beklenen [−3; +4] içinde, ama GA 0'ı dışlıyor). İkincil tab-std fark +0,89 [−2,04; +3,65] | **"dengeli dağılım" ön-kaydı TUTMADI** (ham farkta S > N). Betiğin "desteklendi" etiketi bu kıyas için "Δ_S > Δ_N" anlamına geliyor; makalede "dengeli dağılım desteklendi" diye **yazılmaz** | [T] |
| C0-1 H10 − H02c (lezyon-ayrık alt küme) | +0,28 [−0,83; +1,47] | ayırt edilemedi | [T] |
| C0-2 H08c − H02c | +1,50 [+0,70; +2,28] | desteklendi | [T] |
| **KÇ0** K0_H10 > K0_H08 | K0_H10 +1,99 [+1,06; +2,94], K0_H08 +1,12 [+0,40; +1,80], fark **+0,86 [−0,01; +1,78]** | ayırt edilemedi | [T] |
| **KÇ** K_H10 > K_H08 (resmî → C1) | K_H10 +1,14 [+0,04; +2,31], K_H08 +0,54 [−0,39; +1,47], fark **+0,60 [−0,45; +1,68]**; bileşenler K1_H10 −0,85, K1_H08 −0,59 | ayırt edilemedi. **İfade eşiği FALSE**: oran 0,503 ≥ 0,5, ama KÇ desteklenmedi | [C1] |
| (iii) baseline C1 | 0,8334; beklenti [0,79; 0,85] içinde; > 0,87 değil; çoğunluk tabanının +6,08 puan üstünde. Abhishek vd. 2025 C npz 28 px 0,825 ± 0,006 [LC] §5 (piksel karıştırıcısı var, C2 koşulmadı) | uyumlu | [C1] |

**Okuma:**
- H10'un R18'e kazancı resmî test'te büyük ölçüde lezyonu eğitimde görülmüş görüntülerde. N_test'te 0,59 puan ve ayırt edilemiyor.
- H08'in kazancı N'de de GA > 0: +1,75.
- "H10'un kazancı H08'inkinden daha çok çöküyor" yönündeki nokta tahminleri ön-kayıtlı yönde (KÇ0 +0,86, KÇ +0,60), ama ikisi de **ayırt edilemedi**. Güç beyanı önceden yapılmıştı (MDE ≈ 3 puan [ON] §4.3).

---

## 6. Üç çıkış için kanıt tablosu

| Çıkış | Ön-kayıtlı koşul | Kanıt | Durum |
|---|---|---|---|
| **A. GEÇTİ** | G1 ∧ G2 ∧ G3 | G1 +15,45 [+13,84; +17,01]; G2 (§2.1 teşhisi); G3 +8,26 [+6,63; +9,94]; ablasyon tutarlı (§4); W val → test −0,07 | **SAĞLANIYOR → kademe A-kayıtlı** (V ✔, L ✘) |
| B. ANLAŞILDI-GEÇEMEDİ | G2 ∧ (¬G1 ∨ ¬G3) | G1 ve G3 doğru | **Mekanik olarak uygulanamaz.** "Protokol duyarlılığı" içeriği A-kayıtlı tezin içinde zorunlu olarak yer alıyor (S/N + KÇ) |
| C. TEKRARLANAMADI | Baseline ya da P6 özdeşliği sağlanamadı ve teşhis edilemedi | Replikasyon YEŞİL ×2 (yerel 0,7296 ± 0,0176 [R]; Colab 0,7395 ± 0,0053 [RC]); P6 35/35 [P6] | **Uygulanamaz** |

### 6.A — A-kayıtlı (ÖNERİLEN)

**Tez cümlesi taslağı (iki vurgu seçeneği; mekanik çıkış aynı, vurguyu kullanıcı seçer):**
- **(A1) İyileştirme vurgulu:** "224 px'te son dört bloğu ince ayarlanan DINOv2-B, DermaMNIST'in resmî 28 px ResNet-18 temel çizgisini hem resmî bölmede (+15,5 puan) hem lezyon-ayrık DermaMNIST-C bölmesinde (+8,3 puan) geçer. Aynı önağırlık hakkı verilmiş 224 px ResNet-18'e üstünlüğü (+2,3 puan) ise yalnız lezyon tekrarı içeren resmî bölmede doğrulanır. Lezyon-ayrık bölmede bu üstünlük ayırt edilemez (+1,1 puan, %95 GA [−0,05; +2,30])."
- **(A2) Protokol-duyarlılığı vurgulu (aynı kanıt):** "DermaMNIST'te temel model ince ayarının güçlü bir CNN temel çizgisine üstünlüğü bölme protokolüne duyarlıdır. DINOv2-B'nin ImageNet-ResNet-18'e +2,3 puanlık resmî bölme kazancının %83'ü, lezyonu eğitimde görülmüş test görüntülerinden gelir. Lezyon-ayrık bölmede fark ayırt edilemez. Basit dihedral augmentasyonlu ResNet-18 ise her üç değerlendirmede DINOv2-B ile ayırt edilemez ya da onu geçer."

**Katkı listesi (taslak):**
1. Rakibin iki donanım sınıfında replikasyonu (GTX 1650 ve L4; ikisi de YEŞİL) ve resmî koddaki son-epoch / val-AUC seçim farkının belgelenmesi [R][RC][LC] §1.4.
2. Ön-kayıtlı, test'i bir kez açan, Holm düzeltmeli bir karşılaştırma: DINOv2 sonda / ince ayar ↔ eşit önağırlıklı R18 ↔ rakip [T].
3. **İki katmanlı lezyon-ayrık değerlendirme:** yeniden eğitimsiz C0 ile yeniden eğitimli C1 ve S/N ayrışımı. Kazancın lezyon tekrarına ne kadar bağlı olduğunun nicel ölçüsü [T][C1].
4. Güçlü temel çizgi dersi: R18'e eğitimde yalnız D4 eklemek her üç değerlendirmede tutarlı kazanç veriyor ve DINOv2-B kısmi ince ayarıyla ayırt edilemiyor [T][C1].
5. Başarısız denemelerin mekanizma cümleleriyle raporu: H05, H07, H09, H13 ve tutmayan ön-kayıtlar (§8 tablosu) [G].
6. Tekrar üretilebilir deney paketi: kuyruk, beklenti kapıları, SHA256 dondurma, ön-kayıt + ek b.

### 6.B — ANLAŞILDI-GEÇEMEDİ (yalnız kayıt için)
Tez taslağı yazılmadı, çünkü koşul sağlanmıyor. A2 vurgusu B'nin analitik içeriğini taşıyor; çıkışı B diye etiketlemek ön-kayda aykırı olur.

### 6.C — TEKRARLANAMADI (yalnız kayıt için)
Uygulanamaz. Tez cümlesi yok.

---

## 7. Makaleye girebilecek / girmemesi gereken iddialar

### 7.1 Girebilir (sayı + GA + kaynak)
| # | İddia cümlesi (taslak) | Sayı [GA] | Kaynak |
|---|---|---|---|
| 1 | Rakip replike edildi | test ACC 0,7395 ± 0,0053 (L4, n = 5) ∈ [0,7218; 0,754]; yerel 0,7296 ± 0,0176 | [RC], [R] |
| 2 | DINOv2-B kısmi ince ayarı, resmî 28 px rakibi resmî bölmede geçer (**paket kazancı**: çözünürlük + önağırlık + omurga + tarif) | +15,45 [+13,84; +17,01], Holm t 1,29e-06 | [T] Aile 1 |
| 3 | … ve lezyon-ayrık C bölmesinde yeniden eğitimle de geçer | +8,26 [+6,63; +9,94], Holm t 3e-05 | [C1] |
| 4 | Eşit önağırlıklı R18 ince ayarına üstünlük **resmî bölmede** doğrulandı | +2,26 [+1,15; +3,30], Holm-anlamlı | [T] Aile 2 |
| 5 | Bu üstünlük lezyon tekrarlı görüntülerde yoğunlaşır | Δ_S +5,43 [+3,23; +7,47] / Δ_N +0,59 [−0,58; +1,76]; fark +4,83 [+2,22; +7,18] | [T] S/N |
| 6 | Lezyon-ayrık bölmede DINOv2 ↔ R18 farkı ayırt edilemedi | C1 +1,12 [−0,05; +2,30]; C0 +0,28 [−0,83; +1,47] | [C1], [T] |
| 7 | R18'e eğitimde D4 eklemek üç değerlendirmede de kazandırır (**rakibe D4 hakkı verilmedi** sınırlılığıyla) | test +2,62 [+1,85; +3,36]; C0 +1,50 [+0,70; +2,28]; C1 +2,09 [+1,24; +2,97] | [T], [C1] |
| 8 | DINOv2-B ince ayarı ile R18 + D4 ayırt edilemedi (betimsel, Holm dışı) | test −0,36 [−1,40; +0,70]; C1 −0,96 [−2,10; +0,18]; C0 −1,22 [−2,36; −0,08] | [T], [C1] |
| 9 | H08'in dengeli doğruluk bayrağı ve açıklaması (§2.1 cümlesi) | +8,41 [+5,85; +11,22]; 5 sınıf +5,22 [+3,46; +6,88] | [T] |
| 10 | Donuk sondadan ince ayara geçiş büyük kazanç verir | test +6,80 (n); val +7,24 [+5,48; +9,11] | [T], [T4h] §5 |
| 11 | Seçilen kazanan val → test kararlı | 0,8947 → 0,8940 | [KV], [T] |
| 12 | Sıcaklık ölçekleme kalibrasyonu düzeltir (katman satırı, karar vermez) | H10 ECE 0,0621 → 0,0179; H08c 0,0530 → 0,0168 | [T] |
| 13 | (Ek tablo, betimsel) İki ailenin topluluğu | 0,9169 ± 0,0027; T4 − H10 +2,29 [+1,61; +2,93]; düzeltmesiz; AUC bayrağı notuyla | [T] |

### 7.2 Girmemeli
| Yasak ifade | Neden | Kaynak |
|---|---|---|
| "DINOv2 ince ayarı R18'i lezyon-ayrık bölmede de geçer" | L = FALSE (GA alt −0,05). Tek yönlü p'lerin < 0,05 olması (Holm t 0,018, boot 0,031) kuralı değiştirmez; "anlamlı" kelimesi kural geçmeden kullanılamaz ([ON] §2.5) | [C1] |
| "H10'un kazancının yarısından fazlası lezyon-ayrık bölmede kayboluyor" | İfade eşiği FALSE: oran 0,503, ama KÇ desteklenmedi | [C1] ii_KC |
| "H10'un kazancı lezyon tekrarına H08'inkinden daha duyarlı" | KÇ0 ve KÇ ayırt edilemedi. Yalnız model içi S/N (T-SN1) söylenebilir | [T], [C1] |
| "H08'in kazancı S ve N arasında dengeli dağılır" | T-SN2 ön-kaydı tutmadı (ham fark +2,52 [+0,52; +4,32]) | [T] |
| "En iyi modelimiz / önerilen yöntem T4 (0,917)" | T4 kazanan olamaz; Holm ailesi yok; AUC > 0,99 bayrağı açık | [ON] §1 (8'), §2.2 |
| "H08 (ya da R18 + D4) kazanandır" | W val'de belirlendi; test W'yi değiştirmez | [ON] §1, §6.3 |
| "DINOv2 + D4 / D4, DINOv2 farkını açıklar" | H10 + D4 koşulmadı; omurga ve augmentasyon karışık | [T4h] §8.6 |
| "Wu & Papież'in %92,02'sine ulaştık / yakın" ya da "SOTA" | Bizimki kısmi ince ayar (0,8940). Onlarınki uçtan uca, hakemsiz ve protokolü belirsiz; o sonuç da resmî bölmeye ait | [LC] §3 |
| "DINOv2 rakibi 15 puan geçer" (paket notu olmadan) | Δ; çözünürlük, önağırlık, mimari ve tarifi birlikte içerir. Adil kıyas H02c'dir | §9 |
| C-test ACC'lerini resmî test ACC'leriyle doğrudan kıyaslamak | Çoğunluk tabanları farklı: 0,7726 ↔ 0,669 | [ON] §3, [DM] |
| Klinik kullanım, tanı desteği iddiası | V6; HAM10000 araştırma benchmark'ı | [LC] §5 |
| Hız / verim iddiası | Paralel Colab süreleri yalnız maliyet bilgisidir (R6) | [OD] KARAR 8 |
| H04 katmanlı değerleri birincil gibi sunmak (ör. H08 dengeli 0,8376) | Katmanlar karar vermez | [ON] §2.4 |

---

## 8. Deney kanıt paketi (D5) — eksikler ve K6 öncesi sorumlular

| Öğe | Durum | Eksik | Kim |
|---|---|---|---|
| Günlük | Var [G] | **K5 satırları yok** (test geçişi, C1). Arıza tablosunda 21:49 ve 22:56 kayıtları var, v3 satırı [K5-C1-UYGULAMA-NOTU] tamam. **Bütçe sayacı ≈ 21,5'te kalmış**; K5 test ve C1 L4-sa değerleri `.done`/`canli_*` dosyalarından eklenmeli | şef |
| Replikasyon tablosu | Var [R][RC] (+ test_auc, test_acc_son) | — | — |
| Ablasyon | Zincir §4'te. **`sonuclar/ablasyon.csv` YOK** | [T]/[C1] JSON'larından CSV üretilmeli (CPU, yeni koşum yok) | deney-yürütücü |
| ≥ 3 seed istatistik | Var: 5 seed, eşleştirilmiş t + bootstrap + Holm [T][C1] | — | — |
| Başarısız denemeler tablosu | Günlükte H05/H07/H09/H13 satırları var. **[ON] §6.5'teki ayrı D4 tablosu yok** | Eklenecek satırlar: H05, H07, H09, H13 (val) + **tutmayan ön-kayıtlar**: L (C1-i H10), C0-1, T-SN2 "dengeli dağılım", KÇ0, KÇ, KÇ ifade eşiği | tahlilci taslak (bu tablo) → şef |
| Test / C1 dosyalarının yerel arşivi | **YOK**. `runs_colab/ARSIV_MANIFEST.json` 18:28 tarihli (K5 öncesi); `tahmin_test*` dosya sayısı 0; C1 koşumları yok | [ONb] §3: `colab_arsiv_kopya.py --test-dahil` (KILIT var) ile K5 test + k5-c1-* koşumları + SHA256 manifest | şef / deney-yürütücü |
| P6 kapı raporları | Yalnız Drive'da [P6] | `beklenti-k5-test-asama{1,3}.md` ve `beklenti-k5-c1-*` raporları yerel `sonuclar/`'a kopyalanmalı | şef |
| "Önce/sonra" hata kırılımı | `hata_kirilim.py` baseline, H01, H02, H08, H09 için var (val). **H10 (W) için yok** | `--varyant h10_dinov2b_k4_ft224` (val; CPU). Test kırılımı istenirse "test sonrası betimsel" etiketiyle | deney-yürütücü |
| tahlil-tur5.md | Yok (karar kartta ve günlükte) [K5-ONKOSUL-KONTROL] | İsteğe bağlı | — |

---

## 9. Sınırlılıklar taslağı

1. **Test bir kez açıldı** (2026-10-11 21:12; `ikinci_acilis: null`). C1'de C-test her kol için bir kez açıldı. Test sonrası yeni kol, seed ya da eşik eklenmedi ([ON] §6.3).
2. **Rakibe D4 hakkı verilmedi (M3 = HAYIR, KULLANICI KARARI 9).** H08 satırı "rakibe augmentasyon hakkı verilmedi" cümlesiyle verilir. Adil kıyas satırı augmentasyonsuz H02c'dir ([ON] §2.1).
3. **C2 koşulmadı (KULLANICI KARARI 11).**
   - Yayımlanmış DermaMNIST-C npz pikselleriyle doğrudan kıyas yok.
   - "Görüntü hazırlama etkisi" (C2 − C1) ölçülmedi.
   - Baseline C1 (0,8334) ile Abhishek vd. 0,825 arasındaki uyum, piksel farkı nedeniyle yalnız betimseldir.
4. **28 px rakip ↔ 224 px DINOv2, adil karşılaştırma notu.**
   - Aile 1 farkları (+8,7…+15,8) şunları birlikte içerir: çözünürlük (28 → native 224), önağırlık (yok → ImageNet / LVD-142M), mimari, eğitim tarifi (optimizer, lr, epoch, normalizasyon 0,5/0,5 → ImageNet) ve seçim kuralı (val-AUC ↔ val-ACC).
   - Yalnız seçim kuralının etkisi ölçüldü ve ihmal edilebilir: son-epoch satırı −0,31 [−1,28; +0,73] [T].
   - Literatürde yalnız 28 → 224 büyütme +1,9 puan veriyor (Yang 2023 T3 0,735 → 0,754 [LC] §4(a)).
   - **Bilimsel karşılaştırma H10 ↔ H02c'dir:** aynı çözünürlük, ikisi de önağırlıklı ve ince ayarlı.
   - Ne H02'ye ne H10'a hiperparametre araması yapıldı (tek tarif, simetrik [T4h] §8.6 madde 8).
5. **Tek donanım sınıfı.**
   - Karşılaştırılan bütün test ve C1 sayıları Colab L4, FP32, TF32 kapalı ortamdan geliyor.
   - K4 seçimi kısmen yerel GTX 1650'de yapıldı (Tur-1…3), kısmen L4'te (Tur-4).
   - Tekrar kapısında Colab − yerel val farkı H01/H02/H08 için ≤ 0,3 puan, baseline için +0,62 [OD].
   - Başka GPU sınıflarında ya da AMP/TF32 açıkken sonuçlar sınanmadı.
6. **Seed sayısı 5** {0..4}. Bootstrap görüntü düzeyinde. MDE ≈ 3 puan (KÇ güç beyanı [ON] §4.3).
7. **Resmî bölmede lezyon tekrarı:** test'in %34,6'sı (693/2.005) S [DM]. Resmî bölme sonuçları S/N ayrışımıyla birlikte verilir. Wu & Papież'in 92,02'si de resmî bölmeye aittir [LC] §3.
8. **Nadir sınıf sayıları küçük:** resmî test'te df 23, vasc 29; C-test'te df 8, vasc 17, akiec 35 [DM]. Dengeli metrikler ve H08 bayrağı bu oynaklığa duyarlı. df/vasc hariç satırlar zorunlu.
9. **Protokol notları:**
   - Rakip kodu her epoch'ta C-test'i değerlendiriyor, ama seçimde kullanmıyor ([ONb] §2).
   - C1 rakip kolunda `medmnist.INFO` split boyutları yalnız süreç içinde değiştirildi. Rakip dosyalarına dokunulmadı (veri yolu uyarlaması, K5-C1-UYGULAMA-NOTU).
10. **Kapsam dışı / test'e gitmeyen kartlar:**
    - H03, H05, H07, H09, H13 (ÖLDÜR ya da analiz).
    - H11 C1'e alınmadı (KARAR 9).
    - DINOv2-B tam ince ayar (K = 12) koşulmadı.
    - Eksik ablasyonlar §4'te.
11. **Paralel Colab süreleri** yalnız maliyet bilgisidir; hız iddiası yok (R6).
12. **Klinik iddia yok.** HAM10000 / DermaMNIST bir araştırma benchmark'ıdır; lisans CC BY-NC 4.0 [LC] §6.

---

## 10. Önerilen çıkış ve K6 brief kapsam cümlesi

**Öneri (karar kullanıcının):**
- Çıkış **A. GEÇTİ, kademe A-kayıtlı** (§7 mekanik: G1 ✔, G2 ✔*, G3 ✔, V ✔, L ✘).
  - *G2, §2.1 teşhisinin kullanıcı tarafından kabulüne bağlı.*
- Vurgu için iki seçenek var: A1 (iyileştirme) ve A2 (protokol duyarlılığı). **Tahlilcinin eğilimi A2.** Gerekçeler:
  - En sağlam ve en yeni bulgu, DINOv2 ↔ R18 farkının bölme protokolüne bağımlılığı ve D4'lü R18'in buna denk gelmesi.
  - A1 tek başına okunursa, L = FALSE olduğu için "DINOv2 üstünlüğü" fazla genel algılanır.
- T4 yalnız ek tabloda.

**K6 brief kapsam cümlesi (taslak):**
> "Bu çalışma, DermaMNIST'te (HAM10000) resmî 28 px ResNet-18 temel çizgisinden 224 px DINOv2 ince ayarına geçişi, ön-kayıtlı ve test'i bir kez açan bir protokolle inceler. Sonuçlar iki bölmede raporlanır: lezyon tekrarı içeren resmî MedMNIST bölmesi ve lezyon-ayrık DermaMNIST-C bölmesi (resmî piksellerle, yeniden eğitimle). Kıyaslar beş seed, eşleştirilmiş bootstrap ve Holm düzeltmesiyle verilir. İddialar sınıflandırma doğruluğu, AUC, dengeli doğruluk ve kalibrasyonla sınırlıdır; tek donanım sınıfında (L4, FP32) yapılmıştır. Rakibe augmentasyon hakkı verilmemiştir ve yayımlanmış DermaMNIST-C pikselleriyle doğrudan kıyas yoktur. Klinik bir iddia taşımaz."

---

## 11. Kullanıcıya açık sorular
1. **Çıkış seçimi:** A-kayıtlı (önerilen) ve vurgu A1 ya da A2.
2. **G2 onayı:** H08 dengeli bayrağının teşhisi (§2.1, "arıza yok") kabul ediliyor mu? Kabul edilmezse ¬G2 olur ve çıkış yazılamaz ([ON] §7).
3. **T4:**
   - Yalnız ek tabloda betimsel mi kalsın (önerilen)?
   - Yoksa iddia taşıyacaksa, AUC > 0,99 bayrağı için C0 / N_test AUC denetimi yapılsın mı? Bu denetim dondurulmuş test tahminlerine "test sonrası keşifsel" etiketiyle yeniden bakmak anlamına gelir.
4. **Ek GPU önerilmez.** H10 + D4 ve rakip + D4 (M3) test sonrası ön-kayda eklenemez ([ON] §6.3). Yapılırsa test ikinci kez açılmış olur ([ON] §6.4 beyanı gerekir). Bu yüzden sınırlılık olarak kalmaları öneriliyor.
5. **D5 eksikleri (§8)** K6'dan önce kapatılmalı: arşiv kopyası, günlükte K5 satırları ve bütçe, `ablasyon.csv`, başarısız-ön-kayıtlar tablosu, H10 hata kırılımı.

---

## 12. KULLANICI KARARI (2026-10-12) — BAĞLAYICI
- **Çıkış: A. GEÇTİ (A-kayıtlı), vurgu A2 — protokol duyarlılığı.** Tez cümlesi §6.A (A2). L = FALSE makalede açıkça yazılır.
- **G2:** H08 `dengeli_delta>6,5` teşhisi (§2.1) KABUL; makalede zorunlu beyan cümlesiyle.
- **T4:** yalnız ek tablo, betimsel; iddia taşımaz; test'e yeniden bakılmaz (AUC denetimi YOK).
- Ek GPU/koşum yok; H10+D4 ve rakip+D4 sınırlılık.

## 13. D5 kapanışı (şef, 2026-10-12)
§8 eksiklerinin altısı kapandı: runs_colab/ARSIV_MANIFEST.json (K5 test + C1; arsiv_dogrula hata 0) · sonuclar/DRIVE_SONUC_MANIFEST.json (P6/beklenti raporları, 34 dosya) · sonuclar/ablasyon.csv · sonuclar/basarisiz-onkayitlar.md · hata-analizi.md §9 (H10 + L4 kıyas kolları) · günlük K5 satırları + bütçe ≈30,6/32 oturum-sa (alt sınır ≈26,4). colab_arsiv_kopya.py'ye --test-dahil bayrağı eklendi (ONb §3 md 2; yedek scratchpad). Sapma: arşiv kopyası 849 sn sürdü (>5 dk; GPU işi değil, sonuç etkisiz).
