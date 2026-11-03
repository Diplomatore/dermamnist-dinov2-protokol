# Başarısız denemeler ve tutmayan ön-kayıtlar — 3.1 DermaMNIST · ResNet-18 (28 px) → DINOv2

*deney-yürütücü · 2026-10-12 · D5 eksiği (karar.md §8, K5-ONKAYIT §6.5). Yeni koşum yok, test yeniden açılmadı.*
*Bütün sayılar aşağıda adı geçen dondurulmuş dosyalardan kopyalandı; yeni hesap yapılmadı. Δ'lar puan (×100); köşeli parantez %95 eşleştirilmiş görüntü bootstrap GA'sı (B = 2.000, RNG 0); p_t kesin tek yönlü seed-eşleştirilmiş t (n = 5).*
*Makale kuralı (karar.md §7.2, H05 ön-beyanı): "ayırt edilemedi" ve "n = 5'te saptanamadı" satırları **"etki yok" diye yazılmaz**.*

**Kaynak kısaltmaları:** [T1h]…[T4h] `sonuclar/tahlil-tur{1..4}.md` · [H13] `hipotezler/H13.md` "Sonuç" · [G] `gunluk/deney-gunlugu.md` · [T] `sonuclar/k5_test_sonuc.json` · [C1] `sonuclar/k5_c1_sonuc.json` · [ON] `hipotezler/K5-ONKAYIT.md` · [ONb] `hipotezler/K5-ONKAYIT-b.md` · [İH] `sonuclar/inceleme-h08.md` · [HA] `sonuclar/hata-analizi.md` · [K] `sonuclar/karar.md`

---

## A. Öldürülen kartlar (K4, yalnız validation; n = 5 seed {0..4})

| Kart | Ebeveyn → varyant | Ön-kayıtlı beklenti | Öldürme kuralı (ön-kayıtlı) | Ölçülen | Tetiklenen | Mekanizma cümlesi (ölçülmüş) | Kaynak |
|---|---|---|---|---|---|---|---|
| **H05** sınıf ağırlıklı CE (β = 0,5) | h02 → h05_r18in_ft224_cw05 | dengeli Δ +1,0…+4,0; ACC Δ −1,5…+0,3 | (b) dengeli Δ < +0,5 **veya** p_t ≥ 0,05 **veya** ACC < 0,8572 | dengeli Δ **+1,51 [−0,23; +3,28]**, p_t **0,097**; ACC 0,8662 (Δ −0,10) | (b): p_t ≥ 0,05 | **n = 5'te saptanamadı (MDE ≈ 2,9 puan; ön-beyanlı).** Yön üç seçim kuralında da pozitif (+1,5 / +1,8 / +3,1). Etkin önsel kayması τ ≈ 0,2 (kuramsal 0,5): train kaybı ep 5'te ≈ 0'a inince ağırlık etkisi sönüyor; AUC değişmedi | [T2h] §3.1, [G] Tur-2 |
| **H07** TTA (8 dihedral görünüm) | h02 → h07_h02_tta8 | ACC Δ −0,5…+1,5 (merkez +0,5) | (b) Δ < +0,5 (0,8722) **veya** bootstrap GA alt ≤ 0 | ACC 0,8628 ± 0,0019; Δ **−0,44 [−1,48; +0,60]** | (b) iki koşul | Dönüşmüş görünümler −2,63 puan dağılım dışı; 90° görünümleri D2'den zararlı (+0,94 [+0,08; +1,77]). Ödünleşim: ECE 0,0745 → 0,040, ama mel duyarlılığı −7,39 [−12,46; −2,38] | [T3h] §3.1, [G] Tur-3 |
| **H09** renk oynaması (B/C/S U[0,8; 1,2]) | h02 → h09_r18in_ft224_renk02 | ACC Δ −1,0…+1,0 (merkez 0) | (b) Δ < +0,5 **veya** p_t ≥ 0,05; (b') t ↔ bootstrap çelişkisi → TUT verilmez | ACC 0,8704 ± 0,0037; Δ **+0,32 [−0,58; +1,26]**, p_t 0,033 | (b) ilk koşul (Δ < +0,5); (b') çelişkisi kayıtlı, karar etkisiz | Kaynak kısayolu imzası doğru yönde ama ayırt edilemedi (D_kk +0,60 [−0,89; +2,16]); zayıf düzenleyici (kayıp 2,5×) | [T3h] §3.2, [G] Tur-3 |
| **H13** topluluk H02c ⊕ H10 (½;½ log-olasılık) | h10 → h13_topluluk_h02c_h10 | Δ vs H10 −0,8…+1,5 (merkez +0,4); ÖLDÜR olasılığı yüksek diye ön-beyanlı | (b) Δ < +0,5 (0,8997) **veya** p_t ≥ 0,05 | ACC 0,8961 ± 0,0041; Δ **+0,14 [−0,96; +1,24]**, p_t 0,369 | (b) iki koşul | Topluluk kazancı G = +1,60 (üye ort.'na göre) ama ebeveyn H10 ile üye ortalaması arasındaki fark g/2 = 1,455 bunu yiyor (E − H10 = G − g/2). Nadir sınıf bedeli: dengeli −1,95 [−5,19; +0,98], mel −6,67 [−11,86; −1,89]. ECE/NLL iyileşiyor | [H13] Sonuç, [G] Tur-5 |

*Not: H03 (çözünürlük merdiveni), H06 (logit ayarı genelleşmesi), H12 (DINOv2-S sonda) analiz kartlarıdır; öldürme kuralı yoktur, ön-kayıtlı T maddeleri tuttu ([T1h] §4, [G]). H04, H08, H10, H11 TUT aldı (H08: İNCELE → arıza yok → K5 doğrulama adayı).*

---

## B. K5 ön-kayıt maddelerinden tutmayanlar (resmî test bir kez, C0, C1)

| # | Ön-kayıtlı madde | Ön-kayıtlı kural / beklenti | Ölçülen [GA] | Kural etiketi | Sonucun anlamı (makale) | Kaynak |
|---|---|---|---|---|---|---|
| B1 | **(i) H10 − H02, C1 = "L"** (DINOv2 ince ayarının R18'e kazancı lezyon-ayrık eğitim/test bölmesinde de var) | GA alt > 0 → desteklenir; beklenen Δ ∈ [−1,0; +3,0], merkez +1,0 | **+1,12 [−0,05; +2,30]**; Holm t 0,018, Holm boot 0,031 | **ayırt edilemedi → L = FALSE** | Kademe A-kayıtlı (V ∧ ¬L). "Lezyon-ayrık bölmede de geçer" yazılamaz; tek yönlü p < 0,05 olsa da kural geçmeden "anlamlı" denemez ([ON] §2.5). Nokta tahmin beklenti merkezinde | [C1] `C1.i`, `C1.aile_C1_2`, `olcutler.L`; [K] §0, §7.2 |
| B2 | **C0-1: H10 − H02c, C-test alt kümesi (yeniden eğitimsiz)** | GA alt > 0 → "kazanç lezyon-ayrık görüntülerde de var"; beklenen [−1,0; +3,0], merkez +1,0; önceden beyan: P(GA alt > 0) ≈ 0,4 | **+0,28 [−0,83; +1,47]**, p_t 0,208 | **ayırt edilemedi** | Lezyon-ayrık görüntülerde DINOv2 ↔ R18 farkı ölçülemedi. Nokta beklenti merkezinin altında | [T] `C0.kiyas.h10-h02` |
| B3 | **T-SN2: H08c − H02c, Δ_S ≈ Δ_N** ("D4 kazancı S/N arasında dengeli") | Beklenen ham Δ_S − Δ_N ∈ [−3; +4]; GA 0'ı kapsarsa "dengeli dağılım" ile uyumlu | Δ_S +4,27, Δ_N +1,75; ham fark **+2,52 [+0,52; +4,32]** (aralık içinde ama GA 0'ı dışlıyor). İkincil tabaka-std fark +0,89 [−2,04; +3,65] | **"dengeli dağılım" TUTMADI** (ham farkta S > N). Betiğin "desteklendi" etiketi bu kıyas için "Δ_S > Δ_N" anlamına gelir | "H08'in kazancı S ve N arasında dengeli dağılır" yazılmaz. H08 kazancı N'de de GA > 0 (+1,75 [+0,91; +2,60]) | [T] `SN.h08-h02`, `kiyas.h08-h02.acc_N_ham`; [K] §5 |
| B4 | **KÇ0: K0_H10 > K0_H08** (değerlendirme bileşeni; H10'un kazancı lezyon tekrarına daha duyarlı) | GA alt > 0 → desteklenir; beklenen [+0,5; +2,5], merkez +1,5 | K0_H10 +1,99 [+1,06; +2,94], K0_H08 +1,12 [+0,40; +1,80]; fark **+0,86 [−0,01; +1,78]** | **ayırt edilemedi** (GA alt −0,01) | Nokta ön-kayıtlı yönde ve aralıkta; "H10'un kazancı H08'inkinden daha çok çöküyor" yazılamaz | [T] `C0.KC0` |
| B5 | **KÇ: K_H10 > K_H08** (resmî → C1; ön-kayıtlı yön) | GA alt > 0 → desteklenir; beklenen fark [+0,5; +2,5]; K_H10 ∈ [+0,5; +3,5], K_H08 ∈ [−1,5; +1,5]; güç beyanı MDE ≈ 3 puan, "ayırt edilemedi" olasılığı yüksek | K_H10 +1,14 [+0,04; +2,31], K_H08 +0,54 [−0,39; +1,47]; fark **+0,60 [−0,45; +1,68]**; bileşenler K1_H10 −0,85, K1_H08 −0,59 | **ayırt edilemedi** | K_H10 ve K_H08 kendi beklenti aralıklarında; fark saptanamadı (önceden beyan edilen güç sınırı) | [C1] `C1.ii_KC` |
| B6 | **KÇ ifade eşiği** ("H10'un resmî bölme kazancının yarısından fazlası lezyon-ayrık bölmede kayboluyor") | K_H10 / Δ_H10(resmî) ≥ 0,5 **ve** KÇ desteklendi | oran **0,503** (≥ 0,5), KÇ desteklenmedi | **FALSE** | Cümle makalede kullanılamaz | [C1] `C1.ii_KC.oran_K_H10_bolu_delta_resmi`, `ifade_esigi` |
| B7 | **C2** (yayımlanmış DermaMNIST-C npz ile ikincil değerlendirme) | Ön-kayıtlı ikincil analiz ([ON] §5) | **koşulmadı** | yapılmadı (KULLANICI KARARI 11, bütçe) | Sınırlılık: yayımlanmış C pikselleriyle doğrudan kıyas ve "görüntü hazırlama etkisi" (C2 − C1) yok | [K] §9.3, `orkestrasyon_durumu.md` KARAR 11 |
| B8 | H13 kuralı (§2.3) ve H13-C1 | Yalnız H13 Tur-5'te TUT alırsa | — | **uygulanmadı** (H13 ÖLDÜR; W = H10) | T4 = H08c ⊕ H10 yalnız ek satır, betimsel | [T] `olcutler.W`, [K] §3 |

**Tutan K5 maddeleri (bağlam için; tutmayan tablosuna girmez):** G1 +15,45 [+13,84; +17,01]; H10 kuralı (V) +2,26 [+1,15; +3,30] (beklenti [+1,0; +4,5] içinde); H08 kuralı +2,62 [+1,85; +3,36]; T-SN1 ham Δ_S − Δ_N +4,83 [+2,22; +7,18] (beklenti [+1; +9]); C0-2 +1,50 [+0,70; +2,28]; (i) H08 − H02 C1 +2,09 [+1,24; +2,97]; G3 +8,26 [+6,63; +9,94]; (iii) baseline C1 0,8334 ∈ [0,79; 0,85]; C1 dengeli yön ön-kaydı (H10 − H02 +5,74 [+1,21; +10,17], H08 − H02 +9,15 [+5,64; +12,78]) — [T], [C1].

---

## C. Tetiklenen "görülmemiş kadar iyi" bayrakları (ön-kayıtlı eşik aşıldı → İNCELE)

| # | Bayrak | Ön-kayıtlı eşik | Ölçülen [GA] | Teşhis | Durum | Kaynak |
|---|---|---|---|---|---|---|
| C1 | H08 dengeli Δ (val, Tur-3) | H08 kartı: dengeli Δ > +6,5 → İNCELE | **+8,06 [+4,97; +11,79]** | Arıza yok: 28 px D4 piksel taraması RMSE ≤ 5 kopya 0; nadir sınıf (df/vasc) örnekleme oynaklığı (5 sınıf +5,19); eşik Z2 ailesinden aktarılmış (aile dışı çıpa; gevşetilmedi) | M6 gereği TUT verilmedi; kullanıcı yol A (K5 test doğrulaması) | [İH], [G] arıza 09:40 |
| C2 | H08c − H02c dengeli Δ (resmî test) | [ON] §2.1: dengeli Δ > 6,5 → rapor bayrağı | **+8,41 [+5,85; +11,22]**; df/vasc hariç 5 sınıf +5,22; seed başı 8,08 / 8,55 / 8,65 / 8,06 / 8,72 | Arıza yok, açıklanabilir: df + vasc katkısı toplamın %56'sı (seed başına ≈ 8 görüntü); C0'da +8,26, C1'de +9,15 (sızıntı açıklaması dışlandı) | G2 teşhisi KULLANICI KARARI 12 ile kabul; makalede zorunlu beyan cümlesi | [T] `dogrulama.H08 (§2.1)`, [K] §2.1, §12 |
| C3 | T4 (H08c ⊕ H10) test AUC | **Ön-kayıtta T4 için tanımlı değil**; tahlilci [LC] §4(e) genel eşiğini (> 0,99) uyguladı | **0,9904**; seed başı 0,9893 / 0,9907 / 0,9911 / 0,9902 / 0,9904 (4/5 > 0,99) | Denetim yapılmadı (test'e yeniden bakmayı gerektirir) | KULLANICI KARARI 12: T4 yalnız ek tablo, betimsel; AUC denetimi YOK | [T] `kollar.t4.seed.auc`, [K] §2.2, §12 |

*H10'un üç görülmemiş eşiği (ACC > 0,93, AUC > 0,99, H10 − H02c > +6,0) ve H08'in diğer üç eşiği (ΔACC > 3, ACC > 0,90, AUC > 0,99) FALSE ([T] `olcutler.G2_bayraklar`).*

---

## D. Tutmayan ikincil / keşifsel ön-kayıtlar (karar vermez; ön-beyanlı ikinciller)

| # | Kart | Ön-kayıtlı ikincil / keşifsel yön | Ölçülen | Durum / açıklama | Kaynak |
|---|---|---|---|---|---|
| D1 | H01 | molemax-dışı val ACC ∈ türetilmiş [0,74; 0,80] | **0,730** (Δ vs B +12,1) | Aralığın altında; açıklandı ([T1h] §3d) | [T1h] §3d, §4 |
| D2 | H03 | keşifsel: r ↑ → mel/bkl OvR AUC ↑ | r128 > r224 (224 basamağında tersine döndü) | D-gözlemi, arıza değil; 64 → 128 → 224 ACC adımları da ayırt edilemiyor | [T1h] §3c, satır 96 |
| D3 | H05 | df/vasc/akiec/mel duyarlılığı ↑ | akiec ↓ (GA 0'ı kapsıyor) | beklentinin tersi, keşifsel not; mekanizma ölçülmedi | [T2h] §2 madde 6, defter D |
| D4 | H10 | keşifsel: tabakalı S − N (H10) ≤ H02 (ezber artmaz) | tabakalı S − N +10,09; Δ vs H02c **+3,92 [−2,25; +10,16]** | kural gereği "ayırt edilemedi", nokta yön tersine; kazancın %75'i S → K5'in birincil sorusu oldu (T-SN1 test'te desteklendi) | [T4h] §2, §4 |
| D5 | H11 | manipülasyon denetimi: ep 5 train kaybı H10'dan düşük | 0,085–0,106 > H10 0,040–0,057 | tersi; S omurganın kapasitesi düşük, ezber gecikiyor ama son kayıp daha düşük (0,0005 < 0,0011) | [T4h] §2, H11 Öğrenilen |
| D6 | H13 | ön-kayıtlı not koşulu (f): ΔACC > 0 ve Δdengeli < −1,0 → "topluluk nadir sınıf bedeli" | ΔACC +0,14, Δdengeli **−1,95 [−5,19; +0,98]** | koşul tetiklendi (not); vasc −12,86, mel −6,67 | [H13] Sonuç (2) |
| D7 | H08 | nv payı \|Δ\| ≤ 0,02 (Tur-1 nv kuralı) | \|Δ\| 0,0205 | sınırda; yön nv'den uzaklaşma (kuralın kaygısının tersi), not | [T3h] §2 |

---

## E. Onaydan ÖNCE düzeltilen beklentiler (taslak kart; ön-kayıt ihlali değil — şeffaflık için)

| Taslak beklenti | K3.5 ölçümü | Düzeltme (onay öncesi) | Kaynak |
|---|---|---|---|
| H01/H02/H03: "her model için ham S − N > 0" (ezber) | Baseline'da ham S − N **−18,8 ± 2,9** (Simpson: follow_up/molemax karıştırıcısı) | Ölçü tabakalı S − N'ye (sınıf × dx_type × kaynak) çevrildi | [HA] §4 |
| H01: ECE baseline'dan düşük (Medeiros 0,196 çıpası) | protokol baseline val ECE 0,0365 | kıyas değeri 0,0365, yön keşifsel | [HA] §3 |
| H02: val ACC Δ +8…+15 | B + 15 = 0,890, görülmemiş eşik 0,90'a 1 SE | üst uç +14,5 önerildi (eşik değişmedi) | [HA] §8 |

---

## F. Beklenti kapısı / koşum arızaları (bilim sonucu değil; geçersiz sayılan koşumlar)

| Tarih | Kayıt | Kök neden | Etkilenen | Kaynak |
|---|---|---|---|---|
| 2026-10-10 | baseline_s0 / _s1 rc=95 (ilerleme yok) | ASUS yazılımının tetiklediği uyku | 20261010-094349, -104956 GEÇERSİZ | [G] arıza |
| 2026-10-10 | h03 r64 s1 rc=96 | ArmouryCrate bellek sızıntısı → kuyruk öldürüldü | 20261010-212001 GEÇERSİZ | [G] arıza |
| 2026-10-11 | H04 a3 kapısı s3 seed başı İHLAL (+7,62 > +6,0) | eşik ortalama içindi; kapı seed başına uyguladı | yok | [G], [T2h] |
| 2026-10-11 21:49 | k5-c1-A a1: baseline_c1 / h01_c1 duman rc=1 | medmnist INFO split boyutu sabit; sıralı npz → tek sınıf duman alt kümesi | 2 duman (ölçüm değil) | [G], `sonuclar/beklenti-k5-c1-A-asama1.md` |
| 2026-10-11 22:56 | k5-c1-A-v2 a1 İHLAL n_train/val_n = 256 | beklenti kuralı varyant filtresiz (kapsam hatası; eşik gevşetilmedi) | 4 duman | [G], `sonuclar/beklenti-k5-c1-A-v2-asama1.md` |

*K5 test (k5-test a1/a3) ve C1 (k5-c1-A-v3, k5-c1-B) kapıları GEÇTİ; P6 özdeşliği 35/35 (`sonuclar/beklenti-k5-test-asama{1,3}.md`).*
