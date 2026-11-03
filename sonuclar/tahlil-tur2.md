# Tur-2 Tahlili (K4c) — 3.1 DermaMNIST · H04 / H05 / H06 / R-T · YALNIZ VALIDATION

*Deney-tahlilci, Mod 2 · 2026-10-11 · GPU kullanılmadı. Test tahmini ya da etiketi açılmadı: H04/H05/H06 koşumlarının `metrics.json` ve `gecmis.json` dosyalarında `"test` anahtarı 0, koşum dizinlerinde `*test*` dosyası 0 (betikte sayıldı).*

| Girdi | Değer |
|---|---|
| Koşumlar (rc=0, seed 0–4) | H02 `20261010-222930/225428/231909/234351/20261011-000829` · **H05** `20261011-013813/020249/022727/025205/031643_h05_r18in_ft224_cw05_s0…s4` · H04 `20261011-011434/011443/011453/011503/011512` · H06 `20261011-011758/011808/011818/011827/011837` · H01 `20261010-221110…222554` · R-T `20261011-011856…012039` (20 koşum). Duman ve ön-uçuş koşumları hiçbir karar hesabına girmedi. |
| Betik (tekrar üretilebilir, CPU, RNG 0) | `scripts/tahlil_tur2.py` → `sonuclar/tahlil_tur2_bootstrap.{json,csv}`, `sonuclar/tahlil_tur2_ikincil.csv`, `sonuclar/tahlil_tur2_teshis.json`. Bootstrap, `tahlil_tur1.py` ile aynı tanım ve indekslerle hesaplandı. ACC GA'ları `varyantlar/ortak/eslestirilmis_bootstrap.py` ile 8/8 kıyasta birebir aynı çıktı. |
| Notasyon | "puan" = yüzde puanı. ort ± std = 5 seed (ddof 1). GA = eşleştirilmiş görüntü bootstrap'ı, %95 yüzdelik (2.000, RNG 0). p_t = kesin tek yönlü seed-eşleştirilmiş t (scipy 1.17.1). MDE = %80 güç ve tek yönlü α 0,05'te saptanabilir en küçük etki. "Tam-val eğri" = val'in tamamında ve seçimsiz hesaplanan betimsel nokta (karar değil). |

---

## 0. Şef ön-okumasının doğrulaması

| Şef ön-okuması | Doğrulama |
|---|---|
| H05 val_acc 0,8662 ± 0,0047, dengeli 0,7213 ± 0,0103 | ✓ (0,8662 ± 0,0047; 0,7213 ± 0,0103) |
| H02 0,8672 / 0,7062 | ✓ |
| H04 OOF dengeli 0,7425, ACC 0,8576 | ✓. OOF kararları ayrıca **bağımsız olarak yeniden hesaplandı**: `katlar.json` + τ_k ile H02 log p'den, 5 seed × 1.003 görüntüde **0 argmax uyuşmazlığı** çıktı. |
| H06 dengeli 0,6766 | ✓ |
| Varyant metrics'te test_ yok | ✓ (yukarıdaki sayım) |
| `tur2_cpu_olcum.json` H04/H06 Δ, GA, p_t | ✓. Bu betikte H04 − H02 ve H06 − H01 yeniden hesaplandı, sayılar tam olarak aynı çıktı. |

---

## 1. Ölçüm okuması (val, n = 5, seed {0..4} eşleşmiş)

### 1a. Karar istatistikleri

| Kıyas | ΔACC [GA] | p_t (ACC, B>A) | Δ dengeli [GA] | p_t (dengeli) | sd(Δ_seed) dengeli | MDE dengeli | karar değişen görüntü/seed |
|---|---|---|---|---|---|---|---|
| **H05 − H02** (birincil) | **−0,10** [−0,78; +0,58] | 0,63 | **+1,51 [−0,23; +3,28]** | **0,097** | 2,16 | 2,88 | 68,2 |
| H05(aucsecim) − H02(aucsecim) | +0,14 [−0,56; +0,90] | 0,40 | +3,06 [+1,17; +5,27] | 0,036 | 2,81 | 3,73 | 73,8 |
| H05(son ep) − H02(son ep) | −0,12 [−0,78; +0,56] | 0,61 | +1,83 [+0,22; +3,53] | 0,079 | 2,36 | 3,14 | 71,2 |
| **H04 − H02** (OOF) | **−0,96 [−1,66; −0,32]** | — (B<A p 0,014) | **+3,63 [+1,57; +5,95]** | **0,0176** | 2,59 | 3,45 | 42,8 |
| **H06 − H01** (OOF) | −0,66 [−1,60; +0,32] | — | **+2,72 [+0,65; +5,67]** | 9·10⁻⁵ | 0,45 | 0,60 | 31,6 |
| H05 − H04 (kart ikincil 11, keşifsel) | +0,86 [+0,10; +1,62] | 0,005 | −2,12 [−4,56; +0,16] | 0,85 | 3,93 | 5,23 | 81,0 |
| H05 − H02@τ=0,5 (Bayes eşdeğeri post-hoc, seçimsiz; teşhis) | +0,42 [−0,32; +1,16] | 0,0016 | −1,73 [−4,21; +0,56] | 0,86 | 3,06 | 4,07 | 77,0 |
| H02@τ=0,5 − H02 (teşhis, seçimsiz) | −0,52 [−1,06; −0,02] | — | +3,24 [+1,26; +5,61] | 0,005 | 1,57 | 2,08 | 30,0 |

Seed başına dengeli Δ: **H05** −0,35 / +1,94 / +4,30 / −0,96 / +2,61 · H04 +0,88 / +4,56 / +2,79 / +7,62 / +2,29 · H06 +2,88 / +2,93 / +3,27 / +2,30 / +2,20.

### 1b. İkincil metrikler (tahminlerden, tek tip hesap; seed ortalaması)

| metrik (val) | H02 | **H05** | H04 (OOF) | H02@τ=0,5 (teşhis) | H01 | H06 (OOF) |
|---|---|---|---|---|---|---|
| ACC | 0,8672 | **0,8662** | 0,8576 | 0,8620 | 0,8223 | 0,8158 |
| dengeli ACC | 0,7062 | **0,7213** | 0,7425 | 0,7386 | 0,6494 | 0,6766 |
| makro-F1 | 0,742 | **0,753** | 0,733 | 0,746 | 0,678 | 0,687 |
| AUC (makro OvR) | 0,9697 | **0,9676** | 0,9675 | 0,9686 | 0,9563 | 0,9552 |
| mel / bkl AUC | 0,938 / 0,958 | **0,936 / 0,959** | 0,943 / 0,956 | 0,942 / 0,957 | — | — |
| ECE-15 (ham) | 0,0745 | **0,0776** | 0,0616 | 0,0659 | 0,0309 | 0,0323 |
| ort. güven − ACC (puan) | +7,0 | **+7,0** | +5,7 | +6,1 | +2,5 | +2,0 |
| NLL | 0,518 | 0,521 | 0,526 | 0,514 | 0,518 | 0,529 |
| nv payı (gerçek 0,669) | 0,701 | **0,693** | 0,667 | 0,677 | 0,698 | 0,671 |
| molemax-dışı ACC (n = 619) | 0,800 | **0,798** | 0,786 | 0,792 | 0,730 | 0,721 |
| mel→nv / bkl→nv | 36,0 / 16,0 | **34,4 / 14,4** | 28,2 / 11,0 | 30,2 / 13,0 | 38,6 / 21,4 | 31,8 / 17,8 |
| nv→mel / nv→bkl | 12,8 / 10,0 | **22,0 / 7,2** | 22,0 / 14,0 | 19,4 / 12,4 | 21,8 / 13,2 | 31,2 / 14,6 |
| toplam hata | 133,2 | **134,2** | 142,8 | 138,4 | 178,2 | 184,8 |
| tabakalı S − N (puan) | +6,65 | **+5,24** (Δ −1,41 [−3,84; +1,23]) | +7,43 | +7,50 | +4,02 | +4,41 |
| seed çifti argmax anlaşmazlığı | %9,0 | %8,9 | %11,3 | %10,6 | %0,9 | %0,6 |

### 1c. Sınıf duyarlılığı (destek: akiec 33, bcc 52, bkl 110, df 12, mel 111, nv 671, vasc 14) ve tahmin sayıları

| sınıf | H02 | **H05** | Δ H05 − H02 [GA] | H04 | H02 tahmin sayısı | H05 tahmin sayısı |
|---|---|---|---|---|---|---|
| akiec | 0,582 | 0,552 | **−3,03 [−8,57; +1,94]** | 0,612 | 29,2 | **25,4** |
| bcc | 0,904 | 0,885 | −1,92 [−5,19; +1,00] | 0,900 | 59,0 | 56,8 |
| bkl | 0,707 | 0,718 | +1,09 [−1,55; +3,82] | 0,715 | 96,6 | 95,0 |
| df | 0,533 | 0,617 | +8,33 [+1,54; +16,67] | 0,667 | 8,0 | 10,4 |
| mel | 0,618 | 0,659 | **+4,14 [+0,97; +7,50]** | 0,661 | 96,6 | 110,4 |
| nv | 0,956 | 0,947 | −0,92 [−1,53; −0,32] | 0,929 | 703,2 | 694,8 |
| vasc | 0,643 | 0,671 | +2,86 [−2,67; +8,33] | 0,714 | 10,4 | 10,2 |

Ön-kayıt gereği df ve vasc satırları yorumlanmaz. Bu sınıflarda tek bir örnek duyarlılığı 7–8 puan oynatır.

---

## 2. Kırmızı bayrak taraması

| # | Bayrak | H04 | H05 | H06 |
|---|---|---|---|---|
| 1 | Görülmemiş eşiği | Ortalama dengeli Δ +3,63 ≤ +6,0. ACC Δ −0,96 ≤ +2,0. AUC Δ −0,22 ≤ +1,0. **Seed başına** s3 +7,62 > +6,0: beklenti kapısı İHLAL kaydı düşmüş, §3.2'de değerlendirildi → **bayrak değil** | ACC Δ −0,10 ≤ +2. Dengeli Δ +1,51 ≤ +6,5. 0,8662 < 0,90. AUC 0,968 < 0,99 → **yok** | Δ +2,72 ≤ +6. ACC −0,66. AUC −0,11 → **yok** |
| 2 | Değişiklik devreye girmemiş | Görüntü başına 42,8 karar değişti. τ\* = 0,7. τ = 0 özdeşlik farkı 0 → yok | Görüntü başına 68,2 karar değişti. Ağırlık vektörü kapıda doğrulandı (df 4,424, nv 0,578, akiec 2,620). β = 0 dumanı H02 ile bit bit aynı → yok | 31,6 karar. τ\* = 0,2 → yok |
| 3 | n < 5 | 5 | 5 | 5 |
| 4 | Ağırlık/önsel **sınıf sırası** hatası (kartın soruşturma listesi) | π sırası ✓. Bağımsız yeniden hesapta uyuşmazlık 0 | Sıra yanlış olsaydı nv ağırlığı büyür ve nv payı artardı. Gözlenen: nv payı −0,84 [−1,44; −0,24], mel tahmini 96,6 → 110,4, df 8,0 → 10,4. Yön, doğru sırayla tutarlı → **yok** | π sırası ✓ |
| 5 | İkincil bedel (M7) | **ACC −0,96 [−1,66; −0,32]**, nv duyarlılığı −2,77, molemax-dışı −1,49 [−2,53; −0,50] → **ödünleşim notu** | ECE +0,3 puan (gürültü içinde). akiec duyarlılığı −3,0 (GA 0'ı kapsıyor) → bedel yok, D-gözlemi (§3.1) | ACC −0,66 (GA 0'ı kapsıyor), nv −2,35 → ödünleşim notu |
| 6 | Beklentinin tersi + açıklanamayan mekanizma | Yok | Birincil yön beklentiyle aynı (+1,51 ∈ [+1,0; +4,0]). Yalnız ikincil akiec ↓ beklentinin tersi, GA 0'ı kapsıyor → keşifsel not, alarm değil | **İyimserlik negatif** (OOF > tam-val) → açıklandı (§3.3), arıza değil |
| 7 | t ↔ bootstrap çelişkisi (kural b') | Çelişki yok. p_t 0,0176, GA alt +1,57 > 0 | Çelişki yok. İkisi de anlamsız: p_t 0,097, GA [−0,23; +3,28] | Birincil ölçü bootstrap, GA alt +0,65 > 0 |

**Kırmızı bayrak sayısı: 0.** Ödünleşim/sarı notlar 3: H04'ün ACC bedeli ve kısıta 0,04 puanlık payı, H06'nın ACC/nv bedeli, H06'nın negatif iyimserliği (K5 beklentisini etkiliyor). Açıklanmış kapı kaydı 1: H04 s3.

---

## 3. Kart kararları

### 3.1 H05 — sınıf ağırlıklı CE (β = 0,5) → **ÖLDÜR (n = 5'te saptanamadı; MDE ≈ 2,9 puan)**

**Öldürme kuralları:**
- (a) n = 5 ✓.
- (b) Ortalama dengeli Δ +1,51 ≥ +0,5 ✓. **Kesin tek yönlü p_t = 0,097 ≥ 0,05 → ÖLDÜR tetiklendi.** ACC 0,8662 ≥ 0,8572 ✓.
- (b') Çelişki yok. Bootstrap GA [−0,23; +3,28] 0'ı kapsıyor, t de anlamsız. (Bootstrap tek yönlü p 0,041. Kartın bootstrap ölçütü GA'dır, karar onu değiştirmez.)
- (c) Canlı kural tetiklenmedi. (d) 48,4 s/epoch. (e) Seçilen epoch'lar 20/29/16/7/19, 30 yok. (f) Özdeşlik ✓.
- Görülmemiş eşiği aşılmadı.

**Neden ÖLDÜR, neden "etki yok" değil:** Kartın istatistik ön-beyanı tam olarak bu durumu öngörmüştü: MDE ≈ 3,0, gözlenen 2,88. Beklenen aralığın alt yarısı (+1…+3) bu güçle saptanamayabilirdi. Başarısız denemeler tablosuna **"n = 5'te saptanamadı (MDE ≈ 2,9 puan); yön üç seçim kuralında da pozitif"** diye girer. Kural gevşetilmez. aucsecim kolunun anlamlı çıkması (+3,06, p_t 0,036) ön-kayıtlı birincil kol olmadığı için kararı değiştirmez. Kol seçmek çatallanan yol olurdu.

**Mekanizma (D-grubu; neden küçük çıktı):**
1. **Etkin önsel kaydırması kuramsal değerin altında kaldı.** β = 0,5'in Bayes eşdeğeri τ = 0,5'tir. H02'ye seçimsiz τ = 0,5 uygulanınca nokta 0,8620 / 0,7386 çıkıyor. H05 ise 0,8662 / 0,7213'te. H05'in noktası, H02'nin tam-val τ eğrisinde **τ ≈ 0,2 noktasıyla neredeyse çakışıyor** (0,8654 / 0,7213). nv payı da aynı sırayı izliyor: H02 0,701 → H05 0,693 → τ = 0,5 0,677. Okuma: ağırlık etkisinin yaklaşık %60'ı sönüyor. Bu, Byrd & Lipton 2019'un aşırı parametreli ağ uyarısıyla uyumlu. Train kaybı 5. epoch'ta ≈ 0,017'ye iniyor (s0 `gecmis.json`).
2. **Temsil uyarlanmadı.** Kartın "post-hoc ayardan farkı" savı, azınlık özniteliklerinin uyarlanmasıydı. Eşik bağımsız ölçüler bunu desteklemiyor: AUC 0,9697 → 0,9676. Nadir sınıf AUC'ları akiec 0,959 → 0,953, df 0,983 → 0,976, vasc 0,994 → 0,994. mel AUC 0,938 → 0,936.
3. **Post-hoc sınırının üstüne çıkmıyor.** Aynı ACC'de (0,8662) H02 τ eğrisinin dengeli değeri 0,7167. H05 bunun yalnız **+0,46 puan** üstünde, seed sd'sinin çok altında. Doğrudan kıyas H05 − H02@τ=0,5: dengeli −1,73 [−4,21; +0,56], ACC +0,42 [−0,32; +1,16]. Yani eğitim-zamanı ağırlıklandırma, ödünleşim eğrisi üzerinde daha zayıf bir nokta seçiyor. Eğriyi dışa itmiyor.
4. **Epoch eğrisi:** Seed ortalamalı dengeli ACC, epoch 1–5'te 0,703, 6–15'te 0,716, 16–30'da 0,712. Son-epoch kolunda etki +1,83 [+0,22; +3,53]. Etki seçilen epoch'tan son epoch'a doğru sönmüyor. Baştan beri küçük. H02'nin `gecmis.json`'unda epoch başına dengeli ACC olmadığı için epoch-epoch kıyas yapılamadı (I-grubu sınır).
5. **akiec ↓ (D-gözlemi):** akiec tahmin sayısı 29,2'den 25,4'e düştü (ağırlık 2,62'ye rağmen), duyarlılık −3,0 [−8,6; +1,9]. GA 0'ı kapsıyor. Seed'ler farklı epoch'larda seçildiği için (7–29) sınıf bazlı gürültü büyük. Mekanizma ölçülmedi, karar etkilenmez.

**Ödünleşim:** yok. ECE 0,0776 (H02 0,0745), tabakalı S − N +5,24 (H02 +6,65; Δ GA 0'ı kapsıyor).

### 3.2 H04 — post-hoc logit ayarı (H02, τ OOF) → **TUT (ödünleşim notuyla)**

**Öldürme kuralları:**
- (a) n = 5 ✓.
- (b) Δ +3,63 ≥ +0,5 ✓. p_t 0,0176 < 0,05 ✓. OOF ACC 0,8576 ≥ 0,8572 ✓.
- (b') Bootstrap GA [+1,57; +5,95], alt sınır > 0, t ile aynı yön → **İNCELE yok**.
- (c) τ\* = 0,7 > 0. Hiçbir katta τ_k = 0 değil (0/5).
- (d) Özdeşlik farkı 0. π sırası ✓. Bağımsız OOF yeniden hesabında uyuşmazlık 0.
- (e) τ_k aralığı 0,1 ≤ 0,4.
- Görülmemiş eşiği (ortalama) aşılmadı.

**(i) s3 seed başı İHLAL kaydının değerlendirmesi.** Kartın görülmemiş eşiği ("OOF dengeli Δ > +6,0 puan") tablodaki diğer etki büyüklükleriyle aynı birimde, yani **5-seed ortalama Δ** olarak yazılmış. Öldürme kuralı (b) "5-seed ort. OOF dengeli Δ" der. Eşiğin gerekçesi olan τ = 1 zarfı (+5,4) da seed ortalamasıdır (tahlil-tur1 §6a). Seed başına uygulama, uygulama notundaki beklenti dosyasının mekanik `ikincil_aralik` kuralından geliyor. Bu kural bir **arıza alarmıdır**, karar eşiği değildir. Alarm ölçülebilir testlerle soruşturuldu:

| Hata hipotezi | Test | Sonuç |
|---|---|---|
| OOF birleştirme / kat indeksi hatası | `katlar.json` + τ_k ile H02 log p'den bağımsız yeniden hesap | 5 × 1.003 görüntüde **0 uyuşmazlık** |
| τ'nun kendi katında seçilmesi (sızıntı) | İyimserlik (tam-val − OOF) | +0,08 puan. Sızıntı OOF'u şişirirdi, OOF ≈ tam-val. s3'te tam-val τ = 0,7 Δ'sı +7,47, OOF +7,62 |
| s3'e özgü ayar | τ ortak mı? | Evet: 5 seed ortak τ, `tau_ortak_mi` = true |
| Seed'in kendi zarfını aşıyor mu? | Aynı seed'de τ = 1 Δ'sı | s3 τ = 1: **+7,70** ≥ OOF +7,62. Seçilen τ, seed'in kendi zarfı içinde |
| Aykırı seed mi? | Seed Δ dağılımında z | z = 1,54 < Grubbs kritik değeri (n = 5, α 0,05) 1,715 → aykırı değil |
| Kazancın kaynağı | s3 sınıf ayrışımı (doğru sayısı H02 → H04) | df 7 → 10/12 (**+3,57 puan**), vasc 10 → 13/14 (**+3,06**), mel 74 → 80 (+0,77), akiec +0,43, bcc +0,27, nv 629 → 606 (−0,49). +7,62'nin **6,6 puanı df/vasc'taki 6 görüntüden** geliyor |

**Sonuç:** arıza yok, eşik ihlali de yok. Kart eşiği ortalama içindir, ortalama +3,63. Seed başına uç değer, destekleri 12 ve 14 olan iki sınıfın örnek ayrıklığından geliyor. **İNCELE gerekmez.** Bu bir eşik gevşetmesi değil: kartın kendi metni ortalamayı tanımlıyor. Beklenti kapısının seed başı kuralı alarm olarak çalıştı ve teşhisle kapandı. **Öneri (gelecek beklenti dosyaları):** görülmemiş eşiklerini kart birimine (5-seed ort. Δ) bağlayın. Seed başı bant istenirse, kaynağı "seed başına τ = 1 zarfı" gibi ölçülmüş bir değer olarak yazın. Günlük arıza satırı §9'da.

**(ii) ACC kısıtının 0,04 puanlık payı.** OOF ACC 0,8576, kısıt 0,8572. Pay 0,043 puan. Bu, 5 seed toplamında ≈ **2,2 görüntü kararı**, seed başına < 0,5 görüntü demek. Okuma:
- **Tasarım gereği sınırda:** τ_k, seçim kümelerinde "ACC̄(τ) ≥ ACC̄(0) − 0,010" kısıtının izin verdiği en büyük dengeli değere gider. OOF ACC'nin −1,0 puan sınırının hemen yakınına düşmesi beklenen bir sonuçtur, şans değildir.
- **Bootstrap belirsizliği:** ΔACC = −0,96 [−1,66; −0,32]. GA kısıt sınırını (−1,0) **kapsıyor**. Bootstrap dağılımında P(ΔACC < −1,0) = **0,41**. Seed bazında 5'in 2'si sınırın altında (s0 −1,69, s1 −1,50).
- **Anlamı:** ön-kayıtlı kural nokta tahmini üzerindedir (ort. OOF ACC ≥ 0,8572), mekanik olarak geçer. Ama "ACC bedeli ≤ 1 puan" **istatistiksel olarak gösterilmiş değildir** (eşdeğerlik/non-inferiority ön-kayıtlı değildi). ACC bedelinin varlığı ise gösterilmiştir (GA üst −0,32 < 0). K5 test'te ACC bedeli ≈ %40 olasılıkla 1 puanı aşabilir.
- **Makale cümlesi (I-grubu):** "...ACC'yi ≈ 1 puan (GA 0,3–1,7) düşürerek" yazılır, "≤ 1 puan düşürerek" yazılmaz.
- **İNCELE'ye gerek yok:** t ↔ bootstrap çelişki kuralı birincil dengeli Δ içindir ve orada çelişki yok. Kısıt için ayrı bir belirsizlik kuralı ön-kayıtlı değildi. Sonradan eklemek, eşiği sonuca göre oynatmak olur.

**Ödünleşim (M7):** ACC −0,96, nv duyarlılığı −2,77, molemax-dışı ACC −1,49 [−2,53; −0,50], nv→mel 12,8 → 22,0. Kazanç: mel +4,32 [+2,42; +6,49], akiec +3,03, mel→nv 36,0 → 28,2. ECE iyileşti (0,0745 → 0,0616). Oynaklık: dengeli seed std'si 1,58 → 3,68 puan.

**Ablasyon durumu:** (i) τ = 1 ✓ (ACC 0,8445 / dengeli 0,7605). **(ii) Yalnız-nv kaydırması (çapraz uydurmalı) henüz koşulmadı.** Kart bunu Sonuç bölümünde istiyor. K5 öncesi deney-yürütücü CPU'da koşmalı (dakikalar).

### 3.3 H06 — logit ayarı H01 sondasında (analiz kartı) → **"Genelleşir"**

Ölçütlerin hepsi sağlanıyor: ort. OOF dengeli Δ +2,72 ≥ +0,5. GA alt +0,65 > 0. Ort. OOF ACC 0,8158 ≥ 0,8123. Tam-val τ\* = 0,2 > 0. Görülmemiş eşiği yok. Operasyonel DUR yok (`katlar.json` SHA256 eşit, özdeşlik 0).

- **Ön-kayıtlı keşifsel yön** (kısıt altındaki kazanç H04 > H06): Δ(H04) − Δ(H06) = +0,91 [−1,47; +3,42]. Yön tutuyor, ayırt edilemiyor.
- **Negatif iyimserlik (−1,27 puan; D-grubu, açıklandı).** H01'in tam-val τ eğrisi basamaklı (seçimsiz, betimsel):

  | τ | 0 | 0,1 | 0,2 | 0,3 | 0,4 |
  |---|---|---|---|---|---|
  | ACC | 0,8223 | 0,8221 | 0,8136 | 0,8112 | 0,8046 |
  | dengeli | 0,6494 | 0,6633 | 0,6639 | 0,6826 | 0,6948 |

  Tam-val kısıtı (0,8123) τ = 0,2'de bağlıyor. Dengeli ACC'deki asıl basamak ise 0,2 → 0,3 arasında (+1,9 puan). Kat 4 ve 5'in seçim kümeleri τ = 0,3'e izin verdi, OOF bu yüzden 0,1 ile 0,3'ü karıştırıyor. Arıza değil, nadir sınıflarda birkaç görüntünün karar değiştirmesinden doğan ayrıklık.
- **K5 için sonucu:** test'e tam-val τ\* = 0,2 uygulanacak. Val'deki eşdeğeri **dengeli +1,45 / ACC −0,87**, OOF'un +2,72'si değil. K5 beklentisi bu değere göre yazılmalı. Makalede "Genelleşir" OOF'a dayanır, test Δ'sı ayrıca raporlanır.
- **Ödünleşim:** ACC −0,66 [−1,60; +0,32], nv −2,35, nv→mel 21,8 → 31,2. ECE değişmedi (0,0309 → 0,0323).
- H01 + ayar ile H02 (ayarsız) arasındaki dengeli açık −2,96 [−9,01; +3,73] (Tur-1'de −5,68). Açığın yaklaşık yarısı yalnız karar sınırından geliyor. ACC açığı −5,14 [−7,34; −2,85] sürüyor.
- **Ablasyon:** (i) τ = 1 ✓ (0,7198 / 0,7317). (ii) Yalnız-nv **eksik** (H04 ile aynı iş).

---

## 4. H04 ↔ H05: aynı ebeveyn, aynı amaç (post-hoc ↔ eğitim-zamanı) ve bileşim

| | H04 (post-hoc, τ OOF) | H05 (eğitim-zamanı, β = 0,5) | H02@τ=0,5 (post-hoc, seçimsiz; teşhis) |
|---|---|---|---|
| Δ dengeli vs H02 | **+3,63 [+1,57; +5,95]**, p_t 0,018 | +1,51 [−0,23; +3,28], p_t 0,097 | +3,24 [+1,26; +5,61] |
| Δ ACC vs H02 | −0,96 [−1,66; −0,32] | −0,10 [−0,78; +0,58] | −0,52 [−1,06; −0,02] |
| nv payı | 0,667 | 0,693 | 0,677 |
| AUC | 0,9675 | 0,9676 | 0,9686 |
| ECE (ham) | 0,0616 | 0,0776 | 0,0659 |
| Maliyet | **0 GPU**, ≈ 50 s CPU | **2,2 GPU-sa** | 0 |

**Okuma:**
1. **Doğrudan kıyas** H05 − H04: dengeli −2,12 [−4,56; +0,16], ACC +0,86 [+0,10; +1,62]. İkisi ödünleşim eğrisinin farklı noktalarında. "Hangisi daha iyi" tek ölçütle sorulamaz.
2. **Aynı ACC'de kıyas** (H02'nin tam-val τ eğrisi, betimsel): H05, eğrinin yalnız +0,46 puan üstünde. Etkin önsel kaydırması τ ≈ 0,2. **"Önsel kaydırmanın eğitim sırasında yapılması, post-hoc yapılmasına bir şey katıyor mu?" sorusunun cevabı, bu veri ve tarifte hayırdır:** AUC aynı, sınır aynı, maliyet 2,2 GPU-sa'e karşı sıfır. Bu, makalenin tartışma bölümü için temiz bir **olumsuz bulgu** (I-grubu: başarısız denemeler tablosuna mekanizmasıyla birlikte girer).
3. **Bileşim (H05 + logit ayarı) Tur-3 adayı mı? Hayır.**
   - (a) **Ön-kayıt:** SIRALAMA "H04 + H05 birleşimi yalnız ikisi de TUT alırsa" diyor. H05 ÖLDÜR aldığı için birleşim kartı açılamaz (M2).
   - (b) **Tek değişken:** H05 ebeveyn alınırsa birleşim tek değişkenli olurdu (yalnız τ eklenir). Ama ebeveyn kuralı H05'i ebeveyn yapmıyor (§6a). H02'ye göre ise iki değişken değişir.
   - (c) **Beklenen değer:** H05 post-hoc sınırının üstüne ancak ≈ 0,5 puan çıkıyor. H05 + τ'nun H04'e göre beklenen kazancı bu düzeyde, MDE'nin (≈ 3,5) çok altında. Yani ölçülemez.
   - Bu soruyu zaten Tur-3 kuralı karşılıyor: H04 protokolü bir **değerlendirme katmanı** olarak her Tur-3 varyantına ikinci satırda uygulanacak.
   - **Not (M1):** H05'in τ eğrisi bu tahlilde **hesaplanmadı**. Gelecekte bir bileşim kartı açılırsa val bilgisi temiz kalır.

---

## 5. R-T — sıcaklık ölçekleme raporlaması (ön-kayıtlı protokol; kart değil)

| Kol | ECE ham | **ECE OOF** (± std, seed) | ECE tam-val (iyimser) | T (tam-val, seed) | NLL ham → OOF | güven − ACC ham → OOF |
|---|---|---|---|---|---|---|
| **H02** | 0,0745 | **0,0310 ± 0,0091** (0,032 / 0,027 / 0,044 / 0,032 / 0,019) | 0,0315 | 1,48–1,86 | 0,518 → 0,425 | +7,0 → +0,5 puan |
| H04 (OOF çıktısı) | 0,0616 | **0,0285 ± 0,0034** | 0,0284 | 1,26–1,58 | — | — |
| H01 | 0,0309 | **0,0201 ± 0,0014** | 0,0192 | 1,19 (5/5) | 0,518 → 0,509 | +2,5 → −0,6 |
| Baseline | 0,0365 | **0,0325 ± 0,0042** | 0,0362 | 1,06–1,19 | — | — |

- **Ön-kayıtlı kural:** H02 OOF ECE 0,0310 ∈ beklenti [0,025; 0,045] ve ≤ 0,045 → makale "**ödünleşim ölçeklemeyle giderilir**" der. Ölçekli H02 (0,031), ölçekli baseline (0,0325) ve ham H01 (0,031) düzeyindedir. Görülmemiş eşiği (< 0,010) hiçbir kolda yok. ACC ve dengeli ACC değişimi 20/20 koşumda 0 (argmax korunuyor).
- **Not:** H02 s2'nin OOF ECE'si 0,044, kuralın sınırına (0,045) yakın. 15 kutulu ECE'nin n = 1.003'teki gürültü tabanı ≈ 0,02–0,03 (SIRALAMA R-T). Makale seed aralığını da yazmalı.
- **Göreli düşüş:** H02'de %58 (Guo vd. 2017'nin %66–92'sinin altında, gürültü tabanı nedeniyle beklenen yönde). H04 için %54.
- H05 TUT almadığı için R-T H05'e uygulanmadı (SIRALAMA). Ham ECE'si 0,0776 ile H02 düzeyinde.

---

## 6. Tur-3 ebeveyni, öneriler ve K5 hazırlığı

### 6a. Tur-3 ebeveyni (SIRALAMA "Tur-3 ebeveyn kuralı", mekanik)
- Kuralın istisnası: H05 TUT alırsa **ve** val ACC'si H02'ninkine eşit ya da yüksekse H05 ebeveyn olur. H05 ÖLDÜR aldı, ACC'si de düşük (0,8662 < 0,8672). İstisna iki koşuldan da düşüyor → **eğitim tarifi ebeveyni: `h02_r18in_ft224`.**
- **H04 TUT aldı.** H04 protokolü (yeniden çapraz uydurmalı τ, aynı `katlar.json`) her Tur-3 varyantına ve ebeveynine **ikinci satır** olarak uygulanır. Tur-3 kartlarının birincil kararı ham çıktıyla verilir.
- **R-T** da aynı biçimde ikinci satırda uygulanır (kalibrasyon raporu).

### 6b. Tur-3 önerileri (kart DEĞİL; sıradaki boş id **H07**; Z4 + Z1 kalanı; ACC-hedefli)

| Öncelik | Öneri | Aile / Z | Ebeveyn | Değişen tek şey | Ön-kayıt adayı beklenti (kaynak) | Görülmemiş | Maliyet |
|---|---|---|---|---|---|---|---|
| **1** | **H07 — TTA (8 dihedral görünüm, olasılık ortalaması)** | A8 / Z4 | H02 (`best_model.pth` ×5) | Yalnız çıkarım. Çevirme ve 90° döndürmeyle 8 görünümün softmax ortalaması. Eğitim bit bit aynı | ACC **0…+2 puan** (literatur-cipa §4(d): Medeiros 2026 T2 +1,6, ama sıfırdan 28 px R18 ve ön baskı; ince ayarlı 224 için kaynak yok). ECE ↓ beklenir (ortalama güveni düşürür). Seed'ler arası anlaşmazlık ↓ | ACC Δ **> +3** (§4(d)) | GPU ≈ 5 × 8.024 görüntü / 214 img/s ≈ 3–4 dk → **≈ 0,1 GPU-sa** |
| **2** | **H08 — eğitim augmentasyonu (dihedral + hafif renk)** | A3 / Z4 (+ Z1) | H02 tarifi | Train'de rastgele dihedral + renk oynaması (parlaklık/kontrast/doygunluk küçük genlikle). 30 ep, seçim val ACC, geri kalan her şey aynı. GPU üstünde uygulanmalı, aksi halde epoch ≤ 90 s kuralı riske girer | **Kaynak yok** (§4(d): DermaMNIST'te izole etki bulunamadı). Ön-kayıtta yön belirsiz, aralık geniş yazılmalı. Mekanizma gerekçesi ölçülmüş: train kaybı ≈ 0 (ep ≥ 5), ECE × 2, tabakalı S − N +6,65 (ezber sinyali), molemax-dışı 0,800 ↔ molemax 0,97. Dolaylı: Zhang & Li 2026 T1d 224 CE 0,882 (test). Augmentasyon kullanıp kullanmadığı kart yazımında kaynaktan doğrulanmalı | ACC Δ > +3 **veya** val ACC > 0,90 (H02 tavanı) | **≈ 2,3–2,6 GPU-sa** (+ duman/ön-uçuş) |
| (3) | Seed topluluğu (A9) | A9 / Z3 | H02 | 5-seed olasılık ortalaması (teşhis +1,1 ACC, görülmüş) | — | — | 0 GPU. **Tur-5'te kalsın:** 5-seed topluluğu n = 1 verir, seed tekrarı olmadan istatistik tasarımı (ör. ayrık seed alt kümeleri ya da yalnız bootstrap) önce ön-kayıtla çözülmeli. Rakibe aynı hak verilmeli (M3) |

Notlar:
- **Sıra:** önce H07, sonra H08. H07 en ucuz iş ve Tur-3'ün ACC sorusuna doğrudan cevap veriyor. H08 tek GPU işi, paylaşımsız koşulmalı (R6).
- **TTA riski** (ön-kayıtta yazılmalı): H02 augmentasyonsuz eğitildi. Döndürülmüş görünümler modelin dağılımının dışında olabilir. Medeiros'ta 12 kombinasyonun 11'inde TTA zarar veriyor. Dermoskopide yön klinik olarak anlamsız olduğu için dihedral değişmezlik makul, ama ölçülmedi. **H07 ↔ H08 bileşimi** (augmentasyonla eğitilmiş modelde TTA) ancak ikisi de TUT alırsa ayrı kart olarak açılır (M2).
- **İstatistik:** H07'nin iki kolu aynı modelden geliyor. t testinin varyansı küçük kalır, bu yüzden bağlayıcı belirsizlik görüntü bootstrap'ı olmalı (H04 ön-beyanı ile aynı mantık). H08 yeniden eğitim olduğundan MDE ≈ 0,85 puan ACC (H02 seed sd 0,28, gözlenen sd_Δ ≈ 0,63). Görüntü bootstrap GA yarı genişliği ≈ ±0,7 puan.
- **Hedef ölçüler** (birincil val ACC, ikincil ön-beyan): `val_acc_molemax_disi` (0,800), mel→nv (36,0), tabakalı S − N (H08 için keşifsel yön ↓), ECE, H04 katmanlı dengeli ACC.
- **Durma sayacının Tur-3 eşikleri** Tur-3 ön-kaydına yazılmalı. ACC ölçütü (ham) ≥ **0,8722**. Dengeli ölçüt ≥ **0,7475** (en iyi dengeli = H04 OOF 0,7425 + 0,5). Elmayla elma karşılaştırması için Tur-3'ün dengeli değeri **H04 katmanlı OOF satırından** okunmalı. Bu okuma kuralı şef/tasarımcı kararıdır ve koşumdan önce yazılmalıdır.

### 6c. K5 hazırlığı — ablasyon zinciri (şu ana dek) ve kazanan zincir

| Basamak | Varyant | val ACC | val dengeli | Δ (önceki basamağa) | Durum |
|---|---|---|---|---|---|
| 0 | Rakip: R18, 28 px, sıfırdan (B) | 0,7396 | 0,4621 | — | replikasyon YEŞİL (test 0,7296, donduruldu) |
| 1a | + ImageNet önağırlık + native 224 + ince ayar tarifi (**H02**) | **0,8672** | 0,7062 | +12,76 / +24,42 | TUT. **Paket olarak**: üç bileşen ayrıştırılmadı (sınırlılık) |
| 1b | DINOv2-B donuk sonda r224 (H01) | 0,8223 | 0,6494 | +8,28 / +18,74 (B'ye göre) | TUT. Çözünürlük merdiveni H03 (r28 → r224 +4,85) |
| 2a | H02 + post-hoc logit ayarı τ\* = 0,7 (**H04**) | 0,8576 (OOF) | **0,7425** (OOF) | −0,96 / +3,63 | TUT (ödünleşim). Abl. (i) τ = 1 ✓, (ii) yalnız-nv **eksik** |
| 2b | H01 + logit ayarı τ\* = 0,2 (H06) | 0,8158 | 0,6766 | −0,66 / +2,72 | Genelleşir. Abl. (ii) **eksik** |
| 2c | H02 + sınıf ağırlıklı CE (H05) | 0,8662 | 0,7213 | −0,10 / +1,51 (ns) | ÖLDÜR → başarısız denemeler tablosu |
| 3 | + sıcaklık ölçekleme (R-T) | = | = | ECE 0,0745 → 0,031 (H04: 0,062 → 0,029) | rapor satırı |

- **Kazanan zincir (bugün):** ACC için **B → H02** (0,8672). Dengeli ACC için **B → H02 → H04** (0,7425 OOF) ve kalibrasyon için **+ R-T**. Tur-3 ACC'yi iyileştirirse zincir B → H02 → (H07/H08) → H04 katmanı → R-T olur.
- **K5'te test'e bir kez:** baseline (donduruldu); H02 5 × `best_model.pth`; H04 τ\* = 0,7 aynı olasılıklara; R-T tam-val T (seed başına). H01 test öznitelikleri + C\* = 0,1 sonda; H06 τ\* = 0,2. DermaMNIST-C (kullanıcı kararı 3): baseline + H02 (+ H01). H05 test'e götürülmez (başarısız deneme val'de raporlanır, ek test bakışı gerekmez).
- **D5 paketi eksikleri:** (1) `gunluk/deney-gunlugu.md` Tur-2 satırları ve bütçe sayacı (hâlâ 7,5) → şef. (2) H04 ve H06 için yalnız-nv ablasyonu (ii) → deney-yürütücü, CPU. (3) Başarısız denemeler tablosu taslağı (H05 + Tur-1'den r28 nv notu) → K5'te tahlilci. (4) H02 paketinin ayrıştırılmaması → sınırlılıklar bölümü; makale bileşen iddiası kurmamalı.

---

## 7. Durma kuralı sayacı (konu.md 2026-10-10 20:15, iki ölçüt; SIRALAMA okuma kuralı)

Kurala göre Tur-2'nin en iyisi yalnız **TUT alan H04/H05** arasından okunur. H04 için OOF değeri kullanılır. H06 ve R-T sayılmaz. H05 TUT almadı.

| | Tur-2 öncesi en iyi | Tur-2 en iyi (TUT) | İyileşme | Eşik | ≥ 0,5 puan? |
|---|---|---|---|---|---|
| val ACC | 0,8672 (H02) | 0,8576 (H04 OOF) | −0,96 | ≥ 0,8722 | hayır |
| val dengeli ACC | 0,7062 (H02) | **0,7425 (H04 OOF)** | **+3,63** | ≥ 0,7112 | **evet** |

**Sayaç: 0 / 2.** Tur-3 eşikleri §6b'de: ACC ≥ 0,8722, dengeli ≥ 0,7475.

## 8. Bütçe görüşü

- **Tur-2 GPU:** H05 eğitim süreleri toplamı (`sure_sn`) 7.924 s = **2,20 sa**. Ayrıntı: 4 duman 184 s (a1 iki kuyrukta, `tur2-h05-on` ve `tur2-h05`, iki kez koşmuş; §9), ön-uçuş 475 s, tam koşum 7.265 s. Yükleme payıyla ≈ **2,3 GPU-sa**. Bu, kartın tahminiyle (2,3; üst 2,6) aynı. CPU kolları (H04/H06/R-T/tahlil) ihmal edilebilir.
- **Sayaç:** ≈ 7,5 + 2,3 = **≈ 9,8 / 32 GPU-sa**, **2 / 8 tur**. Kalan ≈ 22 GPU-sa, 6 tur. Durma kuralına yaklaşılmadı.
- **Kalan bütçenin en iyi kullanımı:**
  - Tur-3: H07 (≈ 0,1) + H08 (≈ 2,5) = ≈ 2,6 GPU-sa.
  - K5 rezervi ≈ 1,5 GPU-sa: test çıkarımları, DermaMNIST-C 224 değerlendirmesi, H01 test öznitelikleri.
  - Tur-4 (Z1 kalanı, DINOv2 kısmi ince ayarı) en pahalı kalem. Tasarımcıya öneri: son k blok eğitilecekse, donuk ilk 12 − k bloğun çıktı token'ları **önbelleğe alınabilir** (r224'te 257 × 768 × 7.007 float16 ≈ 2,8 GB). Böylece kısmi ince ayar yerelde makul sürede koşabilir. Ölçülmedi, yalnız fizibilite notu.
- **Getiri gözlemi:** Tur-2'de 2,2 GPU-sa'lik tek GPU kartı (H05) saptanamadı, 0 GPU'lu post-hoc kart (H04) TUT aldı. Kalan Z2 kütlesi (nadir sınıflarda AUC 0,95–0,99, duyarlılık 0,55–0,71) artık karar katmanında değil, örnek sayısında (df 12, vasc 14) sınırlı. Bu yüzden Z2'ye yeni GPU harcaması önerilmiyor.

## 9. Günlük için satır önerileri (`gunluk/deney-gunlugu.md`)

**Tur tablosu (Tur-2):**

| Tur | Tarih | Hipotez | Ebeveyn → Varyant | Koşum id'leri | Ana metrik (ort±std, n) | Δ vs ebeveyn | Karar | Öğrenilen |
|---|---|---|---|---|---|---|---|---|
| 2 | 2026-10-11 | H04 | h02 → h04_h02_logitayar | 20261011-011434…011512 (5) | OOF dengeli 0,7425±0,0368; OOF ACC 0,8576±0,0057 | dengeli +3,63 [+1,57; +5,95] p_t 0,018; ACC −0,96 [−1,66; −0,32] | TUT (ödünleşim) | τ\* = 0,7 kararlı (aralık 0,1). ACC kısıtı nokta tahmininde 0,04 puanla sağlandı, GA sınırı kapsıyor |
| 2 | 2026-10-11 | H05 | h02 → h05_r18in_ft224_cw05 | 20261011-013813…031643 (5) | dengeli 0,7213±0,0103; ACC 0,8662±0,0047 | dengeli +1,51 [−0,23; +3,28] p_t 0,097; ACC −0,10 | ÖLDÜR (n = 5'te saptanamadı, MDE 2,9) | Etkin τ ≈ 0,2 (kuramsal 0,5). AUC değişmedi. Post-hoc sınırının ancak +0,46 puan üstünde |
| 2 | 2026-10-11 | H06 (analiz) | h01 → h06_h01_logitayar | 20261011-011758…011837 (5) | OOF dengeli 0,6766±0,0030; ACC 0,8158 | +2,72 [+0,65; +5,67]; ACC −0,66 | Genelleşir | Tam-val τ\* = 0,2'de val Δ yalnız +1,45 (basamaklı eğri) → K5 beklentisi buna göre |
| 2 | 2026-10-11 | R-T (rapor) | H02/H01/B/H04 → rt_* | 20261011-011856…012039 (20) | H02 ECE 0,0745 → OOF 0,0310±0,0091 | — | "ödünleşim ölçeklemeyle giderilir" | T 1,48–1,86; argmax 20/20 korunuyor |

**Arıza / beklenti ihlali tablosu:**

| Tarih | Belirti | Hipotez | Teşhis yöntemi | Kök neden | Düzeltme | Etkilenen koşumlar |
|---|---|---|---|---|---|---|
| 2026-10-11 01:15 | `tur2-h04` a3 kapısı İHLAL: s3 OOF dengeli 0,8026 > 0,7662 ve Δ +7,62 > +6,0 (seed başına) | OOF birleştirme / kat sızıntısı / π sırası / seed'e özgü ayar | Bağımsız OOF yeniden hesabı (0/5.015 uyuşmazlık); iyimserlik +0,08; ortak τ; s3 τ = 1 zarfı +7,70; z 1,54 < 1,715; sınıf ayrışımı (df +3, vasc +3 görüntü = 6,6 puan) | **Arıza yok.** Kapı, kartın 5-seed ortalaması için yazılmış eşiğini seed başına uyguladı. Uç değer df/vasc ayrıklığından geliyor | Kod düzeltmesi yok. Sonraki beklenti dosyalarında görülmemiş kuralı kart birimine (ort. Δ) bağlanır, seed başı bant ancak ölçülmüş bir kaynakla | Yok (5 koşum geçerli) |
| 2026-10-11 01:22–01:29 | H05 a1 dumanı iki kez koşmuş (`tur2-h05-on` ve `tur2-h05`; cw00/cw05 × 2) | Kuyruk tekrarı | `runs/` zaman damgaları, kapı raporları `beklenti-tur2-h05-on-asama1.md` ve `beklenti-tur2-h05-asama1.md` | Aynı a1 aşaması iki kuyrukta yer aldı. Sonuçlar bit bit aynı (0,8116 / 0,7866), yani seed bağlı ve deterministik | Yok. ≈ 3 dk GPU kaybı | Yok |

---

## 10. Tahlil defteri (S/D/I)

| grup | bulgu | durum |
|---|---|---|
| S | Etiket hizası (9 kol × 5 seed = npz), seed kümeleri {0..4} eşit, test yokluğu, bootstrap ↔ referans betik (8/8), H04 OOF bağımsız yeniden hesap (0 uyuşmazlık), H05 a1 tekrarının bit bit aynı çıkması | geçti |
| D | H05 etkisinin kuramsal değerin altında kalması | açıklandı (etkin τ ≈ 0,2; AUC değişmedi; Byrd & Lipton ile uyumlu) |
| D | H04 s3 seed başı uç değer | açıklandı (df/vasc ayrıklığı; arıza yok) |
| D | H06 negatif iyimserlik | açıklandı (basamaklı τ eğrisi, kısıt τ = 0,2'de bağlıyor) |
| D | H05 akiec ↓ (beklentinin tersi, ikincil) | GA 0'ı kapsıyor; mekanizma ölçülmedi (keşifsel) |
| I | H04'ün ACC bedeli "≤ 1 puan" diye yazılamaz (GA −1,66…−0,32) | makale cümlesi önerildi (§3.2) |
| I | Eğitim-zamanı ağırlıklandırması post-hoc ayara bir şey katmıyor | olumsuz bulgu, başarısız denemeler tablosu (§4) |
| I | H06'nın K5 test beklentisi tam-val τ\* = 0,2'ye göre (+1,45), OOF'a göre değil | K5 notu (§3.3) |
| I | H02 paketi ayrıştırılmadı | sınırlılık (§6c) |

**Bu tahlilde yeni görülen val bilgisi (M1 beyanı; gelecekteki kartlar için):**
- H02 tam-val τ eğrisi (11 nokta, §4) ve H02@τ = 0,5 noktası.
- H01 tam-val τ eğrisi (τ ≤ 0,4, §3.3).
- H05'in seçilmiş/son/aucsecim kolları.
- H05 üzerinde τ eğrisi **hesaplanmadı**.

**Sınırlılıklar:** 5 seed aynı 1.003 val görüntüsünü paylaşıyor. Bootstrap görüntü örneklemini yansıtır. H02@τ=0,5 ve τ eğrileri seçimsiz betimsel noktalardır, kart sonucu değildir. df/vasc duyarlılıkları yorumlanmaz. H02'nin epoch başına dengeli ACC'si kayıtlı değil. Tabakalı S − N ölçüsü N'nin %41,8'ini kapsıyor.

---

## 11. Kart başına hazır sonuç paragrafları

> Kart dosyalarının "Sonuç" bölümleri bu tahlilde düzenlenmedi. H04 ve H06'nın ölçüm satırları deney-yürütücü tarafından zaten dolduruldu, aşağıda yalnız **Karar** ve **Öğrenilen** satırları var. H05'in bölümü boş, paragrafın tamamı aşağıda.

### H04 (Karar + Öğrenilen satırları)
> - Karar: **TUT (ödünleşim notuyla)**. (a) n = 5. (b) Δ +3,63 ≥ +0,5, p_t 0,0176, OOF ACC 0,8576 ≥ 0,8572. (b') bootstrap GA [+1,57; +5,95] t ile tutarlı. (c) τ\* = 0,7, 0/5 katta τ_k = 0. (d) Özdeşlik 0, bağımsız OOF yeniden hesabında 0 uyuşmazlık. (e) τ_k aralığı 0,1. Görülmemiş eşiği (5-seed ort.) aşılmadı. a3 kapısındaki s3 seed başı İHLAL kaydı teşhisle kapandı (`tahlil-tur2.md` §3.2): arıza yok. s3'ün τ = 1 zarfı +7,70, z = 1,54, kazancın 6,6 puanı df/vasc'taki 6 görüntüden geliyor. **Ödünleşim:** ACC −0,96 [−1,66; −0,32]. Kısıt nokta tahmininde 0,04 puanla sağlandı, ancak GA sınırı kapsıyor (P(Δ < −1,0) ≈ 0,41). nv duyarlılığı −2,77, molemax-dışı ACC −1,49. ECE 0,0745 → 0,0616 (R-T OOF ile 0,0285). Ablasyon (ii) (yalnız-nv) K5 öncesi koşulacak.
> - Öğrenilen: Aşırı güvenli, ince ayarlı R18'de çapraz uydurmayla seçilen τ = 0,7, dengeli doğruluğu 3,6 puan artırıyor. Bedel ≈ 1 puan ACC (GA 0,3–1,7). Kısıt τ'yu sınıra ittiği için ACC bedeli "≤ 1 puan" diye değil "≈ 1 puan" diye raporlanmalı. Kazancın seed'ler arası oynaklığı (sd_Δ 2,6) nadir sınıfların örnek ayrıklığından geliyor. τ seçimi bölmeye karşı kararlı. Aynı kaydırmanın eğitim sırasında yapılması (H05) bu kazancı ne aşıyor ne de yakalıyor.

### H05 (Sonuç bölümünün tamamı)
> - Koşumlar (GPU GTX 1650, rc=0, kuyruk `tur2-h05`): duman `20261011-012209_h05_r18in_ft224_cw00_duman_s0` (β = 0 özdeşliği: ilk adım kaybı, ilk epoch kaybı ve val_acc farkı 0), `20261011-012324_h05_r18in_ft224_cw05_duman_s0` (a1, `tur2-h05-on`'da bit bit aynı tekrarı: `012718`/`012826`). Ön-uçuş `20261011-012934_…_cw05_onucus_s0`, `-013350_…_s1`. Tam: `20261011-013813_h05_r18in_ft224_cw05_s0`, `-020249_…_s1`, `-022727_…_s2`, `-025205_…_s3`, `-031643_…_s4`. Seçilen epoch 20/29/16/7/19. 48,4 s/epoch. Toplam ≈ 2,2 GPU-sa.
> - Ölçülen (val, n = 5, H02'ye karşı seed eşleşmeli): **dengeli ACC 0,7213 ± 0,0103** (H02 0,7062 ± 0,0158), **Δ +1,51 puan**. Seed Δ −0,35 / +1,94 / +4,30 / −0,96 / +2,61 (sd 2,16). Kesin tek yönlü p_t = **0,097**. Bootstrap %95 GA **[−0,23; +3,28]**. **ACC 0,8662 ± 0,0047**, Δ −0,10 [−0,78; +0,58] (kısıt 0,8572 ✓). aucsecim kolu: dengeli 0,6856 → 0,7162 (Δ +3,06 [+1,17; +5,27], p_t 0,036), ACC 0,8588 → 0,8602 (karar vermez). Son-epoch kolu: Δ dengeli +1,83 [+0,22; +3,53]. İkinciller: AUC 0,9697 → 0,9676. Nadir sınıf AUC'ları değişmedi (akiec 0,959 → 0,953, df 0,983 → 0,976, vasc 0,994 → 0,994). Makro-F1 0,742 → 0,753. ECE 0,0745 → 0,0776. nv payı 0,701 → 0,693. Duyarlılık: mel 0,618 → 0,659 (+4,1 [+1,0; +7,5]), akiec 0,582 → 0,552 (−3,0 [−8,6; +1,9]), nv 0,956 → 0,947, bkl 0,707 → 0,718 (df 0,533 → 0,617 ve vasc 0,643 → 0,671 yorumlanmaz). mel→nv 36,0 → 34,4, nv→mel 12,8 → 22,0. Molemax-dışı 0,800 → 0,798. Tabakalı S − N +6,65 → +5,24 (Δ GA 0'ı kapsıyor). Epoch başına dengeli ACC (seed ort.): ep 1–5 0,703, 6–15 0,716, 16–30 0,712. **H04 ile kıyas (keşifsel):** H05 − H04 dengeli −2,12 [−4,56; +0,16], ACC +0,86 [+0,10; +1,62]. H02'nin seçimsiz τ = 0,5 noktasına karşı dengeli −1,73 [−4,21; +0,56]. Aynı ACC'de H02 τ eğrisinin +0,46 puan üstünde.
> - Karar: **ÖLDÜR** — kural (b): kesin tek yönlü p_t 0,097 ≥ 0,05. (b') çelişkisi yok (bootstrap GA da 0'ı kapsıyor). Görülmemiş eşiği yok, kırmızı bayrak yok. Ön-beyan gereği başarısız denemeler tablosuna **"n = 5'te saptanamadı (MDE ≈ 2,9 puan)"** diye girer, "etki yok" diye girmez. Yön üç seçim kuralında da pozitif (+1,5 / +1,8 / +3,1). Tur-3 ebeveyn istisnası düşer (ACC 0,8662 < 0,8672). H04 + H05 birleşim kartı açılmaz (M2: ikisi de TUT değil).
> - Öğrenilen: Kare kök ters frekans ağırlıklı CE (β = 0,5), Bayes eşdeğeri olan τ = 0,5 post-hoc kaydırmanın ancak ≈ 0,2'si kadar etkili. Ağırlık etkisi, train kaybının 5. epoch'ta ≈ 0'a indiği aşırı parametreli ağda büyük ölçüde sönüyor (Byrd & Lipton 2019). Temsil nadir sınıflara doğru uyarlanmıyor (AUC değişmedi). Sonuç, post-hoc ödünleşim eğrisinin üzerinde bir nokta. Bu veri ve tarifte, önsel kaydırmanın eğitim sırasında yapılması 2,2 GPU-sa'e mal olup post-hoc ayara (0 GPU) bir şey katmıyor.

### H06 (Karar + Öğrenilen satırları)
> - Karar: **"Genelleşir"**. Ort. OOF dengeli Δ +2,72 ≥ +0,5. Bootstrap GA alt +0,65 > 0. Ort. OOF ACC 0,8158 ≥ 0,8123. Tam-val τ\* = 0,2 > 0. Görülmemiş eşiği ve operasyonel DUR yok. Ön-kayıtlı keşifsel yön (H04 > H06) tutuyor ama ayırt edilemiyor (+0,91 [−1,47; +3,42]). **Not:** iyimserlik negatif (−1,27). H01'in τ eğrisi basamaklı: tam-val kısıtı τ = 0,2'de bağlıyor (dengeli 0,6639), dengeli basamağı 0,2 → 0,3'te (+1,9). K5'te test'e uygulanacak τ\* = 0,2'nin val karşılığı dengeli +1,45 / ACC −0,87. Test beklentisi OOF'a (+2,72) göre değil, buna göre okunmalı. Ödünleşim: ACC −0,66 [−1,60; +0,32], nv duyarlılığı −2,35. ECE değişmedi. Ablasyon (ii) (yalnız-nv) K5 öncesi koşulacak.
> - Öğrenilen: Önsel ayarının dengeli kazancı temsil türünden bağımsız (SSL ViT sondası ve ImageNet CNN ince ayarı, ikisinde de +2,7…+3,6). İyi kalibre sondada kısıt daha küçük τ'da bağlıyor ve kazanç seed'ler arasında çok kararlı (sd_Δ 0,45). Ama karar sınırı nadir sınıflarda birkaç görüntüyle belirlendiği için τ seçimi basamaklı. Tam-val seçimiyle OOF arasında 1,3 puanlık fark çıkabiliyor.
