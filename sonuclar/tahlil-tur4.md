# Tur-4 Tahlili (K4c) — 3.1 DermaMNIST · H10 / H11 (H12 bekliyor) · COLAB L4 · YALNIZ VALIDATION

*Deney-tahlilci, Mod 2 · 2026-10-11 · GPU kullanılmadı (CPU analiz). Test tahmini/etiketi açılmadı: H10/H11 koşumlarının `metrics.json` dosyalarında `test*` anahtarı 0 (betik denetimi). Kartlara dokunulmadı.*

| Girdi | Değer |
|---|---|
| **Ortam kuralı** | SIRALAMA "ÖN-KAYIT — COLAB TEKRAR KAPISI": kıyaslar **yalnız Colab kollarıyla**. Karar ebeveyni **H02c** = `h02_r18in_ft224_colab`, temsil ebeveyni **H01c** = `h01_dinov2b_lp_r224_colab`, bilgi **H08c** = `h08_r18in_ft224_d4_colab`, **baseline_c** = `baseline_colab`. Yerel kollar hiçbir hesaba girmedi. |
| Koşumlar (rc=0, aşama 3, seed 0–4; kök `<DRIVE_KOK>\3-1\runs`) | **H10** `20261011-160211/161411/162611/163811/165010_h10_dinov2b_k4_ft224_s0…s4` · **H11** `20261011-160117/161247/162419/163552/164724_h11_dinov2s_ft224_s0…s4` · H02c `20261011-130855…133837` · H08c `134604…141608` · H01c `125349…130608` · baseline_c `142338…151359`. `config.ebeveyn` = `h02_r18in_ft224_colab:s<seed>` (5/5, H10 ve H11). |
| Betikler (tekrar üretilebilir) | `scripts/tahlil_tur4.py` (karar istatistikleri, ikinciller, H04/R-T katmanları, bütçe) · `scripts/tahlil_tur4_ezber.py` (S/N tabaka-standart, doz-yanıt, sınıf × S/N) · `scripts/tahlil_tur4_cval.py` (DermaMNIST-C val alt kümesi). Komut: `PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I scripts/<betik> --kok "<DRIVE_KOK>/3-1"` |
| Çıktılar | `sonuclar/tahlil_tur4_olcum.json`, `tahlil_tur4_bootstrap.csv`, `tahlil_tur4_ikincil.csv`, `tahlil_tur4_ezber.json`, `tahlil_tur4_cval.json` |
| Notasyon | "puan" = yüzde puanı. ort ± std = 5 seed. GA = eşleştirilmiş görüntü bootstrap'ı %95 (2.000, RNG 0; tur3_olcum ile aynı W; ACC GA'ları `eslestirilmis_bootstrap.py` ile 14/14 aynı). p_t = kesin tek yönlü seed-eşleştirilmiş t (scipy). S = lezyonu train'de olan val görüntüsü (348), N = olmayan (655). |

---

## 0. Bütünlük denetimleri (S grubu)

| Denetim | H10 | H11 |
|---|---|---|
| `tahmin_val.npy` argmax ACC = `metrics.val_acc` | fark 0 (5/5) | fark 0 (5/5) |
| `gecmis.json` en iyi val ACC = metrics | 5/5 | 5/5 |
| Etiket hizası (npz val) | aynı | aynı |
| Seed bağlılığı (çift başına maks \|Δp\|, en küçük) | 0,846 | 0,975 |
| `param_egitilen` / `param_sayisi` | 28.364.551 / 85.730.311 (5/5) | 21.631.495 (5/5) |
| `onbellek_tutarlilik_maks_fark` | 2,43·10⁻³ (≤ 5·10⁻³) | — |
| `n_train` | 7.007 | 7.007 |
| a3 beklenti kuralları (elle; **a3 kapı raporu `.md` Drive'da yazılmamış**) | val_acc 0,8873–0,9053 ∈ [0,80; 0,93]; AUC 0,9799–0,9833 ∈ [0,93; 0,995]; 46,7 s/ep ∈ [13; 260] → GEÇER | val_acc 0,8654–0,8873 ∈ [0,78; 0,93]; AUC 0,9728–0,9813; 44,5 s/ep ∈ [10,8; 216] → GEÇER |
| Öldürme (e) (≥ 3 seed'de en iyi epoch = 15 ve son 3 epoch artış) | 0 seed (en iyi epoch 13/11/10/7/9) | 0 seed (12/9/12/11/8) |
| Test anahtarı | 0 | 0 |
| Katman denetimi (yöntem doğrulaması) | Aynı betikle hesaplanan Colab H01c+H04 OOF ACC **0,8158** = yerel H06 0,8158; H02c/H08c R-T OOF ECE 0,032/0,023 ↔ yerel 0,031/0,021 | |

---

## 1. Ölçüm okuması (val, n = 5, H02c'ye karşı seed eşleşmeli)

### 1a. Karar istatistikleri

| Kol | val ACC (ort ± std) | ΔACC vs H02c [GA] | p_t (B>A) | sd_Δ seed | Δ dengeli [GA] | Δ dengeli 5 sınıf (df/vasc hariç) [GA] | Karar değişen görüntü/seed |
|---|---|---|---|---|---|---|---|
| H02c (ebeveyn) | 0,8656 ± 0,0019 | — | — | — | (0,7100) | (0,7549) | — |
| **H10** DINOv2-B son-4-blok | **0,8947 ± 0,0066** | **+2,91 [+1,34; +4,61]** | **3,4·10⁻⁴** | 0,68 | +8,77 [+4,05; +14,12] | +4,90 [+0,89; +9,27] | — |
| **H11** DINOv2-S tam | 0,8800 ± 0,0086 | **+1,44 [+0,16; +2,81]** | **0,018** | 1,02 | +4,76 [+1,05; +8,69] | +2,62 [−0,42; +5,92] | — |
| *(ref.)* H08c D4 | 0,8865 ± 0,0033 | +2,09 [+1,08; +3,17] | 1,2·10⁻⁴ | 0,38 | +6,63 [+3,58; +10,22] | +3,90 [+1,54; +6,30] | — |
| *(ref.)* H01c sonda | 0,8223 ± 0,0013 | −4,33 [−6,50; −2,11] | 1,00 | 0,15 | −6,05 | −7,81 | — |

**Seçimden bağımsızlık:** son-epoch kolunda H10 0,8939 / H02c 0,8608 (Δ +3,31), val-AUC seçiminde 0,8909 / 0,8596 (Δ +3,13). H11 son epoch 0,8786 (Δ +1,78). Kazanç seçim kuralına dayanmıyor.

### 1b. İkincil metrikler (karar vermez)

| | H02c | H08c | **H10** | **H11** | H01c |
|---|---|---|---|---|---|
| AUC (mel / bkl) | 0,9697 (0,934 / 0,959) | 0,9782 (0,952 / 0,972) | **0,9814 (0,953 / 0,976)** | 0,9788 (0,949 / 0,974) | 0,9563 (0,894 / 0,931) |
| molemax-dışı ACC (Z1 hedefi) | 0,7977 | 0,8281 | **0,8407** (Δ +4,30 [+1,77; +6,97]) | 0,8165 (Δ +1,87 [−0,26; +4,01]) | 0,7299 |
| mel → nv (/111) · mel duyarlılığı | 37,8 · 0,613 | 30,2 · 0,678 | **24,6 · 0,714** (Δ +10,1 [+2,7; +17,9]) | 30,8 · 0,667 | 38,6 · 0,532 |
| makro-F1 | 0,746 | 0,795 | **0,823** | 0,775 | 0,678 |
| ECE-15 / NLL | 0,076 / 0,519 | 0,065 / 0,421 | 0,060 / 0,417 | **0,080** / 0,526 | 0,031 / 0,518 |
| nv payı | 0,702 | 0,686 | 0,690 | 0,682 | 0,698 |
| Toplam hata (/1.003) | 134,8 | 113,8 | **105,6** | 120,4 | 178,2 |
| Seed argmax anlaşmazlığı | %9,2 | %7,4 | %6,4 | %9,8 | %0,9 |
| **Tabakalı S − N** (havuz, [kendi GA]) | +6,17 [−0,35; +11,96] | +5,63 [+0,09; +11,13] | **+10,09 [+4,63; +15,22]** (seed 10,09 ± 1,95) | +7,10 [+1,79; +12,05] | +4,02 [−3,28; +11,69] |
| Train kaybı ep 5 (s0…s4) | 0,012–0,018 | 0,255–0,288 | 0,040–0,057 | 0,085–0,106 | — |
| Son train kaybı | ≈ 0 | — | 0,0011 | 0,0005 | — |
| **2. satır: H04 katmanı OOF dengeli / ACC** (τ\* tam-val) | 0,7377 / 0,8544 (0,7) | 0,8047 / 0,8786 (0,7) | **0,8140 / 0,8855** (0,7) | 0,7992 / 0,8710 (**1,0 — ızgara ucu**) | 0,6766 / 0,8158 (0,2) |
| **3. satır: R-T OOF ECE** (ham → OOF; T tam-val) | 0,076 → 0,032 (1,4–1,8) | 0,065 → 0,023 (1,6–1,8) | 0,060 → **0,018** (1,8–2,1) | 0,080 → 0,023 (2,2–2,5) | 0,031 → 0,020 (1,19) |
| s/epoch (L4) · eğitilen / toplam param | 14,1 · 11,18 M | 14,3 · 11,18 M | 46,7 · 28,36 / 85,73 M | 44,5 · 21,63 M | — |
| GPU (a3, 5 seed, koşum süresi) | ≈ 0,62 sa | 0,62 sa | 1,00 sa | 0,96 sa | — |

---

## 2. Kırmızı bayrak taraması

| Kart | Bayrak | Değerlendirme |
|---|---|---|
| H10 | Görülmemiş eşikleri: 5-seed val ACC > 0,93 / AUC > 0,99 / Δ vs H01c > +10,8 | **Aşılmadı** (0,8947 / 0,9814 / +7,24). Literatür tavanı (Wu & Papież 2025, arXiv:2501.14685: DINOv2-B **uçtan uca** 224 test 92,02) kısmi ince ayarla aşılmıyor; tutarlı. |
| H10 | Beklenti aralığının dışı mı? | Mutlak 0,8947 ∈ [0,845; 0,895] (üst kenar). Δ vs H01c +7,24 ∈ [+2,3; +7,3] (üst kenar). Δ vs H02c +2,91, kart aralığı (−2,2…+2,8; yerel H02'ye göre yazılmıştı) +0,11 puan üstünde; yerel H02'ye göre +2,75 ile içeride → Colab ebeveyn kayması (−0,16), bayrak değil. AUC, mel AUC, ECE, seed std aralıkta. **Manipülasyon denetimi tuttu:** ep 5 kaybı H02c'den yüksek (0,042 > 0,012; ön-kayıtlı yön). |
| H10 | Ebeveynle bit bit aynı? | Hayır (farklı omurga; karar değişen görüntü çok). |
| **H10** | **İkincil bedel (M7 ödünleşim)** | Epoch süresi **×3,3** (46,7 / 14,1 s), eğitilen param ×2,5, toplam param ×7,7. Kalibrasyon ve alt gruplar bozulmuyor (ECE ↓, tüm sınıf duyarlılıkları ≥ H02c'den, bcc −2,3 [−8,9; +4,2] hariç). → **TUT + ödünleşim notu.** |
| **H10** | **Ezber/lezyon tekrarı (keşifsel ön-kayıt; §4)** | Tabakalı S − N +10,09, ön-kayıtlı keşifsel yön "H10 ≤ H02" **nokta tahminde tutmadı** (Δ +3,92 [−2,25; +10,16] → kural gereği "ayırt edilemedi"). Kazancın %75'i S görüntülerinden. **SARI: TUT'u engellemez (ön-kayıt keşifsel), K5'in birincil sorusudur.** Arıza değil: S görüntüleri resmî bölmenin meşru parçası (görüntü düzeyi bölme; Abhishek vd. 2025). |
| H11 | Görülmemiş eşikleri | val ACC 0,8800 / AUC 0,9788 aşılmadı. **3. madde (Δ vs H12 > +12) sınanamadı: H12 koşulmadı.** Eşdeğeri: H12 val ACC **< 0,7600** çıkarsa İNCELE (H12 kart aralığı [0,787; 0,827]; a3 kapı alt sınırı 0,74). **SARI — açık denetim.** |
| H11 | Beklenti | Mutlak 0,8800 ∈ [0,835; 0,890]; Δ +1,44 ∈ [−3,2; +2,3]; AUC, mel AUC aralıkta; **ECE 0,080 = aralığın üst ucu** [0,025; 0,080]; seed std 0,86 (öngörü 0,2–0,8; s0 0,8654 düşük uç). |
| H11 | Manipülasyon denetimi (keşifsel) **tersi** | Ön-kayıt: ep 5 kaybı H10'dan **düşük**; ölçülen 0,085–0,106 > H10 0,040–0,057. Açıklama: S omurganın (384-b) kapasitesi düşük ve tüm katmanlar 1e-5 ile yavaş uyarlanıyor; son kayıp H10'dan düşük (0,0005 < 0,0011) → ezber gecikiyor ama sonunda daha derin. Birincil metrikte tersi yok; İNCELE gerekmez. |
| **H11** | **İkincil bedel (M7)** | Epoch ×3,2, ECE 0,080 (H02c 0,076'dan kötü, R-T ile 0,023'e iniyor), H04 katmanı τ\* = 1,0 ızgara ucunda (aşırı güven + nv önseli; τ ızgarası H04 ön-kaydı, genişletilmez). → **TUT + ödünleşim notu.** |
| H10/H11 | (b') t ↔ bootstrap çelişkisi | Yok (ikisinde de t anlamlı ve GA alt > 0). |

**Kırmızı bayrak sayısı: 2** (ikisi de ödünleşim türü — süre ×3,3 / ×3,2 ve H11 ECE; TUT'u engellemez) **+ 2 SARI** (H10 lezyon tekrarı → K5; H11 görülmemiş 3. madde → H12). TUT'u engelleyen bayrak: **0**.

---

## 3. Kart kararları

### 3.1 H10 — DINOv2-B son 4 blok kısmi ince ayar → **TUT (ödünleşim + ezber notu; kazanç K5'te lezyon-ayrık doğrulamaya bağlanmalı)**

- Kural girdileri (mekanik): (a) n = 5. (b) **sağlanmıyor**: ΔACC +2,91 ≥ +0,5; mutlak 0,8947 ≥ eşik (Colab 0,8706 / yerel 0,8722, ikisi de); p_t 3,4·10⁻⁴ < 0,05. (b') çelişki yok (GA alt +1,34 > 0). (c) canlı kural tetiklenmedi. (d) 46,7 s/ep. (e) 0 seed. (f) özdeşlik a1'de geçti (1,7128, tolerans 1e-3), önbellek 2,43·10⁻³, param doğru. (g) seed'ler farklı. Görülmemiş eşiği yok.
- (h) notları: Tur-1 nv kuralı uygulanmaz (Δdengeli > 0, nv payı −1,24). Z1 hedefi **tuttu** (molemax-dışı +4,30 [+1,77; +6,97]; mel → nv 37,8 → 24,6; mel duyarlılığı +10,1). Δ vs H01c > 0 → "kısmi ince ayar donuk sondayı geçti".
- **Ödünleşim:** maliyet ×3,3 epoch / ×7,7 toplam param. **Ezber notu (§4):** kazancın ¾'ü S'den; lezyon-ayrık görüntülerde H02c'ye karşı ≈ +1 puan ve ayırt edilemiyor; H08c lezyon-ayrık alt kümede H10'u geçiyor.
- **Öğrenilen:** DINOv2-B'nin son 4 bloğunu düşük gövde lr'siyle (1e-5) uyarlamak, donuk sondaya göre +7,2 [+5,5; +9,1], R18 tam ince ayarına göre +2,9 [+1,3; +4,6] puan kazandırıyor; mel ayrımı (AUC 0,953, mel → nv −35 %) ve kalibrasyon (ECE 0,060; R-T ile 0,018) iyileşiyor. Ancak kazanç lezyonu train'de bulunan görüntülerde yoğun (S +6,3 [+3,2; +9,6], N +1,1 [−0,7; +3,2]): DINOv2 ince ayarının avantajının bir kısmı **aynı lezyonun başka çekimini tanımaktan** gelebilir. Kesin yanıt K5 DermaMNIST-C.

### 3.2 H11 — DINOv2-S tam ince ayar → **TUT (ödünleşim notu; görülmemiş 3. maddesi H12'ye bağlı)**

- Kural girdileri: (a) n = 5. (b) sağlanmıyor: ΔACC +1,44 ≥ +0,5; mutlak 0,8800 ≥ 0,8722; p_t 0,018. (b') çelişki yok (GA alt +0,16 > 0). (c)–(g) temiz. Görülmemiş: 1–2. madde temiz, **3. madde H12 bekliyor** (H12 < 0,7600 → TUT askıya alınır, İNCELE).
- (h) notları: nv kuralı uygulanmaz. Z1 hedefi yönde ama ayırt edilemiyor (molemax-dışı +1,87 [−0,26; +4,01]). Δ vs H12: bekliyor.
- **Ödünleşim:** ECE 0,080 (ham), τ\* ızgara ucunda (1,0), epoch ×3,2; seed std 0,86.
- **Pareto (ön-kayıtlı ikincil, yön yok):** H11 − H10 = **−1,48 [−2,91; −0,06]**, p_t(B<A) 0,013 → "büyük omurganın üstünü uyarlamak" (H10) aynı L4 süresinde (46,7 ↔ 44,5 s/ep) "küçük omurgayı tam uyarlamaktan" (H11) daha iyi. Fark da S'de (S −3,97, N −0,15); lezyon-ayrık C-val'de −0,10 [−1,57; +1,40].
- **Öğrenilen:** DINOv2-S'nin tamamını uyarlamak R18 ince ayarını +1,4 puan geçiyor ama H10'un gerisinde; kazancı S/N arasında dengeli (tabaka-std ΔS +2,46, ΔN +1,53, ikisi de ayırt edilemiyor). Tam ince ayar düşük düzey öznitelikleri uyarlıyor ama küçük omurganın temsil açığını kapatmıyor.

### 3.3 H12 (analiz kartı; sekme C) → **BEKLİYOR (koşulmadı)**

Sayaca girmez. Rolü: (1) H11'in temsil ebeveyni (H11 − H12 tam ince ayar kazancı), (2) **H11'in görülmemiş 3. maddesini kapatır**, (3) Pareto satırı. Maliyet ≈ 0,1–0,2 L4-sa. **K5'ten önce koşulması önerilir** (§8).

---

## 4. Ezber / sızıntı — K5'in birincil sorusu (KIRMIZI değil; açıkça)

**Ön-kayıt (H10 kartı ikincil 8; SIRALAMA Tur-4 ortak):** tabakalı S − N keşifsel; yönler **H10 > H01 (+4,0)** ve **H10 ≤ H02 (+6,65)**; GA dışlamazsa "ayırt edilemedi"; kesin yanıt K5 DermaMNIST-C.

| Ölçü | H10 − H02c | H11 − H02c | H08c − H02c | H10 − H08c | H10 − H01c |
|---|---|---|---|---|---|
| Δ(tabakalı S − N) [GA] | +3,92 [−2,25; +10,16] → **ayırt edilemedi; nokta ön-kayıtlı yönün (≤) tersi** | +0,93 [−4,00; +6,20] | −0,54 [−4,07; +3,19] | +4,46 [−1,40; +10,44] | +6,07 [−0,28; +12,29] → **yön tuttu, ayırt edilemedi** |
| Ham Δ_S (348) [GA] | **+6,26 [+3,23; +9,61]** | +2,30 [−0,46; +5,20] | +2,87 [+0,97; +4,70] | **+3,39 [+0,23; +6,51]** | +14,25 |
| Ham Δ_N (655) [GA] | **+1,13 [−0,71; +3,16]** | +0,98 [−0,45; +2,39] | **+1,68 [+0,44; +2,92]** | −0,55 [−2,43; +1,30] | +3,51 [+1,69; +5,38] |
| Tabaka-std Δ_S / Δ_N (sınıf × dx_type × kaynak; N ağırlıklı) | **+5,38 [+1,25; +10,03] / +1,46 [−2,93; +5,90]** | +2,46 / +1,53 (ikisi de 0'ı kapsar) | +2,74 [+0,49; +5,36] / **+3,28 [+0,48; +5,97]** | +2,64 / −1,82 | +13,00 / +6,93 |
| molemax-dışı Δ_S / Δ_N | +6,26 [+3,19; +9,62] / +1,82 [−2,52; +6,20] | +2,09 / +1,61 | +2,84 / **+3,28 [+0,53; +5,99]** | +3,42 [+0,26; +6,57] / −1,46 | +14,32 / +7,01 |
| **Kazancın S payı** (S görüntü payı %34,7) | **%75** (21,8 net görüntü S, 7,4 N) | %56 | %48 | **%144** (S +11,8, N −3,6) | %68 |
| **C-val alt kümesi** (resmî val ∩ C-val, 573; lezyonu resmî train'de **yok**; yeniden eğitimsiz) | **+1,05 [−0,66; +2,79]** (p_t 0,010) | +0,94 [−0,45; +2,34] | **+1,75 [+0,59; +3,04]** | **−0,70 [−2,41; +0,87]** | +3,18 [+1,47; +5,03] |

**Doz-yanıt (S görüntüsünün lezyonunun train'deki görüntü sayısı):** H10 − H02c: ntr = 1 (260) +6,2 [+2,4; +10,2], ntr = 2 (84) +6,9 [+1,4; +13,5] — belirgin doz eğimi yok (H08c'de 2,3 → 4,8). Ezber "kaç kopya gördüm" ile ölçeklenmiyor; "en az bir kez gördüm" ile çalışıyor.

**Sınıf × S/N (H10 − H02c, net görüntü S / N):** mel **+10,4 / +0,8** (mel'in 76/111'i S), bkl +4,8 / 0,0, akiec +2,8 / +1,0, nv +1,8 / +3,8, vasc +1,6 / +2,4. **H10'un Z1 kazancı (mel → nv) neredeyse tamamen S'deki mel görüntülerinde.** H08c'de mel +5,2 / +2,0 ve bkl +2,2 / +3,4 — dengeli.

**Okuma:**
1. **Ön-kayıtlı kural gereği:** Δ(S − N) GA 0'ı kapsıyor → "ayırt edilemedi". TUT kararı değişmez; kural gevşetilmez veya sıkılaştırılmaz.
2. **Ama tahlilci açıkça yazar:** H10'un H02c'ye kazancının ≈ ¾'ü lezyonu train'de görülmüş val görüntülerinde; lezyon-ayrık görüntülerde (N, C-val) H10 − H02c ≈ +1 puan ve GA 0'ı kapsıyor. H08c (D4) ise kazancını S/N arasında dengeli dağıtıyor ve lezyon-ayrık görüntülerde H10'dan iyi (C-val −0,70; N −0,55). **Val'deki sıralama H10 > H08c, lezyon tekrarı çıkarılınca H08c ≥ H10'a dönüyor** (GA'lar 0'ı kapsıyor).
3. **Mekanizma hipotezi (düşük güven, sınanmadı):** DINOv2 öz-denetimli temsilleri örnek düzeyi tanımada güçlü (Oquab vd. 2024, TMLR, örnek tanıma değerlendirmesi: Oxford/Paris/Met/AmsterTime). Son blokların uyarlanması "bu lezyonu gördüm" bilgisini sınıf kararına bağlayabilir. Donuk sonda (H01c) bunu daha az yapıyor (S − N +4,0).
4. **Resmî test bu soruyu çözemez:** resmî test'in %34,6'sı (693/2.005) S — val ile aynı oran. Resmî test'te H10 avantajının val'deki gibi çıkması beklenir. **C-test (1.227) resmî test'in alt kümesi ve lezyonlarının hiçbiri resmî train'de yok** (denetlendi: 0/1.227) → mevcut modellerle **yeniden eğitimsiz** lezyon-ayrık değerlendirme mümkün (§8).
5. **Literatür etkisi (hipotez):** Wu & Papież 2025'in 92,02'si de resmî (görüntü düzeyi) bölmede; aynı S kazancını içeriyor olabilir. Makale bunu tartışmalı; kaynak cümlesi K5 sonucuna bağlanır.

**M1 beyanı:** C-val alt kümesi ve S/N ayrışımı bu tahlilde **ilk kez** görüldü (ön-kayıtlı değil, keşifsel). K5 kuralları (§8.4) bu sayılara göre ayarlanmadı: ölçüt H08 için 2026-10-11 09:42'de yazılmış olanla aynıdır (GA alt > 0).

---

## 5. Diğer ön-beyanlı ikincil karşılaştırmalar (karar vermez)

| Kıyas | ΔACC [GA] | p_t | Etiket (kart kuralı) | Not |
|---|---|---|---|---|
| **H10 − H01c** (temsil: sonda → son-4-blok) | **+7,24 [+5,48; +9,11]** | 1,0·10⁻⁵ | **"ince ayar kazancı saptandı"** (p_t < 0,05 ve GA alt > 0) | Tüm sınıflarda ↑ (mel +18,2, bkl +18,0, bcc +12,7); N'de de +3,51 [+1,69; +5,38] → sonda → ince ayar kazancı lezyon tekrarından bağımsız olarak var. Makale cümlesi: "İnce ayar DINOv2'yi sondadan kurtarır" — destekleniyor. |
| **H10 − H08c** (bilgi; farklı aileler) | +0,82 [−0,76; +2,43] | 0,0085 | t anlamlı, bootstrap 0'ı kapsıyor → **ayırt edilemedi** | Tamamen S'den (S +3,39 [+0,23; +6,51], N −0,55). Dengeli +2,14 [−2,10; +6,64]. bcc −5,8 [−11,0; −0,4]. H04 katmanında +0,70 [−0,96; +2,49]. Lezyon-ayrık C-val'de −0,70. |
| **H11 − H10** (Pareto) | −1,48 [−2,91; −0,06] | p(B<A) 0,013 | H10 > H11 saptandı (val) | Fark S'de; C-val'de −0,10 (ayırt edilemedi). |
| H11 − H01c (çapraz) | +5,76 [+3,77; +7,86] | 9,7·10⁻⁵ | — | Boyut + ince ayar paketi; ayrıştırmak için H12 gerekli. |
| H11 − H08c | −0,66 [−1,91; +0,62] | 0,94 | ayırt edilemedi | — |
| H10 − baseline_c | +14,90 [+12,54; +17,21] | 1,2·10⁻⁶ | — | Zincir bilgisi; tabakalı S − N Δ +9,18 [+1,09; +16,95]. |

---

## 6. Değerlendirme katmanları (H04 OOF, R-T) — Colab tahminlerinden CPU'da hesaplandı

Koşum üretilmedi (analiz). Yöntem `logit_ayari.oof_tau` / `sicaklik.T_sec` ile **birebir** (aynı `katlar.json`, τ 0:1:0,1, kısıt = kolun kendi 5-seed ham ACC'si − 0,010; T 0,5:3,0:0,01; OOF ECE-15). Yöntem doğrulaması: H01c+H04 OOF ACC 0,8158 = yerel H06 0,8158.

| Kol | τ_k (5 kat) / τ\* | OOF dengeli (± std) | OOF dengeli 5 sınıf | OOF ACC (kısıt) | R-T OOF ECE |
|---|---|---|---|---|---|
| H02c | 0,4/0,7/0,8/0,6/0,7 · 0,7 | 0,7377 ± 0,0279 | 0,7571 | 0,8544 (≥ 0,8556? **0,12 puan altında** — kısıt seçim katlarında, OOF'ta garanti değil; yerel H04'te de aynı yapı) | 0,032 |
| H08c | 0,7/0,7/0,7/0,6/0,7 · 0,7 | 0,8047 ± 0,0145 | 0,8084 | 0,8786 (≥ 0,8765 ✓) | 0,023 |
| **H10** | 0,7 ×5 · 0,7 | **0,8140 ± 0,0103** | 0,8111 | 0,8855 (≥ 0,8847 ✓) | **0,018** |
| H11 | 1,0/1,0/1,0/0,9/0,9 · **1,0 (ızgara ucu)** | 0,7992 ± 0,0152 | 0,7994 | 0,8710 (≥ 0,8700 ✓) | 0,023 |
| H01c (= H06) | 0,1/0,1/0,1/0,3/0,3 · 0,2 | 0,6766 | 0,6925 | 0,8158 | 0,020 |

- H10+H04 − H02c+H04: ACC +3,11 [+1,50; +4,91], dengeli **+7,63 [+3,66; +12,02]**, 5 sınıf +5,40 [+1,55; +9,71].
- H10+H04 − H10: dengeli +1,63 [+0,37; +3,18], ACC −0,92 [−1,52; −0,34] (katmanın beklenen ödünleşimi).
- **H11 τ\* = 1,0 ızgara ucunda:** H11'in aşırı güveni (T ≈ 2,3) ve nv önseli logit ayarını tavana itiyor. Izgara ön-kayıtlı; genişletme önerilmez. K5'te H11 için τ\* = 1,0 "ızgara ucu" notuyla uygulanır.
- **H02c H04 notu:** OOF ACC 0,8544, kendi kısıtının (0,8656 − 0,010 = 0,8556) 0,12 puan altında. Bayrak değil (kısıt seçim katlarında uygulanıyor; H07'de de aynı yapı görüldü), raporda açıkça yazılır.

---

## 7. Durma sayacı ve bütçe

### 7a. Durma sayacı (konu.md 2026-10-10 20:15; SIRALAMA Tur-4 okuma kuralı)

Eşikler yerelde H02 zincirinden türetilmişti (0,8722 = 0,8672 + 0,5; 0,7475 = H04 OOF 0,7425 + 0,5). **Colab'da aynı mantıkla H02c zincirinden yeniden hesaplandı ve karar için bu kullanıldı:** ACC ≥ **0,8706** (0,8656 + 0,005), dengeli (H04 katmanı OOF) ≥ **0,7427** (0,7377 + 0,005). Yerel eşiklerle de sonuç aynı.

| | Tur-4 öncesi en iyi (Colab) | Tur-4 en iyi (TUT; Tur-5 kuralı: en yüksek ham val ACC) | Colab eşiği | Yerel eşik | ≥ 0,5 puan? |
|---|---|---|---|---|---|
| val ACC (ham) | 0,8656 (H02c) | **0,8947 (H10)** | 0,8706 | 0,8722 | **evet** |
| val dengeli (H04 katmanı OOF) | 0,7377 (H02c+H04) | **0,8140 (H10+H04)** | 0,7427 | 0,7475 | **evet** |

**Sayaç: 1/2 → 0/2.** Ön-kayıt (SIRALAMA "Tur-5 ebeveyn kuralı"): **H10 = "DINOv2-FT bileşeni", Tur-5 (Z3 topluluk, yalnız CPU) açılır.** Tur 4/8.

### 7b. Bütçe

| Kalem | Koşum süresi toplamı | Oturum (duvar) | Not |
|---|---|---|---|
| Yerel GTX 1650 (K3 … Tur-3 + Tur-4 ön-ölçüm) | — | **15,3 GPU-sa** | günlükteki 14,7 + 0,62; **günlük sayacı 14,7'de kalmış → güncellenmeli** |
| Colab L4 kc1-tekrar (32 koşum) | 2,86 sa | ≈ 3,15 sa (12:18–15:27) | tekrar kapısı |
| Colab L4 kc2-tur4-h10 (A, 8 koşum) | 1,17 sa | ≈ 1,55 sa (15:32–17:05) | önbellek yazımı dahil |
| Colab L4 kc2-tur4-h11 (B, 8 koşum) | 1,11 sa | ≈ 1,55 sa | — |
| **Colab toplamı** | **5,14 L4-sa** | **≈ 6,2 L4-sa** | `durum/erteleme.json` **yok**; değerler `.done` ve `canli_*.json`'dan |
| **Birleşik defter** (SIRALAMA: Colab ayrı birim; H11 erteleme sayacı birleşik sayar) | | **≈ 21,5 / 32** | H11 a3 başında (16:01) ≈ 19,4 ≤ 23,0 → erteleme kuralı tetiklenmedi ✓ |

Kalan ≈ 10,5. K5 + öneriler (§8) ≈ 0,3 (zorunlu) … 9,5 (tüm seçenekler) L4-sa.

---

## 8. K5 ÖNERİSİ (taslak; karar şef/kullanıcıda)

### 8.1 Sıradaki adım: Tur-5 mü, K5 mi?

Ön-kayıtlı durma kuralına göre sayaç 0/2 → **Tur-5 (topluluk) açılır.** Tur-5 yalnız CPU'dur (≈ 0 GPU; `tahmin_val.npy`'ler hazır). Önerilen sıra:
1. **H12 (sekme C, ≈ 0,1–0,2 L4-sa)** — H11'in görülmemiş 3. maddesini kapatır, Pareto satırı.
2. **Tur-5 T1 kartı** (H02c_s ⊕ H10_s, n = 5) ve T2 (+ H01c). **Uyarı:** topluluk H10'un S-ağırlıklı kazancını devralır; Tur-5 kartı S/N ayrışımını ve C-val satırını ikincil olarak ön-kaydetmeli. T4 (H08c ⊕ H10) yalnız koşullu satır (H08 doğrulanırsa).
3. Ardından K5. Kullanıcı Tur-5'i atlayıp doğrudan K5'e geçmek isterse bu, durma kuralından sapmadır → günlükte "kullanıcı kararı" olarak yazılır.

### 8.2 Kazanan zincir (val, Colab)

| Basamak | Varyant | val ACC | val dengeli | Not |
|---|---|---|---|---|
| 0 | baseline_c (R18, 28 px, sıfırdan) | 0,7458 | 0,4333 | replikasyon YEŞİL (test 0,7395) |
| 1a | + ImageNet + 224 + ince ayar (H02c) | 0,8656 | 0,7100 | TUT (Tur-1) |
| 1b | DINOv2-B sonda (H01c) | 0,8223 | 0,6494 | TUT (Tur-1) |
| 2 | **DINOv2-B son-4-blok ince ayar (H10)** | **0,8947** | **0,7977** | **TUT (Tur-4)**; ezber notu |
| 2 + H04 | H10 + logit ayarı (OOF) | 0,8855 | **0,8140** | katman |
| 2 + R-T | H10 + sıcaklık | = | = | ECE 0,060 → 0,018 |
| (aday) | H02c + D4 (H08c) | 0,8865 | 0,7763 | K5 doğrulama adayı (yol A); lezyon-ayrık C-val'de en iyi |
| (Pareto) | DINOv2-S tam (H11) | 0,8800 | 0,7576 | TUT; H10'un gerisinde |

**Kazanan zincir (val):** baseline → H02 (önağırlık/224/FT paketi) → **H10 (omurga A4)** → H04 katmanı → R-T. H08 ayrı R18 dalında aday.

### 8.3 Test'e (bir kez) gidecek kollar — ön-kayıt önerisi (test açılmadan SIRALAMA'ya yazılmalı)

| Kol | Rol | Katmanlar (tam-val τ\*/T) |
|---|---|---|
| baseline_c | rakip (test sayıları kc1'de dondu) | — |
| H01c (+ H06 τ\* 0,2) | sonda satırı | R-T T 1,19 |
| H02c ×5 | R18 ince ayar; adil kıyas satırı | H04 τ\* 0,7; R-T seed başına T |
| **H08c ×5** | K5 doğrulama adayı (SIRALAMA K5 kuralı aynen) | H04 τ\* 0,7; R-T |
| **H10 ×5** | **Tur-4 kazananı (val'de seçildi; test sonucu kazananı DEĞİŞTİRMEZ)** | H04 τ\* 0,7; R-T T 1,79–2,05 |
| H11 ×5 | Pareto satırı (TUT; kazanan değil) | H04 τ\* 1,0 (ızgara ucu notu); R-T |
| H12 | Pareto satırı (koşulursa) | H06 protokolü |
| Tur-5 topluluğu | yalnız Tur-5'te TUT alırsa | — |

Maliyet: `best_model.pth` çıkarımları + H01/H12 test öznitelikleri ≈ **0,2 L4-sa**. H05, H07, H09 test'e gitmez.

### 8.4 K5 kural önerileri (test açılmadan yazılmalı)

1. **H10 doğrulama (resmî test, n = 2.005):** H10 − H02c eşleştirilmiş; "doğrulandı ⇔ ΔACC ≥ +0,5 **ve** görüntü bootstrap GA alt > 0" (H08 kuralının aynısı). Görülmemiş: test ACC > 0,93 / AUC > 0,99 → rapor bayrağı.
2. **Lezyon-ayrık ezber testi (birincil soru; yeniden eğitimsiz, 0 ek GPU):** resmî-train modelleri, **C-test alt kümesinde** (1.227 görüntü, resmî 224 piksel; lezyonları resmî train'de yok — denetlendi). H10 − H02c **ve** H08c − H02c: GA alt > 0 → "kazanç lezyon-ayrık görüntülere genelleşir"; GA 0'ı kapsarsa → "resmî-bölme kazancı lezyon tekrarına duyarlı; lezyon-ayrık ayrım yok". H10 − H08c bilgi satırı. Ek: resmî test'te aynı tabakalı S − N ve Δ_S/Δ_N ayrışımı (bu tahlilin betikleri test için uyarlanır).
3. **Makale cümlesi koşulu:** "DINOv2 ince ayarı R18'i geçer" iddiası yalnız (1) **ve** (2) birlikte geçerse genel ifade olarak yazılır; yalnız (1) geçerse "resmî (görüntü düzeyi) bölmede" kaydıyla ve S/N ayrışımıyla.

### 8.5 DermaMNIST-C (KULLANICI KARARI 3: birincil = resmî piksel + C bölmesi; ikincil = C npz)

| Seçenek | Ne | Kollar | Maliyet (L4) | Not |
|---|---|---|---|---|
| **C0 (öneri, zorunlu)** | Yeniden eğitimsiz: resmî-train modelleri C-test alt kümesinde | tüm test kolları | **0** (test çıkarımının alt kümesi) | Ezber sorusuna en doğrudan yanıt; KARAR 3'ün "yalnız bölme değişir" ruhuna uygun bir **ek** satır |
| **C1 (KARAR 3 birincil; tam protokol)** | Resmî 224 piksel, C-train (8.215) ile yeniden eğitim, C-val (573) seçim, C-test | baseline_c, H02c, H08c, **H10** (+ H01c sonda) | baseline ≈ 1,2 · H02 ≈ 0,75 · H08 ≈ 0,75 · H10 ≈ 1,25 (C-train blok-8 önbelleği dahil) · H01 ≈ 0,15 → **≈ 4,1 L4-sa** (3 paralel oturumda ≈ 1,5 sa duvar); H11 eklenirse +1,1 | C-train 778 resmî-test görüntüsü içerir → C1 koşumları resmî test sonuçlarından **sonra** ve ayrı kuyrukta; C-test yalnız bir kez |
| **C2 (ikincil)** | C npz (kendi küçültmesi) 224, aynı protokol | baseline + H10 (+ H02) | ≈ 2,4–3,2 L4-sa | Literatürle (Abhishek vd. 2025) yayımlanmış hâliyle kıyas |

Tahmin tabanı: kc1/kc2 koşum süreleri × (8.215 / 7.007 = 1,17) + oturum payı. Bütçeye sığar (≈ 21,5 + 0,2 + 4,1 + 2,4 ≈ 28,2 / 32).

### 8.6 Eksik ablasyonlar ve açık kararlar

| # | Madde | Gerekli mi? | Maliyet | Kim |
|---|---|---|---|---|
| 1 | **H12** | Evet (H11 görülmemiş 3. madde + Pareto) | 0,1–0,2 L4-sa | şef (sekme C) |
| 2 | H10 K ablasyonu (K = 2; "kaç blok yeter") | Kart planında "kazanırsa (i)"; makale için faydalı, zorunlu değil | ≈ 0,6 L4-sa | şef/kullanıcı |
| 3 | B tam ince ayar (K = 12) | **KULLANICI ONAYI GEREKİR** (SIRALAMA). Tetikleyici "H10 TUT + (e) sinyali" **oluşmadı** (en iyi epoch ≤ 13) → önerilmez | ≈ 4 L4-sa | kullanıcı |
| 4 | Gövde lr duyarlılığı {1e-4, 1e-6} | M3 asimetrisi; yalnız rakibe aynı arama hakkıyla → önerilmez | — | — |
| 5 | H10 + D4 | H08 doğrulanmadıkça açılmaz (M2); 8 görünüm önbelleği ≈ 25 GB | — | — |
| 6 | H04/H06 yalnız-nv ablasyonu (ii) | Tur-2'den beri eksik | CPU | deney-yürütücü, K5 öncesi |
| 7 | **M3 — rakibe D4 hakkı** | **Yalnız H08 makalede R18 kazanan satırı olarak raporlanırsa.** H10 kazanan olduğundan ana iddia için gerekmez; H08 K5'te doğrulanır ve "R18 + D4" satırı iddia taşırsa gerekir → **K5 öncesi kullanıcı kararı** (öneri: H08 doğrulama sonucuna koşullu ön-kayıt) | ≈ 1,1 L4-sa (baseline 28 px + D4 × 5) | kullanıcı |
| 8 | H10 için M3 | Gerekmez: tek tarif, arama yok (H02 ile simetrik); "rakibe önağırlık hakkı" satırı H02 | — | — |
| 9 | Colab `runs/` yerel arşive kopya; günlük bütçe sayacı (14,7 → ≈ 21,5); h10/h11 a3 kapı raporu `.md` (yok; §0'daki elle denetim kaydı) | D5 | — | şef |

---

## 9. Günlük için satır önerileri (`gunluk/deney-gunlugu.md`)

| Tur | Tarih | Hipotez | Ebeveyn → Varyant | Koşum id'leri | Ana metrik (ort±std, n) | Δ vs ebeveyn | Karar | Öğrenilen |
|---|---|---|---|---|---|---|---|---|
| 4 | 2026-10-11 | H10 | h02_colab → h10_dinov2b_k4_ft224 (Colab L4) | 20261011-160211…165010 (5) | val_acc 0,8947±0,0066; dengeli 0,7977; H04 OOF 0,8140 | +2,91 [+1,34; +4,61], p_t 3,4e-4 | **TUT** (ödünleşim: süre ×3,3; ezber notu) | Kazancın %75'i S'den; N +1,13 [−0,71; +3,16]; C-val +1,05 [−0,66; +2,79]; H10 − H01c +7,24 |
| 4 | 2026-10-11 | H11 | h02_colab → h11_dinov2s_ft224 (Colab L4) | 20261011-160117…164724 (5) | val_acc 0,8800±0,0086; dengeli 0,7576 | +1,44 [+0,16; +2,81], p_t 0,018 | **TUT** (ödünleşim; görülmemiş 3. madde H12'ye bağlı) | H11 − H10 −1,48 [−2,91; −0,06]; S/N dengeli; τ\* ızgara ucunda |
| 4 | — | H12 | h01_colab → h12_dinov2s_lp_r224 | — | — | — | BEKLİYOR | — |

**Arıza / beklenti ihlali tablosu:**

| Tarih | Belirti | Hipotez | Teşhis yöntemi | Kök neden | Düzeltme | Etkilenen |
|---|---|---|---|---|---|---|
| 2026-10-11 | H10 tabakalı S − N +10,09 (ön-kayıtlı keşifsel yön "≤ H02" nokta tahminde tersi); kazancın %75'i S | val'e sızıntı / önbellek karışması / lezyon tekrarının tanınması | npy = metrics, etiket hizası, önbellek tutarlılığı, n_train; S/N tabaka-std, doz-yanıt, sınıf × S/N, C-val alt kümesi (`tahlil_tur4_ezber.py`, `tahlil_tur4_cval.py`) | **Arıza yok.** Resmî bölmenin görüntü düzeyi olmasından gelen meşru S görüntülerinde H10 daha iyi; lezyon-ayrık alt kümede avantaj ≈ +1 ve ayırt edilemiyor | Kod düzeltmesi yok; K5'te C-test ezber kuralı (§8.4) | Yok |
| 2026-10-11 | h10/h11 a3 beklenti kapı raporu Drive'da yok | kapı aşaması "bilgi" olarak koşmadı | kurallar elle denetlendi (§0): hepsi geçer | süreç notu | şef kapıyı yerelde koşabilir | Yok |

**Bütçe satırı:** yerel 15,3 GPU-sa + Colab ≈ 6,2 L4-sa (oturum; koşum 5,14) ≈ 21,5 / 32. Tur 4/8. Durma sayacı **0/2** (H10 TUT; iki ölçüt de sağlandı).

---

## 10. Tahlil defteri (S/D/I)

| grup | bulgu | durum |
|---|---|---|
| S | npy = metrics (10/10), gecmis = metrics, etiket hizası, param sayıları, önbellek tutarlılığı, seed bağlılığı, test anahtarı 0, ebeveyn eşlemesi, ACC GA'ları referans betikle 14/14 aynı, katman yöntemi yerel H06 ile birebir | geçti |
| S | a3 kapı raporu yok | elle denetlendi, geçer |
| D | H10 S − N en yüksek; kazanç S'de | açıklandı (lezyon tekrarı; arıza değil) → K5 C-test |
| D | H10 Δ vs H02c kart aralığının 0,11 puan üstü | Colab ebeveyn kayması (yerel H02'ye göre aralıkta) |
| D | H11 ep 5 kaybı ön-kayıtlı yönün tersi | açıklandı (S kapasitesi; son kayıp daha düşük), keşifsel |
| D | H11 τ\* = 1,0 ızgara ucu | aşırı güven (T ≈ 2,3) + nv önseli; ızgara genişletilmez |
| I | "H10 R18'i +2,9 puan geçer" cümlesi tek başına yazılamaz; S/N ayrışımı ve C-val satırıyla birlikte | makale cümlesi önerisi |
| I | "H10 > H08" val'de ayırt edilemedi ve yalnız S'den; lezyon-ayrık alt kümede tersine dönüyor | K5 notu |
| I | "İnce ayar DINOv2'yi sondadan kurtarır" (H10 − H01c +7,24; N'de de +3,51) | destekleniyor |
| I | H11 görülmemiş denetimi eksik (H12) | H12 koşulmadan H11 makale satırı "koşullu" |

**Bu tahlilde yeni görülen val bilgisi (M1 beyanı):** H10/H11 tüm val sonuçları; S/N tabaka-std ayrışımları; doz-yanıt; sınıf × S/N; **C-val alt kümesi (573) sonuçları** (H10 − H02c +1,05, H08c − H02c +1,75, H10 − H08c −0,70); Colab kollarının H04/R-T katmanları; H10 son-epoch ve AUC-seçimli kolları.

**Sınırlılıklar:** 5 seed aynı 1.003 val görüntüsünü paylaşıyor. S/N ve C-val ayrışımları keşifseldir (ön-kayıtlı karar ölçüsü değil). C-val'de mel yalnız 22, df 5 (lezyon-ayrık alt küme nv ağırlıklı, %78) → C-val sıralaması nadir sınıf kazançlarını az temsil eder. Paralel Colab oturumları → süreler yalnız maliyet raporu (R6). H10'un mekanizma hipotezi (örnek tanıma) sınanmadı.

---

## 11. Kart başına hazır sonuç paragrafları (kart dosyalarına şef/yürütücü ekler)

### H10
> - Koşumlar: Colab L4 (KULLANICI KARARI 6–8), `20261011-160211/161411/162611/163811/165010_h10_dinov2b_k4_ft224_s0…s4`; ebeveyn `h02_r18in_ft224_colab` aynı seed (`20261011-130855…133837`).
> - Ölçülen: val ACC 0,8947 ± 0,0066 (H02c 0,8656); ΔACC +2,91 [+1,34; +4,61], p_t 3,4·10⁻⁴; AUC 0,9814; dengeli 0,7977 (Δ +8,77; 5 sınıf +4,90 [+0,89; +9,27]); molemax-dışı +4,30; mel → nv 24,6; ECE 0,060 (R-T OOF 0,018); H04 OOF dengeli 0,8140 / ACC 0,8855; tabakalı S − N +10,09 (Δ vs H02c +3,92 [−2,25; +10,16], ayırt edilemedi); Δ_S +6,26 / Δ_N +1,13 [−0,71; +3,16]; C-val (lezyon-ayrık, 573) +1,05 [−0,66; +2,79]; H10 − H01c +7,24 [+5,48; +9,11] ("ince ayar kazancı saptandı"); H10 − H08c +0,82 [−0,76; +2,43] (ayırt edilemedi). 46,7 s/ep (L4), 28,36 M eğitilen param.
> - Karar: **TUT** — (b) sağlanmadı, (b') çelişki yok, görülmemiş eşiği yok. Ödünleşim notu (M7): süre ×3,3, toplam param ×7,7. Ezber notu: kazancın %75'i lezyonu train'de görülmüş görüntülerden; K5'te C-test lezyon-ayrık doğrulamasına bağlanır (§8.4). Durma sayacı 0/2; Tur-5 DINOv2-FT bileşeni.
> - Öğrenilen: DINOv2-B'nin son 4 bloğunun düşük lr ile uyarlanması donuk sondayı +7,2, R18 ince ayarını +2,9 puan geçiyor; mel ayrımı ve kalibrasyon iyileşiyor. Kazanç lezyonu train'de olan görüntülerde yoğun; lezyon-ayrık görüntülerde R18'e karşı avantaj ≈ +1 puan ve val'de ayırt edilemiyor, H08 (D4) ise orada H10'dan iyi.

### H11
> - Koşumlar: Colab L4, `20261011-160117/161247/162419/163552/164724_h11_dinov2s_ft224_s0…s4`; ebeveyn `h02_r18in_ft224_colab` aynı seed.
> - Ölçülen: val ACC 0,8800 ± 0,0086; ΔACC +1,44 [+0,16; +2,81], p_t 0,018; AUC 0,9788; dengeli 0,7576 (Δ +4,76; 5 sınıf +2,62 [−0,42; +5,92]); molemax-dışı +1,87 [−0,26; +4,01]; ECE 0,080 (R-T OOF 0,023); H04 OOF dengeli 0,7992 (τ\* 1,0, ızgara ucu); tabakalı S − N +7,10; tabaka-std ΔS +2,46 / ΔN +1,53; C-val +0,94 [−0,45; +2,34]; H11 − H10 −1,48 [−2,91; −0,06]; H11 − H01c +5,76. 44,5 s/ep, 21,63 M param.
> - Karar: **TUT** — (b) sağlanmadı, (b') çelişki yok; ödünleşim notu (ECE 0,080, τ\* ızgara ucu, süre ×3,2). Görülmemiş eşiğinin 3. maddesi (Δ vs H12 > +12 ⇔ H12 < 0,7600) H12 koşulunca kapanır. Tur-5 bileşeni değil (H10 daha yüksek).
> - Öğrenilen: DINOv2-S'nin tamamını uyarlamak R18 ince ayarını +1,4 puan geçiyor, kazancı S/N arasında dengeli; ama aynı L4 süresinde DINOv2-B'nin üst 4 bloğunu uyarlamak (H10) val'de 1,5 puan daha iyi (lezyon-ayrık alt kümede fark yok). Ön-kayıtlı ep 5 kayıp yönü tutmadı (S omurga ezberi geciktiriyor).

---
*Kaynaklar (bu tahlilde kullanılan; kartlardaki künyelere ek):* Oquab, M. vd. (2024). DINOv2: Learning Robust Visual Features without Supervision. TMLR (örnek düzeyi tanıma değerlendirmesi; mekanizma hipotezi için, düşük güven). Abhishek, K. vd. (2025) DermaMNIST-C (`literatur-cipa.md` §5; görüntü düzeyi bölmede lezyon tekrarı). Wu, Y. & Papież, B. W. (2025). arXiv:2501.14685 (H10 kartındaki künye).
