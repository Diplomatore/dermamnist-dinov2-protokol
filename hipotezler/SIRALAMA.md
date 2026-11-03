# Hipotez Sıralaması — Konu 3.1 DermaMNIST

> **TUR-5 ve K5 (2026-10-11):** Tur-5 bölümü → `hipotezler/SIRALAMA-TUR5.md` (kart H13). K5 ön-kaydı (test kolları,
> H10/H08/H13 doğrulama kuralları, C0/C1/C2, karar çıkışları; KULLANICI ONAYI GEREKİR) → `hipotezler/K5-ONKAYIT.md`.
> Aşağıdaki "K5 ÖN-KAYDI — H08 TEST DOĞRULAMA KURALI" aynen geçerlidir ve K5-ONKAYIT §2.1'de değiştirilmeden alıntılanmıştır.


**Durum: TUR-4 ÖN-KAYITLI (2026-10-11).** Kartlar H10, H11, H12 (analiz). "KULLANICI ONAYI GEREKİR" işaretli
(protokol/veri/bütçe/donanım değiştiren) kart **yok**: split, veri (native 224), metrik, epoch seçimi ve donanım (yerel
GTX 1650) değişmiyor. Tur toplamı ≈ 11,8 GPU-sa (üst 13,2); Tur-4 + K5 rezervi ≤ 32 (aşağıda). Tur-4 otomatik koşar.
**DINOv2-B tam ince ayar (Colab) kart olarak YAZILMADI** — kullanıcı onayı gerektiren seçenek olarak aşağıda notlu.
Tur-3, Tur-2 ve Tur-1 bölümleri aşağıda tarihsel kayıt olarak korunuyor; K5 ön-kaydı (H08 test doğrulama kuralı)
dosyanın sonunda değişmeden duruyor.

## Kapanmış kartlar (Tur-3; `sonuclar/tahlil-tur3.md` §3)
| Kart | Karar | Bu tura etkisi |
|---|---|---|
| H07 (TTA-8, H02) | **ÖLDÜR (b)**: ΔACC −0,44 [−1,48; +0,60] | **H07 öldü çünkü** augmentasyonsuz eğitilmiş ağda dönüşmüş görünümler dağılım dışı (görünüm başına −2,6 puan; 90° görünümleri D2'den daha zararlı → anizotropik ölçekleme). TTA katmanı Tur-4'e taşınmaz; durma sayacının ACC ölçütü **ham satırdan** okunur |
| H08 (eğitimde D4, H02) | **İNCELE → arıza yok; M6 gereği TUT verilmedi. KULLANICI: yol A** — K5 test doğrulama adayı (kural dosya sonunda) | Tur-4 tarif ebeveyni **h02_r18in_ft224** (mekanik). Tur-4 augmentasyonsuz (önbellek tek görünüm; omurga değişimi için tek değişken temiz). Ders: **dengeli ACC için aile dışı çıpadan M6 eşiği yazılmaz**, df/vasc hariç 5 sınıflı satır eklenir (Tur-4 kartlarında uygulandı) |
| H09 (renk ±%20, H02) | **ÖLDÜR (b)**: +0,32 [−0,58; +1,26] (p_t 0,033; (b') çelişkisi karar etkisiz) | **H09 öldü çünkü** fotometrik oynama bu tarifte zayıf düzenleyici (kayıp ≈ 2,5×); kaynak kısayolu imzası doğru yönde ama 1.003 görüntüyle ayırt edilemiyor |

Durma sayacı: **1/2** (Tur-3'te TUT alan kart yok). **Tur-4 olası son turdur:** Tur-4'te hiçbir kart TUT alamaz ya da TUT
alanlar iki ölçütün hiçbirini geçmezse sayaç 2/2 olur → K5. Sıradaki boş kart id'si (bu turdan sonra): **H13**.

---

## TUR-4 — Z1 kalanı: DINOv2 ince ayarı (A4 omurga değişimi; tarif ebeveyni `h02_r18in_ft224`)

### Hata kütlesi (H02, val; tahlil-tur3 §5)
- Toplam hata 133,2; ≈ %93'ü molemax-dışı 619 görüntüde (ACC 0,800). En büyük hücre **mel → nv 36,0/111**; mel AUC
  0,938, bkl 0,958 (H08 sonrası bile mel→nv 28,4, mel AUC 0,952). Z2 (H04) ve Z4 (H08, kısmen) ele alındı; kalan kütle
  **temsil düzeyinde** (Z1).
- DINOv2'nin projedeki tek satırı donuk sonda (H01 0,8223); H02'nin 4,5 puan [+2,3; +6,6] altında. Makalenin başlık
  sorusu ("DINOv2'ye geçiş") **ince ayarlı bir DINOv2 satırı olmadan** cevapsız kalıyor → Tur-4'ün makale değeri TUT
  olasılığından bağımsız olarak yüksek.

### Ön-ölçüm (KOMUTLAR §T4/§T4b; ölçüm, karar değil; ≈ 0,62 GPU-sa)
B kısmi K = 2 133 s/ep · K = 4 260 s/ep (önbellek 3,16 GB, yazılı) · S tam ckpt 0 mikro 16 ≈ 216 s/ep (2,24 GB) · S tam
ckpt 1 290 s/ep · **B tam ≈ 1.050 s/ep → yerelde 5 seed × 15 ep ≈ 21,9 GPU-sa → yalnız Colab.** Bellek 4 GB'ı aşarsa
koşum ölmez, 10× yavaşlar (WDDM taşması) → kartlarda `epoch_sure_ort_sn` üst sınırı ve mikro-bs ön-kaydı.

### Puanlama
**öncelik = E[Δ val ACC vs H02, puan] × P(TUT) / maliyet (sa).** Maliyet = GPU-sa + uygulama-sa (kod hazır; kalan iş
kuyruk/beklenti yazımı). E = kart aralığının merkezi.

| Sıra | Kart | Aile / Z | Tek değişken (özet) | Beklenen ΔACC vs H02 (merkez) | P(TUT) | GPU-sa | Uyg.-sa | Öncelik |
|---|---|---|---|---|---|---|---|---|
| 1 | **H12** (analiz) | A4 ablasyonu (boyut) | H01'de B/14 → **S/14** donuk sonda | ≈ −6 (mutlak [0,787; 0,827]); vs H01 −3,5…+0,5 | — (tur kazananı olamaz) | 0,3 | 0,1 | ACC ile puanlanamaz; **H11'in temsil ebeveyni + Pareto satırı**, en ucuz |
| 2 | **H10** | A4 / Z1 | R18-IN tam FT → **DINOv2-B son 4 blok** (ilk 8 donuk, önbellekli) | −2,2…+2,8 (**+0,3**); vs H01 +2,3…+7,3 | 0,25 | 6,3 (üst 7,0) | 0,2 | 0,3 × 0,25 / 6,5 ≈ **0,012** |
| 3 | **H11** | A4 / Z1 | R18-IN tam FT → **DINOv2-S tam FT** (ckpt 0, mikro 16) | −3,2…+2,3 (**−0,5**); vs H12 +3…+9 | 0,15 | 5,2 (üst 5,8) | 0,1 | ≤ 0 → **ACC ile puanlanamaz; Pareto/mekanizma değeri** (erteleme kuralına tabi) |

### Neden bu sıra, neden bu üç kart
1. **H12 önce:** ≈ 0,3 GPU-sa ile H11'in "tam ince ayar kazancı"nı boyut farkından ayıran tek kol (H11 − H12). Yoksa H11'in
   tek DINOv2 kıyası H01 (B) olur ve boyut + ince ayar karışır. Ayrıca makalenin maliyet/Pareto satırı (Tur-1 O2 önerisi).
2. **H10 ikinci:** pozitif merkezli tek kart. **K = 4 ön-kayıtlı, val araması yok:** MAE §4.3 (He vd. 2022) kısmi ince
   ayar kazancının blok sayısıyla arttığını ve "4–6 bloğun tama yakın" olduğunu, güçlü-LP'li kontrastif temsillerde
   (MoCo v3) kazancın daha küçük kaldığını bildiriyor. K = 2'nin beklenen merkezi (≈ −0,7) H02'nin altında. K = 4, 15
   epoch'ta Colab tetiğinin altında kalan en derin seçenek (a3 5,5 GPU-sa).
3. **H11 üçüncü:** ACC beklentisi sıfır civarı (S < B temsil farkı ↔ tam uyarlama). Turda kalma gerekçesi **makale
   değeri**: "küçük omurgayı tam uyarlamak" ↔ "büyük omurganın üstünü uyarlamak" Pareto sorusu ve sayaç için ikinci,
   bağımsız bir atış (düşük düzey öznitelikleri de uyarlayan farklı mekanizma). **Erteleme kuralı (ön-kayıt, sonuca
   bakmaz):** H11 a3 başlarken günlük sayacı **> 23,0 GPU-sa** ise (beklenen ≈ 22,5) H11 a3 ertelenir, karar şef/kullanıcıda.
4. **Tek değişken (M2) ve paket beyanı:** H10/H11'in H02'ye karşı farkı **omurga değişimidir (A4)**; ViT ince ayarının
   gerektirdiği tarif uyarlamaları kartlarda madde madde beyan edildi (baş 1e-3 / gövde 1e-5, LLRD kapalı; **adım başına
   kosinüs**, ısınma 0; **wd yalnız ≥ 2-B, bias/LayerNorm/LayerScale/pos_embed/cls_token muaf**; 30 → 15 epoch; mikro-batch
   birikimi; H10'da fp16 önbellek). Bunlar ayrı ayrı test edilmez; makalede "paket" olarak raporlanır (H01/H02'nin
   baseline'a karşı statüsüyle aynı). Mekanizma düzeyindeki tek-değişken kıyaslar ön-beyanlı ikincillerdir: **H10 − H01**
   (aynı B omurgası: sonda → son-4-blok) ve **H11 − H12** (aynı S omurgası: sonda → tam).
5. **Karar hangi ebeveyne karşı:** **H02** (tarif ebeveyni = mevcut en iyi TUT; sayaç eşikleri bu zincirden). H01/H12'ye
   karşı Δ'lar ön-beyanlı aralıklarla raporlanır, **karar vermez**. H10 − H02 ÖLDÜR alıp H10 − H01 "saptandı" çıkarsa
   bu bir makale bulgusudur ("ince ayar DINOv2'yi sondadan kurtarır ama R18 ince ayarını geçirmez").
6. **Gövde lr: tek değer ön-kayıt (1e-5), ızgara yok.** Wu & Papież (arXiv:2501.14685) ViT'ler için kodlayıcı lr ızgarası
   {1e-4, 1e-5, 1e-6}'dan "1e-5 çoğu durumda en iyi" diyor; 1e-4'ün SSL ViT'i bozduğuna iki bağımsız kaynak var (Doerrich
   DINO E2E 81,31 < LP 84,42; Huang 2024 donuksuz < donuk). Izgara (a2'de 3 lr × 2 seed × 5 ep ≈ 1,1 GPU-sa) val seçim
   iyimserliği ve **M3 asimetrisi** (H02 aramasız tek tarif) ekler. **Reddedildi.**
7. **Kart olmayanlar / reddedilenler:**
   - **DINOv2-B tam ince ayar (K = 12) — KULLANICI ONAYI GEREKTİREN SEÇENEK (kart yazılmadı):** yerelde ≈ 1.050 s/epoch
     (5 seed × 15 ep ≈ 21,9 GPU-sa; bütçe dışı) → yalnız Colab. Gerekenler: konu.md 2026-10-10 09:25 ("Colab yalnız
     tıkanınca, eşleştirilmiş, rakiple aynı defterde"); donanım değişince H02 (ve karşılaştırılan her kol) aynı
     donanımda yeniden koşulur ya da `donanim` etiketiyle leaderboard'dan ayrılır; bütçe birimi ayrı sayılır. Ne zaman
     önerilir: H10 TUT alır ve (e) "az eğitim/derinlik" sinyali verirse, ya da H10 − H01 "saptandı" ama H10 − H02
     ÖLDÜR çıkarsa (Wu'nun 92,02 tavanına tam ince ayarla yaklaşılıp yaklaşılamadığı makalenin açık sorusu kalır).
     Şef kullanıcıya sunar; otomatik geçiş yok.
   - **K = 2:** daha ucuz (3,4 GPU-sa) ama beklenen merkez H02'nin altında. H10 TUT alırsa "kaç blok yeter" ablasyonu (yeni id).
   - **O1 (ViT-L/14 sonda, ≈ 0,25 GPU-sa):** kart sınırı (≤ 3). Bütçe kalırsa sonraki turda Pareto satırı.
   - **H10/H11 + D4 (H08 tarifi):** M2 — H08 TUT değil (yol A); yalnız H08 K5'te doğrulanır ve DINOv2 kolu TUT alırsa
     birleşim kartı. H10'da önbellek tek görünüm (8 görünüm ≈ 25 GB).
   - **H08b (seed {5..9}) ve H08 D2 ablasyonu:** tahlil-tur3 §3.3 / §6c — önerilmez / yalnız H08 kazanan zincirindeyse K5 öncesi.
   - **LLRD, ısınma, drop-path, LoRA:** her biri ayrı karar (A1/A2); M3 asimetrisi ve bütçe nedeniyle açılmadı.

### Tur-4 ortak ön-kayıtlar (kartlar arası)
- **Birincil ölçü: ham val ACC, tam val (n = 1.003), seçilen epoch (val ACC; eşitlikte AUC), tek görünüm.**
- **TUT eşiği (H10/H11):** 5-seed ort. ΔACC vs H02 ≥ **+0,5 puan** (mutlak ≥ **0,8722**) **ve** kesin tek yönlü
  seed-eşleştirilmiş t p < 0,05 **ve** (b') çelişki yok (görüntü bootstrap GA alt > 0). Çelişki → İNCELE, TUT verilmez.
- **MDE (ön-beyan; H02 seed std 0,28):**

  | Karşılaştırma | sd_Δ (seed) | MDE kesin t (1,374 × sd_Δ) | Bootstrap GA yarı genişliği | MDE bootstrap (≈ 2,8 × SE) |
  |---|---|---|---|---|
  | H10/H11 − H02 (farklı aileler) | 0,35–0,65 | **0,5–0,9** | ±1,3–2,0 (H02 − H01 gözlenen ±2,16; aynı aile H08 − H02 ±1,05) | **≈ 1,8–2,9** (bağlayıcı) |
  | H10 − H01, H11 − H12 (aynı omurga, sonda ↔ FT) | ≈ FT seed std 0,2–0,6 | 0,3–0,8 | ±1,3–2,0 | ≈ 1,8–2,9 |
  | H12 − H01 (iki sonda) | ≈ 0,15–0,25 | ≈ 0,2–0,35 | ±0,8–1,5 (tahmin) | ≈ 1,1–2,1 |

  Beklenen merkezler (+0,3 / −0,5) bootstrap MDE'nin çok altında: ÖLDÜR çıkarsa başarısız denemeler tablosuna
  **"n = 5 ve 1.003 val görüntüsüyle saptanamadı (MDE ≈ 2–3 puan); ölçülen Δ = …"** girer (Δ < 0 ise "H02'nin altında").
  Kural gevşetilmez.
- **Değerlendirme katmanları (karar vermez; tahlilden önce CPU'da):** ikinci satır **H04 katmanı** (`logit_ayari.py`,
  aynı `katlar.json` SHA256 `0db03edb…e0193`, τ 0:1:0,1, kısıt = varyantın kendi ham ACC'si − 0,010; OOF dengeli/ACC);
  üçüncü satır **R-T** (OOF ECE). TTA katmanı **yok** (H07 öldü).
- **Durma sayacı (konu.md 2026-10-10 20:15; okuma kuralı):** Tur-4'ün en iyisi **yalnız TUT alan H10/H11** arasından
  okunur (H12 analiz, sayılmaz). ACC ölçütü: kartın ham birincil satırı **≥ 0,8722**. Dengeli ölçütü: **aynı TUT kartın
  H04 katmanı OOF satırı ≥ 0,7475**. TUT ⇒ ACC ölçütü sağlanır (eşik aynı) → sayaç 0/2. Hiçbiri TUT almazsa **sayaç 2/2
  → K5** (Tur-5 açılmaz).
- **Ödünleşim kuralları (her kart):** Tur-1 nv kuralı; Δmolemax-dışı ≤ 0 → "Z1 hedefi tutmadı"; Δ vs kendi temsil
  ebeveyni ≤ 0 → "ince ayar donuk sondayı geçemedi" (makale bulgusu). Maliyet ödünleşimi (M7): `param_egitilen`, s/epoch,
  img/s her satırda (H02 11,18 M / 48,6 s; H10 28,36 M / 260 s; H11 21,63 M / ≈ 216 s).
- **Dengeli ACC için M6 eşiği yazılmadı** (A4 + DermaMNIST'te kaynak yok; H08 dersi). Dengeli Δ ayrıca **df/vasc hariç 5
  sınıflı** raporlanır. Görülmemiş eşikleri yalnız literatür kaynaklı: **5-seed ort. val ACC > 0,93 veya AUC > 0,99**
  (`literatur-cipa.md` §4(e)) + kart başına temsil ebeveynine karşı Δ eşiği.
- **Tabakalı S − N (ön-kayıtlı keşifsel yönler; ezber):** H10 > H01 (+4,0) ve H10 ≤ H02 (+6,65); H11 > H12, H11 ≥ H10;
  H12 ≈ H01. Eşleştirilmiş bootstrap; GA dışlamazsa "ayırt edilemedi". Kesin yanıt K5 DermaMNIST-C.
- **M3:** "Rakibe aynı önağırlık hakkı" satırı H02'dir (değişmez). Tur-4 DINOv2 kollarında **arama yok** (tek K, tek lr,
  tek N) → H02 ile arama bütçesi simetrik. H12'nin C ızgarası H01 ile aynı (sonda protokolü).
- **Birleşimler (M2):** açılmaz (yukarıda). Val bilgisi temiz kalsın diye Tur-4 tahlilinde H10/H11 × H02/H08 topluluk
  satırları **hesaplanmaz** (Tur-5'in ön-kayıtlı işi).
- **Beklenti kapıları:** seed başına `aralik` yalnız arıza alarmıdır (H10 a3 [0,80; 0,93], H11 a3 [0,78; 0,93], H12 a3
  [0,74; 0,88]). Görülmemiş kararı 5-seed ortalamasıyla tahlilcide.
- Test anahtarı yok (M4). K5'te tek test çıkarımı (TUT alan DINOv2 kolu + H12/H01 sondaları Pareto satırı olarak).

### Tur-5 ebeveyn kuralı (ön-kayıt; Tur-4 sonucundan önce)
- H10 ve H11'den TUT alanların 5-seed ham val ACC'si en yüksek olanı **"DINOv2-FT bileşeni"** ve Tur-5 tarif ebeveyni
  olur (eşitlikte val AUC). İkisi de TUT almazsa sayaç 2/2 → K5.
- H04 katmanı ve R-T aynı biçimde sürer.

### Koşum sırası (pratik)
1. **tur4-h12** (GPU ≈ 0,3 sa; a1 → a2 → a3; S öznitelik çıkarımı ilk satırda) → CPU: `h12k`, `rt_h12`.
2. **tur4-h10** (GPU ≈ 6,3 sa; a1 duman s0 1 ep (özdeşlik + önbellek tutarlılığı) → a2 s0,s1 × 5 ep → a3 s0–4 × 15 ep;
   tek başına, R6) → CPU: `h10k_h10_logitayar`, `rt_h10_sicaklik`.
3. **Erteleme denetimi** (şef: günlük sayacı ≤ 23,0?) → **tur4-h11** (GPU ≈ 5,2 sa; aynı kalıp) → CPU: `h11k`, `rt_h11`.
4. Tahlil (K4c) üç kart bittikten sonra, tek seferde.

**Tahmini Tur-4 toplamı:** GPU ≈ **11,8 sa** (üst 13,2), CPU < 0,3 sa, uygulama ≈ 0,4 sa, duvar saati ≈ 13 sa. **Bütçe
sayacı:** ≈ 15,3 (14,7 + Tur-4 ön-ölçümü 0,62) → ≈ **27,1 / 32 GPU-sa** (üst 28,5), tur 4/8. **K5 rezervi ≈ 1,5** (test
çıkarımları, DermaMNIST-C 224, H01/H12 test öznitelikleri) → ≈ 28,6 (üst 30,0) **≤ 32 ✓**. Kalan ≈ 3,4 (üst 2,0): Tur-5
topluluğu yalnız CPU (≈ 0 GPU). **Uyarı:** H08 K5'te doğrulanırsa M3 (rakibe D4 hakkı ≈ 2,7 GPU-sa) üst senaryoda
tam sığmaz → o durumda K5 öncesi kullanıcı kararı.

### Tur-5 ön notu — Z3: topluluk (A9; tablo kapatıcı) — taslak, kart Tur-5 başında yazılır
- **Önkoşul:** Tur-4'te ≥ 1 TUT (sayaç 0/2). Aksi halde K5; topluluk yalnız şef kararıyla K5 tablosunda karar vermeyen,
  eğitimsiz bir satır olarak (aynı ön-kayıtlı tasarımla) raporlanabilir.
- **Birleştirme kuralı (ön-kayıt):** **logit ortalaması** = log-olasılık ortalaması (argmax'ta logit ortalamasıyla özdeş;
  `tahmin_val.npy` olasılık sakladığı için), **ağırlıksız**; heterojende aileler eşit ağırlık. Üye/ağırlık seçimi val'de
  **yapılmaz** (iyimserlik). Olasılık ortalaması ikincil satır (M1: H02 5-seed olasılık ortalaması teşhisi 0,878, ECE
  0,037 görüldü; baseline 0,771).
- **Türler ve n = 1 sorununun çözümü:**
  - **T1 (birincil) — seed-eşleşik heterojen çiftler:** E_s = H02_s ⊕ DINOv2-FT_s (s = 0..4) → **5 ayrık topluluk, n = 5**
    (her üye tek bir topluluğa girer). Karar: E_s − H02_s ve E_s − DINOv2-FT_s üzerinde kesin tek yönlü
    seed-eşleştirilmiş t + eşleştirilmiş görüntü bootstrap'ı (2.000, RNG 0); TUT eşiği mevcut en iyi TUT + 0,5.
    Ek GPU yok. Tamamlayıcılık mekanizması: R18 (ImageNet CNN) ↔ DINOv2 (SSL ViT) hata korelasyonu (keşifsel:
    seed çifti argmax anlaşmazlığı ve hata örtüşmesi raporlanır).
  - **T2 — heterojen üçlü:** T1 + H01 sondası (eğitimsiz; sonda seed varyansı ≈ 0, n = 5 korunur).
  - **T3 — homojen 5-seed (n = 1):** H02 ×5 ve DINOv2-FT ×5. Bağlayıcı belirsizlik **görüntü bootstrap'ı: topluluk ↔
    üyelerin tek-model ortalaması** (H04/H07 mantığı); seed'e genelleme iddiası **yok** (beyan). Betimsel: 5 adet
    "bir seed dışarıda" 4'lü topluluk (örtüşen, bağımsız değil). Ayrık ek seed {5..9}: DINOv2 için ≈ 5,5 GPU-sa →
    **bütçe yok, açılmaz.**
  - **T4 — H08 içeren topluluklar (H08_s ⊕ DINOv2-FT_s):** H08 TUT değil (yol A) → val'de yalnız **koşullu satır**;
    K5'te H08 test doğrulamasını geçerse raporlanır, geçmezse atılır. M2 gereği Tur-5 kararına girmez.
- **M3:** rakibe aynı topluluk hakkı — baseline homojen 5-seed (teşhis 0,771) ve T1'in rakip eşi (baseline_s ⊕ H02_s).
- **Katmanlar:** H04 katmanı (topluluk olasılıklarına, aynı katlar) ve R-T. TTA yok.
- **Beklenti çıpası (kart yazımında kesinleşir):** kütüphane A9 0,3–1 puan; heterojen tamamlayıcılık için DermaMNIST
  kaynağı aranacak. Görülmemiş: val ACC > 0,93 / AUC > 0,99.
- **Maliyet:** CPU < 0,1 sa; GPU 0. Makale katkısı değil (A9: tablo kapatıcı).

### Revizyon notu — 2026-10-11 (Tur-4 ön-kaydı; hiçbir Tur-4 ölçüm koşumundan önce)
Eklenenler: Tur-3 kapanış tablosu (H07/H09 ölüm gerekçeleri, H08 yol A), Tur-4 kartları (H10, H11, H12), puanlama,
ortak ön-kayıtlar (karar ebeveyni H02, temsil ebeveynleri H01/H12, MDE tablosu, sayaç okuma kuralı, dengeli M6 dersi),
gövde lr ön-kaydı (1e-5, ızgara reddi), K = 4 gerekçesi, H11 erteleme kuralı, B tam ince ayar Colab notu (kullanıcı
onayı), bütçe, Tur-5 ebeveyn kuralı ve Tur-5 topluluk ön notu (n = 1 tasarımı). Tahlilcinin önerileriyle eşleme
(tahlil-tur3 §5): öneri 1 = H10, öneri 2 = H11 (Colab gerekmedi: S tam ≈ 216 s/ep, 5 seed 4,5 GPU-sa), öneri (3) O2 =
H12 (O1 açılmadı). Yeni kaynaklar: He vd. 2022 (MAE §4.3 kısmi ince ayar), Huang vd. 2024 (arXiv:2402.07595, düşük güven),
Wu & Papież lr metni (HTML v1, erişim 2026-10-11). Tur-3 bölümü aşağıda "(TARİHSEL)" başlığıyla değiştirilmeden korunuyor.

---
---

# (TARİHSEL) Tur-3 sıralaması — 2026-10-11, ön-kayıtlı (otomatik koşum)

**Durum (o tarihte): TUR-3 ÖN-KAYITLI (2026-10-11).** Kartlar H07, H08, H09. "KULLANICI ONAYI GEREKİR" işaretli (protokol/veri/bütçe
değiştiren) kart **yok**. Split, veri, metrik ve epoch seçimi değişmiyor. Tur toplamı (≈ 4,9 GPU-sa) kalan bütçenin
(≈ 22) içinde. Tur-3 otomatik koşar. Tur-2 ve Tur-1 bölümleri aşağıda tarihsel kayıt olarak korunuyor.

## Kapanmış kartlar (Tur-2; `sonuclar/tahlil-tur2.md` §3)
| Kart | Karar | Bu tura etkisi |
|---|---|---|
| H04 (post-hoc logit ayarı, H02) | **TUT (ödünleşim)**: OOF dengeli 0,7425, Δ +3,63 [+1,57; +5,95]; OOF ACC 0,8576 (Δ −0,96 [−1,66; −0,32]) | Tur-3'te her varyanta **ikinci satır** (değerlendirme katmanı). Durma sayacının dengeli ölçütü buradan okunur |
| H05 (sınıf ağırlıklı CE, β = 0,5) | **ÖLDÜR**: p_t 0,097 (n = 5'te saptanamadı, MDE ≈ 2,9 puan) | **H05 öldü çünkü** eğitim-zamanı ağırlıklandırma, train kaybı ≈ 0'a inen ağda sönüyor (etkin τ ≈ 0,2) ve post-hoc ödünleşim eğrisinin ancak +0,46 puan üstünde kalıyor. Tur-3'te Z2'ye GPU harcanmaz. H04 + H05 birleşimi açılmaz (M2) |
| H06 (logit ayarı, H01; analiz) | **Genelleşir** | K5'te τ\* = 0,2 (val karşılığı +1,45) |
| R-T (sıcaklık; rapor) | H02 OOF ECE 0,0745 → 0,0310 | Tur-3'te **üçüncü satır** |

Sıradaki boş kart id'si (bu turdan sonra): **H10**.

---

## TUR-3 — Z4: aşırı uyum + kaynak kısayolu (ebeveyn `h02_r18in_ft224`)

### Hata kütlesi (H02, val; tahlil-tur1 §6a, tahlil-tur2 §1b)
- Toplam hata 133,2. **≈ %93'ü (≈ 124) molemax-dışı 619 görüntüde** (ACC 0,800). Molemax 384 görüntüde ≈ 0,976,
  yani ≈ 9 hata. Tur-3'ün her ACC kazancı pratikte molemax-dışından gelmek zorunda.
- En büyük hücre mel → nv (36,0/111). 62 hep-yanlış örneğin 21'i mel → nv. ≈ 158 kararsız örnek (1–4 seed yanlış).
- **Z4 sinyalleri:** train kaybı s0'da ep 5'te 0,012 (ezber), ECE 0,0745 (baseline × 2), ort. güven − ACC +7,0 puan,
  tabakalı S − N +6,65 (kol içi p 0,009; lezyon ezberi işareti), seed çifti anlaşmazlığı %9,0. Molemax (≈ 0,976) ile
  molemax-dışı (0,800) arasındaki fark, ölçülmemiş "kaynak kısayolu" hipotezini (hata-analizi §6b) açık tutuyor.
- Z2 (nadir sınıf karar sınırı) H04 ile kapatıldı. Kalan Z2 kütlesi örnek sayısıyla sınırlı (df 12, vasc 14) → yeni GPU
  harcaması yok (tahlil-tur2 §8). Z1 kalanı (mel AUC 0,938) Tur-4'ün işi.

### Puanlama
**öncelik = E[Δ val ACC, puan] × P(TUT) / maliyet (sa).** Maliyet = GPU-sa + uygulama-sa; yalnız CPU süren işler en az
0,1 sa sayılır. E = kart aralığının merkezi. P, MDE (aşağıda) ve literatür tutarlılığından öznel olarak tahmin edildi.

| Sıra | Kart | Aile / Z | Tek değişken | Beklenen ΔACC vs H02 (merkez) | P(TUT) | GPU-sa | Uyg.-sa | Öncelik |
|---|---|---|---|---|---|---|---|---|
| 1 | **H07** | A8 / Z4 | Çıkarımda 8 dihedral görünümün olasılık ortalaması (H02 `best_model.pth` ×5; eğitim yok) | −0,5 … +1,5 (+0,5) | 0,35 | 0,1 | 0,25 | 0,5 × 0,35 / 0,35 ≈ **0,50** |
| 2 | **H08** | A3 / Z4 | Train'de örnek başına rastgele D4 dönüşümü (GPU), 30 ep, geri kalan aynı | −0,5 … +1,5 (+0,5) | 0,30 | 2,4 (üst 2,7) | 0,3 | 0,5 × 0,30 / 2,7 ≈ **0,056** |
| 3 | **H09** | A3 / Z4 (kaynak kısayolu) | Train'de örnek başına parlaklık/kontrast/doygunluk U[0,8; 1,2], ton yok | −1,0 … +1,0 (0) | 0,15 | 2,4 (üst 2,7) | 0,2 | ≈ 0 → **ACC ile puanlanamaz; mekanizma değeri** |

### Neden bu sıra, neden bu üç kart
1. **H07 önce:** en ucuz (≈ 5 dk GPU, yalnız çıkarım) ve Tur-3'ün ACC sorusuna doğrudan cevap. Görünüm başına ACC ve
   D2 ↔ 90° ablasyonu, modelin yönelime ne kadar duyarlı olduğunu da ölçer. **Bu bilgi H08'in beklentisine girmez:**
   H08 ve H09, H07 koşumundan önce yazıldı (M1).
2. **H08 ikinci:** pozitif ACC beklentisi olan tek eğitim kartı ve Z4'ün ana mekanizması (ezber). Dermoskopide yönelim
   tanısal olmadığı için D4 etiket koruyan bir simetridir. Paylaşımsız koşulur (R6).
3. **H09 üçüncü:** ACC beklentisi sıfır merkezli, TUT olasılığı düşük. Turda kalma gerekçesi **makale değeri**: hata
   analizinin açık bıraktığı "kaynak kısayolu" hipotezinin ilk ölçümü (Δmolemax-dışı − Δmolemax imzası). Bu ölçüm,
   makalenin "rakip neden yanılıyor / iyileştirme nereden geliyor" analizini besler. Bütçe izin veriyor (aşağıda).
   **Erteleme kuralı (ön-kayıt):** H09 başlamadan önce günlükteki bütçe sayacı > 17 GPU-sa ise (beklenen ≈ 12,3; tahmin
   aşımı) H09 Tur-4 sonrasına ertelenir. Bu, sonuca bakmayan bir bütçe kuralıdır.
4. **Tek değişken bölünmesi (M2):** Tahlil önerisi "dihedral + hafif renk" tek karttı. İki ayrı mekanizma (geometrik
   simetri ↔ fotometrik/kaynak ipucu) ve ters yönlü olası etkiler (renk tanısal bilgi taşıyor) nedeniyle H08 ve H09
   olarak bölündü. Birleşim yalnız ikisi de TUT alırsa ayrı birleşim kartıdır.
5. **Kart olmayanlar / reddedilenler:**
   - **RandAugment / MixUp / CutMix / rastgele kırpma:** çok parametreli (tek değişken değil). Rastgele kırpma lezyon
     merkezli dermoskopide sınırı kesebilir. DermaMNIST'te kaynak yok. Tur-3'te yok.
   - **Seed topluluğu (A9):** Tur-5 (istatistik tasarımı önce ön-kayıtla çözülmeli; aşağıda).
   - **Epoch uzatma:** yalnız H08/H09 öldürme kuralı (e) tetiklenirse, yeni id'li kart olarak.
   - **H04 / H06 yalnız-nv ablasyonu (ii):** kart değil. K5 öncesi deney-yürütücünün CPU işi (D5 eksiği, tahlil-tur2 §6c).
   - **Focal kayıp:** Tur-2 notu geçerli (Zhang & Li T1d: 224'te ACC −0,5).

### Tur-3 ortak ön-kayıtlar (kartlar arası)
- **Birincil ölçü: ham val ACC, tam val (n = 1.003).** Seçilen epoch, tek görünüm (H08/H09). H07 için 8 görünüm
  ortalaması. **Z4'ün hedef ölçüsü `val_acc_molemax_disi`** her kartta ön-beyanlı **mekanizma ikincilidir**, karar
  vermez. Neden birincil değil: (1) Projenin seçim metriği ve durma sayacının ACC ölçütü tam val ACC'dir. Turlar arası
  kıyas buna bağlı. (2) Molemax alt kümesi tavanda (≈ 0,976), bu yüzden ΔACC ≈ 0,62 × Δmolemax-dışı mekanik olarak
  izler. Birincil yapmak gücü yalnız ≈ 1,27 kat artırır (Δ 1,62×, SE 1,27×), ama ACC ile çelişen sonuçlarda karar
  kuralını bölerdi. (3) İkisi ayrışırsa bu bir bulgudur ve kartlarda **tutarlılık notu** olarak ön-kayıtlıdır (ΔACC > 0 ama
  Δmolemax-dışı ≤ 0 → "Z4 hedefi tutmadı").
- **TUT eşiği:** 5-seed ort. ΔACC ≥ **+0,5 puan** (mutlak ≥ 0,8722 = durma sayacının ACC eşiği).
- **Karar istatistiği:** H07 → iki kol aynı modelden geldiği için **bağlayıcı olan görüntü bootstrap'ıdır** (GA alt ≤ 0 →
  ÖLDÜR). t ile çelişki → İNCELE. H08/H09 → kesin tek yönlü seed-eşleştirilmiş t (scipy 1.17.1, p ≥ 0,05 → ÖLDÜR) +
  eşleştirilmiş görüntü bootstrap'ı (val, 2.000, RNG 0). Çelişki → İNCELE, TUT verilmez.
- **MDE (ön-beyan; ACC'de H02 seed std 0,28):**

  | Kart | sd_Δ (seed) | MDE kesin t (= 1,374 × sd_Δ) | Bootstrap GA yarı genişliği | MDE bootstrap (≈ 2,8 × SE) |
  |---|---|---|---|---|
  | H07 (aynı model) | ≈ 0,2–0,4 (tahmin) | 0,3–0,55 (yanıltıcı derecede güçlü) | ≈ ±0,45–0,65 (H04: 42,8 karar → ±0,67) | **≈ 0,65–0,95** (bağlayıcı) |
  | H08 / H09 (yeniden eğitim) | 0,40 (bağımsız) – 0,63 (gözlenen H05 − H02) | **≈ 0,55–0,87** | ≈ ±0,7 (H05 − H02: ±0,68) | ≈ 1,0 |

  Beklenen merkezler (+0,5 / 0) MDE'nin altında. ÖLDÜR, başarısız denemeler tablosuna "etki yok" diye değil,
  **"n = 5'te saptanamadı (MDE ≈ 0,9 puan)"** diye girer (ölçülen Δ < 0 ise "zarar verdi"). Kural gevşetilmez.
- **Değerlendirme katmanları (karar vermez; tahlilden önce CPU'da):**
  - **İkinci satır: H04 katmanı.** `logit_ayari.py`, aynı `katlar.json` (SHA256 `0db03edb…e0193`), τ ızgarası 0:1:0,1.
    Seçim kümelerinde kısıt = varyantın **kendi** ham ACC'si − 0,010. OOF dengeli ACC ve OOF ACC raporlanır.
  - **Üçüncü satır: R-T.** `sicaklik.py`, aynı katlar, OOF ECE.
  - Keşifsel kıyas: varyant+H04 − H04 (dengeli, bootstrap).
- **Durma sayacı (konu.md 2026-10-10 20:15, iki ölçüt; okuma kuralı):** Tur-3'ün en iyisi **yalnız TUT alan H07/H08/H09**
  arasından okunur.
  - ACC ölçütü: kartın **ham birincil satırı**, eşik **≥ 0,8722** (H02 0,8672 + 0,5).
  - Dengeli ölçütü: **aynı TUT kartın H04 katmanı OOF satırı**, eşik **≥ 0,7475** (H04 OOF 0,7425 + 0,5). Elmayla elma
    karşılaştırma budur: önceki en iyi dengeli değer de bir H04 OOF değeriydi. Kartların ham dengeli ACC'si sayaçta
    kullanılmaz.
  - Hiçbir kart TUT almazsa ya da TUT alanlar iki eşiğin hiçbirini geçmezse sayaç **1/2** olur. Sayaç şu an 0/2.
- **Ödünleşim kuralları (her kart):** Tur-1 nv kuralı (ΔACC > 0, Δdengeli ≤ 0, Δnv payı > +0,02 → "nv yanlılığı" notu).
  Molemax-dışı tutarlılık notu. H09'da ayrıca mel AUC Δ < −1,0 → "renk tanısal bilgisi" notu.
- **Tabakalı S − N (ön-kayıtlı keşifsel yön, üç kartta da ↓):** augmentasyon/TTA lezyon kimliğinin piksel düzeyinde
  ezberini zayıflatır. Karar vermez. Eşleştirilmiş bootstrap'ta GA üst < 0 → "ezberi azaltır" desteklenir; değilse
  "ayırt edilemedi" denir (val bu soruyu çözemeyebilir; kesin yanıt K5 DermaMNIST-C).
- **M3 (rakibe aynı hak):** H07'nin TTA'sı baseline'ın 5 `best_model.pth`'sine yan kol olarak uygulanır
  (`h07x_baseline_tta8`, ayrı kuyruk, karar vermez). H08/H09 TUT alırsa rakibe aynı augmentasyon hakkı (≈ 2,7 GPU-sa/kart)
  **K5 öncesi şef/kullanıcı kararıdır**. Tur-3'te koşulmaz. Adil kıyas satırı augmentasyonsuz H02'dir.
- **Birleşimler (M2):** H08 + H09, H07 × H08, H07 × H09 yalnız ikisi de TUT alırsa ayrı birleşim kartıdır. Val bilgisi
  temiz kalsın diye **Tur-3 tahlilinde birleşim satırları hesaplanmaz** (H08/H09 çıktısına TTA uygulanmaz).
- **Beklenti kapıları:** seed başına `aralik [0.80, 0.90]` yalnız arıza alarmıdır. Üst 0,90 = H02 tavanı, alt 0,80 =
  bozuk eğitim/görünüm. Görülmemiş eşikleri **5-seed ortalaması** biriminde yazıldı ve tahlilcide uygulanır
  (tahlil-tur2 §3.2 önerisi).
- **Kaynak düzeltmesi:** Önceki notlardaki "Zhang & Li 2026 T1d 0,882 (augmentasyonlu)" ifadesi doğrulanmadı. Makale
  (PMC13413053, erişim 2026-10-11) augmentasyon tanımlamıyor. H08/H09 beklentisine girdi değildir.
- Test anahtarı yok (M4). K5'te tek test çıkarımı (H07 TUT alırsa test'e de aynı 8 görünüm, bir kez).

### Tur-4 ebeveyn kuralı (ön-kayıt; Tur-3 sonucundan önce)
- **Eğitim tarifi ebeveyni:** H08 ve H09'dan TUT alanların 5-seed ham val ACC'si en yüksek olanı (eşitlikte val AUC).
  İkisi de TUT almazsa `h02_r18in_ft224` kalır.
- **H07 TUT alırsa:** TTA bir eğitim tarifi değil, H04 gibi bir **değerlendirme katmanı** olur. Tur-4+ varyantlarına
  ayrı satırda uygulanır. Kartların birincil kararı ham satırla verilir. Bu durumda Tur-4+ durma sayacının ACC ölçütü
  **TTA katmanı satırından** okunur ve eşik = (en iyi TTA'lı ACC) + 0,5 olur (elmayla elma). H07 TUT almazsa ham satır
  kullanılır.
- H04 katmanı ve R-T aynı biçimde sürer.

### Koşum sırası (pratik)
1. **tur3-h07** (GPU ≈ 5 dk; a1 duman s0 → a3 s0–4) + **tur3-h07x** (M3 baseline TTA, ≈ 1 dk) → CPU: `h07k_h07_logitayar`
   + `rt_h07_sicaklik` (≈ 3 dk).
2. **tur3-h08** (GPU ≈ 2,4 sa; a1 iki duman (kapalı özdeşlik + açık) → a2 ön-uçuş s0,s1 × 5 ep → a3 s0–4 × 30 ep; tek
   başına, R6) → CPU katmanlar.
3. **tur3-h09** (GPU ≈ 2,4 sa; aynı kalıp; erteleme kuralına tabi) → CPU katmanlar.
4. Tahlil (K4c) üç kart bittikten sonra, tek seferde.

**Tahmini Tur-3 toplamı:** GPU ≈ **4,9 sa** (üst 5,6), CPU < 0,3 sa, uygulama ≈ 0,8 sa, duvar saati ≈ 5,5 sa. Bütçe
sayacı: ≈ 9,8 → ≈ **14,7 / 32 GPU-sa**, tur 3/8. Kalan ≈ 17,3 GPU-sa. K5 rezervi ≈ 1,5 GPU-sa (test çıkarımları,
DermaMNIST-C 224, H01 test öznitelikleri, H07 TUT ise test TTA ≈ 3 dk). Tur-4…8 için ≈ 15,8 GPU-sa kalır.

### Sonraki turlar (taslak; kartlar her tur başında yazılır)

**Tur 4 — Z1 kalanı: DINOv2 ince ayarı (A4 omurga değişimi; Pareto bulgusu)**
- Hedef: mel → nv (36/111), mel AUC 0,938, bkl AUC 0,958 (temsil).
- Çıpa: Wu & Papież 2025 (hakemsiz; split/seçim belirsiz) aynı protokolde DINOv2-B E2E 224 **92,02** ↔ ResNet-18 E2E 224
  **86,05** (Δ ≈ +6) ↔ DINOv2-B LP 83,97. Görülmemiş: val ACC > 0,93 veya AUC > 0,99 (`literatur-cipa.md` §4(e)).
- Seçenekler (tek değişken = omurga; tarif Tur-4 ebeveyn kuralından):
  - **(a) DINOv2-B son k blok kısmi ince ayar** (kart yazımında tek k). Donuk ilk 12 − k bloğun çıkış token'ları önbelleğe
    alınır (r224: 257 × 768 × 7.007 float16 ≈ 2,8 GB; tahlil-tur2 §8). Yerelde olası. **Kısıt:** önbellek tek görünümdür.
    H08/H09 TUT alıp ebeveyn olursa augmentasyon önbellekle çakışır (D4 için 8 × 2,8 ≈ 22 GB disk gerekir). Bu çatallanma
    Tur-4 kart yazımında ebeveyn kuralına göre açıkça çözülür.
  - **(b) DINOv2-S/14 tam ince ayar.** Yerel hız ölçülmedi. B'nin tam ince ayarı 7 img/s'tir (5 seed ≈ 28 sa, bütçe
    dışı; `env/hiz-olcumu.md`). S için ≈ 4 kat hız varsayılırsa (tahmin) 5 seed ≈ 10–12 GPU-sa olur, kalan bütçenin
    ≈ %70'i. Yerelde makul değil.
- **Önce ölçüm:** Tur-4 kartı yazılmadan (a) ve (b) için 1 epoch'luk yerel GPU dumanıyla s/epoch ve tepe bellek
  ölçülür (≈ 15 dk; ölçüm, karar değil).
- **Colab tetiği:** 5-seed a3 tahmini > **6 GPU-sa** (kalan bütçenin ≈ %40'ı) ya da epoch > 300 s ise Colab'a geçiş
  **önerilir**. Bu geçiş **KULLANICI ONAYI GEREKİR** niteliğindedir: konu.md 2026-10-10 09:25 kararı "Colab yalnız
  tıkanınca, eşleştirilmiş, rakiple aynı defterde" der. Donanım değişince ebeveyn (H02/H08) aynı donanımda yeniden
  koşulmalı ya da ayrı `donanim` etiketiyle leaderboard'a karıştırılmamalı. Bütçe birimi de değişir. Otomatik geçiş
  yapılmaz, öneri şefe gider.
- O1/O2 (ViT-L/14, ViT-S/14 sondası; ≈ 0,25 / 0,05 GPU-sa): Tur-4'te kart sınırı (≤ 3) içinde Pareto satırları.

**Tur 5 — Z3: topluluk (A9; son itme, tablo kapatıcı)**
- Görülmüş teşhis (M1 beyanı): H02 5-seed olasılık ortalaması ACC 0,878 (+1,1), dengeli 0,721, ECE 0,037. Baseline'da
  0,771 (+3,1).
- **İstatistik sorunu önce çözülür (ön-kayıt):** 5-seed topluluğu n = 1 verir. Önerilen tasarım: (i) bağlayıcı
  belirsizlik görüntü bootstrap'ı (topluluk ↔ tek modellerin ortalaması, H04/H07 mantığı); (ii) betimsel: 5 adet
  "bir seed dışarıda" 4'lü topluluk (örtüşen, bağımsız değil); (iii) ayrık topluluklar için ek seed {5..9}, ≈ 2,1–2,4
  GPU-sa, yalnız bütçe kalırsa.
- Türler: homojen (en iyi tarif × 5 seed) ve heterojen (DINOv2 sonda + R18 FT; tamamlayıcılık). Rakibe aynı hak (M3;
  baseline 5-seed ortalaması).
- H07 TUT alırsa katman sırası ön-kayıtla yazılır (önce görünüm ortalaması, sonra seed ortalaması).
- Makale katkısı değildir (kütüphane A9: tablo kapatıcı).

### Revizyon notu — 2026-10-11 (Tur-3 ön-kaydı; hiçbir Tur-3 koşumundan önce)
Eklenenler: Tur-2 kapanış tablosu (H05 ölüm gerekçesiyle), Tur-3 kartları (H07, H08, H09), puanlama, ortak ön-kayıtlar
(birincil ölçü, MDE tablosu, katman satırları, durma sayacının okunma kuralı, M3, birleşim yasağı), Tur-4 ebeveyn kuralı,
koşum sırası, Tur-4 Colab tetiği ve Tur-5 taslağı. Tahlilcinin önerileriyle eşleme: H07 = öneri 1. Öneri 2 (dihedral +
renk) tek değişken kuralıyla **H08 (dihedral)** ve **H09 (renk)** olarak bölündü. Öneri 3 (topluluk) Tur-5'te. Kaynak
düzeltmesi: Zhang & Li 2026 augmentasyon kullanmıyor (PMC13413053). Tur-2 bölümü aşağıda "(TARİHSEL)" başlığıyla
değiştirilmeden korunuyor.

---
---

# (TARİHSEL) Tur-2 sıralaması — 2026-10-11, ön-kayıtlı (otomatik koşum)

**Durum (o tarihte): TUR-2 ÖN-KAYITLI (2026-10-11).** Kartlar H04, H05, H06 ve aşağıdaki R-T raporlama protokolü. "KULLANICI ONAYI
GEREKİR" işaretli (protokol/veri/bütçe değiştiren) kart **yok**. Tur-2, bütçe içinde ve protokol değiştirmediği için
otomatik koşar. Tur-1 bölümleri aşağıda tarihsel kayıt olarak korunuyor.

## Kapanmış kartlar (Tur-1; `sonuclar/tahlil-tur1.md` §4)
| Kart | Karar | Bu tura etkisi |
|---|---|---|
| H01 (DINOv2-B sonda r224) | **TUT**: 0,8223 ± 0,0013, Δ +8,28 [+5,66; +10,83] | H06'nın ebeveyni (analiz) |
| H02 (R18-IN ince ayar 224) | **TUT** (ödünleşim: ECE 0,0745): 0,8672 ± 0,0028, Δ +12,76 [+10,67; +14,86] | **Tur-2 ebeveyni** (ebeveyn kuralı: H02 − H01 +4,49 [+2,31; +6,62]) |
| H03 (çözünürlük merdiveni, analiz) | T1 reddedilmedi · T2 kabul · T3 "28 px'te de kazandırır" (nv notuyla) | C\* karıştırıcısı gözlemi → aşağıdaki C\* notu |

Öldürülen kart: **yok**. Sıradaki boş kart id'si: **H07**.

---

## TUR-2 — Z2: nadir sınıf karar sınırı (ebeveyn `h02_r18in_ft224`)

### Hata kütlesi (H02, val; tahlil §6a)
Toplam hata 133,2'dir; nv–mel–bkl üçgeni bunun %66,7'si. En büyük tek hücre mel → nv (36,0/111). nv fazla tahmin
ediliyor (703/671); mel (97/111) ve nadir sınıflar (df 8,0/12, vasc 10,4/14, akiec 29,2/33) eksik. Nadir sınıflarda AUC
0,96–0,99 iken duyarlılık 0,53–0,64: **bilgi var, karar yok (Z2)**. Tur-2 bu kütleyi hedefler. Z1'in kalanı (mel ↔ nv
temsili) Tur-3/4'ün işi.

### Puanlama
**öncelik = E[Δ dengeli ACC, puan] × P(TUT) / maliyet (sa).** Maliyet = GPU-sa + uygulama-sa; yalnız CPU süren işler
en az 0,1 sa sayılır. E = kart aralığının merkezi. P, bootstrap gücü ve literatür tutarlılığından öznel olarak
tahmin edildi.

| Sıra | Kart | Aile / Z | Tek değişken | Beklenen (Δ vs ebeveyn) | P | GPU-sa | Uyg.-sa | Öncelik |
|---|---|---|---|---|---|---|---|---|
| 1 | **H04** | A8 / Z2 | H02 kararına post-hoc logit ayarı, τ çapraz uydurmayla | OOF dengeli **+1,5…+4,0**; ACC −1,0…+0,3 | 0,60 (dengeli GA'nın dar olmaması nedeniyle (b') İNCELE riski %30–40) | 0 | 0,25 | 2,75 × 0,60 / 0,3 ≈ **5,5** |
| 2 | **H06** (analiz) | A8 / Z2, genelleme | aynı ayar H01 sondasında | OOF dengeli +0,5…+3,0; ACC −1,0…0 | 0,50 ("genelleşir") | 0 | 0,05 | 1,75 × 0,5 / 0,1 ≈ 8,8, **ama H04 koduna bağımlı** ve tur kazananı olamaz |
| 3 | **H05** | A5 / Z2 | eğitimde sınıf ağırlıklı CE, β = 0,5 (arama yok) | dengeli **+1,0…+4,0**; ACC −1,5…+0,3 | 0,40 (MDE ≈ 3 puan, n = 5) | 2,3 (üst 2,6) | 0,2 | 2,5 × 0,4 / 2,5 ≈ **0,4** |

### Neden bu sıra, neden bu üç kart
1. **H04 önce:** en ucuz (yalnız CPU) ve Z2'nin doğrudan karşılığı. Görülmüş teşhisler (τ = 1, nv eşitleme) yönü
   destekliyor. Bu yüzden karar iyimserliksiz OOF değeriyle verilir. Makale değeri: "aşırı güvenli ince ayarlı CNN'de
   önsel ayarı ACC'yi ≤ 1 puan düşürerek dengeli doğruluğu artırır mı?"
2. **H06 ikinci:** H04 ile aynı kod ve aynı katlar, marjinal maliyeti ≈ 0. Etkinin temsil türünden (SSL ViT sonda ↔
   ImageNet CNN ince ayar) bağımsız olup olmadığını sınar. Makalenin DINOv2 satırı için de gerekli. Ebeveyni en iyi
   TUT olmadığı için **analiz kartıdır** (H03 statüsü).
3. **H05 son:** aynı önsel kaydırmanın eğitim sırasında yapılması (β = 0,5 ⇔ τ = 0,5 Bayes eşdeğeri). Yeni bilgisi
   şu: "Temsil de uyarlanınca post-hoc ayara bir şey ekleniyor mu?" Maliyeti 2,3 GPU-sa ve gücü sınırlı (MDE ≈ 3 puan),
   bu yüzden en düşük öncelik. GPU paylaşımsız koşsun diye (R6) CPU işlerinden sonra koşulur.
4. **Kart olmayanlar:**
   - **Sıcaklık ölçekleme** (tahlilcinin H07 önerisi): kart değil, aşağıdaki **R-T raporlama protokolü**. Argmax'ı
     değiştirmez, ACC ve dengeli ACC'ye etkisi tanım gereği 0'dır, durma sayacına katkısı yoktur. İşlevi H02'nin M7
     ödünleşimini (ECE × 2) raporlamaktır.
   - **H04 ile sıcaklığın birleştirilmesi gereksiz (M2):** argmax_c [z_c/T − τ log π_c] = argmax_c [z_c − Tτ log π_c].
     T yalnız τ'nun ölçeğini değiştirir ve τ zaten ızgarada seçiliyor. R-T, H04 çıktısına argmax'ı bozmadan uygulanır.
   - **Focal kayıp:** önerilmiyor. Literatürde 224 px'te ACC −0,5 ve makro-ACC −0,5 (Zhang & Li T1d). H04/H05'ten
     ayırt edilemeyecek kadar küçük.
   - **C\* gözlemi:** analiz notu, kart değil (aşağıda).

### Tur-2 ortak ön-kayıtlar (kartlar arası)
- **Birincil etki:** `val_dengeli_acc` (ön-kayıtlı). **Kısıt:** val ACC ≥ ebeveyn − 1,0 puan (H02 için 0,8572, H01 için
  0,8123). Kısıt ihlali ÖLDÜR'dür ve başarısız denemeler tablosuna "ACC bedeli" olarak girer.
- **Karar istatistiği:** **kesin** tek yönlü seed-eşleştirilmiş t testi (scipy 1.17.1, deney venv'i; tahlil §0
  düzeltmesi) + eşleştirilmiş görüntü bootstrap'ı (val, 2.000, RNG 0; dengeli ACC için `scripts/tahlil_tur1.py` ile
  aynı yöntem). İkisi çelişirse İNCELE, TUT verilmez. Post-hoc kartlarda (H04, H06) bağlayıcı olan bootstrap'tır.
- **Post-hoc parametre kuralı (iyimserlik):** val'de seçilen her son işlem parametresi (τ, T) **5 katlı, sınıf-tabakalı,
  lezyon-gruplu çapraz uydurmayla** raporlanır (`StratifiedGroupKFold`, RNG 20261011, tek `katlar.json`, SHA256 kartlar
  arasında eşit). Karar OOF değeriyle verilir. Tam-val değeri iyimser bir ikincil satırdır. K5'te test'e tam-val
  parametresi **bir kez** uygulanır. Train tahminleri seçimde kullanılmaz: ince ayarlı model train'e aşırı uyumludur
  ve önsel yanlılığı orada görünmez (H04 §5).
- **Durma sayacı (konu.md 2026-10-10 20:15, iki ölçüt):** Tur-2'nin en iyisi yalnız **TUT alan H04/H05** arasından
  okunur. H04 için **OOF** değer kullanılır, tam-val değeri kullanılmaz. H06 ve R-T sayılmaz. Sayaç 0/2'de kalır, eğer
  en iyi val ACC ≥ **0,8722** ya da en iyi val dengeli ACC ≥ **0,7112** ise. Tur-2 ön-kayıtlı olarak ACC-nötrdür;
  sayacı dengeli ölçüt belirler. Hiçbiri TUT almazsa sayaç 1/2 olur.
- **Ödünleşim (Tur-2'de ters yön):** Δdengeli > 0 ve ΔACC < 0 beklenir. ACC bedeli her satırda yazılır.
- **M3:** H04 protokolü baseline'a da uygulanır (`h04x_baseline_logitayar`, karar vermez). Rakibe aynı post-hoc hakkı
  verilir.
- Test anahtarı yok (M4). K5'te tek test çıkarımı.

### R-T — Sıcaklık ölçekleme raporlama protokolü (ön-kayıt; KART DEĞİL)
- **Amaç:** H02'nin kalibrasyon bedelinin (ECE 0,0745 = B × 2) makalede dürüst ve iyimserliksiz bir çözüm satırı.
  ACC ve dengeli ACC'yi değiştirmez (Guo vd. 2017: argmax korunur).
- **Yöntem:** seed başına tek T ∈ [0,5; 3,0] (adım 0,01), seçim kümesinde NLL'yi en küçükleyen değer. H04'ün aynı 5
  katıyla çapraz uydurma yapılır, OOF ECE-15 raporlanır. Uygulandığı yerler: H02, H01, baseline, H04 OOF çıktısı ve
  (TUT alırsa) H05. Tam-val T iyimser ikincildir ve K5'te test'e bir kez uygulanır.
- **Görülmüş:** tahlil teşhisi (val'de uydurulan, iyimser) H02 için T 1,48–1,86 ve ECE 0,0315.
- **Beklenti:** H02 OOF ECE **[0,025; 0,045]**. Guo vd. 2017 Tablo 1: ImageNet ResNet-152 5,48 → 1,86, CIFAR-100
  ResNet-110 16,53 → 1,26 (göreli düşüş %66–92). Bizde n = 1.003 ve 15 kutulu ECE'nin gürültü tabanı ≈ 0,02–0,03
  olduğundan alt uç bununla sınırlı. **Görülmemiş:** OOF ECE < 0,010 → İNCELE (tahmin gürültüsünün altında).
  **Kural:** OOF ECE ≤ 0,045 ise makale "ödünleşim ölçeklemeyle giderilir" der. Değilse "kısmen giderilir" der ve iki
  değeri de yazar.
- Maliyet CPU ≈ 2 dk. Durma sayacına ve tur kart sınırına girmez. Kod: `varyantlar/logit_ayari/sicaklik.py` (aynı katlar).

### C\* gözlemi — analiz notu (kart DEĞİL)
- **Gözlem (tahlil §3c):** sondada C, val ACC ile seçildiğinde ACC ≈ eşit iki C arasındaki seçim (C = 0,01 ↔ 0,1)
  dengeli ACC'yi 5–9 puan oynatıyor. Güçlü L2 azınlık logitlerini büzüyor.
- **Neden bu turda kart değil:** (1) **Tek değişken / karışma:** C\*'yi dengeli ACC ile seçmek, H06 ile aynı
  mekanizmaya (sondanın karar sınırını azınlığa kaydırmak) aynı ebeveynde ikinci bir kol açar. H06'nın yorumunu bulandırır
  ve "C mi τ mu?" sorusunu iki değişkenli yapar. (2) **Maliyet/değer:** makalenin ihtiyacı (çözünürlük figüründe dengeli
  ACC'yi sabit C'de göstermek; tahlil I-grubu) mevcut lbfgs teşhis verisiyle (`tahlil_tur1_teshis.json`
  `lbfgs_C_teshisi`) **sıfır ek koşumla** karşılanıyor. (3) Projenin ana seçim metriği (val ACC) sondada değişmemeli;
  turlar arası kıyas ona bağlı.
- **Ne yapılır:** K5/makale figüründe H03 eğrisi, dengeli ACC için sabit C = 0,01 ve C = 0,1'de **ayrıca** çizilir
  (tahlilci, CPU). H06 "genelleşmez" çıkarsa, "C ve τ'nun dengeli ölçütle ortak seçimi" Tur-3+'te ayrı bir kart adayı
  olur (yeni id).

### Tur-3 ebeveyn kuralı (ön-kayıt; Tur-2 sonucundan önce)
- **Eğitim tarifi ebeveyni:** `h02_r18in_ft224` kalır. **İstisna:** H05 TUT alır **ve** 5-seed val ACC'si H02'ninkine
  eşit ya da yüksekse H05 ebeveyn olur. Tur-3 (augmentasyon/TTA) ACC hedefli olduğu için ebeveyn ACC ile seçilir.
- **H04 TUT alırsa:** H04 protokolü (yeniden çapraz uydurmalı τ) Tur-3 varyantlarına ve ebeveynlerine aynı biçimde
  uygulanan bir **değerlendirme katmanı** olarak ikinci satırda raporlanır. Tur-3 kartlarının birincil kararı ham
  çıktıyla verilir (tek değişken).
- H04 + H05 birleşimi yalnız ikisi de TUT alırsa, ayrı bir **birleşim kartı** olarak yazılır (M2).

### Koşum sırası (pratik)
1. **tur2-h04** (CPU, a1 → a3, ≈ 10 dk): yan kollar `h04x_h02aucsecim_logitayar` ve `h04x_baseline_logitayar` aynı
   kuyrukta.
2. **tur2-h06** (CPU, ≈ 5 dk) → **R-T** (CPU, ≈ 2 dk).
3. **tur2-h05** (GPU, a1 → a2 → a3, ≈ 2,3 sa, tek başına).

**Tahmini Tur-2 toplamı:** GPU ≈ **2,3 sa** (üst 2,6), CPU < 0,5 sa, uygulama ≈ 0,5 sa. Bütçe sayacı: ≈ 7,5 → ≈ **9,8 /
32 GPU-sa**, tur 2/8. Kalan ≈ 22 GPU-sa Tur-3 (augmentasyon ≈ 2,1 sa/kart; TTA eğitimsiz) ve Tur-4 (kısmi DINOv2 ince
ayarı) için yeterli.

### Sonraki turlar (taslak; kartlar her tur başında yazılır)
- **Tur 3 — Z4/Z1 kalanı:** H02 tarifinde (a) eğitim augmentasyonu (dihedral + renk), (b) TTA (eğitimsiz). Hedef ölçü
  `val_acc_molemax_disi` ve mel → nv hücresi (36/111; hep yanlış 62 örneğin 21'i). Ebeveyn yukarıdaki kurala göre.
- **Tur 4 — Z1 kalanı:** DINOv2 kısmi ince ayarı (son bloklar; yerelde pahalı → Colab tetiği, plan).
- **Tur 5 — Z3:** topluluk. H02 5-seed olasılık ortalaması teşhiste +1,1 ACC verdi. Rakibe de topluluk hakkı (M3).
- **Omurga önerileri O1/O2** (ViT-L/14, ViT-S/14): Tur 4 ile birlikte Pareto bulgusu olarak.

### Revizyon notu — 2026-10-11 (Tur-2 ön-kaydı; hiçbir Tur-2 koşumundan önce)
Eklenenler: Tur-1 kapanış tablosu, Tur-2 kartları (H04, H05, H06), puanlama, ortak ön-kayıtlar (OOF kuralı, kesin p,
durma sayacının okunma kuralı), R-T protokolü, C\* notu, Tur-3 ebeveyn kuralı ve koşum sırası. Tur-1 bölümleri aşağıda
korunuyor. Yalnız başlıklarına "(TARİHSEL)" eklendi ve durma kuralı satırı "karar verildi" olarak güncellendi.
Tahlilcinin önerileriyle eşleme: H04 = öneri 1, H05 = öneri 2, H06 = öneri 3. Öneri 4 (sıcaklık) kart değil, R-T
protokolü oldu; önerilen id H07 kullanılmadı ve boş kaldı.

---
---

# (TARİHSEL) Tur-1 sıralaması — 2026-10-10, kullanıcı onaylı

**Durum (o tarihte): ONAYA HAZIR (2026-10-10)** — K3 (replikasyon YEŞİL) ve K3.5 (`sonuclar/hata-analizi.md`) sonuçlarıyla
güncellendi. Tur-1 kartları (H01, H02, H03) zorunlu ilk kullanıcı onayına tabi. "KULLANICI ONAYI GEREKİR" (protokol/veri
değişikliği) işaretli kart **yok**.

## İç çıpa (baseline, val, 5 seed; hata-analizi §0)
B = val ACC **0,7396 ± 0,0205** · AUC 0,9228 · dengeli ACC 0,4621 · makro-F1 0,460 · ECE 0,0365 · nv payı 0,672 ·
molemax-dışı ACC 0,609 (n = 619) · tabakalı S − N +2,8 puan. Runlar `20261010-171037/174311/181653/184922/192128_baseline_s0…s4`.
Test: 0,7296 ± 0,0176 (replikasyon YEŞİL; dondurulmuş, K5'te aynen).

## K3.5 hata kütlesi → aile eşlemesi (hata-analizi §7)
| Z | Ölçülen zayıflık | Aile | Nerede |
|---|---|---|---|
| Z1 | nv–mel–bkl üçgeni hataların %64,6'sı; mel/bkl AUC 0,874/0,882 (temsil zayıf); molemax-dışı ACC 0,609 | A4 | **Tur 1** (H01, H02, H03); kalan kısım Tur 4 |
| Z2 | Nadir sınıflarda bilgi var, karar yok: df duyarlılık 0,017 (AUC 0,949), vasc 0,457, akiec 0,382; dengeli ACC 0,462 | A5 + A8 (önsel/eşik) | **Tur 2** |
| Z3 | Seed/seçim varyansı: std 2,05 puan, ACC ↔ nv payı r = 0,88, 296 kararsız örnek | A9 (+ A11 seçim kuralı) | Tur 5; tasarım kuralı: Tur-2/3 kıyasları düşük varyanslı ebeveyne karşı + bootstrap |
| Z4 | Aşırı uyum + olası kaynak kısayolu: train ACC 1,000 (ep ≥ 51); molemax 0,949 ile molemax-dışı 0,609 | A3 + A8 (TTA) | **Tur 3** |
| Z5 | Sızıntı ölçüsü karıştırıcılı (ham −18,8, tabakalı +2,8) | ölçüm (A12 tadında) | Tüm kartlar: yalnız tabakalı S − N; K5 DermaMNIST-C |

## Puanlama (Tur-1)
**öncelik = E[Δ val_acc, puan] × P(Δ ≥ +2 puan) / maliyet (sa)**. maliyet = yerel GPU-sa (GTX 1650 FP32;
`env/hiz-olcumu.md`, `varyantlar/KOMUTLAR.md` §4) + kalan uygulama-sa (varyant kodu yazıldı; kalan iş küçük
düzeltmeler). E[Δ]: kartın mutlak aralığının merkezi − B. P: bağımsız literatür sayılarının tutarlılığından öznel tahmin.

| Sıra | Kart | Aile / Z | Tek değişken | Beklenen val ACC (mutlak; Δ vs B) | P | GPU-sa | Uyg.-sa | Öncelik |
|---|---|---|---|---|---|---|---|---|
| 1 | **H01** | A4 / Z1 | 28 px sıfırdan R18 → donuk DINOv2-B CLS (native 224) + L2 softmax-LR sonda | [0,820; 0,860]; +8,0…+12,0 (merkez +10,5) | 0,90: iki bağımsız LP-224 sayısı (83,97 / 84,42) 0,5 puan içinde | **0,4** (r224 önbelleği hazır) | 0,3 | 9,45 / 0,7 = **13,5** |
| 2 | **H03** (analiz) | A4 ablasyonu / Z1 | girdi bilgi çözünürlüğü r ∈ {28, 64, 128} (bicubic → 224), sonda aynı | r28 [0,730; 0,780] (T3: −1…+4); r64 [0,775; 0,825]; r128 [0,795; 0,845]; T2 Δ(224 − 28) +4…+10 | — (her sonuç bulgu) | **0,6** (üst 1,1) | 0,1 | puanlanamaz; **makale değeri en yüksek** (başlıktaki "düşük çözünürlükte transfer" sorusu + Wu ↔ Ren çelişkisi), en ucuzlardan |
| 3 | **H02** | A4 + adil kıyas (M3) / Z1, Z4 | rakibe ImageNet önağırlığı + native 224 + literatür ince ayar tarifi | [0,820; 0,885]; +8,0…+14,5 (merkez +11,3); ayrıca `val_acc_aucsecim` ile | 0,90: üç farklı tarifin hepsi ≥ +8,8 | **2,0** (üst 2,5) | 0,3 | 10,1 / 2,3 = **4,4** |

### Neden bu sıra (Tur-1)
1. **H01 önce:** konunun tezi; kazanç/maliyet en yüksek; öznitelik önbelleği zaten var. Tur 2–3'ün olası ebeveyni. Sonda
   seed varyansı neredeyse sıfır olduğundan, K3.5'in "küçük etkiler R18'e karşı saptanamaz" bulgusu (Z3; saptanabilir en
   küçük etki ≈ 3,9 puan, sondada ≈ 2,7) H01 ebeveynli turlarda hafifliyor.
2. **H03 ikinci:** ACC kazancı hedeflemiyor, ama 1 GPU-sa'in altında ana analiz figürünü (r'ye karşı ACC/AUC) ve H01
   kazancının temsil/çözünürlük ayrıştırmasını üretiyor. T3'ün gücü sınırda (Δ = +2'de p ≈ 0,047). Bootstrap eşlik ediyor,
   "kararsız" çıkma olasılığı kartta ön-kayıtlı.
3. **H02 üçüncü:** kazancı H01 kadar olası, maliyeti yaklaşık 5 kat. Tur-1'den **çıkarılmaz**: "aynı önağırlık hakkı
   rakibe" satırı olmadan (alan-protokolleri §03, M3) DINOv2 üstünlük iddiası hakemde çöker.

### Koşum sırası (Tur-1; pratik)
Tek GPU, sıralı, `--paralel 1`. Kademeli kapılar (M8): duman → ön-uçuş → tam.
1. **tur1-h01**: r224 önbellek hazır, ≈ 20–25 dk.
2. **tur1-h03** (r28, r64) ≈ 35–45 dk, ardından **tur1-h03-r128** ≈ 20 dk. Önbellek hazırsa daha kısa.
3. **tur1-h02** ≈ 1,9 sa, tek başına (paylaşımsız süre ölçümü, R6).

**Tahmini Tur-1 toplamı:** GPU ≈ **3,0 sa** (üst sınır ≈ 3,7; KOMUTLAR §4 tahmini 3,2 sa, on-ozn önbelleği düşülünce
biraz azalır). Duvar saati, kapılarla ≈ 3,5–4 sa. Bütçe: 32 GPU-sa'ten K3 ve on-ozn sonrası kalanın yaklaşık %11–13'ü
(kesin sayaç şef günlüğünde, M10).

## Tur-1 ortak ön-kayıtlar (kartlar arası)
- **Karar istatistiği (tüm kartlar):** seed-eşleştirilmiş tek yönlü t-testi + **eşleştirilmiş görüntü bootstrap'ı** (val,
  2.000, RNG 0, aynı indeksler iki kola). İkisi çelişirse İNCELE; TUT verilmez.
- **Ödünleşim kuralı (tüm kartlar):** Δval_acc > 0 ve Δdengeli ACC ≤ 0 ve Δ`tahmin_nv_payi` > +0,02 → "ACC kazancı nv
  yanlılığından" notu (hata-analizi §2: corr(ACC, nv payı) = +0,88).
- **Ön-beyanlı ikinciller (tüm kartlar):** `val_dengeli_acc`, `val_makro_f1`, `val_ece` (kıyas 0,0365, yön keşifsel),
  `tahmin_nv_payi`, `val_acc_molemax_disi`, mel/bkl OvR AUC, `val_sizinti_farki_tabakali`.
- **H01 ve H02 karşılaştırması (yön ön-kaydı: belirsiz):** |Δ| ≤ 4 puan beklenir (LP-224 ≈ 84 ile R18-FT-224
  82,3–88,2). H01 kolunun varyansı küçük olduğundan saptanabilir fark ≈ 2,7 puan; altı "ayırt edilemedi".
- **Sızıntı (keşifsel yön):** tabakalı S − N'de H02 > H01 ve H02 > baseline (+2,8); H03'te r ↑ → fark ↑.
- **Tur-2 ebeveyn kuralı:** H01 ve H02'den 5-seed ortalama val_acc'si en yüksek TUT varyant ebeveyn olur (eşitlikte
  val_auc). **Fark < 1,0 puan ve (t-testi p ≥ 0,05 veya bootstrap GA 0'ı kapsıyor) ise H01 seçilir.** Gerekçe: Tur 2–3
  manipülasyonları önbellekli öznitelik üzerinde dakikalar sürer ve sonda ebeveyni seed varyansını yok ederek ±1–2
  puanlık etkileri saptanabilir kılar (Z3).
- **H02 ÖLDÜR alsa da** leaderboard'da "rakip + ImageNet önağırlık" satırı olarak kalır. ÖLDÜR yalnız "Tur-2 ebeveyni
  olamaz" anlamına gelir.
- Test anahtarı yok (M4). Test çıkarımı/özniteliği K5'te bir kez.

## (TARİHSEL) Tur-1'de yazılmış sonraki tur taslakları — yerini yukarıdaki TUR-2 bölümü aldı

### Tur 2 — Z2: dengesizlik / karar sınırı (ebeveyn: Tur-1 kuralına göre, büyük olasılıkla H01)
| Öneri | Değişen tek şey | Birincil etki (ön-kayıt adayı) | ACC beklentisi | Görülmemiş | Maliyet |
|---|---|---|---|---|---|
| D1 | sonda çıktısına **önsel/logit ayarı**: log p(c\|x) − τ·log π_train(c) (Menon vd. 2021); τ ızgarası val'de | **dengeli ACC +2,3…+4,2 puan** (Zhang & Li 2026 T1a/T1d, makro-ACC); df/vasc/akiec duyarlılığı ↑ | ebeveyn ± 1 puan (ılımlı τ). τ = 1'de belirgin düşüş: baseline teşhisi +6,3 dengeli / −15,7 ACC (hata-analizi §2) | ACC'de > +2 puan (literatur-cipa §4(c)) | eğitimsiz, CPU dakikalar |
| D2 | sonda kaybında **sınıf ağırlığı** w_c ∝ π_c^(−β) (tek β) | aynı | ebeveyn ± 1 (WS: −1,5…+0,3; focal −0,4/−0,5) | aynı | 5 seed × sonda, < 0,1 GPU-sa |

- **Seçim metriği ön-kayıt sorunu:** τ/β val ACC ile seçilirse dengesizlik müdahalesi sıfıra çöker (ACC nv yanlılığını
  ödüllendiriyor; Z3). Önerilen ön-kayıt: τ/β seçimi **val dengeli ACC** ile ve **val ACC ≥ ebeveyn − 1,0 puan** kısıtı
  altında yapılır. Bu kart düzeyinde bir seçim kuralıdır, proje protokolü değişmez. Projenin ana seçim metriği (val ACC)
  turlar arası karşılaştırmada aynen kalır.
- **KULLANICI KARARI (durma kuralı etkileşimi) — VERİLDİ 2026-10-10 20:15:** durma kuralı iki ölçütlü (ardışık 2 turda ne
  val ACC ne val dengeli ACC ≥ 0,5 puan iyileşmezse → K5; konu.md).
- Odak kaybını (focal) bir doğrusal sonda için öncelikli görmüyorum: literatürde ACC −0,4/−0,5, AUC +0,2/+0,5; D1/D2'den
  ayırt edilemeyecek kadar küçük. Ebeveyn H02 olursa yeniden değerlendirilir.

### Tur 3 — Z4: augmentasyon + TTA (ayrı kartlar)
| Öneri | Değişen tek şey | Beklenti | Görülmemiş | Maliyet (sonda ebeveyni) |
|---|---|---|---|---|
| U1 | **TTA**: val'in 8 dihedral görünümünün öznitelik/olasılık ortalaması | 0…+2 puan. Medeiros 2026 +1,6 (R18 28 px) yalnız yön için, sonda için kaynak yok | > +3 puan | 8 × 1.003 ÷ 26 img/s ≈ 5 dk |
| U2 | **Eğitim augmentasyonu**: train'in k dihedral + renk kopyasının öznitelikleri | **belirsiz**: DermaMNIST'e özgü kaynak yok (literatur-cipa §4(d)) | > +3 puan | k × 4,5 dk (k = 4 → ≈ 18 dk + sonda) |

- Asıl hedef ölçü **`val_acc_molemax_disi`** (hata kütlesinin %92,6'sı). U2'nin renk bileşeni K3.5'in ölçülmemiş "kaynak
  kısayolu" hipotezini sınar.
- ECE kıyası **ebeveynin kendi ECE'sine** göre yapılır. Medeiros'un 0,196 → 0,039 kazancı yakınsamış R18 içindir, sonda
  için referans değil.

### Tur 4 — Z1'in kalanı: DINOv2 kısmi ince ayar (sondanın tavanı)
Son blokların ince ayarı (S veya B; B/L'nin son blokları önbellekli). Wu: E2E 224 92,02, LP 83,97 → tavan farkı yaklaşık
+8 puan (hakemsiz, protokolü belirsiz). Tam B ince ayarı yerelde 5 seed ≈ 28 sa → Colab tetiği (plan).

### Tur 5 — Z3: topluluk (son itme)
DINOv2 sonda + CNN ince ayar logit ortalaması (tamamlayıcılık). Sonda seed topluluğu anlamsız (varyans ≈ 0). Rakibe de
topluluk hakkı (M3; baseline 5-seed olasılık ortalaması teşhisi 0,771, hata-analizi §5).

## Öneri (Tur-1'de kart DEĞİL): omurga boyutu
| Öneri | Değişen tek şey | Beklenen | Görülmemiş | GPU-sa | Not |
|---|---|---|---|---|---|
| O1 | H01'de ViT-B/14 → **ViT-L/14** (1024-b CLS) | Δ(L − B) **0…+2 puan**: ImageNet LP 84,5 → 86,3 (DINOv2 README / Oquab 2024); DermaMNIST'te S/B/L ≈ eşit (Ren 2025, 28 → 224) | Δ > +4 | ≈ 0,25 (8.010 ÷ 10 img/s = 13,4 dk + sonda) | FP32 sığar |
| O2 | ViT-B/14 → **ViT-S/14** (384-b) | Δ(S − B) **−3…0 puan** (ImageNet LP −3,4; DermaMNIST sayısı **yok**) | Δ > +2 | ≈ 0,05 | Pareto noktası |

Neden Tur 1'de değil: tur başına ≤ 3 kart; "daha büyük model kazandı" makale katkısı değil (A4 riski). Doğru zamanı, H01
TUT aldıktan sonra Tur 4 ile birlikte bir Pareto/maliyet bulgusu olarak. "H01b" adı kullanılmaz (düzeltme kartına
ayrılmış).

## Revizyon notu — 2026-10-10 (K3 + K3.5 sonrası; ölçüm koşumu yok)
Değişenler: iç çıpa B ve Z1–Z5 eşlemesi eklendi. Beklentiler mutlak aralıklarla yazıldı (H02 üst uç +14,5). Bütçe
KOMUTLAR §4 ve on-ozn önbelleğiyle güncellendi (Tur-1 ≈ 3,0 GPU-sa). Öncelik puanları kodun hazır olmasıyla yeniden
hesaplandı; sıra değişmedi. Ortak ön-kayıtlara bootstrap, ödünleşim kuralı ve tabakalı sızıntı eklendi. Tur-2 ebeveyn
kuralına bootstrap koşulu eklendi. Tur 2/3 notları Z2/Z4'e bağlandı; Tur-2 durma kuralı etkileşimi kullanıcı kararı
olarak işaretlendi. Omurga önerilerinin geçici id'leri (H04/H05) kaldırıldı (O1/O2), böylece ardışık id'ler kart
yazımına kalıyor.


## K5 ÖN-KAYDI — H08 TEST DOĞRULAMA KURALI (yol A; KULLANICI KARARI 2026-10-11 09:42; test AÇILMADAN önce yazıldı)
Kaynak: tahlil-tur3.md §3.3 "A yolu için K5 doğrulama ölçütü taslağı" — şef tarafından aynen benimsendi.
1. Test (n = 2.005), seed {0..4}, H08 − H02 eşleştirilmiş. **Doğrulandı ⇔ ΔACC ≥ +0,5 puan VE görüntü bootstrap %95 GA alt sınırı > 0.**
   Aynı test görünümünde iki aday raporlanır; H08 doğrulanırsa makaledeki R18 kazanan satırı H08, doğrulanmazsa H02.
2. Görülmemiş denetimi test'te aynı eşiklerle (ΔACC > 3, ACC > 0,90, AUC > 0,99, dengeli Δ > 6,5 → rapor bayrağı); dengeli Δ ayrıca 5 sınıflı (df/vasc hariç).
3. DermaMNIST-C (birincil: resmî piksel + C bölmesi): H08 − H02 ΔACC GA alt > 0 → "kazanç lezyon ezberinden değil" desteklenir.
4. M3: H08 makale kazananı olursa rakibe aynı augmentasyon hakkı (baseline 28 px + D4, ≈ 2,7 GPU-sa) — K5 öncesi şef/kullanıcı kararı. Adil kıyas satırı augmentasyonsuz H02.
Tur-4 tarif ebeveyni: **h02_r18in_ft224** (mekanik; Tur-3'te TUT alan yok). Durma sayacı: **1/2**.


## ÖN-KAYIT — COLAB TEKRAR KAPISI (KULLANICI KARARI 7, 2026-10-11 11:39; hiçbir Colab ölçüm koşumundan ÖNCE yazıldı)
Ortam: Colab Pro, NVIDIA L4, Colab'ın kendi Python/torch'u (bağlantı testi: Python 3.13.15, torch 2.11.0+cu130), **FP32, TF32 KAPALI**
(`torch.backends.cuda.matmul.allow_tf32=False`, `torch.backends.cudnn.allow_tf32=False`; GTX 1650'de TF32 yok → yerel ile aynı aritmetik sınıf), AMP YOK.
Kod: yerel WS'nin aynı dosyaları (paket SHA256 kaydı). Veri: aynı npz'ler (resmî MD5 doğrulaması Colab'da).
Tekrar kolları (seed 0–4, yerel tariflerin aynısı): baseline (rakip 28 px), H01 (DINOv2-B sonda r224), H02 (R18 ImageNet 224), H08 (H02 + D4).
**Uyum kuralı (her kol için, val, 5-seed ortalaması):**
- baseline: test_acc ortalaması replikasyon aralığında [0,7218; 0,754] (aynı kapı, `replikasyon_kapisi.py`); ayrıca |val_acc_Colab − 0,7396| ≤ 2,0 puan (yerel seed std 2,05).
- H01, H02, H08: |val_acc_Colab − val_acc_yerel| ≤ 1,5 puan (yerel: H01 0,8223, H02 0,8672, H08 0,8893) VE sıralama korunur (H08 > H02 > H01).
- Dengeli ACC farkları bilgi amaçlı raporlanır (karar vermez; nadir sınıf oynaklığı).
**Kapı sonucu:** hepsi uyumlu → Tur-4 ve K5 Colab'da, kıyaslar YALNIZ Colab kollarıyla. Uyumsuzluk → manipülasyon durur; teşhis (yazılım sürümü, deterministiklik,
veri hattı); sonuç "donanım/yazılım duyarlılığı" bulgusu olarak kaydedilir; kullanıcıya sorulur.
Yerel Tur-1–3 kararları (kart TUT/ÖLDÜR, ebeveyn seçimleri) geçerli kalır ("seçim ortamı"); Colab kolları "doğrulama ortamı". K5 testi yalnız Colab kollarıyla, bir kez.
