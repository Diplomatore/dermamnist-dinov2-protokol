# Hipotez Sıralaması — TUR-5 bölümü (SIRALAMA.md'nin devamı) — Konu 3.1 DermaMNIST

**Durum: TUR-5 ÖN-KAYITLI (2026-10-11).**
- Bu turda tek kart var: **H13** (topluluk, yalnız CPU). "KULLANICI ONAYI GEREKİR" işaretli kart **yok**: split, veri, metrik, epoch seçimi ve donanım değişmiyor, GPU kullanılmıyor.
- Tur-5 otomatik koşar.
- K5 ön-kaydı ayrı dosyada: **`hipotezler/K5-ONKAYIT.md`**. O dosya KULLANICI ONAYI GEREKTİRİR.
- Tur-4 ve önceki bölümler `SIRALAMA.md` içinde değiştirilmeden duruyor (tarihsel kayıt).
- Sıradaki boş kart id'si: **H14**.

## 1. Kapanmış kartlar (Tur-4; `sonuclar/tahlil-tur4.md` §3)
| Kart | Karar | Bu tura etkisi |
|---|---|---|
| H10 (DINOv2-B son 4 blok, Colab) | **TUT**: +2,91 [+1,34; +4,61] vs H02c, p_t 3,4·10⁻⁴. Ödünleşim: süre ×3,3. Ezber notu: kazancın %75'i S'de; C-val +1,05 [−0,66; +2,79] | **Tur-5 ebeveyni** ("DINOv2-FT bileşeni"; SIRALAMA Tur-5 ebeveyn kuralı, mekanik) |
| H11 (DINOv2-S tam, Colab) | **TUT**: +1,44 [+0,16; +2,81]. Ödünleşim: ECE 0,080, τ\* ızgara ucunda | Bileşen değil (H10 daha yüksek); K5'te Pareto satırı |
| H12 (S sonda; analiz) | **BEKLİYOR** (koşuyor) | H11'in görülmemiş 3. maddesi; K5 önkoşulu P1 |

**Öldürülen kartlar** (önceki turlar; tekrar edilmez):
- **H05 öldü çünkü** eğitim-zamanı sınıf ağırlığı, train kaybı ≈ 0'a inen ağda sönüyor.
- **H07 öldü çünkü** augmentasyonsuz ağda dönüşmüş görünümler dağılım dışında kalıyor (görünüm başına −2,6 puan).
- **H09 öldü çünkü** fotometrik oynama bu tarifte zayıf düzenleyici.

H07'nin ölümü Tur-5'i doğrudan etkiler: **TTA katmanı topluluğa eklenmez**, topluluk yalnız seed/aile boyutunda kurulur.

## 2. Hata kütlesi (H10, val; tahlil-tur4 §1b)
- Toplam hata 105,6/1.003 (H02c 134,8). Molemax-dışı ACC 0,8407. En büyük hücre mel → nv 24,6/111. Seed std 0,66 puan, seed argmax anlaşmazlığı %6,4.
- **Kalan kütleler:**
  - **Z3 (varyans).** H10'un seed std'si H02c'nin (0,19) 3,5 katı. Aile içi anlaşmazlık %6,4 → A9.
  - **Z1 kalanı.** Temsil ailesi tükendi: H10 > H11 > H02c. B tam ince ayarı kullanıcı onayı gerektiriyor ve tetikleyicisi oluşmadı.
  - **Z2.** H04 katmanı sürüyor.
  - **Z4.** H08 → K5.
- **Ezber boyutu:** H10'un H02c'ye kazancı S'de yoğun (Δ_S +6,26, Δ_N +1,13). İki aile N'de yakın. Bu yüzden topluluğun N'de kazanç üretmesi beklenir. Ön-kayıtlı ikincil olarak H13'te sınanır.

## 3. Puanlama
**öncelik = E[Δ val ACC vs H10, puan] × P(TUT) / maliyet (sa).** Maliyet = GPU-sa + uygulama-sa; yalnız CPU süren işler en az 0,1 sa sayılır.

| Sıra | Kart | Aile / Z | Tek değişken (özet) | Beklenen Δ vs H10 (merkez) | P(TUT) | GPU-sa | Uyg.-sa | Öncelik |
|---|---|---|---|---|---|---|---|---|
| 1 | **H13** | A9 / Z3 | H10_s olasılıklarının log'u, seed-eşleşik H02c_s ile ½ ; ½ ortalanır | −0,8…+1,5 (**+0,4**) | 0,25 | 0 | 0,3 | 0,4 × 0,25 / 0,3 ≈ **0,33** |

### Neden tek kart, neden bu tasarım
1. **H13 = T1 (SIRALAMA Tur-5 ön notunun birincil türü).**
   - Topluluklar seed-eşleşik ve ayrık olduğu için n = 5 ve seed-eşleştirilmiş t geçerli. Ek GPU gerekmez.
   - Üyelerin ikisi de TUT almıştır (H02 Tur-1, H10 Tur-4). Bu nedenle M2'nin birleşim şartı sağlanır, ama kart tek aile (A9) ve tek ayardır.
   - Ağırlık/üye araması yapılmaz (iyimserlik, M3 asimetrisi).
2. **T2 (+ H01c üçlü), T3 (homojen 5-seed), T4 (H08c ⊕ H10) kart değildir.** H13'ün ön-beyanlı, **karar vermeyen** ikincil satırlarıdır.
   - Gerekçe (çatallanan yol yasağı): beş topluluk türünden en iyisini val'de seçmek iyimserlik üretir. Tur-5'in kazananı yalnız T1 olabilir.
   - T4: H08 TUT değil (yol A). Val'de koşullu satırdır. K5'te her durumda yazılır, iddia taşıması H08 doğrulamasına bağlıdır (K5-ONKAYIT §1 (8')).
3. **Bu turda GPU kartı yok:**
   - K5 rezervi (test 0,2 + C1 4,1 + C2 ≤ 3,2 ≈ 7,5 L4-sa) düşülünce kalan ≈ 2,8.
   - Aday GPU kartları kural dışı ya da ACC hedefsiz:
     - K = 2 ablasyonu: Pareto, ACC beklentisi H10'un altında.
     - B tam ince ayar: **KULLANICI ONAYI GEREKİR**; tetikleyici olan (e) sinyali oluşmadı, en iyi epoch ≤ 13.
     - H10 + D4: M2 gereği H08 doğrulanmadan açılmaz.
     - Gövde lr / LLRD: M3 asimetrisi.
   - Kart sınırı ≤ 3 doldurulmadı. Doldurmak yalnız sayı için olurdu.
4. **Beklentinin ana sınırlayıcısı (önceden yazıldı):** üyeler arasında g = 2,91 puan fark var. Eşit ağırlıkta topluluk ancak çeşitlilik kazancı g/2 ≈ 1,45'i aşarsa ebeveyni geçer. P(TUT) bu yüzden düşük (0,25). ÖLDÜR çıkarsa öğrenilen cümle hazır: "Zayıf üye ebeveyni aşağı çekti."

## 4. Tur-5 ortak ön-kayıtlar
- **Birincil ölçü:** ham val ACC, tam val (n = 1.003), topluluk argmax'ı, seed başına bir topluluk.
- **TUT eşiği:** 5-seed ort. ΔACC vs H10 ≥ **+0,5** (mutlak ≥ **0,8997**) **ve** kesin tek yönlü seed-eşleştirilmiş t p < 0,05 **ve** (b') çelişkisi yok (bootstrap GA alt > 0). Ayrıntı H13 kartında.
- **MDE (ön-beyan):** t MDE ≈ 0,4–0,8, MDE_bootstrap ≈ 1,0–1,7 (bağlayıcı). Beklenen merkez +0,4 MDE'nin altında. Kural gevşetilmez.
- **Görülmemiş:** Δ vs H10 > +2,0 (A9 üst × 2), val ACC > 0,93 ya da AUC > 0,99 → İNCELE.
- **Katmanlar (karar vermez):** H04 (aynı `katlar.json`, kısıt = topluluğun kendi ham ACC'si − 0,010) ve R-T. TTA yok.
- **Ön-kayıtlı ikinciller:** S/N ayrışımı ("topluluk ezberi azaltır mı"; yönler H1 Δ_N > Δ_S, H2 tabakalı S − N ↓), C-val alt kümesi, çeşitlilik ölçüleri (çapraz aile anlaşmazlığı > aile içi), olasılık ortalaması satırı.
- **M3:** baseline_c homojen 5-seed (n = 1) ve baseline_c_s ⊕ H02c_s (n = 5). Betimseldir.
- **M1:** aileler arası hiçbir topluluk satırı ve hata örtüşmesi görülmedi (tahlil-tur4 bilerek hesaplamadı). Görülmüş girdiler H13 kartında tek tek listelendi.

## 5. Durma sayacı ve Tur-5 → K5 geçişi
- **Okuma kuralı (H13 kartı):** ACC ≥ 0,8997 veya H13 + H04 OOF dengeli ≥ 0,8190. Yalnız TUT alan H13 sayılır.
  - H13 TUT → **0/2**.
  - H13 ÖLDÜR veya İNCELE → **1/2**.
- **Sayaç Tur-5'te 2/2 olamaz.** Bu yüzden konu.md'deki durma kuralı K5'i **kendiliğinden tetiklemez**. Tur 5/8, birleşik defter ≈ 21,7/32.
- **K5'e geçmek bu nedenle KULLANICI KARARIdır.** K5 zaten kritik kapı olarak kullanıcıya sorulur. Öneri: **Tur-5'ten sonra K5'e geç.** Gerekçeler:
  - (i) K5 rezervinden sonra kalan ≈ 2,8 L4-sa'te kural içi, ACC hedefli bir GPU kartı yok (§3.3).
  - (ii) Z1–Z4 aileleri kapatıldı ya da K5'e bağlandı.
  - (iii) Makalenin açık birincil sorusu (ezber / lezyon-ayrık genelleme) ancak K5'in C0/C1 ayağında yanıtlanabilir.
- Kullanıcı Tur-6 isterse kart yazımı yeni id'yle (H14) ve K5 rezervi korunarak yapılır.

## 6. Koşum sırası (pratik; CPU)
1. **tur5-h13:** a1 duman (s0; tek üye özdeşliği + çift üye) → a3 (s0–4). Beklenti `kuyruk/tur5-h13.beklenti.json` koşumdan önce yazılır.
2. Katmanlar: `h13k_h13_logitayar`, `rt_h13_sicaklik`.
3. İkinciller (`h13x_*`; karar vermez): olasılık ortalaması, T2, T3, T4, M3, ablasyon E_hom2. S/N ve C-val betikleri.
4. Tahlil (K4c → `tahlil-tur5.md`) tek seferde. Ardından K5 önkoşulları (K5-ONKAYIT §0).

**Tahmini Tur-5 toplamı:** GPU 0, CPU < 0,2 sa, uygulama ≈ 0,3 sa. Birleşik defter değişmez (≈ 21,7/32, H12 dahil).

## 7. Revizyon notu — 2026-10-11 (Tur-5 ön-kaydı; hiçbir `h13_*` koşumundan önce)
- **Eklenenler:** Tur-4 kapanış tablosu, H13 kartı (T1), puanlama, ortak ön-kayıtlar, durma sayacı okuma kuralı ve "sayaç 2/2 olamaz → K5 kullanıcı kararı" notu, K5 ön-kaydına bağlantı.
- **Tur-5 ön notundan iki netleştirme** (test öncesi yapıldı; ikisi de sıkılaştırma):
  - (1) **T4 için "K5'te H08 geçmezse atılır" ifadesi "her durumda ek tabloda yazılır, iddia taşımaz" oldu.** Gerekçe: D3, sonuç ne olursa olsun yazılır.
  - (2) **T2/T3, karar vermeyen ikincil satırlar olarak sabitlendi.** Kazanan yalnız T1 olabilir.
- **Kaynaklar:** Gontijo-Lopes, Dauphin, Cubuk 2022 (ICLR, arXiv:2110.12899; özet erişimi 2026-10-11); Lakshminarayanan vd. 2017 (NeurIPS).
- DermaMNIST'te heterojen topluluk için kaynak bulunamadı (2 WebSearch); beklenti iç teşhis ve kuramla yazıldı.

---
**SIRALAMA.md'ye eklenecek bağlantı (şef; dosyanın en üstüne, "Durum" satırından önce):**
> **TUR-5 ve K5 (2026-10-11):** Tur-5 bölümü → `hipotezler/SIRALAMA-TUR5.md` (kart H13). K5 ön-kaydı (test kolları,
> H10/H08/H13 doğrulama kuralları, C0/C1/C2, karar çıkışları; KULLANICI ONAYI GEREKİR) → `hipotezler/K5-ONKAYIT.md`.
> Aşağıdaki "K5 ÖN-KAYDI — H08 TEST DOĞRULAMA KURALI" aynen geçerlidir ve K5-ONKAYIT §2.1'de değiştirilmeden alıntılanmıştır.
