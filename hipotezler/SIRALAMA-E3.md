## E3 / H16 — BCN ön-işleme hattının B1'e katkısı, resmî val iç-veri kestirimi (2026-11-08)
*Yazım: 2026-11-08, hiçbir `e3_*` koşumundan önce. Bu bir tur kartı değildir. Durma sayacına, tur kartı sınırına ve
K5 kararına girmez. SIRALAMA.md, SIRALAMA-E1.md ve SIRALAMA-E2.md'ye dokunulmadı.*

- **Bağlam:** hakem E2 turu B3 (bloklayıcı) ve KULLANICI KARARI 15 (2026-11-08). Soru: B1'in (H10 − H02c BCN dengeli
  +16,34) bir bölümü merkez kaymasından değil, BCN ön-işleme hattından gelebilir mi?
- **Kart: H16.** Değişen tek şey resmî val girdisidir. MedMNIST+ 224 yerine aynı 1.003 HAM görüntüsünün ISIC özgün
  JPEG'i kullanılır ve BCN hattının aynı fonksiyonlarıyla hazırlanır. Kollar H02c, H08c, H10 ve H11'dir (5 seed); hiçbiri
  yeniden eğitilmez. C1 kolları H14 kural (i) gereği dışarıdadır.
- **Birincil nicelik:** Δ_hat = Δ_deng(H10 − H02c | hat) − Δ_deng(H10 − H02c | yerel). Belirsizlik E2 rutiniyle hesaplanır
  (seed × val lezyon kümesi, sınıf tabakalı, B = 10.000).
- **Karar kuralı (E2 donmuş sayılarından):** m = 2,48 (B1 − GA alt) ve G = 10,31 (B1 − iç +6,03). Etiketler: DIŞLANDI
  (GA ⊂ ±2,48) / KÜÇÜK-BELİRSİZ / KISMEN (2,48–10,31) / YORUM DEĞİŞİR (> 10,31) / TERS (≤ −2,48). Görülmemiş-lezyon
  (N) alt kümesi için tutucu kural uygulanır.
- **Puan (bilgi):** E[bilgi] hakemin bloklayıcı B3'ünü kapatmaktır. Sonuç ne çıkarsa çıksın makaleye girer ve her etiket
  için cümle önceden yazıldı. Beklenen Δ_hat [−1,5; +4,0], merkez +0,8. Olasılıklar: P(DIŞLANDI) ≈ 0,40,
  P(KÜÇÜK-BELİRSİZ) ≈ 0,40, P(KISMEN) ≈ 0,15. Maliyet ≈ 0,7 yerel GPU-sa (tavan 2,0) + ≈ 0,5 sa indirme.
  Bilgi/maliyet oranı yüksek: tek bir bloklayıcı, bir saatten kısa GPU süresiyle kapanıyor.
- **Kullanıcı onayı:** çekirdek kapsam Karar 15 ile verildi. Açık kalanlar: (1) duyarlılık kolları S5–S7; (2) q_BCN'nin BCN
  zip başlıklarından kestirilmesi. İkisi de birincil analizi bekletmez.
- Öldürülen ya da kapanan kart yok. Sıradaki boş kart id'si: **H17** (bileşen ayrıştırması gerekirse **H16b**).
