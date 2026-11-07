## E1 / H14 — revizyon ek kartı (2026-11-03; test sonrası keşifsel; KULLANICI ONAYI VERİLDİ)
*Yazım: 2026-11-03, hiçbir `e1-*` koşumundan önce. Tur kartı değildir; durma sayacına, tur kart sınırına ve K5 kararına girmez.*
- **Bağlam:** K5 kapandı (`sonuclar/karar.md` §12: A. GEÇTİ, A-kayıtlı, vurgu A2); makale teslim edildi. Dış eleştiri
  madde 1: D4 yalnız R18'e (H08c) eklendi, H10 augmentasyonsuz kaldı; C1'deki "H10 − H08c ayırt edilemez" (−0,96
  [−2,10; +0,18]) asimetrik bir kıyas. Kullanıcı yalnız **E1**'i onayladı: H10 + D4 (H08 kuralı), C1 protokolü (C-train
  eğitim, C-val seçim, C-test 1.227'de bir kez), "test sonrası keşifsel" etiketiyle.
- **Kart: H14** = `h10_c1` + eğitim-zamanı D4 (ebeveyne göre tek değişken). H10 kartı ablasyon (iv) ve yukarıdaki Tur-4
  §7 "H10/H11 + D4" maddesindeki birleşim koşulu sağlandı: H08 K5'te doğrulandı (+2,62), H10 Tur-4 kazananı. Birincil
  keşifsel kıyaslar H14 − H10 ve H14 − H08c (C-test ACC); ikinciller dengeli ACC ve mel duyarlılığı. Hepsi düzeltmesiz.
- **Puan (bilgi; tur puanlamasıyla aynı biçim):** E[Δ(H14 − H10)] +0,8 × P(GA alt > 0) ≈ 0,30 / ≈ 1,8 sa (1,6 L4-sa +
  0,2 uygulama) ≈ **0,13**. Asıl değer ACC kazancı değil, eleştiri madde 1'e simetrik bir yanıttır. Üç olası sonucun
  (H14 > H08c / ayırt edilemez / H14 < H08c) makale cümlesi kartta önceden yazıldı; her sonuç makaleye girer.
- **Kod yolu:** 8 görünümlü fp16 blok-8 önbelleği (C-train 8.215 × 8 görünüm ≈ 25,9 GB + C-val 0,23 GB; Colab yerel
  disk). Canlı ileri geçişten ≈ 0,8 L4-sa ucuzdur ve H10'un sayısal yolunu korur. a1 kontrol kolu (8 görünüm yolu, D4
  kapalı) `h10_c1_duman_s0`'ı bit bit üretmelidir (Durum A). Ortam sürüklenirse (Durum B) aynı ortamda H10r koşulur.
- **Bütçe:** ≈ 1,6 L4-sa (H10r ile ≈ 2,8); kart tavanı 5,0 (kullanıcı üst sınırı ≤ 10).
- **Değiştirmedikleri:** K5 kararı, Holm aileleri, G1/G2/G3/V/L, `k5_c1_sonuc.*`, `k5_test.KILIT` (resmî test açılmaz;
  C1 modelleri resmî test/val'de asla değerlendirilmez) ve makalenin birincil iddiaları. Sonuç ayrı alt bölümde verilir.
- **Kapsam dışı (yeni onay gerekir):** epoch uzatma, lr araması, ek seed, D2 ablasyonu, tam ince ayar (+ D4), C2, rakibe
  D4, H08r.
- Öldürülen/kapanan kart yok; bu girdi yeni bir tur açmaz. Sıradaki boş kart id'si: **H15**.
- **Not:** bu girdinin eklenmesi `SIRALAMA.md`'nin mtime'ını değiştirir. Önceki turların ön-kayıt damgaları için kart
  dosyaları, beklenti dosyaları (`kuyruk/*.beklenti.json` `_aciklama` alanları) ve deney günlüğü esas alınır.

> Not (şef, 2026-11-03): Bu girdi önce SIRALAMA.md sonuna eklenmişti. Ön-kayıt kanıtı olarak dosyanın bayt içeriği ve mtime değeri (2026-10-11 17:53:06) kullanıldığından SIRALAMA.md orijinal baytlarıyla geri yüklendi (SHA256 e017899a…). E1 girdisi bu dosyada yaşar.
