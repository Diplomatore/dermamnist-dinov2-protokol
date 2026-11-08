## E2 / H15 — BCN20000 dış merkez, yeniden eğitimsiz, tek seferlik kilitli değerlendirme (2026-11-07)
*Yazım: 2026-11-07, hiçbir `e2_*` koşumundan önce. Bu bir tur kartı değildir. Durma sayacına, tur kartı sınırına ve
K5 kararına girmez. SIRALAMA.md ve SIRALAMA-E1.md'ye dokunulmadı.*

- **Bağlam:** KULLANICI KARARI 13 (2026-11-07). Soru: hangi iç protokol (resmî görüntü-düzeyi bölme mi, lezyon-ayrık C1
  mi) DINOv2-B − ResNet-18 farkını dış merkezde daha iyi öngörüyor? Hipotez iki yönlü kuruldu (Taori 2020).
- **Kart: H15.** Değişen tek şey değerlendirme verisidir: BCN20000 figshare train, eşleme E1 (AK→akiec, SCC dışlanır),
  kopyasız 11.933 görüntü / 3.451 lezyon, merkez kare 224 px. 11 arşiv kolu × 5 seed değerlendirilir, hiçbiri yeniden
  eğitilmez. Rakip 28 px kollarının ağırlık dosyası olmadığı için bu kollar dışlanır.
- **Birincil aile (Holm m = 2):**
  - B1: dış dengeli Δ (H10 − H02c).
  - B2: öngörü hatası farkı D = |e_R| − |e_C1|, önsel-sabit ACC ile.
  - Belirsizlik: seed × BCN-lezyon iki düzeyli eşleştirilmiş bootstrap (B = 10.000).
  - Önceden beyan: dengeli ACC'de iki protokol zaten aynı farkı öngörüyor (+6,03 ↔ +5,74). Protokoller yalnız ACC'de
    ayrışıyor (+2,26 ↔ +1,12). Bu yüzden B2'nin çözünürlüğü ≈ 1,1 puanla sınırlı ve en olası etiketi "ayırt edilemedi".
- **Puan (bilgi):** E[bilgi] makalenin A2 tezinin ("üstünlük protokole duyarlı") dış geçerliliğidir; puan yeni bir kazanç
  vaat etmez. Maliyet ≈ 1,5 L4-sa (tavan 3,0) + CPU veri hazırlığı ≈ 2,5 sa. Sonuç ne çıkarsa çıksın makaleye girer. Her
  etiket için cümle kartta önceden yazıldı.
- **Kullanıcı onayı gerekir (açık):**
  1. Rakibin dışlanması.
  2. B2 ölçütü (önsel-sabit ACC).
  3. Karar 13'ün E1 eşlemesi, kopyasız küme ve çerçeve duyarlılığını kapsadığının teyidi.
- Öldürülen ya da kapanan kart yok. Sıradaki boş kart id'si: **H16**.
