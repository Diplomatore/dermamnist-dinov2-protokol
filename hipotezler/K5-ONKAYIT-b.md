# K5 ÖN-KAYDI — EK b (netleştirmeler; K5-ONKAYIT.md üst metni DEĞİŞMEDİ)

*Yazım: 2026-10-11 ~19:00 (deney-yürütücü, ŞEF KARARI üzerine). **TEST AÇILMADAN yazıldı.***
- Bu anda hiçbir `k5-*` kuyruğu başlamamıştır; `runs/` altında k5 koşumu ve `sonuclar/k5_test.KILIT` yoktur.
- Resmî testte hiçbir kolun test tahmini üretilmemiş ve görülmemiştir (baseline_c'nin K3/kc1 sayıları dışında).
- Kesin damga bu dosyanın mtime'ıdır ve SHA256'sı `sonuclar/k5_dondurma.json` → `dosya_sha256` içindedir. İkisi de ilk `*_test` koşumundan önce olmalıdır.
- K5-ONKAYIT.md'ye yalnız sonuna 2 satırlık bağlantı eklendi.

## §1 T-SN1 / T-SN2 kuralının nesnesi (K5-ONKAYIT §2.6 netleştirmesi)
**Karar (ŞEF):** T-SN1 (H10 − H02c) ve T-SN2 (H08c − H02c) kuralları **HAM Δ_S − Δ_N** ile işletilir.
- Ham Δ_S − Δ_N, S ve N alt kümelerindeki seed-ortalamalı ACC farkları arasındaki farktır.
- GA, aynı görüntü bootstrap'ından (B = 2.000, RNG 0) alınır: `ort_S − ort_N` örneklem başına hesaplanır.
- Etiketler aynen kalır:
  - GA alt > 0 → "desteklendi" (T-SN1: "H10'un R18'e kazancı lezyon tekrarlı görüntülerde yoğun").
  - GA üst < 0 → tersi.
  - Aksi → "ayırt edilemedi". T-SN2'de GA 0'ı kapsıyorsa "dengeli dağılım ile uyumlu".

**Tabaka-standardize Δ_S − Δ_N** (sınıf × dx_type × kaynak, N-ağırlıklı) ve tabakalı S − N **ikincil satırdır** ve etiket vermez. Nokta değeri ve GA'sı raporlanır.

**Gerekçe:**
- §2.6'daki beklenen aralık "val +5,1" ifadesiyle verilmiştir. Bu değer val'deki **ham** farka karşılık gelir: ham Δ_S +6,26, ham Δ_N +1,13 → +5,13. Tabaka-std değer +3,92'dir.
- Kuralın nesnesi "GA(Δ_S − Δ_N)" olarak yazılmıştır.
- Metin, tanım cümlesinde iki ölçüyü birlikte andığı için belirsiz kalmıştı (K5-ONKOSUL-KONTROL.md açık madde 1).

**Görülen bilgi (M1):** iki ölçünün val değerleri bu netleştirmeden önce görüldü:
- ham +5,13 [+1,76; +8,93] → desteklendi;
- tabaka-std +3,92 [−2,25; +10,16] → ayırt edilemedi.

Seçim, metnin kendi beklenen değerinin hangi ölçüye ait olduğuna dayanır; test sonucuna bakılmadan yapılmıştır.

**Uygulama:** `scripts/k5_test.py` güncellendi. `SN.<kıyas>.kural` ham farktan üretilir; `tab_*` anahtarları ikincildir. Ardından:
- val kuru koşusu tekrarlandı: `sonuclar/k5_kuru_val_sonuc.md`;
- `k5_dondurma.py` yeniden koşuldu (yeni SHA256).

## §2 Protokol notu — rakibin C-test değerlendirmesi (K5-ONKAYIT §4.2, §5)
- Rakip kodu (MedMNIST/experiments@70b6b3a, dokunulmaz) her epoch sonunda val'i **ve test bölümünü** değerlendirir.
- C1/C2'de bu "test" bölümü C-test'tir (1.227).
- Rakip, test değerlerini **model seçiminde kullanmaz**. Seçim resmî kural olan val-AUC ile yapılır. Raporlanan C-test tahmini, val-AUC ile seçilmiş epoch'unkidir (`tahmin_test.npy`).
- Bu nedenle "C-test kol başına bir kez" ilkesi diğer kollarda katı biçimde sağlanır: `k5_cikarim.py` C-test'i eğitimden sonra, tek satırda açar.
- Rakip kolunda ilke protokol gereği "seçime katılmadan her epoch ölçülür" biçimindedir.
- Rakip koduna dokunmamak için bu davranış değiştirilmez (KARAR 9). Makalede yöntem notu olarak yazılır.
- Epoch başına C-test değerleri yalnız `gecmis.json`'da kalır. Hiçbir analizde kullanılmaz.

## §3 K5 sonrası adımlar (ŞEF KARARI)
1. Resmî test geçişi bittikten ve `k5_test.py --kip test` bir kez koşulduktan sonra şunlar **yerel arşive kopyalanır** (D5):
   - Drive `runs/` altındaki K5 koşumlarının test dosyaları: `tahmin_test*.npy`, `etiket_test.npy`, metrics `test_*`.
   - K4 dönemi baseline test dosyaları.
2. `scripts/colab_arsiv_kopya.py` K4'te adında "test" geçen dosyaları dışlıyordu (M4). K5 sonrası kopyalamada bu dışlama kaldırılır. Bunun için betiğe yeni bir bayrak eklenir (ör. `--test-dahil`) ve SHA256 manifest'e yazılır.
   - Bayrak ancak `sonuclar/k5_test.KILIT` var olduğunda çalışır. Bu adım test açıldıktan **sonra** yapılır.
3. Aynı kural C1/C2 C-test dosyaları için de geçerlidir: `k5-c1-*` / `k5-c2` bitip `k5_test.py --kip c1|c2` koşulduktan sonra arşive kopyalanır.
