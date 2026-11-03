# K5 ÖN KOŞUL KONTROL LİSTESİ (K5-ONKAYIT.md §0) — deney-yürütücü, 2026-10-11 ~18:52

Test kilidi **KAPALI**. Hiçbir test tahmini üretilmedi, açılmadı ve okunmadı. Yerelde yalnız metadata CSV'leri ve val dosyaları kullanıldı.

Damgalar:
- Ön-kayıt SHA256 `9dba23ce…` (tam değer `sonuclar/k5_dondurma.json` → `onkayit_sha256`).
- Dondurma SHA256 güncel değeri `sonuclar/k5_dondurma.json.sha256` dosyasında (EK b sonrası yeniden donduruldu; Drive kopyası `yz/3-1/sonuclar/`).

| # | Ön koşul | Durum | Kanıt |
|---|---|---|---|
| P1 | H12 koşuldu ve tahlil edildi; H11'in 3. maddesi kapandı; H12 C\* ve H06 τ\* belirlendi | **TAMAM** (not: kartın Sonuç bölümü boş) | Koşum: 5/5 rc=0 (`runs_colab`, 5. seed arşive 18:2x'te eklendi).<br>Tahlil: deney-gunlugu.md satır 24 (0,8187; H11 − H12 +6,1; 3. madde kapandı) ve orkestrasyon "Tur-5 KARAR".<br>Dondurma: C\* 0,1 (5 seed aynı), τ\* 0,2 (H06 protokolü), T seed başına.<br>Not: `hipotezler/H12.md` Sonuç bölümü **doldurulmamış** (şefe açık madde). |
| P2 | H13 kararı (TUT/ÖLDÜR/İNCELE) | **TAMAM: ÖLDÜR** | H13.md Sonuç bölümü, `sonuclar/tur5_h13_olcum.json`, orkestrasyon "Tur-5 KARAR" (+0,14 [−0,96; +1,24], p 0,369).<br>Sonuçları: H13 test'e gitmez, W = H10, M3 satırları yok, T4 yalnız ek tabloda.<br>Not: `sonuclar/tahlil-tur5.md` dosyası **yok** (karar günlük ve kartta kayıtlı). |
| P3 | Tur-5 → K5 geçişi kullanıcı kararıyla onaylandı | **TAMAM** | KULLANICI KARARI 10 (orkestrasyon_durumu.md, 18:14) |
| P4 | Donmuş parametre dosyası | **TAMAM** | `sonuclar/k5_dondurma.json` (18:51:22) + `.json.sha256`, Drive'a kopyalandı ve SHA256 eşleşti. Ayrıntı aşağıda. |
| P5 | Analiz betiği yazıldı ve val'de kuru koşuldu; val sayıları tahlil-tur4'ü yeniden üretiyor; SHA256 P4'te | **TAMAM** | Betik `scripts/k5_test.py` (ön-kayıttaki ad; görevdeki `k5_analiz.py` adının yerine bu kullanıldı).<br>Kuru koşu `sonuclar/k5_kuru_val_sonuc.{json,md}`: H10 − H02c **+2,91 [+1,34; +4,61]** = tahlil-tur4. Kuru koşu betik SHA256'sı = dondurmadaki değer. Ayrıntı aşağıda. |
| P6 | Yükleme özdeşliği: argmax 1.003/1.003 ve maks \|Δp\| ≤ 1e-5; H10'da fp16 blok-8 yolu aynen | **COLAB'A BAĞLI (a1 kapısı)**; yerel kanıt TAMAM | `k5-test` a1 aşaması = 35 satır özdeşlik (yalnız val). Kapı geçmezse a3 (test) başlamaz. a3'te özdeşlik bozulursa `k5_cikarim.py` testi açmadan rc=4 ile çıkar. Ayrıntı aşağıda. |
| P7 | C-test lezyon ayrıklığı, C-test ⊂ resmî test, eşleme 1.227/1.227 | **TAMAM** | `sonuclar/k5_c_denetim.json` (`scripts/k5_c_denetim.py`, yalnız CSV). Ayrıntı aşağıda. |
| P8 | Colab runs/ yerel arşivde (D5); bütçe sayacı güncel | **TAMAM** (sayaç güncellemesi şefte) | `runs_colab/` 57 koşum (18:2x artımlı kopya; H12 s4 tamamlandı; test adlı dosyalar hariç).<br>Damgalı Colab toplamı 56 koşum = 5,47 L4-sa. Oturum dahil ≈ 6,2 ve birleşik defter ≈ 21,7 (orkestrasyon). |

**P4 dondurma dosyasının içeriği:**
- Kollar ve run_id'ler: §1 ile birebir; seed {0..4}, rc=0, aşama 3.
- τ\* tam-val: baseline_c 0,2 · H01c 0,2 · H02c 0,7 · H08c 0,7 · H10 0,7 · H11 1,0 · H12 0,2. Ön-kayıttaki değerlerle aynı.
- T seed başına: H01c 1,19 ×5 (ön-kayıtla aynı).
- C\*: H01c 0,1, H12 0,1.
- Baseline'ın dondurulmuş test sayıları.
- `katlar.json` `0db03edb…e0193` ✓; RNG 0, B 2.000; Holm aileleri.
- SHA256'lar: `k5_test.py`, K5-ONKAYIT.md, `varyantlar/k5/*`, `c_bolme_npz.py`, k5-* kuyruk ve beklenti dosyaları, gruplar.json, kaynak `tahmin_val.npy`'ler.
- Drive `best_model.pth` / `sonda_agirlik.npz` / baseline `tahmin_test.npy` için yalnız boyut ve mtime (içerik okunmadı).
- PAKET sürümü.

**P5 kuru koşunun yeniden ürettiği diğer sayılar:**
- C-val: +1,05 [−0,66; +2,79], H08c − H02c +1,75 [+0,59; +3,04], H10 − H08c −0,70 [−2,41; +0,87] (tahlil_tur4_cval ile aynı).
- KÇ0 val +1,52 (ön-kayıt §3 ile aynı).
- Tabaka-std ΔS − ΔN = tahlil_tur4_ezber (+3,92).
- C1/C2 kod yolları vekil veriyle koşuldu.
- Sentetik (rastgele) uçtan uca sınama yapıldı (scratchpad; `--kip test|c1|c2`).
- Test kilidi ikinci açılışı reddediyor. `k5_test.KILIT` yalnız veriler yüklenip P6 geçince yazılıyor.

**P6 yerel kanıt (GTX 1650, ölçüm değil):** `varyantlar/k5/k5_cikarim.py`'nin dört yolu yerel kaynak koşumlarını **bit bit** yeniden üretti (maks \|Δp\| = 0, argmax 1003/1003):
- r18: H02 s0
- vit_kismi: önbellek yeniden hesaplandı
- vit_tam
- sonda: H01 s0

Baseline yolu (dosyadan ACC) da sınandı.

**P7 sonuçları:**
- lezyon(C-test) ∩ lezyon(resmî train ∪ val) = **0/1.227** (lezyon 0).
- C-test ⊂ resmî test **1.227/1.227**.
- (image_id, lesion_id) eşlemesi **1.227/1.227** (tümü 10.015/10.015).
- DermaMNIST-C.csv bölmesi ve sırası ile aynı.

## Çıktılar ve yollar
- Kuyruklar (beklentiler 18:42:55'te, satırlar 18:43'te yazıldı; hiçbiri başlamadı):

  | kuyruk | satır |
  |---|---|
  | `kuyruk/k5-test` | 70 |
  | `k5-c1-A` | 35 |
  | `k5-c1-B` | 27 |
  | `k5-c2` | 35 |

  `gruplar.json`'a k5-test, k5-c1 ve k5-c2 grupları eklendi; izin gereken: aynı üçü.
- Defterler: `<DRIVE_KOK>\3-1-colab-K5-test.ipynb`, `-K5-C1-A`, `-K5-C1-B`, `-K5-C2`. Kod hücreleri Tur-4 v2 ile aynı, hücre 8 düzeltmesi dahil.
- İzin dosyaları **yazılmadı**: `k5-test.izin.json` ŞEF yazar; C1/C2 izinleri test sonrası.
- Komutlar ve sıra: `varyantlar/KOMUTLAR.md` §K5.

## Açık maddeler (şef)
1. **ÇÖZÜLDÜ (ŞEF KARARI → `K5-ONKAYIT-b.md` §1): kural HAM Δ_S − Δ_N ile; tabaka-std ikincil. Val kuru koşusu: ham +5,13 [+1,76; +8,93] → desteklendi.** Eski not: T-SN1 kuralının hangi Δ_S − Δ_N ile uygulanacağı belirsizdi. Ön-kayıt §2.6, "beklenen [+1; +9] (val +5,1)" diyor.
   - +5,1, **ham** ΔS − ΔN'ye karşılık geliyor (+5,13 [+1,76; +8,93]).
   - **Tabaka-std** değer ise +3,92 [−2,25; +10,16].
   - İki seçenek aynı val verisinde farklı etiket veriyor ("desteklendi" ↔ "ayırt edilemedi").
   - Betik ikisini de hesaplayıp raporluyor; şu an etiket tabaka-std'den çıkıyor.
   - **Test öncesi `K5-ONKAYIT-b.md` ile netleştirilmeli.** Değişiklik `k5_test.py`'ye dokunursa `k5_dondurma.py` yeniden koşulur.
2. Ön-kayıtta P5'teki betik adı `scripts/k5_test.py`, görevde `k5_analiz.py`. Ön-kayıttaki ad kullanıldı.
3. **ÇÖZÜLDÜ (`K5-ONKAYIT-b.md` §2, protokol notu).** Rakip (baseline_c1/c2) resmî protokol gereği C-test'i her epoch değerlendiriyor ama seçimde kullanmıyor. "C-test kol başına bir kez" ilkesi diğer kollar için katı biçimde sağlanıyor. Rakip için bu bir protokol notu; kod dokunulmaz (KARAR 9).
4. Bütçe tahminleri ön-kayıttan yüksek:
   - Test geçişi ≈ 0,6–0,9 L4-sa (ön-kayıt ≈ 0,2). Neden: 70 satırın süreç açılışı ve kurulum.
   - C2 ≈ 3,8 (ön-kayıt ≤ 3,2).
   - C1 sonrası defter ≈ 28,0, §5 kapısı 28,8 → C2 sınırda.
5. **ÇÖZÜLDÜ (`K5-ONKAYIT-b.md` §3: K5 sonrası test dosyaları arşive kopyalanır).** `scripts/colab_arsiv_kopya.py` adında "test" geçen dosyaları dışlıyor (M4). K5 sonrası D5 arşivinde test dosyalarının kopyalanıp kopyalanmayacağı şef kararı.
6. H12.md Sonuç bölümü boş, `tahlil-tur5.md` yok. Bunlar belge eksikleri; K5'i bloklamıyor.
