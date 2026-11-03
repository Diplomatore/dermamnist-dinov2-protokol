# K5 C1 uygulama ve protokol notu: k5-c1-A a1 hataları ve k5-c1-A-v2 (2026-10-11 ~22:00, deney-yürütücü)

*Kapsam: yalnız uygulama düzeltmeleri.*
- Tarif, seçim kuralı, kollar, seed'ler ve beklentiler DEĞİŞMEDİ.
- K5-ONKAYIT.md / -b.md'ye ve donmuş `sonuclar/k5_dondurma.json`'a dokunulmadı.
- C-test açılmadı.
- C1 verisi (`data/dermamnist_c_resmipiksel_*.npz`, `kuyruk/k5_c_bolme_beklenen.json`) DEĞİŞMEDİ.

## Olay
`k5-c1-A` a1 (21:42–21:52), `durum/` ve `runs/` kayıtları:
- `c_bolme_veri_c1` rc=0 · `h02_c1_duman` rc=0 (val_acc 0,906).
- `h01_c1_duman` rc=1.
  - Hata: sklearn LogisticRegression, "only one class: 2".
  - Neden: `--alt-kume 256` "ilk N" örneği alır. C npz'si (yayımlanmış C npz ile aynı biçimde) HAM10000 metadata sırasıyla dizilidir, yani sınıfa göre gruplu (bkl önce). Bu yüzden ilk 256 train örneğinin hepsi sınıf 2'dir.
- `baseline_c1_duman` rc=1.
  - Hata: `medmnist/dataset.py:82` `assert info["n_samples"][split] == imgs.shape[0]`. C-train 8.215 ≠ INFO 7.007.
- Sonuç: kapı rc=2 → `KUYRUK-DURDU k5-c1-A asama=1` (21:52:09). a2 ve a3 başlamadı.

## Düzeltmeler (kodda; rakip kodu ve K4 betikleri dokunulmadı)
1. **`varyantlar/k5/c_rakip_kos.py`**
   - Süreç içinde `medmnist.INFO["dermamnist"]["n_samples"]` C npz dosyasından okunan bölme boyutlarına çevrilir: train 8.215 / val 573 / test 1.227.
   - Eski ve yeni değerler `{RUN_DIR}/c_bolme.json`'a yazılır.
   - Bu, MD5 alanıyla aynı türde bir **veri yolu uyarlamasıdır** ("rakip kodu dokunulmaz, yalnız veri yolu override", K5-ONKAYIT §4.2).
   - **Protokol notu:** INFO yalnız süreç içinde değişir. medmnist paketi ve rakip dosyaları değişmez.
   - Aynı düzeltme C2'deki `baseline_c2` için de geçerlidir. `k5-c2` henüz başlamadı ve satırları değişmedi.
2. **`varyantlar/k5/c_bolme.py`**
   - `ortak.veri.yukle` C verisinde, **yalnız `alt_kume` verildiğinde**, "ilk N" yerine **sınıf-tabakalı, sabit RNG'li** alt küme seçer.
     - RNG: `default_rng(20261011 + bölüm kodu)`.
     - Pay: sınıf payıyla orantılı (en büyük kalan yöntemi). Mevcut her sınıftan en az 1 örnek.
     - `idx` gerçek indekslerdir; `val_meta` hizası korunur.
   - `alt_kume` verilmezse özgün `yukle` AYNEN çağrılır. Tam (a2/a3) koşumlar bu yüzden bit düzeyinde etkilenmez.
3. **`varyantlar/k5/c_bolme_sar.py`**
   - C verisinde `ince_ayar_vit.py --alt-kume` reddedilir.
   - Neden: `ince_ayar_vit.etiket_yukle` önbellek varken "ilk N" etiketi okur ve tabakalı alt kümeyle hizasız kalırdı.
   - C kuyruklarında ViT için alt küme kullanılmıyor; bu yalnız koruyucudur.

## Sıra değişikliği GEREKMİYOR (C1 npz değişmedi)
- C1 npz sırası C CSV sırasıdır, yani yayımlanmış C npz sırası (data_manifest §DermaMNIST-C; `k5_c_denetim` `c_csv_sira_ayni`). Bu, ön-kayıtlı yapım kuralıdır (§4.1).
- Eğitimde sıra sonucu belirlemez:
  - R18/ViT: `torch.randperm(N, generator=seed)` + batch.
  - Sonda: `randperm` (seed).
  - Rakip: `DataLoader(shuffle=True)` + global seed.
- Batch'ler her durumda seed'le belirlenen permütasyondur. Farklı dosya sırası farklı ama eşit derecede seed'li bir permütasyon verir.
- Resmî npz ile "aynı permütasyon" zaten anlamsızdır, çünkü veri kümesi farklıdır (8.215 ≠ 7.007).
- Sıradan etkilenen tek şey "ilk N" alt küme dumanıydı. Düzeltme 2 bunu giderir.

## Etki alanı
- **k5-c1-B (H08 + H10, koşuyor):** etkilenmez.
  - Alt küme ve medmnist dataset kullanılmaz. `h08_c1_duman` rc=0 (val_acc 0,901).
  - Oturum kodu başlangıçta kopyaladığı için PAKET güncellemesi koşan sekmeyi değiştirmez.
  - Sekme yeniden açılırsa yeni kod gelir; B yolları için davranış aynıdır.
- **k5-c2:** satırlar aynı; `baseline_c2` düzeltme 1 ile çalışır. v2 gerekmez.
- **k5-c1-A-v2:** `k5-c1-A.jsonl` ve `.beklenti.json`'un BAYT BAYT kopyası. Satır ve beklenti SHA256'ları:
  - `3f13fc7c…c2ba`
  - `9a533229…3bf1`

  k5-c1 grubunda olduğu için izni `kuyruk/k5-c1.izin.json`'dan devralır. a1'deki veri kurulumu yeni oturumda yeniden yapılır (≈ 6 dk); içerik SHA256'sı donmuş değerle aynı olmalı.
- **Analiz:** `scripts/k5_test.py --kip c1` kuyruk öneki `k5-c1` ile yalnız aşama-3 koşumlarını okur. k5-c1-A'nın hatalı aşama-1 koşumları sayıma girmez.

## Yerel CPU mini doğrulama (GTX 1650 makinesi, CPU; scratchpad; ölçüm değil; C-test açılmadı)
- **Rakip veri yolu:** `c_rakip_kos.main()` gerçek yaması koşuldu. Rakip yerine resmî betiğin veri katmanı kuruldu:
  - Süreç içi `n_samples` {8215, 573, 1227}.
  - `DermaMNIST` len train 8.215, val 573.
  - `DataLoader(shuffle)` ilk batch 7 sınıf.
  - Resmî `models.ResNet18` ile 1 Adam adımı, kayıp 2,05. Yerel C1 npz MD5 `c7b115af…`.
  - Tam rakip epoch'u bilinçli olarak koşulmadı: rakip her epoch C-test'i değerlendirirdi ve C-test kol başına bir kez açılır.
- **Sonda dumanı:** `c_bolme_sar --veri c1 -- sonda.py --alt-kume 256 --C 1.0 --secim-seedleri 0,1` rc=0.
  - train sınıfları [9, 14, 29, 3, 32, 165, 4]; val [9, 10, 22, 2, 10, 200, 3].
  - val_acc 0,7305 (a1 aralığı [0,68; 0,96]); `ayni_seed_tahmin_maks_fark` 0; lbfgs çalıştı.
- `ince_ayar_vit --alt-kume` koruması sınandı (reddedildi).

## Düzeltme sonrası dosya SHA256
| dosya | SHA256 |
|---|---|
| `varyantlar/k5/c_bolme.py` | `6759e808…865e` |
| `varyantlar/k5/c_rakip_kos.py` | `a319fc30…bb1b` |
| `varyantlar/k5/c_bolme_sar.py` | `6f2bb61b…78f2` |

## §v3 — k5-c1-A-v2 a1 kapısı: beklenti kapsam hatası (2026-10-11 ~22:30)

### Olay
- k5-c1-A-v2 a1'de dört duman satırı da rc=0 bitti ve val_acc değerleri aralıktaydı (h01 0,7305 · h02 0,9058 · baseline 0,808).
- Kapı yine de iki İHLAL verdi: `h01_c1_duman/s0` için `n_train=256 ∉ [8215, 8215]` ve `val_n=256 ∉ [573, 573]` (`sonuclar/beklenti-k5-c1-A-v2-asama1.md`).

### Neden
- `ikincil_aralik` kuralı varyant filtresi taşımaz. a1'de n_train/val_n kuralları bütün satırlara uygulanır.
- H01 duman satırı ise tasarım gereği 256'lık alt kümeydi. Bu kc1 kalıbından gelir; kc1 beklentisinde n_train/val_n kuralı a1'de yoktu.
- Bu bir beklenti kapsam hatasıdır. Koşum ya da veri hatası değildir.

### Düzeltme
- Düzeltme satırda yapıldı; kurallar değişmedi, eşik gevşetilmedi.
- `k5-c1-A-v3`'te yalnız `h01_c1_duman` satırı değişti:
  - `--alt-kume 256` kaldırıldı; satır tam C-train/C-val ile koşar (C = 1, seçim seed'leri 0,1).
  - Zaman sınırı 15 → 20 dk.
- Diğer 34 satır v2 ile bayt-aynı.
- Beklenti kuralları v2 ile aynıdır; yalnız bir `_notlar` satırı eklendi.

### Süre tahmini (L4)
- Öznitelik çıkarımı (8.788 görüntü, ViT-B/14 fp32): ≈ 1–1,5 dk. Kaynak: kc1 H01c'nin ilk seed'i 3,9 dk sürdü; buna 40 sonda uyumu dahil.
- Duman satırının kendi uyumları: 2 sonda uyumu + aynı seed tekrarı + lbfgs ≈ 0,5–1 dk.
- Toplam ≈ 2–3 dk.
- Öznitelik önbelleği a2/a3 satırlarıyla paylaşıldığı için sonraki satırlarda zaman kazanılır.

### Etkisi olmayanlar
- Kodda değişiklik yok.
- Tabakalı alt küme yaması (§düzeltme 2) yerinde kalır; artık C kuyruklarında kullanılmıyor.

### Günlük arıza satırı önerisi (deney-gunlugu.md)
| 2026-10-11 ~22:2x | k5-c1-A-v2 a1 kapısı İHLAL | h01_c1_duman n_train=256 / val_n=256 ≠ 8215/573 | beklenti raporu | `ikincil_aralik` varyant filtresi taşımıyor; duman satırı alt kümeydi (beklenti kapsam hatası) | `k5-c1-A-v3`: h01 dumanı tam veriyle, kurallar aynı; yeni defter `3-1-colab-K5-C1-A3.ipynb` | yok (yalnız duman; a2/a3 başlamamıştı) |
