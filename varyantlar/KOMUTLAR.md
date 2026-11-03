# K5 komutları: resmî test geçişi, C0, C1, C2 (K5-ONKAYIT.md, KULLANICI KARARI 10) · deney-yürütücü 2026-10-11 ~18:50

*Hazırlık tamam; hiçbir K5 kuyruğu BAŞLAMADI.*
- **2026-10-11 ~22:00 C1-A a1 durdu** (rakip `n_samples` assert'i; sonda duman alt kümesi tek sınıf).
  - Düzeltme kodda: `c_rakip_kos.py`, `c_bolme.py`, `c_bolme_sar.py`.
  - Yeni kuyruk `k5-c1-A-v2` (bayt-aynı kopya) + defter `yz/3-1-colab-K5-C1-A2.ipynb`. Ayrıntı `hipotezler/K5-C1-UYGULAMA-NOTU.md`.
  - Önce eski K5-C1-A sekmesi kapatılır (en çok 2 oturum: B + A2).
  - İzleme: `<uv-python> scripts/colab_izle.py --kuyruklar k5-c1-A-v2,k5-c1-B --aralik 60 --asili-dk 20`.
- **EK b (test öncesi):** `hipotezler/K5-ONKAYIT-b.md` — T-SN1/T-SN2 kuralı ham Δ_S − Δ_N ile işletilir; rakip C-test protokol notu; K5 sonrası arşiv.
- Beklentiler 18:42:55'te, kuyruk satırları 18:43:03–18:43:14'te yazıldı. İkisi de her k5 koşumundan önce.
- Test kilidi KAPALI. Hiçbir test tahmini üretilmedi ya da okunmadı.
- Ön koşul listesi: `hipotezler/K5-ONKOSUL-KONTROL.md`.

## §K5.0 Hazırlık (yerel, CPU; KOŞULDU)
- **P7:** `env/.venv/Scripts/python.exe -I scripts/k5_c_denetim.py` → `sonuclar/k5_c_denetim.json` GEÇTİ.
  Yalnız metadata CSV okundu. C-test ∩ lezyon(resmî train ∪ val) = 0/1.227. C-test ⊂ resmî test 1.227/1.227. Eşleme 10.015/10.015.
- **P5 kuru koşu:** `env/.venv/Scripts/python.exe -I scripts/k5_test.py --kip kuru-val --runs runs_colab` → `sonuclar/k5_kuru_val_sonuc.{json,md}`.
  - H10 − H02c **+2,91 [+1,34; +4,61]** = tahlil-tur4.
  - C-val: H10 − H02c +1,05 [−0,66; +2,79], H08c − H02c +1,75 [+0,59; +3,04] = tahlil_tur4_cval.
  - KÇ0 val +1,52 = ön-kayıt §3.
- **P4 dondurma:** `env/.venv/Scripts/python.exe -I scripts/k5_dondurma.py` → `sonuclar/k5_dondurma.json` + `.json.sha256`. Drive kopyası: `yz/3-1/sonuclar/`.
  - τ*: baseline_c 0,2 · H01c 0,2 · H02c 0,7 · H08c 0,7 · H10 0,7 · H11 1,0 · H12 0,2.
  - T: seed başına. C*: H01c 0,1, H12 0,1. k5_test.py, ön-kayıt, kuyruk ve kod SHA256'ları da dosyada.
  - **k5_test.py, K5-ONKAYIT.md ya da k5-* kuyruk dosyası değişirse dondurma yeniden koşulur (yalnız test öncesi; K5-ONKAYIT-b ile).**
- **P6 yerel kanıt (ölçüm değil):** `varyantlar/k5/k5_cikarim.py` dört yolda yerel kaynakların `tahmin_val.npy`'sini bit bit yeniden üretti (README).
- Kuyruk yazımı `colab/hat/k5_kuyruk_yaz.py`.
- Defterler:
  - `colab/hat/defter_uret.py --k5 --sekmeler K5T=k5-test --dosya K5T=K5-test`
  - `colab/hat/defter_uret.py --k5 --sekmeler C1A=k5-c1-A,C1B=k5-c1-B --dosya C1A=K5-C1-A,C1B=K5-C1-B`
  - `colab/hat/defter_uret.py --k5 --sekmeler C2=k5-c2 --dosya C2=K5-C2`
  - Kod hücreleri v2 ile aynı; hücre 8'de 90 sn bekleme var.
- Paket: `colab/hat/drive_paketle.py`. Sürüm `sonuclar/k5_dondurma.json` → `paket_surum` alanında (KOMUTLAR paketin içinde olduğundan burada yazılmaz).

## §K5.1 Kuyruklar (en çok 2 eşzamanlı Colab oturumu)
| kuyruk | defter (yz/) | satır | içerik | izin (ŞEF yazar) | ne zaman |
|---|---|---|---|---|---|
| `k5-test` | `3-1-colab-K5-test.ipynb` | 70 | a1: 7 kol × 5 seed P6 özdeşliği (yalnız val) → kapı → a3: aynı 35 satır `--test-kilidi-acik` (`<kaynak>-test`) | `kuyruk/k5-test.izin.json` | şimdi (şef izniyle) |
| `k5-c1-A` | `3-1-colab-K5-C1-A.ipynb` | 35 | C1 verisi → baseline_c1, h02_c1, h01_c1 (a1 → a2 → a3) + C-test satırları | `kuyruk/k5-c1.izin.json` | test sonucu SHA256 ile donduktan SONRA |
| `k5-c1-B` | `3-1-colab-K5-C1-B.ipynb` | 27 | C1 verisi → h08_c1, h10_c1 + C-test satırları | aynı izin | A ile paralel |
| `k5-c2` | `3-1-colab-K5-C2.ipynb` | 35 | C2 verisi (Zenodo) → baseline_c2, h08_c2, h10_c2 + C-test | `kuyruk/k5-c2.izin.json` | C1 bitip §5 kapısı (defter ≤ 28,8) geçince; tek sekme |

- Çıkarım: `python colab/fp32_kos.py varyantlar/k5/k5_cikarim.py --tur {rakip|r18|vit_kismi|vit_tam|sonda} --kaynak <run_id> --seed s [--test-kilidi-acik] --out {RUN_DIR}/metrics.json`
  - baseline_c'de yeniden çıkarım yoktur; dondurulmuş `tahmin_test.npy` kopyalanır, ACC ±1e-4 denetlenir.
  - H01c/H12 test öznitelikleri `ozn_cikar.cikar(..., ("test",), test_kilidi_acik=True)` ile çıkarılır.
- C1/C2 eğitimi: `python colab/fp32_kos.py varyantlar/k5/c_bolme_sar.py --veri c1|c2 -- <K4 betiği + aynı argümanlar>`
  - Rakip için: `varyantlar/k5/c_rakip_kos.py --veri c1|c2 -- --seed s --epoch E`.
  - C-test satırı: `k5_cikarim.py --kaynak-varyant <kol>_c1 --kaynak-kuyruk k5-c1 --veri c1 --test-kilidi-acik`.
- İlk C1 satırı `scripts/c_bolme_npz.py --kip c1` şunları yapar:
  - DermaMNIST-C indirmesini bekler.
  - `data/dermamnist_c_resmipiksel_{28,224}.npz` dosyalarını kurar.
  - Denetimleri yapar: sayılar, C etiketi, lezyon 0/0/0, C-test bayt-özdeş.
  - İçerik SHA256'sını `kuyruk/k5_c_bolme_beklenen.json` ile karşılaştırır.

## §K5.2 Sıra (ön-kayıt §8) — şef adımları
1. `kuyruk/k5-test.izin.json`'u Drive'a yaz. Kullanıcı tek sekmede K5-test defterini açar.
   İzleme: `<uv-python> scripts/colab_izle.py --kuyruklar k5-test --aralik 60 --asili-dk 20`.
2. Kuyruk bitince:
   - `kosum.py --kok "<DRIVE_KOK>/3-1" durum` çalıştır: 70 satır `BİTTİ rc=0` olmalı.
   - `sonuclar/beklenti-k5-test-asama{1,3}.md` raporlarına bak.
   - Ardından **BİR KEZ** koş: `env/.venv/Scripts/python.exe -I scripts/k5_test.py --kip test --runs "<DRIVE_KOK>/3-1/runs" --test-kilidi-acik`.
     - Betik P6'yı (a1 ↔ a3 SHA, ozdeslik_gecti) denetler ve `sonuclar/k5_test.KILIT` dosyasını yazar.
     - Çıktı: `sonuclar/k5_test_sonuc.{json,md}`.
   - `sha256sum sonuclar/k5_test_sonuc.json > sonuclar/k5_test_sonuc.json.sha256` ile sonucu dondur (§8 adım 4).
3. C1 verisini dondur (yerel, test SONRASI): `env/.venv/Scripts/python.exe -I scripts/c_bolme_npz.py --kip c1 --ilk-yapim --out <geçici>/metrics.json`.
   - Bu komut `kuyruk/k5_c_bolme_beklenen.json` dosyasını yazar. Dosyayı `<DRIVE_KOK>/3-1/kuyruk/` altına kopyala.
   - SHA256'yı `k5_dondurma` ekine yaz (§4.1).
4. `kuyruk/k5-c1.izin.json`'u yaz. Kullanıcı K5-C1-A ve K5-C1-B defterlerini iki sekmede açar.
   İzleme: `colab_izle.py --kuyruklar k5-c1-A,k5-c1-B`.
5. C1 bitince:
   - `k5_test.py --kip c1 --runs … --test-kilidi-acik` çalıştır.
   - Birleşik defteri hesapla: ≤ 28,8 ise `kuyruk/k5-c2.izin.json` yaz ve K5-C2 defterini tek sekmede açtır, sonra `k5_test.py --kip c2`. Aksi halde C2 koşulmaz ve sınırlılık olarak yazılır.
6. **K5 sonrası arşiv (ŞEF KARARI, K5-ONKAYIT-b §3):** `k5_test.py --kip test` koşulduktan sonra K5 koşumlarının test dosyaları yerel arşive kopyalanır.
   - Kapsam: `tahmin_test*.npy`, `etiket_test.npy` ve baseline'ın K4 test dosyaları.
   - Yöntem: `colab_arsiv_kopya.py`'ye `sonuclar/k5_test.KILIT` şartlı `--test-dahil` bayrağı eklenir; SHA256 manifest'e yazılır.
   - C1/C2 C-test dosyaları için aynı kural geçerlidir.
7. `karar.md` (deney-tahlilci, §7).

## §K5.3 Süre / bütçe tahmini (L4; Colab K4 damgalı süreleri × C-train/val oranı 8.215/7.007; ölçüm değil)
| iş | duvar | L4-sa (oturum) | dayanak |
|---|---|---|---|
| k5-test (1 sekme) | ≈ 50–60 dk | ≈ 0,6–0,9 | 70 satır × ≈ 20–40 s; öznitelik B ≈ 1,5 dk, S ≈ 0,7 dk; H10 token yeniden hesabı ≈ 15 s/satır; kurulum ≈ 15 dk |
| C1 sekme A (baseline + H02 + H01) | ≈ 2,9 sa | ≈ 2,9 | a3: baseline 5 × 14,4 + H02 5 × 8,6 + H01 5 × 3,3 dk; a1/a2 ≈ 20 dk; C verisi ≈ 5 dk; kurulum 15 dk |
| C1 sekme B (H10 + H08) | ≈ 2,7 sa | ≈ 2,7 | a3: H10 5 × 14 + H08 5 × 8,8 dk; a1/a2 ≈ 20 dk; C-test ≈ 8 dk; kurulum 15 dk |
| C2 (1 sekme) | ≈ 3,8 sa | ≈ 3,8 | baseline 5 × 14,4 + H10 5 × 14 + H08 5 × 8,8 dk + a1/a2 ≈ 30 dk + kurulum |

Birleşik defter: ≈ 21,7 (yerel 15,3 + Colab ≈ 6,2 + H12) → test sonrası ≈ 22,4 → C1 sonrası ≈ 28,0 (§5 kapısı 28,8; pay ≈ 0,8).
Bu yol C2 sonrası ≈ 31,8 / 32 eder. C2 tahmini (≈ 3,8) ön-kayıttaki ≈ 3,2'nin üstündedir. Fark a1/a2 aşamaları ve kurulumdan gelir.

---

# Tur 5 komutları — H13 topluluk (CPU, foreground, KOŞULDU 2026-10-11 18:03–18:12)

*deney-yürütücü · kart H13 mtime 17:52:31, SIRALAMA-TUR5 17:51:12. Beklentiler `kuyruk/tur5-h13{,k,rt,x}.beklenti.json`
18:03:23'te, ilk Tur-5 koşumundan (18:03:59) önce yazıldı. Kuyruk adı kart ve SIRALAMA-TUR5 §6'daki gibi `tur5-h13`.*

## §T5
- **Arşiv (K5 öncesi D5):** `env/.venv/Scripts/python.exe -I scripts/colab_arsiv_kopya.py` → `runs_colab/`.
  Drive `runs/` kaynağından yalnız json/npy/log ve `.done`/`.heartbeat` kopyalanır; `.pth`/`.npz`/csv/txt/tfevents ve adında `test` geçen dosyalar HARİÇ.
  SHA256 manifest'i `runs_colab/ARSIV_MANIFEST.json`'da.
- **Koşum:** `bash varyantlar/topluluk/tur5_kos.sh a1|a3|katman|ikincil`.
  - Her satır `kosum.py --zaman-siniri-dk 5` ile foreground, `CUDA_VISIBLE_DEVICES=`, ≈ 5 s.
  - `--gpu-zorunlu` ve `--cpu-izin` YOK: topluluk.py torch kullanmıyor.
  | kuyruk | aşama | varyant | ebeveyn |
  |---|---|---|---|
  | tur5-h13 | 1 / 3 | `h13_h02c_h10_lpo_duman` (s0) / `h13_h02c_h10_lpo` (s0–4) | aynı seed H10 run id'si |
  | tur5-h13k | 3 | `h13k_h13_logitayar` (s0–4) | aynı seed H10 |
  | tur5-h13rt | 3 | `rt_h13_sicaklik` (s0–4) | aynı seed H10 |
  | tur5-h13x | 3 | `h13x_{h02c_h10_op,t2_h01c,t4_h08c_h10_lpo,abl_h10_hom2}` (s0–4; ebeveyn H10_s), `h13x_m3_baseline_h02c_lpo` (s0–4; ebeveyn H02c_s), `h13x_{t3_h10_hom5,t3_h02c_hom5,m3_baseline_hom5}` (seed `0-4`, n = 1) | — |
- **Kapılar:** `beklenti_kontrol.py --kok . --kuyruk <q> --asama 1|3`. Hepsi GEÇTİ.
- **Ölçüm:** `env/.venv/Scripts/python.exe -I scripts/tur5_h13_olcum.py` → `sonuclar/tur5_h13_olcum.json`.
  - Tanımlar tahlil_tur4 / _ezber / _cval ile aynı. O betikler değiştirilmedi: KOL listeleri sabit ve çıktı yolları Tur-4 dosyalarını ezerdi.
- **Leaderboard Tur-5** (baseline H10): scratch kökte Colab H10/H02c/H08c + yerel H13 kollarının `config.json`/`.done` kopyalarıyla üretildi.
  `leaderboard.py --baseline h10_dinov2b_k4_ft224` → `sonuclar/leaderboard-tur5{,-dengeli_acc,-ece}.md`.
  Yerel leaderboard'lar venv ile yeniden üretildi (dengeli → ece → val_acc).

---

# Tur 2 komutları — H04 / H06 / R-T (CPU, foreground, KOŞULDU) · H05 (GPU, kuyruk `tur2-h05`, BAŞLATILMADI)

*deney-yürütücü · 2026-10-11 01:26 · kartlar ön-kayıtlı (H04/H05/H06 mtime 00:56–00:58, SIRALAMA 01:01). Beklentiler
`kuyruk/tur2-{h04,h04x,h06,rt,h05}.beklenti.json` 01:14'te, ilk Tur-2 koşumundan (01:14:17) önce yazıldı.*

## §T2
- **Katlar (bir kez, koşumdan önce, 01:08):** `env/.venv/Scripts/python.exe varyantlar/logit_ayari/logit_ayari.py --katlar-yaz`
  → `varyantlar/logit_ayari/katlar.json` (StratifiedGroupKFold lesion_id, RNG 20261011, SHA256 0db03edb…e0193).
- **CPU kolları** (`CUDA_VISIBLE_DEVICES=` ile; her satır 4–10 s): `kosum.py --kok . --varyant V --seed S --hipotez H --ebeveyn <aynı seed ebeveyn run>
  --kuyruk K --asama A -- <venv python> varyantlar/logit_ayari/{logit_ayari,sicaklik}.py …`
  | kuyruk (config) | aşama | varyant | ebeveyn | betik argümanları |
  |---|---|---|---|---|
  | tur2-h04 | 1 | h04_h02_logitayar_duman (s0) | H02 s0 | `--ebeveyn-varyant h02_r18in_ft224 --tau-izgara 0,1 --alt-bolme 0` |
  | tur2-h04 | 3 | h04_h02_logitayar (s0–4) | H02 aynı seed | `--ebeveyn-varyant h02_r18in_ft224 --ebeveyn-dosya tahmin_val.npy --tau-izgara 0:1:0.1 --kat 5 --kat-rng 20261011 --kisit 0.010` |
  | tur2-h04x | 3 | h04x_h02aucsecim_logitayar (s0–4) | H02 aynı seed | aynı, `--ebeveyn-dosya tahmin_val_aucsecim.npy` |
  | tur2-h04x | 3 | h04x_baseline_logitayar (s0–4) | baseline aynı seed | aynı, `--ebeveyn-varyant baseline` |
  | tur2-h06 | 1 / 3 | h06_h01_logitayar_duman (s0) / h06_h01_logitayar (s0–4) | H01 aynı seed | `--ebeveyn-varyant h01_dinov2b_lp_r224` (duman: `--tau-izgara 0,1 --alt-bolme 0`) |
  | tur2-rt | 3 | rt_{h02,h01,baseline,h04}_sicaklik (s0–4) | kaynak aynı seed | `sicaklik.py --kaynak-varyant {h02_r18in_ft224,h01_dinov2b_lp_r224,baseline,h04_h02_logitayar}` |
  Yan kollar ayrı kuyruk adı (`tur2-h04x`) aldı: `ikincil_aralik` varyant filtresi taşımadığı için baseline kolu H04 aralıklarına takılırdı.
- **H05 GPU dumanı (foreground, kuyruk etiketi `tur2-h05-on`, a1 kapısı `--beklenti kuyruk/tur2-h05.beklenti.json` ile GEÇTİ):**
  `20261011-012209_h05_r18in_ft224_cw00_duman_s0` (β = 0, `--ozdeslik-ref runs/20261010-204010_h02_r18in_ft224_duman_s0/metrics.json`),
  `20261011-012324_h05_r18in_ft224_cw05_duman_s0` (β = 0,5).
- **H05 kuyruğu `kuyruk/tur2-h05.jsonl` (9 satır; ŞEF başlatır: `kuyruk.py --kok . kos tur2-h05 --kademeli`):** a1 cw00_duman + cw05_duman (s0, 1 ep, 8 dk sınır);
  a2 cw05_onucus s0/s1 (5 ep, 15 dk; ebeveyn H02 onucus); a3 cw05 s0–4 (30 ep, 40 dk; ebeveyn aynı seed H02). Hepsi `--gpu-zorunlu`,
  `--kural KURAL_FT` (§3). Tahmini süre ≈ 2,2 sa (a1 2,2 dk + a2 9 dk + a3 5 × ≈ 24,7 dk).
- Ölçüm: `env/.venv/Scripts/python.exe varyantlar/logit_ayari/tur2_olcum.py` → `sonuclar/tur2_cpu_olcum.json`; leaderboard'lar
  venv (scipy) ile yeniden üretildi (`leaderboard.md`, `leaderboard-dengeli_acc.md`, yeni `leaderboard-ece.md`).

---

# Tur 1 komutları — H01 / H02 / H03 (kartlar KULLANICI ONAYLI 2026-10-10 20:15)

*deney-yürütücü · 2026-10-10 20:45 · **GEÇERLİ KUYRUK: `tur1` (§T1, 34 satır, BAŞLATILMADI — şef çift kanalla başlatır).** Aşağıdaki §1–§3
(`tur1-h01`/`tur1-h03`/`tur1-h02` ayrı kuyruk taslakları) ve §5 beklenti taslağı TARİHSEL kayıttır; yerlerini §T1 ve
`kuyruk/tur1.beklenti.json` aldı.*

## §T0 Kart metrik adı ↔ kod anahtarı eşlemesi (kart üstü değişmez; KOD adı geçerlidir)
| kart(lar)daki ad | `metrics.json` anahtarı | not |
|---|---|---|
| `tahmin_nv_payi` | `val_tahmin_nv_payi` | argmax tahminlerinin nv payı (B 0,672) |
| `val_sizinti_farki_tabakali` (tabakalı S − N) | `val_sizinti_tabakali_SN` (+ `val_sizinti_tabakali_kapsam`) | kesir (0,028 = +2,8 puan); `*_ham` anahtarları raporlanmaz |
| `val_acc_molemax_disi` | `val_acc_molemax_disi` | aynı |
| `val_auc_sinif_{4,2}` / "mel ve bkl OvR AUC" | `val_auc_mel`, `val_auc_bkl` (tümü: `val_auc_{akiec,bcc,bkl,df,mel,nv,vasc}`) | baseline 5 koşumunda `hata_kirilim_ozet.json` ile fark 0 (mel 0,8743, bkl 0,8815) |
| `val_acc_lbfgs` | `val_acc_lbfgs` | lbfgs basamak başına bir kez: a3'te yalnız s0 satırı |
| yakınsama ölçütü \|SGD − lbfgs\| (öldürme e) | `yakinsama_fark` (mutlak; + işaretli `lbfgs_sgd_fark`) | SGD tarafı = seçim seed'leri ortalaması @C* (a3 s0: 5 seed) |
| `seed_tahmin_maks_fark` | `seed_tahmin_maks_fark` (+ `seed_tahmin_en_kucuk_fark`, `seed_tahmin_n_karsilastirma`) | TEK KOŞUM İÇİNDE: koşum C*'da bütün seçim seed'lerini zaten uydurur → kendi seed'i ile diğerleri arası max\|Δp\|. Koşumlar arası çapraz denetim: `eslestirilmis_bootstrap.py` çıktısında `seed_tahmin_fark_b` |
| aynı seed tekrarında bit-bit aynı (R5) | `ayni_seed_tahmin_maks_fark` | C*'da kendi seed'i ikinci kez uydurulur (≈ 5 s); GPU dumanında 0,0 |
| `val_acc_aucsecim` (+ "vb.") | `val_acc_aucsecim`, `en_iyi_epoch_aucsecim`, `val_{auc,dengeli_acc,makro_f1,ece,tahmin_nv_payi,acc_molemax_disi,sizinti_tabakali_SN,auc_mel,auc_bkl}_aucsecim` | H02; olasılık `tahmin_val_aucsecim.npy` |
| ödünleşim kuralı Δnv payı | `val_tahmin_nv_payi` | — |
| seçim iyimserliği (C\* − medyan C) | `val_acc_medyanC_farki` | — |
| süre / param / bellek | `egitim_sure_sn`, `param_sayisi` (5.383 / 11.180.103), `param_omurga` (85.724.928), `tepe_gpu_bellek_mb`, `epoch_sure_ort_sn` (H02) | — |

Bootstrap (koşum sonrası, CPU):
```bash
"$PY" varyantlar/ortak/eslestirilmis_bootstrap.py --a-varyant baseline --b-varyant h01_dinov2b_lp_r224 --cikti sonuclar/bs_h01_vs_baseline.json
"$PY" varyantlar/ortak/eslestirilmis_bootstrap.py --a-varyant baseline --b-varyant h02_r18in_ft224 --cikti sonuclar/bs_h02_vs_baseline.json
"$PY" varyantlar/ortak/eslestirilmis_bootstrap.py --a-varyant baseline --b-varyant h02_r18in_ft224 --b-dosya tahmin_val_aucsecim.npy --cikti sonuclar/bs_h02aucsecim_vs_baseline.json
"$PY" varyantlar/ortak/eslestirilmis_bootstrap.py --a-varyant h03_dinov2b_lp_r28 --b-varyant h01_dinov2b_lp_r224 --cikti sonuclar/bs_h01_vs_h03r28.json   # T2
"$PY" varyantlar/ortak/eslestirilmis_bootstrap.py --a-varyant baseline --b-varyant h03_dinov2b_lp_r28 --cikti sonuclar/bs_h03r28_vs_baseline.json      # T3
```

## §T1 Kuyruk `tur1` (yazıldı 2026-10-10 ~20:45; BAŞLATILMADI)
Beklenti: `kuyruk/tur1.beklenti.json` (koşumdan önce). Başlatma (şef, K2 çift kanal):
`"$PY" $SK/kuyruk.py --kok $WS kos tur1 --kademeli --paralel 1` + `izle tur1`.
| aşama | satırlar | içerik | tahmini süre |
|---|---|---|---|
| a1 duman | 3 | H03 r28 ve H01 r224 sonda (256 + 256, C = 1, seçim s0,s1; `_n256` önbelleği çıkarılır), H02 1 epoch; seed 0 | ≈ 3,5 dk |
| a2 ön-uçuş | 6 | H03 r28 s0,s1 · H01 r224 s0,s1 (8 C, C* = ort(s0,s1), lbfgs) · H02 5 epoch s0,s1 | ≈ 18 dk |
| a3 tam | 25 | H03 r28/r64/r128 × s0–4 → H01 r224 × s0–4 (C* = ort(s0..s4); lbfgs yalnız s0, kart "bir kez") → H02 30 ep × s0–4 | ≈ 3,4 sa |
Ebeveyn: H01/H02 a3 = aynı seed'in baseline run_id'si (M5); H03 a3 = `h01_dinov2b_lp_r224` (kart; T3 baseline'a karşı tahlilcide);
a1/a2 = `baseline`. Canlı kurallar: sonda `{"nan_kes","ilerleme_zorunlu_sn":300,"sabit_sn":300,"min_adim":8,"alt_sinir":{"val_acc":0.69}}`,
H02 aynı + `min_adim 3`, `alt_sinir val_acc_eniyi 0.69` (kart c). Zaman sınırı: sonda a1 10 / a2 15 / a3 20 dk; H02 a1 8 / a2 15 / a3 40 dk.
r64/r128 a2'de yok: önbellek meta (girdi SHA256, model, r), etiketlerin 28 px ile bayt eşitliği, şekil, sonluluk ve
basamaklar arası öznitelik farkı (28↔64 maks|Δ| 9,9; 64↔128 6,0; 128↔224 3,9) CPU'da val'e bakmadan doğrulandı (20:38).

## §T2 Yerel GPU dumanı (2026-10-10 20:19–20:30; `runs_yerel_duman/runs/*gpu_duman*`; ölçüm DEĞİL)
| koşum | rc | süre | bulgu |
|---|---|---|---|
| `gpu_duman_h03_r28` s0 (256+256, C=1, seçim s0,s1) | 0 | 33,6 s | çıkarım 24,4 img/s, determinizm 0; s0/s1 val_acc ikisi de 0,6445 (kartın `farkli_seedler_farkli` yasağının gerekçesi), `seed_tahmin_maks_fark` 0,135, `ayni_seed_tahmin_maks_fark` 0,0; test_ anahtarı yok |
| `gpu_duman_h01_r224` s0 (aynı) | 0 | 50,9 s | val_acc 0,652; seed farkı 0,139; aynı seed 0,0. **Alt kümede SGD yakınsamamış:** 100 adım, amaç 0,138 / lbfgs 0,051, val farkı 12,9 puan → `yakinsama_fark` kuralı a1'de yok, a2'de tam veride (2.800 adım) sınanır |
| `gpu_duman_h02_1ep` s0 | 0 | 253 s | **epoch 232 s** (kart 43 s; öldürme d eşiği 90 s) → teşhis aşağıda |
| `gpu_duman_h02_1ep_v2` s0 (düzeltme sonrası) | 0 | 71 s | **epoch 48,7 s**, val_acc 0,812, `ilk_adim_loss` 1,85 (≈ ln 7), param 11.180.103, tepe bellek 1,74 GB, `tahmin_val_aucsecim.npy` ✓ |
**H02 yavaşlık teşhisi (sentetik mikro ölçüm, scratchpad):** GPU'da hazır NCHW girdiyle det=True 414 ms/adım (det=False 318);
CPU uint8 → GPU → `permute(0,3,1,2)` girdiyle 2.117 ms/adım. Neden: permute sonrası tensör NHWC adımlı kalıyor, deterministik
cuDNN bütün ağı yavaş channels-last yoluna sokuyor. Düzeltme `ince_ayar.py` `hazirla()`: `.permute(...).contiguous()` → 424 ms/adım.
Model/veri/tarif/seed aynı; yalnız bellek düzeni (hiçbir ölçüm koşumu düzeltmeden önce yapılmadı).
Tam veri sonda uyumu (sahte val ile, sonuç bilgisi yok): ≈ 5 s/uyum (C'den bağımsız); lbfgs C*=1: 5 s, C*=1000: 37 s.

## §T3 Tahmini süre (yerel ölçümden)
a1 ≈ 3,5 dk · a2 ≈ 18 dk (sonda satırı ≈ 2–2,5 dk, H02 5 ep ≈ 4,5 dk) · a3: sonda 20 satır × ≈ 3,5–4,3 dk ≈ 1,3 sa + H02 5 × (30 × 48,7 s + 25 s)
≈ 2,1 sa → **toplam ≈ 3,7 sa GPU** (SIRALAMA üst sınırı 3,7 sa; kart bütçeleri: H01 ≈ 0,45 sa ≤ 0,4–0,5; H03 ≈ 1,05 sa ≤ 1,1 üst;
H02 ≈ 2,25 sa ≤ 2,5 üst). Risk: GPU 85 °C'de termal yavaşlama bayrağı (0x20) görüldü; epoch > 90 s olursa kural (d) yakalar.

---

## (TARİHSEL) Onay öncesi taslak — Tur 1 komutları — H01 / H02 / H03 (TASLAK: kartlar onaylanınca geçerli)

*deney-yürütücü · 2026-10-10 (K3.5 sonrası güncelleme 20:10: sonda varsayılan lr 1e-2, yeni val metrikleri, eşleştirilmiş bootstrap) · kartlar henüz ONAYLI DEĞİL. Bu dosyadaki kuyruk satırları YAZILMADI, koşum BAŞLATILMADI.*
*Onaydan sonra şef önce `kuyruk/<ad>.beklenti.json` dosyasını yazdırır (§5 taslağı), sonra satırları ekler ve kuyruğu çift kanalla başlatır (K1).*

Donanım: yerel GTX 1650 4 GB, FP32 (`env/hiz-olcumu.md`). Hepsi `--gpu-zorunlu`, kuyruk `--paralel 1`.
**K3 kuyruğu (`k3-baseline-yerel`) bitmeden GPU'ya iş verilmez.**

```bash
WS=<WS>
PY='<WS>\env\.venv\Scripts\python.exe'   # deney venv (MUTLAK)
UVPY=<PYTHON311>                           # orkestra betikleri
SK=<WS>/tools
export PYTHONIOENCODING=utf-8
EKLE="$UVPY $SK/kuyruk.py --kok $WS ekle"
```
Komutlar kuyruğun cwd'si olan WS kökünden göreli betik yolu kullanır (`varyantlar/...`). `{RUN_DIR}` kosum.py doldurur.

## Varyant adları (leaderboard/replikasyon betikleri varyant adına bakar → aşama adları AYRI)
| kart | aşama 1 (duman) | aşama 2 (ön-uçuş) | aşama 3 (tam; leaderboard'a giren) | ebeveyn |
|---|---|---|---|---|
| H01 | `h01_dinov2b_lp_r224_duman` | `h01_dinov2b_lp_r224_onucus` | `h01_dinov2b_lp_r224` | `baseline` s0–4 run_id'leri |
| H03 | `h03_dinov2b_lp_r28_duman` | `h03_dinov2b_lp_r28_onucus` (r128 için `_r128_onucus`) | `h03_dinov2b_lp_r28`, `_r64`, `_r128` | `h01_dinov2b_lp_r224` run_id'leri |
| H02 | `h02_r18in_ft224_duman` | `h02_r18in_ft224_onucus` | `h02_r18in_ft224` | `baseline` s0–4 run_id'leri |

`--ebeveyn` aynı seed'in ebeveyn run_id'si (M5 eşleştirme): `<BASE_s0>` … `<BASE_s4>` = K3 aşama-3 `*_baseline_s{0..4}` klasör adları
(`runs/` altında; K3 bitince şef yazar). H03 için `<H01_s{s}>`.

## Ön koşullar (hepsi GPU'suz; durum 2026-10-10 ~11:00)
| öğe | durum |
|---|---|
| `data/medmnist/dermamnist.npz` (28) | var, MD5 ✓ |
| `dermamnist_64.npz` | var, MD5 ✓ (`data/medmnist/indirme_dogrulama.json`) |
| `dermamnist_128.npz` | var, MD5 `2defd784…` ✓ (11:00) |
| `dermamnist_224.npz` | **indiriliyor** (11:00: 294/1.091 MB) — betikler MD5 tutmazsa rc=2 ile durur (yarım dosyayla koşmaz) |
| DINOv2 ViT-B/14 ağırlığı | **önceden indirildi**: `data/hf_cache/.../snapshots/4685c99d…/model.safetensors`, 346.334.872 B, SHA256 `55cbb5d8…6d8c7b` |
| ResNet-18 IMAGENET1K_V1 | indirildi: `data/torch_cache/hub/checkpoints/resnet18-f37072fd.pth` |

---
## §0 Öznitelik önbelleği (genelde ELLE GEREKMEZ)
`sonda.py`, önbellek yoksa çıkarımı aynı süreçte GPU'da yapar (ilk kuyruk satırı; her 500 görüntüde progress.json).
Elle/önceden doldurmak istenirse (GPU işi, kuyruk dışında yapılmaz → kuyruk satırı olarak):
`$PY varyantlar/dinov2_ozn/ozn_cikar.py --model b --r 224 --bs 32 --progress-dir {RUN_DIR}`
Dosyalar: `data/oznitelik/dinov2_vitb14_r{28,64,128,224}.npz` (+ duman `_n256`). Test öznitelikleri K4'te ÇIKARILMAZ.

## §1 H01 — kuyruk `tur1-h01` (224 npz MD5 doğrulandıktan sonra)
```bash
KURAL_LP='{"nan_kes":true,"ilerleme_zorunlu_sn":300,"sabit_sn":300,"min_adim":8,"alt_sinir":{"val_acc":0.69}}'
# a1 duman: 256+256 görüntü, tek C, seed 0
$EKLE tur1-h01 --asama 1 --varyant h01_dinov2b_lp_r224_duman --seed 0 --hipotez H01 --gpu-zorunlu --zaman-siniri-dk 15 \
  --etiket "H01 duman: DINOv2-B LP r224, 256+256, C=1" --kural "$KURAL_LP" -- \
  "$PY" varyantlar/dinov2_sonda/sonda.py --seed 0 --r 224 --alt-kume 256 --C 1.0 --secim-seedleri 0 --out "{RUN_DIR}/metrics.json"
# a2 ön-uçuş: tam train+val çıkarımı (ilk satır) + seed 0,1 × 8 C
for s in 0 1; do
$EKLE tur1-h01 --asama 2 --varyant h01_dinov2b_lp_r224_onucus --seed $s --hipotez H01 --gpu-zorunlu --zaman-siniri-dk 20 \
  --etiket "H01 ön-uçuş: DINOv2-B LP r224, C ızgarası, C* = ort(s0,s1)" --kural "$KURAL_LP" -- \
  "$PY" varyantlar/dinov2_sonda/sonda.py --seed $s --r 224 --C izgara --secim-seedleri 0,1 --out "{RUN_DIR}/metrics.json"
done
# a3 tam: seed 0–4 × 8 C, tek C* = argmax 5-seed ortalaması
for s in 0 1 2 3 4; do
$EKLE tur1-h01 --asama 3 --varyant h01_dinov2b_lp_r224 --seed $s --hipotez H01 --ebeveyn "<BASE_s$s>" --gpu-zorunlu --zaman-siniri-dk 15 \
  --etiket "H01: donuk DINOv2 ViT-B/14 CLS (native 224) + L2 softmax-LR, C* val'de" --kural "$KURAL_LP" -- \
  "$PY" varyantlar/dinov2_sonda/sonda.py --seed $s --r 224 --C izgara --secim-seedleri 0,1,2,3,4 --out "{RUN_DIR}/metrics.json"
done
```
Not: `min_adim 8` → canlı `alt_sinir` yalnız ilk seed'in 8 C'si bittikten sonra devreye girer; progress `val_acc` =
o ana dek EN İYİ uyum (küçük C'nin çoğunluk tabanına çökmesi meşru, ızgara bunu içerir). Çıkarım sırasında `adim=0`.

## §2 H03 — kuyruk `tur1-h03` (r28 + r64; veri yerelde) ve `tur1-h03-r128` (128 indirilince)
```bash
# a1 duman (yalnız r28) ve a2 ön-uçuş (r28, seed 0,1) — H01 ile aynı kalıp, --r 28
$EKLE tur1-h03 --asama 1 --varyant h03_dinov2b_lp_r28_duman --seed 0 --hipotez H03 --gpu-zorunlu --zaman-siniri-dk 15 \
  --etiket "H03 duman: r28→224 bicubic, 256+256, C=1" --kural "$KURAL_LP" -- \
  "$PY" varyantlar/dinov2_sonda/sonda.py --seed 0 --r 28 --alt-kume 256 --C 1.0 --secim-seedleri 0 --out "{RUN_DIR}/metrics.json"
for s in 0 1; do
$EKLE tur1-h03 --asama 2 --varyant h03_dinov2b_lp_r28_onucus --seed $s --hipotez H03 --gpu-zorunlu --zaman-siniri-dk 20 \
  --etiket "H03 ön-uçuş r28" --kural "$KURAL_LP" -- \
  "$PY" varyantlar/dinov2_sonda/sonda.py --seed $s --r 28 --C izgara --secim-seedleri 0,1 --out "{RUN_DIR}/metrics.json"
done
# a3: r28 ve r64 × seed 0–4 (her r'nin ilk satırı çıkarımı yapar, ≈5 dk)
for r in 28 64; do for s in 0 1 2 3 4; do
$EKLE tur1-h03 --asama 3 --varyant h03_dinov2b_lp_r$r --seed $s --hipotez H03 --ebeveyn "<H01_s$s>" --gpu-zorunlu --zaman-siniri-dk 20 \
  --etiket "H03 r$r: DINOv2-B LP, girdi ${r}px→224 bicubic" --kural "$KURAL_LP" -- \
  "$PY" varyantlar/dinov2_sonda/sonda.py --seed $s --r $r --C izgara --secim-seedleri 0,1,2,3,4 --out "{RUN_DIR}/metrics.json"
done; done
# tur1-h03-r128: yeni veri dosyası olduğundan a2 (seed 0,1; varyant h03_dinov2b_lp_r128_onucus) + a3 (seed 0–4) — aynı kalıp, --r 128
```
H01'den önce koşulursa `--ebeveyn` boş bırakılır ve H01 bitince kart Sonuç'unda eşleme yazılır (kart: "koşum sırası sonucu değiştirmez").

## §3 H02 — kuyruk `tur1-h02` (en uzun; tek başına, R6)
```bash
KURAL_FT='{"nan_kes":true,"ilerleme_zorunlu_sn":300,"sabit_sn":300,"min_adim":3,"alt_sinir":{"val_acc_eniyi":0.69}}'
$EKLE tur1-h02 --asama 1 --varyant h02_r18in_ft224_duman --seed 0 --hipotez H02 --gpu-zorunlu --zaman-siniri-dk 10 \
  --etiket "H02 duman: R18-IN ince ayar 224, 1 epoch" --kural "$KURAL_FT" -- \
  "$PY" varyantlar/r18_imagenet/ince_ayar.py --seed 0 --epoch 1 --out "{RUN_DIR}/metrics.json"
for s in 0 1; do
$EKLE tur1-h02 --asama 2 --varyant h02_r18in_ft224_onucus --seed $s --hipotez H02 --gpu-zorunlu --zaman-siniri-dk 15 \
  --etiket "H02 ön-uçuş: 5 epoch (T_max=5; kısaltılmış)" --kural "$KURAL_FT" -- \
  "$PY" varyantlar/r18_imagenet/ince_ayar.py --seed $s --epoch 5 --out "{RUN_DIR}/metrics.json"
done
for s in 0 1 2 3 4; do
$EKLE tur1-h02 --asama 3 --varyant h02_r18in_ft224 --seed $s --hipotez H02 --ebeveyn "<BASE_s$s>" --gpu-zorunlu --zaman-siniri-dk 35 \
  --etiket "H02: ResNet-18 IMAGENET1K_V1 native 224 ince ayar (AdamW 1e-4, kosinüs, 30 ep, bs 64)" --kural "$KURAL_FT" -- \
  "$PY" varyantlar/r18_imagenet/ince_ayar.py --seed $s --epoch 30 --out "{RUN_DIR}/metrics.json"
done
```
Başlatma (şef, çift kanal): `"$PY" $SK/kuyruk.py --kok $WS kos <ad> --kademeli --paralel 1`.

## §3b Sonda öğrenme oranı ve yeni metrikler (K3.5 sonrası)
- `sonda.py --lr` varsayılanı **1e-2** (kart revizyonu yönünde; özgün kart 1e-3, sentetik N=7.007/D=768 denemede
  C=0,1'de amaç 0,21 vs lbfgs 0,06 — yakınsamadı; 1e-2 ile amaç farkı < %1). Komutlarda `--lr` verilmez → 1e-2.
  Kart revizyonu farklı bir değer derse komuta `--lr <değer>` eklenir. Değer `config.json` → `protokol.lr`'de damgalı.
- Her `metrics.json`'a (H01/H02/H03) eklenen ön-beyanlı ikinciller: `val_tahmin_nv_payi`, `val_acc_molemax_disi`
  (+ `val_acc_molemax`, `val_dengeli_acc_molemax_disi`), `val_sizinti_tabakali_SN` (+ `_kapsam`). Ham S/N anahtarları
  `*_ham` sonekiyle kalır, raporlanmaz. H02'de `val_acc_aucsecim` (rakibin val-AUC ilk-kesin-artış kuralı) var.
- Eşleştirilmiş görüntü bootstrap'ı (koşumlar bittikten sonra, CPU):
  `"$PY" varyantlar/ortak/eslestirilmis_bootstrap.py --a-varyant baseline --b-varyant h01_dinov2b_lp_r224 --cikti sonuclar/bs_h01_vs_baseline.json`

## §4 Tahmini süreler (GTX 1650, FP32; `env/hiz-olcumu.md` + yerel CPU ölçümleri)
| kuyruk | a1 | a2 | a3 | toplam | dayanak |
|---|---|---|---|---|---|
| tur1-h01 | ≈1,5 dk | ≈7 + 2 dk | 5 × ≈2–3 dk | **≈20–25 dk** | çıkarım 8.010 ÷ 26 img/s = 5,1 dk; sonda uyumu CPU'da 3–5 s (N=7.007, 2.800 adım; sentetik ölçüm), her a3 satırı 5×8 = 40 uyum yapar |
| tur1-h03 (r28, r64) | ≈1,5 dk | ≈9 dk | 2 × (5,1 + 5×2,5) ≈ 35 dk | **≈45 dk** | büyütme PIL bicubic CPU'da, 8.010 görüntü < 30 s |
| tur1-h03-r128 | — | ≈9 dk | ≈12 dk | **≈20 dk** | aynı |
| tur1-h02 | ≈1,5 dk | 2 × ≈4 dk | 5 × ≈21,5 dk | **≈1,9 sa** | 43 s/epoch (kart bütçesi); npz yükleme + MD5 ≈ 15–20 s/satır |
Tur-1 toplamı ≈ 3,2 sa (SIRALAMA.md tahmini 2,65 sa; fark: her sonda satırı C* için 5 seed × 8 C ızgarasını kendi içinde yeniden uydurur, ≈+0,3 sa).

## §5 Beklenti kuralları TASLAĞI (şef onaydan sonra `kuyruk/<ad>.beklenti.json` olarak yazdırır; `metrik: val_acc`, `yon: max`)
Kaynaklar kartların üst tablosundan; `<B>` = K3 baseline 5-seed val_acc ortalaması (K3 bitince; ön-uçuş 20 ep: 0,7817).
**Önemli kısıt:** `yon_vs`/`tavan`/`sure_orani` referansı (baseline) aynı kuyrukta ve aşamada koşum ister
(`beklenti_kontrol.py` yalnız o kuyruğun koşumlarına bakar) → baseline bu kuyruklarda yok; yerine mutlak `aralik`
(alt = `<B>` + 0,02 = öldürme kuralı (b) eşiği, üst = kartın "görülmemiş" eşiği) önerilir.

| kuyruk/aşama | kurallar (kaynak) |
|---|---|
| h01 a1 | `sonlu`; `min_kosum 1`; `aralik [0.55, 0.95]` (256 train, tek C: gevşek; alt ≈ çoğunluk tabanı 0,669 − 0,12); `ikincil_aralik param_sayisi [5383, 5383]` (768×7+7, H01 kartı); `oznitelik_boyut [768,768]`; `oznitelik_determinizm_maks_fark [0, 1e-5]` (H01 "Seed'in rolü"); `egitim_sure_sn [1, 900]` |
| h01 a2 | `sonlu`; `min_kosum 2`; `aralik [<B>+0.02, 0.88]` (öldürme b; tavan literatur-cipa §4b); `ikincil_aralik val_auc [0.93, 0.99]` (kart: 0,95–0,97; üst §4e); `oznitelik_img_s [10, 1000]` (öldürme d); `lbfgs_sgd_fark [-0.005, 0.005]` (öldürme e — **aşağıdaki açık soru 1'e bak**); `param_sayisi [5383,5383]`; `egitim_sure_sn [10, 1200]` |
| h01 a3 | a2 ile aynı + `min_kosum 5`. `farkli_seedler_farkli` → **açık soru 2** |
| h03 a1/a2 | H01 ile aynı; a2 `aralik [0.69, 0.80]` (r28: operasyonel DUR 0,69; tavan 0,80 kart) ; `ikincil_aralik giris_cozunurluk [28,28]` |
| h03 a3 | `sonlu`; `min_kosum 10` (r128 ayrı kuyrukta 5); `kollar_farkli` (r28/r64 kolları bit-bit aynıysa büyütme devreye girmemiş); seed başına `aralik` YOK (varyanta özel aralık için `aralik` + `"varyant"`: r28 [0.69, 0.80], r64 [0.69, 0.85], r128 [0.69, 0.87]; kart "görülmemiş" eşikleri) |
| h02 a1 | `sonlu`; `min_kosum 1`; `aralik [0.62, 0.90]` (1 epoch; alt = çoğunluk tabanı − 0,05, rakip duman gevşetmesiyle aynı; üst kart tavanı); `param_sayisi [11180103, 11180103]` (kart; CPU'da ölçüldü ✓); `egitim_sure_sn [10, 300]` |
| h02 a2 | `sonlu`; `min_kosum 2`; `farkli_seedler_farkli`; `aralik [0.72, 0.90]` (5 ep, kısaltılmış: alt ≈ rakip testi 0,735 − 0,015; üst tavan); `epoch_sure_ort_sn [10, 90]` (öldürme d); `val_auc [0.88, 0.99]`; `param_sayisi` |
| h02 a3 | `sonlu`; `min_kosum 5`; `farkli_seedler_farkli`; `aralik [<B>+0.02, 0.90]`; `val_auc [0.93, 0.99]` (§4e); `epoch_sure_ort_sn [10, 90]`; `param_sayisi`. Öldürme (e) (`en_iyi_epoch = 30` ve artıyor) tahlilcide |

## §6 Yerel CPU mini doğrulama (2026-10-10, GPU KULLANILMADI; ölçüm DEĞİL) — `runs_yerel_duman/runs/`
CPU iş parçacığı 2 ile sınırlı (koşan K3 GPU işinin CPU veri hattını yavaşlatmamak için). kosum.py ile sarmalandı (`--kok runs_yerel_duman`),
öznitelik önbelleği `DERMA_OZN_DIZIN=runs_yerel_duman/oznitelik` (gerçek önbellek dizinine yazılmadı).
| koşum | rc | süre | sonuç |
|---|---|---|---|
| `cpu_mini_h03_vits_r28` s0 (ViT-S/14, r28→224, 32+32 görüntü, C {0.1,1}, seçim seed 0,1) | 0 | 23,6 s | çıkarım 8,7 img/s CPU; determinizm maks fark 0; tüm anahtarlar sonlu; test_ anahtarı yok |
| aynı s1 / aynı s0 tekrar | 0 / 0 | 7,8 / 8,0 s | önbellekten (=1); s0 tekrar `tahmin_val.npy` **bit-bit aynı**; s0↔s1 maks fark 0,046 (seed bağlı ✓); iki seed de aynı C* |
| `cpu_mini_h03_vits_r28_izgara` s2 (8 noktalı ızgara) | 0 | 11,6 s | ızgara/C* seçimi/lbfgs yolu çalışıyor |
| `cpu_mini_h02_r18_s28` s0, s0, s1 (R18-IN, 28 px, 64 görüntü, 1 adım) | 0 | 23 / 8 / 9 s | `param_sayisi` 11.180.103 (= kart); `ilk_adim_loss` s0 = s0 tekrar = 2.018967628479004 (bit-bit), s1 = 2.180549 (farklı ✓) |
**K3.5 ek doğrulaması (20:10, CPU):** `degerlendir()` baseline'ın 5 aşama-3 koşumunun `tahmin_val.npy`'sinde
`scripts/hata_kirilim.py` (`sonuclar/hata_kirilim_ozet.json`) ile seed başına eşleşti: `val_tahmin_nv_payi` 0,672 ± 0,056
(maks fark 0), `val_sizinti_tabakali_SN` +2,78 ± 4,07 puan (maks fark 1e-17; kapsam 0,418), ham S−N −18,8 ± 2,9,
`val_acc_molemax_disi` 0,609 ± 0,033, `val_acc_molemax` 0,949, val_acc 0,7396. Eşleştirilmiş bootstrap: baseline↔kendisi
Δ = 0 [0, 0]; son epoch − seçilen Δ −0,10 puan [−1,58; +1,44]. Yeni API ile sonda/ince_ayar mini koşumları rc=0
(`cpu_mini_k35_*`), R18 sonuçları önceki koşumla bit-bit aynı.
ViT-B/14 img_size=224: `pos_embed` (1, 257, 768), CLS 768-b, param 85.724.928 (kartta yazan 86,6 M = 518 px konum gömmeli sürüm).

---

# Tur 3 komutları — H07 (KOŞULDU, foreground) · H08 + H09 (kuyruk `tur3`, BAŞLATILMADI)

*deney-yürütücü · 2026-10-11 ~04:50 · kartlar H07/H08/H09 mtime 04:02/04:03/04:04, SIRALAMA 04:08. Beklentiler
`kuyruk/tur3{,-on,-h07,-h07x,-h07a,-h07k,-h07rt,-h08k,-h08rt,-h09k,-h09rt}.beklenti.json` 04:22:18'de, ilk Tur-3 koşumundan
(04:22:28) önce yazıldı.*

## §T3
- **H07 (GPU çıkarım, foreground; her satır ≈ 25 s):** `tur3-h07` a1 `h07_h02_tta8_duman` s0 → a3 `h07_h02_tta8` s0–4
  (`varyantlar/tta/tta.py --ebeveyn-varyant h02_r18in_ft224 --gorunum d4 --val-bs 128`); `tur3-h07x` a3 `h07x_baseline_tta8` s0–4
  (`--ebeveyn-varyant baseline`). Kapılar GEÇTİ.
- **H07 CPU satırları (`CUDA_VISIBLE_DEVICES=`):** `tur3-h07a` `h07a_h02_tta4{d2,r90}` (`tta_alt.py --kaynak-varyant h07_h02_tta8 --alt d2|r90`);
  `tur3-h07k` `h07k_h07_logitayar` (`logit_ayari.py --ebeveyn-varyant h07_h02_tta8 --tau-izgara 0:1:0.1 --kat 5 --kat-rng 20261011 --kisit 0.010`);
  `tur3-h07rt` `rt_h07_sicaklik` (`sicaklik.py --kaynak-varyant h07_h02_tta8`). `--ebeveyn` = aynı seed H02 run (kart). Kapılar GEÇTİ.
- **H08/H09 özdeşlik dumanları (GPU, foreground, `tur3-on` a1, GEÇTİ):** `h08_r18in_ft224_d4kapali_duman`, `h09_r18in_ft224_renk02kapali_duman`
  (`ince_ayar.py --seed 0 --epoch 1 [--augment-renk 0] --ozdeslik-ref runs/20261010-204010_h02_r18in_ft224_duman_s0/metrics.json`).
- **Kuyruk `kuyruk/tur3.jsonl` (16 satır; ŞEF başlatır: `kuyruk.py --kok . kos tur3 --kademeli --paralel 1` + `izle tur3`):**
  | aşama | satırlar | içerik | zaman sınırı | ebeveyn |
  |---|---|---|---|---|
  | a1 | 2 | `h08_r18in_ft224_d4_duman`, `h09_r18in_ft224_renk02_duman` (s0, 1 ep, `--ozdeslik-ref` H02 dumanı → fark > 0 = devrede) | 8 dk | H02 duman |
  | a2 | 4 | `h08_r18in_ft224_d4_onucus`, `h09_r18in_ft224_renk02_onucus` × s0, s1 × 5 ep | 15 dk | H02 onucus s0/s1 |
  | a3 | 10 | `h08_r18in_ft224_d4` s0–4 → `h09_r18in_ft224_renk02` s0–4 × 30 ep | 40 dk | H02 a3 aynı seed |
  Hepsi `--gpu-zorunlu`, `--kural KURAL_FT` (§3). Tahmini süre: a1 ≈ 2,5 dk + a2 ≈ 4 × 4,6 dk ≈ 18 dk + a3 ≈ 10 × 25,5 dk ≈ 4,25 sa
  → **≈ 4,6 sa GPU** (kart bütçeleri 2 × 2,4; üst 2 × 2,7). H09 erteleme kuralı (SIRALAMA): H09 a3 başlarken sayaç > 17 GPU-sa ise ertelenir
  (beklenen ≈ 12,3).
- **Kuyruk sonrası (deney-yürütücü, CPU):** `h08k_h08_logitayar` / `h09k_h09_logitayar` (`tur3-h08k` / `tur3-h09k`), `rt_h08_sicaklik` /
  `rt_h09_sicaklik` (`tur3-h08rt` / `tur3-h09rt`) — beklentileri şimdiden yazılı; ölçüm `scripts/tur3_olcum.py --kart h08|h09`;
  H09 kaynak kırılımı `scripts/hata_kirilim.py --varyant h09_r18in_ft224_renk02`.
- **Ablasyon (ii) yalnız-nv (H04/H06):** kartta parametreleme/ızgara tanımsız → **koşulmadı** (H04/H06 Sonuç'a kaydedildi).
- **Kuyruk sonrası KOŞULDU (2026-10-11 09:11–09:14, CPU, `CUDA_VISIBLE_DEVICES=`, kosum.py foreground):** `h08k_h08_logitayar` /
  `h09k_h09_logitayar` (`logit_ayari.py --ebeveyn-varyant h08_r18in_ft224_d4|h09_r18in_ft224_renk02 --ebeveyn-dosya tahmin_val.npy
  --tau-izgara 0:1:0.1 --kat 5 --kat-rng 20261011 --kisit 0.010`), `rt_h08_sicaklik` / `rt_h09_sicaklik` (`sicaklik.py --kaynak-varyant …
  --kaynak-dosya tahmin_val.npy`); `--ebeveyn` aynı seed H02 run, `--asama 3`, kuyruk etiketleri `tur3-h08k/h09k/h08rt/h09rt`; 20/20 rc=0.
  Kapılar (`beklenti-tur3-{h08k,h09k,h08rt,h09rt}-asama3.md`, `beklenti-tur3-asama3.md`) GEÇTİ. Ölçüm `tur3_olcum.py --kart h08|h09`,
  ek `scripts/tur3_ek_olcum.py . sonuclar/tur3_h08h09_ek_olcum.json` (D_kk, kaynak başına GA, epoch/kayıp denetimi). Kırılım:
  `hata_kirilim.py --varyant V --etiket V --cikti-csv sonuclar/hata-kirilim-V.csv --cikti-json sonuclar/hata_kirilim_ozet_V.json`
  (çıktı yolu verilmezse baseline'ın `hata-kirilim.csv`/`hata_kirilim_ozet.json`'ı ezilir — ezildi ve `hata_kirilim.py` varsayılanıyla
  baseline için yeniden üretildi). Leaderboard'lar venv ile: dengeli → ece → val_acc (en son).

---

# Tur 4 ön-ölçüm — DINOv2 ince ayarı (KART YOK; SIRALAMA Tur-4 notu + tahlil-tur3 §5) · deney-yürütücü 2026-10-11 10:45

## §T4 Yerel GPU ölçümü (GTX 1650 4 GB, FP32, TF32 kapalı; `runs_yerel_duman/runs/*t4_*`; ÖLÇÜM, karar değil)
Betik: `varyantlar/dinov2_ft/ince_ayar_vit.py` (README aynı dizinde). Etkin bs 64, AdamW baş 1e-3 / gövde 1e-5, adım başına kosinüs
(1 epoch'luk koşumda lr epoch içinde 0'a iner → val_acc yalnız bilgi). Her satır `kosum.py --gpu-zorunlu` ile foreground, GPU'da başka iş yokken
(nvidia-smi 0 MiB). GPU 77–87 °C, SM saati 1.830/2.100 MHz (sürekli ısıl durum ölçüme dahil).
| koşum | yapılandırma | s/epoch (eğitim + val) | img/s (kararlı) | tepe GPU ayrılan / tahsis | tepe RAM özel / ws | 1 ep val_acc | disk |
|---|---|---|---|---|---|---|---|
| `t4_gpu_onbellek_b_k4` | B, blok 0–7 token önbelleği (train+val 8.010) | **263 s** yazma (hesap ≈ 219 s, 36,5 img/s) | 30,5 (yazma dahil) | 0,64 GB | 3,9 / 4,6 GB | — | **3,16 GB** (`dinov2_vitb14_r224_blok8_tok_*`) |
| `t4_gpu_b_kismi_k4_1ep` | B kısmi K = 4 (28,4 M param), mikro 32 | **260,5** (248,4 + 12,0) | 28,2 | 3,08 / 2,35 GB | 5,6\* / 4,7 GB | 0,7956 | best_model 0,33 GB |
| `t4_gpu_onbellek_b_k2` | B, blok 0–9 token önbelleği | **290 s** yazma | 27,7 (yazma dahil) | 0,64 GB | 3,9 / 4,7 GB | — | **3,16 GB** (`…_blok10_tok_*`) |
| `t4_gpu_b_kismi_k2_1ep` | B kısmi K = 2 (14,2 M param), mikro 32 | **132,9** (126,6 + 6,2) | 55,3 | 1,95 / 1,26 GB | 4,5 / 4,6 GB | 0,7667 | 0,33 GB |
| `t4_gpu_s_tam_1ep` | S tam (21,6 M), grad ckpt, mikro 16 × 4 | **289,6** (280,8 + 8,7) | 25,0 | 1,04 / 0,68 GB | 4,1 / 2,4 GB | 0,7797 | 0,09 GB |
| `t4_gpu_s_tam_hiz_ck0_mb16` (20 adım) | S tam, ckpt yok, mikro 16 × 4 | **≈ 216** (7.007 / 33,8 + 8,5; ekstrapolasyon) | 33,8 | 2,24 / 1,71 GB | 5,3 / 2,3 GB | — | — |
| `t4_gpu_s_tam_hiz_ck0_mb32` (20 adım) | S tam, ckpt yok, mikro 32 × 2 | ≈ 207 | 35,4 | **3,61** / 3,04 GB (sınırda) | 6,7 / 2,3 GB | — | — |
| `t4_gpu_s_tam_hiz_ck1_mb64` (20 adım) | S tam, ckpt, mikro 64 | ≈ 272 | 26,6 | 1,29 / 1,12 GB | 4,4 / 2,3 GB | — | — |
| `t4_gpu_s_tam_hiz_ck0_mb64` (20 adım) | S tam, ckpt yok, mikro 64 | **rc = 98** (6 dk sınırı) | **3,8** | > 4 GB → WDDM paylaşımlı belleğe taşma | — | — | — |
| `t4_gpu_b_tam_hiz_ck1_mb16` (15 adım) | B tam (85,7 M), ckpt, mikro 16 × 4 | **≈ 1.050** (7.007 / 6,91 + 35,6) | 6,9 | 2,53 / 1,96 GB | 6,2 / 2,8 GB | — | 0,33 GB |
\* K = 4 satırı önbellek-varken-görüntü-yükleme düzeltmesinden önce koştu (train görüntüleri RAM'de, +≈ 1,1 GB). Sonraki koşumlar yalnız etiket + 8 val görüntüsü okur.
**Doğrulamalar:** önbellek ↔ tam ağ (görüntüden, fp32) normlanmış CLS maks |Δ| 2,2–2,4e-3 = fp16 yuvarlama payının kendisi (|CLS| ≤ 7,35; token maks |x| 8,6);
K = 2 ve K = 4 `ilk_adim_loss` 1,712790 / 1,712789 (iki önbellek aynı ağı veriyor); GPU'da aynı seed tekrarı `tahmin_val.npy` **bit-bit aynı** (s0 ↔ s0 maks |Δp| 0,
s0 ↔ s1 0,60; `t4_gpu_b_kismi_k2_det_*`, 15 adım); CPU mini (16 görüntü): aynı seed bit-bit aynı, grad ckpt açık ↔ kapalı fark 0 (kısmi ve tam); test_ anahtarı yok.
Ölçümün GPU maliyeti ≈ **0,62 GPU-sa** (13 koşum; plan ≈ 0,25 — fark: iki önbellek yazımı + S hız varyantları + rc=98 taşma koşumu 6 dk).

## §T4b 5 seed × N epoch tahmini (a3; GPU-sa = 5 × (N × s/epoch + ≈ 20 s açılış) + önbellek bir kez)
| seçenek | s/epoch | N = 10 | N = 15 | N = 20 | a1 + a2 (1 ep + 2 seed × 5 ep) | Colab tetiği (> 6 GPU-sa ya da > 300 s/epoch) |
|---|---|---|---|---|---|---|
| (a) B kısmi K = 2 | 133 | **1,96** | **2,88** | **3,80** | ≈ 0,43 | yok |
| (a) B kısmi K = 4 | 260 | **3,72** | **5,53** | 7,34 | ≈ 0,80 | N ≤ 15 yok; N = 20 tetikler |
| (b) S tam, ckpt yok, mikro 16 | ≈ 216 | **3,03** | **4,53** | 6,03 | ≈ 0,66 | N ≤ 15 yok |
| (b) S tam, ckpt, mikro 16 | 290 | 4,06 | 6,06 | 8,07 | ≈ 0,88 | N = 15'te sınırda tetikler |
| (c) B tam, ckpt, mikro 16 | ≈ 1.050 | 14,6 | 21,9 | 29,2 | ≈ 3,2 | **tetikler (epoch > 300 s, 5-seed > 6) → Colab, KULLANICI ONAYI** |
Önbellek (a): K = 2 ≈ 0,08, K = 4 ≈ 0,07 GPU-sa (seed'den bağımsız, bir kez; ilk a1 satırı yazar). Bütçe çerçevesi (tahlil-tur3 §7): kalan Tur-4…8 ≈ 15,8 (A yolu) / 10,9 GPU-sa (B yolu).
Disk: kullanılmayan K'nın önbelleği silinebilir (3,16 GB); a3 best_model B 5 × 0,33 GB, S 5 × 0,09 GB.
**Kısıtlar:** önbellek tek görünüm → B yolu (H08 D4 ebeveyni) kısmi modla uyumsuz (8 görünüm ≈ 25 GB); S tam görüntüden çalıştığı için D4 eklenebilir.
GPU belleği 4 GB'ı aşarsa koşum ölmez, 10× yavaşlar → kartta `epoch_sure_ort_sn` üst sınırı (kural d) ve mikro-bs ön-kaydı gerekir.

---

# Colab hattı (KULLANICI KARARI 6–7) — kuyruklar kc1-tekrar, kc2-tur4-{h12,h10,h11} · deney-yürütücü 2026-10-11 ~12:10

*Beklentiler 11:58–11:59 (kartlar H10/H11/H12 10:53–10:55, SIRALAMA tekrar kapısı ön-kaydı 11:39), hiçbir Colab koşumundan ÖNCE.
Kuyruklar `colab/hat/kuyruk_yaz.py` ile `kuyruk.py ekle` üzerinden yazıldı; BAŞLATILMADI (kullanıcı defteri Colab'da koşar).*

## §C1 Düzen
- Drive kökü `<DRIVE_KOK>\3-1` ↔ Colab `<DRIVE_KOK>/3-1` (paket + `PAKET.json` + kalıcı `runs/ kuyruk/ sonuclar/ env/ durum/`);
  defter `yz/3-1-colab.ipynb` (ilk hücrede `KUYRUK = 'kc1-tekrar' | 'kc2-tur4'`). Colab'da kod `/content/is`'e kopyalanır + SHA256;
  `runs/ kuyruk/ sonuclar/ env/` Drive'a bağ; `data/medmnist`, `data/dermamnist_c` → `/content/veri` (oturum başına Zenodo + MD5/SHA256);
  `data/{oznitelik,hf_cache,torch_cache}` yerel diskte yeniden üretilir.
- Her komut `python colab/fp32_kos.py <betik> …` (rakip köprüsü TF32'yi kapatmıyordu; L4'te cudnn.allow_tf32 varsayılanı True).
  Ortam: `NVIDIA_TF32_OVERRIDE=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `TZ=Europe/Istanbul`, kuyrukta `HF_HUB_OFFLINE=1` (SHA256'sı doğrulanmış önbellek).
  Yerel parite: GTX 1650'de TF32 yok → sarmalayıcı yerel aritmetiği değiştirmez. Öz-sınama: fp64'e göre göreli hata < 1e-4 (TF32 ≈ 1e-3) yoksa DUR.
- CPU RNG akışları (randperm, randint, normal_, nn.Linear init) torch 2.5.1 ↔ 2.11.0 BİT BİT AYNI (scratch sınaması) → batch sırası/D4/baş init aynı.

## §C2 Kuyruklar
| kuyruk | a1 | a2 | a3 | ebeveyn | not |
|---|---|---|---|---|---|
| kc1-tekrar (32) | `baseline_colab_duman` 3 ep, `h01_…_colab_duman`, `h02_…_colab_duman` (+`--ozdeslik-ref colab/yerel_ref/…h02…`), `h08_…_colab_duman` (+ref yerel H08) | 4 kol × s0,s1 (rakip 20 ep, sonda ızgara, H02/H08 5 ep) | H01 ×5 (s1–4 `--lbfgs-yok`) → H02 30 ep ×5 → H08 30 ep ×5 → rakip 100 ep ×5 | tekrarlanan YEREL run_id | zs rakip a3 150 dk (yerel 75; Colab CPU payı) |
| kc2-tur4-h12 (8) | `h12_dinov2s_lp_r224_duman` | s0,s1 | s0–4 (s1–4 `--lbfgs-yok`) | `h01_dinov2b_lp_r224_colab[_…]:s<seed>` (sembolik) | kart aynen |
| kc2-tur4-h10 (8) | 1 ep (önbellek L4'te yazılır) | 5 ep × s0,s1 | 15 ep × s0–4 | `h02_r18in_ft224_colab[_…]:s<seed>` | özdeşlik toleransı 1e-3 (kc1 a1 ampirik farkı ≤ 1e-4 ŞART) |
| kc2-tur4-h11 (8) | 1 ep | 5 ep × s0,s1 | 15 ep × s0–4, `colab/erteleme_kapisi.py --esik 23.0 --baslangic-sa 15.3` | aynı | sayaç L4-sa ile |
Gruplar `kuyruk/gruplar.json`; kc2 tek kuyruk değil, çünkü `ikincil_aralik` varyant filtresi taşımaz (param_sayisi çakışması).
kc2 defterde `kuyruk/kc2-tur4.izin.json` (şef, tekrar kapısı geçince) olmadan BAŞLAMAZ.

## §C3 Tahmini L4 süreleri (ölçüm yok; yerel ölçüm + L4/GTX 1650 FP32 oranı 3,4× (R18 bağlantı testi, TF32 belirsiz) … 10× (tepe FLOPS))
kc1: a1 ≈ 5 dk · a2 ≈ 25–35 dk · a3: H01 5 × ≈ 3,5 dk + H02 5 × 4–8 dk + H08 5 × 4–8 dk + rakip 5 × 12–30 dk → **≈ 2,5–4 sa** (+ kurulum/veri/öz-sınama ≈ 10–20 dk/oturum).
kc2: H12 ≈ 25 dk · H10 ≈ 45–90 dk · H11 ≈ 45–85 dk → **≈ 2–3,3 sa**.

## §C4 İzleme (yerel)
`<uv-python> scripts/colab_izle.py --grup kc1-tekrar --aralik 60 --asili-dk 20` (Monitor; olay satırları; grup bitince/durunca çıkar; PID yok).
Kendi-testi: `scripts/colab_izle.py --kendi-testi`. Devam mantığı uçtan uca sınandı (gerçek kuyruk.py/kosum.py/beklenti_kontrol.py, sahte komutlar).

## §C5 Paralel sekmeler (v2; kullanıcı onayı: Tur-4 ve K5 için 3 paralel Colab oturumu, aynı L4 sınıfı) · deney-yürütücü 2026-10-11 ~14:00
- Defterler `yz/3-1-colab-A.ipynb` (KUYRUK `kc2-tur4-h10`, en uzun) · `-B` (`kc2-tur4-h11`) · `-C` (`kc2-tur4-h12`); aynı kod, yalnız
  hücre 1 (`SEKME`, `KUYRUK`) farklı (`colab/hat/defter_uret.py` denetler). v1 `yz/3-1-colab.ipynb` (kc1-tekrar) DEĞİŞMEDİ.
- Kuyruk/beklenti dosyaları ve `gruplar.json` DEĞİŞMEDİ (ön-kayıt; PAKET `kuyruk_dosyalari` SHA256'ları önceki paketle aynı).
  Gruptaki sıra (H12 → H10 → H11) paralelde kalkar: hiçbir kapı başka bir Tur-4 kuyruğunun sonucuna bakmaz (ebeveynler kc1
  Colab kolları); H11'in temsil ebeveyni H12 yalnız tahlilde eşlenir.
- Oturuma ayrılan Drive dosyaları, çakışma bekçisi, izin devri: `colab/hat/README.md` §Paralel sekmeler.
- **H11 erteleme sayacı paralelde:** 15,3 + Σ bütün Colab koşumları (her sekme, her kuyruk, kc1 dahil): damgalı `sure_sn` +
  damgasız/yetim koşumların `.heartbeat` `gecen_sn`'i; satırın kendi koşumu hariç. 3 sekme paralelken duvar saatinin 1 saati
  sayaca ≈ 3 GPU-sa ekler → her H11 a3 seed'i başlamadan sayaç yeniden hesaplanır; eşik aşılırsa kalan seed'ler rc=94
  (erteleme.json'da kuyruk bazında döküm). Drive'daki sayaç 2026-10-11 ~14:00: Colab 1,03 L4-sa (kc1, 21 koşum) → 16,3.
- Tahmini duvar süresi (§C3 tahminleri; L4 ölçümü yok): A ≈ 45–90 dk, B ≈ 45–85 dk, C ≈ 25 dk, + kurulum/veri/öz-sınama
  ≈ 10–20 dk/sekme → **≈ 1–1,8 sa** (sıralı ≈ 2–3,3 sa). GPU-sa toplamı değişmez (≈ 2–3,3 sa + 3 × kurulum).
- İzleme: `<uv-python> scripts/colab_izle.py --kuyruklar kc2-tur4-h10,kc2-tur4-h11,kc2-tur4-h12 --aralik 60 --asili-dk 20`
  (bir kuyruk DURURSA diğerleri izlenmeye devam; hepsi bitince/durunca çıkar: 0 tamam · 1 başarısız koşum var · 3 en az biri DURDU).
- K5: `defter_uret.py --sekmeler A=<kuyruk1>,B=<kuyruk2>[,C=<kuyruk3>]` → `drive_paketle.py` → `colab_izle.py --kuyruklar …`
  (izin gerekiyorsa `gruplar.json`'a grup + `"izin_gereken"`; kod değişmez).
