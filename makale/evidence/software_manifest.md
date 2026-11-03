# Yazılım Manifesti — DermaMNIST · DINOv2 ↔ ResNet-18 · protokol duyarlılığı (A2)

*yazilim-analisti · Faz 1 TAM mod · 2026-10-12 · önbellek yok.*
*Kaynak (DW): `<WS>` (standart deney workspace'i, `/konudan-makale` devri).*
*Kapsam: `notes/brief.md` + `notes/taslak.md` (A2 — protokol duyarlılığı). Kapsam genişletilmedi.*
*Bu manifest için yeni koşum yapılmadı. Test tahmin dosyaları açılmadı. Bütün sayılar dondurulmuş JSON/MD dosyalarından okundu. Satır numaraları 2026-10-12 itibarıyla geçerli.*

**Kısaltmalar.** DW = deney workspace'i. [T] = `sonuclar/k5_test_sonuc.json` (SHA256 `e7ae79c3…`). [C1] = `sonuclar/k5_c1_sonuc.json` (SHA256 `00962fd1…`). [KV] = `sonuclar/k5_kuru_val_sonuc.json`. [K] = `sonuclar/karar.md`. [ON] / [ONb] = `hipotezler/K5-ONKAYIT.md` / `-b.md`. [G] = `gunluk/deney-gunlugu.md`. [DM] = `data/data_manifest.md`. Δ'lar puan (×100), köşeli parantez %95 görüntü-bootstrap GA'sıdır.

---

## Genel Bakış

- **Ne:** Resmî MedMNIST v2 DermaMNIST ResNet-18 (28 px, sıfırdan) rakibi ile 224 px ImageNet-ResNet-18 ve DINOv2 (sonda / kısmi-tam ince ayar) kollarının ön-kayıtlı karşılaştırması. Değerlendirme üç katmanlıdır: resmî test (lezyon tekrarlı), C0 (aynı modeller, lezyon-ayrık C-test alt kümesi) ve C1 (DermaMNIST-C bölmesi, resmî piksellerle yeniden eğitim). Resmî test ayrıca S/N (lezyonu train'de görülmüş / görülmemiş) olarak ayrıştırılır.
- **Akış:** K3 rakip replikasyonu → K3.5 hata analizi → K4 seçim döngüsü (5 tur, 13 kart; yalnız val) → Colab tekrar kapısı → K5 (test bir kez + C0 + C1) → karar (A-kayıtlı, vurgu A2; [K] §12).
- **Kod düzeni:**
  - Rakip köprüsü `scripts/rakip_kos.py`. Resmî kod `rakip/experiments` @`70b6b3a`, dokunulmadı.
  - Varyant kodları `varyantlar/{ortak,r18_imagenet,dinov2_sonda,dinov2_ozn,dinov2_ft,logit_ayari,tta,topluluk,k5}/`.
  - Analiz betikleri `scripts/` altında.
  - Colab hattı `colab/hat/` (`fp32_kos.py`, defterler, kuyruk yazıcıları).
  - Orkestra (kuyruk/kosum/beklenti) skill tarafında: `.claude/skills/konudan-makale/scripts/`.
- **Kanıt kökleri:**
  - `runs/`: yerel GTX 1650, Tur 0–3 ve Tur-5 CPU (218 dizin).
  - `runs_colab/`: Colab L4. K4 Colab kolları, K5 test ve C1 (197 koşum dizini; `ARSIV_MANIFEST.json`, 2.352 dosya, test dosyaları dahil, SHA256'lı).
  - `sonuclar/`: dondurulmuş analiz çıktıları.
- **Donanım (karşılaştırılan bütün sayılar):** Colab NVIDIA L4 24 GB, Python 3.13.15, torch 2.11.0+cu130, timm 1.0.30, medmnist 3.0.2, sklearn 1.6.1. Ortam FP32, TF32 kapalı, deterministik cuDNN, AMP yok. Kanıt her koşumda `runs_colab/<id>/config.json → donanim` ve `fp32_durum.json`. Yerel GTX 1650 (torch 2.5.1+cu121, Python 3.11.16) yalnız K3 replikasyonu, Tur 1–3 seçimi ve analizler için kullanıldı.

---

## Önerilen Yöntem

**Çerçeve (karar.md §6.A-A2, §12):** Önerilen bir "yöntem" yoktur; makalenin katkısı bir **analiz/değerlendirme çerçevesidir**:
1. Val'de seçilmiş kazanan **W = H10** (DINOv2-B son 4 blok ince ayarı). Test W'yi değiştirmez ([ON] §1 satır 72).
2. İki katmanlı lezyon-ayrık değerlendirme: C0 (yeniden eğitimsiz) + C1 (yeniden eğitimli).
3. Resmî test S/N ayrışımı.
4. Ön-kayıt + tek test açılışı + eşleştirilmiş bootstrap + Holm.

### Kollar (hepsi seed {0..4}; Colab kolları `_colab` soneki ya da `kc2-tur4-*` kuyruğu)

| Kol | Rol | Tanım (ebeveynden tek fark) | Kod (dosya:satır) | Colab kaynak koşumları (s0…s4) |
|---|---|---|---|---|
| **B** = baseline_c | rakip (otoriter) | MedMNIST resmî `models.ResNet18` (3×3 stem, maxpool yok, 11.172.423 param), 28 px, sıfırdan. Adam 1e-3, MultiStepLR [50, 75] γ 0,1, 100 ep, bs 128, CE, augmentasyon yok, Normalize(0,5; 0,5). Seçim **val-AUC** (resmî kural). Ek satır: son epoch | köprü `scripts/rakip_kos.py:56` (blob pin), `:85-92` (R2 doğrulama), `:97-104` (seed + deterministik), `:264` (resmî `main()` çağrısı); protokol tablosu `rakip/KURULUM.md:60-77` | `20261011-142338/143613/144846/150122/151359_baseline_colab_s*` |
| **H01** (H01c) | sonda satırı | Native 224 px **donuk DINOv2 ViT-B/14 CLS** (768-b) + L2'li softmax-LR sonda. Adam 1e-2, bs 256, 100 ep, kosinüs, StandardScaler yalnız train. C ızgarası 8 değer; tek C\* = 5 seed ortalamasının argmax'ı (resmî: C\* = 0,1; C1: 0,01) | `varyantlar/dinov2_sonda/sonda.py:4-25` (tarif), `:54` (C ızgarası), `:193` (C\* seçimi), `:219-250` (lbfgs denetimi); öznitelik `varyantlar/dinov2_ozn/ozn_cikar.py:45-48` | `20261011-125349/125744/130031/130320/130608_h01_dinov2b_lp_r224_colab_s*` |
| **H02** (H02c) | **adil kıyas satırı** (eşit önağırlık hakkı) | torchvision ResNet-18 IMAGENET1K_V1, native 224, uçtan uca ince ayar. AdamW 1e-4, wd 0,01, CosineAnnealingLR(T_max = 30), bs 64, CE, augmentasyon yok, FP32. Seçim val ACC (eşitlikte AUC) | `varyantlar/r18_imagenet/ince_ayar.py:4-14` (tarif), `:67-68` (ImageNet norm), `:136-137` (model), `:148-152` (`hazirla`), `:185-186` (opt/takvim), `:251` (val-ACC seçimi), `:254` (rakip kuralıyla ikincil seçim); kart `hipotezler/H02.md:15` | `20261011-130855/131620/132346/133112/133837_h02_r18in_ft224_colab_s*` |
| **H08** (H08c) | doğrulanmış güçlü R18 satırı | H02 + eğitimde örnek başına düzgün rastgele **D4** (8 dihedral dönüşüm). GPU'da, normalizasyondan sonra uygulanır. Ayrı üreteç `g_aug` = seed + 100003, bu yüzden batch sırası H02 ile aynı | `ince_ayar.py:155-160` (üreteç + öz-sınama), `:217-223` (uygulama); D4 tanımı `varyantlar/ortak/dihedral.py:1-40` (`donustur`), `:41` (`oz_sinama`); kart `hipotezler/H08.md:25` | `20261011-134604/135335/140107/140837/141608_h08_r18in_ft224_d4_colab_s*` |
| **H10** = W | val'de seçilmiş kazanan | DINOv2 ViT-B/14 (`vit_base_patch14_dinov2.lvd142m`, img 224). İlk 8 blok donuk (fp16 token önbelleği); son 4 blok + norm + Linear(768→7) eğitilir (28.364.551 param). AdamW, baş lr 1e-3 / gövde 1e-5, wd 0,01 (1-B muaf), adım başına kosinüs, ısınma 0, 15 ep, etkin bs 64 (mikro 32 × 2), **augmentasyon yok**. Seçim val ACC | `varyantlar/dinov2_ft/ince_ayar_vit.py:242-267` (argümanlar), `:293` (`timm.create_model`), `:341-357` (önbellek tutarlılık denetimi), `:391-398` (dondurma), `:417` (lr grupları), `:424` (AdamW), `:431-436` (kosinüs), `:532` (seçim); "paket beyanı" (H02 → H10'un 9 bit-düzeyi farkı) `hipotezler/H10.md:30-45` | `20261011-160211/161411/162611/163811/165010_h10_dinov2b_k4_ft224_s*` |
| **H11** | Pareto satırı (betimsel) | DINOv2 ViT-S/14 tam ince ayar, tarif H10 ile aynı; grad-ckpt 0, mikro-bs 16 | `ince_ayar_vit.py` `--mod tam --model s`; kart `hipotezler/H11.md:17` | `20261011-160117/161247/162419/163552/164724_h11_dinov2s_ft224_s*` |
| **H12** | Pareto satırı (betimsel) | DINOv2 ViT-S/14 donuk CLS + sonda (H01 tarifi); C\* = 0,1 | `sonda.py --model s`; kart `hipotezler/H12.md:17` | `20261011-174515/174846/175135/175422/175710_h12_dinov2s_lp_r224_s*` |
| **T4** = H08c ⊕ H10 | **yalnız ek tablo, betimsel** (KARAR 12) | Seed-eşli ½ ; ½ log-olasılık topluluğu (kırpma 1e-12) + softmax. Test tahminlerinden CPU'da | `scripts/k5_test.py:384-385`; H13 birleştiricisi `varyantlar/topluluk/topluluk.py` | — (türetilmiş) |
| **H04 / R-T katmanları** | 2. ve 3. satır, **karar vermez** | H04: `argmax[log p − τ·log π_train]`, τ\* tam-val donmuş: B 0,2 · H01 0,2 · H02 0,7 · H08 0,7 · H10 0,7 · H11 1,0 · H12 0,2. R-T: seed başına T ∈ [0,5; 3,0], NLL ile seçilir (H10 T = 2,02/1,94/1,91/1,79/2,05) | `k5_test.py:160-161` (`tah_tau`), `:185-197` (katman özetleri), `:421-430` (H04 kıyasları); seçim `varyantlar/logit_ayari/logit_ayari.py:178-201`, `sicaklik.py:45`; donmuş değerler `sonuclar/k5_dondurma.json → katman`; R-T protokolü `hipotezler/SIRALAMA.md:440-453` | — (post-hoc) |

### K5 çıkarım ve C1 yolu
- **Test çıkarımı:** `varyantlar/k5/k5_cikarim.py`. Türler: `rakip|r18|vit_kismi|vit_tam|sonda` (`:80`).
  - P6 özdeşliği: val argmax N/N ∧ maks |Δp| ≤ 1e-5 (`:17`, `:88`, `:278`). Test tahmini yalnız özdeşlik geçerse ve `--test-kilidi-acik` verildiyse yazılır (`:143`, `:283`).
  - baseline_c'de yeniden çıkarım yapılmaz; dondurulmuş `tahmin_test.npy` kullanılır.
- **C1 eğitimi:**
  - Veri yaması `varyantlar/k5/c_bolme.py`; sarmalayıcı `c_bolme_sar.py` (aynı süreçte K4 betiği + C verisi).
  - Rakip için `c_rakip_kos.py:54-58`: süreç içinde `medmnist.INFO` MD5 ve `n_samples` 8.215/573/1.227'ye çevrilir. Rakip dosyalarına dokunulmaz.
- **C1 verisinin yapımı:** `scripts/c_bolme_npz.py:97-131`.
  - Resmî pikseller `dermamnist_c_esleme.csv` sırasıyla yeniden dizilir.
  - Denetimler: lezyon kesişimi 0/0/0, etiket = C etiketi = CSV dx, C-test = resmî test aynı indeksin **bayt-özdeş** kopyası.
  - Çıktı `data/dermamnist_c_resmipiksel_{28,224}.npz` + `.json` (`denetim_gecti: true`).
- **FP32 sarmalayıcı:** `colab/hat/fp32_kos.py:48-59`. TF32 kapatılır (eski + yeni API), `float32_matmul_precision = highest`, deterministik. Durum `{RUN_DIR}/fp32_durum.json`'a yazılır. Varyant betikleri de `varyantlar/ortak/kayit.py:44-55` ile TF32'yi kapatır.
- **Test kilidi (M4):**
  - `varyantlar/ortak/veri.py:72-73`: `test_izin` olmadan test bölümü okunamaz.
  - `kayit.py:109-113`: K4'te `test_` anahtarı yazılamaz.
  - `k5_test.py:343-356`: SHA256 dondurma denetimi ve KILIT; `:391-394` KILIT yazımı.

---

## Mevcut Karşılaştırma Yöntemleri

| Baseline | Durum | Kanıt |
|---|---|---|
| **baseline (rakip, otoriter)** = MedMNIST v2 resmî ResNet-18 28 px | Kodda var, pinli: experiments `70b6b3a7…`, medmnist 3.0.2 = `805b742`; **yama YOK** (`rakip/KURULUM.md:5-13`, `:32`). Replikasyon iki kez YEŞİL | yerel GTX 1650: test ACC **0,7296 ± 0,0176** (`sonuclar/replikasyon.md`); Colab L4: **0,7395 ± 0,0053**, seed değerleri 0,7436/0,7317/0,7411/0,7441/0,7367 (`replikasyon-colab.md`); kabul aralığı [0,7218; 0,754] (Yang 2023 T3 0,735; Doerrich 2025 73,38 ± 0,61; Medeiros 2026 0,742). Ek: `test_auc` 0,9148 ± 0,0034 ve `test_acc_son` 0,7395 ± 0,0076 YEŞİL |
| **baseline_c (son epoch)** | ikincil rakip satırı | baseline_son − baseline −0,31 [−1,28; +0,73] [T] → seçim kuralının etkisi ihmal edilebilir |
| **H02c** (eşit önağırlık hakkı verilmiş R18) | adil kıyas satırı; aile 2'nin referansı | [T] `kiyas.h10-h02`, `h08-h02` |
| **H01c / H12** | sonda tabanları | [T] `kiyas.h01-baseline`, `h12-h01` |
| Literatür çıpaları (koşulmadı) | Wu & Papież 2025 DINOv2-B uçtan uca 92,02 ± 0,32 (hakemsiz; protokolü belirsiz, resmî bölme); DINOv2-B LP 83,97; Abhishek vd. 2025 C npz R18-28 0,825 ± 0,006; Zhang & Li 2026 R18-IN-224 0,882 | `notlar/literatur-cipa.md` §3 (satır 126-135), §4(a), §5 (203-214) |

---

## Değerlendirme Metrikleri ve Çıktı Dosyaları

### Metrik tanımları (`scripts/k5_test.py`)
| Metrik | Tanım | Kod |
|---|---|---|
| ACC | argmax doğruluğu (= medmnist `getACC`) | `k5_test.py:172`; eşdeğerlik `varyantlar/ortak/degerlendir.py:46-48`, `:140` |
| AUC | makro OvR; sınıf başına `roc_auc_score` ortalaması (= `getAUC`), softmax olasılıkları | `k5_test.py:156-157` |
| dengeli ACC / dengeli-5 | sınıf duyarlılıklarının ortalaması / df ve vasc hariç 5 sınıf (`BES = [0,1,2,4,5]`) | `:49`, `:141-142` |
| makro-F1, ECE-15, NLL, nv payı, `duyarlilik_<sınıf>` | ikincil satırlar | `:145-153`, `:176-182` |
| S/N | S = lezyonu resmî train'de olan test görüntüsü (693/2.005), N = kalan 1.312. Ham Δ_S, Δ_N, Δ_S − Δ_N (**kural nesnesi**, [ONb] §1). Tabaka-std (sınıf × dx_type × kaynak, N-ağırlıklı, 20 tabaka) ikincildir, etiket vermez | `:75-91` (`bolum_meta`), `:251-270`, `:291-294`, `:469-486` |
| C0 | resmî modellerin test tahminleri C-test alt kümesinde (1.227; çoğunluk tabanı 0,7726) yeniden puanlanır | `:488-516` |
| K0 / KÇ0 | K0_X = Δ_X(resmî) − Δ_X(C0); ortak bootstrap (2.005 indeks) | `:297-305`, `:507-515` |
| C1 + K / KÇ | C1 C-test Δ'ları; K_X = Δ_X(resmî) − Δ_X(C1) = K0 + K1; ifade eşiği | `:571-658` |
| Ölçütler G1/G3/V/L | mekanik işaretleme | `:524-532`, `:652-654` |

### İstatistik protokolü ([ON] §2.4–2.5; `k5_test.py:15-28`)
- **Seed düzeyi:** kesin tek yönlü eşleştirilmiş t (`scipy.stats.ttest_rel`, alternative = "greater"), n = 5 (`:245-249`).
- **Görüntü bootstrap'ı:**
  - B = 2.000, `default_rng(0)`, aynı W matrisi iki kola uygulanır.
  - Δ\* = seed-ortalamalı görüntü doğruluğu farkı.
  - %95 yüzdelik GA; p = (1 + #{Δ\* ≤ 0})/(B + 1), dolayısıyla en küçük p ≈ 0,0005 (`:201-221`).
  - Alt kümelerde (C0/C1) ayrı rng(0) bootstrap'ı kullanılır (`:274-277`).
- **Holm:** α = 0,05, tek yönlü; t p'lerine ve bootstrap p'lerine **ayrı ayrı** uygulanır (`:280-288`, `:433-451`).
  - Aile 1 = {H01c, H02c, H08c, H10} − B; marj ≥ +2,0 puan.
  - Aile 2 = {H08c, H10} − H02c; marj ≥ +0,5.
  - C1 aileleri: C1-1 = {H01, H02, H08, H10} − B; C1-2 = {H08, H10} − H02 (`:614-627`).
- **"Anlamlı" kuralı:** doğrulama kuralı geçti ∧ Holm t < 0,05 ∧ Holm boot < 0,05 (`:444-447`).
- **Etiket kuralı:** GA alt > 0 → "desteklendi"; GA üst < 0 → "tersi"; diğer durumlar → "ayırt edilemedi" (`:308-309`).
- **Betimsel kıyaslar (Holm dışı):** H10 − H08c, H11/H12 satırları, T4 satırları, baseline_son (`:63-64`).

### Çıktı dosyaları ve JSON şemaları
**`sonuclar/k5_test_sonuc.json`** (kip test, 2026-10-11T21:12:03, `ikinci_acilis: null`). Üst anahtarlar:
`kip, tarih, runs, betik_sha256 (13a71cf1…), onkayit_sha256 (ac8215ea…), yontem, kosumlar{kol: [5 run_id]}, P6{gecti, hatalar, satir}, ikinci_acilis, n (2005), n_S (693), n_N (1312), kollar, n_tabaka (20), kiyas, kiyas_H04_katmani, aile1, aile2, dogrulama, SN, C0, olcutler`.
- `kollar.<kol>` (kol ∈ baseline, h01, h02, h08, h10, h11, h12, baseline_son, t4):
  - `{ort, std, seed}`; `seed` alanında 5 elemanlı listeler var: `acc, auc, dengeli_acc, dengeli_acc_5sinif, makro_f1, ece15, nll, nv_payi, duyarlilik_{akiec,bcc,bkl,df,mel,nv,vasc}`.
  - Ek alanlar: `katman_H04{tau, acc_ort, dengeli_ort, dengeli5_ort, acc_seed}` ve `katman_RT{T[5], ece15_ort, nll_ort, argmax_degismedi}`.
- `kiyas.<B-A>` (15 çift: `h01|h02|h08|h10-baseline`, `h08|h10-h02`, `h10-h08`, `h11|h12-baseline`, `h11-h02`, `h12-h01`, `t4-h10|h02|h08`, `baseline_son-baseline`):
  - `acc{A, B, delta, ga, p_boot}`.
  - `dengeli_acc{…, n_gecerli_ornek}` ve `dengeli_acc_5sinif{…}`.
  - **`t_acc{fark_seed[5], ort, sd, p_tek}`** ve **`t_dengeli{fark_seed[5], ort, sd, p_tek}`**.
  - `farkli_dogru_goruntu`.
  - **`acc_S_ham{delta, ga}`, `acc_N_ham{delta, ga}`, `ham_dS_eksi_dN{nokta, ga}`, `S_payi_net`**.
  - `tab_std_dS / tab_std_dN / tab_std_dS_eksi_dN{nokta, ga}`.
- `aile1 / aile2.<B-A>`: `{delta_puan, ga_puan, p_t, p_boot, holm_t, holm_boot, kural_gecti, anlamli, ifade}`.
- `dogrulama`:
  - `"H08 (§2.1)"{delta_puan, ga_puan, dogrulandi, r18_satiri, gorulmemis{dACC>3, ACC>0,90, AUC>0,99, dengeli_delta>6,5}, dengeli_5sinif_delta_puan}`.
  - `"H10 (§2.2)"{…, gorulmemis{ACC>0,93, AUC>0,99, H10-H02c>+6,0}, beklenti_[+1,0;+4,5]_icinde}`.
- `SN.<h10-h02|h08-h02|h10-h08>`: `{hipotez, kural_kaynagi, kural, beklenen, ham_dS_eksi_dN_puan, ham_ga_puan, tab_dS_eksi_dN_puan, tab_ga_puan, tab_bilgi_etiketi, dS_ham_puan, dN_ham_puan, dS_tab_puan, dN_tab_puan}`.
- `C0`:
  - `{n 1227, cogunluk_tabani 0,7726}`.
  - `kol.<kol>{acc_ort, acc_std, dengeli_ort, dengeli5_ort, acc_eksi_cogunluk}`.
  - `kiyas.<h10-h02|h08-h02|h10-h08|h10-baseline>{kural, delta_puan, ga_puan, sonuc, p_t, dengeli_delta_puan, dengeli_ga_puan}`.
  - `KC0{K0_H10_puan, K0_H10_ga, K0_H08_puan, K0_H08_ga, K0_H10_eksi_K0_H08_puan, ga_puan, kural, beklenen}`.
- `olcutler{W, G1, G2_bayraklar, V}`.

**`sonuclar/k5_c1_sonuc.json`** (kip c1, 2026-10-12 02:11). Resmî test bölümlerinin aynısını yeniden içerir (KÇ hesabı için ön-kayıtlı). Aile 1 Δ'ları [T] ile bire bir aynı, bu manifestte denetlendi. Ek olarak `C1` anahtarı:
- `kol.<baseline|h01|h02|h08|h10>{acc_ort, acc_std, acc_eksi_cogunluk, dengeli_ort, dengeli5_ort, auc_ort}`.
- `kiyas.<7 çift>`: [T] `kiyas` ile aynı şema, S/N hariç; `t_acc.fark_seed` var.
- `kosumlar`, `iii_baseline{acc, beklenti_[0,79;0,85], gorulmemis_>0,87_INCELE, cogunluk_notu}`.
- `aile_C1_1`, `aile_C1_2`: `{delta_puan, ga_puan, holm_t, holm_boot, kural_gecti, dengeli_*}`.
- `i.<h10-h02|h08-h02|h10-h08>{kural, delta_puan, ga_puan, dengeli_yon}`.
- `ii_KC{K_H10_puan, K_H10_ga, K_H08_puan, K_H08_ga, K_H10_eksi_K_H08_puan, ga_puan, kural, bilesen_K0_H10, bilesen_K1_H10, bilesen_K0_H08, bilesen_K1_H08, oran_K_H10_bolu_delta_resmi, ifade_esigi}`.
- `G3`, `L`.

**Diğer dondurulmuş çıktılar:**
- `sonuclar/k5_kuru_val_sonuc.{json,md}`: P5 kuru koşusu; val + C-val ∩ val (573). Val → test kayması tablosu bu dosyadan gelir ([K] §1.2).
- `sonuclar/k5_dondurma.json` (+ `.sha256` = `c984fd0a…`, 18:54:44):
  - `katman{tau_tamval, T_tamval_seed}`, `sonda_C_yildiz`, `en_iyi_epoch`, `baseline_c_dondurulmus_test`, `istatistik`.
  - `kaynak_tahmin_val_sha256` (35 koşum), `dosya_sha256` (ön-kayıt, betik, kuyruk dosyaları).
- `sonuclar/k5_dondurma_ek_c1.json`: C1 veri beklenti SHA256 `895a50ab…` ve test sonucu SHA256.
- `sonuclar/k5_test.KILIT`: `{"zaman":"2026-10-11T21:12:03","ikinci_acilis":null,"betik_sha256":"13a71cf1…"}`.
- `sonuclar/ablasyon.csv` (10 satır): `sira, basamak, kol, rol, ne_degisti, taban, val_*` (ort/std/dengeli/Δ taban ve B'ye göre GA), `test_*` (+ holm_t/holm_boot), `c0_*`, `c1_*` (+ holm), `not`, `kaynak_*`.
  - Üretici `scripts/ablasyon_csv.py:1-25`. Yalnız üç JSON'dan okur; JSON'da olmayan değer boş bırakılır.
- `sonuclar/k5_test_sonuc.md`, `k5_c1_sonuc.md`: insan okunur tablolar (`k5_test.py:695-759`).
- **Koşum düzeyi** (`runs_colab/<id>/`):
  - `config.json`: `run_id, varyant, seed, etiket, ebeveyn, hipotez, kuyruk, asama, komut[], baslangic, makine`, `k5{kaynak_run, kaynak_sha256{best_model.pth, sonda_agirlik.npz, tahmin_val.npy, …}, ozdeslik_tol, vit{…}}`, `donanim{nvidia_smi, torch, cuda, cudnn, python, timm, sklearn, medmnist, cihaz}`.
  - `.done`: `{rc, sure_sn, bitis, metrics_var, metrics{…}}`.
  - `metrics.json` (K5 test): `val_acc, ozdeslik_argmax_ayni, ozdeslik_p_maks_fark, ozdeslik_gecti, test_acc, test_auc, test_dengeli_acc, test_ece, test_nll, test_n, alt_grup{sinif_0..6}`.
  - Tahmin dosyaları `tahmin_test.npy` (5 × N × 7, float32), `etiket_test.npy`, `fp32_durum.json`, `progress.json`, `gecmis.json` (eğitim koşumlarında).
- **Val (K4) leaderboard'ları:**
  - `sonuclar/leaderboard*.csv/md` (208 satır, tüm yerel + Colab val koşumları), `leaderboard-tur5*.csv`, `leaderboard-kc1.csv` (Colab tekrar kapısı).
  - `leaderboard-colab-k5-test.csv`, `-k5-c1-*.csv`: kapı/izleme amaçlı.
- **Hata kırılımı (val):**
  - `sonuclar/hata-kirilim-*.csv` (uzun biçim: `bolum, olcu, alt_kume, sinif, deger_ort, deger_std, n_seed, not`).
  - `hata_kirilim_ozet_*.json` (`zor.hep_yanlis_indeks` dahil).
  - `hata-kirilim-h10-karsilastirma.csv`, `hata-kirilim-h10-zor-gecis.json`.
  - Üretici `scripts/hata_kirilim.py:1-22` (test dosyası açmaz), `hata_kirilim_karsilastir.py`.
- **Nitel panel verisi (val)** *(makale için sonradan eklendi — deney-gelistirici, 2026-10-12)*:
  - Üretici `scripts/nitel_panel_val.py` (SHA256 `f16bbf36…`). Yalnız val okur; `veri.yukle(224, ("val",))`, `sizinti_maskesi("val")`. Kaynak: B/H02c/H08c/H10 Colab `tahmin_val.npy`, 20/20 SHA256 dondurma ve ARSIV_MANIFEST ile doğrulandı.
  - Çıktı `makale WS/figures/data/nitel_panel_val.json`: seçim ölçütü, başlık cümlesi, kategori aday sayıları, P/E denetimleri, 12 görüntü × 4 kol (çoğunluk tahmini, doğru seed sayısı, ort. olasılık, seed tahminleri).
  - Karolar `makale WS/figures/tiles/nitel/val_<indeks>.png` (12 adet, 224 px, eksensiz).
  - Ön-kayıtlı kural `notes/eksik-kod-teklifi.md`; uygulama notu `notes/eksik-kod-uygulama.md`. Betimseldir, karar taşımaz. Görüntüler CC BY-NC 4.0.

### Anahtar dondurulmuş sayılar (manifest denetimi; yazımda [T]/[C1]/[K]'dan alıntılanır)
- Resmî test ACC (n = 2.005, 5 seed):

| Kol | ACC (ort ± std) |
|---|---|
| B | 0,7395 ± 0,0053 |
| H01c | 0,8260 |
| H02c | 0,8713 |
| H08c | 0,8976 ± 0,0027 |
| H10 | 0,8940 ± 0,0033 |
| H11 | 0,8837 |
| H12 | 0,8160 |
| T4 (ek) | 0,9169 |

- Aile 1: H10 − B **+15,45 [+13,84; +17,01]**, Holm t 1,29e-06, Holm boot 0,002.
- Aile 2:
  - H10 − H02c **+2,26 [+1,15; +3,30]**, Holm t 1,07e-04, Holm boot 0,001.
  - H08c − H02c +2,62 [+1,85; +3,36].
- S/N (H10 − H02c): Δ_S +5,43 [+3,23; +7,47], Δ_N +0,59 [−0,58; +1,76], ham fark +4,83 [+2,22; +7,18], `S_payi_net` 0,828.
- T-SN2 (H08c − H02c): Δ_S +4,27, Δ_N +1,75 [+0,91; +2,60], ham fark +2,52 [+0,52; +4,32]. "Dengeli dağılım" TUTMADI.
- C0 (n 1.227):

| Kol | ACC |
|---|---|
| B | 0,8289 |
| H02c | 0,9102 |
| H08c | 0,9252 |
| H10 | 0,9130 |
| T4 (ek) | 0,9335 |

  - C0-1 H10 − H02c +0,28 [−0,83; +1,47].
  - C0-2 H08c − H02c +1,50 [+0,70; +2,28].
  - KÇ0 +0,86 [−0,01; +1,78].
- C1:

| Kol | ACC |
|---|---|
| B | 0,8334 ± 0,0054 |
| H01 | 0,8755 |
| H02 | 0,9048 ± 0,0099 |
| H08 | 0,9257 |
| H10 | 0,9161 ± 0,0040 |

  - G3 H10 − B **+8,26 [+6,63; +9,94]**.
  - **L: H10 − H02 +1,12 [−0,05; +2,30] → FALSE.** Holm t 0,018, boot 0,031; tek yönlü p'ler < 0,05 olsa da "anlamlı" denmez.
  - H08 − H02 +2,09 [+1,24; +2,97].
  - KÇ +0,60 [−0,45; +1,68]; ifade eşiği FALSE (oran 0,503).
- H08 bayrağı:
  - Dengeli Δ +8,41 [+5,85; +11,22]; seed başına 8,08 / 8,55 / 8,65 / 8,06 / 8,72 (`kiyas.h08-h02.t_dengeli.fark_seed`).
  - 5 sınıflı +5,22; df/vasc payı %56.

---

## Veri Setleri

| Veri | Yükleme | Bütünlük | Not |
|---|---|---|---|
| **DermaMNIST (resmî, MedMNIST v3.0, Zenodo 10.5281/zenodo.10519652)** 28 px + MedMNIST+ 64/128/224 | `varyantlar/ortak/veri.py:31-73` (npz yolu, MD5 denetimi, test kilidi); rakip için `medmnist.DermaMNIST(root=data/medmnist, download=False)` | MD5 resmî: 28 `0744692d…`, 224 `8974907d…` (`data/medmnist/indirme_dogrulama.json`, hepsi `ok: true`) | 7.007 / 1.003 / 2.005; 7 sınıf; nv payı test %66,88 (çoğunluk tabanı 0,669). Test'te df 23, vasc 29 ([DM] satır 59-71) |
| **Lezyon eşlemesi** (resmî bölmenin lezyon etiketlemesi; yeni bölme değil) | `data/splits/dermamnist_lezyon_esleme.csv` (`split, index, image_id, lesion_id, dx`; 10.015 satır) + `data/meta/HAM10000_metadata.csv` (Dataverse v4.0) + `dermamnist_split_info.csv` (Abhishek vd.) | 10.015/10.015 etiket uyumu; lezyon sayıları Abhishek vd. ile bire bir ([DM] 37-40, 142-147) | **S/N tanımı:** S_test = lezyonu resmî train'de olan test görüntüsü, **693/2.005 (%34,56)**; val'de 348/1.003 (%34,70) ([DM] 82-92; `k5_test.py:75-91`). Tabaka = sınıf × dx_type × kaynak (`dataset`) |
| **DermaMNIST-C** (Abhishek, Jain, Hamarneh 2025, Sci Data 12:196; Zenodo 10.5281/zenodo.12739457 v1.1) | `scripts/indir_dermamnist_c.py`; eşleme `data/splits/dermamnist_c_esleme.csv` (`split_C, image_id, lesion_id, dx, split_resmi, index_resmi`) | CSV MD5 `1c54bc9f…`; `dermamnist_corrected_28.npz` MD5 `96e28629…`; 224 npz de indirildi (`data/dermamnist_c/`) | 8.215 / 573 / 1.227; lezyon-ayrık 0/0/0; **C-test ⊂ resmî test**; C-train = resmî train + 430 val + 778 test. C-test nv %77,26 (**çoğunluk tabanı 0,7726**); C-test'te df 8, vasc 17, akiec 35 ([DM] 222-289) |
| **C1 verisi = DermaMNIST-C bölmesi + resmî pikseller** | `data/dermamnist_c_resmipiksel_{28,224}.npz` (`scripts/c_bolme_npz.py:97-146`), yükleme `varyantlar/k5/c_bolme.py` | `dermamnist_c_resmipiksel_224.json`: `denetim_gecti true`, dosya SHA256 `d43a13b5…`, içerik `3b4d8b7d…`; ön kontrol P7 `sonuclar/k5_c_denetim.json` | Yayımlanmış C npz pikselleri resmî 28 px ile piksel-özdeş değil (RMSE medyan 14,05). Bu nedenle birincil = resmî piksel (KULLANICI KARARI 2026-10-10, [DM] 312-315). **C2 (yayımlanmış C npz) koşulmadı** |
| Lisans | CC BY-NC 4.0 (DermaMNIST, HAM10000, C npz); MedMNIST kodu Apache-2.0; DINOv2 Apache-2.0 | [DM] 122-136; `literatur-cipa.md` §6 | Klinik iddia yok |

---

## Çalıştırma Talimatı

Ortam (yerel, analiz):
```bash
WS=<WS>
PY=$WS/env/.venv/Scripts/python.exe   # Python 3.11.16, torch 2.5.1+cu121 (env/KURULUM.md)
cd $WS
```
Colab (ölçüm ortamı): `colab/hat/3-1-colab-*.ipynb` defterleri pinleri kurar (medmnist 3.0.2, timm 1.0.30, tensorboardX 2.6.5; torch Colab'ın kendisininkidir). Defter L4 değilse durur. Her kuyruk satırı `python colab/fp32_kos.py <betik> …` biçimindedir.

**Kopyala-çalıştır komutları** (arşivdeki `config.json → komut` alanından; `{RUN_DIR}` kosum.py tarafından doldurulur):
```bash
# B (rakip, 100 ep)
python colab/fp32_kos.py scripts/rakip_kos.py --seed S --epoch 100 --veri-kok data/medmnist --out {RUN_DIR}/metrics.json
# H01 / H12 (sonda)
python colab/fp32_kos.py varyantlar/dinov2_sonda/sonda.py --seed S [--model s] --r 224 --C izgara --secim-seedleri 0,1,2,3,4 --out {RUN_DIR}/metrics.json
# H02 / H08
python colab/fp32_kos.py varyantlar/r18_imagenet/ince_ayar.py --seed S --epoch 30 [--augment-d4] --out {RUN_DIR}/metrics.json
# H10
python colab/fp32_kos.py varyantlar/dinov2_ft/ince_ayar_vit.py --mod kismi --model b --son-k 4 --seed S --epoch 15 --lr-bas 1e-3 --lr-govde 1e-5 --wd 0.01 --katman-azalma 1 --isinma-epoch 0 --drop-path 0 --mikro-bs 32 --out {RUN_DIR}/metrics.json
# H11
python colab/fp32_kos.py varyantlar/dinov2_ft/ince_ayar_vit.py --mod tam --model s --grad-ckpt 0 --mikro-bs 16 --seed S --epoch 15 --lr-bas 1e-3 --lr-govde 1e-5 --wd 0.01 --katman-azalma 1 --isinma-epoch 0 --drop-path 0 --out {RUN_DIR}/metrics.json
# K5 test çıkarımı (P6 özdeşliği a1'de; a3'te --test-kilidi-acik)
python colab/fp32_kos.py varyantlar/k5/k5_cikarim.py --tur {rakip|sonda|r18|vit_kismi|vit_tam} --kaynak <K4 run_id> --seed S [--test-kilidi-acik] --out {RUN_DIR}/metrics.json
# C1 verisi + eğitim + C-test
python scripts/c_bolme_npz.py --kip c1 --out {RUN_DIR}/metrics.json
python colab/fp32_kos.py varyantlar/k5/c_bolme_sar.py --veri c1 -- <yukarıdaki K4 komutu, aynı argümanlar>
python colab/fp32_kos.py varyantlar/k5/c_rakip_kos.py --veri c1 -- --seed S --epoch 100 --out {RUN_DIR}/metrics.json
python colab/fp32_kos.py varyantlar/k5/k5_cikarim.py --tur <tür> --kaynak-varyant <kol>_c1 --kaynak-kuyruk k5-c1 --seed S --veri c1 --test-kilidi-acik --out {RUN_DIR}/metrics.json
```
Kuyruk sarmalayıcısı (örnek, `rakip/KURULUM.md:93-97`): `$PY .claude/skills/konudan-makale/scripts/kosum.py --kok <runs kökü> --varyant <ad> --seed S --asama 1|2|3 --gpu-zorunlu --kural '{…}' -- <komut>`. K5 kuyrukları `colab/hat/k5_kuyruk_yaz.py` ile yazıldı (`kuyruk/k5-test.jsonl`, `k5-c1-A-v3.jsonl`, `k5-c1-B.jsonl`; SHA256'lar `k5_dondurma.json → dosya_sha256`). Sıra `varyantlar/KOMUTLAR.md:54-76`.

**Analiz (yerel CPU; makale için yeniden koşulabilir olanlar):**
```bash
PYTHONIOENCODING=utf-8 $PY -I scripts/k5_test.py --kip kuru-val --runs runs_colab          # val kuru koşusu (test açmaz)
PYTHONIOENCODING=utf-8 $PY -I scripts/ablasyon_csv.py                                     # ablasyon.csv (yalnız JSON okur)
PYTHONIOENCODING=utf-8 $PY -I scripts/hata_kirilim.py --kosumlar <5 dizin> --etiket <ad> --cikti-csv … --cikti-json …   # val kırılımı
PYTHONIOENCODING=utf-8 $PY -I scripts/arsiv_dogrula.py                                     # runs_colab SHA256 denetimi
```
**YENİDEN KOŞULMAZ:** `k5_test.py --kip test` (KILIT var; betik ikinci açılışı `--ikinci-acilis` olmadan reddeder, `:354-356`). Aynı yasak `--kip c1` için de geçerlidir: C-test kol başına bir kez açıldı. Makale bu sayıları dondurulmuş JSON'lardan alır.

---

## Boşluk analizi girdisi

| Gereksinim | Gerçek-çıktı verisi var mı? | Kaynak / not |
|---|---|---|
| **Nitel panel** (örnek görüntüler + kolların tahminleri; figure-style §A2) | **VAR (val) — makale için sonradan eklendi, deney-gelistirici 2026-10-12:** `figures/data/nitel_panel_val.json` + `figures/tiles/nitel/*.png` (12 karo; üretici `scripts/nitel_panel_val.py`). *Önceki durum:* **Kısmen. Hazır figür yok**; tek PNG `data/raporlar/en_yakin_16_val_train.png` (val↔train yakın-kopya). Ham veri **val için tam**: görüntüler `data/medmnist/dermamnist_224.npz → val_images`; tahminler `runs_colab/<kol>_s*/tahmin_val.npy` (B, H01, H02, H08, H10, H11, H12; 5 seed); S/N maskesi `dermamnist_lezyon_esleme.csv`; "hep yanlış" indeksleri `sonuclar/hata_kirilim_ozet_*.json → zor.hep_yanlis_indeks` (H10 58, ortak çekirdek 14 görüntü, `hata-analizi.md` §9.6) | **Önerilen:** val'den panel. Test'e dokunmaz. Örnek kurgu: S ve N görüntülerinde B/H02/H08/H10 tahminleri; df/vasc örnekleri; ortak zor çekirdek. Test görüntüleri ve `tahmin_test.npy` arşivde var (`runs_colab/*-test_s*/`), ama **test tabanlı panel "test sonrası keşifsel" etiketi ister** ([ON] §6.3) ve KARAR 12'nin "test'e yeniden bakılmaz" ruhuyla çelişebilir. Kullanıcı kararı gerekir. Görüntü gösterimi CC BY-NC 4.0 atfıyla yapılır |
| **Önce/sonra hata kırılımı** | **VAR (val).** `hata-analizi.md` §9 (satır 218-356): B → H02c → H08c → H10 özet, sınıf duyarlılıkları, karışıklık hücreleri, S/N, kaynak/dx_type, zor örnekler; CSV `hata-kirilim-h10-karsilastirma.csv`. **Test sınıf duyarlılıkları da dondurulmuş:** [T] `kollar.<kol>.ort/seed.duyarlilik_*` (yeniden bakma gerekmez) | Brief'teki figür 3 (sınıf duyarlılığı, df/vasc vurgusu) [T]'den doğrudan çizilebilir. Sınıf-farkı GA'ları JSON'da yok; GA eklemek test tahminlerine yeniden bakmayı gerektirir → "test sonrası keşifsel" |
| **Tur/bütçe eğrisi** (val, orkestrasyon anlatısı) | **VAR.** Günlük tur satırları [G] satır 11-28 (Tur 0–5 + K5; en iyi val ACC: B 0,7396 → H02 0,8672 → H08 0,8893 → H10 0,8947 [Colab] → H13 0,8961 ÖLDÜR). Seed düzeyi `sonuclar/leaderboard.csv` (+ `-tur5.csv`). Kümülatif bütçe: Tur-1 ≈ 7–7,5 · Tur-2 ≈ 9,8 · Tur-3 ≈ 14,4–14,7 GPU-sa (`tahlil-tur1.md:234`, `tahlil-tur2.md:265`, `tahlil-tur3.md:184`) · Tur-4 sonu ≈ 21,5 · H12 +1,17 · K5 test +1,37 · C1 +6,55 → **≈ 30,6 / 32 oturum-sa** (alt sınır ≈ 26,4) ([G] 44-57) | Uyarı: Tur 1–3 yerel GTX 1650 val, Tur-4/5 Colab L4 val. Eğri iki donanımı karıştırır; tekrar kapısı farkı ≤ 0,3 puan (baseline +0,62). Figürde işaretlenmeli. Tur-4 bütçesi tek kalem olarak verilmemiş (≈ 21,5 − 14,7 ≈ 6,8; türetilmiş) |
| **S/N figürü** (Δ_S vs Δ_N, GA'lı) | **Nokta + GA VAR**, seed düzeyi **YOK**. [T] `kiyas.<h10-h02|h08-h02|h10-h08>.acc_S_ham / acc_N_ham / ham_dS_eksi_dN` (görüntü bootstrap GA'ları). Val karşılığı `hata-analizi.md` §9.3 ve `tahlil_tur4_ezber.json` (val Δ_S +6,26 / Δ_N +1,13) | GA'lı nokta-aralık figürü için yeterli. Seed başına S/N noktaları JSON'da yok; `tahmin_test.npy`'den türetmek **test sonrası keşifsel** olur |
| **C0 / C1 figürleri** (ablasyon zinciri × üç değerlendirme, seed dağılımlı) | **Resmî test:** seed düzeyi VAR ([T] `kollar.*.seed.acc`; `kiyas.*.t_acc.fark_seed`). **C1:** kol başına seed değerleri VAR (her `runs_colab/*_c1-test_s*/metrics.json → test_acc` ve `baseline_c1_s*/metrics.json → test_acc`; bunlar tek C-test açılışında yazılmış kayıtlardır); [C1] `kiyas.*.t_acc.fark_seed` VAR. **C0:** yalnız `acc_ort/acc_std` ve kıyas Δ/GA/p_t VAR; **kol başına seed değerleri YOK** | C0 için ort ± std ile çizilebilir. C0 seed noktaları `tahmin_test.npy`'den türetilirse **test sonrası keşifsel** etiketi gerekir. K/KÇ figürü için nokta + GA var ([T] `C0.KC0`, [C1] `ii_KC`) |
| Ablasyon tablosu | **VAR.** `sonuclar/ablasyon.csv` + [K] §4 | — |
| Başarısız denemeler | **VAR.** `sonuclar/basarisiz-onkayitlar.md` (A: H05/H07/H09/H13; B: tutmayan K5 maddeleri B1–B8; C: bayraklar; D: ikincil; E/F) | H03, H06, H12 analiz kartıdır, öldürülmedi. Brief'teki "öldürülen H03" ifadesi düzeltilmeli (bkz. Belirsizlikler 12) |
| Kalibrasyon (R-T) | **VAR.** [T] `kollar.*.katman_RT` (ECE ham → R-T: H10 0,0621 → 0,0179; H08c 0,0530 → 0,0168) | Karar vermez |
| Replikasyon tablosu | **VAR.** `replikasyon.md`, `replikasyon-colab.md`, `-test_auc.md`, `-test_acc_son.md` | — |

---

## Belirsizlikler/Eksikler

**Protokol beyanları (yazıma girmesi zorunlu):**
1. **Test bir kez açıldı:** 2026-10-11 21:12:03, `ikinci_acilis: null` (`k5_test.KILIT`). P6 özdeşliği 35/35 (`ozdeslik_p_maks_fark = 0`). C-test kol başına bir kez açıldı. `--kip c1` analizi resmî test tahminlerini KÇ için ön-kayıtlı olarak yeniden yükledi; sayılar aynı. Test sonrası kol, seed ya da eşik eklenmedi.
2. **Seed sayısı 5** {0..4}. MDE ≈ 3 puan (KÇ güç beyanı [ON] §4.3). Bootstrap p'nin çözünürlüğü 1/2001; Holm boot 0,002 = 4 × 0,0005.
3. **Ön-kayıt zinciri ve tarihler:**
   - K5-ONKAYIT onayı 18:14 (KARAR 10).
   - Ek b 18:55 (ham Δ_S − Δ_N kuralı; val'de iki ölçü önceden görülmüştü, M1 beyanı [ONb] §1).
   - H08 test kuralı `SIRALAMA.md:626-634` (09:42).
   - Colab tekrar kapısı `SIRALAMA.md:636-647` (11:39).
   - Dondurma SHA256 `c984fd0a…`.
4. **C1 uygulama düzeltmesi (v2/v3):** test sonrası, C1 öncesi `c_rakip_kos.py`, `c_bolme.py`, `c_bolme_sar.py` değişti (INFO yaması, tabakalı duman alt kümesi). Bu yüzden `k5_dondurma.json → dosya_sha256` içindeki bu üç dosyanın SHA'ları güncel dosyalarla **uyuşmaz**: dondurmada `e19fa3d8…` (c_bolme.py), düzeltme sonrası `6759e808…`.
   - Tarif, kural ve veri değişmedi (`hipotezler/K5-C1-UYGULAMA-NOTU.md`).
   - `k5_test.py` SHA'sı değişmedi (`13a71cf1…`).
   - Makalede "C1 v3 düzeltmesi" olarak beyan edilir.
   - Küçük tutarsızlık: `c_rakip_kos.py:58` yorumunda "K5-ONKAYIT-b §4" geçiyor, ama ONKAYIT-b'de §4 yok (§1–3 var).
5. **Rakibin C-test'i her epoch değerlendirmesi** (resmî kod davranışı; seçimde kullanılmaz) protokol notu olarak yazılır ([ONb] §2).

**Yazılım/ortam tutarsızlıkları (yazımda düzeltilmeli):**

6. `rakip/KURULUM.md` §1 ve §3 (satır 28, 76-77) Colab için `torch==2.5.1` pinini ve "TF32 kapatılmadı" bilgisini veriyor. **Gerçek ölçüm ortamı farklıdır:**
   - Ortam: torch 2.11.0+cu130, Python 3.13.15, sklearn 1.6.1, numpy 2.1.3. Defter "Colab'ın kendi torch'u, pin yok" (`colab/hat/defter_uret.py:124`; `SIRALAMA.md:637`).
   - TF32 **kapalı**, `fp32_kos.py` ile (`fp32_durum.json`). Yöntem bölümü bu gerçek ortamı yazmalı.
   - Not: `fp32_durum.json → bitis.cudnn.conv.fp32_precision = "none"`, `allow_tf32 = false`. Başlangıç değeri "ieee"; torch'un yeni/eski API etkileşimi gibi görünüyor. Ortam düzeyinde `NVIDIA_TF32_OVERRIDE=0` ayrıca var.
   - `orkestrasyon_durumu.md:198`: torch 2.5.1 ↔ 2.11 CPU RNG akışları bit bit aynı.
7. `data/data_manifest.md` "64/128/224 px indirilmedi" diyor (satır 12-27, 166) ve "C 224 indirilmedi" diyor (211). Bu bilgi **eski**: dosyalar mevcut ve MD5'leri doğrulanmış (`data/medmnist/indirme_dogrulama.json`, `data/dermamnist_c/`).
8. `notlar/literatur-cipa.md` §5 (satır 205) "DermaMNIST-C kapsam dışı" diyor. Bu bilgi KULLANICI KARARI 2026-10-10 ([DM] 178-184, 312-315) ile geçersizleşti.
9. Abhishek vd.'nin görüntü bazlı sızıntı sayıları (train–test 886) bizim tanımımızla (693) yeniden üretilemedi. Lezyon sayıları (641) bire bir aynı ([DM] 144-147). Makalede tanım farkı yazılmalı.

**Kapsam dışı / eksik deneyler (sınırlılık; ek koşum YOK, KARAR 12):**

10. **Rakibe D4 hakkı verilmedi** (M3 = HAYIR, KARAR 9).
    - **H10 + D4 koşulmadı**; H10 ↔ H08c kıyası omurga ile augmentasyonu karıştırır.
    - **C2 koşulmadı** (KARAR 11).
    - H11 C1'e alınmadı.
    - DINOv2-B tam ince ayar (K = 12) ve H10 K = 2 koşulmadı.
    - Yalnız D2 ablasyonu ve H04 yalnız-nv ablasyonu da koşulmadı.
11. **Tek donanım sınıfı** (L4). K4 seçimi Tur 1–3'te yerel GTX 1650'de yapıldı. Başka GPU'da ya da AMP/TF32 açıkken sınanmadı. Paralel Colab süreleri yalnız maliyettir (R6, KARAR 8).
12. Brief/taslakta "Öldürülen/kapanan kartlar (H03, H05, H07, H09, H13)" deniyor. `basarisiz-onkayitlar.md:20`'ye göre **H03 öldürülmedi, bir analiz kartıdır** (H06 ve H12 de öyle). Öldürülenler H05, H07, H09 ve H13'tür. "Başarısız denemeler" bölümünde H03 "test'e gitmeyen analiz kartı" diye anılmalı.
13. **T4 AUC 0,9904 > 0,99 bayrağı denetlenmedi** (KARAR 12). T4 yalnız ek tabloda yer alır.
14. HAM10000'de hasta kimliği yok. Lezyon-ayrıklık var, **hasta-ayrıklık denetlenemez** ([DM] 288).

**Kaynakça boşlukları:** `notlar/references-cekirdek.bib` içinde gerekçelerde atıf alan şu kaynaklar **yok**:
- Tupper & Gagné 2024 (arXiv:2403.09828)
- Guo vd. 2017 (sıcaklık ölçekleme)
- Menon vd. 2021 (logit ayarı)
- Balestriero vd. 2022
- Perez vd. 2018
- Byrd & Lipton 2019
- Holm 1979
- Efron (bootstrap)

Literatür ajanı bu künyeleri doğrulamalıdır.

**Test'e yeniden bakma kısıtı (bağlayıcı):** Aşağıdaki öneriler test tahmin dosyalarından yeni sayı türetir. Hepsi "**test sonrası keşifsel**" etiketiyle ve ayrı bölümde verilmelidir ([ON] §6.3). Hiçbiri karar ya da tez taşımaz ve hepsi kullanıcı onayına bağlıdır:
- (a) C0 kol başına seed değerleri;
- (b) seed düzeyi S/N;
- (c) sınıf-duyarlılık farklarının GA'ları;
- (d) test tabanlı nitel panel;
- (e) T4'ün C0/N AUC'si.

Önerilen yol: figürleri dondurulmuş JSON alanlarıyla (nokta + GA, ort ± std) ve val verisiyle (nitel panel, hata kırılımı) kurmak.

**Kullanıcıya sorulacak eksikler (orkestratör için):**
- Nitel panel val'den mi kurulsun (önerilen), yoksa test sonrası keşifsel test paneli mi?
- C0 figüründe seed noktaları gerekli mi? Gerekliyse test sonrası keşifsel türetme gerekir; aksi halde ort ± std yeterli.
- Eksik baseline: rakip + D4 ve H10 + D4 yok. Karar 12 gereği eklenmeyecek ve sınırlılık olarak kalacak; onay teyidi.
