# varyantlar/k5 — K5 test geçişi, C1 ve C2 yardımcıları (hipotezler/K5-ONKAYIT.md; KULLANICI KARARI 10)

**Ebeveynden farkı (tek cümle):** yeni tarif yoktur. Donmuş K4 modelleri yalnız yeniden yüklenip çıkarım yapılır (`k5_cikarim.py`).
C1/C2'de K4 tarifleri bit düzeyinde aynı komutla koşar; tek fark veri dosyası ve bölmedir (`c_bolme.py` yaması).

| dosya | iş |
|---|---|
| `k5_cikarim.py` | Kaynak koşumu (best_model.pth / sonda_agirlik.npz / rakip tahminleri) yükler. Val'de **P6 özdeşliğini** sınar: argmax N/N ve maks \|Δp\| ≤ 1e-5. Ancak `--test-kilidi-acik` verilirse ve özdeşlik geçerse test tahmini yazar (`tahmin_test.npy`, `test_*`). Türler: `r18`, `vit_kismi` (H10; fp16 blok-8 token yolu aynen), `vit_tam` (H11), `sonda` (H01c/H12), `rakip` (baseline_c; yeniden çıkarım yok). |
| `c_bolme.py` | `ortak.veri` yaması: `npz_yolu`, `npz_dogrula`, `resmi_md5`, `etiket_tutarliligi`, `sizinti_maskesi` ve `val_meta` C1 (`data/dermamnist_c_resmipiksel_*.npz`) ya da C2 (`data/dermamnist_c/dermamnist_corrected_*.npz`) verisine yönlenir. Öznitelik önbelleği `data/oznitelik_<veri>` altına yazılır. |
| `c_bolme_sar.py` | Bir K4 betiğini aynı süreçte C verisiyle koşar: `c_bolme_sar.py --veri c1 -- <betik> <argümanlar>`. |
| `c_rakip_kos.py` | Rakibi (scripts/rakip_kos.py → MedMNIST/experiments@70b6b3a, dokunulmaz) C verisiyle koşar. Yalnız veri kökü ve süreç içi MD5 değeri değişir. |

İlgili betikler:
- `scripts/c_bolme_npz.py`: C1 verisini kurar ve denetler. **Yalnız resmî test geçişinden SONRA** koşulur.
- `scripts/k5_test.py`: analiz betiği (P5; `--kip kuru-val|test|c1|c2`).
- `scripts/k5_dondurma.py`: P4 dondurma dosyası.
- `scripts/k5_c_denetim.py`: P7 denetimi.
- Kuyruklar `colab/hat/k5_kuyruk_yaz.py` ile yazılır. Komutlar ve sıra için bkz. `varyantlar/KOMUTLAR.md` §K5.

## Nasıl koşulur
Colab'da kuyruk satırı olarak koşulur, elle koşulmaz. Yerel sınama (yalnız val, ölçüm değil):
```
env/.venv/Scripts/python.exe varyantlar/k5/k5_cikarim.py --tur r18 --kaynak 20261010-222930_h02_r18in_ft224_s0 --seed 0 --out <geçici>/metrics.json
```

## Yerel doğrulama (2026-10-11, GTX 1650, scratchpad; ölçüm değil)
Dört çıkarım yolu yerel kaynak koşumlarının `tahmin_val.npy` dosyasını **bit bit** yeniden üretti (argmax 1003/1003, maks \|Δp\| = 0):
- `r18`: H02 s0
- `vit_kismi`: t4 B-K4 1 epoch (önbellek yeniden hesaplandı)
- `vit_tam`: t4 S tam 1 epoch
- `sonda`: H01 s0

Ayrıca:
- `rakip` val yolu baseline_colab s0 üzerinde sınandı.
- C2 yaması üç yolda sınandı: R18 28 px, sonda alt küme 16 ve ViT kısmi alt küme 64 (CPU). C-test hiç açılmadı.
- `--alt-kume` ile `--test-kilidi-acik` birlikte verilemez (sınandı).
