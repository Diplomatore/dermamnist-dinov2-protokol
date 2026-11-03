# varyantlar/tta — H07 (test-zamanı artırma, 8 dihedral görünüm)

**Ebeveynden farkı (tek cümle):** H02'nin seçilmiş-epoch ağırlıklarıyla val çıkarımı tek görünüm yerine D4'ün 8 görünümünün
softmax olasılıklarının aritmetik ortalamasıyla yapılır; eğitim, ağırlık, normalizasyon ve val bs (128) bit bit aynıdır.

- `tta.py` — ana betik (GPU, yalnız çıkarım). `--ebeveyn-varyant h02_r18in_ft224` (H07) ya da `baseline` (M3 yan kolu:
  rakibin `models.ResNet18`'i salt-okunur içe aktarılır, 28 px, (x/255 − 0,5)/0,5). Kimlik görünümü ebeveynin
  `tahmin_val.npy`'sini yeniden üretmezse (max|Δp| > 1e-4 ya da ACC farkı ≠ 0) rc = 3 (DUR).
- `tta_alt.py` — ablasyon (i): kaydedilmiş `tahmin_val_gorunum.npy`'den D2 / 90° alt kümelerinin ortalaması (CPU, ek GPU yok).
- D4 tanımı `varyantlar/ortak/dihedral.py` (H08 ile ortak). Ters-transpoz = flip(3)(rot90(x)); kartın "rot90(1)∘flip(3)" yazımı
  boru hattı sırasıyla okundu (diğer okuma transpozu tekrarlar, 7 ayrık eleman kalır). `oz_sinama()` 8 dönüşümün ayrık ve
  kapalı olduğunu her koşumda denetler.

Koşum (WS kökü, foreground, her satır < 1 dk):
```bash
PY='…\env\.venv\Scripts\python.exe'; SK=../../.claude/skills/konudan-makale/scripts
"$PY" $SK/kosum.py --kok . --varyant h07_h02_tta8 --seed S --hipotez H07 --ebeveyn <H02 s{S} run> --kuyruk tur3-h07 --asama 3 \
  --gpu-zorunlu --zaman-siniri-dk 5 -- "$PY" varyantlar/tta/tta.py --ebeveyn-varyant h02_r18in_ft224 --seed S --gorunum d4 --val-bs 128 --out "{RUN_DIR}/metrics.json"
CUDA_VISIBLE_DEVICES= "$PY" $SK/kosum.py --kok . --varyant h07a_h02_tta4d2 --seed S --hipotez H07 --ebeveyn <H02 s{S} run> --kuyruk tur3-h07a \
  --asama 3 -- "$PY" varyantlar/tta/tta_alt.py --kaynak-varyant h07_h02_tta8 --alt d2 --seed S --out "{RUN_DIR}/metrics.json"
```
Ölçüm: `env/.venv/Scripts/python.exe -I scripts/tur3_olcum.py --kart h07` → `sonuclar/tur3_h07_olcum.json`.
