# varyantlar/e3 — H16 (E3): BCN ön-işleme hattı denetimi, resmî val

**Ebeveynden farkı (tek cümle):** H02c, H08c, H10 ve H11'in arşivdeki eğitilmiş resmî kolları yeniden eğitilmeden, aynı
oturumda hem MedMNIST+ val girdisinde (yerel) hem de aynı 1.003 HAM görüntüsünün ISIC özgün JPEG'inden BCN hattıyla
(`indir_bcn20000.geometri` + `adaylar()["PIL_BILINEAR"]`, 224 px) üretilmiş girdide (+ JPEG q_BCN, ölçek ×1,25 / ×0,8
duyarlılık kolları) çalıştırılır; değişen yalnız val girdisinin piksel kaynağı ve ön-işleme hattıdır.

Kart: `hipotezler/H16.md` (SHA256 `f3c6998196e06248…`; KULLANICI KARARI 15). Uygulama notu: `hipotezler/H16-UYGULAMA-NOTU.md`.

| dosya | iş |
|---|---|
| `e3_cikarim.py` | kol-seed başına çıkarım (kip duman / onucus / tam); hat kolları etiket-kör; K-c Durum A/B/C, K-e, K-f, K-g |
| `e3_kuyruk_yaz.py` | a0 bittikten ve beklenen dosyası donduktan sonra `kuyruk/e3-onisleme.jsonl`'ın 26 satırını yazar |
| `../../scripts/e3_ham_val_hazirla.py` | a0: indirme, q_BCN, 4 hat npz, kapılar, `kuyruk/e3_onisleme_beklenen.json` |
| `../../scripts/e3_onisleme_analiz.py` | `--kuru` · `--kilit-ac` · analiz (bir kez) |

## Nasıl koşulur (şef; WS kökü; `PY = env/.venv/Scripts/python.exe`, `KQ = <skill>/scripts/kuyruk.py`)
1. `PY -I KQ --kok WS kos e3-veri` (CPU + ağ; a0) — bitince `kuyruk/e3_onisleme_beklenen.json` ve
   `data/ham_val_isic/e3_a0_rapor.md`.
2. `PY -I varyantlar/e3/e3_kuyruk_yaz.py` → `kuyruk/e3-onisleme.jsonl` (26 satır).
3. `PY -I KQ --kok WS kos e3-onisleme --kademeli` (GPU; kuyruk venv Python'uyla başlatılmalı, `gpu_dogrula` torch ister).
4. `PY -I scripts/e3_onisleme_analiz.py --kilit-ac`, ardından `PY -I scripts/e3_onisleme_analiz.py` (BİR KEZ).

Yerel kod sınaması (ölçüm değil): `PY -I varyantlar/e3/e3_cikarim.py --kol h02c --seed 0 --kip onucus --cihaz cpu --cpu-izin
--alt-kume 20 --hat-dizin <vekil> --beklenen <vekil>/beklenen.json --out <geçici>/metrics.json`.

Lisans: `data/ham_val_isic/` (HAM10000 / ISIC, CC BY-NC) depoya girmez; WS `.gitignore` `data/`'yı dışlar, `kopyala.py` izin
listesi bu yolu içermez.
