# logit_ayari — H04 (`h04_h02_logitayar`), H06 (`h06_h01_logitayar`), R-T (`rt_*_sicaklik`)

**Ebeveynden fark (tek cümle):** ebeveynin kayıtlı val olasılıklarında karar `argmax p` yerine
`argmax [log p − τ·log π_train]` olur; τ ∈ {0; 0,1; …; 1} 5 seed için ortak, val içinde 5 katlı (lesion_id gruplu)
çapraz uydurmayla seçilir (H04 kartı §"τ seçim protokolü"). Eğitim/çıkarım yok, GPU yok.

| Dosya | İş |
|---|---|
| `logit_ayari.py` | H04/H06 (ve yan kollar `h04x_*`). `--katlar-yaz` bir kez, koşumdan önce → `katlar.json` |
| `katlar.json` | ortak kat bölmesi (StratifiedGroupKFold, RNG 20261011, grup = lesion_id). SHA256 `0db03edb…0193`; H04/H06/R-T aynı |
| `sicaklik.py` | R-T raporlama protokolü (SIRALAMA): seed başına T ∈ [0,5; 3,0], NLL ile, aynı katlarla OOF ECE-15 |
| `tur2_olcum.py` | kart ölçümleri (kesin tek yönlü eşleştirilmiş t + görüntü bootstrap 2.000 / RNG 0) → `sonuclar/tur2_cpu_olcum.json` |

Koşum (WS kökü, CPU; `--ebeveyn` = aynı seed'in ebeveyn run_id'si; komutlar `../KOMUTLAR.md` §T2):
```
python <SK>/kosum.py --kok . --varyant h04_h02_logitayar --seed S --hipotez H04 --ebeveyn <H02 run> --kuyruk tur2-h04 --asama 3 -- \
  env/.venv/Scripts/python.exe varyantlar/logit_ayari/logit_ayari.py --ebeveyn-varyant h02_r18in_ft224 \
  --ebeveyn-dosya tahmin_val.npy --tau-izgara 0:1:0.1 --kat 5 --kat-rng 20261011 --kisit 0.010 --seed S --out {RUN_DIR}/metrics.json
```
Denetimler: τ = 0 özdeşliği (`tau0_ozdeslik_fark` = 0, değilse rc=3), π sınıf sırası (`pi_sira_ok`), etiket hizası.
`metrics.json`'da `test_` anahtarı yok; tahmin_val.npy = OOF olasılıkları (leaderboard/bootstrap bunu okur).
