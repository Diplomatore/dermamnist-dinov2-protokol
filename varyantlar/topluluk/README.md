# topluluk — H13 (`h13_h02c_h10_lpo`) ve ikincil topluluk satırları (`h13x_*`)

**Ebeveynden fark (tek cümle):** seed s'de H10_s'nin kayıtlı val olasılıklarının logaritması, aynı seed'in H02c_s
olasılıklarının logaritmasıyla ½ ; ½ ortalanır (1e-12 kırpma), softmax ile normalize edilir ve argmax alınır.
Eğitim, çıkarım ve GPU yok (kart `hipotezler/H13.md`).

| Dosya | İş |
|---|---|
| `topluluk.py` | Birleştirici. `--uyeler` (ilk üye = referans; H13'te H10_s), `--agirlik`, `--birlesim logolasilik\|olasilik`, `--kirp 1e-12`, `--ozdeslik` (tek üye modu, öldürme c1), `--seed-denet hepsi\|ilk\|yok`. Çıktı: `metrics.json` (`ortak/degerlendir.py` anahtarları + üye/kazanç G + çeşitlilik + C-val + özdeşlik alanları; `test_` yok), `tahmin_val.npy`, `etiket_val.npy`, config `uyeler` (run_id, ebeveyn, donanım, SHA256) |
| `tur5_kos.sh` | Tur-5 koşum sürücüsü (kosum.py foreground, CPU, satır başına ≈ 5 s): `a1` · `a3` · `katman` (H04 `h13k_h13_logitayar`, R-T `rt_h13_sicaklik`) · `ikincil` (`h13x_*`) |

Üyeler: `runs_colab/` = Colab Drive `<DRIVE_KOK>/3-1/runs` yerel arşiv kopyası (`scripts/colab_arsiv_kopya.py`;
json/npy/log + damga; `.pth`/`.npz` ve adında `test` geçen dosyalar HARİÇ; SHA256 `runs_colab/ARSIV_MANIFEST.json`).

Koşum (WS kökü): `bash varyantlar/topluluk/tur5_kos.sh a1` → kapı → `a3` → kapı → `katman` → `ikincil`.
Kapılar: `beklenti_kontrol.py --kuyruk tur5-h13|tur5-h13k|tur5-h13rt|tur5-h13x --asama 1|3`.
Ölçüm: `env/.venv/Scripts/python.exe -I scripts/tur5_h13_olcum.py` → `sonuclar/tur5_h13_olcum.json`.

Denetimler (öldürme c/d): tek üye modu argmax 1.003/1.003 ve max|Δp| ≤ 1e-6 · üye etiketleri = npz val etiketi ·
H10 üyesi `config.ebeveyn = h02_r18in_ft224_colab:s<s>` · satır toplamı 1 ± 1e-6 · NaN/inf → rc=3.
