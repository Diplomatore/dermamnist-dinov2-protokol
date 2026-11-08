# varyantlar/e2 — H15 / E2: BCN20000 dış merkez değerlendirmesi (yeniden eğitimsiz, kilitli; KULLANICI ONAYI H15, 2026-11-07)

**Ebeveynden farkı (tek cümle):** yeni tarif yoktur; arşivdeki 11 eğitilmiş kol (55 kol-seed) aynı yükleyici ve dönüşümlerle
yeniden yüklenir ve yalnız değerlendirme verisi değişir (iç test yerine BCN20000 figshare train, etiketsiz girdi, KİLİT altında).

| dosya | iş |
|---|---|
| `kollar.py` | 11 kol × 5 seed kaynak run_id'leri, tür (r18 / vit_kismi / vit_tam / sonda), veri (resmî / c1), rol, iç test arşivi. Tek kaynak. |
| `e2_cikarim.py` | `--kip duman / onucus / bcn`. Val özdeşliği (K-c Durum A/B/C), KİLİT ve ikinci çıkarım denetimi (K-f), BCN olasılıkları (`tahmin_bcn_{merkez_kare,cerceve}.npy`). Etiket okumaz. Yükleyiciler `k5/k5_cikarim.py` ile aynı (o dosya değişmedi). |
| `bcn_veri.py` | Etiketsiz model girdisi: SHA256 (dosya + içerik), Colab'da npz → npy açma (`ac`), a1 veri kapısı (`kapi`). |
| `cerceve.py` | S9: yazarların siyah çerçeve kırpıcısı (`ucuncu/image_cropper.py` @7ea35f5, bayt bayt) + ana 224 px yolu. |
| `ucuncu/` | Üçüncü taraf kod ve kaynağı (`KAYNAK.md`, CC BY 4.0). |

İlgili betikler: `scripts/e2_bcn_hazirla.py` (a0, yerel CPU), `scripts/e2_kilit.py` (a3 ilk satırı), `scripts/e2_bcn_analiz.py`
(analiz; `--kuru`), `colab/hat/e2_kuyruk_yaz.py`, `colab/hat/defter_uret_e2.py`, `colab/hat/e2_veri_kopya.py`.
Uygulama kararları: `hipotezler/H15-UYGULAMA-NOTU.md`.

## Nasıl koşulur
Colab'da `e2-degerlendirme` kuyruğunun satırı olarak koşulur, elle koşulmaz (defter `colab/e2-h15.ipynb`). Örnek satır:
```
python colab/erteleme_kapisi.py --esik 3.0 --baslangic-sa 0 --kuyruk-onek e2-degerlendirme --kural "…" -- \
  python colab/fp32_kos.py varyantlar/e2/e2_cikarim.py --kol h10 --kaynak 20261011-160211_h10_dinov2b_k4_ft224_s0 --seed 0 \
  --kip bcn --dis-kilit-acik --out {RUN_DIR}/metrics.json
```
Yerel sınama (ölçüm değil, BCN yok): `--cihaz cpu --cpu-izin --alt-kume 16` (+ vekil girdi için `--bcn-dizin/--beklenen/--kilit`,
öznitelik önbelleği için `E2_BCN_OZN_DIZIN`).

## Yerel doğrulama (2026-11-07, CPU, scratchpad; ölçüm değil; hiçbir BCN görüntüsü modele verilmedi)
- `--kip onucus` dört yolda `k5_cikarim.py` ile **bit bit aynı** `tahmin_val_yeniden.npy` üretti: r18 (yerel H02 s0, val 32),
  sonda (yerel H01 s0, val 1.003), vit_kismi ve vit_tam (rastgele başlı sahte ViT-S kaynakları, val 16), C1 yaması (r18, C-val 573).
- BCN yolu VEKİL girdiyle (DermaMNIST val'in ilk 96 görüntüsü, etiketsiz npz): r18 ve vit_kismi (`--bcn-parca 64`) BCN tahmini =
  aynı görüntülerin val tahmini (bit bit); sonda öznitelik önbelleği yeniden kullanımda bit bit aynı; KİLİT yok/uyuşmaz → rc 7;
  ikinci çıkarım → rc 8; Durum C → BCN açılmadı (rc 4); dejenere çıktı → rc 5; duman: kalibrasyon 8/8, 48 görüntü yazılmadı.
