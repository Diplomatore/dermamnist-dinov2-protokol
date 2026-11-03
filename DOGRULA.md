# Dondurulmuş sonuçların ve ön-kayıt zincirinin doğrulanması

Depo kökünde (yalnız Python standart kütüphanesi gerekir):
```bash
python tools/dogrula.py              # özet + yalnız sorunlu satırlar
python tools/dogrula.py --ayrinti    # her dosyanın satırı
```
Çıkış kodu `0` ise doğrulama geçmiştir.

## Denetim katmanları
| katman | ne denetlenir | zorunlu |
|---|---|---|
| Z1 | `sonuclar/*.sha256` yan dosyaları: `k5_test_sonuc.json`, `k5_c1_sonuc.json`, `k5_dondurma.json` | evet |
| Z2 | `sonuclar/k5_test.KILIT` → `scripts/k5_test.py` (test açılışındaki betik) | evet |
| Z3 | `sonuclar/k5_dondurma.json`: ön-kayıt (`K5-ONKAYIT.md`, `-b.md`), analiz ve C bölme betikleri, k5 kuyruk ve beklenti dosyaları, `katlar.json`. `k5_dondurma_ek_c1.json`: C1 veri beklentisi ve test sonucu özeti | evet |
| B1 | `makale/evidence/lkb`, `makale/evidence/c1mel` içindeki donmuş kaynak özetleri | bilgi |
| B2 | `makale/evidence/evidence.json` → `deney_meta.kaynak_sha256` | bilgi |
| B3 | `runs_meta/runs_colab/ARSIV_MANIFEST.json` → depodaki koşum JSON damgaları | bilgi |

Durum etiketleri:
- **GEÇTİ:** özet bayt düzeyinde eşleşiyor.
- **HİJYEN-EŞLEMESİ:** dosyada yalnız kişisel yollar yer tutucuya çevrildi. Kayıttaki eski özet
  `HIJYEN_DEGISIKLIKLERI.json` içindeki `onceki_sha256` ile, depodaki dosya ise `yeni_sha256` ile eşleşiyor.
- **BELGELİ SAPMA:** dondurmadan sonra belgelenmiş uygulama düzeltmesi var (`hipotezler/K5-C1-UYGULAMA-NOTU.md`):
  `varyantlar/k5/c_bolme.py`, `c_rakip_kos.py` ve `c_bolme_sar.py`. Bu üç dosyanın güncel özeti notun "Düzeltme sonrası
  dosya SHA256" tablosuyla eşleşmelidir. `kuyruk/gruplar.json`'daki değişiklik ise v2/v3 kuyruklarının eklenmesidir
  (dosya içi `_aciklama_k5_v2` / `_v3`).
- **DEPODA-YOK:** veri, tahmin `.npy` dosyaları veya makale notları bu depoya alınmadı. Bu durum hata sayılmaz.
- **UYUŞMAZ:** doğrulama başarısızdır.

Ön-kayıt zincirindeki dosyalar (Z1–Z3) depoya **bayt bayt** kopyalandı ve hiçbir temizlik uygulanmadı. Bu yüzden
`scripts/k5_test.py`, `scripts/k5_dondurma.py`, `sonuclar/k5_test_sonuc.json` ve `sonuclar/k5_c1_sonuc.json` içinde
Drive bağlama yolu (`G:/Drive'ım/yz/3-1/...`) özgün hâliyle kalmıştır.

`.gitattributes` bütün dosyalar için satır sonu dönüşümünü kapatır (`* -text`). Böylece Windows ve Linux'ta checkout
sonrasında özetler aynı kalır.

## Elle doğrulama (araç kullanmadan)
```bash
sha256sum sonuclar/k5_test_sonuc.json sonuclar/k5_c1_sonuc.json sonuclar/k5_dondurma.json
cat sonuclar/*.sha256
sha256sum scripts/k5_test.py hipotezler/K5-ONKAYIT.md     # k5_dondurma.json: k5_test_sha256, onkayit_sha256
```
PowerShell: `Get-FileHash -Algorithm SHA256 sonuclar\k5_test_sonuc.json`.
