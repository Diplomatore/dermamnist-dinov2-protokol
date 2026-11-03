# makale/: kanıt defteri, tablo ve figür üreticileri

Makaledeki her sayı `evidence/evidence.json` içindeki bir kayıttan gelir. Her kayıtta kaynak dosya, alan ya da satır
bilgisi ve kip (val / test / C1) bulunur. Bu dizinde makale çalışma alanı (`<MAKALE_WS>`) yer alır. Deney çalışma
alanı (`DW`) ise deponun köküdür.

| betik | girdi | depodan çalışır mı |
|---|---|---|
| `evidence/scripts/build_tables.py` | `evidence/evidence.json` | evet → `evidence/tables/*.tex` |
| `evidence/scripts/build_derivations.py` | `evidence/evidence.json` | evet → `notes/derivations.md` |
| `figures/scripts/fig_*.py` | `sonuclar/`, `data/data_manifest.md`, `runs_meta/runs_colab/*/metrics.json` | evet → `figures/data/*.dat`. `fig_nitel.py` görüntü karolarını (`figures/tiles/`) gerektirir. Bu karolar DermaMNIST görüntüsü olduğu için depoda yok |
| `evidence/scripts/build_evidence.py` | dondurulmuş JSON/MD/CSV dosyaları ve makale notları (`notes/literature.md`, `review.md` …) | kısmen: makale notları depoda yok. Üretilmiş `evidence.json` depodadır ve kaynak özetleri `tools/dogrula.py` (B2) ile denetlenir |
| `evidence/scripts/lkb_bootstrap.py`, `c1_mel_bootstrap.py` | test tahmin dosyaları (`runs_colab/*/tahmin_test.npy`) | hayır: tahmin `.npy` dosyaları depoda yok. Çıktılar (`evidence/lkb/`, `evidence/c1mel/`) ve okudukları tahmin dosyalarının SHA256 değerleri depodadır |
| `paper/araclar/derle.sh`, `birlestir.py` | `sections/*.tex` (makale metni) | hayır: makale metni depoda yok; yazar bloğu yer tutucudur |

Kullanım (depo kökünden): `python makale/evidence/scripts/build_tables.py`.

Depo düzenine uyarlama olarak betiklerde `DW = WS.parent` ve `RC = runs_meta/runs_colab` tanımları yapıldı. Hangi
dosyaların uyarlandığı `HIJYEN_DEGISIKLIKLERI.json` dosyasında `elle_yama: true` ile işaretlidir.
