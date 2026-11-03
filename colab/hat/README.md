# Colab hattı (Drive köprüsü) — KULLANICI KARARI 6–7

Yerel tariflerin Colab L4'te aynen koşulması; yerel kod değişmez, yalnız sarılır. Eski `colab/k3_baseline.ipynb` + `3-1-paket.zip` arşivdir.

| dosya | Colab'a gider | iş |
|---|---|---|
| `fp32_kos.py` | evet | her kuyruk komutunun önünde: TF32 kapalı (eski + yeni API), deterministik cuDNN, `{RUN_DIR}/fp32_durum.json`; betiği aynı süreçte (runpy) koşar |
| `erteleme_kapisi.py` | evet | H11 a3 erteleme kuralı (sayaç = 15,3 yerel + **bütün eşzamanlı Colab oturumlarının** L4-sa'i > 23,0 → rc=94); `--kendi-testi` |
| `oz_sinama.py` | evet | TF32 sayısal sınaması (fp64 referansına göre göreli hata) |
| `defter_yardim.py` | evet | paket kopyala + SHA256, bağlar, veri indirme, kuyruk/grup çözümü + izin + DEVAM, canlı kalp atışı, eşzamanlı çakışma bekçisi, tekrar özeti |
| `oturum_sar.py` | evet | (v2) `kuyruk.py`/`kosum.py`'yi aynı süreçte koşar, yalnız `bildir()`'i `runs/BILDIRIM_<oturum>.log`'a yönlendirir (orkestra kodu değişmez) |
| `veri_listesi.json` | evet | Zenodo URL + resmî MD5 + yerel SHA256 (MedMNIST 28/224, DermaMNIST-C 28/224) + önağırlık SHA256 |
| `yerel_ref/*.json` | evet | H02/H08 yerel duman metrics (özdeşlik referansı `--ozdeslik-ref`) |
| `kuyruk_yaz.py` | hayır | `kuyruk.py ekle` ile kc1-tekrar ve kc2-tur4-h12/-h10/-h11 kuyruklarını yazar |
| `defter_uret.py` | hayır | **v2** paralel sekme defterleri `3-1-colab-<SEKME>.ipynb` (`--sekmeler A=kc2-tur4-h10,B=kc2-tur4-h11,C=kc2-tur4-h12`); yalnız hücre 1 farklı (denetlenir) |
| `defter_uret_v1.py` | hayır | **v1** tek defter `3-1-colab.ipynb` (DONDURULMUŞ; kc1-tekrar oturumu bununla koşuyor; yeniden üretim bayt bayt aynı) |
| `drive_paketle.py` | hayır | `<DRIVE_KOK>\3-1\` paketini + `PAKET.json`'u yazar (başlamış kuyruğun dosyalarını ezmez; v1 defteri YAZMAZ; Colab'da kaydedilmiş v2 defterini ezmez) |

## Paralel sekmeler (v2, 2026-10-11)
- Defter ilk hücresi: `SEKME`, `KUYRUK` (tek kuyruk adı → paralel; grup adı → tek sekmede sırayla, v1 gibi). Hücre 2–10 bütün sekmelerde aynı.
- `OTURUM = <SEKME>-<YYYYmmdd-HHMMSS>-<4 hex>` (yeniden açılışta yenisi). Oturuma ayrılan Drive dosyaları:
  `durum/canli_<oturum>.json`, `durum/canli_hata_<oturum>.txt`, `durum/hata_<oturum>_<zaman>.log`, `durum/veri_c_<oturum>.json|.log`,
  `durum/veri_medmnist_<oturum>.json`, `runs/BILDIRIM_<oturum>.log`, `env/ortam-colab-<oturum>.json`, `env/paket-<oturum>.json`,
  `env/requirements-colab-<oturum>.lock`, `env/oz-sinama-<oturum>.json`, `sonuclar/leaderboard-colab-<KUYRUK>.*`.
  Kuyruğa göre zaten ayrık: `runs/kuyruklar/<ad>.*`, `kuyruk/<ad>-devamN.*`, `sonuclar/beklenti-<ad>-asama<k>.md`; koşum klasörleri
  zaman+varyant+seed ile benzersiz. Ortak ve salt okunur: `env/ortam-colab-ilk.json`, `kuyruk/*.jsonl|.beklenti.json`, `gruplar.json`.
- İzin: tek kuyruk, içinde bulunduğu grubun iznini devralır (`kc2-tur4-h10` → `kuyruk/kc2-tur4.izin.json`). Yeni izinli grup için
  `gruplar.json`'a isteğe bağlı `"izin_gereken": [...]` (kod değişmez).
- Çakışma bekçisi: başka bir sekmenin taze (< 600 s) ve bitmemiş canlı dosyası aynı kuyruğu kapsıyorsa hücre 5 ve 9 DURUR
  (iki sekmenin aynı kuyruğu koşması devam mantığında diğerinin koşumlarını yetim sayardı).
- Devam: yalnız o sekmenin kuyruk ailesi yetim damgalanır; diğer sekmelerin koşumlarına dokunulmaz.
- `canli_<oturum>.json`'daki `aktif` yalnız o sekmenin kuyruk ailesini listeler (runs/ ortak klasör).
- Hücre 10'daki `kosum.py durum` diğer sekmelerin koşan koşumlarını "ÖLDÜ damgasız" gösterebilir (PID'ler başka VM'de) — salt gösterim.
- v1 uyumu: `oturum=None` çağrıları (v1 defteri) değişmeden `durum/canli.json` + `runs/BILDIRIM.log` yazar; v1 defterindeki her `DY.*`
  çağrısının v2 imzasına bağlandığı ve v1 canlı JSON anahtarlarının birebir aynı olduğu sınandı.
- K5: aynı defter; `defter_uret.py --sekmeler A=<k5-kuyruk-1>,B=<k5-kuyruk-2>[,C=…]` → `drive_paketle.py` → `colab_izle.py --kuyruklar …`.

Yeniden üretim (WS kökünden, uv Python 3.11, `PYTHONIOENCODING=utf-8`):
`colab/hat/defter_uret.py` → `colab/hat/drive_paketle.py`.
İzleme: `scripts/colab_izle.py --kuyruklar kc2-tur4-h10,kc2-tur4-h11,kc2-tur4-h12` (v2) · `--grup kc1-tekrar` (v1).
Kendi-testleri: `scripts/colab_izle.py --kendi-testi` (v1 grup + v2 paralel kipi) · `colab/erteleme_kapisi.py --kendi-testi`.
