#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
drive_paketle.py — Colab hattı paketini Google Drive for desktop klasörüne yazar (<DRIVE_KOK>\\3-1\\) ve PAKET.json üretir.

Paket (Drive çalışma kökü yz/3-1/; Colab'da <DRIVE_KOK>/3-1):
  scripts/      WS/scripts/*            (yerel kod, DEĞİŞMEDEN)
  varyantlar/   WS/varyantlar/**        (yerel kod + README + KOMUTLAR.md + logit_ayari/katlar.json)
  rakip/        WS/rakip/** (.git HARİÇ; MedMNIST/experiments@70b6b3a baytları aynen — CRLF korunur, blob SHA1 köprüde doğrulanır)
  orkestra/     SK: kosum.py, kuyruk.py, beklenti_kontrol.py, leaderboard.py, replikasyon_kapisi.py (dashboard_guncelle.py YOK)
  data/         data_manifest.json + splits/*.csv + meta/*.csv  (npz YOK — Colab'da Zenodo'dan iner)
  colab/        WS/colab/hat/: fp32_kos.py, erteleme_kapisi.py, oz_sinama.py, defter_yardim.py, oturum_sar.py, veri_listesi.json, yerel_ref/
  kuyruk/       kc1-tekrar, kc2-tur4-h12/-h10/-h11, k5-test, k5-c1-A/-B, k5-c2 (.jsonl + .beklenti.json), gruplar.json  (KALICI; başlamış kuyruk EZİLMEZ)
  OKU-BENI.md, PAKET.json
  ../3-1-colab-<SEKME>.ipynb   (yz/ kökünde; v2 paralel sekme defterleri = defter_uret.py çıktısı colab/hat/3-1-colab-*.ipynb)
  ../3-1-colab.ipynb           v1 tek defter (kc1-tekrar oturumu) — YAZILMAZ: Colab açık sekmenin çıktılarını otomatik kaydeder;
                               ezmek açık oturumla çakışır. PAKET.json'daki "defter" kaydı önceki paketten aynen taşınır.
Defter koruması: yz/3-1-colab-<S>.ipynb Drive'da varsa ve ne yerel kopyayla ne de önceki PAKET'te kayıtlı bir sürümle
aynıysa (Colab'da açılıp kaydedilmiş) EZİLMEZ — uyarı + çıkış kodu 3 (--defter-ez ile zorlanır; yalnız sekme kapalıyken).
HARİÇ: *.npz *.npy *.pth __pycache__ .git env/.venv runs/ (runs/, sonuclar/, env/, durum/ Colab'da oluşur, Drive'da kalıcıdır).
PAKET.json: her dosyanın SHA256'sı, baytı, yerel kaynağı ve kopyanın yerel kaynakla eşleşmesi (Drive'dan geri okunarak).

Kullanım (WS kökünden): PYTHONIOENCODING=utf-8 <uv-python> colab/hat/drive_paketle.py [--drive "<DRIVE_KOK>"]
"""
import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
WS = Path(__file__).resolve().parents[2]
SK = WS.parents[1] / ".claude" / "skills" / "konudan-makale" / "scripts"
HAT = WS / "colab" / "hat"
ORKESTRA = ["kosum.py", "kuyruk.py", "beklenti_kontrol.py", "leaderboard.py", "replikasyon_kapisi.py"]
COLAB = ["fp32_kos.py", "erteleme_kapisi.py", "oz_sinama.py", "defter_yardim.py", "oturum_sar.py", "veri_listesi.json"]
KUYRUKLAR = ["kc1-tekrar", "kc2-tur4-h12", "kc2-tur4-h10", "kc2-tur4-h11",
             "k5-test", "k5-c1-A", "k5-c1-B", "k5-c2", "k5-c1-A-v2", "k5-c1-A-v3"]  # K5 (2026-10-11; colab/hat/k5_kuyruk_yaz.py)
HARIC_PARCA = {"__pycache__", ".git", ".venv"}
HARIC_UZANTI = {".pyc", ".pth", ".pt", ".ckpt", ".npz", ".npy", ".tmp"}
YONETILEN = ["scripts", "varyantlar", "rakip", "orkestra", "colab", "data"]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def kod_girdileri():
    g = []  # (kaynak, paket içi ad)

    def agac(kok: Path, onek: str):
        for p in sorted(kok.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(kok)
            if HARIC_PARCA & set(rel.parts) or p.suffix in HARIC_UZANTI:
                continue
            g.append((p, f"{onek}/{rel.as_posix()}"))

    agac(WS / "scripts", "scripts")
    agac(WS / "varyantlar", "varyantlar")
    agac(WS / "rakip", "rakip")
    for ad in ORKESTRA:
        g.append((SK / ad, f"orkestra/{ad}"))
    g.append((WS / "data" / "data_manifest.json", "data/data_manifest.json"))
    for alt in ("splits", "meta"):
        for p in sorted((WS / "data" / alt).glob("*.csv")):
            g.append((p, f"data/{alt}/{p.name}"))
    for ad in COLAB:
        g.append((HAT / ad, f"colab/{ad}"))
    for p in sorted((HAT / "yerel_ref").glob("*.json")):
        g.append((p, f"colab/yerel_ref/{p.name}"))
    return g


def kuyruk_girdileri():
    g = [(WS / "kuyruk" / "gruplar.json", "kuyruk/gruplar.json")]
    for ad in KUYRUKLAR:
        g.append((WS / "kuyruk" / f"{ad}.jsonl", f"kuyruk/{ad}.jsonl"))
        g.append((WS / "kuyruk" / f"{ad}.beklenti.json", f"kuyruk/{ad}.beklenti.json"))
    return g


OKU_BENI = """# yz/3-1 — Colab hattı (3.1 DermaMNIST)

Bu klasörü şef (Claude) yazar. **Elle değiştirme.**

Defterler (`yz/` içinde):
- `3-1-colab.ipynb` — v1 tek defter (kc1-tekrar). Yalnız o oturum koparsa yeniden açılır.
- `3-1-colab-A.ipynb`, `-B`, `-C` — v2 paralel sekmeler. Hepsi aynı kod; yalnız ilk hücre (SEKME, KUYRUK) farklı.
- K5: `3-1-colab-K5-test.ipynb` (resmî test; tek sekme), `3-1-colab-K5-C1-A.ipynb` + `-K5-C1-B.ipynb` (C1; test SONRASI, iki sekme),
  `3-1-colab-K5-C2.ipynb` (C2; C1 bitince tek sekme). Her biri şef izni ister. **Aynı anda en çok 2 Colab oturumu.**

Her sekme için:
1. Drive'da defter dosyasına sağ tıkla → *Birlikte aç* → *Google Colaboratory* (her defter ayrı tarayıcı sekmesinde).
2. *Çalışma zamanı* → *Çalışma zamanı türünü değiştir* → **L4 GPU** → *Kaydet*. Başka GPU seçme, defter durur.
3. İlk hücreyi DEĞİŞTİRME. *Çalışma zamanı* → *Tümünü çalıştır*. Drive izni sorulursa aynı hesapla izin ver.
4. Sekmeleri açık tut, bilgisayar uyumasın. Bir sekme koparsa yalnız o defteri yeniden aç ve *Tümünü çalıştır*
   (≈ 10 dk beklemen gerekebilir; aynı kuyruğu iki sekme koşamaz). Bitmiş koşumlar atlanır.
   Bittiğinde şefe "bitti" demen yeterli; dosya indirmen gerekmez.

Klasörler: `runs/` koşumlar (+ `BILDIRIM_<oturum>.log`) · `kuyruk/` kuyruklar ve ön-kayıtlı beklentiler ·
`sonuclar/` kapı raporları · `env/` ortam kayıtları (oturum başına) · `durum/` canlı kalp atışı
(`canli_<oturum>.json`; v1: `canli.json`) ve hata günlükleri (`hata_*.log`).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default="<DRIVE_KOK>")
    ap.add_argument("--kuru", action="store_true", help="yalnız listele, yazma")
    ap.add_argument("--defter-ez", action="store_true", help="Colab'da kaydedilmiş v2 defterini de ez (sekme KAPALIYKEN)")
    a = ap.parse_args()
    yz = Path(a.drive)
    hedef = yz / "3-1"
    if not yz.exists():
        raise SystemExit(f"HATA: Drive klasörü yok: {yz}")
    kod, kuy = kod_girdileri(), kuyruk_girdileri()
    eksik = [str(k) for k, _ in kod + kuy if not k.exists()]
    defter = HAT / "3-1-colab.ipynb"   # v1 (yalnız kayıt; Drive'a yazılmaz)
    sekme_defterleri = sorted(HAT.glob("3-1-colab-*.ipynb"))
    if not defter.exists():
        eksik.append(str(defter))
    if not sekme_defterleri:
        eksik.append(str(HAT / "3-1-colab-<SEKME>.ipynb (önce defter_uret.py)"))
    if eksik:
        raise SystemExit("HATA: eksik kaynak:\n  " + "\n  ".join(eksik))
    if a.kuru:
        for k, ad in kod + kuy:
            print(ad)
        print(f"{len(kod)} kod + {len(kuy)} kuyruk dosyası")
        return 0
    # başlamış kuyruğun dosyaları EZİLMEZ (ön-kayıt)
    for k, ad in kuy:
        d = hedef / ad
        q = Path(ad).name.split(".")[0]
        if d.exists() and sha256(d) != sha256(k) and (hedef / "runs" / "kuyruklar").exists() and \
                any((hedef / "runs" / "kuyruklar").glob(f"{q}*.baslat.json")):
            raise SystemExit(f"HATA: {ad} Drive'da farklı ve kuyruk BAŞLAMIŞ — ön-kayıt ezilemez")
    hedef.mkdir(parents=True, exist_ok=True)
    onceki = json.loads((hedef / "PAKET.json").read_text(encoding="utf-8")) if (hedef / "PAKET.json").exists() else {}
    bilinen = {v.get("sha256") for v in (onceki.get("defterler") or {}).values()}
    # v2 sekme defterleri: Colab'da açılıp kaydedilmiş bir kopya EZİLMEZ (açık sekmenin otomatik kaydıyla çakışır)
    defterler, korunan = {}, []
    for yerel in sekme_defterleri:
        s_ad = yerel.stem[len("3-1-colab-"):]
        d = yz / yerel.name
        if d.exists() and sha256(d) != sha256(yerel) and sha256(d) not in bilinen and not a.defter_ez:
            korunan.append(d.name)
        elif not d.exists() or sha256(d) != sha256(yerel):
            shutil.copyfile(yerel, d)
        nb = json.loads(yerel.read_text(encoding="utf-8"))
        ilk = "".join(nb["cells"][1]["source"])
        kq = next((x.split("'")[1] for x in ilk.splitlines() if x.startswith("KUYRUK =")), None)
        defterler[s_ad] = {"yol": f"yz/{yerel.name}", "kuyruk": kq, "sha256": sha256(yerel),
                           "drive_sha256": sha256(d), "yerel_eslesme": sha256(d) == sha256(yerel)}
    if korunan:
        print(f"UYARI: Colab'da kaydedilmiş defter(ler) EZİLMEDİ: {korunan} (sekme kapalıysa --defter-ez)")
    dosyalar, kuyruk_dosyalari = {}, {}
    for kaynak, ad in kod + kuy:
        d = hedef / ad
        d.parent.mkdir(parents=True, exist_ok=True)
        if not d.exists() or sha256(d) != sha256(kaynak):
            shutil.copyfile(kaynak, d)
        h_k, h_d = sha256(kaynak), sha256(d)
        try:
            kaynak_rel = kaynak.relative_to(WS).as_posix()
        except ValueError:
            kaynak_rel = "SK/" + kaynak.relative_to(SK).as_posix()
        kayit = {"sha256": h_d, "bayt": d.stat().st_size, "kaynak": kaynak_rel, "yerel_eslesme": h_k == h_d}
        (kuyruk_dosyalari if ad.startswith("kuyruk/") else dosyalar)[ad] = kayit
    # yönetilen dizinlerde pakette olmayan eski dosyaları sil (runs/kuyruk/sonuclar/env/durum'a DOKUNULMAZ)
    silinen = []
    for ust in YONETILEN:
        for p in (hedef / ust).rglob("*") if (hedef / ust).exists() else []:
            if p.is_file():
                rel = p.relative_to(hedef).as_posix()
                if rel not in dosyalar:
                    p.unlink(); silinen.append(rel)
    (hedef / "OKU-BENI.md").write_text(OKU_BENI, encoding="utf-8")
    # v1 defter (yz/3-1-colab.ipynb) YAZILMAZ; kaydı önceki paketten taşınır (yoksa yerel v1'den)
    v1 = dict(onceki.get("defter") or {"yol": "yz/3-1-colab.ipynb", "sha256": sha256(defter), "yerel_eslesme": True})
    v1["not"] = ("v1 tek defter (kc1-tekrar oturumu); drive_paketle v2'den beri Drive kopyasını YAZMAZ (Colab otomatik kaydı "
                 "çıktıları ekler → Drive SHA256 farklı olabilir; kod hücreleri yerel colab/hat/3-1-colab.ipynb ile aynı)")
    surum = hashlib.sha256(json.dumps({k: v["sha256"] for k, v in sorted(dosyalar.items())}).encode()).hexdigest()[:12]
    meta = {"surum": surum, "sema": 2, "onceki_surum": onceki.get("surum") if onceki.get("surum") != surum else onceki.get("onceki_surum"),
            "olusturma": datetime.now().isoformat(timespec="seconds"), "konu": "3.1 DermaMNIST",
            "hat": "Colab L4 (KULLANICI KARARI 6–7)", "rakip": "MedMNIST/experiments@70b6b3a7ad7afddff1df2a3b735235830fbdb142",
            "drive_kok": "yz/3-1", "colab_kok": "<DRIVE_KOK>/3-1",
            "defter": v1,
            "defterler": defterler,
            "paralel": {"oturum_dosyalari": ["durum/canli_<oturum>.json", "durum/canli_hata_<oturum>.txt",
                                             "durum/hata_<oturum>_<zaman>.log", "durum/veri_c_<oturum>.json",
                                             "runs/BILDIRIM_<oturum>.log", "env/ortam-colab-<oturum>.json",
                                             "env/paket-<oturum>.json"],
                        "v1_uyum": "oturum=None → durum/canli.json + runs/BILDIRIM.log (kc1 defteri, değişmeden)"},
            "haric": "npz/npy/pth, .git, .venv, __pycache__, runs/",
            "dosyalar": dosyalar, "kuyruk_dosyalari": kuyruk_dosyalari}
    (hedef / "PAKET.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    uyusmayan = [k for k, v in {**dosyalar, **kuyruk_dosyalari}.items() if not v["yerel_eslesme"]]
    print(f"PAKET sürüm {surum}: {len(dosyalar)} kod/veri dosyası + {len(kuyruk_dosyalari)} kuyruk dosyası → {hedef}")
    ozet_d = ", ".join(f"{v['yol']} ({v['kuyruk']})" for v in defterler.values())
    print(f"defterler → {ozet_d} · v1 {yz / '3-1-colab.ipynb'} YAZILMADI · silinen eski: {len(silinen)} · "
          f"yerel eşleşmeyen: {uyusmayan or 'yok'}")
    for ust in sorted({k.split('/')[0] for k in dosyalar}):
        print(f"  {ust:<11} {sum(1 for k in dosyalar if k.split('/')[0] == ust)} dosya")
    if uyusmayan:
        return 2
    return 3 if korunan else 0


if __name__ == "__main__":
    sys.exit(main())
