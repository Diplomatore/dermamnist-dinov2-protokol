#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
paket_hazirla.py — Colab (hedef D) için Drive'a yüklenecek kod paketini üretir: WS/colab/3-1-paket.zip

İçerik (zip kökü = Drive'daki çalışma kökü <DRIVE_KOK>/3-1-dermamnist-dinov2):
  scripts/            deney betikleri (rakip_kos.py, veri_hazirla.py, ...)
  rakip/              resmî rakip kodu (MedMNIST/experiments @ 70b6b3a; .git HARİÇ, baytlar aynen — CRLF korunur)
                      + rakip/KURULUM.md, rakip/YAMALAR.md (varsa)
  kuyruk/             k3-baseline.jsonl + k3-baseline.beklenti.json (ön-kayıt)
  orkestra/           SK kopyası: kosum.py, kuyruk.py, beklenti_kontrol.py, leaderboard.py, replikasyon_kapisi.py
                      (dashboard_guncelle.py BİLEREK yok: Colab'da HERE.parents[3] → /content/drive altına yazardı;
                       kuyruk.py onu yalnız dosya varsa çağırır)
  env/requirements.lock · data/data_manifest.json · colab/k3_baseline.ipynb · colab/OKU-BENI.md
  PAKET.json          sürüm (içerik özeti) + dosya SHA256 listesi — defter hücre 1 sürüm değişince yeniden açar
HARİÇ: data/*.npz ve tüm data/ (manifest dışında), env/.venv, runs/, runs_yerel_duman/, __pycache__, *.pth

Kullanım (WS kökünden ya da herhangi bir yerden; uv Python 3.11):
  PYTHONIOENCODING=utf-8 <uv-python> colab/paket_hazirla.py
"""
import hashlib
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WS = Path(__file__).resolve().parents[1]
SK = WS.parents[1] / ".claude" / "skills" / "konudan-makale" / "scripts"
CIKTI = WS / "colab" / "3-1-paket.zip"
ORKESTRA = ["kosum.py", "kuyruk.py", "beklenti_kontrol.py", "leaderboard.py", "replikasyon_kapisi.py"]
HARIC_PARCA = {"__pycache__", ".git", ".venv"}
HARIC_UZANTI = {".pyc", ".pth", ".pt", ".ckpt", ".npz", ".tmp"}


def topla():
    girdiler = []  # (kaynak yol, zip içi ad)

    def agac(kok: Path, onek: str):
        for p in sorted(kok.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(kok)
            if HARIC_PARCA & set(rel.parts) or p.suffix in HARIC_UZANTI:
                continue
            girdiler.append((p, f"{onek}/{rel.as_posix()}"))

    agac(WS / "scripts", "scripts")
    agac(WS / "rakip", "rakip")
    for ad in ("k3-baseline.jsonl", "k3-baseline.beklenti.json"):
        girdiler.append((WS / "kuyruk" / ad, f"kuyruk/{ad}"))
    for ad in ORKESTRA:
        girdiler.append((SK / ad, f"orkestra/{ad}"))
    girdiler.append((WS / "env" / "requirements.lock", "env/requirements.lock"))
    girdiler.append((WS / "data" / "data_manifest.json", "data/data_manifest.json"))
    girdiler.append((WS / "colab" / "k3_baseline.ipynb", "colab/k3_baseline.ipynb"))
    girdiler.append((WS / "colab" / "OKU-BENI.md", "colab/OKU-BENI.md"))
    eksik = [str(k) for k, _ in girdiler if not k.exists()]
    if eksik:
        raise SystemExit("HATA: eksik dosya(lar):\n  " + "\n  ".join(eksik))
    return girdiler


def main():
    girdiler = topla()
    dosyalar = {ad: hashlib.sha256(k.read_bytes()).hexdigest() for k, ad in girdiler}
    surum = hashlib.sha256(json.dumps(dosyalar, sort_keys=True).encode()).hexdigest()[:12]
    meta = {"surum": surum, "olusturma": datetime.now().isoformat(timespec="seconds"),
            "konu": "3.1 DermaMNIST", "kuyruk": "k3-baseline",
            "rakip": "MedMNIST/experiments@70b6b3a7ad7afddff1df2a3b735235830fbdb142", "dosyalar": dosyalar}
    CIKTI.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(CIKTI, "w", zipfile.ZIP_DEFLATED) as z:
        for k, ad in girdiler:
            z.write(k, ad)
        z.writestr("PAKET.json", json.dumps(meta, ensure_ascii=False, indent=1))
    # doğrulama: zip yeniden okunur, her dosyanın özeti tutmalı
    with zipfile.ZipFile(CIKTI) as z:
        bozuk = [ad for ad, h in dosyalar.items() if hashlib.sha256(z.read(ad)).hexdigest() != h]
        assert not bozuk, f"zip doğrulaması başarısız: {bozuk}"
        adlar = z.namelist()
    assert not any(a.endswith(".npz") or "/.venv/" in a or a.startswith("runs") for a in adlar)
    print(f"yazıldı: {CIKTI}")
    print(f"sürüm {surum} · {len(adlar)} dosya · {CIKTI.stat().st_size:,} bayt ({CIKTI.stat().st_size / 1024:.1f} KB)")
    for ust in sorted({a.split('/')[0] for a in adlar}):
        print(f"  {ust:<10} {sum(1 for a in adlar if a.split('/')[0] == ust)} dosya")
    return 0


if __name__ == "__main__":
    sys.exit(main())
