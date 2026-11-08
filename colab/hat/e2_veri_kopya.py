#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
e2_veri_kopya.py — E2 (H15) ETİKETSİZ BCN model girdisini Drive'a taşır ve geri okuyarak SHA256 doğrular (şef, a0'dan SONRA,
drive_paketle.py'den ÖNCE ya da sonra; paket dosyalarına dokunmaz).

Kopyalanan (kaynak WS → hedef Drive):
  data/bcn20000/e2_girdi/bcn_e2_merkez_kare_224.npz   → yz/3-1/veri_dis/bcn20000/e2_girdi/   (≈ 1,2 GB)
  data/bcn20000/e2_girdi/bcn_e2_cerceve_fark_224.npz  → aynı yer (yalnız kırpılan satırlar)
  data/bcn20000/e2_girdi/bcn_e2_kal_224.npz           → aynı yer (8 HAM kalibrasyon görüntüsü)
  sonuclar/k5_c1_sonuc.json (+ .sha256)               → yz/3-1/sonuclar/ YALNIZ yoksa (KİLİT satırının K-d'si donmuş C1
                                                        değerleriyle de karşılaştırsın; var ve farklıysa EZİLMEZ, uyarı)
`veri_dis/` drive_paketle.py'nin YÖNETTİĞİ dizinlerden değildir (paketleme onu silmez). Etiket (tanı/lezyon) TAŞINMAZ.
Doğrulama: her npz kopyası Drive'dan geri okunur; SHA256 = kuyruk/e2_bcn_beklenen.json (aksi rc 2). Var ve aynıysa yeniden
yazılmaz. Drive for desktop yüklemesi arka planda sürer: Colab'da defter hücre 8 aynı SHA256'yı yeniden denetler.
Kullanım (WS kökü): PYTHONIOENCODING=utf-8 <uv-python> colab/hat/e2_veri_kopya.py [--drive "<DRIVE_KOK>"] [--kuru]
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
WS = Path(__file__).resolve().parents[2]
BEKLENEN = WS / "kuyruk" / "e2_bcn_beklenen.json"
KAYNAK = WS / "data" / "bcn20000" / "e2_girdi"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for parca in iter(lambda: f.read(1 << 22), b""):
            h.update(parca)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive", default="<DRIVE_KOK>")
    ap.add_argument("--kuru", action="store_true")
    a = ap.parse_args()
    if not BEKLENEN.exists():
        raise SystemExit(f"HATA: {BEKLENEN} yok — önce a0 (scripts/e2_bcn_hazirla.py --asama hepsi)")
    B = json.loads(BEKLENEN.read_text(encoding="utf-8"))
    hedef = Path(a.drive) / "3-1" / "veri_dis" / "bcn20000" / "e2_girdi"
    rc, rapor = 0, {}
    for ad, bilgi in B["dosyalar"].items():
        k, h = KAYNAK / ad, hedef / ad
        if sha256(k) != bilgi["sha256"]:
            raise SystemExit(f"HATA: yerel {ad} SHA256 beklenenle aynı değil (a0 sonrası değişmiş?)")
        if a.kuru:
            print(f"KOPYALANACAK {k} → {h} ({k.stat().st_size:,} B)")
            continue
        h.parent.mkdir(parents=True, exist_ok=True)
        if not (h.exists() and sha256(h) == bilgi["sha256"]):
            tmp = h.with_name(h.name + ".tmp")
            shutil.copyfile(k, tmp)
            tmp.replace(h)
        s = sha256(h)
        rapor[ad] = {"bayt": h.stat().st_size, "sha256": s, "eslesti": s == bilgi["sha256"]}
        if s != bilgi["sha256"]:
            rc = 2
        print(f"{ad}: {rapor[ad]['bayt']:,} B · SHA256 {s[:16]}… {'EŞLEŞTİ' if s == bilgi['sha256'] else 'UYUŞMADI'}")
    for ad in ("k5_c1_sonuc.json", "k5_c1_sonuc.json.sha256"):
        k, h = WS / "sonuclar" / ad, Path(a.drive) / "3-1" / "sonuclar" / ad
        if a.kuru:
            print(f"{'VAR' if h.exists() else 'KOPYALANACAK'} {h}")
            continue
        if not h.exists():
            shutil.copyfile(k, h)
            print(f"{ad} → Drive sonuclar (yoktu)")
        elif sha256(h) != sha256(k):
            print(f"UYARI: Drive'daki {ad} yereldekinden farklı — EZİLMEDİ (şef incelesin)")
    if not a.kuru:
        (hedef / "KOPYA.json").write_text(json.dumps({"beklenen_sha256": sha256(BEKLENEN), "dosyalar": rapor}, ensure_ascii=False,
                                                     indent=1), encoding="utf-8")
        print("TAMAM" if rc == 0 else "HATA: SHA256 uyuşmazlığı (kopyayı yinele)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
