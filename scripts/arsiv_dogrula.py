# -*- coding: utf-8 -*-
"""
arsiv_dogrula.py — runs_colab/ARSIV_MANIFEST.json'ın bağımsız doğrulaması (D5; K5 sonrası arşiv).

Denetimler (Drive'a dokunmadan; yalnız yerel):
 1. Manifest'teki her dosya yerelde var ve SHA256'sı manifest ile aynı.
 2. Manifest'teki koşum dizini sayısı = yereldeki koşum dizini sayısı (Drive'daki dizinlerin hepsi kopyalanmış).
 3. K5 dizinleri tam: k5_test_sonuc.json / k5_c1_sonuc.json `kosumlar` listelerindeki her dizin yerelde var ve
    test kollarında tahmin_test.npy + etiket_test.npy mevcut (içerik OKUNMAZ; yalnız varlık + manifest SHA256).
Kullanım (WS kökü): env/.venv/Scripts/python.exe -I scripts/arsiv_dogrula.py
"""
import hashlib
import json
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
ARS = WS / "runs_colab"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for parca in iter(lambda: f.read(1 << 20), b""):
            h.update(parca)
    return h.hexdigest()


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    man = json.loads((ARS / "ARSIV_MANIFEST.json").read_text(encoding="utf-8"))
    hata = []
    for d in man["dosyalar"]:
        q = ARS / d["yol"]
        if not q.exists():
            hata.append(f"eksik: {d['yol']}")
        elif sha256(q) != d["sha256"]:
            hata.append(f"SHA256 farklı: {d['yol']}")
    yerel_dizin = sorted(p.name for p in ARS.iterdir() if p.is_dir() and p.name[:8].isdigit())
    man_dizin = sorted({d["yol"].split("/")[0] for d in man["dosyalar"] if d["yol"].split("/")[0][:8].isdigit()})
    eksik_dizin = sorted(set(man_dizin) - set(yerel_dizin))
    if len(yerel_dizin) < man["n_kosum_dizini"] - 1:  # n_kosum_dizini 'kuyruklar' dizinini de sayar
        hata.append(f"koşum dizini sayısı: yerel {len(yerel_dizin)} < manifest {man['n_kosum_dizini']}")
    hata += [f"manifest dizini yerelde yok: {d}" for d in eksik_dizin]
    man_yol = {d["yol"] for d in man["dosyalar"]}
    k5 = {}
    for ad, alan in (("k5_test_sonuc.json", None), ("k5_c1_sonuc.json", "C1")):
        J = json.loads((WS / "sonuclar" / ad).read_text(encoding="utf-8"))
        kos = J[alan]["kosumlar"] if alan else J["kosumlar"]
        n_dizin = n_test = 0
        for kol, liste in kos.items():
            for k in liste:
                if not (ARS / k).is_dir():
                    hata.append(f"{ad}: koşum dizini yok: {k}")
                    continue
                n_dizin += 1
                for f in ("tahmin_test.npy", "etiket_test.npy"):
                    if f"{k}/{f}" not in man_yol:
                        hata.append(f"{ad}: {k}/{f} manifest'te yok")
                    else:
                        n_test += 1
        k5[ad] = {"dizin": n_dizin, "test_dosyasi": n_test}
    sonuc = {"n_dosya_dogrulandi": len(man["dosyalar"]), "n_yerel_kosum_dizini": len(yerel_dizin),
             "manifest_n_kosum_dizini": man["n_kosum_dizini"], "test_dahil": man.get("test_dahil"),
             "n_kopya_test_adli": man.get("n_kopya_test_adli"), "k5": k5, "hata": hata}
    print(json.dumps(sonuc, ensure_ascii=False, indent=1))
    return 1 if hata else 0


if __name__ == "__main__":
    sys.exit(main())
