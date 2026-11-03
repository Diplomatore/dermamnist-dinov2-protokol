# -*- coding: utf-8 -*-
"""
drive_sonuc_kopya.py — Colab Drive sonuclar/ (<DRIVE_KOK>/3-1/sonuclar) → WS/sonuclar/ kopyası (K5 D5 eksiği, karar.md §8).

Kopyalanan: beklenti kapı raporları (beklenti-*.md: P6 k5-test, k5-c1-*, kc1/kc2) + Colab leaderboard'ları
(leaderboard-colab-*, leaderboard-kc1.*) + tekrar-ozeti-colab.md.
Kural:
- Yerelde AYNI ADLI dosya varsa: SHA256 aynıysa atlanır; farklıysa ÜZERİNE YAZILMAZ, "catisma" olarak manifest'e yazılır.
- Drive replikasyon.md yerelde replikasyon-colab.md adıyla zaten var (SHA256 ile denetlenir; kopyalanmaz).
- Her kopya SHA256 ile kaynağa karşı doğrulanır; manifest sonuclar/DRIVE_SONUC_MANIFEST.json.
Kullanım (WS kökü): env/.venv/Scripts/python.exe -I scripts/drive_sonuc_kopya.py
"""
import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
DESEN = ["beklenti-*.md", "leaderboard-colab-*", "leaderboard-kc1.*", "tekrar-ozeti-colab.md"]
ESDEGER = {"replikasyon.md": "replikasyon-colab.md"}  # Drive adı → yerel adı


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for parca in iter(lambda: f.read(1 << 20), b""):
            h.update(parca)
    return h.hexdigest()


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaynak", default="<DRIVE_KOK>/3-1/sonuclar")
    ap.add_argument("--hedef", default=str(WS / "sonuclar"))
    a = ap.parse_args()
    kay, hed = Path(a.kaynak), Path(a.hedef)
    secili = sorted({p for d in DESEN for p in kay.glob(d) if p.is_file()})
    kopya, ayni, catisma = [], [], []
    for p in secili:
        h = sha256(p)
        q = hed / p.name
        if q.exists():
            (ayni if sha256(q) == h else catisma).append({"ad": p.name, "sha256": h})
            continue
        shutil.copy2(p, q)
        if sha256(q) != h:
            raise SystemExit(f"HATA: kopya SHA256 tutmuyor: {p.name}")
        kopya.append({"ad": p.name, "bayt": p.stat().st_size, "sha256": h})
    esdeger = []
    for dad, yad in ESDEGER.items():
        dp, yp = kay / dad, hed / yad
        if dp.exists() and yp.exists():
            esdeger.append({"drive": dad, "yerel": yad, "ayni": sha256(dp) == sha256(yp)})
    ortak = []
    for p in sorted(kay.iterdir()):
        if p.is_file() and p not in secili and p.name not in ESDEGER and (hed / p.name).exists():
            ortak.append({"ad": p.name, "ayni": sha256(p) == sha256(hed / p.name)})
    man = {"aciklama": "Drive sonuclar/ → WS/sonuclar/ (beklenti kapı raporları + Colab leaderboard'ları). SHA256 doğrulandı; "
                       "yerel dosya üzerine yazılmadı.",
           "kaynak": str(kay), "hedef": str(hed), "zaman": datetime.now().isoformat(timespec="seconds"),
           "n_kopya": len(kopya), "n_zaten_ayni": len(ayni), "n_catisma": len(catisma),
           "kopya": kopya, "zaten_ayni": ayni, "catisma": catisma, "esdeger_ad": esdeger, "ortak_ad_denetim": ortak}
    (hed / "DRIVE_SONUC_MANIFEST.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[drive-sonuc] kopya {len(kopya)} · zaten aynı {len(ayni)} · çatışma {len(catisma)} · "
          f"eşdeğer {esdeger} · ortak ad {[(o['ad'], o['ayni']) for o in ortak]}")
    return 1 if catisma or not all(e["ayni"] for e in esdeger) or not all(o["ayni"] for o in ortak) else 0


if __name__ == "__main__":
    sys.exit(main())
