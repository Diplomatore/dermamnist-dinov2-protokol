# -*- coding: utf-8 -*-
"""
colab_arsiv_kopya.py — Colab Drive koşumlarının (<DRIVE_KOK>/3-1/runs) YEREL arşiv kopyası → WS/runs_colab/
(K5 öncesi D5 eksiği; deney-yürütücü, Tur-5). Ağırlık ve test dosyası KOPYALANMAZ.

Kural:
- Yalnız *.json, *.npy, *.log ve kosum.py damga dosyaları (.done, .heartbeat) kopyalanır.
- *.pth (ağırlık), *.npz (sonda ağırlığı), tfevents, csv/txt (rakip resmî çıktısı) HARİÇ.
- Adında "test" geçen HİÇBİR dosya kopyalanmaz (M4: tahmin_test*.npy / etiket_test.npy rakip köprüsünün yan ürünü;
  K4'te yerel diske alınmaz, açılmaz). Dışlanan dosyalar yalnız AD olarak manifest'e yazılır (içerik okunmaz).
- Her kopya SHA256 ile kaynağa karşı doğrulanır; manifest runs_colab/ARSIV_MANIFEST.json.
- Var olan ve SHA256'sı aynı dosya yeniden kopyalanmaz (artımlı); farklıysa HATA (arşiv üzerine yazılmaz).
Kullanım (WS kökü): env/.venv/Scripts/python.exe -I scripts/colab_arsiv_kopya.py [--kaynak "<DRIVE_KOK>/3-1/runs"]

K5 sonrası (K5-ONKAYIT-b §3 madde 2): --test-dahil → "test" adı dışlaması kaldırılır (tahmin_test*.npy,
etiket_test.npy, *-test koşumları dahil). Bayrak YALNIZ sonuclar/k5_test.KILIT varsa çalışır (test açıldıktan sonra).
Dosyaların İÇERİĞİ okunmaz; yalnız bayt kopyası + SHA256. Tür filtresi (.pth/.npz/csv/txt/tfevents hariç) aynen kalır.
Manifest'e test_dahil + KILIT içeriği yazılır; önceki manifest üzerine yazılmadan önce ARSIV_MANIFEST_onceki.json'a alınır.
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
IZINLI_SONEK = {".json", ".npy", ".log"}
IZINLI_AD = {".done", ".heartbeat"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for parca in iter(lambda: f.read(1 << 20), b""):
            h.update(parca)
    return h.hexdigest()


def izinli(p: Path, test_dahil: bool = False) -> bool:
    if "test" in p.name.lower() and not test_dahil:
        return False
    return p.name in IZINLI_AD or p.suffix.lower() in IZINLI_SONEK


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaynak", default="<DRIVE_KOK>/3-1/runs")
    ap.add_argument("--hedef", default=str(WS / "runs_colab"))
    ap.add_argument("--test-dahil", action="store_true",
                    help="K5 sonrası: adında 'test' geçen dosyaları da kopyala (k5_test.KILIT şart)")
    a = ap.parse_args()
    kilit = WS / "sonuclar" / "k5_test.KILIT"
    kilit_icerik = None
    if a.test_dahil:
        if not kilit.exists():
            raise SystemExit("HATA: --test-dahil yalnız sonuclar/k5_test.KILIT varken çalışır (test henüz açılmadı).")
        kilit_icerik = json.loads(kilit.read_text(encoding="utf-8"))
    kay, hed = Path(a.kaynak), Path(a.hedef)
    hed.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    kopya, ayni, disla = [], 0, []
    for p in sorted(kay.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(kay)
        if not izinli(p, a.test_dahil):
            disla.append({"yol": rel.as_posix(), "bayt": p.stat().st_size,
                          "neden": "test adı" if ("test" in p.name.lower() and not a.test_dahil)
                          else f"tür ({p.suffix or p.name})"})
            continue
        h = sha256(p)
        q = hed / rel
        if q.exists():
            if sha256(q) == h:
                ayni += 1
                kopya.append({"yol": rel.as_posix(), "bayt": p.stat().st_size, "sha256": h})
                continue
            raise SystemExit(f"HATA: arşivde farklı içerikli dosya var: {q} (üzerine yazılmaz)")
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
        if sha256(q) != h:
            raise SystemExit(f"HATA: kopya SHA256 tutmuyor: {rel}")
        kopya.append({"yol": rel.as_posix(), "bayt": p.stat().st_size, "sha256": h})
    test_kopya = [k for k in kopya if "test" in k["yol"].lower().rsplit("/", 1)[-1]]
    if not a.test_dahil:
        assert not test_kopya, test_kopya
    aciklama = ("Colab Drive runs/ → WS/runs_colab/ yerel arşiv kopyası (json/npy/log + .done/.heartbeat; "
                ".pth/.npz/csv/txt/tfevents ve adında 'test' geçen dosyalar HARİÇ). SHA256 kaynağa karşı doğrulandı.")
    if a.test_dahil:
        aciklama = ("Colab Drive runs/ → WS/runs_colab/ yerel arşiv kopyası, K5 SONRASI (--test-dahil; k5_test.KILIT var): "
                    "json/npy/log + .done/.heartbeat, adında 'test' geçen dosyalar DAHİL; .pth/.npz/csv/txt/tfevents HARİÇ. "
                    "SHA256 kaynağa karşı doğrulandı. İçerik okunmadı (yalnız bayt kopyası).")
    onceki = hed / "ARSIV_MANIFEST.json"
    if onceki.exists():
        shutil.copy2(onceki, hed / "ARSIV_MANIFEST_onceki.json")
    man = {"aciklama": aciklama, "test_dahil": bool(a.test_dahil), "k5_test_kilit": kilit_icerik,
           "kaynak": str(kay), "hedef": str(hed), "zaman": datetime.now().isoformat(timespec="seconds"),
           "sure_sn": round(time.time() - t0, 1), "n_kosum_dizini": len([d for d in kay.iterdir() if d.is_dir()]),
           "n_kopya": len(kopya), "n_zaten_ayni": ayni, "bayt_kopya": sum(k["bayt"] for k in kopya),
           "n_dislanan": len(disla), "bayt_dislanan": sum(d["bayt"] for d in disla),
           "dislanan_test_adli": sum(1 for d in disla if d["neden"] == "test adı"),
           "n_kopya_test_adli": len(test_kopya),
           "dosyalar": kopya, "dislanan": disla}
    (hed / "ARSIV_MANIFEST.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[arsiv] {man['n_kosum_dizini']} koşum dizini · {len(kopya)} dosya ({man['bayt_kopya']/1e6:.1f} MB; "
          f"{ayni} zaten aynı) · dışlanan {len(disla)} ({man['bayt_dislanan']/1e9:.2f} GB; test adlı "
          f"{man['dislanan_test_adli']}) · {man['sure_sn']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
