#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
e3_kuyruk_yaz.py — H16 (E3) `kuyruk/e3-onisleme.jsonl` satırlarını yazar (a1 2 + a2 4 + a3 20 = 26 satır; hepsi gpu_zorunlu,
yerel GTX 1650). Satırlar YALNIZ şu koşullarda yazılır (kart: "beklenen ve beklenti dosyası satırlardan ÖNCE dondurulur"):
  · kuyruk/e3_onisleme_beklenen.json VAR (a0 = kuyruk e3-veri bitti), K-b kapısı geçti, n ≥ 993, 4 npz'nin dosya SHA256'sı tutuyor;
  · kuyruk/e3-onisleme.beklenti.json VAR ve beklenen dosyasından ÖNCE yazılmış;
  · kuyruk/e3-onisleme.jsonl henüz YOK (ikinci kez yazılmaz).
Kullanım (WS kökü; a0 bittikten sonra, şef):
  env/.venv/Scripts/python.exe -I varyantlar/e3/e3_kuyruk_yaz.py            # yazar
  env/.venv/Scripts/python.exe -I varyantlar/e3/e3_kuyruk_yaz.py --kuru     # yalnız komutları gösterir
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "varyantlar"))
from e2 import bcn_veri as BV  # noqa: E402
from e2 import kollar as KL  # noqa: E402

KUYRUK_PY = WS.parents[1] / ".claude" / "skills" / "konudan-makale" / "scripts" / "kuyruk.py"
PY = str(WS / "env" / ".venv" / "Scripts" / "python.exe")
AD = "e3-onisleme"
BEKLENEN = WS / "kuyruk" / "e3_onisleme_beklenen.json"
BEKLENTI = WS / "kuyruk" / f"{AD}.beklenti.json"
HAT_DIZIN = WS / "data" / "ham_val_isic"
KURAL = json.dumps({"nan_kes": True, "ilerleme_zorunlu_sn": 600, "sabit_sn": 600}, separators=(",", ":"))
SINIR_DK = {"h10": 30, "h11": 20, "h02c": 15, "h08c": 15}      # v3: H10 196–307 s, H11 ≈ 61 s, R18 13–60 s (5 koşul) + Drive kopyası
SIRA = ("h10", "h02c", "h08c", "h11")                         # birincil kollar önce (bütçe tükenirse); sıra sonucu etkilemez


def satirlar():
    out = []
    for k in ("h02c", "h10"):
        out.append((1, k, 0, "duman", f"E3 a1 duman: {k} s0 yerel + birincil hat (etiket-kör)"))
    for k in SIRA:
        out.append((2, k, 0, "onucus", f"E3 a2 ön-uçuş: {k} s0 yerel + 4 hat (hat/jpeg/z125/z080; etiket-kör; süre → a3 bütçesi)"))
    for k in SIRA:
        for s in KL.SEEDLER:
            out.append((3, k, s, "tam", f"E3 a3 tam: {k} s{s} yerel + 4 hat (etiket-kör; tahmin dosyaları)"))
    return out


def varyant(k, kip):
    return {"duman": f"e3_{k}_duman", "onucus": f"e3_{k}_onucus", "tam": f"e3_{k}"}[kip]


def denetle() -> list:
    hata = []
    if not BEKLENEN.exists():
        return [f"{BEKLENEN.name} yok — önce kuyruk e3-veri (a0) bitmeli"]
    B = json.loads(BEKLENEN.read_text(encoding="utf-8"))
    if not ((B.get("kapilar") or {}).get("K-b") or {}).get("gecti"):
        hata.append("a0 K-b kapısı geçmemiş")
    if not 993 <= int(B.get("n", 0)) <= 1003:
        hata.append(f"n = {B.get('n')} ∉ [993; 1003]")
    for ad, b in (B.get("dosyalar") or {}).items():
        p = HAT_DIZIN / ad
        if not p.exists() or BV.dosya_sha256(p) != b["sha256"]:
            hata.append(f"{ad} yok ya da SHA256 tutmuyor")
    if len(B.get("dosyalar") or {}) != 4:
        hata.append("beklenen dosyasında 4 hat npz'si yok")
    if not BEKLENTI.exists():
        hata.append(f"{BEKLENTI.name} yok")
    elif BEKLENTI.stat().st_mtime > BEKLENEN.stat().st_mtime:
        hata.append("beklenti dosyası beklenen dosyasından SONRA değişmiş (ön-kayıt sırası)")
    if (WS / "kuyruk" / f"{AD}.jsonl").exists():
        hata.append(f"kuyruk/{AD}.jsonl zaten var — ikinci kez yazılmaz")
    return hata


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kuru", action="store_true")
    a = ap.parse_args()
    hata = denetle()
    if hata and not a.kuru:
        print("DUR (satır yazılmadı): " + " · ".join(hata))
        return 2
    for asama, k, s, kip, et in satirlar():
        kom = [PY, "-I", str(KUYRUK_PY), "--kok", str(WS), "ekle", AD, "--varyant", varyant(k, kip), "--seed", str(s),
               "--asama", str(asama), "--hipotez", "H16", "--ebeveyn", KL.run_id(k, s), "--etiket", et, "--gpu-zorunlu",
               "--zaman-siniri-dk", str(SINIR_DK[k]), "--kural", KURAL, "--",
               PY, "-I", "varyantlar/e3/e3_cikarim.py", "--kol", k, "--seed", str(s), "--kip", kip, "--out", "{RUN_DIR}/metrics.json"]
        if a.kuru:
            print(" ".join(kom[4:]))
            continue
        r = subprocess.run(kom, cwd=str(WS), capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(r.stdout, r.stderr)
            return r.returncode
    if a.kuru:
        print(f"(KURU: {len(satirlar())} satır yazılmadı; denetim: {hata or 'tamam'})")
    else:
        print(f"yazıldı: kuyruk/{AD}.jsonl ({len(satirlar())} satır)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
