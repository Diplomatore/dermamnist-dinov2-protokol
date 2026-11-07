#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
erteleme_kapisi.py — H11 erteleme kuralının (ön-kayıt; H11.md "Bütçe" + SIRALAMA Tur-4 §Neden 3) MEKANİK uygulaması.

Kural (kart, sonuca bakmaz): "H11 a3 başlamadan önce günlükteki bütçe sayacı > 23,0 GPU-sa ise H11 a3 ertelenir ve
karar şef/kullanıcıya gider." Colab hattında H11 a3 kuyrukta otomatik başladığı için şef araya giremez; bu betik her
H11 a3 satırının önünde sayacı hesaplar:
    sayaç = --baslangic-sa (yerel GTX 1650 GPU-sa, Colab'dan ÖNCE; SIRALAMA Tur-4 "Bütçe sayacı ≈ 15,3")
          + Σ Colab koşum süresi / 3600        ← L4-saat (KULLANICI KARARI 6–7; birim notu aşağıda)
  sayaç > --esik (23,0) → koşum BAŞLATILMAZ: {RUN_DIR}/erteleme.json yazılır, çıkış kodu 94 (metrics.json yok;
  kosum.py damgası rc=94). Aksi halde komut os.execvp ile AYNI süreçte başlatılır (PID değişmez → kosum.py --kural
  öldürmesi eğitimi öldürür).

PARALEL OTURUMLAR (v2, 3 sekme; 2026-10-11): sayaç bütün eşzamanlı Colab oturumlarının TOPLAM GPU-saatidir.
runs/ ortak Drive klasörü olduğundan her sekmenin her kuyruğunun koşumları sayılır:
  (1) damgalı koşum (.done'da sure_sn)          → sure_sn
  (2) damgasız koşum (başka sekmede ŞU AN koşan ya da kopan oturumdan kalan) → .heartbeat'teki gecen_sn
      (kosum.py her 15 sn yazar; Drive eşitleme gecikmesi kadar eksik sayabilir, ≈ 1–3 dk)
  (3) damgalı ama sure_sn'siz (yetim rc=96)      → .heartbeat gecen_sn (koşum GPU'yu kullandı; sayaç monoton kalsın)
  Bu satırın kendi RUN_DIR'i sayılmaz. 3 sekme paralelken duvar saatinin 1 saati sayaca ≈ 3 GPU-sa ekler.
  erteleme.json'da kuyruk bazında döküm (damgalı / koşan / yetim ayrı) kayıtlıdır (denetim).
Birim notu: Colab koşumları L4-saatle sayılır (1 L4-sa = 1 GPU-sa sayılır; donanım farkı sayaçta düzeltilmez).
Kurulum/indirme/öz-sınama süreleri (koşum dışı) sayaca girmez — yerel sayaçla aynı tanım (koşum süreleri).

Kullanım (kuyruk satırında):
  python colab/erteleme_kapisi.py --esik 23.0 --baslangic-sa 15.3 -- python colab/fp32_kos.py varyantlar/... --out {RUN_DIR}/metrics.json
  python colab/erteleme_kapisi.py --kendi-testi
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _oku(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — eşitleme sırasında yarım dosya
        return None


def colab_saat_dokum(runs: Path, haric: Path | None = None, kuyruk_onek: str | None = None) -> dict:
    """Σ koşum süresi (sa) + döküm. haric: bu satırın kendi RUN_DIR'i (sayılmaz).
    kuyruk_onek (E1/H14 kural h, 2026-11-03): verilirse YALNIZ config.kuyruk'u bu önekle başlayan koşumlar sayılır
    (ör. 'e1-' → E1 birleşik L4-sa; Ekim koşumları sayılmaz). None → eski davranış (hepsi)."""
    out = {"toplam_sa": 0.0, "damgali_n": 0, "damgali_sa": 0.0, "kosan_n": 0, "kosan_sa": 0.0,
           "yetim_n": 0, "yetim_sa": 0.0, "kuyruk_sa": {}}
    if not runs.exists():
        return out
    haric_ad = Path(haric).name if haric else None
    for d in runs.iterdir():
        if not d.is_dir() or d.name == "kuyruklar" or d.name == haric_ad:
            continue
        if kuyruk_onek is not None and not str(((_oku(d / "config.json") or {}).get("kuyruk")) or "").startswith(kuyruk_onek):
            continue
        done = d / ".done"
        s, tur = None, None
        if done.exists():
            j = _oku(done) or {}
            if isinstance(j.get("sure_sn"), (int, float)):
                s, tur = float(j["sure_sn"]), "damgali"
            else:
                hb = _oku(d / ".heartbeat") or {}
                if isinstance(hb.get("gecen_sn"), (int, float)):
                    s, tur = float(hb["gecen_sn"]), "yetim"
        else:
            hb = _oku(d / ".heartbeat") or {}
            if isinstance(hb.get("gecen_sn"), (int, float)):
                s, tur = float(hb["gecen_sn"]), "kosan"
        if s is None:
            continue
        out[f"{tur}_n"] += 1
        out[f"{tur}_sa"] += s / 3600.0
        out["toplam_sa"] += s / 3600.0
        kq = ((_oku(d / "config.json") or {}).get("kuyruk")) or "?"
        out["kuyruk_sa"][kq] = out["kuyruk_sa"].get(kq, 0.0) + s / 3600.0
    for k in ("toplam_sa", "damgali_sa", "kosan_sa", "yetim_sa"):
        out[k] = round(out[k], 6)
    out["kuyruk_sa"] = {k: round(v, 6) for k, v in sorted(out["kuyruk_sa"].items())}
    return out


def colab_saat(runs: Path) -> tuple[float, int]:
    """v1 arayüzü (geriye uyum): (toplam saat, sayılan koşum)."""
    d = colab_saat_dokum(runs)
    return d["toplam_sa"], d["damgali_n"] + d["kosan_n"] + d["yetim_n"]


def main() -> int:
    argv = sys.argv[1:]
    if argv == ["--kendi-testi"]:
        return kendi_testi()
    if "--" not in argv:
        raise SystemExit("kullanım: erteleme_kapisi.py --esik X --baslangic-sa Y -- <komut...>")
    i = argv.index("--")
    p = argparse.ArgumentParser()
    p.add_argument("--esik", type=float, required=True)
    p.add_argument("--baslangic-sa", type=float, required=True)
    p.add_argument("--kok", default=".")
    p.add_argument("--kuyruk-onek", default=None, help="yalnız bu önekli kuyrukların koşumları (E1: 'e1-')")
    p.add_argument("--kural", default="H11 erteleme (H11.md Bütçe; SIRALAMA Tur-4 §3)", help="kayıt metni")
    a = p.parse_args(argv[:i])
    komut = argv[i + 1:]
    if not komut:
        raise SystemExit("HATA: '--' sonrası komut yok")
    kok = Path(a.kok).resolve()
    rd = os.environ.get("RUN_DIR")
    D = colab_saat_dokum(kok / "runs", Path(rd) if rd else None, a.kuyruk_onek)
    saat = D["toplam_sa"]
    sayac = a.baslangic_sa + saat
    karar = "ERTELE" if sayac > a.esik else "DEVAM"
    kayit = {"zaman": time.strftime("%Y-%m-%dT%H:%M:%S"), "kural": a.kural, "kuyruk_onek": a.kuyruk_onek,
             "baslangic_sa_yerel": a.baslangic_sa, "colab_l4_sa": round(saat, 3),
             "colab_damgali_kosum": D["damgali_n"], "colab_kosan_kosum": D["kosan_n"], "colab_yetim_kosum": D["yetim_n"],
             "dokum": D, "tanim": "tüm eşzamanlı Colab oturumları: damgalı sure_sn + damgasız/yetim .heartbeat gecen_sn",
             "sayac": round(sayac, 3), "esik": a.esik, "karar": karar, "komut": komut}
    if rd:
        try:
            (Path(rd) / "erteleme.json").write_text(json.dumps(kayit, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError:
            pass
    print(f"[erteleme_kapisi] sayaç {sayac:.2f} GPU-sa (yerel {a.baslangic_sa} + Colab L4 {saat:.2f}: damgalı {D['damgali_n']} "
          f"{D['damgali_sa']:.2f} sa, koşan {D['kosan_n']} {D['kosan_sa']:.2f} sa, yetim {D['yetim_n']} {D['yetim_sa']:.2f} sa) "
          f"{'>' if karar == 'ERTELE' else '≤'} eşik {a.esik} → {karar}", flush=True)
    if karar == "ERTELE":
        print(f"[erteleme_kapisi] {a.kural}: ERTELENDİ/DUR — karar şef/kullanıcıda (rc=94, metrics yok).", flush=True)
        return 94
    sys.stdout.flush()
    os.execvp(komut[0], komut)
    return 0  # erişilmez


def kendi_testi() -> int:
    t = Path(tempfile.mkdtemp(prefix="erteleme_test_"))
    try:
        runs = t / "runs"
        (runs / "kuyruklar").mkdir(parents=True)

        def kosum(ad, kuyruk, done=None, hb=None):
            d = runs / ad
            d.mkdir()
            (d / "config.json").write_text(json.dumps({"kuyruk": kuyruk}), encoding="utf-8")
            if done is not None:
                (d / ".done").write_text(json.dumps(done), encoding="utf-8")
            if hb is not None:
                (d / ".heartbeat").write_text(json.dumps({"gecen_sn": hb}), encoding="utf-8")
            return d

        kosum("r1", "kc1-tekrar", {"rc": 0, "sure_sn": 3600.0}, 3590)   # damgalı 1,0 sa (hb yok sayılır)
        kosum("r2", "kc2-tur4-h10", None, 1800)                         # başka sekmede koşuyor 0,5 sa
        kosum("r3", "kc2-tur4-h12", {"rc": 96, "hata": "yetim"}, 900)   # yetim 0,25 sa
        kosum("r4", "kc2-tur4-h12", {"rc": 0, "sure_sn": 1800.0})       # damgalı 0,5 sa
        ben = kosum("r5", "kc2-tur4-h11", None, 30)                     # kendisi: sayılmaz
        kosum("r6", "kc2-tur4-h10", None, None)                         # henüz kalp atışı yok: 0
        D = colab_saat_dokum(runs, ben)
        assert abs(D["toplam_sa"] - 2.25) < 1e-6, D
        assert (D["damgali_n"], D["kosan_n"], D["yetim_n"]) == (2, 1, 1), D
        assert abs(D["kuyruk_sa"]["kc2-tur4-h12"] - 0.75) < 1e-6, D
        D2 = colab_saat_dokum(runs)
        assert abs(D2["toplam_sa"] - (2.25 + 30 / 3600)) < 1e-6, D2
        h, n = colab_saat(runs)
        assert n == 5 and abs(h - D2["toplam_sa"]) < 1e-9, (h, n)
        # E1 (kural h): kuyruk öneki filtresi — yalnız 'e1-' kuyrukları
        kosum("e1a", "e1-h14", {"rc": 0, "sure_sn": 1800.0})
        kosum("e1b", "e1-h10r", None, 900)
        De = colab_saat_dokum(runs, ben, "e1-")
        assert abs(De["toplam_sa"] - 0.75) < 1e-6 and set(De["kuyruk_sa"]) == {"e1-h14", "e1-h10r"}, De
        r = subprocess.run([sys.executable, __file__, "--esik", "0.5", "--baslangic-sa", "0", "--kuyruk-onek", "e1-",
                            "--kural", "E1 bütçe (H14 kural h)", "--kok", str(t), "--", sys.executable, "-c", "print('KOSTU')"],
                           capture_output=True, text=True, env={**os.environ, "RUN_DIR": str(ben)})
        e = json.loads((ben / "erteleme.json").read_text(encoding="utf-8"))
        assert r.returncode == 94 and "KOSTU" not in r.stdout and abs(e["sayac"] - 0.75) < 1e-3, (r, e)
        (runs / "e1a").rename(runs / "e1a_x"); (runs / "e1b").rename(runs / "e1b_x")
        shutil.rmtree(runs / "e1a_x"); shutil.rmtree(runs / "e1b_x")
        # karar: 21,0 + 2,25 = 23,25 > 23 → ERTELE (rc 94, komut koşmaz); 15,3 + 2,25 ≤ 23 → DEVAM (komut koşar)
        env = {**os.environ, "RUN_DIR": str(ben)}
        r = subprocess.run([sys.executable, __file__, "--esik", "23.0", "--baslangic-sa", "21.0", "--kok", str(t), "--",
                            sys.executable, "-c", "print('KOSTU')"], capture_output=True, text=True, env=env)
        e = json.loads((ben / "erteleme.json").read_text(encoding="utf-8"))
        assert r.returncode == 94 and "KOSTU" not in r.stdout and e["karar"] == "ERTELE" and abs(e["sayac"] - 23.25) < 1e-3, (r, e)
        r = subprocess.run([sys.executable, __file__, "--esik", "23.0", "--baslangic-sa", "15.3", "--kok", str(t), "--",
                            sys.executable, "-c", "print('KOSTU')"], capture_output=True, text=True, env=env)
        e = json.loads((ben / "erteleme.json").read_text(encoding="utf-8"))
        assert r.returncode == 0 and "KOSTU" in r.stdout and e["karar"] == "DEVAM", (r.returncode, r.stdout, r.stderr, e)
        print(f"KENDİ-TESTİ GEÇTİ — döküm: {json.dumps(D, ensure_ascii=False)}")
        return 0
    finally:
        shutil.rmtree(t, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
