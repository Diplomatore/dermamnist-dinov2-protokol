#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
oturum_sar.py — PARALEL Colab oturumları için bildirim ayırıcı (orkestra kodu DEĞİŞMEZ, yalnız sarılır).

Sorun: orkestra/kuyruk.py ve orkestra/kosum.py her olayı `runs/BILDIRIM.log`'a EKLER. runs/ Drive'dadır; üç Colab VM'i
aynı Drive dosyasına aynı anda eklerse Drive dosya düzeyinde "son yazan kazanır" davranır → satır kaybı.
Çözüm: betik bu süreçte modül olarak yüklenir, yalnız `bildir()` işlevi `runs/BILDIRIM_<OTURUM>.log`'a yazacak biçimde
değiştirilir (satır biçimi AYNI: "<iso-zaman> <olay>"). kuyruk.py'nin kosum.py'yi çağıran komutu da bu sarmalayıcıdan
geçirilir (alt süreçler aynı oturum dosyasına yazar). Başka hiçbir davranış değişmez (argümanlar, damgalar, kapılar).

Kullanım (defter_yardim.kuyruk_kos oturum kipinde):
  KC_OTURUM=<oturum> python colab/oturum_sar.py orkestra/kuyruk.py --kok . kos <ad> --kademeli --paralel 1
KC_OTURUM yoksa betik değiştirilmeden (runpy, __main__) koşar → eski tek-oturum davranışı (runs/BILDIRIM.log).
"""
import importlib.util
import os
import re
import runpy
import sys
from pathlib import Path

SAR = Path(__file__).resolve()


def bildirim_dosyasi(kok: Path, oturum: str) -> Path:
    guvenli = re.sub(r"[^A-Za-z0-9_.-]", "_", oturum)
    return Path(kok) / "runs" / f"BILDIRIM_{guvenli}.log"


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("kullanım: oturum_sar.py <orkestra/betik.py> [argümanlar…]")
    betik = Path(sys.argv[1]).resolve()
    argumanlar = sys.argv[2:]
    oturum = os.environ.get("KC_OTURUM", "").strip()
    sys.argv = [str(betik)] + argumanlar
    if not oturum:  # geriye uyum: hiçbir şey değiştirmeden koş
        runpy.run_path(str(betik), run_name="__main__")
        return 0
    spec = importlib.util.spec_from_file_location(f"_oturum_sar_{betik.stem}", betik)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # __name__ != "__main__" → betiğin main()'i burada koşmaz
    if not hasattr(mod, "bildir") or not hasattr(mod, "main"):
        raise SystemExit(f"HATA: {betik.name} beklenen bildir()/main() işlevlerini içermiyor — sarmalayıcı uygulanamaz")
    simdi = mod.simdi

    def bildir(kok, satir):
        try:
            yol = bildirim_dosyasi(Path(kok), oturum)
            yol.parent.mkdir(parents=True, exist_ok=True)
            with open(yol, "a", encoding="utf-8") as f:
                f.write(f"{simdi()} {satir}\n")
        except OSError as e:
            print(f"[oturum_sar] UYARI bildirim yazılamadı: {e}", file=sys.stderr, flush=True)

    mod.bildir = bildir
    if hasattr(mod, "kosum_komutu"):  # kuyruk.py → kosum.py alt süreci de sarmalayıcıdan geçer
        asil = mod.kosum_komutu

        def kosum_komutu(kok, ad, j):
            cmd = asil(kok, ad, j)
            return [cmd[0], str(SAR)] + list(cmd[1:])

        mod.kosum_komutu = kosum_komutu
    return mod.main()


if __name__ == "__main__":
    sys.exit(main())
