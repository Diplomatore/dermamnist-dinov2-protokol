# -*- coding: utf-8 -*-
"""
defter_yardim.py — 3-1-colab.ipynb'nin yardımcıları (Colab hattı; yerel kodu DEĞİŞTİRMEZ, yalnız sarar).

Düzen (Colab):
  <DRIVE_KOK>/3-1/      Drive çalışma kökü (KALICI): kod paketi + PAKET.json + kuyruk/ runs/ sonuclar/ env/ durum/
  /content/is/                        çalışma kopyası (yerel disk): kod dizinleri Drive'dan KOPYALANIR ve SHA256 ile doğrulanır;
                                      runs/ kuyruk/ sonuclar/ env/ → Drive'a sembolik bağ (kalıcı);
                                      data/medmnist, data/dermamnist_c → /content/veri/... (oturum başına indirilir, MD5+SHA256);
                                      data/oznitelik, data/hf_cache, data/torch_cache → yerel disk (oturum başına yeniden üretilir)
  Neden kopya: kod WS'yi `Path(__file__).parents[k]` ile bulur ve veriyi WS/data/medmnist'te arar; Drive FUSE'ta sembolik
  bağ kurulamaz. Kopya + bağ ile hiçbir yerel dosya değişmeden yollar tutar.

Kuyruk aileleri: <ad>, <ad>-devam1, <ad>-devam2 … (kuyruk.py bitmiş satırları atlamaz, aynı ad yeniden koşulmaz → devam
mantığı burada). Gruplar kuyruk/gruplar.json'da (ör. kc2-tur4 → kc2-tur4-h12, -h10, -h11 sırayla).

PARALEL OTURUM (v2 defterleri 3-1-colab-A/B/C.ipynb): defterdeki KUYRUK bir grup adı YA DA tek kuyruk adı olabilir
(`kuyruk_cozumle`). Oturum kipinde (`oturum=` verilince) Drive'a yazılan her ORTAK ekleme dosyası oturuma ayrılır:
  runs/BILDIRIM_<oturum>.log  (kuyruk.py/kosum.py → colab/oturum_sar.py; yetim/koptu satırları da)
  durum/canli_<oturum>.json   (kalp atışı; yalnız bu oturumun kuyruk ailelerinin aktif koşumları)
  durum/canli_hata_<oturum>.txt
Kuyruk damgaları (runs/kuyruklar/<ad>.*), devam kuyrukları ve beklenti raporları zaten kuyruk adına göre ayrıktır;
koşum klasörleri zaman+varyant+seed ile benzersizdir. `oturum=None` → v1 davranışı BİREBİR (kc1 defteri, canli.json,
runs/BILDIRIM.log). Aynı kuyruğun iki sekmede koşmasını `es_zamanli_cakisma` engeller (taze kalp atışı + kuyruk kesişimi).
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

KOD_DIZINLERI = ("scripts", "varyantlar", "rakip", "orkestra", "colab")
DRIVE_KALICI = ("runs", "kuyruk", "sonuclar", "env")
UA = {"User-Agent": "3-1-colab-hatti/1.0 (akademik; urllib)"}


def simdi() -> str:
    return datetime.now().isoformat(timespec="seconds")


def damga_adi() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def atomik_json(yol: Path, veri) -> None:
    yol = Path(yol)
    yol.parent.mkdir(parents=True, exist_ok=True)
    tmp = yol.with_name(yol.name + ".tmp")
    tmp.write_text(json.dumps(veri, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, yol)


def sha256(yol: Path) -> str:
    h = hashlib.sha256()
    with open(yol, "rb") as f:
        for parca in iter(lambda: f.read(1 << 20), b""):
            h.update(parca)
    return h.hexdigest()


def md5_sha256(yol: Path):
    h5, h2 = hashlib.md5(), hashlib.sha256()
    with open(yol, "rb") as f:
        for parca in iter(lambda: f.read(1 << 22), b""):
            h5.update(parca)
            h2.update(parca)
    return h5.hexdigest(), h2.hexdigest()


def bildirim_yolu(kok: Path, oturum: str | None = None) -> Path:
    """oturum None → runs/BILDIRIM.log (v1); aksi halde runs/BILDIRIM_<oturum>.log (oturum_sar.py ile AYNI ad)."""
    if not oturum:
        return Path(kok) / "runs" / "BILDIRIM.log"
    import re
    return Path(kok) / "runs" / f"BILDIRIM_{re.sub(r'[^A-Za-z0-9_.-]', '_', oturum)}.log"


def bildirim_yaz(kok: Path, satir: str, oturum: str | None = None) -> None:
    try:
        with open(bildirim_yolu(kok, oturum), "a", encoding="utf-8") as f:
            f.write(f"{simdi()} {satir}\n")
    except OSError:
        pass


def kos(cmd, cwd=None, env=None) -> int:
    """Komutu koşar, çıktıyı canlı basar, dönüş kodunu verir."""
    cmd = [str(c) for c in cmd]
    print("$", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
    for satir in p.stdout:
        print(satir, end="", flush=True)
    return p.wait()


# ───────────────────────── paket ─────────────────────────
def paket_kopyala_dogrula(drive_kok: Path, is_kok: Path) -> dict:
    """Drive'daki kod paketini is_kok'a kopyalar ve PAKET.json SHA256'larıyla doğrular (kopya = koşulan kod)."""
    paket = json.loads((drive_kok / "PAKET.json").read_text(encoding="utf-8"))
    dosyalar = paket["dosyalar"]
    for d in KOD_DIZINLERI:
        hedef = is_kok / d
        if hedef.is_symlink() or hedef.is_file():
            hedef.unlink()
        elif hedef.exists():
            shutil.rmtree(hedef)
    (is_kok / "data").mkdir(parents=True, exist_ok=True)
    kopyalanan, bozuk, eksik = 0, [], []
    for rel, bilgi in sorted(dosyalar.items()):
        if rel.split("/")[0] not in KOD_DIZINLERI and rel.split("/")[0] != "data":
            continue
        kaynak, hedef = drive_kok / rel, is_kok / rel
        if not kaynak.exists():
            eksik.append(rel)
            continue
        hedef.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(kaynak, hedef)
        if sha256(hedef) != bilgi["sha256"]:
            bozuk.append(rel)
        kopyalanan += 1
    shutil.copyfile(drive_kok / "PAKET.json", is_kok / "PAKET.json")
    # pakette olmayan fazladan dosya (Drive'a elle eklenmiş kod) — uyarı
    fazla = []
    for d in KOD_DIZINLERI:
        for p in (drive_kok / d).rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                rel = p.relative_to(drive_kok).as_posix()
                if rel not in dosyalar:
                    fazla.append(rel)
    return {"surum": paket["surum"], "kopyalanan": kopyalanan, "bozuk": bozuk, "eksik": eksik, "fazla": fazla,
            "dosya_sayisi": len(dosyalar), "olusturma": paket.get("olusturma")}


def baglantilari_kur(drive_kok: Path, is_kok: Path, veri_kok: Path) -> dict:
    out = {}
    for d in DRIVE_KALICI + ("durum",):
        (drive_kok / d).mkdir(parents=True, exist_ok=True)
    for d in DRIVE_KALICI:
        bag = is_kok / d
        if bag.is_symlink() or bag.is_file():
            bag.unlink()
        elif bag.exists():
            shutil.rmtree(bag)
        bag.symlink_to(drive_kok / d, target_is_directory=True)
        out[d] = str(drive_kok / d)
    for alt in ("medmnist", "dermamnist_c"):
        (veri_kok / alt).mkdir(parents=True, exist_ok=True)
        bag = is_kok / "data" / alt
        if bag.is_symlink() or bag.is_file():
            bag.unlink()
        elif bag.exists():
            shutil.rmtree(bag)
        bag.symlink_to(veri_kok / alt, target_is_directory=True)
        out[f"data/{alt}"] = str(veri_kok / alt)
    for alt in ("oznitelik", "hf_cache", "torch_cache"):
        (is_kok / "data" / alt).mkdir(parents=True, exist_ok=True)
    return out


# ───────────────────────── veri ─────────────────────────
def indir(url: str, hedef: Path, bayt: int | None = None, deneme: int = 8, log=print) -> float:
    """HTTP Range ile kaldığı yerden devam eden indirme (.part). Dönüş: süre (s)."""
    part = hedef.with_name(hedef.name + ".part")
    t0, son_log = time.time(), 0.0
    for i in range(deneme):
        mevcut = part.stat().st_size if part.exists() else 0
        if bayt and mevcut == bayt:
            break
        istek = urllib.request.Request(url, headers={**UA, **({"Range": f"bytes={mevcut}-"} if mevcut else {})})
        try:
            with urllib.request.urlopen(istek, timeout=120) as r:
                if mevcut and r.status != 206:  # sunucu Range desteklemedi → baştan
                    mevcut = 0
                    part.unlink(missing_ok=True)
                with open(part, "ab") as f:
                    while True:
                        parca = r.read(1 << 22)
                        if not parca:
                            break
                        f.write(parca)
                        if time.time() - son_log > 30:
                            son_log = time.time()
                            n = part.stat().st_size
                            log(f"  {hedef.name}: {n / 1e6:.0f}/{(bayt or 0) / 1e6:.0f} MB "
                                f"({n / 1e6 / max(time.time() - t0, 1e-9):.1f} MB/s)")
            if not bayt or part.stat().st_size == bayt:
                break
        except Exception as e:  # noqa: BLE001
            log(f"  {hedef.name}: deneme {i + 1}/{deneme} hata: {e}")
            time.sleep(min(60, 5 * (i + 1)))
    os.replace(part, hedef)
    return time.time() - t0


def veri_hazirla(liste_json: Path, veri_kok: Path, grup: str, kayit_json: Path | None = None, log=print) -> dict:
    L = json.loads(Path(liste_json).read_text(encoding="utf-8"))
    sonuc = {}
    for ad in L["gruplar"][grup]:
        b = L["dosyalar"][ad]
        hedef = veri_kok / b["alt"] / ad
        hedef.parent.mkdir(parents=True, exist_ok=True)
        durum, sure = "mevcut", 0.0
        if not hedef.exists() or hedef.stat().st_size != b["bayt"]:
            log(f"indiriliyor: {ad} ({b['bayt'] / 1e6:.0f} MB) ← {b['url']}")
            sure = indir(b["url"], hedef, b["bayt"], log=log)
            durum = "indirildi"
        md5, s256 = md5_sha256(hedef)
        ok = (md5 == b["md5"] and s256 == b["sha256"] and hedef.stat().st_size == b["bayt"])
        sonuc[ad] = {"durum": durum, "bayt": hedef.stat().st_size, "md5": md5, "sha256": s256, "md5_resmi": b["md5"],
                     "sha256_yerel": b["sha256"], "dogrulandi": ok, "sure_sn": round(sure, 1), "zaman": simdi()}
        log(f"{'OK ' if ok else 'HATA'} {ad}: md5 {md5} sha256 {s256[:12]}… ({durum}, {sure:.0f} s)")
        if not ok:
            hedef.unlink(missing_ok=True)
            if kayit_json:
                atomik_json(kayit_json, {"grup": grup, "dosyalar": sonuc, "durum": "HATA"})
            raise RuntimeError(f"{ad}: MD5/SHA256/bayt uyuşmuyor — dosya silindi, hücreyi yeniden çalıştır")
    if kayit_json:
        atomik_json(kayit_json, {"grup": grup, "dosyalar": sonuc, "durum": "OK", "zaman": simdi()})
    return sonuc


def veri_arka_plan(liste_json: Path, veri_kok: Path, grup: str, kayit_json: Path, log_dosya: Path) -> threading.Thread:
    def calis():
        with open(log_dosya, "a", encoding="utf-8") as f:
            def log(s):
                f.write(f"{simdi()} {s}\n"); f.flush()
            try:
                veri_hazirla(liste_json, veri_kok, grup, kayit_json, log=log)
            except Exception as e:  # noqa: BLE001
                log(f"HATA: {e}")
                atomik_json(kayit_json, {"grup": grup, "durum": "HATA", "hata": str(e), "zaman": simdi()})
    t = threading.Thread(target=calis, name=f"veri-{grup}", daemon=True)
    t.start()
    return t


# ───────────────────────── kuyruk aileleri / devam ─────────────────────────
def gruplar(kok: Path) -> dict:
    return json.loads((kok / "kuyruk" / "gruplar.json").read_text(encoding="utf-8"))["gruplar"]


IZIN_GEREKEN = {"kc2-tur4"}  # + gruplar.json'daki isteğe bağlı "izin_gereken" listesi (K5 vb. için kod değişmeden)


def kuyruk_cozumle(kok: Path, ad: str) -> list:
    """Defterdeki KUYRUK değeri → koşulacak kuyruk listesi. Grup adı → gruptaki kuyruklar (SIRAYLA);
    tek kuyruk adı (kuyruk/<ad>.jsonl var, '-devamN' değil) → [ad]. Bilinmeyen ad → KeyError."""
    G = gruplar(kok)
    if ad in G:
        return list(G[ad])
    import re
    if re.search(r"-devam\d+$", ad):
        raise KeyError(f"{ad!r} bir devam kuyruğu; defterde ailenin ana adını yaz (devam otomatik)")
    if (Path(kok) / "kuyruk" / f"{ad}.jsonl").exists():
        return [ad]
    raise KeyError(f"bilinmeyen kuyruk/grup {ad!r}; gruplar: {sorted(G)}; tek kuyruk için kuyruk/{ad}.jsonl gerekir")


def izin_dosyasi(kok: Path, ad: str):
    """KUYRUK (grup ya da tek kuyruk) şef izni gerektiriyorsa izin dosyasının yolu, gerekmiyorsa None.
    Tek kuyruk, içinde bulunduğu grubun iznini devralır (kc2-tur4-h10 → kuyruk/kc2-tur4.izin.json)."""
    ham = json.loads((Path(kok) / "kuyruk" / "gruplar.json").read_text(encoding="utf-8"))
    G, gereken = ham["gruplar"], IZIN_GEREKEN | set(ham.get("izin_gereken") or [])
    for g in sorted(gereken):
        if ad == g or ad in G.get(g, []):
            return Path(kok) / "kuyruk" / f"{g}.izin.json"
    return None


def aile_adi_mi(kosum_kuyrugu, kuyruklar: list) -> bool:
    """Koşumun config.kuyruk değeri verilen kuyruklardan birinin ailesinde mi (<ad> ya da <ad>-devamN)."""
    if not kosum_kuyrugu:
        return False
    for q in kuyruklar:
        if kosum_kuyrugu == q or (kosum_kuyrugu.startswith(q + "-devam") and kosum_kuyrugu[len(q) + 6:].isdigit()):
            return True
    return False


def aile(kok: Path, ad: str) -> list:
    adlar = [ad]
    devam = []
    for p in (kok / "kuyruk").glob(f"{ad}-devam*.jsonl"):
        ek = p.stem[len(ad) + len("-devam"):]
        if ek.isdigit():
            devam.append((int(ek), p.stem))
    return adlar + [s for _, s in sorted(devam)]


def satirlar(kok: Path, ad: str) -> list:
    out = []
    for line in (kok / "kuyruk" / f"{ad}.jsonl").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(json.loads(line))
    return out


def kuyruk_damgasi(kok: Path, ad: str):
    p = kok / "runs" / "kuyruklar" / f"{ad}.KUYRUK.done"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def baslamis(kok: Path, ad: str) -> bool:
    return (kok / "runs" / "kuyruklar" / f"{ad}.baslat.json").exists()


def aile_kosumlari(kok: Path, adlar: list) -> list:
    out = []
    runs = kok / "runs"
    if not runs.exists():
        return out
    for d in runs.iterdir():
        if not d.is_dir() or d.name == "kuyruklar":
            continue
        cfg = d / "config.json"
        if not cfg.exists():
            continue
        try:
            c = json.loads(cfg.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if c.get("kuyruk") not in adlar:
            continue
        dn = d / ".done"
        j = json.loads(dn.read_text(encoding="utf-8")) if dn.exists() else None
        out.append({"dir": d, "cfg": c, "done": j})
    return out


def asama_gecti(kok: Path, adlar: list, k: int) -> bool:
    for ad in adlar:
        rap = kok / "sonuclar" / f"beklenti-{ad}-asama{k}.md"
        if rap.exists():
            ilk = rap.read_text(encoding="utf-8").splitlines()[0] if rap.stat().st_size else ""
            if "**GEÇTİ" in ilk:
                return True
    return False


def yetim_damgala(kok: Path, adlar: list, oturum: str | None = None) -> list:
    """Bu oturumda henüz kuyruk başlatılmadan çağrılır: aileye ait damgasız koşumlar önceki (kopan) oturumdandır.
    PID denetimi YAPILMAZ (eski VM'nin PID'leri yeni VM'de anlamsız); kosum.py'nin rc=96 damgasıyla aynı biçim.
    Yalnız BU ailenin koşumlarına dokunur (paralel sekmelerin kuyrukları farklı aile → etkilenmez)."""
    out = []
    for r in aile_kosumlari(kok, adlar):
        if r["done"] is None:
            atomik_json(r["dir"] / ".done", {"rc": 96, "hata": "yetim: Colab oturumu koptu, damga yoktu (defter devam hücresi)",
                                              "bitis": simdi(), "metrics_var": False, "metrics": None})
            try:
                (r["dir"] / ".pid").unlink()
            except OSError:
                pass
            out.append(r["dir"].name)
            bildirim_yaz(kok, f"YETİM-DAMGA {r['dir'].name} rc=96 (Colab oturumu koptu)", oturum)
    return out


def devam_plani(kok: Path, ad: str, uygula: bool = True, oturum: str | None = None) -> dict:
    """Ailenin durumuna göre ne yapılacağını söyler (uygula=True ise yetim damgalar ve devam kuyruğunu YAZAR).
    durum: 'durdu' | 'bitti' | 'baslat' | 'devam' | 'kapat' (her satır bitmiş ama kuyruk damgası yok)."""
    adlar = aile(kok, ad)
    for x in adlar:
        d = kuyruk_damgasi(kok, x)
        if d and d.get("durduruldu_asama") is not None:
            return {"durum": "durdu", "kuyruk": x, "asama": d["durduruldu_asama"], "aile": adlar}
    son = adlar[-1]
    if kuyruk_damgasi(kok, son) is not None:
        return {"durum": "bitti", "kuyruk": son, "aile": adlar}
    if not baslamis(kok, son):
        return {"durum": "baslat", "kuyruk": son, "aile": adlar}
    # başlamış ama damgasız → önceki oturum koptu
    S = satirlar(kok, ad)
    asamalar = sorted({int(j.get("asama", 3)) for j in S})
    son_asama = asamalar[-1]
    k_gecti = 0
    for k in asamalar[:-1]:
        if asama_gecti(kok, adlar, k):
            k_gecti = k
        else:
            break
    tum_kapilar = all(asama_gecti(kok, adlar, k) for k in asamalar[:-1])
    yetimler = yetim_damgala(kok, adlar, oturum) if uygula else []
    if tum_kapilar:
        bitmis = {(r["cfg"].get("varyant"), str(r["cfg"].get("seed")), int(r["cfg"].get("asama", 3)))
                  for r in aile_kosumlari(kok, adlar)
                  if r["done"] and r["done"].get("rc") == 0 and r["done"].get("metrics_var")}
        kalan = [j for j in S if int(j.get("asama", 3)) == son_asama
                 and (j.get("varyant"), str(j.get("seed")), son_asama) not in bitmis]
        neden = f"tüm ara kapılar GEÇTİ → yalnız bitmemiş aşama-{son_asama} satırları"
    else:
        kalan = [j for j in S if int(j.get("asama", 3)) > k_gecti]
        neden = f"geçilen son kapı aşama {k_gecti} → aşama {k_gecti + 1}'den itibaren bütün satırlar"
    n = len(adlar)  # ad + (n-1) devam
    yeni = f"{ad}-devam{n}"
    plan = {"durum": "devam" if kalan else "kapat", "kuyruk": yeni, "onceki": son, "aile": adlar,
            "kalan": len(kalan), "neden": neden, "yetim_damgalanan": yetimler}
    if uygula:
        (kok / "kuyruk" / f"{yeni}.jsonl").write_text("".join(json.dumps(j, ensure_ascii=False) + "\n" for j in kalan),
                                                     encoding="utf-8")
        # ön-kayıt değişmez: beklenti BAYT BAYT aynı kopya
        shutil.copyfile(kok / "kuyruk" / f"{ad}.beklenti.json", kok / "kuyruk" / f"{yeni}.beklenti.json")
        atomik_json(kok / "runs" / "kuyruklar" / f"{son}.KOPTU.json",
                    {"zaman": simdi(), "kuyruk": son, "devam_kuyrugu": yeni, "kalan_satir": len(kalan),
                     "neden": neden, "yetim_damgalanan": yetimler})
        bildirim_yaz(kok, f"KUYRUK-KOPTU {son} → devam {yeni} kalan={len(kalan)}", oturum)
    return plan


def gpu_bos() -> str:
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,name", "--format=csv,noheader"],
                       capture_output=True, text=True)
    return r.stdout.strip()


def kuyruk_kos(kok: Path, ad: str, oturum: str | None = None) -> int:
    mesgul = gpu_bos()
    if mesgul:
        raise RuntimeError(f"GPU başka süreçle meşgul: {mesgul} (aynı GPU'da ikinci eğitim yok)")
    t0 = time.time()
    arg = ["orkestra/kuyruk.py", "--kok", ".", "kos", ad, "--kademeli", "--paralel", "1"]
    if oturum:  # paralel oturum: bildirimler runs/BILDIRIM_<oturum>.log'a (colab/oturum_sar.py)
        rc = kos(["python", "colab/oturum_sar.py"] + arg, cwd=str(kok), env={**os.environ, "KC_OTURUM": oturum})
    else:
        rc = kos(["python"] + arg, cwd=str(kok))
    print(f"\n[defter] kuyruk {ad} rc={rc} · {(time.time() - t0) / 60:.1f} dk "
          f"(0 hepsi başarılı · 1 başarısız koşum var · 3 aşama kapısında DURDU)", flush=True)
    return rc


def grup_kos(kok: Path, grup: str, durda_devam: bool = False, oturum: str | None = None) -> dict:
    """KUYRUK (grup ya da tek kuyruk) içindeki kuyrukları sırayla koşar; her biri için devam_plani.
    DURDU → (varsayılan) grup durur. oturum verilirse bildirimler oturuma ayrılır (paralel sekmeler)."""
    try:
        liste = kuyruk_cozumle(kok, grup)
    except KeyError as e:
        raise KeyError(f"bilinmeyen kuyruk grubu {grup!r}; {e}") from None
    ozet = {}
    for ad in liste:
        for _ in range(20):
            plan = devam_plani(kok, ad, oturum=oturum)
            print(f"[defter] {ad}: {json.dumps({k: v for k, v in plan.items() if k != 'aile'}, ensure_ascii=False)}", flush=True)
            if plan["durum"] == "durdu":
                ozet[ad] = plan
                print(f"[defter] ✖ {plan['kuyruk']} aşama {plan['asama']} beklenti kapısında DURDU → şef teşhis eder "
                      f"(sonuclar/beklenti-{plan['kuyruk']}-asama{plan['asama']}.md). Devam YOK.", flush=True)
                if not durda_devam:
                    return ozet
                break
            if plan["durum"] == "bitti":
                ozet[ad] = plan
                break
            onceki = kuyruk_damgasi(kok, plan["kuyruk"])
            kuyruk_kos(kok, plan["kuyruk"], oturum=oturum)
            if kuyruk_damgasi(kok, plan["kuyruk"]) is None and onceki is None:
                raise RuntimeError(f"{plan['kuyruk']} damga yazmadan bitti (kuyruk süreci çöktü?) — şefe bildir")
        else:
            raise RuntimeError(f"{ad}: devam döngüsü 20 turu aştı — şefe bildir")
    return ozet


# ───────────────────────── canlı kalp atışı ─────────────────────────
def _nvidia() -> dict:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        u, t, ut, tc = [x.strip() for x in r.stdout.strip().split(",")]
        return {"gpu_bellek_mb": float(u), "gpu_bellek_toplam_mb": float(t), "gpu_kullanim_yuzde": float(ut),
                "gpu_sicaklik_c": float(tc)}
    except Exception as e:  # noqa: BLE001
        return {"gpu_hata": str(e)}


def canli_durum(kok: Path, grup: str, oturum: str, kuyruklar: list | None = None, sekme: str | None = None) -> dict:
    """kuyruklar verilirse (oturum kipi) yalnız bu kuyruk ailelerinin aktif koşumları listelenir (runs/ ortak Drive
    klasörüdür; diğer sekmelerin koşumları bu oturumun kalp atışına karışmaz) ve JSON'a 'kuyruklar'/'sekme' eklenir."""
    aktif = []
    runs = kok / "runs"
    if runs.exists():
        for d in runs.iterdir():
            if d.is_dir() and d.name != "kuyruklar" and (d / ".pid").exists() and not (d / ".done").exists():
                c = {}
                try:
                    c = json.loads((d / "config.json").read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    pass
                if kuyruklar is not None and not aile_adi_mi(c.get("kuyruk"), kuyruklar):
                    continue
                pr = {}
                try:
                    pr = json.loads((d / "progress.json").read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    pass
                hb = d / ".heartbeat"
                aktif.append({"run_id": d.name, "kuyruk": c.get("kuyruk"), "varyant": c.get("varyant"),
                              "seed": c.get("seed"), "asama": c.get("asama"),
                              "epoch": pr.get("adim"), "epoch_toplam": pr.get("epoch_toplam"),
                              "val_acc": pr.get("val_acc"), "val_acc_eniyi": pr.get("val_acc_eniyi"),
                              "asama_ad": pr.get("asama_ad"),
                              "hb_yas_sn": round(time.time() - hb.stat().st_mtime, 1) if hb.exists() else None})
    ek = {} if kuyruklar is None else {"kuyruklar": list(kuyruklar), "sekme": sekme}
    return {"zaman": simdi(), "kalp_epoch": time.time(), "oturum": oturum, "grup": grup, **ek, "aktif": aktif, **_nvidia()}


def canli_dosyasi(durum_dizin: Path, oturum: str | None = None) -> Path:
    """oturum None → durum/canli.json (v1, kc1 defteri); aksi halde durum/canli_<oturum>.json."""
    return Path(durum_dizin) / ("canli.json" if not oturum else f"canli_{oturum}.json")


def canli_baslat(kok: Path, durum_dizin: Path, grup: str, oturum: str, aralik: int = 60,
                 oturum_dosyasi: bool = False, kuyruklar: list | None = None, sekme: str | None = None):
    """oturum_dosyasi=False → v1: durum/canli.json + canli_hata.txt. True → durum/canli_<oturum>.json +
    canli_hata_<oturum>.txt (paralel sekmeler aynı dosyaya yazmaz)."""
    dur = threading.Event()
    hedef = canli_dosyasi(durum_dizin, oturum if oturum_dosyasi else None)
    hata = Path(durum_dizin) / (f"canli_hata_{oturum}.txt" if oturum_dosyasi else "canli_hata.txt")

    def dongu():
        while True:
            try:
                atomik_json(hedef, canli_durum(kok, grup, oturum, kuyruklar, sekme) if oturum_dosyasi
                            else canli_durum(kok, grup, oturum))
            except Exception as e:  # noqa: BLE001
                try:
                    hata.write_text(f"{simdi()} {e}\n", encoding="utf-8")
                except OSError:
                    pass
            if dur.wait(aralik):
                break

    t = threading.Thread(target=dongu, name="canli-kalp", daemon=True)
    t.start()
    return t, dur


def canli_kuyruklari(cj: dict, G: dict) -> set:
    """Bir canli*.json'un kapsadığı kuyruklar: v2 'kuyruklar' alanı; v1 'grup' (grup adı → gruptakiler, değilse kendisi)."""
    if cj.get("kuyruklar"):
        return set(cj["kuyruklar"])
    g = cj.get("grup")
    return set(G.get(g, [g])) if g else set()


def es_zamanli_cakisma(kok: Path, durum_dizin: Path, kuyruklar: list, oturum: str, taze_sn: float = 600) -> list:
    """Başka bir sekmenin TAZE (kalp atışı < taze_sn) ve bitmemiş canlı dosyası aynı kuyruklardan birini kapsıyor mu?
    Kapsıyorsa iki sekme aynı kuyruğu koşacak demektir (devam mantığı diğerinin koşumlarını yetim sayar) → DUR.
    Kopan bir sekmenin dosyası en geç taze_sn sonra bayatlar; yeniden açılış o kadar beklemeli. Saat: kalp_epoch (UNIX)."""
    G = gruplar(kok)
    istenen, out = set(kuyruklar), []
    for p in sorted(Path(durum_dizin).glob("canli*.json")):
        if p.name == canli_dosyasi(durum_dizin, oturum).name:
            continue
        try:
            cj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if cj.get("bitti") or cj.get("oturum") == oturum:
            continue
        yas = time.time() - float(cj.get("kalp_epoch") or 0)
        ortak = istenen & canli_kuyruklari(cj, G)
        if ortak and yas < taze_sn:
            out.append({"dosya": p.name, "oturum": cj.get("oturum"), "sekme": cj.get("sekme"), "ortak": sorted(ortak),
                        "kalp_yas_sn": round(yas)})
    return out


# ───────────────────────── son hücre: tekrar kapısı özeti (BİLGİ; karar şefte) ─────────────────────────
YEREL_CIPA = {  # SIRALAMA.md "ÖN-KAYIT — COLAB TEKRAR KAPISI" (yerel 5-seed val_acc ortalamaları ve toleranslar)
    "baseline_colab": (0.7396, 0.020, "baseline"),
    "h01_dinov2b_lp_r224_colab": (0.8223, 0.015, "H01"),
    "h02_r18in_ft224_colab": (0.8672, 0.015, "H02"),
    "h08_r18in_ft224_d4_colab": (0.8893, 0.015, "H08"),
}


def tekrar_ozeti(kok: Path, adlar: list) -> str:
    from statistics import mean, stdev
    K = [r for r in aile_kosumlari(kok, adlar) if r["done"] and r["done"].get("rc") == 0 and r["done"].get("metrics_var")]
    L = ["# Colab tekrar kapısı — ÖZET (BİLGİ; kapı kararı şefte, SIRALAMA.md ön-kaydı)",
         f"*{simdi()}* · kuyruklar: {', '.join(adlar)}", "",
         "| kol | n | Colab val_acc ort ± std | yerel | |Δ| puan | tolerans | uyum |", "|---|---|---|---|---|---|---|"]
    ort = {}
    for v, (yerel, tol, ad) in YEREL_CIPA.items():
        x = [float(r["done"]["metrics"]["val_acc"]) for r in K if r["cfg"].get("varyant") == v]
        if not x:
            L.append(f"| {ad} (`{v}`) | 0 | — | {yerel:.4f} | — | {tol * 100:.1f} | VERİ YOK |"); continue
        m = mean(x); ort[ad] = m
        s = stdev(x) if len(x) > 1 else float("nan")
        d = abs(m - yerel) * 100
        L.append(f"| {ad} (`{v}`) | {len(x)} | {m:.4f} ± {s:.4f} | {yerel:.4f} | {d:.2f} | {tol * 100:.1f} | "
                 f"{'UYUMLU' if d <= tol * 100 and len(x) == 5 else ('n<5' if len(x) < 5 else 'UYUMSUZ')} |")
    if all(k in ort for k in ("H01", "H02", "H08")):
        sira = ort["H08"] > ort["H02"] > ort["H01"]
        L += ["", f"Sıralama H08 > H02 > H01: **{'KORUNDU' if sira else 'BOZULDU'}** "
                  f"({ort['H08']:.4f} / {ort['H02']:.4f} / {ort['H01']:.4f})"]
    bt = [float(r["done"]["metrics"]["test_acc"]) for r in K
          if r["cfg"].get("varyant") == "baseline_colab" and "test_acc" in (r["done"]["metrics"] or {})]
    if bt:
        mb = mean(bt)
        L += ["", f"baseline_colab test_acc ort {mb:.4f} (n={len(bt)}) ∈ [0,7218; 0,754]: "
                  f"**{'EVET' if 0.7218 <= mb <= 0.754 else 'HAYIR'}** (karar verici: replikasyon_kapisi.py)"]
    # FP32 denetimi: her koşumda fp32_durum.json ve TF32 kapalı
    eksik, acik = [], []
    for r in aile_kosumlari(kok, adlar):
        f = r["dir"] / "fp32_durum.json"
        if not f.exists():
            eksik.append(r["dir"].name); continue
        j = json.loads(f.read_text(encoding="utf-8"))
        for anahtar in ("baslangic", "bitis"):
            st = j.get(anahtar) or {}
            if st.get("cudnn.allow_tf32") is True or st.get("cuda.matmul.allow_tf32") is True:
                acik.append(f"{r['dir'].name}:{anahtar}")
    L += ["", f"FP32 denetimi: fp32_durum.json eksik {len(eksik)} {eksik[:5]} · TF32 açık görülen {len(acik)} {acik[:5]}"]
    return "\n".join(L) + "\n"
