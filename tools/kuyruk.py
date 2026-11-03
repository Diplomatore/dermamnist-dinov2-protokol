#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
kuyruk.py — Koşum KUYRUĞU: sub-agent'ların yazdığı kuyruk dosyasını (kuyruk/<ad>.jsonl) ŞEF
arka planda başlatır; her satır kosum.py ile sırayla koşar; bitişte KUYRUK DAMGASI yazılır.
Bitişten haberdar olmanın iki kanalı bu betiktedir (bkz. references/kosum-izleme.md):
  KANAL 1 — `kos` alt komutu ŞEF tarafından Bash(run_in_background=true) ile başlatılır;
            süreç kuyruk bitince çıkar → harness şefi TEK bildirimle uyandırır.
  KANAL 2 — `izle` alt komutu Monitor aracıyla koşar; her koşumun BAŞLADI/BİTTİ/ASILI/ÖLDÜ
            olayını stdout'a satır satır basar, kuyruk bitince (veya kuyruk süreci ölünce) çıkar.
            Sessizlik = başarı DEĞİLDİR: çökme, asılma ve yetim koşum da olay üretir.

Kuyruk dosyası: WS/kuyruk/<ad>.jsonl — her satır bir koşum:
  {"varyant":"baseline","seed":0,"etiket":"PatchTST resmi","hipotez":null,"ebeveyn":null,
   "gpu_zorunlu":true,"zaman_siniri_dk":90,"komut":["python","scripts/rakip_kos.py","--seed","0",
   "--out","{RUN_DIR}/metrics.json"]}

Alt komutlar:
  kuyruk.py --kok WS ekle <ad> --varyant V --seed S [--etiket ..] [--hipotez H] [--ebeveyn R]
            [--gpu-zorunlu] [--zaman-siniri-dk N] -- <komut...>       # satır ekler (sub-agent kullanır)
  kuyruk.py --kok WS kos <ad> [--paralel N]     # ŞEF, arka planda. GPU işi: paralel=1 (varsayılan)
  kuyruk.py --kok WS izle <ad> [--aralik 5] [--asili-sn 600]   # ŞEF, Monitor ile. Olay akışı.
  kuyruk.py --kok WS bekle <ad>                 # sessiz bekleme; tek satır + rc. (bg Bash alternatifi)
  kuyruk.py --kok WS liste                      # kuyruklar ve durumları

Kuyruk damgası: WS/runs/kuyruklar/<ad>.KUYRUK.done → {"toplam":N,"basarili":k,"basarisiz":m,...}
Başlatma kaydı: WS/runs/kuyruklar/<ad>.baslat.json → {"pid":..,"baslangic":..}
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

HERE = Path(__file__).resolve().parent
KOSUM = HERE / "kosum.py"


def simdi():
    return datetime.now().isoformat(timespec="seconds")


def atomik_yaz(yol: Path, veri: dict):
    yol.parent.mkdir(parents=True, exist_ok=True)
    tmp = yol.with_suffix(yol.suffix + ".tmp")
    tmp.write_text(json.dumps(veri, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, yol)


def pid_canli(pid: int) -> bool:
    if pid <= 0:
        return False
    if platform.system() == "Windows":
        r = subprocess.run(f'tasklist /FI "PID eq {pid}" /NH', shell=True, capture_output=True, text=True)
        return str(pid) in r.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def yollar(kok: Path, ad: str):
    return {
        "jsonl": kok / "kuyruk" / f"{ad}.jsonl",
        "baslat": kok / "runs" / "kuyruklar" / f"{ad}.baslat.json",
        "done": kok / "runs" / "kuyruklar" / f"{ad}.KUYRUK.done",
        "bildirim": kok / "runs" / "BILDIRIM.log",
    }


def oku_kuyruk(jsonl: Path):
    if not jsonl.exists():
        raise SystemExit(f"HATA: kuyruk dosyası yok: {jsonl}")
    satirlar = []
    for i, line in enumerate(jsonl.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        j = json.loads(line)
        if "komut" not in j or not isinstance(j["komut"], list):
            raise SystemExit(f"HATA: satır {i+1}: 'komut' listesi zorunlu")
        satirlar.append(j)
    return satirlar


def bildir(kok: Path, satir: str):
    with open(kok / "runs" / "BILDIRIM.log", "a", encoding="utf-8") as f:
        f.write(f"{simdi()} {satir}\n")


# ───────────────────────── ekle ─────────────────────────
def cmd_ekle(a, kok):
    y = yollar(kok, a.ad)
    y["jsonl"].parent.mkdir(parents=True, exist_ok=True)
    komut = a.komut
    if not komut:
        raise SystemExit("HATA: '--' sonrasına komut yaz")
    j = {"varyant": a.varyant, "seed": a.seed, "etiket": a.etiket, "hipotez": a.hipotez,
         "ebeveyn": a.ebeveyn, "gpu_zorunlu": bool(a.gpu_zorunlu), "asama": int(a.asama),
         "zaman_siniri_dk": a.zaman_siniri_dk, "kural": a.kural, "komut": komut}
    with open(y["jsonl"], "a", encoding="utf-8") as f:
        f.write(json.dumps(j, ensure_ascii=False) + "\n")
    n = len(oku_kuyruk(y["jsonl"]))
    print(f"eklendi → {y['jsonl']} (toplam {n} koşum)")
    return 0


# ───────────────────────── kos ─────────────────────────
def kosum_komutu(kok: Path, ad: str, j: dict):
    cmd = [sys.executable, str(KOSUM), "--kok", str(kok), "--varyant", str(j.get("varyant", "baseline")),
           "--seed", str(j.get("seed", 0)), "--kuyruk", ad]
    if j.get("etiket"):
        cmd += ["--etiket", str(j["etiket"])]
    if j.get("hipotez"):
        cmd += ["--hipotez", str(j["hipotez"])]
    if j.get("ebeveyn"):
        cmd += ["--ebeveyn", str(j["ebeveyn"])]
    if j.get("gpu_zorunlu"):
        cmd += ["--gpu-zorunlu"]
    if j.get("zaman_siniri_dk"):
        cmd += ["--zaman-siniri-dk", str(j["zaman_siniri_dk"])]
    cmd += ["--asama", str(j.get("asama", 3))]
    if j.get("kural"):
        cmd += ["--kural", j["kural"] if isinstance(j["kural"], str) else json.dumps(j["kural"])]
    return cmd + ["--"] + list(j["komut"])


def cmd_kos(a, kok):
    y = yollar(kok, a.ad)
    satirlar = oku_kuyruk(y["jsonl"])
    if y["done"].exists() and not a.zorla:
        raise SystemExit(f"HATA: {y['done']} zaten var — bu kuyruk koşulmuş. Yeni ad ver veya --zorla.")
    if y["baslat"].exists():
        eski = json.loads(y["baslat"].read_text(encoding="utf-8"))
        if pid_canli(int(eski.get("pid", 0))) and not a.zorla:
            raise SystemExit(f"HATA: kuyruk '{a.ad}' zaten koşuyor (pid {eski['pid']}).")
    if y["done"].exists():
        y["done"].unlink()
    atomik_yaz(y["baslat"], {"pid": os.getpid(), "baslangic": simdi(), "toplam": len(satirlar),
                             "paralel": a.paralel, "makine": platform.node()})
    (kok / "runs").mkdir(exist_ok=True)
    beklenti = kok / "kuyruk" / f"{a.ad}.beklenti.json"
    if a.kademeli and not beklenti.exists():
        raise SystemExit(f"HATA: --kademeli için ön-kayıtlı beklenti dosyası şart: {beklenti} "
                         f"(şablon: templates/beklenti.json). Beklentisiz kademeli koşum yapılmaz.")
    asamalar = sorted(set(int(j.get("asama", 3)) for j in satirlar)) if a.kademeli else [None]
    bildir(kok, f"KUYRUK-BAŞLADI {a.ad} toplam={len(satirlar)} paralel={a.paralel} pid={os.getpid()}"
                f"{' kademeli=' + ','.join(map(str, asamalar)) if a.kademeli else ''}")
    print(f"[kuyruk] {a.ad}: {len(satirlar)} koşum, paralel={a.paralel}"
          + (f", kademeli aşamalar {asamalar}" if a.kademeli else ""))
    t0 = time.time()
    sonuc = []
    durduruldu_asama = None
    kapi_raporlari = []

    def tek(i_j):
        i, j = i_j
        cmd = kosum_komutu(kok, a.ad, j)
        r = subprocess.run(cmd, cwd=str(kok))
        return {"sira": i, "varyant": j.get("varyant"), "seed": j.get("seed"),
                "asama": int(j.get("asama", 3)), "rc": r.returncode}

    def grubu_kos(grup):
        if a.paralel <= 1:
            for i, j in grup:
                s = tek((i, j))
                sonuc.append(s)
                print(f"[kuyruk] {len(sonuc)}/{len(satirlar)} a{s['asama']} {s['varyant']} s{s['seed']} rc={s['rc']}")
                if a.ilk_hatada_dur and s["rc"] != 0:
                    print("[kuyruk] ilk hatada dur: kalan koşumlar başlatılmadı")
                    return False
        else:
            with ThreadPoolExecutor(max_workers=a.paralel) as ex:
                for s in ex.map(tek, grup):
                    sonuc.append(s)
                    print(f"[kuyruk] a{s['asama']} {s['varyant']} s{s['seed']} rc={s['rc']}")
        return True

    for k in asamalar:
        grup = [(i, j) for i, j in enumerate(satirlar) if k is None or int(j.get("asama", 3)) == k]
        if k is not None:
            print(f"[kuyruk] ── AŞAMA {k} başlıyor ({len(grup)} koşum) ──")
            bildir(kok, f"AŞAMA-BAŞLADI {a.ad} asama={k} kosum={len(grup)}")
        devam = grubu_kos(grup)
        if not devam:
            durduruldu_asama = k; break
        if k is not None and k != asamalar[-1]:
            # AŞAMA KAPISI: ilk sonuçlar beklendiği gibi mi? Değilse büyük koşuma geçilmez.
            kapi = subprocess.run([sys.executable, str(HERE / "beklenti_kontrol.py"), "--kok", str(kok),
                                   "--kuyruk", a.ad, "--asama", str(k)], cwd=str(kok), capture_output=True, text=True)
            rapor = kok / "sonuclar" / f"beklenti-{a.ad}-asama{k}.md"
            kapi_raporlari.append({"asama": k, "rc": kapi.returncode, "rapor": str(rapor)})
            if kapi.returncode in (2, 3):
                durduruldu_asama = k
                bildir(kok, f"KUYRUK-DURDU {a.ad} asama={k} beklenti-kapisi rc={kapi.returncode} rapor={rapor}")
                print(f"[kuyruk] ✖ AŞAMA {k} beklenti kapısını GEÇEMEDİ (rc={kapi.returncode}) — sonraki aşama BAŞLATILMADI. Rapor: {rapor}")
                break
            bildir(kok, f"AŞAMA-GEÇTİ {a.ad} asama={k} rc={kapi.returncode} rapor={rapor}")
            print(f"[kuyruk] ✔ AŞAMA {k} beklenti kapısı geçti (rc={kapi.returncode}); sıradaki aşama başlıyor")
    basarili = sum(1 for s in sonuc if s["rc"] == 0)
    ozet = {"kuyruk": a.ad, "toplam": len(satirlar), "koşulan": len(sonuc), "basarili": basarili,
            "basarisiz": len(sonuc) - basarili, "sure_sn": round(time.time() - t0, 1),
            "bitis": simdi(), "kademeli": bool(a.kademeli), "durduruldu_asama": durduruldu_asama,
            "kapi_raporlari": kapi_raporlari, "sonuclar": sonuc}
    atomik_yaz(y["done"], ozet)
    # dashboard'u best-effort güncelle (proje kökü: WS'nin iki üstü = <proje>/deney_workspace/<ws>)
    try:
        dg = HERE / "dashboard_guncelle.py"
        if dg.exists():
            subprocess.run([sys.executable, str(dg), "--proje-koku", str(kok.parents[1])], capture_output=True, timeout=60)
    except Exception:
        pass
    if durduruldu_asama is not None:
        bildir(kok, f"KUYRUK-BİTTİ {a.ad} DURDURULDU asama={durduruldu_asama} basarili={basarili}/{len(satirlar)} sure={ozet['sure_sn']}s")
        print(f"[kuyruk] DURDURULDU {a.ad} (aşama {durduruldu_asama}); {basarili}/{len(satirlar)} koşum başarılı → {y['done']}")
        return 3
    bildir(kok, f"KUYRUK-BİTTİ {a.ad} basarili={basarili}/{len(satirlar)} sure={ozet['sure_sn']}s")
    print(f"[kuyruk] BİTTİ {a.ad}: {basarili}/{len(satirlar)} başarılı, {ozet['sure_sn']}s → {y['done']}")
    return 0 if basarili == len(satirlar) else 1


# ───────────────────────── izle / bekle ─────────────────────────
def kuyruk_runlari(kok: Path, ad: str):
    out = {}
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
        except Exception:
            continue
        if c.get("kuyruk") == ad:
            out[d.name] = (d, c)
    return out


def run_hali(d: Path, asili_sn: int):
    done, pidf, hb = d / ".done", d / ".pid", d / ".heartbeat"
    if done.exists():
        j = json.loads(done.read_text(encoding="utf-8"))
        m = j.get("metrics") or {}
        ozet = ", ".join(f"{k}={v}" for k, v in list(m.items())[:4]) if m else "METRİK YOK"
        return "BITTI", f"rc={j.get('rc')} {j.get('sure_sn','?')}s {ozet}"
    pid = 0
    if pidf.exists():
        try:
            pid = int(pidf.read_text().strip() or 0)
        except ValueError:
            pid = 0
    hb_yas = (time.time() - hb.stat().st_mtime) if hb.exists() else None
    if pid and pid_canli(pid):
        if hb_yas is not None and hb_yas > asili_sn:
            return "ASILI", f"pid={pid} kalp atışı {int(hb_yas)}s önce"
        return "KOSUYOR", f"pid={pid}"
    if pidf.exists():
        return "OLDU", f"pid {pid} yok, damga yok"
    return "BASLAMAMIS", ""


def cmd_izle(a, kok):
    y = yollar(kok, a.ad)
    gorulen = {}        # run_id → son bildirilen hal
    asili_son = {}      # run_id → son ASILI bildirimi zamanı
    print(f"IZLE {a.ad} başladı {simdi()} (aralık {a.aralik}s, asılı eşiği {a.asili_sn}s)")
    while True:
        # kuyruk bitti mi?
        if y["done"].exists():
            j = json.loads(y["done"].read_text(encoding="utf-8"))
            # kuyruğa ait ama henüz bildirilmemiş bitişleri bas
            for rid, (d, c) in sorted(kuyruk_runlari(kok, a.ad).items()):
                hal, acik = run_hali(d, a.asili_sn)
                if hal == "BITTI" and gorulen.get(rid) != "BITTI":
                    print(f"BİTTİ {rid} {acik}")
                    gorulen[rid] = "BITTI"
            if j.get("durduruldu_asama") is not None:
                rap = (j.get("kapi_raporlari") or [{}])[-1].get("rapor", "?")
                print(f"KUYRUK-DURDU {a.ad} asama={j['durduruldu_asama']} BEKLENTİ İHLALİ — büyük koşum başlatılmadı. "
                      f"Rapor: {rap} → teşhis: beklenti-ihlali.md; düzeltme sonra yeni kuyruk (<ad>-v2)")
                return 3
            print(f"KUYRUK-BİTTİ {a.ad} basarili={j['basarili']}/{j['toplam']} basarisiz={j['basarisiz']} "
                  f"sure={j['sure_sn']}s → sonraki adım: kosum.py durum + leaderboard.py")
            return 0 if j["basarisiz"] == 0 else 1
        # kuyruk süreci yaşıyor mu?
        if y["baslat"].exists():
            b = json.loads(y["baslat"].read_text(encoding="utf-8"))
            if not pid_canli(int(b.get("pid", 0))):
                # kısa gecikme: done yazılmak üzere olabilir
                time.sleep(2)
                if not y["done"].exists():
                    print(f"KUYRUK-SÜRECİ-ÖLDÜ {a.ad} pid={b.get('pid')} damga yok → kosum.py durum --yetim-damgala")
                    return 2
                continue
        for rid, (d, c) in sorted(kuyruk_runlari(kok, a.ad).items()):
            hal, acik = run_hali(d, a.asili_sn)
            onceki = gorulen.get(rid)
            if hal == "KOSUYOR" and onceki is None:
                print(f"BAŞLADI {rid} {acik} asama={c.get('asama', 3)} varyant={c.get('varyant')} seed={c.get('seed')}")
            elif hal == "BITTI" and onceki != "BITTI":
                erken = d / ".erken_durdurma.json"
                ek = ""
                if erken.exists():
                    try:
                        ek = " ERKEN-DURDURULDU: " + json.loads(erken.read_text(encoding="utf-8")).get("neden", "")
                    except Exception:
                        ek = " ERKEN-DURDURULDU"
                print(f"BİTTİ {rid} {acik}{ek}")
            elif hal == "ASILI":
                son = asili_son.get(rid, 0)
                if time.time() - son > a.asili_sn:
                    print(f"ASILI {rid} {acik} → teşhis gerekir (log: {d / 'stdout.log'})")
                    asili_son[rid] = time.time()
            elif hal == "OLDU" and onceki != "OLDU":
                print(f"ÖLDÜ {rid} {acik} → yetim; kuyruk devam ediyorsa sıradaki koşum başlar")
            if hal != "BASLAMAMIS":
                gorulen[rid] = hal
        time.sleep(a.aralik)


def cmd_bekle(a, kok):
    y = yollar(kok, a.ad)
    while True:
        if y["done"].exists():
            j = json.loads(y["done"].read_text(encoding="utf-8"))
            if j.get("durduruldu_asama") is not None:
                print(f"KUYRUK-DURDU {a.ad} asama={j['durduruldu_asama']} beklenti ihlali — büyük koşum başlatılmadı")
                return 3
            print(f"KUYRUK-BİTTİ {a.ad} basarili={j['basarili']}/{j['toplam']} sure={j['sure_sn']}s")
            return 0 if j["basarisiz"] == 0 else 1
        if y["baslat"].exists():
            b = json.loads(y["baslat"].read_text(encoding="utf-8"))
            if not pid_canli(int(b.get("pid", 0))):
                time.sleep(2)
                if not y["done"].exists():
                    print(f"KUYRUK-SÜRECİ-ÖLDÜ {a.ad} pid={b.get('pid')} damga yok")
                    return 2
        time.sleep(a.aralik)


def cmd_liste(a, kok):
    kd = kok / "kuyruk"
    if not kd.exists():
        print("kuyruk/ yok")
        return 0
    for f in sorted(kd.glob("*.jsonl")):
        ad = f.stem
        y = yollar(kok, ad)
        n = len(oku_kuyruk(f))
        if y["done"].exists():
            j = json.loads(y["done"].read_text(encoding="utf-8"))
            st = f"BİTTİ {j['basarili']}/{j['toplam']} ({j['bitis']})"
        elif y["baslat"].exists():
            b = json.loads(y["baslat"].read_text(encoding="utf-8"))
            st = f"KOŞUYOR pid={b['pid']} ({b['baslangic']})" if pid_canli(int(b["pid"])) else "ÖLDÜ damgasız"
        else:
            st = "BAŞLATILMAMIŞ"
        print(f"{ad:<30} {n:>3} koşum  {st}")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kok", required=True)
    sp = p.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("ekle"); e.add_argument("ad"); e.add_argument("--varyant", required=True)
    e.add_argument("--seed", required=True); e.add_argument("--etiket", default="")
    e.add_argument("--hipotez"); e.add_argument("--ebeveyn"); e.add_argument("--gpu-zorunlu", action="store_true")
    e.add_argument("--zaman-siniri-dk", type=float)
    e.add_argument("--asama", type=int, default=3, help="1 duman · 2 ön-uçuş · 3 tam (kademeli koşum sırası)")
    e.add_argument("--kural", default=None, help="kosum.py erken-durdurma kural JSON'u (progress.json'a bakar)")
    k = sp.add_parser("kos"); k.add_argument("ad"); k.add_argument("--paralel", type=int, default=1)
    k.add_argument("--ilk-hatada-dur", action="store_true"); k.add_argument("--zorla", action="store_true")
    k.add_argument("--kademeli", action="store_true",
                   help="aşamaları sırayla koş; her aşama sonunda beklenti_kontrol.py kapısı; ihlalde DUR")
    i = sp.add_parser("izle"); i.add_argument("ad"); i.add_argument("--aralik", type=float, default=5)
    i.add_argument("--asili-sn", type=int, default=600)
    b = sp.add_parser("bekle"); b.add_argument("ad"); b.add_argument("--aralik", type=float, default=10)
    sp.add_parser("liste")
    # '--' sonrası koşulacak komuttur; argparse'a hiç gösterilmez (positional 'ad' ile çakışmasın)
    argv = sys.argv[1:]
    komut = []
    if "--" in argv:
        i = argv.index("--")
        argv, komut = argv[:i], argv[i + 1:]
    a = p.parse_args(argv)
    a.komut = komut
    kok = Path(a.kok).resolve()
    return {"ekle": cmd_ekle, "kos": cmd_kos, "izle": cmd_izle, "bekle": cmd_bekle, "liste": cmd_liste}[a.cmd](a, kok)


if __name__ == "__main__":
    sys.exit(main())
