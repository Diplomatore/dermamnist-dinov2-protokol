#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
kosum.py — Tek bir deney koşumunu SARMALAR: run klasörü açar, komutu koşturur, stdout/stderr'i
log'a yazar, çalışırken KALP ATIŞI (.heartbeat) bırakır, metrics.json'ı doğrular ve bitişte
DAMGA (.done) + BİLDİRİM satırı (runs/BILDIRIM.log) yazar.
"Bitti mi?" sorusunun tek cevabı damgadır (bkz. references/kosum-izleme.md).

Kullanım:
  python kosum.py --kok <WS> --varyant baseline --seed 0 --etiket "PatchTST resmi" \
      -- python rakip/run.py --seed 0 --out {RUN_DIR}/metrics.json

  {RUN_DIR} yer tutucusu komutta run klasörünün mutlak yoluna çevrilir. Koşulan komut
  metrics.json'ı {RUN_DIR}/metrics.json'a düz bir JSON sözlük olarak yazmalıdır:
      {"val_accuracy": 0.9312, "egitim_sure_sn": 812}
  Yazmazsa koşum BAŞARISIZ sayılır (rc=99, komut 0 döndürse bile) — sessiz başarısızlık yakalayıcı.

  --gpu-zorunlu : koşum öncesi torch.cuda.is_available() doğrulanır; yoksa başlatılmaz (rc=97).
  --ebeveyn RUN : ablasyon ağacı için ebeveyn koşum id'si.
  --kuyruk AD   : kuyruk.py tarafından doldurulur (hangi kuyruğa ait).

Damga: {RUN_DIR}/.done  → {"rc":0,"sure_sn":..,"bitis":..,"metrics_var":true,"metrics":{..}}
Kalp atışı: {RUN_DIR}/.heartbeat (her 15 sn) → {"zaman":..,"gecen_sn":..,"log_satir":N}
Durum:
  python kosum.py --kok <WS> durum [--yetim-damgala]
     BİTTİ rc / KOŞUYOR (kalp atışı N sn önce) / ASILI (kalp atışı bayat, pid canlı) /
     ÖLDÜ damgasız (pid yok, damga yok; --yetim-damgala ile rc=96 damgası yazılır)

UZUN KOŞUMU DOĞRUDAN BU BETİKLE FOREGROUND'DA BAŞLATMA — kuyruk.py + kosum-izleme.md protokolü.
"""
import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

HEARTBEAT_SN = 15
ASILI_ESIK_SN = 600  # kalp atışı bu kadar bayatsa "ASILI"


def simdi():
    return datetime.now().isoformat(timespec="seconds")


def atomik_yaz(yol: Path, veri: dict):
    tmp = yol.with_suffix(yol.suffix + ".tmp")
    tmp.write_text(json.dumps(veri, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, yol)


def bildir(kok: Path, satir: str):
    """runs/BILDIRIM.log — her koşum olayı tek satır; şef ve izleyiciler okur."""
    try:
        with open(kok / "runs" / "BILDIRIM.log", "a", encoding="utf-8") as f:
            f.write(f"{simdi()} {satir}\n")
    except OSError:
        pass


def yeni_run_id(runs: Path, varyant: str, seed):
    zaman = datetime.now().strftime("%Y%m%d-%H%M%S")
    ad = f"{zaman}_{varyant}_s{seed}"
    i, cand = 0, ad
    while (runs / cand).exists():
        i += 1
        cand = f"{ad}-{i}"
    return cand


def gpu_dogrula():
    try:
        import torch  # type: ignore
        ok = torch.cuda.is_available()
        print(f"[kosum] torch {torch.__version__} cuda_available={ok}")
        return ok
    except Exception as e:
        print(f"[kosum] torch import edilemedi: {e}")
        return False


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


def run_durumu(d: Path):
    """Tek run klasörünün durumu → (kod, açıklama). kod: BITTI | KOSUYOR | ASILI | OLDU | BASLAMAMIS"""
    done, pidf, hb = d / ".done", d / ".pid", d / ".heartbeat"
    if done.exists():
        j = json.loads(done.read_text(encoding="utf-8"))
        return "BITTI", (f"BİTTİ rc={j.get('rc')} metrics={'var' if j.get('metrics_var') else 'YOK'} "
                         f"{j.get('sure_sn', '?')}s"), j
    if pidf.exists():
        try:
            pid = int(pidf.read_text().strip() or 0)
        except ValueError:
            pid = 0
        hb_yas = None
        if hb.exists():
            hb_yas = time.time() - hb.stat().st_mtime
        canli = pid_canli(pid)
        if canli and hb_yas is not None and hb_yas > ASILI_ESIK_SN:
            return "ASILI", f"ASILI pid={pid} kalp atışı {int(hb_yas)}s önce (eşik {ASILI_ESIK_SN}s)", None
        if canli:
            return "KOSUYOR", f"KOŞUYOR pid={pid} kalp atışı {int(hb_yas) if hb_yas is not None else '?'}s önce", None
        return "OLDU", f"ÖLDÜ damgasız (pid {pid} yok, .done yok)", None
    if (d / "config.json").exists():
        return "BASLAMAMIS", "BAŞLAMAMIŞ/YARIM (config var, pid/damga yok)", None
    return "BASLAMAMIS", "?", None


def durum(kok: Path, yetim_damgala=False):
    runs = kok / "runs"
    if not runs.exists():
        print("runs/ yok")
        return 0
    sayac = {}
    for d in sorted(p for p in runs.iterdir() if p.is_dir() and p.name != "kuyruklar"):
        kod, acik, _ = run_durumu(d)
        if kod == "OLDU" and yetim_damgala:
            atomik_yaz(d / ".done", {"rc": 96, "hata": "yetim: süreç öldü, damga yoktu", "bitis": simdi(),
                                     "metrics_var": False, "metrics": None})
            bildir(kok, f"YETİM-DAMGA {d.name} rc=96")
            acik += " → rc=96 damgalandı"
        sayac[kod] = sayac.get(kod, 0) + 1
        print(f"{d.name:<48} {acik}")
    print("ÖZET: " + ", ".join(f"{k}={v}" for k, v in sorted(sayac.items())) if sayac else "(koşum yok)")
    return 0


def kural_denetle(rd: Path, kural: dict, t0: float):
    """ERKEN DURDURMA: koşulan kod {RUN_DIR}/progress.json'a ilerleme yazarsa
    ({"adim": 120, "loss": 0.71, "val_accuracy": 0.61, ...}) kurallar canlı sınanır.
    Dönüş: None (devam) veya neden (str → süreç öldürülür, rc=95)."""
    pj = rd / "progress.json"
    if not pj.exists():
        if kural.get("ilerleme_zorunlu_sn") and time.time() - t0 > kural["ilerleme_zorunlu_sn"]:
            return f"{kural['ilerleme_zorunlu_sn']}s geçti, progress.json hiç yazılmadı (kod ilerleme raporlamıyor)"
        return None
    try:
        pr = json.loads(pj.read_text(encoding="utf-8"))
    except Exception:
        return None
    yas = time.time() - pj.stat().st_mtime
    if kural.get("sabit_sn") and yas > kural["sabit_sn"]:
        return f"progress.json {int(yas)}s güncellenmedi (eğitim takıldı; eşik {kural['sabit_sn']}s)"
    if kural.get("nan_kes", True):
        for k, v in pr.items():
            if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
                return f"{k}={v} (NaN/Inf) — ıraksama"
    adim = pr.get("adim", pr.get("step", 0)) or 0
    if adim >= kural.get("min_adim", 0):
        for k, alt in (kural.get("alt_sinir") or {}).items():
            if k in pr and isinstance(pr[k], (int, float)) and pr[k] < alt:
                return f"adım {adim}: {k}={pr[k]} < alt sınır {alt} (beklenti ihlali — şans düzeyi/öğrenmiyor)"
        for k, ust in (kural.get("ust_sinir") or {}).items():
            if k in pr and isinstance(pr[k], (int, float)) and pr[k] > ust:
                return f"adım {adim}: {k}={pr[k]} > üst sınır {ust} (patlama veya sızıntı şüphesi)"
    return None


def heartbeat_dongusu(rd: Path, t0: float, dur: threading.Event, proc=None, kural=None):
    log = rd / "stdout.log"
    while not dur.wait(HEARTBEAT_SN):
        try:
            n = sum(1 for _ in open(log, "rb")) if log.exists() else 0
            hb = {"zaman": simdi(), "gecen_sn": round(time.time() - t0), "log_satir": n}
            pj = rd / "progress.json"
            if pj.exists():
                try:
                    hb["ilerleme"] = json.loads(pj.read_text(encoding="utf-8"))
                except Exception:
                    pass
            atomik_yaz(rd / ".heartbeat", hb)
            if kural and proc is not None and proc.poll() is None:
                neden = kural_denetle(rd, kural, t0)
                if neden:
                    atomik_yaz(rd / ".erken_durdurma.json", {"zaman": simdi(), "neden": neden, "kural": kural})
                    with open(log, "a", encoding="utf-8") as f:
                        f.write(f"\n# {simdi()} ERKEN DURDURMA (kosum.py --kural): {neden}\n")
                    proc.kill()
        except OSError:
            pass


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kok", required=True, help="deney workspace kökü")
    p.add_argument("--varyant", default="baseline")
    p.add_argument("--seed", default="0")
    p.add_argument("--etiket", default="")
    p.add_argument("--ebeveyn", default=None)
    p.add_argument("--hipotez", default=None)
    p.add_argument("--kuyruk", default=None)
    p.add_argument("--gpu-zorunlu", action="store_true")
    p.add_argument("--zaman-siniri-dk", type=float, default=None)
    p.add_argument("--asama", type=int, default=3, help="1 duman · 2 ön-uçuş · 3 tam")
    p.add_argument("--kural", default=None,
                   help='erken durdurma JSON: {"nan_kes":true,"min_adim":200,"alt_sinir":{"val_accuracy":0.55},'
                        '"ust_sinir":{"loss":50},"sabit_sn":900,"ilerleme_zorunlu_sn":600}')
    p.add_argument("--yetim-damgala", action="store_true", help="durum ile: ölü/damgasız koşumları rc=96 damgala")
    p.add_argument("komut", nargs=argparse.REMAINDER)
    a = p.parse_args()

    kok = Path(a.kok).resolve()
    if a.komut and a.komut[0] == "durum":
        return durum(kok, a.yetim_damgala)
    komut = a.komut[1:] if a.komut and a.komut[0] == "--" else a.komut
    if not komut:
        p.error("koşulacak komut yok ('--' sonrasına yaz) veya 'durum'")

    runs = kok / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    run_id = yeni_run_id(runs, a.varyant, a.seed)
    rd = runs / run_id
    rd.mkdir()
    komut = [c.replace("{RUN_DIR}", str(rd)) for c in komut]

    if a.gpu_zorunlu and not gpu_dogrula():
        atomik_yaz(rd / ".done", {"rc": 97, "hata": "GPU zorunlu ama CUDA yok", "bitis": simdi(),
                                  "metrics_var": False, "metrics": None})
        bildir(kok, f"DURDU {run_id} rc=97 GPU yok")
        print(f"[kosum] DURDU: GPU zorunlu, CUDA yok. run={run_id}")
        return 97

    kural = None
    if a.kural:
        try:
            kural = json.loads(a.kural)
        except json.JSONDecodeError:
            kp = Path(a.kural)
            kural = json.loads(kp.read_text(encoding="utf-8")) if kp.exists() else None
            if kural is None:
                print(f"[kosum] UYARI: --kural ayrıştırılamadı, erken durdurma kapalı: {a.kural}")
    cfg = {"run_id": run_id, "varyant": a.varyant, "seed": a.seed, "etiket": a.etiket,
           "ebeveyn": a.ebeveyn, "hipotez": a.hipotez, "kuyruk": a.kuyruk, "asama": a.asama,
           "kural": kural, "komut": komut, "cwd": str(kok), "baslangic": simdi(), "makine": platform.node()}
    atomik_yaz(rd / "config.json", cfg)

    t0 = time.time()
    dur = threading.Event()
    rc = None
    with open(rd / "stdout.log", "w", encoding="utf-8", errors="replace") as log:
        log.write(f"# {simdi()} $ {' '.join(shlex.quote(c) for c in komut)}\n")
        log.flush()
        proc = subprocess.Popen(komut, cwd=str(kok), stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, "RUN_DIR": str(rd), "SEED": str(a.seed),
                                     "PYTHONIOENCODING": "utf-8"})
        (rd / ".pid").write_text(str(proc.pid), encoding="utf-8")
        atomik_yaz(rd / ".heartbeat", {"zaman": simdi(), "gecen_sn": 0, "log_satir": 1})
        hb = threading.Thread(target=heartbeat_dongusu, args=(rd, t0, dur, proc, kural), daemon=True)
        hb.start()
        bildir(kok, f"BAŞLADI {run_id} pid={proc.pid} varyant={a.varyant} seed={a.seed}")
        print(f"[kosum] başladı run={run_id} pid={proc.pid}")
        try:
            rc = proc.wait(timeout=a.zaman_siniri_dk * 60 if a.zaman_siniri_dk else None)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = 98
            log.write(f"\n# {simdi()} ZAMAN SINIRI AŞILDI ({a.zaman_siniri_dk} dk) — öldürüldü\n")
        except KeyboardInterrupt:
            proc.kill()
            rc = 130
    dur.set()
    sure = round(time.time() - t0, 1)
    if (rd / ".erken_durdurma.json").exists():
        rc = 95  # canlı beklenti ihlali: kural tarafından öldürüldü
        neden = json.loads((rd / ".erken_durdurma.json").read_text(encoding="utf-8")).get("neden", "")
        bildir(kok, f"ERKEN-DURDURULDU {run_id} rc=95 {neden}")

    mfile = rd / "metrics.json"
    metrics_var, metrics = False, {}
    if mfile.exists():
        try:
            metrics = json.loads(mfile.read_text(encoding="utf-8"))
            metrics_var = isinstance(metrics, dict) and len(metrics) > 0
        except Exception as e:
            metrics = {"_hata": f"metrics.json okunamadı: {e}"}
    if rc == 0 and not metrics_var:
        rc = 99  # sessiz başarısızlık
    atomik_yaz(rd / ".done", {"rc": rc, "sure_sn": sure, "bitis": simdi(), "metrics_var": metrics_var,
                              "metrics": metrics if metrics_var else None})
    for f in (".pid",):
        try:
            (rd / f).unlink()
        except OSError:
            pass
    ozet = ", ".join(f"{k}={v}" for k, v in list(metrics.items())[:6]) if metrics_var else "METRİK YOK"
    bildir(kok, f"BİTTİ {run_id} rc={rc} sure={sure}s {ozet}")
    print(f"[kosum] bitti run={run_id} rc={rc} sure={sure}s  {ozet}")
    return 0 if rc == 0 else (rc if isinstance(rc, int) and 0 < rc < 256 else 1)


if __name__ == "__main__":
    sys.exit(main())
