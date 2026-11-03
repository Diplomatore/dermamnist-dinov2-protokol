"""Kaldığı yerden devam eden (HTTP Range) + MD5 doğrulamalı MedMNIST indirici.
Kullanım: python scripts/indir_devamli.py 128 224   (64 için yalnız doğrulama yapılır)
Kopan bağlantıda yeniden dener; yarım dosyayı SİLMEZ, üzerine ekler. MD5 = medmnist.INFO."""
import hashlib, json, os, sys, time, urllib.request
import medmnist
KOK = os.path.join(os.path.dirname(__file__), "..", "data", "medmnist")
INFO = medmnist.INFO["dermamnist"]

def md5(yol):
    h = hashlib.md5()
    with open(yol, "rb") as f:
        for p in iter(lambda: f.read(1 << 20), b""): h.update(p)
    return h.hexdigest()

def indir(boyut, deneme=60):
    ad = f"dermamnist_{boyut}.npz"
    url = INFO[f"url_{boyut}"]; beklenen = INFO[f"MD5_{boyut}"]
    yol = os.path.join(KOK, ad)
    for d in range(deneme):
        mevcut = os.path.getsize(yol) if os.path.exists(yol) else 0
        try:
            bas = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60)
            toplam = int(bas.headers["Content-Length"])
            if mevcut == toplam: break
            if mevcut > toplam: raise SystemExit(f"{ad}: yerel {mevcut} > uzak {toplam} — elle incele")
            req = urllib.request.Request(url, headers={"Range": f"bytes={mevcut}-"})
            with urllib.request.urlopen(req, timeout=60) as r, open(yol, "ab") as f:
                if mevcut and r.status != 206: raise RuntimeError(f"Range desteklenmedi (status {r.status})")
                t0, n = time.time(), 0
                while True:
                    p = r.read(1 << 20)
                    if not p: break
                    f.write(p); n += len(p)
                    if n % (50 << 20) < (1 << 20):
                        print(f"[{ad}] {(mevcut+n)/1e6:.0f}/{toplam/1e6:.0f} MB · {n/1e3/max(time.time()-t0,1e-3):.0f} kB/s", flush=True)
        except Exception as e:
            print(f"[{ad}] deneme {d+1}: {type(e).__name__}: {e} — 20 sn sonra devam", flush=True)
            time.sleep(20)
    m = md5(yol)
    ok = m == beklenen
    print(f"[{ad}] bayt={os.path.getsize(yol)} md5={m} resmi={beklenen} -> {'OK' if ok else 'UYUSMUYOR'}", flush=True)
    kayit = os.path.join(KOK, "indirme_dogrulama.json")
    d = json.load(open(kayit)) if os.path.exists(kayit) else {}
    d[ad] = {"bayt": os.path.getsize(yol), "md5": m, "md5_resmi": beklenen, "ok": ok, "zaman": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(d, open(kayit, "w"), indent=1)
    return ok

if __name__ == "__main__":
    sonuc = {b: indir(int(b)) for b in sys.argv[1:]}
    print("SONUÇ", sonuc); sys.exit(0 if all(sonuc.values()) else 1)
