#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
beklenti_kontrol.py — Kademeli koşumun AŞAMA KAPISI: bir aşamanın (duman=1, ön-uçuş=2, tam=3)
damgalı koşumlarını, KOŞUMDAN ÖNCE yazılmış beklenti dosyasına (kuyruk/<ad>.beklenti.json)
karşı sınar. Amaç: "ilk sonuçlar beklendiği gibi mi?" sorusunu betikle cevaplamak; değilse
büyük koşuma GEÇİLMEZ (kuyruk.py kos --kademeli bu betiği aşamalar arasında kendisi çağırır).

Beklenti dosyası (ön-kayıt; şablon: templates/beklenti.json):
{
  "metrik": "val_accuracy", "yon": "max",
  "asama": {
    "1": {"aciklama": "duman: çalışıyor mu",
          "kurallar": [{"tur": "sonlu"}, {"tur": "aralik", "alt": 0.5, "ust": 1.0}]},
    "2": {"aciklama": "ön-uçuş: doğru ve beklendiği gibi mi",
          "kurallar": [{"tur": "sonlu"},
                        {"tur": "aralik", "alt": 0.88, "ust": 0.96, "kaynak": "Devlin 2019 dev %92.7±reprod"},
                        {"tur": "farkli_seedler_farkli"},
                        {"tur": "kollar_farkli"},
                        {"tur": "yon_vs", "varyant": "H02_revin", "referans": "baseline", "beklenen": "+", "en_az": 0.0},
                        {"tur": "tavan", "varyant": "H02_revin", "referans": "baseline", "en_cok": 0.02,
                         "kaynak": "literatürde RevIN kazancı ≤2 puan"},
                        {"tur": "sure_orani", "en_cok": 3.0, "referans": "baseline"}]},
    "3": {"aciklama": "tam", "kurallar": [{"tur": "sonlu"}, {"tur": "farkli_seedler_farkli"}]}
  }
}
Kural türleri: sonlu · aralik(alt,ust) · farkli_seedler_farkli · kollar_farkli · yon_vs(varyant,
referans,beklenen +/-,en_az) · tavan(varyant,referans,en_cok) · sure_orani(referans,en_cok) ·
min_kosum(n) · ikincil_aralik(metrik,alt,ust)
Kaynaksız 'aralik'/'tavan' kuralı UYARI üretir (çıpa ilkesi: beklenti kaynaklı olmalı).

Kullanım:
  python beklenti_kontrol.py --kok WS --kuyruk k3-baseline --asama 2 [--beklenti kuyruk/x.beklenti.json]
Çıkış: 0 GEÇTİ · 1 UYARI (geçti, eksik kaynak/az koşum) · 2 İHLAL (büyük koşuma geçilmez) · 3 veri yok
Rapor: WS/sonuclar/beklenti-<kuyruk>-asama<k>.md
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def kosumlar(kok: Path, kuyruk: str, asama: int):
    out = []
    for d in (kok / "runs").glob("*/"):
        cfg, done = d / "config.json", d / ".done"
        if not (cfg.exists() and done.exists()):
            continue
        c = json.loads(cfg.read_text(encoding="utf-8")); j = json.loads(done.read_text(encoding="utf-8"))
        if c.get("kuyruk") != kuyruk or int(c.get("asama", 3)) != asama:
            continue
        out.append({"run_id": c["run_id"], "varyant": c["varyant"], "seed": str(c["seed"]), "rc": j.get("rc"),
                    "sure": j.get("sure_sn"), "m": j.get("metrics") or {}})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kok", required=True); p.add_argument("--kuyruk", required=True)
    p.add_argument("--asama", type=int, required=True); p.add_argument("--beklenti", default=None)
    a = p.parse_args()
    kok = Path(a.kok).resolve()
    bpath = Path(a.beklenti) if a.beklenti else kok / "kuyruk" / f"{a.kuyruk}.beklenti.json"
    if not bpath.is_absolute():
        bpath = kok / bpath
    if not bpath.exists():
        print(f"İHLAL: beklenti dosyası yok: {bpath} — ön-kayıtsız kademeli koşum yapılmaz"); return 2
    B = json.loads(bpath.read_text(encoding="utf-8"))
    metrik, yon = B["metrik"], B.get("yon", "max")
    kurallar = (B.get("asama", {}).get(str(a.asama)) or {}).get("kurallar", [])
    K = kosumlar(kok, a.kuyruk, a.asama)
    out = kok / "sonuclar" / f"beklenti-{a.kuyruk}-asama{a.asama}.md"; out.parent.mkdir(parents=True, exist_ok=True)
    if not K:
        out.write_text(f"# Beklenti Kapısı — VERİ YOK (kuyruk {a.kuyruk}, aşama {a.asama})\n", encoding="utf-8")
        print("VERİ YOK: bu aşamaya ait damgalı koşum bulunamadı"); return 3

    bulgular = []  # (seviye, kural, mesaj) seviye: OK / UYARI / İHLAL
    basarili = [k for k in K if k["rc"] == 0 and metrik in k["m"]]
    cokme = [k for k in K if k["rc"] != 0]
    if cokme:
        bulgular.append(("İHLAL", "mekanik", f"{len(cokme)} koşum rc≠0: " + ", ".join(f"{k['run_id']}(rc={k['rc']})" for k in cokme)))
    if not basarili:
        bulgular.append(("İHLAL", "mekanik", f"'{metrik}' metriği olan başarılı koşum yok"))
    grup = defaultdict(dict)  # varyant → seed → değer
    sure = defaultdict(list)
    for k in basarili:
        grup[k["varyant"]][k["seed"]] = float(k["m"][metrik]); sure[k["varyant"]].append(k["sure"] or 0)

    def ort(v):
        return mean(grup[v].values()) if grup.get(v) else None

    for r in kurallar:
        t = r["tur"]
        if t == "sonlu":
            kotu = [f"{v}/s{s}" for v, ss in grup.items() for s, x in ss.items() if not math.isfinite(x)]
            bulgular.append(("İHLAL" if kotu else "OK", t, f"sonlu değil: {kotu}" if kotu else "tüm değerler sonlu"))
        elif t == "aralik":
            if "kaynak" not in r:
                bulgular.append(("UYARI", t, "kaynaksız aralık kuralı (çıpa ilkesi)"))
            for v, ss in grup.items():
                if r.get("varyant") and v != r["varyant"]:
                    continue
                m_ = mean(ss.values())
                ok = r["alt"] <= m_ <= r["ust"]
                bulgular.append(("OK" if ok else "İHLAL", t, f"{v}: ort {m_:.4g} {'∈' if ok else '∉'} [{r['alt']}, {r['ust']}] ({r.get('kaynak','?')})"))
        elif t == "farkli_seedler_farkli":
            for v, ss in grup.items():
                if len(ss) >= 2 and len(set(round(x, 10) for x in ss.values())) == 1:
                    bulgular.append(("İHLAL", t, f"{v}: {len(ss)} seed bit-bit aynı → seed bağlanmamış"))
                elif len(ss) >= 2:
                    bulgular.append(("OK", t, f"{v}: seed'ler farklı"))
        elif t == "kollar_farkli":
            vs = list(grup)
            ayni = [(x, y) for i, x in enumerate(vs) for y in vs[i+1:]
                    if set(grup[x]) & set(grup[y]) and all(abs(grup[x][s] - grup[y][s]) < 1e-12 for s in set(grup[x]) & set(grup[y]))]
            bulgular.append(("İHLAL" if ayni else "OK", t, f"plasebo kol: {ayni}" if ayni else "kollar birbirinden farklı"))
        elif t == "yon_vs":
            v, ref = r["varyant"], r["referans"]
            if ort(v) is None or ort(ref) is None:
                bulgular.append(("UYARI", t, f"{v} veya {ref} için veri yok")); continue
            fark = (ort(v) - ort(ref)) * (1 if yon == "max" else -1)
            ok = fark >= r.get("en_az", 0.0) if r["beklenen"] == "+" else fark <= -r.get("en_az", 0.0)
            bulgular.append(("OK" if ok else "İHLAL", t, f"{v} vs {ref}: yönlü fark {fark:+.4g}, beklenen {r['beklenen']} (en az {r.get('en_az',0)})"))
        elif t == "tavan":
            v, ref = r["varyant"], r["referans"]
            if "kaynak" not in r:
                bulgular.append(("UYARI", t, "kaynaksız tavan kuralı"))
            if ort(v) is None or ort(ref) is None:
                bulgular.append(("UYARI", t, f"{v} veya {ref} için veri yok")); continue
            fark = (ort(v) - ort(ref)) * (1 if yon == "max" else -1)
            ok = fark <= r["en_cok"]
            bulgular.append(("OK" if ok else "İHLAL", t, f"{v} vs {ref}: kazanç {fark:+.4g} {'≤' if ok else '>'} tavan {r['en_cok']} "
                             + ("" if ok else "→ LİTERATÜRDE GÖRÜLMEMİŞ KADAR İYİ: sızıntı/protokol/rakip bozulması soruşturması")))
        elif t == "sure_orani":
            ref = r["referans"]
            if not sure.get(ref):
                bulgular.append(("UYARI", t, f"{ref} süresi yok")); continue
            for v, ss in sure.items():
                if v == ref:
                    continue
                oran = mean(ss) / max(1e-9, mean(sure[ref]))
                ok = oran <= r["en_cok"]
                bulgular.append(("OK" if ok else "İHLAL", t, f"{v}/{ref} süre oranı {oran:.2f} {'≤' if ok else '>'} {r['en_cok']}"))
        elif t == "min_kosum":
            ok = len(basarili) >= r["n"]
            bulgular.append(("OK" if ok else "UYARI", t, f"{len(basarili)} başarılı koşum (en az {r['n']})"))
        elif t == "ikincil_aralik":
            mm = r["metrik"]
            for k in basarili:
                if mm in k["m"]:
                    x = float(k["m"][mm]); ok = r["alt"] <= x <= r["ust"]
                    bulgular.append(("OK" if ok else "İHLAL", t, f"{k['varyant']}/s{k['seed']} {mm}={x:.4g} {'∈' if ok else '∉'} [{r['alt']},{r['ust']}]"))
        else:
            bulgular.append(("UYARI", t, "bilinmeyen kural türü"))

    ihlal = [b for b in bulgular if b[0] == "İHLAL"]; uyari = [b for b in bulgular if b[0] == "UYARI"]
    karar = "İHLAL — BÜYÜK KOŞUMA GEÇİLMEZ" if ihlal else ("GEÇTİ (uyarılı)" if uyari else "GEÇTİ")
    rc = 2 if ihlal else (1 if uyari else 0)
    L = [f"# Beklenti Kapısı — kuyruk `{a.kuyruk}` aşama {a.asama} → **{karar}**",
         f"*{datetime.now():%Y-%m-%d %H:%M}* · metrik `{metrik}` ({yon}) · beklenti: `{bpath.name}` · {len(K)} koşum", "",
         "| varyant | n | ort | seed değerleri | ort süre (s) |", "|---|---|---|---|---|"]
    for v, ss in grup.items():
        L.append(f"| {v} | {len(ss)} | {mean(ss.values()):.4g} | {', '.join(f'{x:.4g}' for x in ss.values())} | {mean(sure[v]):.1f} |")
    L += ["", "## Kurallar", "| seviye | kural | bulgu |", "|---|---|---|"]
    for s, k, m in bulgular:
        L.append(f"| {'🔴' if s=='İHLAL' else ('🟡' if s=='UYARI' else '🟢')} {s} | {k} | {m} |")
    if ihlal:
        L += ["", "## Yapılacak", "1. Koşum kuyruğu durdu; sıradaki aşama başlatılmadı.",
              "2. Her İHLAL için hata hipotezi kur (beklenti-ihlali.md): kurulum/veri yolu/seed/metrik hesabı.",
              "3. Düzeltme sonra duman aşamasından yeniden; aynı kuyruk adı tekrar koşulmaz (`<ad>-v2`).",
              "4. Beklenti yanlışsa: kaynağıyla düzelt (eşiği sonucu geçirmek için gevşetme yasak)."]
    md = "\n".join(L) + "\n"; out.write_text(md, encoding="utf-8"); print(md); print(f"yazıldı: {out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
