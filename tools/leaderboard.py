#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
leaderboard.py — runs/*/ damgalarını tek tabloya toplar (sonuclar/leaderboard.csv) ve
varyant bazında özet çıkarır (ortalama ± std, n seed, baseline'a göre fark, eşleştirilmiş test).

Kullanım:
  python leaderboard.py --kok deney_workspace/5-1-etth1 --metrik mse --yon min
  python leaderboard.py --kok <ws> --metrik accuracy --yon max --baseline baseline --md sonuclar/leaderboard.md

Kurallar (deney-sozlesmesi.md):
  * Yalnız damgalı (.done, rc=0, metrics_var) koşumlar tabloya girer; yarım koşum girmez.
  * Aynı varyant için seed'ler eşleştirilir (aynı seed → aynı satır); eşleşmeyen seed'ler
    eşleştirilmiş testte dışlanır ve raporlanır.
  * n<3 olan varyant "YETERSİZ SEED" işaretlenir — makaleye giremez.
  * p-değeri scipy varsa Welch/eşleştirilmiş t-testi; yoksa saf Python eşleştirilmiş t (normal yaklaşımı).
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean, stdev

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def topla(kok: Path):
    satirlar = []
    for d in sorted((kok / "runs").glob("*/")):
        done = d / ".done"; cfg = d / "config.json"
        if not (done.exists() and cfg.exists()):
            continue
        j = json.loads(done.read_text(encoding="utf-8")); c = json.loads(cfg.read_text(encoding="utf-8"))
        if j.get("rc") != 0 or not j.get("metrics_var"):
            continue
        m = j.get("metrics") or {}
        satirlar.append({"run_id": c["run_id"], "varyant": c["varyant"], "seed": str(c["seed"]),
                         "hipotez": c.get("hipotez") or "", "ebeveyn": c.get("ebeveyn") or "",
                         "etiket": c.get("etiket") or "", "sure_sn": j.get("sure_sn"),
                         "makine": c.get("makine", ""), "bitis": j.get("bitis"),
                         **{k: v for k, v in m.items() if isinstance(v, (int, float))}})
    return satirlar


def yaz_csv(satirlar, yol: Path):
    if not satirlar:
        return
    anahtarlar = []
    for s in satirlar:
        for k in s:
            if k not in anahtarlar:
                anahtarlar.append(k)
    yol.parent.mkdir(parents=True, exist_ok=True)
    with open(yol, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=anahtarlar); w.writeheader(); w.writerows(satirlar)


def _t_cdf_yaklasik(t, df):
    # Student-t sağ kuyruk için yeterli doğrulukta yaklaşım (Abramowitz-Stegun tabanlı normal yaklaşımı)
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    # düzenli tamamlanmamış beta yerine sürekli kesir yaklaşımı yerine normal yaklaşımı (df>=3 için kabul edilebilir)
    z = t * (1 - 1 / (4 * df)) / math.sqrt(1 + t * t / (2 * df))
    return 0.5 * math.erfc(z / math.sqrt(2))


def eslesmis_test(a, b):
    """a, b: aynı seed sırasında iki liste. İki kuyruklu eşleştirilmiş t-testi p-değeri."""
    n = len(a)
    if n < 2:
        return float("nan"), n
    d = [x - y for x, y in zip(a, b)]
    try:
        from scipy import stats  # type: ignore
        return float(stats.ttest_rel(a, b).pvalue), n
    except Exception:
        md = mean(d); sd = stdev(d) if n > 1 else 0.0
        if sd == 0:
            return (0.0 if md != 0 else 1.0), n
        t = md / (sd / math.sqrt(n))
        return 2 * _t_cdf_yaklasik(abs(t), n - 1), n


def ozet(satirlar, metrik, yon, baseline):
    gruplar = {}
    for s in satirlar:
        if metrik in s:
            gruplar.setdefault(s["varyant"], {})[s["seed"]] = s[metrik]
    if baseline not in gruplar:
        print(f"UYARI: baseline varyantı '{baseline}' tabloda yok; farklar hesaplanmadı.")
    bs = gruplar.get(baseline, {})
    rows = []
    for v, seeds in gruplar.items():
        vals = list(seeds.values()); n = len(vals)
        mu = mean(vals); sd = stdev(vals) if n > 1 else 0.0
        ortak = sorted(set(seeds) & set(bs))
        if bs and v != baseline and len(ortak) >= 2:
            p, _ = eslesmis_test([seeds[s] for s in ortak], [bs[s] for s in ortak])
            fark = mu - mean(bs.values())
        else:
            p, fark = float("nan"), (mu - mean(bs.values()) if bs else float("nan"))
        rows.append({"varyant": v, "n_seed": n, "ort": mu, "std": sd, "min": min(vals), "max": max(vals),
                     "fark_vs_baseline": fark, "p_eslesmis": p, "ortak_seed": len(ortak),
                     "yeterli": n >= 3})
    rows.sort(key=lambda r: r["ort"], reverse=(yon == "max"))
    return rows


def md_tablo(rows, metrik, yon, baseline):
    out = [f"# Leaderboard — metrik `{metrik}` ({'yüksek iyi' if yon=='max' else 'düşük iyi'}), baseline `{baseline}`", "",
           "| # | varyant | n seed | ort ± std | min / max | Δ vs baseline | p (eşleştirilmiş) | durum |",
           "|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        p = "—" if math.isnan(r["p_eslesmis"]) else f"{r['p_eslesmis']:.3g}"
        fark = "—" if math.isnan(r["fark_vs_baseline"]) else f"{r['fark_vs_baseline']:+.4g}"
        durum = "OK" if r["yeterli"] else "YETERSİZ SEED (<3)"
        if r["varyant"] == baseline:
            durum = "BASELINE"
        out.append(f"| {i} | {r['varyant']} | {r['n_seed']} | {r['ort']:.4g} ± {r['std']:.3g} | "
                   f"{r['min']:.4g} / {r['max']:.4g} | {fark} | p={p} (n={r['ortak_seed']}) | {durum} |")
    out += ["", "_Yalnız damgalı (rc=0, metrics var) koşumlar sayıldı. n<3 varyant makaleye giremez._"]
    return "\n".join(out) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kok", required=True); p.add_argument("--metrik", required=True)
    p.add_argument("--yon", choices=["max", "min"], default="max")
    p.add_argument("--baseline", default="baseline")
    p.add_argument("--csv", default="sonuclar/leaderboard.csv"); p.add_argument("--md", default="sonuclar/leaderboard.md")
    a = p.parse_args()
    kok = Path(a.kok).resolve()
    satirlar = topla(kok)
    if not satirlar:
        print("Damgalı başarılı koşum yok."); return 1
    yaz_csv(satirlar, kok / a.csv)
    rows = ozet(satirlar, a.metrik, a.yon, a.baseline)
    md = md_tablo(rows, a.metrik, a.yon, a.baseline)
    (kok / a.md).parent.mkdir(parents=True, exist_ok=True)
    (kok / a.md).write_text(md, encoding="utf-8")
    print(md); print(f"yazıldı: {kok / a.csv}, {kok / a.md}  ({len(satirlar)} koşum)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
