# -*- coding: utf-8 -*-
"""tahlil_tur4_cval.py — Tur-4: DermaMNIST-C val alt kümesi (resmî val ∩ C-val, 573 görüntü; lezyonu resmî train'de YOK)
üzerinde Colab kollarının ACC/dengeli farkları. Yeniden eğitim YOK (resmî-train modelleri). YALNIZ VAL; test okunmaz.
Bootstrap 2.000 RNG 0 (alt küme içinde görüntü yeniden örneklemesi), kesin tek yönlü seed-eşleştirilmiş t.
Kullanım: PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I scripts/tahlil_tur4_cval.py --kok "<DRIVE_KOK>/3-1"
Çıktı: sonuclar/tahlil_tur4_cval.json"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
from scipy import stats
WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "scripts"))
import tahlil_tur4 as T4  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--kok", required=True); a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
y = np.load(WS / "data/medmnist/dermamnist.npz")["val_labels"].ravel().astype(int)
cv = set(int(r["index_resmi"]) for r in csv.DictReader(open(WS / "data/splits/dermamnist_c_esleme.csv", encoding="utf-8"))
         if r["split_C"] == "val" and r["split_resmi"] == "val")
m = np.array([i in cv for i in range(len(y))]); idx = np.where(m)[0]; n = len(idx)
P = {}
for ad in ("baseline", "h01", "h02", "h08", "h10", "h11"):
    Pa, yy, seed, _, _ = T4.yukle(Path(a.kok), T4.KOL[ad]); assert np.array_equal(yy, y) and seed == [0, 1, 2, 3, 4]
    P[ad] = Pa[:, idx]
yc = y[idx]
rng = np.random.default_rng(0); I = rng.integers(0, n, size=(2000, n))
def den(t): return float(np.mean([(t[yc == c] == c).mean() for c in range(7) if (yc == c).any()]))
out = {"n": n, "nv_payi": float((yc == 5).mean()), "sinif_n": np.bincount(yc, minlength=7).tolist(), "kol": {}, "kiyas": {}}
for ad in P:
    acc = (P[ad].argmax(2) == yc[None]).mean(1)
    out["kol"][ad] = {"acc_ort": float(acc.mean()), "acc_std": float(acc.std(ddof=1)),
                      "dengeli_ort": float(np.mean([den(P[ad][s].argmax(1)) for s in range(5)]))}
for A, B in [("h02", "h10"), ("h02", "h11"), ("h02", "h08"), ("h08", "h10"), ("h01", "h10"), ("h10", "h11"), ("baseline", "h10")]:
    dA = (P[A].argmax(2) == yc[None]).mean(0); dB = (P[B].argmax(2) == yc[None]).mean(0); f = dB - dA
    bs = f[I].mean(1)
    aA = (P[A].argmax(2) == yc[None]).mean(1); aB = (P[B].argmax(2) == yc[None]).mean(1)
    out["kiyas"][f"{B}-{A}"] = {"dACC_puan": float(100 * f.mean()), "ga": [float(100 * np.quantile(bs, .025)), float(100 * np.quantile(bs, .975))],
                                "p_t_B>A": float(stats.ttest_rel(aB, aA, alternative="greater").pvalue), "sd_seed_puan": float(100 * (aB - aA).std(ddof=1))}
    k = out["kiyas"][f"{B}-{A}"]
    print(f"C-val {B}−{A}: Δ {k['dACC_puan']:+.2f} [{k['ga'][0]:+.2f}; {k['ga'][1]:+.2f}] p_t {k['p_t_B>A']:.3g}")
print(json.dumps({k: v for k, v in out.items() if k != "kiyas"}, ensure_ascii=False))
(WS / "sonuclar" / "tahlil_tur4_cval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
