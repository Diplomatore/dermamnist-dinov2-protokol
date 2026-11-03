# -*- coding: utf-8 -*-
"""
tahlil_tur4_ezber.py — Tur-4 ezber/sızıntı teşhisi (deney-tahlilci; YALNIZ VAL, CPU; karar vermez).
Soru: H10'un (ve H11/H08'in) H02c'ye karşı kazancının ne kadarı lezyonu train'de görülmüş (S) val görüntülerinden?
- Tabaka-standartlaştırılmış Δ_S ve Δ_N: tabaka = sınıf × dx_type × kaynak (degerlendir.tabakali_fark ile aynı tabakalar);
  ağırlık = tabakanın N sayısı (yalnız hem S hem N içeren tabakalar). Δ(S−N)_tab = Δ_S_tab − Δ_N_tab.
- Doz-yanıt: S görüntüleri için aynı lezyonun train'deki görüntü sayısı (1, 2, ≥3) → Δ ACC (ham), N ayrı satır.
- Sınıf × S/N ve kaynak × S/N ham Δ'lar (bilgi).
Bootstrap: 2.000, RNG 0, tur3_olcum/tahlil_tur4 ile aynı W. Colab kolları (--kok).
Kullanım (WS kökü): PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I scripts/tahlil_tur4_ezber.py --kok "<DRIVE_KOK>/3-1"
Çıktı: sonuclar/tahlil_tur4_ezber.json
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "scripts"))
import tahlil_tur4 as T4  # noqa: E402

SINIF = T4.SINIF
NB = 2000
KIYAS = [("h02", "h10"), ("h02", "h11"), ("h02", "h08"), ("h08", "h10"), ("h01", "h10"), ("h02", "h01")]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--kok", required=True)
    a = ap.parse_args()
    kok = Path(a.kok)
    y = np.load(WS / "data/medmnist/dermamnist.npz")["val_labels"].ravel().astype(int)
    N = len(y)
    tr_say, val_lez, vimg = Counter(), {}, {}
    with open(WS / "data/splits/dermamnist_lezyon_esleme.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "train":
                tr_say[r["lesion_id"]] += 1
            elif r["split"] == "val":
                val_lez[int(r["index"])] = r["lesion_id"]
                vimg[int(r["index"])] = r["image_id"]
    ntr = np.array([tr_say.get(val_lez[i], 0) for i in range(N)])
    S = ntr > 0
    meta = {}
    with open(WS / "data/meta/HAM10000_metadata.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            meta[r["image_id"]] = r
    dxt = np.array([meta[vimg[i]]["dx_type"] for i in range(N)])
    kay = np.array([meta[vimg[i]]["dataset"] for i in range(N)])
    tab = np.array([f"{y[i]}|{dxt[i]}|{kay[i]}" for i in range(N)])
    tabakalar = [(tab == k) & S for k in sorted(set(tab)) if ((tab == k) & S).any() and ((tab == k) & ~S).any()]
    tabakalarN = [(tab == k) & ~S for k in sorted(set(tab)) if ((tab == k) & S).any() and ((tab == k) & ~S).any()]

    d = {}
    for ad in ("h01", "h02", "h08", "h10", "h11"):
        P, yy, seed, _, _ = T4.yukle(kok, T4.KOL[ad])
        assert np.array_equal(yy, y) and seed == [0, 1, 2, 3, 4]
        d[ad] = (P.argmax(2) == y[None]).mean(0)

    rng = np.random.default_rng(0)
    IDX = rng.integers(0, N, size=(NB, N))
    W = np.zeros((NB, N))
    for b in range(NB):
        W[b] = np.bincount(IDX[b], minlength=N)

    def ga(x):
        return [float(np.quantile(x, .025)), float(np.quantile(x, .975))]

    def std_tab(f):
        """N-ağırlıklı tabaka-standartlaştırılmış ort. (S ve N için ayrı; aynı tabaka ağırlıkları) + bootstrap."""
        top = {"S": 0.0, "N": 0.0}; w = 0.0
        topb = {"S": np.zeros(NB), "N": np.zeros(NB)}; wb = np.zeros(NB)
        for mS, mN in zip(tabakalar, tabakalarN):
            wk = mN.sum()
            top["S"] += wk * f[mS].mean(); top["N"] += wk * f[mN].mean(); w += wk
            sS, sN = W[:, mS].sum(1), W[:, mN].sum(1)
            ok = (sS > 0) & (sN > 0)
            topb["S"] += np.where(ok, sN * (W[:, mS] @ f[mS]) / np.maximum(sS, 1), 0)
            topb["N"] += np.where(ok, sN * (W[:, mN] @ f[mN]) / np.maximum(sN, 1), 0)
            wb += np.where(ok, sN, 0)
        return {k: (top[k] / w, topb[k] / wb) for k in ("S", "N")}

    sonuc = {"n_S": int(S.sum()), "n_N": int((~S).sum()),
             "S_ntr_dagilim": {str(k): int(v) for k, v in sorted(Counter(ntr[S].tolist()).items())},
             "S_payi_sinif": {SINIF[c]: [int((S & (y == c)).sum()), int((y == c).sum())] for c in range(7)},
             "S_payi_kaynak": {k: [int((S & (kay == k)).sum()), int((kay == k).sum())] for k in sorted(set(kay))},
             "kiyas": {}}
    binler = {"N (0)": ntr == 0, "S ntr=1": ntr == 1, "S ntr=2": ntr == 2, "S ntr>=3": ntr >= 3}
    for A, B in KIYAS:
        f = d[B] - d[A]
        e = {}
        st = std_tab(f)
        e["tab_std_dS"] = {"nokta": float(100 * st["S"][0]), "ga": [100 * x for x in ga(st["S"][1])]}
        e["tab_std_dN"] = {"nokta": float(100 * st["N"][0]), "ga": [100 * x for x in ga(st["N"][1])]}
        e["tab_std_dS_eksi_dN"] = {"nokta": float(100 * (st["S"][0] - st["N"][0])), "ga": [100 * x for x in ga(st["S"][1] - st["N"][1])]}
        e["doz"] = {}
        for bn, m in binler.items():
            if m.sum() == 0:
                continue
            bs = (W[:, m] @ f[m]) / np.maximum(W[:, m].sum(1), 1)
            e["doz"][bn] = {"n": int(m.sum()), "dACC_puan": float(100 * f[m].mean()), "ga": [100 * x for x in ga(bs)]}
        e["sinif_SN"] = {SINIF[c]: {"dS": float(100 * f[S & (y == c)].mean()) if (S & (y == c)).any() else None,
                                    "dN": float(100 * f[~S & (y == c)].mean()) if (~S & (y == c)).any() else None,
                                    "net_S_goruntu": float(f[S & (y == c)].sum()), "net_N_goruntu": float(f[~S & (y == c)].sum())}
                         for c in range(7)}
        e["kaynak_SN"] = {k: {"dS": float(100 * f[S & (kay == k)].mean()) if (S & (kay == k)).any() else None,
                              "dN": float(100 * f[~S & (kay == k)].mean()) if (~S & (kay == k)).any() else None}
                          for k in sorted(set(kay))}
        sonuc["kiyas"][f"{B}-{A}"] = e
        print(f"{B}−{A}: tabStd ΔS {e['tab_std_dS']['nokta']:+.2f} [{e['tab_std_dS']['ga'][0]:+.2f}; {e['tab_std_dS']['ga'][1]:+.2f}] "
              f"ΔN {e['tab_std_dN']['nokta']:+.2f} [{e['tab_std_dN']['ga'][0]:+.2f}; {e['tab_std_dN']['ga'][1]:+.2f}] "
              f"ΔS−ΔN {e['tab_std_dS_eksi_dN']['nokta']:+.2f} [{e['tab_std_dS_eksi_dN']['ga'][0]:+.2f}; {e['tab_std_dS_eksi_dN']['ga'][1]:+.2f}] | doz "
              + " ".join(f"{k}:{v['dACC_puan']:+.1f}[{v['ga'][0]:+.1f};{v['ga'][1]:+.1f}](n{v['n']})" for k, v in e["doz"].items()))
        print("   sınıf ΔS/ΔN (net görüntü S/N): " + " ".join(
            f"{c}:{v['dS'] if v['dS'] is None else round(v['dS'],1)}/{v['dN'] if v['dN'] is None else round(v['dN'],1)}({v['net_S_goruntu']:+.1f}/{v['net_N_goruntu']:+.1f})"
            for c, v in e["sinif_SN"].items()))
    print(json.dumps({k: sonuc[k] for k in ("n_S", "n_N", "S_ntr_dagilim", "S_payi_sinif", "S_payi_kaynak")}, ensure_ascii=False))
    (WS / "sonuclar" / "tahlil_tur4_ezber.json").write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
