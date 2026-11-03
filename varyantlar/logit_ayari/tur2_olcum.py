#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tur2_olcum.py — Tur-2 CPU kollarının (H04, H06, H04 yan kolları, R-T) kart ölçümleri. YALNIZ VAL, CPU. KARAR VERMEZ
(karar deney-tahlilcinin); öldürme kurallarının girdisi olan sayıları üretir.

- Seed-eşleştirilmiş kesin tek yönlü t (scipy.stats.ttest_rel, alternative="greater"), seed {0..4}.
- Eşleştirilmiş görüntü bootstrap'ı: 2.000, RNG 0, ağırlık matrisi scripts/tahlil_tur1.py ile AYNI
  (rng.integers(0,N,(2000,N)) → bincount); dengeli ACC bootstrap'ı aynı yöntem (seed-ortalamalı görüntü doğruluğu,
  sınıf duyarlılıkları ağırlıklı). ACC GA'sı varyantlar/ortak/eslestirilmis_bootstrap.py ile aynıdır.
- Kollar tahmin_val.npy'den okunur (H04/H06 için OOF olasılıkları). Test dosyası açılmaz (M4).
Çıktı: sonuclar/tur2_cpu_olcum.json
Kullanım (WS kökü): PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe varyantlar/logit_ayari/tur2_olcum.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from logit_ayari import ebeveyn_kosumlari, val_etiket  # noqa: E402

K, NB, RNG = 7, 2000, 0
SINIF = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def yukle(varyant, dosya="tahmin_val.npy"):
    kos = ebeveyn_kosumlari(varyant, dosya)
    assert list(kos) == [0, 1, 2, 3, 4], (varyant, list(kos))
    P = np.stack([np.load(d / dosya).astype(np.float64) for d in kos.values()])
    M = [json.loads((d / "metrics.json").read_text(encoding="utf-8")) for d in kos.values()]
    return P, M, [d.name for d in kos.values()]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    y = val_etiket()
    N = len(y)
    rng = np.random.default_rng(RNG)
    IDX = rng.integers(0, N, size=(NB, N))
    W = np.stack([np.bincount(IDX[b], minlength=N) for b in range(NB)]).astype(float)
    oh = np.stack([(y == c).astype(float) for c in range(K)], 1)

    def dengeli(t):
        return float(np.mean([(t[y == c] == c).mean() for c in range(K)]))

    def uc(P):
        d = (P.argmax(2) == y[None]).mean(0)
        acc = (d.mean(), W @ d / N)
        rec_b = (W @ (d[:, None] * oh)) / (W @ oh)
        den = (float(np.mean([d[y == c].mean() for c in range(K)])), rec_b.mean(1))
        return {"acc": acc, "dengeli_acc": den}

    def ga(a):
        return [float(np.quantile(a, .025)), float(np.quantile(a, .975))]

    kol = {"h02": ("h02_r18in_ft224", "tahmin_val.npy"), "h02_aucsecim": ("h02_r18in_ft224", "tahmin_val_aucsecim.npy"),
           "h01": ("h01_dinov2b_lp_r224", "tahmin_val.npy"), "baseline": ("baseline", "tahmin_val.npy"),
           "h04": ("h04_h02_logitayar", "tahmin_val.npy"), "h06": ("h06_h01_logitayar", "tahmin_val.npy"),
           "h04x_aucsecim": ("h04x_h02aucsecim_logitayar", "tahmin_val.npy"),
           "h04x_baseline": ("h04x_baseline_logitayar", "tahmin_val.npy")}
    D = {}
    for ad, (v, f) in kol.items():
        P, M, kos = yukle(v, f)
        T = P.argmax(2)
        D[ad] = {"P": P, "M": M, "kos": kos, "U": uc(P),
                 "acc_s": np.array([(T[s] == y).mean() for s in range(5)]),
                 "den_s": np.array([dengeli(T[s]) for s in range(5)]),
                 "nv_s": np.array([(T[s] == 5).mean() for s in range(5)]),
                 "rec": np.array([[(T[s][y == c] == c).mean() for c in range(K)] for s in range(5)])}

    def kiyas(A, B):
        a, b = D[A], D[B]
        out = {"A": A, "B": B, "kosum_B": b["kos"]}
        for m, key in (("acc", "acc_s"), ("dengeli_acc", "den_s")):
            fa, fb = a["U"][m], b["U"][m]
            fark = fb[1] - fa[1]
            t = stats.ttest_rel(b[key], a[key], alternative="greater")
            tl = stats.ttest_rel(b[key], a[key], alternative="less")
            out[m] = {"A_ort": float(a[key].mean()), "A_std": float(a[key].std(ddof=1)),
                      "B_ort": float(b[key].mean()), "B_std": float(b[key].std(ddof=1)),
                      "B_seed": b[key].tolist(), "delta": float(fb[0] - fa[0]),
                      "delta_seed": (b[key] - a[key]).tolist(), "sd_delta_seed": float((b[key] - a[key]).std(ddof=1)),
                      "p_t_tek_B>A": float(t.pvalue), "p_t_tek_B<A": float(tl.pvalue),
                      "bootstrap_ga95": ga(fark), "bootstrap_p_tek_B>A": float((np.sum(fark <= 0) + 1) / (NB + 1))}
        out["nv_payi"] = {"A": float(a["nv_s"].mean()), "B": float(b["nv_s"].mean())}
        out["duyarlilik"] = {SINIF[c]: {"A": float(a["rec"][:, c].mean()), "B": float(b["rec"][:, c].mean())} for c in range(K)}
        out["auc"] = {"A": float(np.mean([m["val_auc"] for m in a["M"]])) if "val_auc" in a["M"][0] else None,
                      "B": float(np.mean([m["val_auc"] for m in b["M"]]))}
        return out

    R = {"aciklama": "Tur-2 CPU kolları ölçümü (val; KARAR VERMEZ). Δ = B − A; bootstrap 2.000 RNG 0 (tahlil_tur1 yöntemi).",
         "H04_vs_H02": kiyas("h02", "h04"), "H06_vs_H01": kiyas("h01", "h06"),
         "H04x_aucsecim_vs_H02aucsecim": kiyas("h02_aucsecim", "h04x_aucsecim"),
         "H04x_baseline_vs_baseline": kiyas("baseline", "h04x_baseline"),
         "H06_vs_H02_ayarsiz": kiyas("h02", "h06")}
    # H04 − H06 kısıt altı kazanç farkı (keşifsel, ön-kayıtlı yön H04 > H06)
    g4 = D["h04"]["U"]["dengeli_acc"][1] - D["h02"]["U"]["dengeli_acc"][1]
    g6 = D["h06"]["U"]["dengeli_acc"][1] - D["h01"]["U"]["dengeli_acc"][1]
    n4 = D["h04"]["U"]["dengeli_acc"][0] - D["h02"]["U"]["dengeli_acc"][0]
    n6 = D["h06"]["U"]["dengeli_acc"][0] - D["h01"]["U"]["dengeli_acc"][0]
    R["H04_eksi_H06_dengeli_kazanc"] = {"delta": float(n4 - n6), "bootstrap_ga95": ga(g4 - g6),
                                        "bootstrap_p_tek_H04>H06": float((np.sum((g4 - g6) <= 0) + 1) / (NB + 1))}
    # τ ve ikinciller (koşum metrics.json'larından)
    for ad in ("h04", "h06", "h04x_aucsecim", "h04x_baseline"):
        M = D[ad]["M"]
        R[f"{ad}_tau"] = {"tau_kat": [M[0][f"tau_kat_{k}"] for k in range(1, 6)], "tau_tamval": M[0]["tau_tamval"],
                          "tau_kat_aralik": M[0]["tau_kat_aralik"], "tau_kat_medyan": M[0]["tau_kat_medyan"],
                          "tau_ortak_mi": all(m["tau_tamval"] == M[0]["tau_tamval"] for m in M),
                          "katlar_sha256": sorted(set(m["katlar_sha256"] for m in M))}
        R[f"{ad}_ikincil"] = {k: float(np.mean([m[k] for m in M])) for k in
                              ("val_acc_tamval", "val_dengeli_acc_tamval", "val_iyimserlik_dengeli", "val_makro_f1",
                               "val_makro_f1_ebeveyn", "val_ece", "val_ece_ebeveyn", "val_nll", "val_auc", "val_auc_ebeveyn",
                               "val_tahmin_nv_payi", "val_acc_molemax_disi", "val_acc_molemax_disi_ebeveyn",
                               "val_sizinti_tabakali_SN", "val_acc_tau1", "val_dengeli_acc_tau1",
                               "val_dengeli_acc_alt_bolme_std", "val_dengeli_acc_alt_bolme_ort",
                               "val_karisiklik_mel_nv", "val_karisiklik_mel_nv_ebeveyn", "val_karisiklik_bkl_nv",
                               "val_karisiklik_bkl_nv_ebeveyn", "val_karisiklik_nv_mel", "val_karisiklik_nv_mel_ebeveyn",
                               "val_karisiklik_nv_bkl", "val_karisiklik_nv_bkl_ebeveyn", "val_karar_degisen_goruntu",
                               "tau0_ozdeslik_fark") if k in M[0]}
        R[f"{ad}_ikincil"]["val_dengeli_acc_alt_bolme_std_seedler"] = [m.get("val_dengeli_acc_alt_bolme_std") for m in M]
    # R-T
    rt = {}
    for v in ("rt_h02_sicaklik", "rt_h01_sicaklik", "rt_baseline_sicaklik", "rt_h04_sicaklik"):
        kos = ebeveyn_kosumlari(v, "tahmin_val.npy")
        M = [json.loads((d / "metrics.json").read_text(encoding="utf-8")) for d in kos.values()]
        rt[v] = {"kosumlar": [d.name for d in kos.values()],
                 **{k: [m[k] for m in M] for k in ("val_ece", "val_ece_ham", "val_ece_tamval", "T_tamval", "T_kat_medyan",
                                                    "val_nll", "val_nll_ham", "val_asiri_guven", "val_asiri_guven_ham",
                                                    "acc_degisim", "dengeli_degisim")},
                 "val_ece_ort": float(np.mean([m["val_ece"] for m in M])), "val_ece_std": float(np.std([m["val_ece"] for m in M], ddof=1)),
                 "val_ece_ham_ort": float(np.mean([m["val_ece_ham"] for m in M])),
                 "val_ece_tamval_ort": float(np.mean([m["val_ece_tamval"] for m in M]))}
    R["R_T"] = rt
    out = WS / "sonuclar" / "tur2_cpu_olcum.json"
    out.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
    for k in ("H04_vs_H02", "H06_vs_H01", "H04x_aucsecim_vs_H02aucsecim", "H04x_baseline_vs_baseline", "H06_vs_H02_ayarsiz"):
        r = R[k]
        print(f"{k:<30} dengeli {r['dengeli_acc']['A_ort']:.4f}→{r['dengeli_acc']['B_ort']:.4f}±{r['dengeli_acc']['B_std']:.4f} "
              f"Δ {r['dengeli_acc']['delta']*100:+.2f} GA[{r['dengeli_acc']['bootstrap_ga95'][0]*100:+.2f},{r['dengeli_acc']['bootstrap_ga95'][1]*100:+.2f}] "
              f"p_t {r['dengeli_acc']['p_t_tek_B>A']:.3g} | ACC {r['acc']['A_ort']:.4f}→{r['acc']['B_ort']:.4f}±{r['acc']['B_std']:.4f} "
              f"Δ {r['acc']['delta']*100:+.2f} GA[{r['acc']['bootstrap_ga95'][0]*100:+.2f},{r['acc']['bootstrap_ga95'][1]*100:+.2f}] "
              f"| nv {r['nv_payi']['A']:.3f}→{r['nv_payi']['B']:.3f}")
    h = R["H04_eksi_H06_dengeli_kazanc"]
    print(f"H04 − H06 dengeli kazanç farkı {h['delta']*100:+.2f} GA[{h['bootstrap_ga95'][0]*100:+.2f},{h['bootstrap_ga95'][1]*100:+.2f}]")
    for ad in ("h04", "h06", "h04x_aucsecim", "h04x_baseline"):
        print(ad, json.dumps(R[f"{ad}_tau"], ensure_ascii=False))
        print("   ", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in R[f"{ad}_ikincil"].items()}, ensure_ascii=False))
    for v, r in rt.items():
        print(f"{v:<22} ECE ham {r['val_ece_ham_ort']:.4f} → OOF {r['val_ece_ort']:.4f} ± {r['val_ece_std']:.4f} (tam-val {r['val_ece_tamval_ort']:.4f}) T* {r['T_tamval']}")
    print(f"yazıldı {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
