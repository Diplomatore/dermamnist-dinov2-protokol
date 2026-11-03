# -*- coding: utf-8 -*-
"""F1 — ablasyon zinciri × üç değerlendirme (resmî test / C0 / C1).
Kaynaklar:
  [T]  sonuclar/k5_test_sonuc.json : kollar.<kol>.seed.acc, kollar.<kol>.ort/std.acc,
                                     kiyas.{h10,h08}-h02.acc (delta, ga), C0.kol.<kol>.acc_ort/acc_std,
                                     C0.kiyas.{h10,h08}-h02, C0.cogunluk_tabani
  [C1] sonuclar/k5_c1_sonuc.json   : C1.kol.<kol>.acc_ort/acc_std, C1.kiyas.{h10,h08}-h02.acc,
                                     C1.kosumlar → runs_colab/<run>/metrics.json:test_acc (C-test tek açılış)
  veri manifesti: resmî test çoğunluk tabanı = nv 1341 / 2005 (data/data_manifest.md satır 67, 71)
C0 seed noktaları JSON'da yok → çizilmez (ort ± std).
"""
import json
import statistics as st
from _ortak import DW, RC, dat_yaz, eng, jload

KOLLAR = ["baseline", "h01", "h02", "h10", "h08"]   # x = 0..4
JIT = [-0.16, -0.08, 0.0, 0.08, 0.16]

T = jload("sonuclar/k5_test_sonuc.json")
C = jload("sonuclar/k5_c1_sonuc.json")["C1"]

# resmî test
oz, sd = [], []
for k, kol in enumerate(KOLLAR):
    v = T["kollar"][kol]
    oz.append([k, k, f"{v['ort']['acc']:.6f}", f"{v['std']['acc']:.6f}"])
    for j, a in enumerate(v["seed"]["acc"]):
        sd.append([f"{k + JIT[j]:.3f}", k, f"{a:.6f}"])
dat_yaz("fig_ablasyon_test_ozet.dat", ["x", "k", "m", "s"], oz, "[T] kollar.<kol>.ort/std.acc")
dat_yaz("fig_ablasyon_test_seed.dat", ["x", "k", "acc"], sd, "[T] kollar.<kol>.seed.acc")

# C0 (ort ± std)
oz = [[k, k, f"{T['C0']['kol'][kol]['acc_ort']:.6f}", f"{T['C0']['kol'][kol]['acc_std']:.6f}"]
      for k, kol in enumerate(KOLLAR)]
dat_yaz("fig_ablasyon_c0_ozet.dat", ["x", "k", "m", "s"], oz, "[T] C0.kol.<kol>.acc_ort/acc_std")

# C1 (ort ± std + seed: koşum metrics.json, ort/std JSON ile birebir doğrulanır)
oz, sd = [], []
for k, kol in enumerate(KOLLAR):
    v = C["kol"][kol]
    seeds = [json.loads((RC / r / "metrics.json").read_text(encoding="utf-8"))["test_acc"]
             for r in C["kosumlar"][kol]]
    assert abs(st.mean(seeds) - v["acc_ort"]) < 1e-9 and abs(st.stdev(seeds) - v["acc_std"]) < 1e-9
    oz.append([k, k, f"{v['acc_ort']:.6f}", f"{v['acc_std']:.6f}"])
    for j, a in enumerate(seeds):
        sd.append([f"{k + JIT[j]:.3f}", k, f"{a:.6f}"])
dat_yaz("fig_ablasyon_c1_ozet.dat", ["x", "k", "m", "s"], oz, "[C1] C1.kol.<kol>.acc_ort/acc_std")
dat_yaz("fig_ablasyon_c1_seed.dat", ["x", "k", "acc"], sd, "runs_colab/<C1.kosumlar>/metrics.json:test_acc")

# çoğunluk tabanları (kesikli çizgi)
nv_satir = next(s for s in (DW / "data/data_manifest.md").read_text(encoding="utf-8").splitlines()
                if "melanocytic nevi (nv) |" in s)
nv_test = int([x.strip() for x in nv_satir.split("|")][5].replace(".", ""))
taban_test = nv_test / T["n"]   # nv test sayısı (data_manifest) / n
taban_c = T["C0"]["cogunluk_tabani"]
dat_yaz("fig_ablasyon_taban.dat", ["x", "test", "c"],
        [[-0.5, f"{taban_test:.6f}", f"{taban_c:.6f}"], [4.5, f"{taban_test:.6f}", f"{taban_c:.6f}"]],
        "test: nv 1341 / [T] n (data_manifest satır 67, 71); c: [T] C0.cogunluk_tabani")


# panel içi Δ notları (pp, %95 GA)
def satir(d, g):
    return f"{eng(d)} [{eng(g[0])}, {eng(g[1])}]"


notlar = []
for panel, kay in [("test", T["kiyas"]), ("c0", None), ("c1", C["kiyas"])]:
    r = []
    for c in ["h10-h02", "h08-h02"]:
        if panel == "c0":
            x = T["C0"]["kiyas"][c]
            d, g = x["delta_puan"], x["ga_puan"]
        else:
            d, g = 100 * kay[c]["acc"]["delta"], [100 * y for y in kay[c]["acc"]["ga"]]
        r.append(satir(d, g))
    notlar.append([panel] + r)
dat_yaz("fig_ablasyon_not.dat", ["panel", "h10", "h08"], notlar,
        "[T] kiyas.{h10,h08}-h02.acc ×100; [T] C0.kiyas.*.delta_puan/ga_puan; [C1] C1.kiyas.*.acc ×100")
