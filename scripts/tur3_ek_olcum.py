# -*- coding: utf-8 -*-
"""tur3_ek_olcum.py — H08/H09 kart ikincilleri (YALNIZ VAL, CPU, karar vermez): D_kk = Δmolemax-dışı − Δmolemax (tur3_olcum ile aynı W, RNG 0, 2.000),
kaynak başına ACC Δ [GA], seçilen epoch / son-3 epoch artışı (öldürme e), train kaybı denetimi, renk çarpanları, τ/T özetleri.
Kullanım (WS kökü): env/.venv/Scripts/python.exe scripts/tur3_ek_olcum.py . sonuclar/tur3_h08h09_ek_olcum.json"""
# Tur-3 H08/H09 ek ölçümler (yalnız val; tur3_olcum ile aynı W). Kart ikincilleri: D_kk GA, seçilen epoch, öldürme (e), kayıp denetimi, renk çarpanları.
import sys, json, csv, glob
from pathlib import Path
import numpy as np
WS = Path(sys.argv[1]); sys.path.insert(0, str(WS / "scripts")); sys.path.insert(0, str(WS / "varyantlar"))
import tur3_olcum as T
npz = np.load(WS / "data/medmnist/dermamnist.npz"); y = npz["val_labels"].ravel().astype(int); N = len(y)
vimg = {}
with open(WS / "data/splits/dermamnist_lezyon_esleme.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["split"] == "val": vimg[int(r["index"])] = r["image_id"]
meta = {}
with open(WS / "data/meta/HAM10000_metadata.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f): meta[r["image_id"]] = r
kay = np.array([meta[vimg[i]]["dataset"] for i in range(N)]); MM = kay == "vidir_molemax"
rng = np.random.default_rng(0); IDX = rng.integers(0, N, size=(2000, N)); W = np.zeros((2000, N))
for b in range(2000): W[b] = np.bincount(IDX[b], minlength=N)
def d_of(v): P, yy, s, k = T.yukle(v); assert np.array_equal(yy, y) and s == [0,1,2,3,4]; return (P.argmax(2) == y[None]).mean(0), P
def mm(d): return d[~MM].mean(), (W[:, ~MM] @ d[~MM]) / W[:, ~MM].sum(1), d[MM].mean(), (W[:, MM] @ d[MM]) / W[:, MM].sum(1)
out = {}
dA, _ = d_of("h02_r18in_ft224")
for ad, v in [("h08", "h08_r18in_ft224_d4"), ("h09", "h09_r18in_ft224_renk02")]:
    dB, P = d_of(v)
    a1, a1b, a2, a2b = mm(dA); b1, b1b, b2, b2b = mm(dB)
    D = (b1 - a1) - (b2 - a2); Db = (b1b - a1b) - (b2b - a2b)
    kaynak = {}
    for k in sorted(set(kay)):
        m = kay == k
        kaynak[k] = {"n": int(m.sum()), "h02": float(dA[m].mean()), ad: float(dB[m].mean()), "delta": float(dB[m].mean() - dA[m].mean()),
                     "ga": [float(np.quantile((W[:, m] @ (dB[m]-dA[m])) / W[:, m].sum(1), q)) for q in (.025, .975)]}
    out[ad] = {"D_kk": float(D), "D_kk_ga": [float(np.quantile(Db, .025)), float(np.quantile(Db, .975))], "D_kk_p_tek_gt0": float((np.sum(Db <= 0)+1)/2001), "kaynak": kaynak}
# kosum başına
def mets(v):
    r = []
    for d in sorted(glob.glob(str(WS / "runs" / f"*_{v}_s?"))):
        c = json.load(open(Path(d)/"config.json", encoding="utf-8"))
        if int(c.get("asama", 3)) != 3: continue
        m = json.load(open(Path(d)/"metrics.json", encoding="utf-8")); g = json.load(open(Path(d)/"gecmis.json", encoding="utf-8")) if (Path(d)/"gecmis.json").exists() else None
        r.append((Path(d).name, m, g))
    return r
def gk(e, *ks):
    for k in ks:
        if k in e: return e[k]
for v in ["h02_r18in_ft224", "h08_r18in_ft224_d4", "h09_r18in_ft224_renk02"]:
    rows = []
    for name, m, g in mets(v):
        tl = [gk(e, "train_loss") for e in g] if g else None
        va = [gk(e, "val_acc") for e in g] if g else None
        rows.append({"run": name, "val_acc": m["val_acc"], "en_iyi_epoch": m.get("en_iyi_epoch"), "en_iyi_epoch_auc": m.get("en_iyi_epoch_aucsecim"),
                     "val_acc_son": m.get("val_acc_son"), "val_acc_aucsecim": m.get("val_acc_aucsecim"), "epoch_sure": m.get("epoch_sure_ort_sn"),
                     "egitim_sure": m.get("egitim_sure_sn"), "train_loss_son": m.get("train_loss_son"), "aug_sapma": m.get("augment_frekans_maks_sapma"),
                     "tl_e1_5_10_30": [tl[i] for i in (0, 4, 9, 29)] if tl else None, "va_son3": va[-3:] if va else None,
                     "son3_artiyor": bool(va[-1] > va[-2] > va[-3]) if va else None, "tepe_mb": m.get("tepe_gpu_bellek_mb"),
                     "renk": {k: m[k] for k in m if k.startswith("renk_carpan")}, "ilk_adim_loss": m.get("ilk_adim_loss"), "param": m.get("param_sayisi")})
    out[v] = rows
for v in ["h08k_h08_logitayar", "h09k_h09_logitayar"]:
    out[v] = [{k: m.get(k) for k in ["tau_kat_1","tau_kat_2","tau_kat_3","tau_kat_4","tau_kat_5","tau_tamval","kat_sapma","val_acc","val_dengeli_acc","val_acc_tamval","val_dengeli_acc_tamval","val_iyimserlik_dengeli","val_ece","val_acc_ebeveyn"]} | {"run": n} for n, m, g in mets(v)]
for v in ["rt_h08_sicaklik", "rt_h09_sicaklik"]:
    out[v] = [{k: m.get(k) for k in ["T_tamval","T_kat_medyan","val_ece","val_ece_ham","val_nll","val_nll_ham","acc_degisim","dengeli_degisim"]} | {"run": n} for n, m, g in mets(v)]
for v in ["h08_r18in_ft224_d4kapali_duman","h09_r18in_ft224_renk02kapali_duman","h08_r18in_ft224_d4_duman","h09_r18in_ft224_renk02_duman"]:
    out[v] = [{"run": n, "ilk_adim_loss": m.get("ilk_adim_loss")} for n, m, g in mets(v)]
out["h02_duman_ilk_adim_loss"] = json.load(open(WS/"runs/20261010-204010_h02_r18in_ft224_duman_s0/metrics.json", encoding="utf-8"))["ilk_adim_loss"]
json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
