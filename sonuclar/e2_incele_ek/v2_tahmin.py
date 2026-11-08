# İNCELE v2: arşiv BCN tahminlerinden bağımsız yeniden hesap + betimsel mekanizma (model ÇALIŞTIRMAZ; keşifsel)
import json, csv, sys
from pathlib import Path
import numpy as np
WS = Path(r"<WS>")
DR = Path("<DRIVE_KOK>/3-1/runs")
sys.path.insert(0, str(WS / "varyantlar"))
from e2 import kollar as KL
R = json.loads((WS / "sonuclar/e2_bcn_sonuc.json").read_text(encoding="utf-8"))
kos = R["a3_denetim"]["kosumlar"]
B = json.loads((WS / "kuyruk/e2_bcn_beklenen.json").read_text(encoding="utf-8"))
E1 = B["esleme"]["E1"]
ad = np.load(WS / "data/bcn20000/e2_girdi/bcn_e2_merkez_kare_224.npz")["bcn_filename"].tolist()
tr = {r["bcn_filename"]: r for r in csv.DictReader(open(WS / "data/bcn20000/meta/bcn_20k_train.csv", encoding="utf-8", newline=""))}
dis = set(json.loads((WS / "data/bcn20000/kopyasiz_liste.json").read_text(encoding="utf-8"))["dislanan"])
dx = np.array([tr[a]["diagnosis"] for a in ad])
sel = np.array([(d in E1) and (a not in dis) for a, d in zip(ad, dx)])
y = np.array([E1.get(d, -1) for d in dx])[sel]
cer = json.loads((WS / "data/bcn20000/e2_girdi/cerceve_kayit.json").read_text(encoding="utf-8"))
assert [c["ad"] for c in cer] == ad
kirp = np.array([c["kirp"] for c in cer])[sel]
# piksel istatistikleri (görüntü başına)
X = np.load(WS / "data/bcn20000/e2_girdi/bcn_e2_merkez_kare_224.npz")["images"]
ort = np.zeros((len(X), 3)); kose = np.zeros(len(X))
for i in range(0, len(X), 500):
    xb = X[i:i + 500].astype(np.float32)
    ort[i:i + 500] = xb.mean((1, 2))
    g = xb.mean(3)
    kose[i:i + 500] = np.stack([g[:, :16, :16].mean((1, 2)), g[:, :16, -16:].mean((1, 2)), g[:, -16:, :16].mean((1, 2)), g[:, -16:, -16:].mean((1, 2))], 1).mean(1)
ort, kose = ort[sel], kose[sel]
zd = np.load(WS / "data/medmnist/dermamnist_224.npz")
dm_ort = np.stack([zd["train_images"][i:i + 500].astype(np.float32).mean((1, 2)) for i in range(0, 7007, 500)]).reshape(-1, 3) if False else None
def dengeli(yy, t):
    return float(np.mean([(t[yy == c] == c).mean() for c in range(7) if (yy == c).any()]))
O = {"n": int(sel.sum())}
P = {}
for k in KL.SIRA:
    P[k] = np.stack([np.load(DR / kos[k][s] / "tahmin_bcn_merkez_kare.npy")[sel] for s in KL.SEEDLER])
    t = P[k].argmax(2)
    O.setdefault("dengeli", {})[k] = [round(dengeli(y, t[s]), 4) for s in range(5)] + [round(float(np.mean([dengeli(y, t[s]) for s in range(5)])), 4)]
    O.setdefault("nv_pay", {})[k] = round(float((t == 5).mean()), 4)
    cm = np.mean([np.array([[((y == a) & (t[s] == b)).sum() for b in range(7)] for a in range(7)]) for s in range(5)], 0)
    O.setdefault("cm", {})[k] = cm.round(1).tolist()
    # sınıf başına ortalama olasılık (önsel kayması mı, sıralama mı?)
    O.setdefault("ort_p", {})[k] = P[k].mean((0, 1)).round(3).tolist()
O["Delta_B1_bagimsiz"] = round(100 * (O["dengeli"]["h10"][-1] - O["dengeli"]["h02c"][-1]), 2)
# alt gruplar: çerçeve (yazar kırpıcısı tetiklendi) / köşe parlaklığı / R kanalı
def alt(mask, ad_):
    r = {"n": int(mask.sum())}
    for k in ("h10", "h02c", "h08c", "h11", "h01c"):
        t = P[k].argmax(2)[:, mask]
        r[k] = {"dengeli": round(float(np.mean([dengeli(y[mask], t[s]) for s in range(5)])), 4), "nv_pay": round(float((t == 5).mean()), 4)}
    r["Delta_h10_h02c"] = round(100 * (r["h10"]["dengeli"] - r["h02c"]["dengeli"]), 2)
    return r
O["alt"] = {"cerceve_var": alt(kirp, "c"), "cerceve_yok": alt(~kirp, "nc"),
            "kose<40": alt(kose < 40, "k"), "kose>=40": alt(kose >= 40, "k2")}
qs = np.quantile(ort[:, 0], [0.25, 0.5, 0.75])
for i, (lo, hi) in enumerate(zip([-1, *qs], [*qs, 999])):
    O["alt"][f"R_ceyrek{i+1}[{lo:.0f},{hi:.0f})"] = alt((ort[:, 0] >= lo) & (ort[:, 0] < hi), "q")
O["bcn_R_ort_ceyrek"] = [round(float(q), 1) for q in qs]
O["kose_oran<40"] = round(float((kose < 40).mean()), 4)
O["kirp_oran"] = round(float(kirp.mean()), 4)
json.dump(O, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: O[k] for k in ("dengeli", "nv_pay", "Delta_B1_bagimsiz", "alt", "bcn_R_ort_ceyrek", "kose_oran<40", "kirp_oran")}, ensure_ascii=False, indent=0))
