import json, csv, sys
from pathlib import Path
import numpy as np
WS = Path(r"<WS>"); DR = Path("<DRIVE_KOK>/3-1/runs")
R = json.loads((WS / "sonuclar/e2_bcn_sonuc.json").read_text(encoding="utf-8")); kos = R["a3_denetim"]["kosumlar"]
B = json.loads((WS / "kuyruk/e2_bcn_beklenen.json").read_text(encoding="utf-8")); E1 = B["esleme"]["E1"]
ad = np.load(WS / "data/bcn20000/e2_girdi/bcn_e2_merkez_kare_224.npz")["bcn_filename"].tolist()
tr = {r["bcn_filename"]: r for r in csv.DictReader(open(WS / "data/bcn20000/meta/bcn_20k_train.csv", encoding="utf-8", newline=""))}
dis = set(json.loads((WS / "data/bcn20000/kopyasiz_liste.json").read_text(encoding="utf-8"))["dislanan"])
nn = json.loads((WS / "data/bcn20000/bcn20000_nn_tam.json").read_text(encoding="utf-8"))
e = nn["en_yakin_30"]; e = e if isinstance(e, list) else eval(e)
yakin = {x["bcn"] for x in e if x["rmse"] <= 10}
print("rmse<=10:", len(yakin), sorted({x['dx'] for x in e if x['rmse'] <= 10}))
sel = np.array([(tr[a]["diagnosis"] in E1) and (a not in dis) for a in ad]); sel2 = sel & np.array([a not in yakin for a in ad])
def deng(y, t): return np.mean([(t[y == c] == c).mean() for c in range(7) if (y == c).any()])
for nm, m in (("ana", sel), ("nn10_haric", sel2)):
    y = np.array([E1.get(tr[a]["diagnosis"], -1) for a in ad])[m]
    d = {k: np.mean([deng(y, np.load(DR / kos[k][s] / "tahmin_bcn_merkez_kare.npy")[m].argmax(1)) for s in range(5)]) for k in ("h10", "h02c")}
    print(nm, int(m.sum()), {k: round(v, 4) for k, v in d.items()}, "Delta", round(100 * (d["h10"] - d["h02c"]), 2))
