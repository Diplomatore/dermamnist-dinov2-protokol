import json, sys, hashlib
from pathlib import Path
import numpy as np, torch
K = Path(sys.argv[1]); ciftler = [("eski", "yeni"), ("eski_onb", "yeni_onb"), ("eski", "eski_onb")]
ZAMAN = ("sn", "img_s", "tepe_", "tarih", "sure", "baslangic", "bitis", "zaman")
rapor = {}
for a, b in ciftler:
    A, B = K / a, K / b
    r = {}
    ma, mb = json.loads((A / "metrics.json").read_text("utf-8")), json.loads((B / "metrics.json").read_text("utf-8"))
    r["anahtar_kumesi_ayni"] = sorted(ma) == sorted(mb)
    sayisal_fark = {}
    for k in sorted(set(ma) & set(mb)):
        if any(z in k for z in ZAMAN): continue
        if ma[k] != mb[k]: sayisal_fark[k] = (ma[k], mb[k])
    r["metrics_farkli_alanlar(zaman/bellek hariç)"] = sayisal_fark
    r["ilk_adim_loss"] = (ma["ilk_adim_loss"], mb["ilk_adim_loss"], abs(ma["ilk_adim_loss"] - mb["ilk_adim_loss"]))
    ga, gb = json.loads((A / "gecmis.json").read_text("utf-8")), json.loads((B / "gecmis.json").read_text("utf-8"))
    ga = ga.get("gecmis", ga) if isinstance(ga, dict) else ga; gb = gb.get("gecmis", gb) if isinstance(gb, dict) else gb
    r["epoch_train_loss"] = [(x["train_loss"], y["train_loss"]) for x, y in zip(ga, gb)]
    r["epoch_val"] = all(x[k] == y[k] for x, y in zip(ga, gb) for k in ("train_loss", "val_acc", "val_auc", "val_dengeli_acc"))
    for f in ("tahmin_val.npy", "tahmin_val_son.npy", "tahmin_val_aucsecim.npy", "etiket_val.npy"):
        xa, xb = np.load(A / f), np.load(B / f)
        r[f] = {"bit_esit": bool(xa.shape == xb.shape and xa.dtype == xb.dtype and xa.tobytes() == xb.tobytes()),
                "maks_fark": float(np.max(np.abs(xa.astype(np.float64) - xb.astype(np.float64))))}
    sa, sb = torch.load(A / "best_model.pth", map_location="cpu", weights_only=False), torch.load(B / "best_model.pth", map_location="cpu", weights_only=False)
    sa = sa.get("state_dict", sa) if isinstance(sa, dict) and "state_dict" in sa else sa
    sb = sb.get("state_dict", sb) if isinstance(sb, dict) and "state_dict" in sb else sb
    tens = [k for k, v in sa.items() if torch.is_tensor(v)]
    esit = sum(int(torch.equal(sa[k], sb[k])) for k in tens)
    r["best_model_tensor_bit_esit"] = f"{esit}/{len(tens)}"
    r["best_model_maks_fark"] = max(float((sa[k].double() - sb[k].double()).abs().max()) if sa[k].numel() else 0.0 for k in tens)
    rapor[f"{a}↔{b}"] = r
print(json.dumps(rapor, ensure_ascii=False, indent=1))
