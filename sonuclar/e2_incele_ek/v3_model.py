# İNCELE v3 (model çalıştırır; BCN GÖRÜNTÜSÜ YOK): (a) 8 HAM kalibrasyon argmax uyumu, 55 kol-seed
# (b) keşifsel: resmî val üzerinde BCN-benzeri renk/çerçeve bozulmaları (yalnız resmî kollar; C1 kolları resmî val'e sokulmaz)
import json, os, sys, time
from pathlib import Path
WS = Path(r"<WS>")
os.environ.setdefault("HF_HOME", str(WS / "data" / "hf_cache")); os.environ.setdefault("TORCH_HOME", str(WS / "data" / "torch_cache"))
os.environ["HF_HUB_OFFLINE"] = "1"
DR = Path("<DRIVE_KOK>/3-1/runs")
sys.path.insert(0, str(WS / "varyantlar"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, timm
from timm.layers import set_fused_attn
from e2 import kollar as KL
from k5.k5_cikarim import IMAGENET_ORT, IMAGENET_STD
from dinov2_ft import ince_ayar_vit as IV
from dinov2_ozn import ozn_cikar as OZ
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
cih = torch.device("cuda")
ort = torch.tensor(IMAGENET_ORT, device=cih).view(1, 3, 1, 1); std = torch.tensor(IMAGENET_STD, device=cih).view(1, 3, 1, 1)
def hazirla(xb):
    x = torch.from_numpy(np.ascontiguousarray(xb)).to(cih).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    return (x - ort) / std
B = json.loads((WS / "kuyruk/e2_bcn_beklenen.json").read_text(encoding="utf-8"))
zk = np.load(WS / "data/bcn20000/e2_girdi/bcn_e2_kal_224.npz"); Kx, kidx = zk["images"], zk["idx"]
zd = np.load(WS / "data/medmnist/dermamnist_224.npz")
Rx = zd["train_images"][kidx]; yk = zd["train_labels"].reshape(-1)[kidx]
Xv, yv = zd["val_images"], zd["val_labels"].reshape(-1)
# BCN global kanal istatistikleri (v1'den: piksel, etiket yok) ve DermaMNIST train istatistikleri
mu_b, sd_b = np.array([148.03, 128.69, 123.79]), np.array([65.41, 61.84, 63.09])
mu_d, sd_d = np.array([195.48, 136.70, 142.80]), np.array([35.01, 40.04, 44.86])
yy, xx = np.mgrid[0:224, 0:224]; r = np.sqrt((yy - 111.5) ** 2 + (xx - 111.5) ** 2)
maske = (r <= 112 * 1.05).astype(np.float32)[..., None]     # daire dışı siyah (dermatoskop çerçevesi; köşeler)
def boz(X, tur):
    x = X.astype(np.float32)
    if tur == "orijinal": return X
    if tur == "renk_ort": y = x - mu_d + mu_b
    elif tur == "renk_ort_std": y = (x - mu_d) / sd_d * sd_b + mu_b
    elif tur == "cerceve": y = x * maske
    elif tur == "renk+cerceve": y = ((x - mu_d) / sd_d * sd_b + mu_b) * maske
    return np.clip(np.round(y), 0, 255).astype(np.uint8)
BOZ = ["orijinal", "renk_ort", "renk_ort_std", "cerceve", "renk+cerceve"]
def dengeli(y, t): return float(np.mean([(t[y == c] == c).mean() for c in range(7) if (y == c).any()]))
OZN = {}
import shutil, tempfile
TMP = Path(tempfile.mkdtemp())
def yerel(p):
    for i in range(5):
        try:
            q = TMP / p.name; shutil.copyfile(p, q); return q
        except OSError as e:
            print("yeniden dene", p, e, flush=True); time.sleep(10)
    raise
def kol_yukle(k, s):
    K = KL.KOLLAR[k]; kd = DR / KL.run_id(k, s); tur = K["tur"]
    km = json.loads((kd / "metrics.json").read_text(encoding="utf-8"))
    if tur == "r18":
        from torchvision.models import resnet18
        m = resnet18(weights=None); m.fc = nn.Linear(512, 7)
        m.load_state_dict(torch.load(yerel(kd / "best_model.pth"), map_location="cpu", weights_only=True), strict=True); m.to(cih).eval()
        def f(X):
            with torch.no_grad():
                return np.concatenate([torch.softmax(m(hazirla(X[i:i + 64])), 1).cpu().numpy() for i in range(0, len(X), 64)])
        return f
    if tur in ("vit_kismi", "vit_tam"):
        bm = json.loads((kd / "best_model_meta.json").read_text(encoding="utf-8"))
        set_fused_attn(bool(km.get("fused_attn", 1)))
        m = timm.create_model(bm["timm_ad"], pretrained=False, img_size=224, num_classes=0, drop_path_rate=0.0)
        m.head = nn.Linear(m.num_features, 7); m.num_classes = 7
        m.load_state_dict(torch.load(yerel(kd / "best_model.pth"), map_location="cpu", weights_only=True), strict=True); m.to(cih).eval()
        L0 = int(bm["donuk_blok"])
        def f(X):
            out = []
            with torch.no_grad():
                for i in range(0, len(X), 16):
                    if tur == "vit_kismi":
                        t = IV.donuk_ileri(m, hazirla(X[i:i + 16]), L0).half().float()
                        out.append(torch.softmax(IV.egitilen_ileri(m, t, L0, False), 1).cpu().numpy())
                    else:
                        out.append(torch.softmax(m(hazirla(X[i:i + 16])), 1).cpu().numpy())
            return np.concatenate(out)
        return f
    z = np.load(kd / "sonda_agirlik.npz", allow_pickle=False)
    kisa = next(kk for kk, v in OZ.MODELLER.items() if v[0] == str(z["model"]))
    W = torch.from_numpy(np.asarray(z["W"], np.float32)).to(cih); b = torch.from_numpy(np.asarray(z["b"], np.float32)).to(cih)
    so, ss = np.asarray(z["scaler_ort"], np.float64), np.asarray(z["scaler_std"], np.float64)
    if kisa not in OZN:
        OZN[kisa] = OZ.model_olustur(kisa, cih)
    mm, _, o2, s2 = OZN[kisa]
    def f(X):
        with torch.no_grad():
            Fz = np.concatenate([OZ._ileri(mm, np.ascontiguousarray(X[i:i + 16]), o2, s2, cih) for i in range(0, len(X), 16)])
            Xs = torch.from_numpy(((Fz - so) / ss).astype(np.float32)).to(cih)
            return torch.softmax(F.linear(Xs, W, b), 1).cpu().numpy()
    return f
yol0 = Path(sys.argv[1])
O = json.loads(yol0.read_text(encoding="utf-8")) if yol0.exists() else {"kal": {}, "val": {}}
cache = {t: boz(Xv, t) for t in BOZ}
yol = Path(sys.argv[1])
for k in KL.SIRA:
    for s in KL.SEEDLER:
        if f"{k}/s{s}" in O["kal"]: continue
        t0 = time.time()
        f = kol_yukle(k, s)
        pk, pr = f(Kx), f(Rx)
        O["kal"][f"{k}/s{s}"] = {"uyum": int((pk.argmax(1) == pr.argmax(1)).sum()), "maks_dp": float(np.abs(pk - pr).max()),
                                "dogru_bcn_yolu": int((pk.argmax(1) == yk).sum()), "dogru_npz": int((pr.argmax(1) == yk).sum())}
        if KL.KOLLAR[k]["veri"] == "resmi":
            arsiv = np.load(DR / KL.run_id(k, s) / "tahmin_val.npy")
            for t in BOZ:
                p = f(cache[t]); tt = p.argmax(1)
                d = {"dengeli": dengeli(yv, tt), "acc": float((tt == yv).mean()), "nv_pay": float((tt == 5).mean())}
                if t == "orijinal":
                    d["arsiv_argmax_uyum"] = float((tt == arsiv.argmax(1)).mean()); d["arsiv_maks_dp"] = float(np.abs(p - arsiv).max())
                O["val"].setdefault(k, {}).setdefault(t, []).append(d)
        print(k, s, O["kal"][f"{k}/s{s}"], f"{time.time() - t0:.0f}s", flush=True)
        del f; torch.cuda.empty_cache()
        yol.write_text(json.dumps(O, ensure_ascii=False, indent=1), encoding="utf-8")
print("BITTI", flush=True)
