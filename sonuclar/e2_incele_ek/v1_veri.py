# İNCELE v1: eşleme, hiza, kopyasız/lezyon, nn_tam, kol<->run_id<->config (model çalıştırmaz)
import json, csv, hashlib, sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
WS = Path(r"<WS>")
DR = Path("<DRIVE_KOK>/3-1/runs")
sys.path.insert(0, str(WS / "varyantlar"))
from e2 import kollar as KL, bcn_veri as BV
from ortak import veri as V
O = {}
B = json.loads((WS / "kuyruk/e2_bcn_beklenen.json").read_text(encoding="utf-8"))
# 1) eşleme
import medmnist
O["medmnist_label"] = medmnist.INFO["dermamnist"]["label"]
O["SINIF_KISA"] = V.SINIF_KISA
z = np.load(WS / "data/medmnist/dermamnist_224.npz")
ytr = z["train_labels"].reshape(-1)
O["train_pay"] = (np.bincount(ytr, minlength=7) / len(ytr)).round(4).tolist()
kl = json.loads((WS / "data/bcn20000/kalibrasyon_ham/kalibrasyon_listesi.json").read_text(encoding="utf-8"))["goruntuler"]
kal = []
for g in kl:
    meta = json.loads((WS / f"data/bcn20000/kalibrasyon_ham/{g['image_id']}.json").read_text(encoding="utf-8"))["metadata"]["clinical"]
    kal.append({"id": g["image_id"], "idx": g["index"], "dx_liste": g["dx"], "isic_dx3": meta.get("diagnosis_3"),
                "npz_label": int(ytr[g["index"]]), "npz_ad": V.SINIF_KISA[int(ytr[g["index"]])]})
O["kal_esleme"] = kal
tr = list(csv.DictReader(open(WS / "data/bcn20000/meta/bcn_20k_train.csv", encoding="utf-8", newline="")))
O["bcn_dx_say"] = dict(Counter(r["diagnosis"] for r in tr))
O["esleme_E1"] = {k: (v, V.SINIF_KISA[v]) for k, v in B["esleme"]["E1"].items()}
# 2) hiza
ze = np.load(WS / "data/bcn20000/e2_girdi/bcn_e2_merkez_kare_224.npz")
zl = np.load(WS / "data/bcn20000/bcn20000_train_merkez_kare_224.npz")
ad_e, ad_l = ze["bcn_filename"].tolist(), zl["bcn_filename"].tolist()
O["hiza"] = {"e2_files": ze.files, "etiketli_files": zl.files, "n_e": len(ad_e), "n_l": len(ad_l), "sira_ayni": ad_e == ad_l,
             "sira_sha_e": BV.sira_sha256(ad_e) == B["sira_sha256"], "sirali": ad_e == sorted(ad_e), "tekil": len(set(ad_e)) == len(ad_e)}
Xe, Xl = ze["images"], zl["images"]
O["hiza"]["shape_dtype"] = [str(Xe.shape), str(Xe.dtype), str(Xl.shape), str(Xl.dtype)]
O["hiza"]["goruntu_bayt_ayni"] = bool(np.array_equal(Xe, Xl))
O["hiza"]["icerik_sha_eslesti"] = BV.icerik_sha256(Xe) == B["icerik"]["merkez_kare"]
csvmap = {r["bcn_filename"]: r["diagnosis"] for r in tr}
O["hiza"]["npz_diag_csv_ayni"] = all(csvmap[a] == d for a, d in zip(ad_l, zl["diagnosis"].tolist()))
# RGB: kanal ortalamaları
rng = np.random.default_rng(0)
ib = rng.choice(len(Xe), 1500, replace=False); idm = rng.choice(len(z["train_images"]), 1500, replace=False)
Xb = Xe[np.sort(ib)].astype(np.float64); Xd = z["train_images"][np.sort(idm)].astype(np.float64)
O["kanal"] = {"bcn_ort": Xb.mean((0, 1, 2)).round(2).tolist(), "bcn_std": Xb.std((0, 1, 2)).round(2).tolist(),
              "dm_ort": Xd.mean((0, 1, 2)).round(2).tolist(), "dm_std": Xd.std((0, 1, 2)).round(2).tolist(),
              "dm_val_shape": str(z["val_images"].shape), "dm_dtype": str(z["val_images"].dtype)}
# 3) kopyasız + lezyon
K = json.loads((WS / "data/bcn20000/kopyasiz_liste.json").read_text(encoding="utf-8"))
LK = json.loads((WS / "data/bcn20000/lezyon_kumeleri.json").read_text(encoding="utf-8"))
E1 = B["esleme"]["E1"]; dis = set(K["dislanan"])
sel = [r for r in tr if r["diagnosis"] in E1 and r["bcn_filename"] not in dis]
O["kopyasiz"] = {"n_goruntu": len(sel), "n_lezyon": len({r["lesion_id"] for r in sel}), "n_dislanan": len(dis),
                 "dislanan_hepsi_csvde": all(d in csvmap for d in dis),
                 "sinif_goruntu": [sum(1 for r in sel if E1[r["diagnosis"]] == c) for c in range(7)]}
lez_img = defaultdict(list)
for r in tr: lez_img[r["lesion_id"]].append(r["bcn_filename"])
selles = {r["lesion_id"] for r in sel}
O["birlesik"] = [{"cift": g, "kopyasizda": [u in selles for u in g], "goruntu_sayisi": [len(lez_img[u]) for u in g],
                  "dx": [next(r["diagnosis"] for r in tr if r["lesion_id"] == u) for u in g],
                  "dislanan_goruntu": [[f for f in lez_img[u] if f in dis] for u in g]} for g in LK["birlesik"]]
# çelişkili
O["celiskili_dislanan"] = sorted([(f, csvmap[f], next(r["lesion_id"] for r in tr if r["bcn_filename"] == f)) for f in dis
                                  if f in ("BCN_0003560", "BCN_0000237") or True])[:0]
# kopya grubu yeniden türetme (zip CRC)
zc = list(csv.DictReader(open(WS / "data/bcn20000/meta/zip_icerik_train.csv", encoding="utf-8", newline="")))
O["zip_cols"] = list(zc[0].keys())
grp = defaultdict(list)
for r in zc:
    fn = Path(r["ad"]).name
    if fn in csvmap: grp[(r["crc32"], r["boyut"])].append(fn)
gr = [sorted(v) for v in grp.values() if len(v) > 1]
fazla = {f for g in gr for f in g[1:]}
celis = [g for g in gr if len({csvmap[f] for f in g}) > 1]
yeniden = fazla | {f for g in celis for f in g}
O["kopya_turet"] = {"grup": len(gr), "fazla": len(fazla), "celiskili_grup": [(g, [csvmap[f] for f in g]) for g in celis],
                    "dislanan_ayni": yeniden == dis, "n_yeniden": len(yeniden)}
# nn_tam
nn = json.loads((WS / "data/bcn20000/bcn20000_nn_tam.json").read_text(encoding="utf-8"))
O["nn_tam"] = {k: nn[k] for k in ("n_bcn", "n_ref", "nn_rmse", "esik5_alti", "esik10_alti", "esik15_alti")}
e30 = nn["en_yakin_30"] if isinstance(nn["en_yakin_30"], list) else []
O["nn_tam"]["esik10_ref_tekrar"] = Counter(x["ref"] for x in e30[:20]).most_common(3)
O["nn_tam"]["esik10_dx"] = Counter(x["dx"] for x in e30[:20])
# 4) kol <-> run_id <-> config
Kj = json.loads((WS / "sonuclar/e2_bcn.KILIT").read_text(encoding="utf-8"))
R = json.loads((WS / "sonuclar/e2_bcn_sonuc.json").read_text(encoding="utf-8"))
kos = R["a3_denetim"]["kosumlar"]
tab = []; sorun = []
for k in KL.SIRA:
    shas = []
    for s in KL.SEEDLER:
        d = DR / kos[k][s]
        c = json.loads((d / "config.json").read_text(encoding="utf-8")); m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        e2 = c["e2"]; rid = KL.run_id(k, s)
        src = DR / rid
        sc = json.loads((src / "config.json").read_text(encoding="utf-8")); sm = json.loads((src / "metrics.json").read_text(encoding="utf-8"))
        cs_src = hashlib.sha256((src / "config.json").read_bytes()).hexdigest()
        ok = {"kaynak=kollar": e2["kaynak_run"] == rid, "ebeveyn": c.get("ebeveyn") == rid, "kilit": Kj["kollar"][k][s] == rid,
              "komut_kol": c["komut"][c["komut"].index("--kol") + 1] == k, "seed": str(c["seed"]) == str(s) == str(sc["seed"]),
              "src_varyant": sc["varyant"] == KL.KOLLAR[k]["varyant"] == e2["kaynak_varyant"],
              "src_config_sha": cs_src == e2["kaynak_sha256"]["config.json"],
              "durum": e2["ozdeslik"]["durum"], "val_acc=kaynak": m["val_acc"] == m["kaynak_val_acc"] == sm["val_acc"],
              "maxdp": m["ozdeslik_p_maks_fark"],
              "size": (sc.get("protokol") or {}).get("size", (sc.get("protokol") or {}).get("r", None)),
              "src_npz": (sc.get("veri") or {}).get("npz"), "giris_cozunurluk": sm.get("giris_cozunurluk"),
              "normalizasyon": str((sc.get("protokol") or {}).get("normalizasyon")), "augment": str((sc.get("protokol") or {}).get("augment")),
              "c_bolme": (src / "c_bolme.json").exists(), "bcn_dosya": kos[k][s]}
        shas.append(e2["kaynak_sha256"]["best_model.pth"] or e2["kaynak_sha256"]["sonda_agirlik.npz"])
        for q in ("kaynak=kollar", "ebeveyn", "kilit", "komut_kol", "seed", "src_varyant", "src_config_sha", "val_acc=kaynak"):
            if not ok[q]: sorun.append((k, s, q))
        tab.append((k, s, ok))
    if len(set(shas)) != 5: sorun.append((k, "agirlik_sha_tekil_degil"))
O["kol_tab"] = tab; O["kol_sorun"] = sorun
json.dump(O, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print("bitti")
