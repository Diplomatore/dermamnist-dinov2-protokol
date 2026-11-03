#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
topluluk.py — H13 (Tur-5, A9): üye koşumların KAYITLI val olasılıklarından topluluk. Eğitim / çıkarım YOK; GPU YOK
(torch import edilmez). Kart: hipotezler/H13.md (ön-kayıtlı; kart üstü değişmez).

Birleşim (kart "Değişen tek şey"):
- `--birlesim logolasilik` (birincil): L = Σ_i w_i · log(clip(p_i, kirp, 1)), p = softmax(L), argmax.
  Kırpma `--kirp 1e-12` ön-kayıtlıdır (öldürme (d)); NaN/inf → rc=3 (DUR).
- `--birlesim olasilik` (ikincil satır (7)): p = Σ_i w_i p_i, satır toplamıyla normalize.
- `--agirlik` virgüllü (varsayılan eşit; toplam 1'e normalize edilir). Tek üye modu `--agirlik 1,0` (özdeşlik (c1)).
Denetimler (öldürme (c); `--ozdeslik` ile tek üye modu da ayrıca hesaplanır):
 (c1) tek üye modu (ilk üye, ağırlık 1) argmax'ı ilk üyenin kayıtlı argmax'ıyla 1.003/1.003 aynı ve max|Δp| ≤ 1e-6;
 (c2) her üyenin etiket_val.npy = npz val etiketi (dermamnist.npz val_labels);
 (c3) H10 üyesinin config.ebeveyn = "h02_r18in_ft224_colab:s<seed>";
 (c4) topluluk satır toplamları 1 ± 1e-6. Herhangi biri tutmazsa metrics yazılır ve rc=3.
Seed denetimi `--seed-denet hepsi|ilk|yok`: hepsi = bütün üyelerin seed'i --seed (seed-eşleşik topluluk);
 ilk = yalnız ilk üye (homojen 2'li ablasyon H10_s ⊕ H10_{s+1}); yok = homojen 5-seed (n = 1).
TEST OKUNMAZ (M4): yalnız tahmin_val.npy / etiket_val.npy / metrics.json (yalnız val_acc, val_dengeli_acc, param_sayisi)
/ config.json / .done okunur; yolunda "test" geçen dosya reddedilir; kayit.metrics_yaz test_ anahtarı yazamaz.
Çıktılar ({RUN_DIR}): metrics.json (ortak/degerlendir.py anahtarları + S/N + C-val + çeşitlilik + özdeşlik) ·
 tahmin_val.npy (normalize olasılık, float32) · etiket_val.npy · config.json/varyant_bilgi.json eki (üyeler, SHA256).

Kullanım (WS kökü, kosum.py altında; komutlar varyantlar/KOMUTLAR.md §T5):
  env/.venv/Scripts/python.exe varyantlar/topluluk/topluluk.py --uyeler <H10_s dizini> <H02c_s dizini> \
     --agirlik 0.5,0.5 --birlesim logolasilik --kirp 1e-12 --ozdeslik --seed S --out {RUN_DIR}/metrics.json
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ortak.degerlendir import degerlendir, dengeli_acc  # noqa: E402
from ortak.kayit import config_ekle, metrics_yaz  # noqa: E402

NPZ28 = WS / "data" / "medmnist" / "dermamnist.npz"
C_ESLEME = WS / "data" / "splits" / "dermamnist_c_esleme.csv"
H10_VARYANT = "h10_dinov2b_k4_ft224"
SINIF_KISA = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
BES = [0, 1, 2, 4, 5]  # df (3), vasc (6) hariç (tahlil_tur4 ile aynı)
MEL, NV, BKL = 4, 5, 2
OZDESLIK_TOL = 1e-6
TOPLAM_TOL = 1e-6


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def val_etiket() -> np.ndarray:
    with np.load(NPZ28) as z:
        return z["val_labels"].reshape(-1).astype(np.int64)


def cval_maskesi(n: int) -> np.ndarray:
    """Resmî val ∩ C-val (tahlil_tur4_cval.py ile aynı tanım: split_C == val ve split_resmi == val)."""
    with open(C_ESLEME, encoding="utf-8") as f:
        cv = {int(r["index_resmi"]) for r in csv.DictReader(f) if r["split_C"] == "val" and r["split_resmi"] == "val"}
    return np.array([i in cv for i in range(n)])


def uye_oku(d: Path):
    d = Path(d).resolve()
    for ad in ("tahmin_val.npy", "etiket_val.npy", "metrics.json", "config.json", ".done"):
        if not (d / ad).exists():
            raise SystemExit(f"HATA: üye {d.name}: {ad} yok")
    yol = d / "tahmin_val.npy"
    if "test" in yol.name.lower():
        raise SystemExit("HATA: test dosyası okunamaz (M4)")
    c = json.loads((d / "config.json").read_text(encoding="utf-8"))
    j = json.loads((d / ".done").read_text(encoding="utf-8"))
    if j.get("rc") != 0:
        raise SystemExit(f"HATA: üye {d.name} damgası rc={j.get('rc')}")
    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    return {"dizin": d, "run_id": c.get("run_id", d.name), "varyant": c.get("varyant"), "seed": str(c.get("seed")),
            "asama": int(c.get("asama", 3)), "ebeveyn": c.get("ebeveyn"),
            "donanim": (c.get("donanim") or {}).get("nvidia_smi") if isinstance(c.get("donanim"), dict) else None,
            "P": np.load(yol).astype(np.float64), "y": np.load(d / "etiket_val.npy").reshape(-1).astype(np.int64),
            "val_acc": float(m["val_acc"]), "val_dengeli_acc": float(m.get("val_dengeli_acc", float("nan"))),
            "param_sayisi": m.get("param_sayisi"), "sha256": sha256(yol)}


def birlestir(Ps, w, birlesim: str, kirp: float) -> np.ndarray:
    if birlesim == "logolasilik":
        L = sum(wi * np.log(np.clip(P, kirp, 1.0)) for wi, P in zip(w, Ps))
        z = L - L.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)
    if birlesim == "olasilik":
        p = sum(wi * P for wi, P in zip(w, Ps))
        return p / p.sum(axis=1, keepdims=True)
    raise SystemExit(f"HATA: bilinmeyen birleşim {birlesim}")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uyeler", nargs="+", required=True, help="üye koşum dizinleri (ilk üye = ebeveyn/ref; H13'te H10_s)")
    ap.add_argument("--agirlik", default=None, help="virgüllü ağırlıklar (varsayılan eşit)")
    ap.add_argument("--birlesim", choices=["logolasilik", "olasilik"], default="logolasilik")
    ap.add_argument("--kirp", type=float, default=1e-12)
    ap.add_argument("--ozdeslik", action="store_true", help="öldürme (c1): tek üye modunu da hesapla")
    ap.add_argument("--seed-denet", choices=["hepsi", "ilk", "yok"], default="hepsi")
    ap.add_argument("--seed", required=True, help="seed (int) ya da homojen topluluk için etiket (ör. 0-4)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out).resolve()
    run_dir = out.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    if a.kirp != 1e-12:
        print(f"UYARI: kırpma {a.kirp} ≠ ön-kayıtlı 1e-12")

    U = [uye_oku(Path(d)) for d in a.uyeler]
    k = len(U)
    w = np.ones(k) / k if a.agirlik is None else np.array([float(x) for x in a.agirlik.split(",")])
    if len(w) != k or (w < 0).any() or w.sum() <= 0:
        raise SystemExit(f"HATA: ağırlık sayısı/değeri üye sayısıyla uyuşmuyor ({a.agirlik}, {k} üye)")
    w = w / w.sum()
    y = val_etiket()
    N = len(y)
    denetim_hatasi = []

    # (c2) etiket hizası + şekil
    hiza = all(np.array_equal(u["y"], y) and u["P"].shape == (N, 7) for u in U)
    if not hiza:
        denetim_hatasi.append("c2 etiket/şekil hizası")
    # seed denetimi
    if a.seed_denet == "hepsi":
        seed_ok = all(u["seed"] == str(a.seed) for u in U)
    elif a.seed_denet == "ilk":
        seed_ok = U[0]["seed"] == str(a.seed)
    else:
        seed_ok = True
    if not seed_ok:
        denetim_hatasi.append(f"seed ({[u['seed'] for u in U]} ↔ {a.seed}, kip {a.seed_denet})")
    if any(u["asama"] != 3 for u in U):
        denetim_hatasi.append("üye aşama ≠ 3")
    # (c3) H10 üyelerinin ebeveyni
    h10 = [u for u in U if u["varyant"] == H10_VARYANT]
    eb_ok = [u["ebeveyn"] == f"h02_r18in_ft224_colab:s{u['seed']}" for u in h10]
    if not all(eb_ok):
        denetim_hatasi.append(f"c3 H10 ebeveyni {[u['ebeveyn'] for u in h10]}")

    # birleşim
    Ps = [u["P"] for u in U]
    p = birlestir(Ps, w, a.birlesim, a.kirp)
    if not np.isfinite(p).all():
        print("DUR: topluluk olasılıklarında NaN/inf (öldürme (d))")
        return 3
    p = p.astype(np.float32).astype(np.float64)  # kaydedilen dizinin aynısı üzerinden ölç (H04 τ=0 özdeşliği için)
    toplam_sapma = float(np.max(np.abs(p.sum(1) - 1.0)))
    if toplam_sapma > TOPLAM_TOL:
        denetim_hatasi.append(f"c4 satır toplamı {toplam_sapma}")

    # (c1) tek üye modu
    ozd = {}
    if a.ozdeslik:
        p1 = birlestir([Ps[0]], [1.0], a.birlesim, a.kirp)
        ayni = int((p1.argmax(1) == Ps[0].argmax(1)).sum())
        mf = float(np.max(np.abs(p1 - Ps[0])))
        ozd = {"ozdeslik_tek_uye_argmax_ayni": ayni, "ozdeslik_tek_uye_argmax_ayni_oran": ayni / N,
               "ozdeslik_tek_uye_maks_fark": mf,
               "ozdeslik_tek_uye_acc_fark": float(abs((p1.argmax(1) == y).mean() - U[0]["val_acc"]))}
        if ayni != N or mf > OZDESLIK_TOL or ozd["ozdeslik_tek_uye_acc_fark"] > 1e-12:
            denetim_hatasi.append(f"c1 tek üye ({ayni}/{N}, max|Δp| {mf:.3g})")

    # metrikler (ortak/degerlendir.py; val_idx → S/N ve molemax alanları)
    vidx = np.arange(N)
    m = degerlendir(y, p, on="val", val_idx=vidx)
    alt_grup = m.pop("alt_grup")
    tah = p.argmax(1)
    Tu = np.stack([P.argmax(1) for P in Ps])  # (k, N)
    du = (Tu == y[None])
    # üye ve kazanç
    ek = {f"val_acc_uye{i + 1}": float(du[i].mean()) for i in range(k)}
    ek["val_acc_uye_ort"] = float(du.mean())
    ek["val_topluluk_kazanci_G"] = m["val_acc"] - ek["val_acc_uye_ort"]
    ek["val_acc_delta_vs_uye1"] = m["val_acc"] - float(du[0].mean())
    ek["val_dengeli_acc_delta_vs_uye1"] = m["val_dengeli_acc"] - dengeli_acc(y, Tu[0])
    ek["val_karar_degisen_vs_uye1"] = int((tah != Tu[0]).sum())
    # çeşitlilik (ikincil 6): üye çiftleri argmax anlaşmazlığı ve "ikisi de yanlış"
    if k >= 2:
        an, iy = [], []
        for i in range(k):
            for j in range(i + 1, k):
                an.append(float((Tu[i] != Tu[j]).mean()))
                iy.append(float((~du[i] & ~du[j]).mean()))
        ek["val_uye_anlasmazlik"] = float(np.mean(an))
        ek["val_uye_ikisi_de_yanlis"] = float(np.mean(iy))
        ek["val_uye_hepsi_yanlis"] = float((~du).all(0).mean())
        ek["val_uye_en_az_biri_dogru"] = float(du.any(0).mean())
    # ikincil 2: 5 sınıflı dengeli, mel→nv, nv payı, mel/bkl AUC (degerlendir'de), sınıf duyarlılığı
    rec = [float((tah[y == c] == c).mean()) for c in range(7)]
    ek["val_dengeli_acc_5sinif"] = float(np.mean([rec[c] for c in BES]))
    ek.update({f"val_duyarlilik_{SINIF_KISA[c]}": rec[c] for c in range(7)})
    ek["val_karisiklik_mel_nv"] = int(((y == MEL) & (tah == NV)).sum())
    ek["val_karisiklik_bkl_nv"] = int(((y == BKL) & (tah == NV)).sum())
    ek["val_toplam_hata"] = int((tah != y).sum())
    # ikincil 5: C-val alt kümesi (resmî val ∩ C-val; lezyonu resmî train'de yok)
    cm = cval_maskesi(N)
    ek["val_n_cval"] = int(cm.sum())
    ek["val_acc_cval"] = float((tah[cm] == y[cm]).mean())
    ek["val_dengeli_acc_cval"] = dengeli_acc(y[cm], tah[cm])
    ek["val_acc_cval_uye1"] = float(du[0][cm].mean())
    # özdeşlik/denetim alanları (kapı)
    ek.update(ozd)
    ek["etiket_hizasi"] = int(hiza)
    ek["seed_denet_ok"] = int(seed_ok)
    ek["uye_ebeveyn_ok"] = int(all(eb_ok)) if h10 else -1
    ek["uye_ebeveyn_h10_n"] = len(h10)
    ek["satir_toplam_maks_sapma"] = toplam_sapma
    ek["kirp"] = float(a.kirp)
    ek["uye_sayisi"] = k
    ek["birlesim_logolasilik"] = int(a.birlesim == "logolasilik")
    ek.update({f"agirlik_uye{i + 1}": float(w[i]) for i in range(k)})
    sure = round(time.time() - t0, 3)
    params = [u["param_sayisi"] for u in U]
    metrics = {**m, **ek,
               "egitim_sure_sn": sure, "sure_sn": sure,
               "param_sayisi": int(sum(params)) if all(x is not None for x in params) else -1,
               "param_ek_skaler": 0, "egitilen_param": 0, "cikarim_omurga_sayisi": k,
               "seed": int(a.seed) if str(a.seed).lstrip("-").isdigit() else str(a.seed),
               "cihaz": "cpu-birlesim (numpy; GPU kullanılmadı; üyeler L4)", "alt_grup": alt_grup}
    np.save(run_dir / "tahmin_val.npy", p.astype(np.float32))
    np.save(run_dir / "etiket_val.npy", y)
    config_ekle(run_dir, {
        "donanim": "cpu-birlesim",
        "topluluk": {"kural": "softmax(Σ w_i log clip(p_i, kirp, 1))" if a.birlesim == "logolasilik" else "Σ w_i p_i / satır toplamı",
                     "birlesim": a.birlesim, "agirlik": w.tolist(), "kirp": a.kirp, "seed_denet": a.seed_denet,
                     "kart": "hipotezler/H13.md", "kaynak": "Gontijo-Lopes vd. 2022 (ICLR); Lakshminarayanan vd. 2017"},
        "uyeler": [{"run_id": u["run_id"], "varyant": u["varyant"], "seed": u["seed"], "ebeveyn": u["ebeveyn"],
                    "donanim": u["donanim"], "tahmin_val_sha256": u["sha256"], "dizin": str(u["dizin"]),
                    "kaynak": "<DRIVE_KOK>/3-1/runs (yerel arşiv kopyası runs_colab/, SHA256 ARSIV_MANIFEST.json)"}
                   for u in U],
        "denetim_hatasi": denetim_hatasi})
    sd = metrics_yaz(out, metrics)
    if sd:
        print(f"DUR: sonlu olmayan anahtarlar {sd}")
        return 3
    if denetim_hatasi:
        print(f"DUR: özdeşlik/denetim ihlali {denetim_hatasi} — metrics yazıldı, rc=3")
        return 3
    print(f"[topluluk] BİTTİ {sure}s · {k} üye {[u['run_id'] for u in U]} · {a.birlesim} w={np.round(w, 4).tolist()} · "
          f"val_acc {m['val_acc']:.4f} (üye1 {du[0].mean():.4f}, üye ort {du.mean():.4f}) · AUC {m['val_auc']:.4f} · "
          f"ECE {m['val_ece']:.4f} · özdeşlik {ozd or '—'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
