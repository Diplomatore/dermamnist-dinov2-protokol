#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sicaklik.py — R-T: sıcaklık ölçekleme RAPORLAMA protokolü (SIRALAMA.md "R-T", ön-kayıt; KART DEĞİL).
GPU YOK, eğitim YOK; kaynağın kayıtlı val olasılıklarına uygulanır.

Yöntem (ön-kayıt):
- Seed başına tek T ∈ [0,5; 3,0] (adım 0,01); seçim kümesinde NLL'yi en küçükleyen T (eşitlikte küçük T).
  p_T = softmax(log p / T)  (log p, logitlerin sabit kaymasıdır → z/T ile aynı).
- H04'ün aynı 5 katıyla (varyantlar/logit_ayari/katlar.json; SHA256 metrics'e) çapraz uydurma: k katı için T_k kalan
  4 kattan seçilir, yalnız k katına uygulanır → OOF olasılık; **OOF ECE-15 raporlanır** (birincil R-T sayısı).
- Tam-val T (tüm val'de seçilen) iyimser ikincildir (val_*_tamval); K5'te test'e bir kez uygulanacak T budur.
- argmax korunur (Guo vd. 2017): acc_degisim ve dengeli_degisim = 0 olmalı (değilse rc=3).
Kaynaklar: H02, H01, baseline, H04 OOF çıktısı ve (TUT alırsa) H05.

Kullanım (WS kökü, kosum.py altında):
  env/.venv/Scripts/python.exe varyantlar/logit_ayari/sicaklik.py --kaynak-varyant h02_r18in_ft224 \
     --kaynak-dosya tahmin_val.npy --seed 0 --out {RUN_DIR}/metrics.json
Çıktılar: metrics.json (düz; val_*; TEST YOK) · tahmin_val.npy (OOF kalibre) · tahmin_val_tamval.npy · etiket_val.npy ·
  sicaklik_secim.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from logit_ayari import KATLAR, ebeveyn_kosumlari, izgara, sha256, softmax, val_etiket  # noqa: E402
from ortak.degerlendir import degerlendir  # noqa: E402
from ortak.kayit import atomik_json, config_ekle, metrics_yaz  # noqa: E402


def nll(L, y, T):
    z = L / T
    z = z - z.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=1))
    return float(np.mean(lse - z[np.arange(len(y)), y]))


def T_sec(L, y, idx, Ts):
    v = np.array([nll(L[idx], y[idx], T) for T in Ts])
    i = int(np.argmin(v))  # np.argmin ilk (en küçük T) en küçüğü döndürür
    return float(Ts[i]), v


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kaynak-varyant", required=True)
    p.add_argument("--kaynak-dosya", default="tahmin_val.npy")
    p.add_argument("--T-izgara", default="0.5:3.0:0.01")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    t0 = time.time()
    out = Path(a.out).resolve()
    run_dir = out.parent
    if not KATLAR.exists():
        print(f"HATA: {KATLAR} yok")
        return 3
    KJ = json.loads(KATLAR.read_text(encoding="utf-8"))
    kat = np.array(KJ["kat"], dtype=np.int64)
    y = val_etiket()
    kos = ebeveyn_kosumlari(a.kaynak_varyant, a.kaynak_dosya)
    if a.seed not in kos:
        print(f"HATA: {a.kaynak_varyant} seed {a.seed} yok (bulunan {list(kos)})")
        return 3
    d = kos[a.seed]
    if not np.array_equal(np.load(d / "etiket_val.npy").reshape(-1).astype(np.int64), y):
        print("HATA: kaynak etiketleri ≠ npz val")
        return 3
    P = np.load(d / a.kaynak_dosya).astype(np.float64)
    L = np.log(np.maximum(P, 1e-300))
    Ts = izgara(a.T_izgara)
    n_kat = int(kat.max()) + 1
    T_k = []
    for k in range(n_kat):
        T_k.append(T_sec(L, y, np.where(kat != k)[0], Ts)[0])
    T_k = np.array(T_k)
    p_oof = softmax(L / T_k[kat][:, None])
    T_t, egri = T_sec(L, y, np.arange(len(y)), Ts)
    p_tam = softmax(L / T_t)
    p_ham = softmax(L)
    vidx = np.arange(len(y))
    m = degerlendir(y, p_oof, on="val", val_idx=vidx)
    alt_grup = m.pop("alt_grup")
    mh = degerlendir(y, p_ham, on="val", bootstrap=False)
    mt = degerlendir(y, p_tam, on="val", bootstrap=False)
    acc_deg = abs(m["val_acc"] - mh["val_acc"])
    den_deg = abs(m["val_dengeli_acc"] - mh["val_dengeli_acc"])
    if not np.array_equal(p_oof.argmax(1), P.argmax(1)):
        acc_deg = max(acc_deg, 1.0)

    def asiri_guven(pp):
        return float(pp.max(1).mean() - (pp.argmax(1) == y).mean())

    sure = round(time.time() - t0, 2)
    metrics = {
        **m,
        "val_ece_ham": mh["val_ece"], "val_nll_ham": mh["val_nll"], "val_ece_tamval": mt["val_ece"], "val_nll_tamval": mt["val_nll"],
        "val_auc_ham": mh["val_auc"],
        "val_asiri_guven": asiri_guven(p_oof), "val_asiri_guven_ham": asiri_guven(p_ham), "val_asiri_guven_tamval": asiri_guven(p_tam),
        "val_ece_dusus": mh["val_ece"] - m["val_ece"],
        **{f"T_kat_{k + 1}": float(t) for k, t in enumerate(T_k)},
        "T_kat_medyan": float(np.median(T_k)), "T_tamval": T_t,
        "acc_degisim": float(acc_deg), "dengeli_degisim": float(den_deg),
        "katlar_sha256": sha256(KATLAR), "kat_yontemi": KJ["yontem"],
        "kaynak_varyant": a.kaynak_varyant, "kaynak_dosya": a.kaynak_dosya, "kaynak_kosum": d.name,
        "egitim_sure_sn": sure, "sure_sn": sure, "param_ek_skaler": 1, "seed": a.seed,
        "cihaz": "cpu (numpy; GPU kullanılmadı)", "alt_grup": alt_grup,
    }
    np.save(run_dir / "tahmin_val.npy", p_oof.astype(np.float32))
    np.save(run_dir / "tahmin_val_tamval.npy", p_tam.astype(np.float32))
    np.save(run_dir / "etiket_val.npy", y)
    atomik_json(run_dir / "sicaklik_secim.json", {"T_kat": T_k.tolist(), "T_tamval": T_t,
                                                  "tamval_nll_egrisi": [[float(t), float(v)] for t, v in zip(Ts, egri)]})
    config_ekle(run_dir, {"sicaklik": {"protokol": "SIRALAMA R-T", "kaynak_varyant": a.kaynak_varyant,
                                       "kaynak_dosya": a.kaynak_dosya, "kaynak_kosum": d.name, "T_izgara": a.T_izgara,
                                       "secim": "NLL min (eşitlikte küçük T)", "katlar_sha256": sha256(KATLAR)}})
    metrics_yaz(out, metrics)
    if acc_deg != 0.0 or den_deg != 0.0:
        print(f"DUR: argmax değişti (acc_degisim {acc_deg}, dengeli_degisim {den_deg}) — rc=3")
        return 3
    print(f"[sicaklik] BİTTİ {sure}s · {a.kaynak_varyant} s{a.seed} · T_k {T_k.tolist()} T* {T_t} · "
          f"ECE ham {mh['val_ece']:.4f} → OOF {m['val_ece']:.4f} (tam-val {mt['val_ece']:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
