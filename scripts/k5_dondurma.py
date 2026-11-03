# -*- coding: utf-8 -*-
"""
k5_dondurma.py — K5-ONKAYIT §0 P4: donmuş parametre dosyası sonuclar/k5_dondurma.json (+ .json.sha256) — VAL'DEN, TEST AÇILMADAN.

İçerik:
  - kollar + run_id'ler (§1), seed {0..4}, rc=0, aşama 3; K4 val metrics.json'larında test_* anahtarı sayısı (baseline hariç 0 olmalı).
  - katman: τ* (tam-val; H04 protokolü = logit_ayari.tau_sec, τ ızgarası 0:1:0,1, kısıt 0,010; H01c/H12 için H06 protokolü = aynı
    kural) ve R-T: seed başına tam-val T (sicaklik.T_sec, T ızgarası 0,5:3,0:0,01, NLL). Ön-kayıttaki değerlerle karşılaştırılır
    (H02c 0,7 · H08c 0,7 · H10 0,7 · H11 1,0 · H01c 0,2; H01c T 1,19) — fark → rc 2.
  - sonda C* (H01c, H12): metrics secilen_C (5 seed aynı olmalı). Epoch seçimi: en_iyi_epoch (bilgi; best_model'e gömülü).
  - baseline_c dondurulmuş test sayıları (K3/kc1 tekrar kapısında görüldü, izinli): test_acc, test_auc, test_acc_son (P6 eşi).
  - katlar.json SHA256 (0db03edb…e0193), RNG 0, B 2.000, Holm aileleri, W = H10 (H13 ÖLDÜR).
  - SHA256: K5-ONKAYIT.md (+ mtime), scripts/k5_test.py, varyantlar/k5/*.py, scripts/c_bolme_npz.py, scripts/k5_c_denetim.py,
    sonuclar/k5_c_denetim.json, kuyruk/k5-*.jsonl|beklenti.json, kuyruk/gruplar.json, kaynak tahmin_val.npy'ler (yerel arşiv),
    Drive best_model.pth / sonda_agirlik.npz boyut + mtime (SHA256'ları Colab a1'de hesaplanır; a1 ↔ a3 aynı olmalı).
Kullanım (WS kökü): PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I scripts/k5_dondurma.py [--drive "G:/Drive'ım/yz/3-1"]
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "varyantlar" / "logit_ayari"))
sys.path.insert(0, str(WS / "scripts"))
import logit_ayari as LA  # noqa: E402
import sicaklik as SC  # noqa: E402
import k5_test as KT  # noqa: E402

ONKAYIT_TAU = {"h02": 0.7, "h08": 0.7, "h10": 0.7, "h11": 1.0, "h01": 0.2}
ONKAYIT_T_H01 = 1.19


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(WS / "runs_colab"))
    ap.add_argument("--drive", default="G:/Drive'ım/yz/3-1")
    a = ap.parse_args()
    runs, drive = Path(a.runs), Path(a.drive)
    logpi = KT.train_logpi()
    taus, Ts = LA.izgara("0:1:0.1"), LA.izgara("0.5:3.0:0.01")
    hatalar = []
    kollar, katman, cyildiz, epoch, baseline_test, kaynak_sha, drive_agirlik = {}, {}, {}, {}, {}, {}, {}
    for k, v in KT.KOL.items():
        P, y, adlar = KT.yukle_kol(runs, v, "tahmin_val.npy", "etiket_val.npy")
        kollar[k] = {"varyant": v, "run_id": adlar}
        M = [json.loads((runs / d / "metrics.json").read_text(encoding="utf-8")) for d in adlar]
        ntest = sum(1 for m in M for q in m if q.startswith("test"))
        kollar[k]["metrics_test_anahtari"] = ntest
        if k != "baseline" and ntest:
            hatalar.append(f"{k}: K4 metrics'te test anahtarı {ntest}")
        L = np.log(np.maximum(P, 1e-300))
        tau, tablo = LA.tau_sec(L, logpi, y, np.arange(len(y)), taus, 0.010)
        T = [SC.T_sec(L[s], y, np.arange(len(y)), Ts)[0] for s in range(5)]
        katman[k] = {"tau_tamval": tau, "T_tamval_seed": T, "protokol": "H06" if k in ("h01", "h12") else "H04",
                     "tau_tablo": tablo}
        if k in ONKAYIT_TAU and abs(tau - ONKAYIT_TAU[k]) > 1e-9:
            hatalar.append(f"{k}: τ* {tau} ≠ ön-kayıt {ONKAYIT_TAU[k]}")
        epoch[k] = [m.get("en_iyi_epoch") for m in M]
        if k in ("h01", "h12"):
            cs = [m["secilen_C"] for m in M]
            cyildiz[k] = cs[0]
            if len(set(cs)) != 1:
                hatalar.append(f"{k}: secilen_C seed'ler arasında farklı {cs}")
        if k == "baseline":
            baseline_test = {q: [m[q] for m in M] for q in ("test_acc", "test_auc", "test_acc_son", "test_auc_son")}
            baseline_test["test_acc_ort"] = float(np.mean(baseline_test["test_acc"]))
        kaynak_sha[k] = {d: sha(runs / d / "tahmin_val.npy") for d in adlar}
        for d in adlar:
            for ad in ("best_model.pth", "sonda_agirlik.npz", "tahmin_test.npy"):
                f = drive / "runs" / d / ad
                if f.exists():
                    st = f.stat()
                    drive_agirlik.setdefault(d, {})[ad] = {"bayt": st.st_size,
                                                            "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")}
            gerek = {"baseline": "tahmin_test.npy", "h01": "sonda_agirlik.npz", "h12": "sonda_agirlik.npz"}.get(k, "best_model.pth")
            if gerek not in drive_agirlik.get(d, {}):
                hatalar.append(f"{d}: Drive'da {gerek} yok")
    if abs(np.mean(katman["h01"]["T_tamval_seed"]) - ONKAYIT_T_H01) > 0.005:
        hatalar.append(f"H01c T {katman['h01']['T_tamval_seed']} ≠ ön-kayıt {ONKAYIT_T_H01}")
    katlar = WS / "varyantlar" / "logit_ayari" / "katlar.json"
    katlar_sha = sha(katlar)
    if not (katlar_sha.startswith("0db03edb") and katlar_sha.endswith("e0193")):
        hatalar.append(f"katlar.json SHA256 {katlar_sha} ≠ 0db03edb…e0193")
    dosyalar = [WS / "hipotezler" / "K5-ONKAYIT.md", WS / "hipotezler" / "K5-ONKAYIT-b.md", WS / "scripts" / "k5_test.py", WS / "scripts" / "c_bolme_npz.py",
                WS / "scripts" / "k5_c_denetim.py", WS / "scripts" / "k5_dondurma.py", WS / "sonuclar" / "k5_c_denetim.json",
                WS / "kuyruk" / "gruplar.json", WS / "colab" / "hat" / "k5_kuyruk_yaz.py"] + \
        sorted((WS / "varyantlar" / "k5").glob("*.py")) + sorted((WS / "kuyruk").glob("k5-*.jsonl")) + \
        sorted((WS / "kuyruk").glob("k5-*.beklenti.json"))
    c_den = json.loads((WS / "sonuclar" / "k5_c_denetim.json").read_text(encoding="utf-8"))
    if c_den.get("karar") != "GEÇTİ":
        hatalar.append("P7 k5_c_denetim GEÇMEDİ")
    paket = json.loads((drive / "PAKET.json").read_text(encoding="utf-8")) if (drive / "PAKET.json").exists() else {}
    ok = WS / "hipotezler" / "K5-ONKAYIT.md"
    D = {
        "aciklama": "K5 DONMUŞ PARAMETRELER (K5-ONKAYIT §0 P4). Val'den, test açılmadan. Test'te hiçbiri seçilmez/aranmaz/ayarlanmaz.",
        "zaman": datetime.now().isoformat(timespec="seconds"),
        "onkayit": {"dosya": "hipotezler/K5-ONKAYIT.md", "mtime": datetime.fromtimestamp(ok.stat().st_mtime).isoformat(timespec="seconds"),
                    "onay": "KULLANICI KARARI 10 (orkestrasyon_durumu.md, 2026-10-11 18:14)"},
        "onkayit_sha256": sha(ok), "k5_test_sha256": sha(WS / "scripts" / "k5_test.py"),
        "kazanan_W": "H10 (h10_dinov2b_k4_ft224) — H13 Tur-5'te ÖLDÜR (tahlil-tur5 / SIRALAMA-TUR5); T4 ek tablo; M3 yok",
        "test_kollari": {k: kollar[k] for k in KT.KOL},
        "test_disi": ["H03", "H05", "H07", "H09", "H13 (ÖLDÜR)"],
        "katman": katman,
        "sonda_C_yildiz": cyildiz,
        "en_iyi_epoch": epoch,
        "baseline_c_dondurulmus_test": baseline_test,
        "istatistik": {"B": 2000, "rng": 0, "rng_cagri": "numpy.random.default_rng(0).integers(0, N, (B, N))",
                       "t": "scipy.stats.ttest_rel(alternative='greater'), n = 5",
                       "holm_aile1": ["h01-baseline", "h02-baseline", "h08-baseline", "h10-baseline"],
                       "holm_aile2": ["h08-h02", "h10-h02"], "alfa": 0.05,
                       "pratik_marj": {"aile1_puan": 2.0, "aile2_puan": 0.5}},
        "katlar_json_sha256": katlar_sha,
        "c_denetim_P7": {k: c_den[k] for k in ("P7_ctest_lezyon_kesisim_resmi_train_val", "P7_ctest_resmi_test_alt_kume",
                                                "P7_ctest_esleme_tutarli", "karar")},
        "kaynak_tahmin_val_sha256": kaynak_sha,
        "drive_kaynak_dosyalar": drive_agirlik,
        "dosya_sha256": {str(p.relative_to(WS)).replace("\\", "/"): sha(p) for p in dosyalar},
        "paket_surum": paket.get("surum"),
        "c1_veri_sha256": "TEST GEÇİŞİNDEN SONRA eklenir (§4.1): kuyruk/k5_c_bolme_beklenen.json (c_bolme_npz.py --ilk-yapim)",
        "hatalar": hatalar,
    }
    out = WS / "sonuclar" / "k5_dondurma.json"
    out.write_text(json.dumps(D, ensure_ascii=False, indent=1), encoding="utf-8")
    h = sha(out)
    out.with_suffix(".json.sha256").write_text(f"{h}  k5_dondurma.json  {D['zaman']}\n", encoding="utf-8")
    kopya = []
    if drive.exists():
        (drive / "sonuclar").mkdir(exist_ok=True)
        for f in (out, out.with_suffix(".json.sha256")):
            shutil.copyfile(f, drive / "sonuclar" / f.name)
            kopya.append(str(drive / "sonuclar" / f.name))
        if sha(drive / "sonuclar" / out.name) != h:
            hatalar.append("Drive kopyası SHA256 uyuşmuyor")
    print(json.dumps({"tau": {k: v["tau_tamval"] for k, v in katman.items()},
                      "T": {k: v["T_tamval_seed"] for k, v in katman.items()}, "C*": cyildiz, "sha256": h,
                      "drive": kopya, "hatalar": hatalar}, ensure_ascii=False, indent=1))
    return 0 if not hatalar else 2


if __name__ == "__main__":
    sys.exit(main())
