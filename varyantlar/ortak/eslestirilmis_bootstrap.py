#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eslestirilmis_bootstrap.py — İki koşum kümesi (A = ebeveyn/referans, B = varyant) arasında AYNI val görüntüleri
üzerinde eşleştirilmiş görüntü bootstrap'ı: Δ val ACC = ACC(B) − ACC(A), %95 yüzdelik GA.

Gerekçe (sonuclar/hata-analizi.md §5): 5 seed'li eşleştirilmiş t-testinin en küçük saptanabilir etkisi ≈ 2,7–3,9 puan;
Tur-2/3'ün ±1–2 puanlık etkileri için görüntü düzeyinde eşleştirilmiş bootstrap eşlik eder (K4c'de leaderboard eki).

Tanım:
- Her küme için görüntü başına seed-ortalamalı doğruluk d̄_X(i) = (1/n_X) Σ_s 1[argmax p_s(i) = y_i].
- Nokta tahmini Δ = ort_i(d̄_B − d̄_A) = ort(ACC_B seed'leri) − ort(ACC_A seed'leri).
- Bootstrap: n = 2.000 yeniden örnekleme, RNG = np.random.default_rng(0); her örneklemede AYNI görüntü indeksleri
  iki kola uygulanır (eşleştirme). Varsayılan düz görüntü yeniden örneklemesi; --sinif-tabakali ile sınıf içi
  (hata_kirilim.py'deki sızıntı GA'sı gibi).
- Ek çıktı: p_tek = P*(Δ* ≤ 0) (bootstrap tek yönlü, B > A yönü), seed eşleşmesi raporu (M5: aynı seed kümesi).
- Yalnız tahmin_val.npy / etiket_val.npy okunur (TEST YOK, M4). Etiketler bütün koşumlarda aynı olmalı.

Kullanım (WS kökünden):
  env/.venv/Scripts/python.exe varyantlar/ortak/eslestirilmis_bootstrap.py --a-varyant baseline --b-varyant h01_dinov2b_lp_r224
  ... --a <koşum_dizini ...> --b <koşum_dizini ...> [--sinif-tabakali] [--n 2000] [--cikti sonuc.json]
  H02 seçim-kuralı kolu: --a-varyant baseline --b-varyant h02_r18in_ft224 --b-dosya tahmin_val_aucsecim.npy
Python: from ortak.eslestirilmis_bootstrap import eslestirilmis_bootstrap
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]


def eslestirilmis_bootstrap(PA, PB, y, n: int = 2000, rng_seed: int = 0, sinif_tabakali: bool = False,
                            alfa: float = 0.05) -> dict:
    """PA: (nA, N, K) veya (N, K) olasılık; PB aynı biçimde; y: (N,). Dönüş: delta, ga_alt, ga_ust, p_tek, ..."""
    y = np.asarray(y).reshape(-1).astype(np.int64)
    PA, PB = np.asarray(PA), np.asarray(PB)
    PA = PA[None] if PA.ndim == 2 else PA
    PB = PB[None] if PB.ndim == 2 else PB
    dA = (PA.argmax(2) == y[None]).mean(0)
    dB = (PB.argmax(2) == y[None]).mean(0)
    fark = dB - dA
    N = len(y)
    rng = np.random.default_rng(rng_seed)
    if sinif_tabakali:
        gruplar = [np.where(y == c)[0] for c in np.unique(y)]
        bs = np.empty(n)
        for b in range(n):
            idx = np.concatenate([rng.choice(g, len(g)) for g in gruplar])
            bs[b] = fark[idx].mean()
    else:
        bs = fark[rng.integers(0, N, size=(n, N))].mean(1)
    return {"delta": float(fark.mean()), "ga_alt": float(np.quantile(bs, alfa / 2)),
            "ga_ust": float(np.quantile(bs, 1 - alfa / 2)), "p_tek": float((np.sum(bs <= 0) + 1) / (n + 1)),
            "acc_a": float(dA.mean()), "acc_b": float(dB.mean()), "n_a": int(len(PA)), "n_b": int(len(PB)),
            "n_goruntu": int(N), "bootstrap_n": int(n), "rng": int(rng_seed), "sinif_tabakali": bool(sinif_tabakali),
            "farkli_dogru_goruntu": int((fark != 0).sum())}


def cift_fark(P) -> dict:
    """(n, N, K) → {maks, en_kucuk}: koşum çiftleri arası max|p_i − p_j|; n < 2 → boş."""
    if len(P) < 2:
        return {}
    f = [float(np.max(np.abs(P[i] - P[j]))) for i in range(len(P)) for j in range(i + 1, len(P))]
    return {"maks": max(f), "en_kucuk": min(f), "n_cift": len(f)}


def kosum_bul(varyant: str, kok: Path = WS, dosya: str = "tahmin_val.npy") -> list:
    out = []
    for d in sorted((kok / "runs").glob("*/")):
        cfg, done = d / "config.json", d / ".done"
        if not (cfg.exists() and done.exists() and (d / dosya).exists()):
            continue
        c = json.loads(cfg.read_text(encoding="utf-8"))
        j = json.loads(done.read_text(encoding="utf-8"))
        if c.get("varyant") == varyant and j.get("rc") == 0:
            out.append(d)
    return out


def yukle(dizinler, dosya: str = "tahmin_val.npy"):
    P, Y, seed = [], None, []
    for d in dizinler:
        d = Path(d)
        y = np.load(d / "etiket_val.npy").reshape(-1).astype(np.int64)
        if Y is None:
            Y = y
        elif not np.array_equal(Y, y):
            raise SystemExit(f"HATA: {d.name} etiket_val diğer koşumlarla hizasız")
        P.append(np.load(d / dosya).astype(np.float64))
        try:
            seed.append(str(json.loads((d / "metrics.json").read_text(encoding="utf-8")).get("seed")))
        except Exception:  # noqa: BLE001
            seed.append("?")
    return np.stack(P), Y, seed


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", nargs="*", default=None, help="A (referans) koşum dizinleri")
    p.add_argument("--b", nargs="*", default=None, help="B (varyant) koşum dizinleri")
    p.add_argument("--a-varyant", default=None)
    p.add_argument("--b-varyant", default=None)
    p.add_argument("--kok", default=str(WS))
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--rng", type=int, default=0)
    p.add_argument("--sinif-tabakali", action="store_true")
    p.add_argument("--a-dosya", default="tahmin_val.npy", help="A kolunun tahmin dosyası")
    p.add_argument("--b-dosya", default="tahmin_val.npy",
                   help="B kolunun tahmin dosyası (H02 val_acc_aucsecim kolu: tahmin_val_aucsecim.npy)")
    p.add_argument("--cikti", default=None)
    a = p.parse_args()
    kok = Path(a.kok)
    A = a.a or kosum_bul(a.a_varyant, kok, a.a_dosya)
    B = a.b or kosum_bul(a.b_varyant, kok, a.b_dosya)
    if not A or not B:
        raise SystemExit("HATA: A veya B kümesi boş")
    PA, yA, sA = yukle(A, a.a_dosya)
    PB, yB, sB = yukle(B, a.b_dosya)
    if not np.array_equal(yA, yB):
        raise SystemExit("HATA: A ve B val etiketleri hizasız (aynı val görüntüleri değil)")
    r = eslestirilmis_bootstrap(PA, PB, yA, a.n, a.rng, a.sinif_tabakali)
    r.update({"a": a.a_varyant or "elle", "b": a.b_varyant or "elle", "seed_a": sA, "seed_b": sB,
              "seed_eslesmesi": sorted(sA) == sorted(sB),
              "kosum_a": [Path(d).name for d in A], "kosum_b": [Path(d).name for d in B],
              "a_dosya": a.a_dosya, "b_dosya": a.b_dosya,
              # seed bağlılığı (H01/H03 R5; koşumlar arası): farklı seed tahminleri arasındaki en büyük/en küçük
              # çift-maks mutlak fark (sonda kuyruklarında farkli_seedler_farkli yerine; > 0 beklenir)
              "seed_tahmin_fark_a": cift_fark(PA), "seed_tahmin_fark_b": cift_fark(PB)})
    if not r["seed_eslesmesi"]:
        print(f"UYARI (M5): seed kümeleri farklı A={sA} B={sB}")
    print(json.dumps(r, ensure_ascii=False, indent=1))
    if a.cikti:
        Path(a.cikti).write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
