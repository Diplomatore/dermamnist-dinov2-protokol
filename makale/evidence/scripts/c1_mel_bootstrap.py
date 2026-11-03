#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
c1_mel_bootstrap.py — TEST-SONRASI ANALİZ (revizyon 2026-11-03; eleştiri M3, notes/elestiri-plani.md R3).

C1 (C-train ile yeniden eğitim, C-test n = 1.227) için DONDURULMUŞ C-test tahminlerinden:
  (a) her kolun (baseline, H01c, H02c, H08c, H10) mel duyarlılığı: seed başına duyarlılık → ort ± std (ddof=1, n=5);
  (b) eşleştirilmiş farkların (H10−H02c, H10−H08c, H08c−H02c) mel duyarlılığı Δ'sı + %95 bootstrap GA'sı.
Yeni eğitim YOK; resmî test tahminleri AÇILMAZ (yalnız C1 C-test tahminleri ve resmî test ETİKETLERİ sıra denetimi için);
hiçbir seçim/karar/ön-kayıt sonucu değişmez. Statü: LKB analizi (2026-10-17) ile aynı — test-sonrası, betimsel.

Yöntem (K5 C1 analiziyle birebir aynı): dondurulmuş `scripts/k5_test.py` (SHA256 13a71cf1…) modül olarak yüklenir;
c1_analiz'in C-test kod yolu aynen izlenir: C1 tahminleri C CSV sırasından artan resmî indekse sıralanır (sira),
`K5.Bootstrap(1227)` (B = 2.000, numpy default_rng(0)) TEK W matrisi iki kola da uygulanır, Δ = B − A
seed-ortalamalı görüntü doğruluğundan. Mel duyarlılığı Δ'sı, K5.kiyas'taki dengeli-ACC sınıf duyarlılığı
formülünün tek sınıflı (mel = 4) hâlidir: r* = W[:, mel]·d[mel] / ΣW[:, mel]; GA %2,5–%97,5 yüzdelik (K5.ga);
p = (1 + #{Δ* ≤ 0}) / (B_geçerli + 1). Önce K5.kiyas ile 7 C1 çifti ve C1.kol özetleri yeniden üretilir ve
`sonuclar/k5_c1_sonuc.json` ile alan alan karşılaştırılır; uyuşmazlık varsa betik DURUR (rc 2).

Kullanım: <DW>/env/.venv/Scripts/python.exe -X utf8 -I evidence/scripts/c1_mel_bootstrap.py
Çıktı: evidence/c1mel/c1_mel_sonuc.json
"""
import csv
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
WS = Path(__file__).resolve().parents[2]
DW = WS.parent  # depo kökü = deney çalışma alanı (makale/ = WS)
RC = DW / "runs_colab" if (DW / "runs_colab").is_dir() else DW / "runs_meta" / "runs_colab"  # depoda yalnız JSON damgaları runs_meta/ altında
RUNS = RC  # tahmin .npy'leri depoda YOK (bkz. makale/README.md)
CIKTI = WS / "evidence" / "c1mel" / "c1_mel_sonuc.json"
LKB = WS / "evidence" / "lkb" / "lkb_bootstrap_sonuc.json"
DONMUS_K5_SHA = "13a71cf1375fc98a2e8c891bb4f5fdc884e43deb6819ba740aa4035428179b4c"
TOL = 1e-9
MEL = 4


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


k5_yol = DW / "scripts" / "k5_test.py"
if sha256(k5_yol) != DONMUS_K5_SHA:
    raise SystemExit("HATA: k5_test.py SHA256 dondurulmuş değerle aynı değil")
spec = importlib.util.spec_from_file_location("k5_test", k5_yol)
K5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(K5)
assert K5.SINIF[MEL] == "mel" and K5.NB == 2000 and K5.RNG == 0

C = json.loads((DW / "sonuclar/k5_c1_sonuc.json").read_text(encoding="utf-8"))
LK = json.loads(LKB.read_text(encoding="utf-8"))
LKB_SHA = LK["tahmin_dosyalari_sha256"]


def kiyasla(donmus, yeni, yol, rapor):
    if isinstance(donmus, dict):
        for k, v in donmus.items():
            if k in yeni:
                kiyasla(v, yeni[k], f"{yol}.{k}", rapor)
            else:
                rapor["eksik"].append(f"{yol}.{k}")
    elif isinstance(donmus, list):
        if len(donmus) != len(yeni):
            rapor["uyusmaz"].append(f"{yol}: uzunluk")
        for i, (a, b) in enumerate(zip(donmus, yeni)):
            kiyasla(a, b, f"{yol}[{i}]", rapor)
    elif isinstance(donmus, bool) or isinstance(donmus, str) or donmus is None:
        rapor["n"] += 1
        if donmus != yeni:
            rapor["uyusmaz"].append(f"{yol}: {donmus!r} ≠ {yeni!r}")
    else:
        rapor["n"] += 1
        d = abs(float(donmus) - float(yeni))
        rapor["maks"] = max(rapor["maks"], d)
        if d > TOL:
            rapor["uyusmaz"].append(f"{yol}: {donmus} ≠ {yeni}")


def main():
    t0 = time.time()
    # ── C1 tahminleri: dondurulmuş JSON'daki koşum adları ──
    dosya_sha, sha_uyusmaz = {}, []
    PC, yc0 = {}, None
    for k, adlar in C["C1"]["kosumlar"].items():
        seedler = [int(json.loads((RUNS / a / "config.json").read_text(encoding="utf-8"))["seed"]) for a in adlar]
        if seedler != [0, 1, 2, 3, 4]:
            raise SystemExit(f"HATA: {k} seed sırası {seedler}")
        P = []
        for a in adlar:
            for f in ("tahmin_test.npy", "etiket_test.npy"):
                anahtar = f"runs_colab/{a}/{f}"
                h = sha256(RUNS / a / f)
                dosya_sha[anahtar] = h
                if LKB_SHA.get(anahtar) != h:
                    sha_uyusmaz.append(anahtar)
            y = np.load(RUNS / a / "etiket_test.npy").reshape(-1).astype(np.int64)
            if yc0 is not None and not np.array_equal(y, yc0):
                raise SystemExit(f"HATA: {k}: C-test etiketleri kollar/seed'ler arasında farklı")
            yc0 = y
            P.append(np.load(RUNS / a / "tahmin_test.npy").astype(np.float64))
        PC[k] = np.stack(P)
    if sha_uyusmaz:
        raise SystemExit(f"HATA: tahmin dosyası SHA256 LKB kaydıyla uyuşmuyor: {sha_uyusmaz[:5]}")
    # ── sıra: c1_analiz ile aynı ──
    idx_c = K5.c_alt_kume("test", "test")
    if len(idx_c) != len(yc0):
        raise SystemExit(f"HATA: C-test {len(yc0)} ≠ eşleme {len(idx_c)}")
    with np.load(DW / "data/medmnist/dermamnist.npz") as z:  # yalnız ETİKET (sıra denetimi); tahmin değil
        y_res = z["test_labels"].reshape(-1).astype(np.int64)
    if not np.array_equal(yc0, y_res[idx_c]):
        raise SystemExit("HATA: C-test etiketleri resmî testin aynı indeksleriyle aynı değil")
    sira = np.argsort(idx_c)
    PC = {k: v[:, sira] for k, v in PC.items()}
    yc = yc0[sira]
    BSc = K5.Bootstrap(len(yc))

    # ── (1) yeniden üretim: C1.kol özetleri + 7 çift K5.kiyas ──
    rp = {"n": 0, "maks": 0.0, "uyusmaz": [], "eksik": []}
    kol_yeni = {}
    for k in PC:
        acc = (PC[k].argmax(2) == yc[None]).mean(1)
        kol_yeni[k] = {"acc_ort": float(acc.mean()), "acc_std": float(acc.std(ddof=1)),
                       "acc_eksi_cogunluk": float(acc.mean() - K5.CTEST_COGUNLUK),
                       "dengeli_ort": float(np.mean([K5.dengeli(yc, PC[k][s].argmax(1)) for s in range(5)])),
                       "dengeli5_ort": float(np.mean([K5.dengeli(yc, PC[k][s].argmax(1), K5.BES) for s in range(5)])),
                       "auc_ort": float(np.mean([K5.auc(yc, PC[k][s]) for s in range(5)]))}
    kiyasla(C["C1"]["kol"], kol_yeni, "C1.kol", rp)
    CIFT = [("baseline", "h01"), ("baseline", "h02"), ("baseline", "h08"), ("baseline", "h10"), ("h02", "h08"),
            ("h02", "h10"), ("h08", "h10")]
    kiy = {f"{B}-{A}": K5.kiyas(PC[A], PC[B], yc, BSc) for A, B in CIFT}
    kiyasla(C["C1"]["kiyas"], kiy, "C1.kiyas", rp)
    print(f"[yeniden üretim] karşılaştırılan alan {rp['n']}, maks |fark| {rp['maks']:.3e}, "
          f"uyuşmaz {len(rp['uyusmaz'])}, eksik {len(rp['eksik'])}  ({time.time() - t0:.1f} sn)")
    if rp["uyusmaz"] or rp["eksik"]:
        print("\n".join((rp["uyusmaz"] + rp["eksik"])[:40]))
        raise SystemExit(2)

    # ── (2) kol başına sınıf duyarlılıkları (seed başına; mel birincil) ──
    m = yc == MEL
    n_mel = int(m.sum())
    lez = {}
    with open(DW / "data/splits/dermamnist_lezyon_esleme.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "test":
                lez[int(r["index"])] = r["lesion_id"]
    lez_c = np.array([lez[int(i)] for i in idx_c[sira]])
    n_mel_lez = int(len(set(lez_c[m])))
    sinif_n = {K5.SINIF[c]: int((yc == c).sum()) for c in range(K5.K)}
    kol = {}
    for k in PC:
        t = PC[k].argmax(2)
        d = {}
        for c in range(K5.K):
            if (yc == c).any():
                v = [float((t[s][yc == c] == c).mean()) for s in range(5)]
                d[K5.SINIF[c]] = {"seed": v, "ort": float(np.mean(v)), "std": float(np.std(v, ddof=1))}
        dv = [K5.dengeli(yc, t[s]) for s in range(5)]
        assert abs(np.mean(dv) - C["C1"]["kol"][k]["dengeli_ort"]) < TOL
        kol[k] = {"duyarlilik": d, "dengeli_acc": {"seed": dv, "ort": float(np.mean(dv)), "std": float(np.std(dv, ddof=1))}}

    # ── (3) eşleştirilmiş mel duyarlılığı Δ (aynı BSc.W) ──
    def mel_kiyas(A, B):
        dA = (PC[A].argmax(2) == yc[None]).mean(0)
        dB = (PC[B].argmax(2) == yc[None]).mean(0)
        Wm = BSc.W[:, m]
        sm = Wm.sum(1)
        ok = sm > 0
        rA = (Wm @ dA[m]) / np.maximum(sm, 1)
        rB = (Wm @ dB[m]) / np.maximum(sm, 1)
        fb = (rB - rA)[ok]
        fd = np.array(kol[B]["duyarlilik"]["mel"]["seed"]) - np.array(kol[A]["duyarlilik"]["mel"]["seed"])
        return {"A": float(dA[m].mean()), "B": float(dB[m].mean()), "delta": float(dB[m].mean() - dA[m].mean()),
                "ga": K5.ga(fb), "p_boot": float((np.sum(fb <= 0) + 1) / (len(fb) + 1)), "n_gecerli_ornek": int(ok.sum()),
                "ga_genislik": float(K5.ga(fb)[1] - K5.ga(fb)[0]),
                "fark_seed": fd.tolist(), "fark_seed_ort": float(fd.mean()), "fark_seed_sd": float(fd.std(ddof=1)),
                "kural": K5.etiket_kural(K5.ga(fb))}

    HEDEF = [("h02", "h10"), ("h08", "h10"), ("h02", "h08")]
    mel = {f"{B}-{A}": mel_kiyas(A, B) for A, B in HEDEF}

    # ── (4) lezyon-kümeli duyarlılık (LKB ile aynı KumeBootstrap; tab:protokol ‡ işaretinin tutarlılığı için) ──
    class KumeBootstrap:  # lkb_bootstrap.py ile aynı (örnekleme birimi lezyon; rng(0); B = 2.000)
        def __init__(self, kume_etiket):
            self.N = len(kume_etiket)
            _, kod = np.unique(kume_etiket, return_inverse=True)
            L = int(kod.max()) + 1
            rng = np.random.default_rng(K5.RNG)
            idx = rng.integers(0, L, size=(K5.NB, L))
            Ck = np.zeros((K5.NB, L))
            for b in range(K5.NB):
                Ck[b] = np.bincount(idx[b], minlength=L)
            self.W = Ck[:, kod]
            self.L = L
            self.toplam = self.W.sum(1)

        def ort(self, f, m=None):
            if m is None:
                return self.W @ f / self.toplam
            return (self.W[:, m] @ f[m]) / np.maximum(self.W[:, m].sum(1), 1)

    KB = KumeBootstrap(lez_c)
    assert np.array_equal(KumeBootstrap(np.arange(len(yc))).W, BSc.W), "tekil-küme sınaması"
    BSc_yedek = BSc
    BSc = KB
    mel_lez = {f"{B}-{A}": mel_kiyas(A, B) for A, B in HEDEF}
    BSc = BSc_yedek
    kiy_lez = {f"{B}-{A}": K5.kiyas(PC[A], PC[B], yc, KB) for A, B in HEDEF}
    LZ = LK["tam_lezyon"]["C1"]["kiyas"]
    rl = {"n": 0, "maks": 0.0, "uyusmaz": [], "eksik": []}
    for c in kiy_lez:
        for q in ("acc", "dengeli_acc", "dengeli_acc_5sinif"):
            kiyasla({"delta": LZ[c][q]["delta"], "ga": LZ[c][q]["ga"], "p_boot": LZ[c][q]["p_boot"]},
                    kiy_lez[c][q], f"LKB.C1.kiyas.{c}.{q}", rl)
    print(f"[lezyon-kümeli yeniden üretim, LKB'ye karşı] alan {rl['n']}, maks |fark| {rl['maks']:.3e}, "
          f"uyuşmaz {len(rl['uyusmaz'])}")
    if rl["uyusmaz"] or rl["eksik"]:
        print("\n".join((rl["uyusmaz"] + rl["eksik"])[:20]))
        raise SystemExit(4)

    def konum(g):
        return K5.etiket_kural(g).split(" (")[0]

    lezyon = {}
    for c in mel:
        lezyon[c] = {
            "mel": {"ga": mel_lez[c]["ga"], "p_boot": mel_lez[c]["p_boot"], "n_gecerli_ornek": mel_lez[c]["n_gecerli_ornek"],
                    "konum_goruntu": konum(mel[c]["ga"]), "konum_lezyon": konum(mel_lez[c]["ga"]),
                    "isaret_degisti": konum(mel[c]["ga"]) != konum(mel_lez[c]["ga"])},
            "dengeli_acc": {"ga": kiy_lez[c]["dengeli_acc"]["ga"], "p_boot": kiy_lez[c]["dengeli_acc"]["p_boot"],
                            "konum_goruntu": konum(kiy[c]["dengeli_acc"]["ga"]),
                            "konum_lezyon": konum(kiy_lez[c]["dengeli_acc"]["ga"]),
                            "isaret_degisti": konum(kiy[c]["dengeli_acc"]["ga"]) != konum(kiy_lez[c]["dengeli_acc"]["ga"])}}
    # A/B tutarlılığı: bootstrap dışı nokta = kol seed-ortalaması
    for c, e in mel.items():
        b, a = c.split("-")
        assert abs(e["A"] - kol[a]["duyarlilik"]["mel"]["ort"]) < 1e-12
        assert abs(e["B"] - kol[b]["duyarlilik"]["mel"]["ort"]) < 1e-12
    dengeli_yeniden = {c: {"delta_puan": 100 * kiy[c]["dengeli_acc"]["delta"],
                           "ga_puan": [100 * v for v in kiy[c]["dengeli_acc"]["ga"]],
                           "p_boot": kiy[c]["dengeli_acc"]["p_boot"],
                           "donmus_delta_puan": 100 * C["C1"]["kiyas"][c]["dengeli_acc"]["delta"],
                           "donmus_ga_puan": [100 * v for v in C["C1"]["kiyas"][c]["dengeli_acc"]["ga"]]}
                       for c in mel}

    out = {
        "etiket": "test-sonrası analiz (revizyon 2026-11-03) — dondurulmuş C1 C-test tahminleri; yeni eğitim yok; "
                  "resmî test tahminleri açılmadı; hiçbir seçim/karar/ön-kayıt sonucu değişmez",
        "gerekce": "dış eleştiri M3 (paper/eleştiri.txt; notes/elestiri-plani.md R3)",
        "tarih": datetime.now().isoformat(timespec="seconds"),
        "betik": "evidence/scripts/c1_mel_bootstrap.py", "betik_sha256": sha256(__file__),
        "k5_test_sha256": DONMUS_K5_SHA,
        "donmus_kaynak_sha256": {"sonuclar/k5_c1_sonuc.json": sha256(DW / "sonuclar/k5_c1_sonuc.json"),
                                 "data/splits/dermamnist_c_esleme.csv": sha256(DW / "data/splits/dermamnist_c_esleme.csv"),
                                 "data/splits/dermamnist_lezyon_esleme.csv":
                                     sha256(DW / "data/splits/dermamnist_lezyon_esleme.csv")},
        "yontem": {"B": K5.NB, "rng": f"numpy.random.default_rng({K5.RNG}); tek Bootstrap(1227) nesnesi (K5 c1_analiz BSc ile aynı)",
                   "sira": "C CSV sırası → artan resmî indeks (np.argsort(idx_c)), c1_analiz ile aynı",
                   "mel_delta": "seed-ortalamalı görüntü doğruluğu d; r* = W[:,mel]·d[mel]/ΣW[:,mel]; Δ* = r*_B − r*_A; "
                                "GA %2,5–%97,5 (K5.ga); p = (1+#{Δ*≤0})/(B_geçerli+1) — K5.kiyas dengeli-ACC formülünün tek sınıflı hâli",
                   "kol_std": "seed başına duyarlılık, np.std(ddof=1), n = 5 seed"},
        "dogrulama": {"karsilastirilan_alan": rp["n"], "maks_mutlak_fark": rp["maks"], "uyusmaz": len(rp["uyusmaz"]),
                      "tolerans": TOL, "kapsam": "k5_c1_sonuc.json: C1.kol (5 kol × 6 alan), C1.kiyas (7 çift, tüm alanlar)",
                      "tahmin_sha_lkb_ile_ayni": True, "n_tahmin_dosyasi": len(dosya_sha)},
        "n_ctest": int(len(yc)), "n_mel": n_mel, "n_mel_lezyon": n_mel_lez, "sinif_n": sinif_n,
        "kol": kol, "mel_kiyas": mel, "dengeli_yeniden_uretim": dengeli_yeniden,
        "lezyon_kumeli": {"aciklama": "LKB (lkb_bootstrap.py) ile aynı KumeBootstrap; C-test lezyon kümesi "
                                      f"{KB.L}; acc/dengeli/dengeli5 LKB tam_lezyon.C1.kiyas ile yeniden üretildi",
                          "n_lezyon_ctest": int(KB.L),
                          "dogrulama": {"karsilastirilan_alan": rl["n"], "maks_mutlak_fark": rl["maks"],
                                        "uyusmaz": len(rl["uyusmaz"]), "karsi": "evidence/lkb/lkb_bootstrap_sonuc.json"},
                          "cift": lezyon},
        "kosumlar": C["C1"]["kosumlar"], "tahmin_dosyalari_sha256": dosya_sha,
        "sure_sn": round(time.time() - t0, 1),
    }
    CIKTI.parent.mkdir(parents=True, exist_ok=True)
    CIKTI.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"C-test n = {len(yc)}, mel n = {n_mel} ({n_mel_lez} lezyon); sınıflar {sinif_n}")
    for k in kol:
        v = kol[k]["duyarlilik"]["mel"]
        print(f"  {k:<9} mel duyarlılığı {100 * v['ort']:6.2f} ± {100 * v['std']:5.2f}  seed {[round(100 * x, 1) for x in v['seed']]}")
    for c, e in mel.items():
        print(f"  {c:<8} Δmel {100 * e['delta']:+6.2f} [{100 * e['ga'][0]:+6.2f}; {100 * e['ga'][1]:+6.2f}] p={e['p_boot']:.4f} "
              f"| Δdengeli {dengeli_yeniden[c]['delta_puan']:+6.2f} [{dengeli_yeniden[c]['ga_puan'][0]:+6.2f}; "
              f"{dengeli_yeniden[c]['ga_puan'][1]:+6.2f}]  {e['kural']}")
        lz = lezyon[c]
        print(f"           lezyon-kümeli: mel [{100 * lz['mel']['ga'][0]:+6.2f}; {100 * lz['mel']['ga'][1]:+6.2f}] "
              f"{'DEĞİŞTİ' if lz['mel']['isaret_degisti'] else ''} | dengeli [{100 * lz['dengeli_acc']['ga'][0]:+6.2f}; "
              f"{100 * lz['dengeli_acc']['ga'][1]:+6.2f}] {'DEĞİŞTİ' if lz['dengeli_acc']['isaret_degisti'] else ''}")
    print(f"yazıldı: {CIKTI}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
