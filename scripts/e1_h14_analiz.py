# -*- coding: utf-8 -*-
"""
e1_h14_analiz.py — E1 / H14 test sonrası KEŞİFSEL analiz (hipotezler/H14.md "İstatistik ve karşılaştırma"). CPU; karar
VERMEZ (TUT/ÖLDÜR yok), etiketleri ve önceden yazılmış makale cümlesini MEKANİK seçer. K5 Holm ailelerine girmez.

Sıra (kart):
 0) `scripts/k5_test.py` SHA256'sı donmuş değerle (k5_dondurma.json / k5_c1_sonuc.json betik_sha256) aynı olmalı: bootstrap ve
    metrik işlevleri ondan İÇE AKTARILIR (Bootstrap, kiyas, kol_ozet, dengeli, c_alt_kume, kosumlar; yeniden yazılmaz).
 1) KURU KOŞU ÖZDEŞLİĞİ (DUR koşulu; H14 C-test okunmadan ÖNCE): arşiv kollarıyla `sonuclar/k5_c1_sonuc.json` (i) satırları
    yeniden üretilir — H10 − H08 −0,96 [−2,10; +0,18], H08 − H02 +2,09 [+1,24; +2,97], H10 − H02 +1,12 [−0,05; +2,30].
    Nokta ve GA uçlarında |fark| ≤ 0,005 puan; aksi → DUR (rc 2).
 2) Kollar: H14 (`h14_c1-test`, kuyruk e1-h14), arşiv `h10_c1-test`, `h08_c1-test`, `h02_c1-test` (kuyruk k5-c1), Durum B'de
    H10r (`h10r_c1-test`, kuyruk e1-h10r). Etiket dizileri bütün kollarda aynı olmalı (aksi DUR). `veri_c` ≠ 1 olan bir
    h14/h10r çıkarımı → DUR (kural i). C-test sırası → artan resmî indeks (k5_test c1_analiz ile AYNI bootstrap örneklemi).
 3) Kural (f) Durum A/B/C kontrol kolundan (h14_c1_d4kapali_duman) mekanik; C → DUR; B → H10r zorunlu + h10r_c1_duman ↔
    kontrol bit-bit (|Δ ilk_adim_loss| ≤ 1e-6 ∧ maks|Δp| ≤ 1e-5) aksi DUR. Kural (g) denetimleri (D4 duman ≠ kontrol;
    augment frekans sapması ≤ 0,01; görünüm tutarlılığı ≤ 5e-3; özdeş görünüm 0) aksi DUR.
 4) Kıyaslar (düzeltmesiz): K1 = H14 − H10 (Durum B: − H10r; ikincil − H10 arşiv + sürüklenme H10r − H10), K2 = H14 − H08c,
    betimsel H14 − H02c. Her biri: ACC, dengeli, dengeli-5, mel duyarlılığı — görüntü bootstrap'ı B = 2.000, rng(0), aynı W
    iki kola; yüzdelik %95 GA; tek yönlü p'ler (1 + #{Δ* ≤ 0})/(B+1) ve tersi; kesin seed-eşleştirilmiş t (iki yönlü; bilgi).
    Etiket: GA alt > 0 "H14 üstün" · GA üst < 0 "H14 geride" · aksi "ayırt edilemedi" + "(test sonrası keşifsel, düzeltmesiz)".
 5) Görülmemiş (İNCELE) eşikleri, manipülasyon denetimi, kural (e)/(j) notları, ikinciller, önceden yazılmış cümle seçimi.

Kipler:
  --kuru   H14 YERİNE arşiv h10_c1 konur (K1 = 0 beklenir; K2 = H10 − H08). Yeni hiçbir C-test açılmaz; çıktı
           sonuclar/e1_h14_kuru_sonuc.{json,md}. Durum/kural (g) denetimleri bu kipte arşiv H10 C1 eğitim koşumlarına uygulanır.
  (varsayılan) gerçek analiz: --runs <E1 koşumlarının runs/ dizini> → sonuclar/e1_h14_sonuc.{json,md} + .sha256.
           Var olan sonuç dosyası yalnız --yeniden "<neden>" ile ezilir (neden JSON'a yazılır).
Kullanım (WS kökü; deney venv'i):
  PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I scripts/e1_h14_analiz.py --kuru
  PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I scripts/e1_h14_analiz.py --runs "<DRIVE_KOK>/3-1/runs"
Dokunulmayanlar: k5_c1_sonuc.*, k5_test_sonuc.*, karar.md, k5_test.KILIT (yalnız OKUNUR).
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "scripts"))
import k5_test as K  # noqa: E402  (bootstrap + metrikler; donmuş betik)

KART = WS / "hipotezler" / "H14.md"
C1_SONUC = WS / "sonuclar" / "k5_c1_sonuc.json"
DONDURMA = WS / "sonuclar" / "k5_dondurma.json"
TOL_PUAN = 0.005
MEL, NV, BKL = 4, 5, 2
SEEDLER = [0, 1, 2, 3, 4]
EK = "(test sonrası keşifsel, düzeltmesiz)"
# kart eşikleri
GORULMEMIS = {"K1_delta_puan": 4.2, "H14_ctest_acc": 0.9457, "H14_cval_acc": 0.96, "K1_dengeli_delta_puan": 16.0}
H10C1_EP5_LOSS = 0.0345
H10C1_SECILEN_EPOCH_ORT = 7.0


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dur(msg: str):
    print(f"DUR: {msg}", flush=True)
    raise SystemExit(2)


def tr(x: float, n: int = 2, isaret: bool = True) -> str:
    s = f"{x:+.{n}f}" if isaret else f"{x:.{n}f}"
    return s.replace("-", "−").replace(".", ",")


def ga_tr(g) -> str:
    return f"[{tr(g[0])}; {tr(g[1])}]"


# ───────────────────────── yükleme ─────────────────────────
def kosum_dizinleri(runs: Path, varyant: str, kuyruk: str, asama: int = 3) -> dict:
    """seed → koşum dizini (K.kosumlar: rc=0, aşama, kuyruk öneki; seed başına birden çok → HATA)."""
    return {int(json.loads((d / "config.json").read_text(encoding="utf-8"))["seed"]): d
            for d in K.kosumlar(runs, varyant, kuyruk, asama)}


def kol_yukle(runs: Path, varyant: str, kuyruk: str, seedler=None):
    dz = kosum_dizinleri(runs, varyant, kuyruk)
    if seedler is not None:
        eksik = [s for s in seedler if s not in dz]
        if eksik:
            dur(f"{varyant} ({kuyruk}): seed {eksik} yok")
        dz = {s: dz[s] for s in seedler}
    P, y0 = [], None
    for s in sorted(dz):
        y = np.load(dz[s] / "etiket_test.npy").reshape(-1).astype(np.int64)
        if y0 is not None and not np.array_equal(y, y0):
            dur(f"{varyant}: C-test etiketleri seed'ler arasında farklı")
        y0 = y
        P.append(np.load(dz[s] / "tahmin_test.npy").astype(np.float64))
    P = np.stack(P) if P else None
    if P is not None and (not np.isfinite(P).all() or not np.allclose(P.sum(2), 1.0, atol=1e-3)):
        dur(f"{varyant}: olasılıklar sonlu/normalize değil")
    return P, y0, {s: dz[s].name for s in sorted(dz)}, dz


def metrics(d: Path) -> dict:
    return json.loads((d / "metrics.json").read_text(encoding="utf-8"))


def tek_kosum(runs: Path, varyant: str, kuyruk: str, asama: int, seed: int = 0):
    """Aşama-1 kontrol/duman koşumları: rc ∈ {0, 5} (5 = Durum C çıkışı), en yeni."""
    bul = []
    for d in sorted(Path(runs).glob("*/")):
        cf, dn = d / "config.json", d / ".done"
        if not (cf.exists() and dn.exists() and (d / "metrics.json").exists()):
            continue
        c = json.loads(cf.read_text(encoding="utf-8"))
        if c.get("varyant") == varyant and str(c.get("seed")) == str(seed) and int(c.get("asama", 3)) == asama \
                and str(c.get("kuyruk", "")).startswith(kuyruk):
            bul.append(d)
    return bul[-1] if bul else None


# ───────────────────────── istatistik (k5_test işlevleri + mel / iki yönlü ek) ─────────────────────────
def mel_kiyas(PA, PB, y, BS):
    dA, dB = (PA.argmax(2) == y[None]).mean(0), (PB.argmax(2) == y[None]).mean(0)
    m = y == MEL
    w = BS.W[:, m].sum(1)
    ok = w > 0
    rA = (BS.W[:, m] @ dA[m]) / np.maximum(w, 1)
    rB = (BS.W[:, m] @ dB[m]) / np.maximum(w, 1)
    fb = (rB - rA)[ok]
    return {"A": float(dA[m].mean()), "B": float(dB[m].mean()), "delta": float(dB[m].mean() - dA[m].mean()),
            "ga": K.ga(fb), "p_ust": float((np.sum(fb <= 0) + 1) / (len(fb) + 1)),
            "p_alt": float((np.sum(fb >= 0) + 1) / (len(fb) + 1)), "n_mel": int(m.sum()), "n_gecerli_ornek": int(ok.sum())}


def etiket(ga, b_ad="H14") -> str:
    return (f"{b_ad} üstün" if ga[0] > 0 else (f"{b_ad} geride" if ga[1] < 0 else "ayırt edilemedi")) + f" {EK}"


def kiyasla(PA, PB, y, BS, b_ad="H14") -> dict:
    r = K.kiyas(PA, PB, y, BS)
    dA, dB = (PA.argmax(2) == y[None]).mean(0), (PB.argmax(2) == y[None]).mean(0)
    bs = BS.ort(dB - dA)
    out = {"n_seed": int(len(PA))}
    for ad in ("acc", "dengeli_acc", "dengeli_acc_5sinif"):
        e = r[ad]
        out[ad] = {"A": e["A"], "B": e["B"], "delta_puan": 100 * e["delta"], "ga_puan": [100 * v for v in e["ga"]],
                   "etiket": etiket(e["ga"], b_ad)}
    out["acc"]["p_ust"] = float((np.sum(bs <= 0) + 1) / (K.NB + 1))   # P(Δ* ≤ 0): B > A yönü
    out["acc"]["p_alt"] = float((np.sum(bs >= 0) + 1) / (K.NB + 1))   # tersi
    mk = mel_kiyas(PA, PB, y, BS)
    out["mel_duyarlilik"] = {"A": mk["A"], "B": mk["B"], "delta_puan": 100 * mk["delta"], "ga_puan": [100 * v for v in mk["ga"]],
                             "p_ust": mk["p_ust"], "p_alt": mk["p_alt"], "n_mel": mk["n_mel"], "etiket": etiket(mk["ga"], b_ad)}
    aA, aB = (PA.argmax(2) == y[None]).mean(1), (PB.argmax(2) == y[None]).mean(1)
    xA = np.array([K.dengeli(y, PA[s].argmax(1)) for s in range(len(PA))])
    xB = np.array([K.dengeli(y, PB[s].argmax(1)) for s in range(len(PB))])
    for ad, xa, xb in (("t_acc", aA, aB), ("t_dengeli", xA, xB)):
        fd = xb - xa
        if len(fd) > 1 and (np.any(fd != fd[0]) or fd[0] != 0):
            p2 = float(stats.ttest_rel(xb, xa).pvalue)
        else:
            p2 = 1.0
        out[ad] = {"fark_seed_puan": (100 * fd).tolist(), "ort_puan": float(100 * fd.mean()), "p_iki_yonlu_bilgi": p2}
    out["farkli_dogru_goruntu"] = r["farkli_dogru_goruntu"]
    return out


def ikinciller(P, y) -> dict:
    o = K.kol_ozet(P, y)
    t = P.argmax(2)
    hucre = lambda g, tah: float(np.mean([int(((y == g) & (t[s] == tah)).sum()) for s in range(len(P))]))  # noqa: E731
    cift = [float((t[a] != t[b]).mean()) for a in range(len(P)) for b in range(a + 1, len(P))]
    return {"ort": o["ort"], "std": o["std"], "acc_eksi_cogunluk": o["ort"]["acc"] - K.CTEST_COGUNLUK,
            "mel_to_nv": hucre(MEL, NV), "nv_to_mel": hucre(NV, MEL), "bkl_to_nv": hucre(BKL, NV),
            "seed_cifti_argmax_anlasmazlik_ort": float(np.mean(cift)) if cift else None,
            "guven_eksi_acc": float(np.mean([P[s].max(1).mean() - (t[s] == y).mean() for s in range(len(P))]))}


# ───────────────────────── 1) kuru koşu özdeşliği ─────────────────────────
def kuru_ozdeslik(arsiv: Path, C1: dict):
    idx_csv = K.c_alt_kume("test", "test")
    sira = np.argsort(idx_csv)
    P, y = {}, None
    kos = {}
    for k in ("h02", "h08", "h10"):
        Pk, yk, ad, _ = kol_yukle(arsiv, f"{k}_c1-test", "k5-c1", SEEDLER)
        if Pk is None:
            dur(f"arşiv {k}_c1-test bulunamadı ({arsiv})")
        if len(yk) != len(idx_csv):
            dur(f"{k}: {len(yk)} ≠ {len(idx_csv)} C-test görüntüsü")
        if y is not None and not np.array_equal(y, yk):
            dur(f"{k}: C-test etiketleri arşiv kollarında farklı")
        y = yk
        P[k], kos[k] = Pk, ad
    PC = {k: v[:, sira] for k, v in P.items()}
    yc = y[sira]
    BS = K.Bootstrap(len(yc))
    sat = {}
    ok = True
    for A, B in (("h08", "h10"), ("h02", "h08"), ("h02", "h10")):
        r = K.kiyas(PC[A], PC[B], yc, BS)
        ref = C1["i"][f"{B}-{A}"]
        d = 100 * r["acc"]["delta"]
        g = [100 * v for v in r["acc"]["ga"]]
        fark = max(abs(d - ref["delta_puan"]), abs(g[0] - ref["ga_puan"][0]), abs(g[1] - ref["ga_puan"][1]))
        sat[f"{B}-{A}"] = {"delta_puan": d, "ga_puan": g, "k5_c1_delta_puan": ref["delta_puan"], "k5_c1_ga_puan": ref["ga_puan"],
                           "maks_fark_puan": fark, "gecti": bool(fark <= TOL_PUAN)}
        ok &= fark <= TOL_PUAN
    for k in ("h02", "h08", "h10"):
        a = float((PC[k].argmax(2) == yc[None]).mean(1).mean())
        sat[f"kol_{k}_acc"] = {"bu": a, "k5_c1": C1["kol"][k]["acc_ort"], "fark": abs(a - C1["kol"][k]["acc_ort"])}
        ok &= abs(a - C1["kol"][k]["acc_ort"]) <= 1e-12
    return {"gecti": bool(ok), "tolerans_puan": TOL_PUAN, "satirlar": sat, "kosumlar": kos}, PC, yc, sira, BS


# ───────────────────────── 3) Durum ve kural (g) ─────────────────────────
def durum_sinifla(m: dict) -> str:
    d, n_ok, p = m["kontrol_ilk_adim_loss_fark"], m["kontrol_argmax_oran"], m["kontrol_p_maks_fark"]
    if d <= 1e-6 and n_ok >= 1.0 and p <= 1e-5:
        return "A"
    if d <= 1e-3 and n_ok * 573 >= 570 - 1e-9 and m.get("kontrol_surum_farkli"):
        return "B"
    return "C"


def egitim_denetim(runs: Path, varyant: str, kuyruk: str, aug: bool) -> dict:
    dz = kosum_dizinleri(runs, varyant, kuyruk, 3)
    out = {"kosumlar": {s: d.name for s, d in dz.items()}, "seed": {}}
    for s, d in sorted(dz.items()):
        m = metrics(d)
        g = json.loads((d / "gecmis.json").read_text(encoding="utf-8"))
        va = [r["val_acc"] for r in g]
        out["seed"][s] = {"val_acc": m["val_acc"], "en_iyi_epoch": m["en_iyi_epoch"], "train_loss_ep5": g[4]["train_loss"] if len(g) >= 5 else None,
                          "son3_artiyor": bool(len(va) >= 3 and va[-3] < va[-2] < va[-1]),
                          "epoch_sure_ort_sn": m.get("epoch_sure_ort_sn"), "val_ece": m.get("val_ece"),
                          "augment_frekans_maks_sapma": m.get("augment_frekans_maks_sapma"),
                          "onbellek_gorunum_tutarlilik_maks_fark": m.get("onbellek_gorunum_tutarlilik_maks_fark"),
                          "onbellek_gorunum_ozdes_cift": m.get("onbellek_gorunum_ozdes_cift")}
    S = out["seed"]
    if S:
        out["cval_acc_ort"] = float(np.mean([v["val_acc"] for v in S.values()]))
        out["secilen_epoch_ort"] = float(np.mean([v["en_iyi_epoch"] for v in S.values()]))
        out["epoch_sure_ort_sn"] = float(np.mean([v["epoch_sure_ort_sn"] for v in S.values() if v["epoch_sure_ort_sn"] is not None]))
        out["kural_e_az_egitim"] = bool(sum(1 for v in S.values() if v["en_iyi_epoch"] == 15 and v["son3_artiyor"]) >= 3)
        if aug:
            ihl = [s for s, v in S.items() if (v["augment_frekans_maks_sapma"] is None or v["augment_frekans_maks_sapma"] > 0.01
                                               or v["onbellek_gorunum_tutarlilik_maks_fark"] is None
                                               or v["onbellek_gorunum_tutarlilik_maks_fark"] > 5e-3
                                               or v["onbellek_gorunum_ozdes_cift"] != 0)]
            out["kural_g_ihlal_seed"] = ihl
    return out


def butce_l4sa(runs: Path, onek: str = "e1-") -> float:
    t = 0.0
    for d in Path(runs).glob("*/"):
        cf, dn = d / "config.json", d / ".done"
        if cf.exists() and dn.exists():
            c = json.loads(cf.read_text(encoding="utf-8"))
            if str(c.get("kuyruk", "")).startswith(onek):
                s = json.loads(dn.read_text(encoding="utf-8")).get("sure_sn")
                t += float(s or 0)
    return t / 3600


def veri_c_denetim(runs: Path) -> list:
    """Kural (i): h14/h10r öneki taşıyan her k5_cikarim koşumunda veri_c = 1; C-test seed başına en çok bir kez."""
    hata, sayac = [], {}
    for d in Path(runs).glob("*/"):
        cf = d / "config.json"
        if not cf.exists() or not (d / "metrics.json").exists():
            continue
        c = json.loads(cf.read_text(encoding="utf-8"))
        v = str(c.get("varyant", ""))
        if not v.startswith(("h14_", "h10r_")):
            continue
        m = metrics(d)
        if "veri_c" in m and m["veri_c"] != 1:
            hata.append(f"{d.name}: veri_c={m['veri_c']}")
        if (d / "tahmin_test.npy").exists():
            sayac[(v, str(c.get("seed")))] = sayac.get((v, str(c.get("seed"))), 0) + 1
    hata += [f"{v} s{s}: C-test {n} kez açılmış" for (v, s), n in sayac.items() if n > 1]
    return hata


# ───────────────────────── 5) cümleler ─────────────────────────
def cumleler(K1, K2, durum_, kiyas_h10_arsiv, surukleme, gorulmemis_acik, n_seed):
    if gorulmemis_acik:
        return {"not": "Görülmemiş (İNCELE) bayrağı açık → soruşturma kapanana dek sayı/cümle YAZILMAZ (kart)."}
    k2 = K2["acc"]
    yari = (k2["ga_puan"][1] - k2["ga_puan"][0]) / 2
    e2 = k2["etiket"].split(" (")[0]
    if e2 == "H14 üstün":
        c2 = ("Test sonrası keşifsel bir analizde DINOv2-B'ye ResNet-18 ile aynı eğitim-zamanı dihedral augmentasyonu "
              f"eklendiğinde (H14), lezyon-ayrık C1 bölmesinde D4'lü ResNet-18'i (H08c) {tr(k2['delta_puan'], isaret=False)} puan "
              f"{ga_tr(k2['ga_puan'])} geçti. Ana metindeki 'H08c ile H10 ayırt edilemez' bulgusu bu nedenle kısmen augmentasyon "
              "asimetrisinden kaynaklanıyor olabilir. Analiz test sonuçları görüldükten sonra tasarlandı, düzeltmesizdir ve K5 "
              "kararını ya da ön-kayıtlı iddiaları değiştirmez. Doğrulanması bağımsız bir lezyon-ayrık test kümesi gerektirir.")
    elif e2 == "H14 geride":
        c2 = ("Simetrik augmentasyonla bile kısmi ince ayarlı DINOv2-B (H14), lezyon-ayrık C1 bölmesinde D4'lü ResNet-18'in "
              f"{tr(abs(k2['delta_puan']), isaret=False)} puan {ga_tr(k2['ga_puan'])} gerisinde kaldı. Bu keşifsel bulgu, kısmi ince ayarlı "
              "DINOv2-B'nin lezyon-ayrık bölmede ResNet-18'e üstünlüğü olmadığı yönündeki ana sonuçla uyumludur. Analiz test "
              "sonrası ve düzeltmesizdir.")
    else:
        c2 = ("Augmentasyon simetrik hâle getirildiğinde de (H14 = H10 + D4) DINOv2-B ile D4'lü ResNet-18 lezyon-ayrık C1 "
              f"bölmesinde ayırt edilemedi (Δ {tr(k2['delta_puan'])} puan {ga_tr(k2['ga_puan'])}). Bu, asimetrik kıyasın ana bulguyu "
              f"değiştirmediği yönünde keşifsel bir kanıttır. GA genişliği yaklaşık ±{tr(yari, isaret=False)} puanlık farkları dışlamaz. "
              "Analiz test sonrası ve düzeltmesizdir.")
    k1 = K1["acc"]
    e1 = k1["etiket"].split(" (")[0]
    if e1 == "H14 üstün":
        c1 = f"D4 augmentasyonu DINOv2-B'ye C1'de {tr(k1['delta_puan'], isaret=False)} puan {ga_tr(k1['ga_puan'])} kazandırdı"
    elif e1 == "H14 geride":
        c1 = f"D4 augmentasyonu DINOv2-B'ye C1'de {tr(abs(k1['delta_puan']), isaret=False)} puan {ga_tr(k1['ga_puan'])} zarar verdi"
    else:
        c1 = f"D4 augmentasyonu DINOv2-B'de C1'de ayırt edilebilir bir değişiklik yaratmadı (Δ {tr(k1['delta_puan'])} {ga_tr(k1['ga_puan'])})"
    c1 += ". Karşılaştırma için aynı augmentasyonun ResNet-18'e kazancı +2,09 [+1,24; +2,97] idi."
    dm = []
    for ad, kk in (("K1 (H14 − H10" + ("r" if durum_ == "B" else "") + ")", K1), ("K2 (H14 − H08c)", K2)):
        d_, m_ = kk["dengeli_acc"], kk["mel_duyarlilik"]
        dm.append(f"{ad}: dengeli doğruluk Δ {tr(d_['delta_puan'])} puan {ga_tr(d_['ga_puan'])} ({d_['etiket'].split(' (')[0]}); "
                  f"mel duyarlılığı Δ {tr(m_['delta_puan'])} puan {ga_tr(m_['ga_puan'])} ({m_['etiket'].split(' (')[0]}; "
                  f"n_mel = {m_['n_mel']})")
    out = {"K2_etiketi": e2, "K2_cumle": c2, "K1_etiketi": e1, "K1_eslik_cumle": c1,
           "dengeli_mel_cumle": "; ".join(dm) + f" {EK}.", "n_seed": n_seed}
    if durum_ == "B" and surukleme is not None:
        out["durum_B_eki"] = ("H14 ve karşılaştırma kolu H10r aynı yazılım ortamında koşuldu. Ekim koşumlarıyla ortam farkının "
                              f"etkisi H10r − H10 = {tr(surukleme['acc']['delta_puan'])} puan {ga_tr(surukleme['acc']['ga_puan'])} olarak ölçüldü.")
        out["K2_durum_B_notu"] = ("kıyas Ekim/Kasım ortamları arasındadır; ortam sürüklenmesi H10r − H10 = "
                                  f"{tr(surukleme['acc']['delta_puan'])} puan ile ölçüldü")
    out["sonuctan_bagimsiz_oneri"] = ("Özet ve sonuç bölümündeki H08c ↔ H10 cümlesine '(ResNet-18'e D4 eklendi, DINOv2-B'ye "
                                      "eklenmedi; simetrik kıyas için bkz. Bölüm [E1])' kaydı (yazım kararı makale-yazıcı/kullanıcıda).")
    return out


# ───────────────────────── rapor ─────────────────────────
def rapor_md(R: dict) -> str:
    L = [f"# E1 / H14 — test sonrası keşifsel analiz{' — KURU KOŞU (H14 yerine arşiv H10)' if R['kuru'] else ''}",
         f"*{R['tarih']}* · betik `{R['betik_sha256'][:16]}…` · kart `{R['kart_sha256'][:16]}…` · k5_test `{R['k5_test_sha256'][:16]}…`",
         "", "**Etiketlerin hepsi:** " + EK + ". K5 Holm ailelerine girmez; K5 kararı değişmez.", ""]
    if R["kuru"]:
        L += ["> **KURU KOŞU:** H14 kolu = arşiv `h10_c1-test` (aynı tahminler). K1 = 0 beklenir; K2 = H10 − H08 (C1 (i) satırı). "
              "Yeni C-test açılmadı; yalnız K5 C1'de açılmış arşiv tahminleri okundu.", ""]
    oz = R["kuru_ozdeslik"]
    L += [f"## 1) Kuru koşu özdeşliği (k5_c1_sonuc.json yeniden üretimi) → **{'GEÇTİ' if oz['gecti'] else 'DUR'}** "
          f"(tolerans {oz['tolerans_puan']} puan)", "| kıyas | Δ (bu) | GA (bu) | Δ (k5_c1) | GA (k5_c1) | maks fark |", "|---|---|---|---|---|---|"]
    for k, v in oz["satirlar"].items():
        if k.startswith("kol_"):
            continue
        L.append(f"| {k} | {v['delta_puan']:+.4f} | [{v['ga_puan'][0]:+.4f}; {v['ga_puan'][1]:+.4f}] | {v['k5_c1_delta_puan']:+.4f} | "
                 f"[{v['k5_c1_ga_puan'][0]:+.4f}; {v['k5_c1_ga_puan'][1]:+.4f}] | {v['maks_fark_puan']:.2e} |")
    L += ["", f"## 2) Kural (f) Durum: **{R['durum']['durum']}** · K1 karşılaştırıcısı `{R['durum']['k1_karsilastirici']}`",
          f"- {json.dumps(R['durum'].get('ayrinti'), ensure_ascii=False)}", "", "## 3) Kollar (C-test, seed ortalaması)",
          "| kol | n | ACC | dengeli | dengeli-5 | mel duy. | makro-F1 | ECE-15 | NLL | nv payı | mel→nv | ACC − 0,7726 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, o in R["kollar"].items():
        e = o["ort"]
        L.append(f"| {k} | {R['n_seed']} | {e['acc']:.4f} ± {o['std']['acc']:.4f} | {e['dengeli_acc']:.4f} | {e['dengeli_acc_5sinif']:.4f} | "
                 f"{e.get('duyarlilik_mel', float('nan')):.4f} | {e['makro_f1']:.4f} | {e['ece15']:.4f} | {e['nll']:.4f} | {e['nv_payi']:.4f} | "
                 f"{o['mel_to_nv']:.1f} | {o['acc_eksi_cogunluk']:+.4f} |")
    L += ["", "## 4) Kıyaslar (B − A; görüntü bootstrap'ı B = 2.000, rng 0; düzeltmesiz)",
          "| kıyas | metrik | Δ (puan) | %95 GA | p(Δ*≤0) / p(Δ*≥0) | etiket |", "|---|---|---|---|---|---|"]
    for ad, kk in R["kiyaslar"].items():
        for mt in ("acc", "dengeli_acc", "dengeli_acc_5sinif", "mel_duyarlilik"):
            e = kk[mt]
            pp = f"{e['p_ust']:.3g} / {e['p_alt']:.3g}" if "p_ust" in e else "—"
            L.append(f"| {ad} | {mt} | {e['delta_puan']:+.2f} | [{e['ga_puan'][0]:+.2f}; {e['ga_puan'][1]:+.2f}] | {pp} | {e['etiket']} |")
        L.append(f"| {ad} | seed t (ACC, iki yönlü, bilgi) | {kk['t_acc']['ort_puan']:+.2f} | — | p = {kk['t_acc']['p_iki_yonlu_bilgi']:.3g} | — |")
    L += ["", "## 5) Görülmemiş (İNCELE), manipülasyon ve kural notları", f"- Görülmemiş: {json.dumps(R['gorulmemis'], ensure_ascii=False)}",
          f"- Manipülasyon/kural notları: {json.dumps(R['notlar'], ensure_ascii=False)}", "", "## 6) Önceden yazılmış cümle (mekanik seçim)"]
    for k, v in R["cumleler"].items():
        L.append(f"- **{k}:** {v}")
    return "\n".join(L) + "\n"


# ───────────────────────── ana ─────────────────────────
def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kuru", action="store_true", help="H14 yerine arşiv h10_c1 (Δ = 0 beklenir); yeni C-test açılmaz")
    ap.add_argument("--runs", default="runs_colab", help="E1 koşumlarının runs/ dizini (Drive runs ya da yerel kopya)")
    ap.add_argument("--arsiv", default="runs_colab", help="K5 C1 arşiv koşumları (h02/h08/h10_c1, h10_c1-test …)")
    ap.add_argument("--yeniden", default=None, help="var olan e1_h14_sonuc.json'u ezmenin NEDENİ")
    ap.add_argument("--cikti", default=None, help="yalnız kod sınaması: çıktı JSON yolu (WS/sonuclar dışı)")
    a = ap.parse_args()
    runs = Path(a.runs) if Path(a.runs).is_absolute() else WS / a.runs
    arsiv = Path(a.arsiv) if Path(a.arsiv).is_absolute() else WS / a.arsiv
    out = Path(a.cikti) if a.cikti else WS / "sonuclar" / ("e1_h14_kuru_sonuc.json" if a.kuru else "e1_h14_sonuc.json")
    if out.exists() and not a.kuru and not a.yeniden:
        dur(f"{out.name} zaten var — yalnız --yeniden '<neden>' ile (neden JSON'a yazılır)")

    # 0) donmuş k5_test
    k5s = sha(WS / "scripts" / "k5_test.py")
    C1S = json.loads(C1_SONUC.read_text(encoding="utf-8"))
    don = json.loads(DONDURMA.read_text(encoding="utf-8"))
    if k5s != C1S.get("betik_sha256") or k5s != don.get("k5_test_sha256"):
        dur(f"scripts/k5_test.py SHA256 {k5s[:12]} donmuş değerle aynı değil (bootstrap işlevleri değişmiş olabilir)")
    R = {"kuru": a.kuru, "tarih": datetime.now().isoformat(timespec="seconds"), "betik_sha256": sha(__file__),
         "kart_sha256": sha(KART), "k5_test_sha256": k5s, "runs": str(runs), "arsiv": str(arsiv),
         "yontem": "k5_test.Bootstrap (B=2000, rng 0, aynı W iki kola) + k5_test.kiyas; mel duyarlılığı aynı W ile; "
                   "seed t iki yönlü (bilgi); düzeltme yok", "yeniden_neden": a.yeniden}

    # 1) kuru koşu özdeşliği (H14 C-test okunmadan ÖNCE)
    oz, PC, yc, sira, BS = kuru_ozdeslik(arsiv, C1S["C1"])
    R["kuru_ozdeslik"] = oz
    print(f"[e1] kuru koşu özdeşliği: {'GEÇTİ' if oz['gecti'] else 'DUR'} · "
          + ", ".join(f"{k} Δ {v['delta_puan']:+.2f} [{v['ga_puan'][0]:+.2f}; {v['ga_puan'][1]:+.2f}] (fark {v['maks_fark_puan']:.1e})"
                      for k, v in oz["satirlar"].items() if not k.startswith("kol_")), flush=True)
    if not oz["gecti"]:
        dur("kuru koşu özdeşliği geçmedi (|fark| > 0,005 puan) — analiz yapılmaz")

    notlar, gorulmemis = {}, {}
    # 2) kollar
    if a.kuru:
        seedler = SEEDLER
        P = {"H14": PC["h10"].copy(), "H10": PC["h10"], "H08c": PC["h08"], "H02c": PC["h02"]}
        R["kosumlar"] = {"H14 (kuru = arşiv h10_c1-test)": oz["kosumlar"]["h10"], **{k: v for k, v in oz["kosumlar"].items()}}
        egit = egitim_denetim(arsiv, "h10_c1", "k5-c1", aug=False)
        R["durum"] = {"durum": "KURU (atlandı)", "k1_karsilastirici": "h10_c1 (arşiv)",
                      "ayrinti": "kuru koşu: kontrol kolu yok; Durum A/B/C gerçek analizde kontrol_* metriklerinden"}
        durum_ = "A"
    else:
        hata = veri_c_denetim(runs)
        if hata:
            dur(f"kural (i): {hata[:5]}")
        mk = tek_kosum(runs, "h14_c1_d4kapali_duman", "e1-h14", 1)
        if mk is None:
            dur("kontrol kolu h14_c1_d4kapali_duman (e1-h14 a1) bulunamadı")
        m_k = metrics(mk)
        durum_ = durum_sinifla(m_k)
        ayr = {k: m_k.get(k) for k in ("kontrol_ilk_adim_loss_fark", "kontrol_argmax_oran", "kontrol_p_maks_fark",
                                         "kontrol_val_acc_fark", "kontrol_surum_fark", "kontrol_durum")}
        ayr["kontrol_run"] = mk.name
        if durum_ != m_k.get("kontrol_durum"):
            ayr["uyari"] = f"betik Durum {durum_} ≠ koşumdaki {m_k.get('kontrol_durum')}"
        if durum_ == "C":
            dur(f"kural (f) Durum C (kod yolu hatası şüphesi): {ayr}")
        md = tek_kosum(runs, "h14_c1_duman", "e1-h14", 1)
        if md is None:
            dur("D4 dumanı h14_c1_duman bulunamadı")
        d_ilk = abs(metrics(md)["ilk_adim_loss"] - m_k["ilk_adim_loss"])
        ayr["d4_duman_ilk_adim_loss_fark"] = d_ilk
        if d_ilk <= 1e-6:
            dur("kural (g): D4 dumanının ilk_adim_loss'u kontrolle aynı (D4 devrede değil)")
        seed_h14 = sorted(kosum_dizinleri(runs, "h14_c1-test", "e1-h14"))
        if not seed_h14:
            dur("h14_c1-test koşumu yok")
        seedler = [s for s in SEEDLER if s in seed_h14]
        if len(seedler) < 5:
            notlar["kural_a_eksik_seed"] = f"n = {len(seedler)}, eksik (seed {seedler})"
        P14, y14, ad14, _ = kol_yukle(runs, "h14_c1-test", "e1-h14", seedler)
        y14 = y14[sira]
        if not np.array_equal(y14, yc):
            dur("H14 C-test etiketleri arşiv kollarıyla aynı değil (sıra/eşleme)")
        si = [SEEDLER.index(s) for s in seedler]
        P = {"H14": P14[:, sira], "H10": PC["h10"][si], "H08c": PC["h08"][si], "H02c": PC["h02"][si]}
        R["kosumlar"] = {"H14": ad14, **oz["kosumlar"]}
        k1_ref = "H10"
        if durum_ == "B":
            mr = tek_kosum(runs, "h10r_c1_duman", "e1-h10r", 1)
            if mr is None:
                dur("Durum B: H10r zorunlu, h10r_c1_duman yok")
            m_r = metrics(mr)
            ayr["h10r_duman"] = {"run": mr.name, "kontrol_ilk_adim_loss_fark": m_r.get("kontrol_ilk_adim_loss_fark"),
                                 "kontrol_p_maks_fark": m_r.get("kontrol_p_maks_fark")}
            if not (m_r.get("kontrol_ilk_adim_loss_fark", 1) <= 1e-6 and m_r.get("kontrol_p_maks_fark", 1) <= 1e-5):
                dur(f"Durum B ek koşulu: h10r_c1_duman ↔ kontrol kolu bit-bit değil ({ayr['h10r_duman']})")
            P10r, y10r, ad10r, _ = kol_yukle(runs, "h10r_c1-test", "e1-h10r", seedler)
            if P10r is None or not np.array_equal(y10r[sira], yc):
                dur("Durum B: h10r_c1-test eksik ya da etiketleri hizasız")
            P["H10r"] = P10r[:, sira]
            R["kosumlar"]["H10r"] = ad10r
            k1_ref = "H10r"
        R["durum"] = {"durum": durum_, "k1_karsilastirici": k1_ref, "ayrinti": ayr}
        egit = egitim_denetim(runs, "h14_c1", "e1-h14", aug=True)
        if egit.get("kural_g_ihlal_seed"):
            dur(f"kural (g): augment/görünüm denetimi ihlali seed {egit['kural_g_ihlal_seed']}")
        e1sa = butce_l4sa(runs)
        notlar["butce_E1_L4sa"] = round(e1sa, 3)
        if e1sa > 5.0:
            notlar["kural_h"] = f"E1 birleşik L4-sa {e1sa:.2f} > 5,0 (DUR kuralı; kullanıcıya rapor)"
    R["n_seed"] = len(seedler)
    R["egitim"] = egit

    # 4) kıyaslar
    BSx = BS
    k1_ref = "H10r" if durum_ == "B" and "H10r" in P else "H10"
    KIY = {f"K1: H14 − {k1_ref}": kiyasla(P[k1_ref], P["H14"], yc, BSx),
           "K2: H14 − H08c": kiyasla(P["H08c"], P["H14"], yc, BSx),
           "betimsel: H14 − H02c": kiyasla(P["H02c"], P["H14"], yc, BSx)}
    surukleme = None
    if k1_ref == "H10r":
        KIY["ikincil: H14 − H10 (arşiv)"] = kiyasla(P["H10"], P["H14"], yc, BSx)
        surukleme = kiyasla(P["H10"], P["H10r"], yc, BSx, b_ad="H10r")
        KIY["sürüklenme: H10r − H10 (arşiv)"] = surukleme
    R["kiyaslar"] = KIY
    R["kollar"] = {k: ikinciller(v, yc) for k, v in P.items()}
    K1, K2 = KIY[f"K1: H14 − {k1_ref}"], KIY["K2: H14 − H08c"]

    # 5) görülmemiş + notlar
    gorulmemis = {"(i) K1 Δ > +4,2": K1["acc"]["delta_puan"] > GORULMEMIS["K1_delta_puan"],
                  "(ii) H14 C-test ACC > 0,9457": R["kollar"]["H14"]["ort"]["acc"] > GORULMEMIS["H14_ctest_acc"],
                  "(iii) H14 C-val ACC > 0,96": bool(egit.get("cval_acc_ort", 0) > GORULMEMIS["H14_cval_acc"]),
                  "(iv) dengeli Δ(K1) > +16,0": K1["dengeli_acc"]["delta_puan"] > GORULMEMIS["K1_dengeli_delta_puan"]}
    R["gorulmemis"] = {**gorulmemis, "INCELE": any(gorulmemis.values())}
    s0 = (egit.get("seed") or {}).get(0) or {}
    l5 = s0.get("train_loss_ep5")
    notlar.update({
        "manipulasyon_s0_ep5_loss": l5, "manipulasyon_esik_>0,069": (l5 > 2 * H10C1_EP5_LOSS) if l5 is not None else None,
        "kural_j_notu": (l5 is not None and l5 <= 1.2 * H10C1_EP5_LOSS),
        "secilen_epoch_ort": egit.get("secilen_epoch_ort"), "secilen_epoch_>7,0": (egit.get("secilen_epoch_ort") or 0) > H10C1_SECILEN_EPOCH_ORT,
        "cval_acc_ort": egit.get("cval_acc_ort"), "epoch_sure_ort_sn": egit.get("epoch_sure_ort_sn"),
        "kural_e_az_egitim": egit.get("kural_e_az_egitim"),
        "ece_H14_<=_K1ref": R["kollar"]["H14"]["ort"]["ece15"] <= R["kollar"][k1_ref]["ort"]["ece15"],
        "nv_yanliligi_notu": bool(K1["acc"]["delta_puan"] > 0 and K1["dengeli_acc"]["delta_puan"] <= 0
                                  and (R["kollar"]["H14"]["ort"]["nv_payi"] - R["kollar"][k1_ref]["ort"]["nv_payi"]) > 0.02),
    })
    if notlar["kural_j_notu"]:
        notlar["kural_j"] = "D4 devrede ama ezberi geciktirmedi"
    if notlar["kural_e_az_egitim"]:
        notlar["kural_e"] = "15 epoch D4 için kısa olabilir (epoch uzatma onay kapsamı DIŞINDA)"
    R["notlar"] = notlar
    R["cumleler"] = cumleler(K1, K2, durum_, KIY.get("ikincil: H14 − H10 (arşiv)"), surukleme, R["gorulmemis"]["INCELE"], len(seedler))
    if a.kuru:
        R["kuru_beklenen"] = {"K1_delta_puan_sifir": abs(K1["acc"]["delta_puan"]) < 1e-12,
                              "K1_ga_sifir": max(abs(v) for v in K1["acc"]["ga_puan"]) < 1e-12,
                              "K2_eşittir_H10-H08": abs(K2["acc"]["delta_puan"] - C1S["C1"]["i"]["h10-h08"]["delta_puan"]) < 1e-9}

    temiz = json.loads(json.dumps(R, ensure_ascii=False, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else
                                  (int(o) if isinstance(o, (np.integer,)) else (bool(o) if isinstance(o, np.bool_) else float(o)))))
    out.write_text(json.dumps(temiz, ensure_ascii=False, indent=1), encoding="utf-8")
    md = rapor_md(temiz)
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    out.with_name(out.name + ".sha256").write_text(f"{sha(out)}  {out.name}\n", encoding="utf-8")
    print(md)
    print(f"yazıldı: {out} (+ .md, .sha256)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
