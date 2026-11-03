#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
logit_ayari.py — H04 / H06 (Tur-2): ebeveynin kayıtlı val olasılıklarına post-hoc logit (önsel) ayarı.
Yeniden eğitim / çıkarım YOK; GPU YOK (torch import edilmez).

Karar kuralı (Menon vd. 2021): argmax_c [log p(c|x) − τ·log π_train(c)], tek τ ∈ ızgara (5 seed ortak).
τ seçim protokolü (H04 kartı §"τ seçim protokolü", ön-kayıt; H06 aynısı):
 1. Kat bölmesi `katlar.json` (bu dizinde) — `--katlar-yaz` ile, KOŞUMDAN ÖNCE, yalnız etiket + lesion_id'den:
    StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261011), tabaka = sınıf, grup = lesion_id
    (eşleme: data/splits/dermamnist_lezyon_esleme.csv — scripts/hata_kirilim.py ile AYNI kaynak).
    Eşleme okunamazsa StratifiedKFold (aynı RNG) ve sapma katlar.json + metrics.json'a yazılır.
 2. k katı için seçim kümesi = kalan 4 kat. Her τ'da 5 seed'in ORTALAMA dengeli ACC'si ve ORTALAMA ACC'si.
    Uygun küme: ACC̄(τ) ≥ ACC̄(0) − kısıt (0,010). τ_k = uygun kümede dengeli ACC'yi en büyük yapan τ;
    eşitlikte (1e-9) küçük τ. τ = 0 her zaman uygundur.
 3. OOF: τ_k yalnız k katındaki görüntülere uygulanır (her seed'de aynı τ_k) → seed başına 1.003'lük OOF karar
    vektörü; metrikler bundan. Kaydedilen tahmin_val.npy = softmax(log p − τ_{k(i)}·log π).
 4. Tam-val τ*: aynı kural tüm val'e → val_*_tamval (iyimser, ikincil; K5'te test'e uygulanacak tek τ).
 6. π_train: yalnız train etiket frekansları (dermamnist.npz train_labels); sınıf sırası medmnist INFO ile denetlenir.
Denetimler: τ = 0 özdeşliği (OOF metrikleri = ebeveynin metrics.json val_acc / val_dengeli_acc; fark ≠ 0 → rc=3,
DUR), π sınıf sırası (uyuşmazlık → rc=3), ebeveyn etiketleri = npz val etiketleri.
Kararlılık (ikincil 12): RNG 1–20 ile 20 alternatif bölmede (aynı yöntem) bu seed'in OOF dengeli ACC'si (ort/std).

Kullanım (WS kökü):
  # bir kez, koşumlardan önce:
  env/.venv/Scripts/python.exe varyantlar/logit_ayari/logit_ayari.py --katlar-yaz
  # koşum (kosum.py altında):
  env/.venv/Scripts/python.exe varyantlar/logit_ayari/logit_ayari.py --ebeveyn-varyant h02_r18in_ft224 \
     --ebeveyn-dosya tahmin_val.npy --tau-izgara 0:1:0.1 --kat 5 --kat-rng 20261011 --kisit 0.010 \
     --seed 0 --out {RUN_DIR}/metrics.json
  Duman (a1): --tau-izgara 0,1 --alt-bolme 0
Çıktılar ({RUN_DIR}): metrics.json (düz; val_*; TEST YOK) · tahmin_val.npy (OOF) · tahmin_val_tamval.npy ·
  etiket_val.npy · tau_secim.json (kat başına seçim tabloları + tam-val τ eğrisi; τ = 1 referansı dahil) ·
  varyant_bilgi.json / config.json eki.
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

from ortak.degerlendir import degerlendir  # noqa: E402
from ortak.kayit import atomik_json, config_ekle, metrics_yaz  # noqa: E402

KATLAR = Path(__file__).resolve().parent / "katlar.json"
LEZYON_CSV = WS / "data" / "splits" / "dermamnist_lezyon_esleme.csv"
NPZ28 = WS / "data" / "medmnist" / "dermamnist.npz"
SINIF_KISA = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
# medmnist INFO['dermamnist']['label'] metinlerinde aranacak anahtar sözcükler (sınıf sırası denetimi, kart (d))
SINIF_ANAHTAR = ["actinic", "basal", "benign keratosis", "dermatofibroma", "melanoma", "melanocytic", "vascular"]
PI_KART = [0.0325, 0.0512, 0.1097, 0.0114, 0.1112, 0.6698, 0.0141]  # H04 kartı §6 (4 hane)
K = 7
ESIT_TOL = 1e-9   # kart: eşitlikte (1e-9) küçük τ
KISIT_TOL = 1e-12  # kayan nokta: ACC̄(τ) ≥ ACC̄(0) − kısıt karşılaştırması için


# ───────────────────────── yardımcılar ─────────────────────────
def sha256(yol: Path) -> str:
    return hashlib.sha256(Path(yol).read_bytes()).hexdigest()


def val_etiket() -> np.ndarray:
    with np.load(NPZ28) as z:
        return z["val_labels"].reshape(-1).astype(np.int64)


def train_pi() -> np.ndarray:
    with np.load(NPZ28) as z:
        ytr = z["train_labels"].reshape(-1).astype(np.int64)
    return np.bincount(ytr, minlength=K) / len(ytr)


def sinif_sirasi_dogru() -> bool:
    from medmnist import INFO
    lab = INFO["dermamnist"]["label"]
    return all(SINIF_ANAHTAR[c] in lab[str(c)].lower() for c in range(K))


def lezyon_gruplari(n: int):
    """Resmî val indeks sırasıyla lesion_id dizisi; okunamazsa None."""
    try:
        g = {}
        with open(LEZYON_CSV, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["split"] == "val":
                    g[int(r["index"])] = r["lesion_id"]
        if len(g) != n or any(i not in g or not g[i] for i in range(n)):
            return None
        return np.array([g[i] for i in range(n)])
    except Exception:  # noqa: BLE001
        return None


def kat_bol(y: np.ndarray, gruplar, n_kat: int, rng: int):
    """(kat_id dizisi (N,), yöntem adı). Yalnız etiket ve lesion_id'ye bakar (tahminlere bakmaz)."""
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    kat = np.full(len(y), -1, dtype=np.int64)
    if gruplar is not None:
        bol = StratifiedGroupKFold(n_splits=n_kat, shuffle=True, random_state=rng).split(np.zeros(len(y)), y, gruplar)
        yontem = "StratifiedGroupKFold(lesion_id)"
    else:
        bol = StratifiedKFold(n_splits=n_kat, shuffle=True, random_state=rng).split(np.zeros(len(y)), y)
        yontem = "StratifiedKFold(SAPMA: lesion_id eşlemesi okunamadı)"
    for k, (_, te) in enumerate(bol):
        kat[te] = k
    assert (kat >= 0).all()
    return kat, yontem


def katlar_yaz(n_kat: int, rng: int) -> int:
    y = val_etiket()
    g = lezyon_gruplari(len(y))
    kat, yontem = kat_bol(y, g, n_kat, rng)
    if g is not None:  # grup bütünlüğü denetimi: aynı lezyon tek katta
        for lez in set(g.tolist()):
            assert len(set(kat[g == lez].tolist())) == 1, lez
    veri = {"aciklama": "H04/H06/R-T ortak val kat bölmesi (ön-kayıt; koşumdan önce yazıldı). Yalnız etiket ve lesion_id'ye bakar.",
            "yontem": yontem, "n_splits": n_kat, "shuffle": True, "random_state": rng,
            "sapma": None if g is not None else "lesion_id eşlemesi okunamadı → StratifiedKFold",
            "grup_kaynak": str(LEZYON_CSV.relative_to(WS)).replace("\\", "/"), "etiket_kaynak": "data/medmnist/dermamnist.npz val_labels",
            "n": int(len(y)), "n_lezyon": int(len(set(g.tolist()))) if g is not None else None,
            "kat_boyut": [int((kat == k).sum()) for k in range(n_kat)],
            "kat_sinif_sayilari": [[int(((kat == k) & (y == c)).sum()) for c in range(K)] for k in range(n_kat)],
            "kat": kat.tolist()}
    if KATLAR.exists():
        eski = json.loads(KATLAR.read_text(encoding="utf-8"))
        if eski.get("kat") != veri["kat"]:
            print(f"HATA: {KATLAR} zaten var ve FARKLI — üzerine yazılmaz (ön-kayıt).")
            return 3
        print(f"[logit_ayari] {KATLAR.name} zaten var ve aynı; dokunulmadı. SHA256 {sha256(KATLAR)}")
        return 0
    KATLAR.write_text(json.dumps(veri, ensure_ascii=False), encoding="utf-8")
    print(f"[logit_ayari] yazıldı {KATLAR} · {yontem} · kat boyutları {veri['kat_boyut']} · SHA256 {sha256(KATLAR)}")
    for k in range(n_kat):
        print(f"  kat {k + 1}: sınıf sayıları {veri['kat_sinif_sayilari'][k]}")
    return 0


def izgara(s: str) -> np.ndarray:
    if ":" in s:
        a, b, h = (float(x) for x in s.split(":"))
        t = np.round(np.arange(a, b + h / 2, h), 10)
    else:
        t = np.array(sorted(float(x) for x in s.split(",")))
    return t


def ebeveyn_kosumlari(varyant: str, dosya: str):
    """runs/ altında varyant adı eşleşen, damgalı rc=0, aşama 3 koşumlar; seed başına tam bir koşum."""
    bul = {}
    for d in sorted((WS / "runs").glob("*/")):
        cfg, done = d / "config.json", d / ".done"
        if not (cfg.exists() and done.exists() and (d / dosya).exists()):
            continue
        c = json.loads(cfg.read_text(encoding="utf-8"))
        j = json.loads(done.read_text(encoding="utf-8"))
        if c.get("varyant") != varyant or j.get("rc") != 0 or int(c.get("asama", 3)) != 3:
            continue
        s = int(c["seed"])
        if s in bul:
            raise SystemExit(f"HATA: {varyant} seed {s} için birden çok damgalı koşum: {bul[s].name}, {d.name}")
        bul[s] = d
    return dict(sorted(bul.items()))


def dengeli(y, t):
    v = [(t[y == c] == c).mean() for c in range(K) if (y == c).any()]
    return float(np.mean(v))


def tau_sec(L, logpi, y, idx, taus, kisit):
    """L: (S,N,K) log p. idx: seçim kümesi indeksleri. Dönüş (τ, tablo)."""
    yy = y[idx]
    acc, den = [], []
    for t in taus:
        tah = (L[:, idx, :] - t * logpi[None, None, :]).argmax(2)  # (S, n)
        acc.append(float(np.mean([(tah[s] == yy).mean() for s in range(len(L))])))
        den.append(float(np.mean([dengeli(yy, tah[s]) for s in range(len(L))])))
    acc, den = np.array(acc), np.array(den)
    i0 = int(np.where(np.isclose(taus, 0.0))[0][0])
    uygun = acc >= acc[i0] - kisit - KISIT_TOL
    uygun[i0] = True
    eniyi = den[uygun].max()
    sec = min(i for i in range(len(taus)) if uygun[i] and den[i] >= eniyi - ESIT_TOL)
    tablo = [{"tau": float(taus[i]), "acc_ort": float(acc[i]), "dengeli_ort": float(den[i]), "uygun": bool(uygun[i])}
             for i in range(len(taus))]
    return float(taus[sec]), tablo


def oof_tau(L, logpi, y, kat, taus, kisit):
    n_kat = int(kat.max()) + 1
    tau_k, tablolar = [], []
    for k in range(n_kat):
        t, tab = tau_sec(L, logpi, y, np.where(kat != k)[0], taus, kisit)
        tau_k.append(t)
        tablolar.append(tab)
    return np.array(tau_k), tablolar


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def karisiklik_hucreleri(y, tah, on, ek=""):
    ad = {"mel": 4, "nv": 5, "bkl": 2}
    out = {}
    for a, b in [("mel", "nv"), ("bkl", "nv"), ("nv", "mel"), ("nv", "bkl")]:
        out[f"{on}_karisiklik_{a}_{b}{ek}"] = int(((y == ad[a]) & (tah == ad[b])).sum())
    return out


# ───────────────────────── ana ─────────────────────────
def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--katlar-yaz", action="store_true", help="katlar.json'ı yaz ve çık (koşumdan önce, bir kez)")
    p.add_argument("--ebeveyn-varyant")
    p.add_argument("--ebeveyn-dosya", default="tahmin_val.npy")
    p.add_argument("--tau-izgara", default="0:1:0.1")
    p.add_argument("--kat", type=int, default=5)
    p.add_argument("--kat-rng", type=int, default=20261011)
    p.add_argument("--kisit", type=float, default=0.010)
    p.add_argument("--alt-bolme", type=int, default=20, help="kararlılık: RNG 1..n alternatif bölme (0 = kapalı)")
    p.add_argument("--seed", type=int)
    p.add_argument("--out")
    a = p.parse_args()
    if a.katlar_yaz:
        return katlar_yaz(a.kat, a.kat_rng)
    if a.ebeveyn_varyant is None or a.seed is None or a.out is None:
        p.error("--ebeveyn-varyant, --seed, --out gerekli")
    t0 = time.time()
    out = Path(a.out).resolve()
    run_dir = out.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── denetimler: π sırası, katlar ──
    if not sinif_sirasi_dogru():
        print("DUR: medmnist INFO etiket sırası akiec,bcc,bkl,df,mel,nv,vasc ile uyuşmuyor (kart (d))")
        return 3
    pi = train_pi()
    if np.max(np.abs(np.round(pi, 4) - np.array(PI_KART))) > 1e-4 + 1e-12:
        print(f"DUR: π_train {np.round(pi, 4).tolist()} kart değeriyle uyuşmuyor {PI_KART}")
        return 3
    logpi = np.log(pi)
    if not KATLAR.exists():
        print(f"HATA: {KATLAR} yok — önce --katlar-yaz (ön-kayıt; koşum içinde yazılmaz)")
        return 3
    KJ = json.loads(KATLAR.read_text(encoding="utf-8"))
    kat = np.array(KJ["kat"], dtype=np.int64)
    if KJ["n_splits"] != a.kat or KJ["random_state"] != a.kat_rng:
        print(f"HATA: katlar.json ({KJ['n_splits']}, {KJ['random_state']}) ≠ argümanlar ({a.kat}, {a.kat_rng})")
        return 3
    y = val_etiket()
    if len(kat) != len(y):
        print("HATA: katlar.json uzunluğu val ile uyuşmuyor")
        return 3
    taus = izgara(a.tau_izgara)
    if not np.isclose(taus, 0.0).any():
        print("HATA: ızgara τ = 0 içermeli")
        return 3

    # ── ebeveyn olasılıkları (5 seed; ortak τ seçimi için hepsi okunur) ──
    kos = ebeveyn_kosumlari(a.ebeveyn_varyant, a.ebeveyn_dosya)
    if a.seed not in kos:
        print(f"HATA: {a.ebeveyn_varyant} için seed {a.seed} koşumu yok (bulunan {list(kos)})")
        return 3
    seeds = list(kos)
    P, Mebe = [], []
    for s, d in kos.items():
        ye = np.load(d / "etiket_val.npy").reshape(-1).astype(np.int64)
        if not np.array_equal(ye, y):
            print(f"HATA: {d.name} etiket_val ≠ npz val etiketleri")
            return 3
        P.append(np.load(d / a.ebeveyn_dosya).astype(np.float64))
        Mebe.append(json.loads((d / "metrics.json").read_text(encoding="utf-8")))
    P = np.stack(P)                         # (S, N, K)
    L = np.log(np.maximum(P, 1e-300))       # float32 denormalleri ayırt edilir; argmax(L) = argmax(P)
    si = seeds.index(a.seed)
    print(f"[logit_ayari] ebeveyn {a.ebeveyn_varyant}/{a.ebeveyn_dosya} seed'ler {seeds} · ızgara {taus.tolist()} · "
          f"kat {KJ['yontem']} · kısıt {a.kisit}")

    # ── 1-3. OOF τ seçimi ve uygulama ──
    tau_k, tablolar = oof_tau(L, logpi, y, kat, taus, a.kisit)
    tau_i = tau_k[kat]                       # görüntü başına τ
    Z_oof = L[si] - tau_i[:, None] * logpi[None, :]
    p_oof = softmax(Z_oof)
    # ── 4. tam-val τ* ──
    tau_t, tablo_t = tau_sec(L, logpi, y, np.arange(len(y)), taus, a.kisit)
    p_tam = softmax(L[si] - tau_t * logpi[None, :])
    # ── τ = 0 özdeşlik denetimi (kart (d)) ──
    p_0 = softmax(L[si])
    m0 = degerlendir(y, p_0, on="val", bootstrap=False)
    me = Mebe[si]
    # ebeveyn dosyası tahmin_val<ek>.npy → ebeveyn metrics anahtarları val_acc<ek>, val_dengeli_acc<ek> (ör. _aucsecim)
    ek = Path(a.ebeveyn_dosya).stem.replace("tahmin_val", "", 1)
    ka, kd = f"val_acc{ek}", f"val_dengeli_acc{ek}"
    if ka not in me or kd not in me:
        print(f"DUR: ebeveyn metrics.json'da {ka}/{kd} yok — τ = 0 özdeşliği sınanamaz")
        return 3
    ozdeslik = max(abs(m0["val_acc"] - float(me[ka])), abs(m0["val_dengeli_acc"] - float(me[kd])))
    if not np.array_equal(p_0.argmax(1), P[si].argmax(1)):
        ozdeslik = max(ozdeslik, 1.0)
    print(f"[logit_ayari] τ_k = {tau_k.tolist()} · tam-val τ* = {tau_t} · τ=0 özdeşlik farkı {ozdeslik}")

    # ── metrikler ──
    vidx = np.arange(len(y))
    m = degerlendir(y, p_oof, on="val", val_idx=vidx)
    alt_grup = m.pop("alt_grup")
    mt = degerlendir(y, p_tam, on="val", val_idx=vidx, bootstrap=False)
    tamval = {f"{k}_tamval": mt[k] for k in ("val_acc", "val_dengeli_acc", "val_makro_f1", "val_auc", "val_ece", "val_nll",
                                              "val_tahmin_nv_payi", "val_acc_molemax_disi") if k in mt}
    tamval.update({f"val_duyarlilik_{SINIF_KISA[int(k.split('_')[1])]}_tamval": v for k, v in mt["alt_grup"].items()})
    m0full = degerlendir(y, p_0, on="val", val_idx=vidx, bootstrap=False)
    ebe = {f"{k}_ebeveyn": m0full[k] for k in ("val_acc", "val_dengeli_acc", "val_makro_f1", "val_auc", "val_ece",
                                                "val_tahmin_nv_payi", "val_acc_molemax_disi") if k in m0full}
    # τ = 1 sabit (ablasyon (i) referansı; seçimsiz Menon noktası) — karar vermez
    p_1 = softmax(L[si] - 1.0 * logpi[None, :])
    t1 = p_1.argmax(1)
    tau1 = {"val_acc_tau1": float((t1 == y).mean()), "val_dengeli_acc_tau1": dengeli(y, t1)}
    tah_oof, tah0 = p_oof.argmax(1), P[si].argmax(1)
    # ── 12. kararlılık: 20 alternatif bölme ──
    alt = {}
    if a.alt_bolme > 0:
        g = lezyon_gruplari(len(y)) if KJ.get("sapma") is None else None
        dlist, tlist = [], []
        for r in range(1, a.alt_bolme + 1):
            kat_r, _ = kat_bol(y, g, a.kat, r)
            tk, _ = oof_tau(L, logpi, y, kat_r, taus, a.kisit)
            tah_r = (L[si] - tk[kat_r][:, None] * logpi[None, :]).argmax(1)
            dlist.append(dengeli(y, tah_r))
            tlist.append(float(np.median(tk)))
        alt = {"val_dengeli_acc_alt_bolme_ort": float(np.mean(dlist)), "val_dengeli_acc_alt_bolme_std": float(np.std(dlist, ddof=1)),
               "val_dengeli_acc_alt_bolme_min": float(np.min(dlist)), "val_dengeli_acc_alt_bolme_max": float(np.max(dlist)),
               "alt_bolme_tau_medyan_ort": float(np.mean(tlist)), "alt_bolme_n": int(a.alt_bolme)}
    sure = round(time.time() - t0, 2)
    n_param_ebe = me.get("param_sayisi")
    metrics = {
        **m,
        "val_dengeli_acc_delta": m["val_dengeli_acc"] - ebe["val_dengeli_acc_ebeveyn"],
        "val_acc_delta": m["val_acc"] - ebe["val_acc_ebeveyn"],
        **{f"tau_kat_{k + 1}": float(t) for k, t in enumerate(tau_k)},
        "tau_kat_medyan": float(np.median(tau_k)), "tau_kat_aralik": float(tau_k.max() - tau_k.min()),
        "tau_kat_sifir_sayisi": int((tau_k == 0).sum()), "tau_tamval": tau_t,
        **tamval,
        "val_iyimserlik_dengeli": tamval["val_dengeli_acc_tamval"] - m["val_dengeli_acc"],
        **ebe, **tau1,
        **karisiklik_hucreleri(y, tah_oof, "val"), **karisiklik_hucreleri(y, tah0, "val", "_ebeveyn"),
        "val_karar_degisen_goruntu": int((tah_oof != tah0).sum()),
        **{f"val_duyarlilik_{SINIF_KISA[int(k.split('_')[1])]}": v for k, v in alt_grup.items()},
        **alt,
        "tau0_ozdeslik_fark": float(ozdeslik), "pi_sira_ok": 1,
        "katlar_sha256": sha256(KATLAR), "kat_yontemi": KJ["yontem"], "kat_sapma": 0 if KJ.get("sapma") is None else 1,
        "kisit": a.kisit, "tau_izgara_n": int(len(taus)), "n_seed_secim": int(len(seeds)),
        "ebeveyn_varyant": a.ebeveyn_varyant, "ebeveyn_dosya": a.ebeveyn_dosya, "ebeveyn_kosum": kos[a.seed].name,
        "egitim_sure_sn": sure, "sure_sn": sure,
        "param_sayisi": int(n_param_ebe) if n_param_ebe is not None else -1, "param_ek_skaler": 1,
        "seed": a.seed, "cihaz": "cpu (numpy; GPU kullanılmadı)", "alt_grup": alt_grup,
    }
    np.save(run_dir / "tahmin_val.npy", p_oof.astype(np.float32))
    np.save(run_dir / "tahmin_val_tamval.npy", p_tam.astype(np.float32))
    np.save(run_dir / "etiket_val.npy", y)
    egri = []
    for t in taus:
        tt = (L[si] - t * logpi[None, :]).argmax(1)
        egri.append({"tau": float(t), "acc": float((tt == y).mean()), "dengeli": dengeli(y, tt), "nv_payi": float((tt == 5).mean())})
    atomik_json(run_dir / "tau_secim.json", {
        "tau_kat": tau_k.tolist(), "tau_tamval": tau_t, "kat_tablolari": tablolar, "tamval_tablo": tablo_t,
        "bu_seed_tamval_egrisi": egri, "pi_train": pi.tolist(), "ebeveyn_kosumlar": {str(s): d.name for s, d in kos.items()}})
    config_ekle(run_dir, {"logit_ayari": {
        "kural": "argmax_c [log p − τ·log π_train]", "kaynak": "Menon vd. 2021 post-hoc logit adjustment",
        "ebeveyn_varyant": a.ebeveyn_varyant, "ebeveyn_dosya": a.ebeveyn_dosya,
        "ebeveyn_kosumlar": {str(s): d.name for s, d in kos.items()}, "tau_izgara": taus.tolist(), "kisit": a.kisit,
        "kat": KJ["yontem"], "kat_rng": a.kat_rng, "katlar_sha256": sha256(KATLAR), "pi_train": pi.tolist(),
        "esitlik_tol": ESIT_TOL, "kisit_tol": KISIT_TOL, "alt_bolme": a.alt_bolme}})
    metrics_yaz(out, metrics)
    if ozdeslik != 0.0:
        print(f"DUR: τ = 0 özdeşlik farkı {ozdeslik} ≠ 0 (kart (d)) — metrics yazıldı, rc=3")
        return 3
    print(f"[logit_ayari] BİTTİ {sure}s · OOF val_acc {m['val_acc']:.4f} (ebeveyn {ebe['val_acc_ebeveyn']:.4f}) · "
          f"OOF dengeli {m['val_dengeli_acc']:.4f} (ebeveyn {ebe['val_dengeli_acc_ebeveyn']:.4f}) · τ_k {tau_k.tolist()} τ* {tau_t}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
