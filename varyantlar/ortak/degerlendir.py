# -*- coding: utf-8 -*-
"""
ortak/degerlendir.py — H01/H02/H03 ortak değerlendirme (yalnız VAL; test K5).

Tanımlar:
- ACC = argmax doğruluğu, AUC = makro OvR (sınıf başına roc_auc_score(y==c, p[:,c]) ortalaması):
  `medmnist.evaluator.getACC/getAUC` (3.0.2, task "multi-class") ile BİREBİR — doğrudan onlar çağrılır.
  Alt kümede bir sınıf hiç yoksa getAUC hata verir → yalnız mevcut sınıfların ortalaması alınır ve
  `<on>_auc_eksik_sinif` > 0 yazılır (yalnız duman/alt-küme; tam val'de 7/7 sınıf var).
- dengeli_acc = mevcut sınıfların duyarlılık (recall) ortalaması (rakip köprüsüyle aynı tanım).
- makro_f1 = sklearn f1_score(average="macro", labels=0..6, zero_division=0).
- ece = 15 eşit genişlikte kutu, en yüksek sınıf olasılığı (softmax) güveni (Guo vd. 2017).
- nll = ortalama -log p(y) (bilgi).
- <on>_auc_<kisa> = sınıf başına OvR AUC (akiec, bcc, bkl, df, mel, nv, vasc; ör. val_auc_mel, val_auc_bkl —
  kart "mel/bkl OvR AUC", hata_kirilim.py auc_sinif ile aynı tanım).
- alt_grup = sınıf başına val duyarlılığı {"sinif_0".."sinif_6"} (yoksa NaN değil, anahtar atlanır).
- K3.5 ekleri (sonuclar/hata-analizi.md; tanımlar scripts/hata_kirilim.py ile AYNI, sayısal eşleşme
  baseline 5 koşumunda doğrulandı — varyantlar/KOMUTLAR.md §6):
  <on>_tahmin_nv_payi = argmax tahminlerinin nv (sınıf 5) payı.
  <on>_acc_molemax / <on>_acc_molemax_disi (+ <on>_dengeli_acc_molemax_disi): HAM10000 `dataset` == "vidir_molemax".
  <on>_sizinti_tabakali_SN = tabaka (sınıf × dx_type × kaynak) içi S−N doğruluk farklarının, tabakanın N sayısıyla
  ağırlıklı ortalaması (yalnız hem S hem N içeren tabakalar); <on>_sizinti_tabakali_kapsam = bu tabakaların kapsadığı
  N payı. RAPORLANAN SIZINTI ÖLÇÜSÜ BUDUR.
- Ham sızıntı ölçüleri (`*_ham` soneki; S = lezyonu train'de olan, N = olmayan): <on>_acc_sizan_ham,
  <on>_acc_sizmayan_ham, <on>_sizinti_farki_ham, dengeli_acc karşılıkları, <on>_acc_sizan_Nagirlikli_ham (sınıf
  bileşimi düzeltmeli). Kaynak/dx_type karıştırıcısı nedeniyle (Simpson paradoksu; baseline ham −18,8, tabakalı
  +2,8 puan) bunlar RAPORLANMAZ; yalnız denetim/izlenebilirlik için yazılır.
- bootstrap: val ACC %95 yüzdelik GA, 2.000 yeniden örnekleme, sabit RNG 0 (kart: H01 Seed'in rolü).
"""
import math

import numpy as np

SINIF_SAYISI = 7
SINIF_KISA = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]  # ortak.veri.SINIF_KISA ile aynı sıra
NV = 5
MOLEMAX = "vidir_molemax"


def softmax(logit: np.ndarray) -> np.ndarray:
    z = logit - logit.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def _acc(y, p):
    from medmnist.evaluator import getACC
    return float(getACC(y, p, "multi-class"))


def _auc(y, p):
    """(auc, eksik_sinif_sayisi)."""
    from medmnist.evaluator import getAUC
    from sklearn.metrics import roc_auc_score
    mevcut = [c for c in range(p.shape[1]) if 0 < (y == c).sum() < len(y)]
    if len(mevcut) == p.shape[1]:
        return float(getAUC(y, p, "multi-class")), 0
    if not mevcut:
        return float("nan"), p.shape[1]
    return float(np.mean([roc_auc_score((y == c).astype(int), p[:, c]) for c in mevcut])), p.shape[1] - len(mevcut)


def sinif_duyarlilik(y, tah, n_sinif=SINIF_SAYISI) -> dict:
    return {c: float((tah[y == c] == c).mean()) for c in range(n_sinif) if (y == c).any()}


def dengeli_acc(y, tah) -> float:
    d = sinif_duyarlilik(y, tah)
    return float(np.mean(list(d.values()))) if d else float("nan")


def ece(y, p, n_kutu: int = 15) -> float:
    guven = p.max(axis=1)
    dogru = (p.argmax(axis=1) == y).astype(np.float64)
    sinirlar = np.linspace(0.0, 1.0, n_kutu + 1)
    toplam = 0.0
    for i in range(n_kutu):
        lo, hi = sinirlar[i], sinirlar[i + 1]
        m = (guven > lo) & (guven <= hi) if i > 0 else (guven >= lo) & (guven <= hi)
        if m.any():
            toplam += m.mean() * abs(dogru[m].mean() - guven[m].mean())
    return float(toplam)


def bootstrap_acc_ga(y, p, n: int = 2000, rng_seed: int = 0, alfa: float = 0.05):
    rng = np.random.default_rng(rng_seed)
    dogru = (p.argmax(axis=1) == y).astype(np.float64)
    N = len(y)
    ornek = rng.integers(0, N, size=(n, N))
    acc = dogru[ornek].mean(axis=1)
    return float(np.quantile(acc, alfa / 2)), float(np.quantile(acc, 1 - alfa / 2))


def tabakali_fark(dogru: np.ndarray, S: np.ndarray, tabaka: np.ndarray):
    """hata_kirilim.py `tabakali` ile aynı: Σ_k n_N,k·(acc_S,k − acc_N,k) / Σ_k n_N,k (yalnız S ve N içeren
    tabakalar). Dönüş: (fark, kapsam = kullanılan tabakaların N sayısı / toplam N) — tabaka yoksa (nan, 0)."""
    dogru = np.asarray(dogru, dtype=float)
    top = w = 0.0
    for k in set(tabaka.tolist()):
        m = tabaka == k
        if (m & S).any() and (m & ~S).any():
            wk = (m & ~S).sum()
            top += wk * (dogru[m & S].mean() - dogru[m & ~S].mean())
            w += wk
    nN = (~S).sum()
    return (top / w if w else float("nan")), (w / nN if nN else 0.0)


def degerlendir(y: np.ndarray, p: np.ndarray, on: str = "val", val_idx: np.ndarray | None = None,
                bootstrap: bool = True) -> dict:
    """y: (N,) int, p: (N,K) softmax olasılığı. val_idx: resmî val indeksleri (alt-küme dahil) → sızıntı/kaynak
    alt-grup metrikleri (ortak.veri.val_meta). Dönüş: düz sözlük + "alt_grup"."""
    if on == "test":
        raise RuntimeError("degerlendir: 'test' öneki K4'te yasak (M4); K5 test betiği ayrı çağırır")
    y = np.asarray(y).reshape(-1).astype(np.int64)
    p = np.asarray(p, dtype=np.float64)
    tah = p.argmax(axis=1)
    from sklearn.metrics import f1_score
    auc, eksik = _auc(y, p)
    m = {
        f"{on}_acc": _acc(y, p),
        f"{on}_auc": auc,
        f"{on}_dengeli_acc": dengeli_acc(y, tah),
        f"{on}_makro_f1": float(f1_score(y, tah, average="macro", labels=list(range(p.shape[1])), zero_division=0)),
        f"{on}_ece": ece(y, p),
        f"{on}_nll": float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)))),
        f"{on}_n": int(len(y)),
        f"{on}_tahmin_nv_payi": float((tah == NV).mean()),
    }
    if eksik:
        m[f"{on}_auc_eksik_sinif"] = int(eksik)
    # sınıf başına OvR AUC (K3.5 Z1: mel/bkl; kart adı "val_auc_sinif_{4,2}" / "mel ve bkl OvR AUC" →
    # kod adı <on>_auc_<kisa>, ör. val_auc_mel, val_auc_bkl). Tanım hata_kirilim.py auc_sinif ile aynı.
    from sklearn.metrics import roc_auc_score
    for c in range(p.shape[1]):
        if 0 < (y == c).sum() < len(y):
            m[f"{on}_auc_{SINIF_KISA[c]}"] = float(roc_auc_score((y == c).astype(int), p[:, c]))
    # bağımsız çapraz kontrol: Evaluator ACC = argmax doğruluğu
    if abs(m[f"{on}_acc"] - float((tah == y).mean())) > 1e-9:
        raise RuntimeError("ACC tanım uyuşmazlığı (getACC ≠ argmax doğruluğu)")
    if bootstrap:
        lo, hi = bootstrap_acc_ga(y, p)
        m[f"{on}_acc_ga_alt"], m[f"{on}_acc_ga_ust"] = lo, hi
    if val_idx is not None:
        from ortak.veri import val_meta
        vm = val_meta()
        idx = np.asarray(val_idx, dtype=np.int64)
        S, dxt, kay = vm["S"][idx], vm["dx_type"][idx], vm["kaynak"][idx]
        dogru = (tah == y)
        mm = kay == MOLEMAX
        if mm.any():
            m[f"{on}_acc_molemax"] = float(dogru[mm].mean())
            m[f"{on}_n_molemax"] = int(mm.sum())
        if (~mm).any():
            m[f"{on}_acc_molemax_disi"] = float(dogru[~mm].mean())
            m[f"{on}_dengeli_acc_molemax_disi"] = dengeli_acc(y[~mm], tah[~mm])
        N_ = ~S
        if S.any() and N_.any():
            tab = np.array([f"{y[i]}|{dxt[i]}|{kay[i]}" for i in range(len(y))])
            fark, kapsam = tabakali_fark(dogru, S, tab)
            if np.isfinite(fark):
                m[f"{on}_sizinti_tabakali_SN"] = float(fark)
            m[f"{on}_sizinti_tabakali_kapsam"] = float(kapsam)
            # ham ölçüler (RAPORLANMAZ; Simpson karıştırıcısı)
            m[f"{on}_acc_sizan_ham"] = float(dogru[S].mean())
            m[f"{on}_acc_sizmayan_ham"] = float(dogru[N_].mean())
            m[f"{on}_sizinti_farki_ham"] = m[f"{on}_acc_sizan_ham"] - m[f"{on}_acc_sizmayan_ham"]
            m[f"{on}_dengeli_acc_sizan_ham"] = dengeli_acc(y[S], tah[S])
            m[f"{on}_dengeli_acc_sizmayan_ham"] = dengeli_acc(y[N_], tah[N_])
            m[f"{on}_dengeli_sizinti_farki_ham"] = m[f"{on}_dengeli_acc_sizan_ham"] - m[f"{on}_dengeli_acc_sizmayan_ham"]
            dS = sinif_duyarlilik(y[S], tah[S])
            pN = {c: float((y[N_] == c).mean()) for c in dS}
            w = sum(pN.values())
            if w > 0:
                m[f"{on}_acc_sizan_Nagirlikli_ham"] = float(sum(pN[c] * dS[c] for c in dS) / w)
                m[f"{on}_sizinti_farki_Nagirlikli_ham"] = m[f"{on}_acc_sizan_Nagirlikli_ham"] - m[f"{on}_acc_sizmayan_ham"]
            m[f"{on}_n_sizan"] = int(S.sum())
            m[f"{on}_n_sizmayan"] = int(N_.sum())
    m["alt_grup"] = {f"sinif_{c}": v for c, v in sinif_duyarlilik(y, tah, p.shape[1]).items()}
    return m


def sonlu_mu(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
