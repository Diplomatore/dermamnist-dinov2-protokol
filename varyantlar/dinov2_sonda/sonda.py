#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sonda.py — Donuk DINOv2 CLS öznitelikleri üzerinde seed'li mini-batch softmax-LR sondası (H01; H03 --r).

Kart karşılığı (H01 "Seed'in sondadaki rolü"):
- Birincil sonda: nn.Linear(D,7); kayıp = CE (ortalama) + λ‖W‖², λ = 1/(C·N_train), sapma cezasız.
  Adam lr 1e-2 (varsayılan; K3.5 sonrası kart revizyonu — özgün kart 1e-3 sentetik N=7007/D=768 denemesinde
  optimuma ulaşmadı; --lr ile değiştirilebilir) (weight_decay=0), bs 256, 100 epoch, CosineAnnealingLR(T_max=epoch, epoch başına adım).
  Seed'in değiştirdiği: ağırlık başlatması N(0, 0.01²) (sapma 0) ve mini-batch sırası.
- StandardScaler YALNIZ train özniteliklerinden (ort/std, ddof=0; std=0 → 1).
- C ızgarası {1e-4,…,1e3} (8 nokta). TEK C* = argmax(seçim seed'leri üzerinde ORTALAMA val_acc);
  eşitlikte ortalama val_auc, o da eşitse küçük C. Seed başına ayrı C seçilmez.
  Uygulama: her koşum (seed s) ızgarayı --secim-seedleri'nin hepsi için kendi içinde uydurur (deterministik,
  önbellekli öznitelik üzerinde ucuz), C*'ı seçer, metrics.json'a KENDİ seed'inin C*'daki sonucunu yazar.
  Böylece 5 koşumun hepsi aynı C*'ı bulur (secilen_C anahtarı ile denetlenir) ve koşumlar bağımsız kalır.
- Denetim (kural e): aynı C* ve aynı ölçeklemeyle sklearn LogisticRegression(lbfgs, max_iter=5000).
  AMAÇ EŞDEĞERİ: sklearn amacı C_sk·ΣCE + ½‖W‖² ⇔ ortalama CE + ‖W‖²/(2·C_sk·N); kartın λ = 1/(C·N)
  cezasıyla aynı amaç için C_sk = C*/2 kullanılır (lbfgs_C_sklearn anahtarı). Fark = val_acc_lbfgs −
  (seçim seed'leri üzerinde ortalama SGD val_acc @C*) → `lbfgs_sgd_fark`; ayrıca train amaç değerleri.
- Öznitelik önbelleği yoksa (data/oznitelik/...) bu süreç ozn_cikar.cikar()'ı GPU'da kendisi çağırır
  (ilk kuyruk satırı; progress.json her 500 görüntüde). Test öznitelikleri ASLA okunmaz.

Çıktılar ({RUN_DIR} = --out klasörü): metrics.json (val_*; TEST ANAHTARI YOK) · progress.json ·
  tahmin_val.npy (kendi seed'i @C*, softmax float32) · etiket_val.npy · c_izgara.json (seed×C val acc/auc,
  süre) · sonda_agirlik.npz (W, b, scaler ort/std, C*; K5 test değerlendirmesi için) · config.json eki.

Kullanım (WS kökünden):
  env/.venv/Scripts/python.exe varyantlar/dinov2_sonda/sonda.py --seed 0 --r 224 --out {RUN_DIR}/metrics.json
      [--model b] [--C izgara | --C 1.0] [--secim-seedleri 0,1,2,3,4] [--alt-kume 256]
      [--cihaz cuda | --cihaz cpu --cpu-izin] [--lbfgs-yok]
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(WS / "data" / "hf_cache"))
os.environ.setdefault("TORCH_HOME", str(WS / "data" / "torch_cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from dinov2_ozn import ozn_cikar as OZ  # noqa: E402
from ortak import veri as V  # noqa: E402
from ortak.degerlendir import degerlendir, softmax  # noqa: E402
from ortak.kayit import (cihaz_sec, config_ekle, donanim_bilgisi, metrics_yaz, progress_yaz,  # noqa: E402
                         seed_bagla, tahmin_kaydet, atomik_json)

C_IZGARA = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]


def sgd_sonda(Xtr, ytr, Xva, C, seed, epoch, bs, lr, cihaz):
    """Kartın birincil sondası. Dönüş: (val_olasılık np (N,7), W np, b np, train_amac, son_loss)."""
    import torch
    import torch.nn.functional as F
    N, D = Xtr.shape
    K = V.SINIF_SAYISI
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    W = torch.empty(K, D).normal_(0.0, 0.01, generator=g)
    lin = torch.nn.Linear(D, K).to(cihaz)
    with torch.no_grad():
        lin.weight.copy_(W.to(cihaz))
        lin.bias.zero_()
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=0.0)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epoch)
    lam = 1.0 / (C * N)
    son = float("nan")
    for _ in range(epoch):
        perm = torch.randperm(N, generator=g).to(cihaz)
        for i in range(0, N, bs):
            j = perm[i:i + bs]
            loss = F.cross_entropy(lin(Xtr[j]), ytr[j]) + lam * lin.weight.pow(2).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sch.step()
        son = float(loss.detach())
    with torch.no_grad():
        amac = float(F.cross_entropy(lin(Xtr), ytr) + lam * lin.weight.pow(2).sum())
        p = torch.softmax(lin(Xva), dim=1).cpu().numpy().astype(np.float32)
    return p, lin.weight.detach().cpu().numpy(), lin.bias.detach().cpu().numpy(), amac, son


def train_amac(Xtr_np, ytr_np, W, b, C):
    """ortalama CE + λ‖W‖² (λ = 1/(C·N)) — numpy, lbfgs/SGD kıyası için."""
    z = Xtr_np @ W.T + b
    z = z - z.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=1))
    ce = float(np.mean(lse - z[np.arange(len(ytr_np)), ytr_np]))
    return ce + float((W.astype(np.float64) ** 2).sum()) / (C * len(ytr_np))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True, help="{RUN_DIR}/metrics.json")
    p.add_argument("--model", choices=sorted(OZ.MODELLER), default="b")
    p.add_argument("--r", type=int, choices=V.BOYUTLAR, default=224, help="kaynak çözünürlük (H03: 28/64/128)")
    p.add_argument("--C", default="izgara", help="'izgara' (8 nokta, kart) veya virgüllü değer(ler) (duman: 1.0)")
    p.add_argument("--secim-seedleri", default="0,1,2,3,4", help="C* için ortalaması alınan seed'ler (kart: 0..4)")
    p.add_argument("--epoch", type=int, default=100)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-2,
                   help="1e-2: K3.5 sonrası kart revizyonu (1e-3 sentetik denemede yakınsamadı); kart özgün değeri 1e-3")
    p.add_argument("--alt-kume", type=int, default=None, help="duman: her bölümün ilk N örneği (ayrı önbellek)")
    p.add_argument("--ozn-bs", type=int, default=32, help="önbellek yoksa çıkarım batch'i")
    p.add_argument("--cihaz", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--cpu-izin", action="store_true", help="yalnız hata ayıklama/mini doğrulama")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--lbfgs-yok", action="store_true", help="lbfgs denetimini atla (kart: lbfgs basamak başına BİR KEZ → tur1 a3'te yalnız s0 satırı koşar; "
                        "s0 satırı --secim-seedleri 0..4 ile 5-seed ortalamasına karşı yakinsama_fark yazar)")
    p.add_argument("--tekrar-yok", action="store_true", help="C*'da aynı seed tekrar uyumunu atla (determinizm denetimi)")
    a = p.parse_args()

    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    out = Path(a.out).resolve()
    run_dir = out.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    t_kosum = time.time()
    cihaz, cihaz_ad = cihaz_sec(a.cihaz, a.cpu_izin)
    seed_bagla(a.seed)

    # ── 1. öznitelik önbelleği (seed'den bağımsız) ─────────────────────────────────────
    yol = OZ.onbellek_yolu(a.model, a.r, a.alt_kume)
    onbellekten = yol.exists()
    if not onbellekten:
        print(f"[sonda] önbellek yok → çıkarım: {yol.name}", flush=True)
        OZ.cikar(a.model, a.r, ("train", "val"), a.ozn_bs, a.alt_kume, a.cihaz, a.cpu_izin,
                 progress_dir=run_dir)
    D, meta = OZ.onbellek_oku(yol)
    # önbellek ↔ girdi doğrulaması (yanlış/eskimiş önbelleği yakala)
    _, sha_simdi = V.dosya_hash(V.npz_yolu(a.r))
    if meta["veri"]["sha256"] != sha_simdi or meta["model"] != OZ.MODELLER[a.model][0] or meta["kaynak_px"] != a.r:
        raise SystemExit(f"HATA: önbellek {yol.name} meta uyuşmuyor (model/r/girdi SHA256) — --zorla ile yeniden çıkar")
    Xtr_np, ytr_np = D["train_oz"].astype(np.float32), D["train_etiket"].astype(np.int64)
    Xva_np, yva_np, va_idx = D["val_oz"].astype(np.float32), D["val_etiket"].astype(np.int64), D["val_idx"]

    # ── 2. ölçekleme: YALNIZ train istatistiği ─────────────────────────────────────────
    ort = Xtr_np.mean(axis=0, dtype=np.float64)
    std = Xtr_np.std(axis=0, dtype=np.float64)
    std[std == 0] = 1.0
    Xtr_s = ((Xtr_np - ort) / std).astype(np.float32)
    Xva_s = ((Xva_np - ort) / std).astype(np.float32)
    Xtr = torch.from_numpy(Xtr_s).to(cihaz)
    ytr = torch.from_numpy(ytr_np).to(cihaz)
    Xva = torch.from_numpy(Xva_s).to(cihaz)

    C_liste = C_IZGARA if a.C == "izgara" else [float(c) for c in a.C.split(",")]
    secim = [int(s) for s in a.secim_seedleri.split(",")]
    if a.seed not in secim:
        secim.append(a.seed)
    print(f"[sonda] {meta['model']} r={a.r} D={Xtr_np.shape[1]} · train {len(ytr_np)} val {len(yva_np)} · "
          f"C {C_liste} · seçim seed'leri {secim} · cihaz {cihaz_ad} · önbellekten={onbellekten}", flush=True)

    # ── 3. ızgara: seçim seed'leri × C ────────────────────────────────────────────────
    from ortak.degerlendir import _acc, _auc
    izgara, kendi, tum_p = [], {}, {}
    t_sonda = time.time()
    eniyi = -1.0
    adim = 0
    for s in secim:
        for C in C_liste:
            t0 = time.time()
            pv, W, b, amac, son = sgd_sonda(Xtr, ytr, Xva, C, s, a.epoch, a.bs, a.lr, cihaz)
            acc, (auc, _) = _acc(yva_np, pv), _auc(yva_np, pv)
            izgara.append({"seed": s, "C": C, "val_acc": acc, "val_auc": auc, "train_amac": amac,
                           "son_batch_loss": son, "sure_sn": round(time.time() - t0, 2)})
            tum_p[(s, C)] = pv
            if s == a.seed:
                kendi[C] = (pv, W, b, amac, time.time() - t0)
            adim += 1
            eniyi = max(eniyi, acc)
            # progress: val_acc = şimdiye dek EN İYİ uyum (küçük C'nin çoğunluk tabanına çökmesi meşru;
            # canlı alt_sinir kuralı min_adim ≥ ızgara boyu ile kullanılmalı)
            progress_yaz(run_dir, adim=adim, adim_toplam=len(secim) * len(C_liste), asama_ad="sonda",
                         seed_uyum=s, C=C, loss=son, val_acc_son_uyum=acc, val_acc=eniyi, val_acc_eniyi=eniyi,
                         gecen_sn=round(time.time() - t_kosum, 1))
            print(f"[sonda] s{s} C={C:g}: val_acc {acc:.4f} val_auc {auc:.4f} amaç {amac:.4f} "
                  f"({time.time() - t0:.1f}s)", flush=True)
    sonda_sure = time.time() - t_sonda

    def ortalama(C, k):
        return float(np.mean([g[k] for g in izgara if g["C"] == C and g["seed"] in secim]))

    # C* = argmax ort val_acc; eşitlikte ort val_auc; sonra küçük C (sıralı listede ilk)
    C_yildiz = sorted(C_liste, key=lambda C: (-round(ortalama(C, "val_acc"), 12), -round(ortalama(C, "val_auc"), 12), C))[0]
    pv, W, b, amac, tek_sure = kendi[C_yildiz]

    # ── 3b. seed bağlılığı (kart R5: farkli_seedler_farkli YERİNE tahmin farkı) ─────────
    # Seçim seed'lerinin hepsi bu koşumda C*'da zaten uyduruldu → farklı seed tahminleri tek koşumda kıyaslanır.
    # seed_tahmin_maks_fark = max_{s'≠s} max|p_s − p_s'| (kart adı); seed_tahmin_en_kucuk_fark = min_{s'≠s} max|p_s − p_s'|
    # (> 0 ⇒ kendi seed'i diğer HER seed'den farklı). Aynı seed tekrarı: C*'da kendi seed'i yeniden uydurulur →
    # ayni_seed_tahmin_maks_fark (bit-bit aynı beklenir = 0; deterministik hat kanıtı).
    seed_ek = {}
    digerleri = [float(np.max(np.abs(pv.astype(np.float64) - tum_p[(s, C_yildiz)].astype(np.float64))))
                 for s in secim if s != a.seed]
    if digerleri:
        seed_ek = {"seed_tahmin_maks_fark": max(digerleri), "seed_tahmin_en_kucuk_fark": min(digerleri),
                   "seed_tahmin_n_karsilastirma": len(digerleri)}
    if not a.tekrar_yok:
        p_tekrar = sgd_sonda(Xtr, ytr, Xva, C_yildiz, a.seed, a.epoch, a.bs, a.lr, cihaz)[0]
        seed_ek["ayni_seed_tahmin_maks_fark"] = float(np.max(np.abs(pv.astype(np.float64) - p_tekrar.astype(np.float64))))
    print(f"[sonda] seed bağlılığı @C*: {seed_ek}", flush=True)

    # ── 4. kendi seed'inin C* değerlendirmesi ─────────────────────────────────────────
    m = degerlendir(yva_np, pv, on="val", val_idx=va_idx)
    alt_grup = m.pop("alt_grup")
    kendi_acc = {C: _acc(yva_np, kendi[C][0]) for C in C_liste}
    medyan_acc = float(np.median(list(kendi_acc.values())))

    # ── 5. lbfgs denetimi (aynı amaç) ─────────────────────────────────────────────────
    ek = {}
    if not a.lbfgs_yok:
        from sklearn.linear_model import LogisticRegression
        t0 = time.time()
        C_sk = C_yildiz / 2.0
        lr_ = LogisticRegression(C=C_sk, solver="lbfgs", max_iter=5000, tol=1e-6)
        import warnings
        with warnings.catch_warnings(record=True) as uy:
            warnings.simplefilter("always")
            lr_.fit(Xtr_s, ytr_np)
        yakinsadi = not any("converge" in str(w.message).lower() for w in uy)
        p_lb = lr_.predict_proba(Xva_s)
        # predict_proba sütunları sınıf etiketleri sırasında; eksik sınıf (alt küme) varsa 7'ye genişlet
        P = np.zeros((len(yva_np), V.SINIF_SAYISI), dtype=np.float64)
        P[:, lr_.classes_] = p_lb
        sgd_ort = ortalama(C_yildiz, "val_acc")
        if lr_.coef_.shape[0] == len(lr_.classes_):  # çok sınıflı (alt kümede 2 sınıf kalırsa ikili model → atla)
            Wl = np.zeros((V.SINIF_SAYISI, Xtr_s.shape[1])); bl = np.zeros(V.SINIF_SAYISI)
            Wl[lr_.classes_], bl[lr_.classes_] = lr_.coef_, lr_.intercept_
            if len(lr_.classes_) < V.SINIF_SAYISI:
                bl[np.setdiff1d(np.arange(V.SINIF_SAYISI), lr_.classes_)] = -1e4
            amac_lb = train_amac(Xtr_s.astype(np.float64), ytr_np, Wl, bl, C_yildiz)
        else:
            amac_lb = float("nan")
        amac_sgd = train_amac(Xtr_s.astype(np.float64), ytr_np, W.astype(np.float64), b.astype(np.float64), C_yildiz)
        ek = {"val_acc_lbfgs": float((P.argmax(1) == yva_np).mean()), "lbfgs_C_sklearn": C_sk,
              "lbfgs_sgd_fark": float((P.argmax(1) == yva_np).mean()) - sgd_ort,
              # kart öldürme kuralı (e): |SGD seçim-seed ort val_acc − lbfgs val_acc| ≤ 0,005 (0,5 puan)
              "yakinsama_fark": abs(float((P.argmax(1) == yva_np).mean()) - sgd_ort),
              "lbfgs_yakinsadi": int(yakinsadi), "lbfgs_iter": int(np.max(lr_.n_iter_)),
              "lbfgs_sure_sn": round(time.time() - t0, 2),
              "train_amac_lbfgs": amac_lb, "train_amac_sgd": amac_sgd,
              "train_amac_goreli_fark": (amac_sgd - amac_lb) / max(abs(amac_lb), 1e-12)}
        print(f"[sonda] lbfgs denetimi C_sk={C_sk:g}: val_acc {ek['val_acc_lbfgs']:.4f} · SGD ort {sgd_ort:.4f} "
              f"· fark {ek['lbfgs_sgd_fark']:+.4f} · amaç SGD {amac_sgd:.5f} / lbfgs {amac_lb:.5f}", flush=True)

    # ── 6. kayıt ──────────────────────────────────────────────────────────────────────
    tahmin_kaydet(run_dir, pv, yva_np)
    np.savez(run_dir / "sonda_agirlik.npz", W=W, b=b, scaler_ort=ort, scaler_std=std, C=C_yildiz,
             seed=a.seed, model=np.array(meta["model"]), kaynak_px=a.r)
    atomik_json(run_dir / "c_izgara.json", {"secim_seedleri": secim, "C_izgara": C_liste, "secilen_C": C_yildiz,
                                             "ortalama": {str(C): {"val_acc": ortalama(C, "val_acc"),
                                                                   "val_auc": ortalama(C, "val_auc")} for C in C_liste},
                                             "uyumlar": izgara})
    egitim = round(time.time() - t_kosum, 2)
    metrics = {
        **m,
        "secilen_C": C_yildiz, "log10_secilen_C": math.log10(C_yildiz),
        "val_acc_secimC_ort": ortalama(C_yildiz, "val_acc"),
        "val_acc_medyanC_farki": kendi_acc[C_yildiz] - medyan_acc,
        "n_secim_seed": len(secim), "n_C": len(C_liste),
        **seed_ek,
        **ek,
        "giris_cozunurluk": a.r, "oznitelik_boyut": int(Xtr_np.shape[1]),
        "oznitelik_sure_sn": float(meta["sure_sn"]), "oznitelik_img_s": float(meta["img_s"]),
        "oznitelik_onbellekten": int(onbellekten),
        "oznitelik_determinizm_maks_fark": float(meta["determinizm_maks_fark"]),
        "oznitelik_tepe_gpu_bellek_mb": float(meta.get("tepe_gpu_bellek_mb", 0.0)),
        "egitim_sure_sn": egitim, "sure_sn": egitim, "sonda_izgara_sure_sn": round(sonda_sure, 2),
        "sonda_tek_sure_sn": round(tek_sure, 2), "train_amac": amac,
        "param_sayisi": int(W.size + b.size), "param_omurga": int(meta["param_omurga"]),
        "n_train": int(len(ytr_np)), "seed": a.seed, "cihaz": cihaz_ad, "alt_grup": alt_grup,
    }
    if cihaz.type == "cuda":
        metrics["tepe_gpu_bellek_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    if a.alt_kume:
        metrics["alt_kume"] = a.alt_kume
    config_ekle(run_dir, {
        "donanim": {**donanim_bilgisi(), "cihaz": cihaz_ad},
        "veri": meta["veri"], "oznitelik_onbellek": {"dosya": str(yol), "meta": meta},
        "protokol": {"sonda": "nn.Linear + CE + λ‖W‖², λ=1/(C·N), sapma cezasız", "optimizer": "Adam",
                     "lr": a.lr, "weight_decay": 0.0, "bs": a.bs, "epoch": a.epoch, "takvim": "CosineAnnealingLR(T_max=epoch)",
                     "baslatma": "W~N(0,0.01²), b=0", "olcekleme": "StandardScaler (yalnız train)",
                     "C_izgara": C_liste, "secim": "argmax ort val_acc (seçim seed'leri) → val_auc → küçük C",
                     "secim_seedleri": secim, "lbfgs_denetim": None if a.lbfgs_yok else "C_sk=C*/2 (aynı amaç), max_iter 5000",
                     "seed": a.seed, "cudnn_deterministic": True},
    })
    metrics_yaz(out, metrics)
    print(f"[sonda] BİTTİ {egitim}s · C*={C_yildiz:g} · val_acc {m['val_acc']:.4f} val_auc {m['val_auc']:.4f} "
          f"→ {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
