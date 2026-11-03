#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ince_ayar.py — H02: torchvision ResNet-18 (IMAGENET1K_V1) native 224 px DermaMNIST uçtan uca ince ayarı.

Kart karşılığı (H02 "Değişen tek şey" + "Uygulama notu"):
- Model: torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1), fc = nn.Linear(512, 7)
  (standart 7×7 stem + maxpool). Ağırlık dosyası SHA256 config.json'a yazılır.
- Veri: dermamnist_224.npz (MD5 resmî), uint8 CPU'da tutulur; batch uint8 olarak GPU'ya taşınır, /255 ve
  ImageNet normalizasyonu GPU'da. Augmentasyon YOK. FP32 (AMP yok).
- AdamW(lr=1e-4, weight_decay=0.01), CosineAnnealingLR(T_max=epoch) epoch başına adım, ısınma yok,
  bs 64, CrossEntropyLoss(), 30 epoch (kısaltılmış aşamalarda --epoch 1/5 → T_max da ölçeklenir).
- Her epoch val değerlendirilir (TEST HİÇ DEĞERLENDİRİLMEZ, M4). Seçim: en yüksek val ACC (kesin artış;
  eşitlikte val AUC). Seçilen epoch'un state_dict'i (deepcopy) best_model.pth olarak saklanır (K5 test için).
  Ek: rakibin kuralıyla (val AUC, ilk kesin artış) seçilmiş epoch'un val ACC'si `val_acc_aucsecim`.
- Seed: random/numpy/torch/cuda + cudnn.deterministic=True, benchmark=False (baseline köprüsüyle aynı);
  mini-batch sırası seed'li torch.Generator ile.

Çıktılar ({RUN_DIR}): metrics.json (val_*; TEST ANAHTARI YOK) · progress.json (her epoch) · gecmis.json ·
  tahmin_val.npy (seçilen epoch softmax) · tahmin_val_aucsecim.npy (val-AUC seçimli epoch) · tahmin_val_son.npy ·
  etiket_val.npy · best_model.pth · config.json eki.

H05 eki (Tur-2, 2026-10-11; kart H05 "Uygulama notu"):
- `--sinif-agirlik-us β` (varsayılan 0 → H02 kod yolu DEĞİŞMEZ: ağırlıksız F.cross_entropy, özdeşlik kuralı (f)).
  β > 0: w_c = π_c^(−β) / Σ_j π_j^(1−β) (Σ_c π_c·w_c = 1), π = YALNIZ train etiket frekansı;
  F.cross_entropy(weight=w) (= nn.CrossEntropyLoss(weight=w), PyTorch ağırlıklı ortalaması). w config.json →
  protokol.sinif_agirlik ve metrics `sinif_agirlik_<sınıf>` olarak damgalanır.
- gecmis.json / progress.json'a epoch başına `val_dengeli_acc` (keşifsel; yalnız ölçüm, eğitim yolunu değiştirmez).
- `--ozdeslik-ref <metrics.json>` (yalnız ölçüm): ilk_adim_loss / ilk_epoch_loss / val_acc'nin referansla mutlak
  farkını `ozdeslik_*_fark` olarak yazar (H05 özdeşlik kuralı (f) beklenti kapısında mekanik sınansın diye).

H08 / H09 ekleri (Tur-3, 2026-10-11; kartlar H08/H09 "Uygulama notu"; ikisi BİRLİKTE verilemez):
- `--augment-d4` (varsayılan kapalı → H02 kod yolu DEĞİŞMEZ, özdeşlik kuralı (f)): train mini-batch'inde her örneğe
  D4'ün 8 dönüşümünden biri düzgün dağılımla (`ortak/dihedral.py`; H07 ile aynı tanım), hazirla()'dan SONRA GPU'da;
  k = torch.randint(0, 8, (B,), generator=g_aug), g_aug = torch.Generator("cpu").manual_seed(seed + 100_003)
  (ana üreteç g'ye dokunulmaz → batch sırası H02 ile bit bit aynı); ardından xb.contiguous(). Val yolu değişmez.
  Sayaçlar: gecmis.json `augment_sayim` (epoch başına 8 sayı), metrics `augment_frekans_{0..7}`, `augment_frekans_maks_sapma`.
- `--augment-renk G` (varsayılan 0 → H02 yolu): G > 0 iken train batch'i hazirla() yerine: uint8 → GPU →
  permute.contiguous().float()/255 → örnek başına b, c, s ∼ U[1−G, 1+G] (g_renk = Generator("cpu").manual_seed(seed + 200_003))
  → parlaklık x·b → kırp → kontrast (x − m)·c + m (m = görüntünün gri ortalaması; gri = 0,299R + 0,587G + 0,114B) → kırp →
  doygunluk gri + (x − gri)·s → kırp → ImageNet normalizasyonu. Ton yok. Val yolu değişmez.
  metrics `renk_carpan_{b,c,s}_{ort,min,max}`, `augment_renk`.
- `train_loss_son` (son epoch train kaybı) metrics'e eklendi (yalnız ölçüm; manipülasyon denetimi).

Kullanım (WS kökünden):
  env/.venv/Scripts/python.exe varyantlar/r18_imagenet/ince_ayar.py --seed 0 --out {RUN_DIR}/metrics.json [--epoch 30]
  Hata ayıklama: --size 28 --alt-kume 64 --azami-adim 1 --cihaz cpu --cpu-izin  (ölçüm değildir)
"""
import argparse
import copy
import os
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
os.environ.setdefault("TORCH_HOME", str(WS / "data" / "torch_cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ortak import veri as V  # noqa: E402
from ortak.degerlendir import _acc, _auc, degerlendir, dengeli_acc  # noqa: E402
from ortak.kayit import (atomik_json, cihaz_sec, config_ekle, donanim_bilgisi, metrics_yaz,  # noqa: E402
                         progress_yaz, seed_bagla, tahmin_kaydet)

IMAGENET_ORT = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epoch", type=int, default=30)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--size", type=int, choices=V.BOYUTLAR, default=224, help="kart: native 224; diğerleri yalnız hata ayıklama")
    p.add_argument("--val-bs", type=int, default=128)
    p.add_argument("--alt-kume", type=int, default=None, help="hata ayıklama: her bölümün ilk N örneği")
    p.add_argument("--azami-adim", type=int, default=None, help="hata ayıklama: epoch başına en çok adım")
    p.add_argument("--cihaz", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--cpu-izin", action="store_true")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--sinif-agirlik-us", type=float, default=0.0,
                   help="H05: β; w_c ∝ π_c^-β (train frekansı), Σ π·w = 1. 0 = ağırlıksız CE (H02 yolu, özdeşlik)")
    p.add_argument("--ozdeslik-ref", default=None,
                   help="yalnız ölçüm: referans metrics.json (ör. H02 dumanı) → ozdeslik_*_fark anahtarları")
    p.add_argument("--augment-d4", action="store_true",
                   help="H08: train'de örnek başına düzgün rastgele D4 dönüşümü (GPU, g_aug seed+100003). Kapalı = H02 yolu")
    p.add_argument("--augment-renk", type=float, default=0.0,
                   help="H09: G; parlaklık/kontrast/doygunluk çarpanı U[1-G,1+G], ton yok (g_renk seed+200003). 0 = H02 yolu")
    a = p.parse_args()
    if a.augment_d4 and a.augment_renk > 0:
        p.error("--augment-d4 ile --augment-renk birlikte verilemez (H08/H09 ayrı kartlar; birleşim yalnız ayrı kartla, M2)")
    if a.augment_renk < 0 or a.augment_renk >= 1:
        p.error("--augment-renk 0 ≤ G < 1 olmalı")

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    from torchvision.models import ResNet18_Weights, resnet18
    if a.threads:
        torch.set_num_threads(a.threads)
    out = Path(a.out).resolve()
    run_dir = out.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    if a.size != 224:
        print(f"[ince_ayar] UYARI: size={a.size} — kart native 224; bu koşum ölçüm sayılmaz")
    cihaz, cihaz_ad = cihaz_sec(a.cihaz, a.cpu_izin)
    seed_bagla(a.seed)

    veri_bilgi = V.npz_dogrula(a.size)
    t_yuk = time.time()
    D = V.yukle(a.size, ("train", "val"), alt_kume=a.alt_kume)
    xtr, ytr = torch.from_numpy(D["train"]["x"]), torch.from_numpy(D["train"]["y"])
    xva, yva = torch.from_numpy(D["val"]["x"]), D["val"]["y"]
    print(f"[ince_ayar] veri {veri_bilgi['npz']} md5 OK · train {len(ytr)} val {len(yva)} ({time.time() - t_yuk:.1f}s yükleme)")

    # ── H05: sınıf ağırlığı (β = 0 → None: H02 yolu bit bit aynı) ──
    beta = float(a.sinif_agirlik_us)
    pi_train = np.bincount(D["train"]["y"], minlength=V.SINIF_SAYISI) / len(D["train"]["y"])
    if beta != 0.0:
        if (pi_train == 0).any():
            raise SystemExit("HATA: train alt kümesinde eksik sınıf — sınıf ağırlığı tanımsız")
        w_np = pi_train ** (-beta) / np.sum(pi_train ** (1.0 - beta))
        print(f"[ince_ayar] sınıf ağırlığı β={beta}: " + " · ".join(f"{V.SINIF_KISA[c]} {w_np[c]:.3f}" for c in range(V.SINIF_SAYISI))
              + f" (Σπw = {float(np.sum(pi_train * w_np)):.6f})")
    else:
        w_np = None

    # ── model ────────────────────────────────────────────────────────────────────────
    w = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=w)
    model.fc = nn.Linear(512, V.SINIF_SAYISI)
    torch.manual_seed(a.seed)  # fc başlatması seed'e bağlı (resnet18() kendi init'ini global RNG ile yapar)
    model.fc.reset_parameters()
    model.to(cihaz)
    agirlik_dosya = Path(os.environ["TORCH_HOME"]) / "hub" / "checkpoints" / Path(w.url).name
    agirlik_sha = V.dosya_hash(agirlik_dosya)[1] if agirlik_dosya.exists() else None
    n_param = int(sum(q.numel() for q in model.parameters()))
    ort = torch.tensor(IMAGENET_ORT, device=cihaz).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=cihaz).view(1, 3, 1, 1)

    def hazirla(xb_u8):
        # .contiguous(): permute sonrası NHWC-adımlı tensör deterministik cuDNN'de tüm ağı yavaş channels-last
        # yoluna sokuyordu (GTX 1650: 2,1 s/adım → 0,42 s/adım; 2026-10-10 duman teşhisi). Matematik aynı.
        x = xb_u8.to(cihaz, non_blocking=True).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
        return (x - ort) / std

    # ── H08 / H09: ayrı üreteçler (ana üreteç g'ye ve global RNG'ye dokunmaz → batch sırası H02 ile aynı) ──
    if a.augment_d4:
        from ortak import dihedral as D4
        if not D4.oz_sinama():
            raise SystemExit("DUR: D4 öz-sınaması başarısız")
        g_aug = torch.Generator(device="cpu").manual_seed(a.seed + 100_003)
        aug_sayim_top = np.zeros(8, dtype=np.int64)
    G_renk = float(a.augment_renk)
    if G_renk > 0:
        g_renk = torch.Generator(device="cpu").manual_seed(a.seed + 200_003)
        renk_ist = {k: {"top": 0.0, "n": 0, "min": float("inf"), "max": float("-inf")} for k in "bcs"}
        gri_w = torch.tensor([0.299, 0.587, 0.114], device=cihaz).view(1, 3, 1, 1)

    def hazirla_renk(xb_u8):
        # H09: [0,1] ölçeğinde, normalizasyondan ÖNCE; sıra B → C → S, her adımdan sonra [0,1] kırpma (ton yok)
        x = xb_u8.to(cihaz, non_blocking=True).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
        # (B, 3): b, c, s. float64'te: float32'de 0,8 + 0,4·u üst uçta 1,2f = 1,20000005'e yuvarlanabiliyordu (U[0,8; 1,2) dışı)
        f = (1.0 - G_renk) + (2.0 * G_renk) * torch.rand((x.shape[0], 3), generator=g_renk).double()
        for i_, k_ in enumerate("bcs"):
            col = f[:, i_]
            renk_ist[k_]["top"] += float(col.sum()); renk_ist[k_]["n"] += int(col.numel())
            renk_ist[k_]["min"] = min(renk_ist[k_]["min"], float(col.min())); renk_ist[k_]["max"] = max(renk_ist[k_]["max"], float(col.max()))
        fd = f.float().to(cihaz).view(-1, 3, 1, 1)
        b_, c_, s_ = fd[:, 0:1], fd[:, 1:2], fd[:, 2:3]
        x = (x * b_).clamp_(0.0, 1.0)                                   # parlaklık
        m_ = (x * gri_w).sum(dim=1, keepdim=True).mean(dim=(2, 3), keepdim=True)
        x = ((x - m_) * c_ + m_).clamp_(0.0, 1.0)                       # kontrast (gri ortalama etrafında)
        gri = (x * gri_w).sum(dim=1, keepdim=True)
        x = (gri + (x - gri) * s_).clamp_(0.0, 1.0)                     # doygunluk (piksel grisi etrafında)
        return ((x - ort) / std).contiguous()

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epoch)
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    w_t = torch.tensor(w_np, dtype=torch.float32, device=cihaz) if w_np is not None else None

    def val_olasilik():
        model.eval()
        ps = []
        with torch.no_grad():
            for i in range(0, len(yva), a.val_bs):
                ps.append(torch.softmax(model(hazirla(xva[i:i + a.val_bs])), dim=1).float().cpu().numpy())
        model.train()
        return np.concatenate(ps).astype(np.float32)

    if cihaz.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    gecmis = []
    en = {"acc": -1.0, "auc": -1.0, "epoch": 0, "p": None, "state": None}
    en_auc = {"auc": 0.0, "epoch": 0, "acc": None, "p": None}
    ilk_adim_loss, ilk_epoch_loss = None, None
    t0 = time.time()
    N = len(ytr)
    model.train()
    for ep in range(1, a.epoch + 1):
        t_ep = time.time()
        perm = torch.randperm(N, generator=g)
        top, n_ornek, adim = 0.0, 0, 0
        aug_sayim_ep = np.zeros(8, dtype=np.int64) if a.augment_d4 else None
        for i in range(0, N, a.bs):
            j = perm[i:i + a.bs]
            xb = hazirla_renk(xtr[j]) if G_renk > 0 else hazirla(xtr[j])
            if a.augment_d4:  # H08: örnek başına D4; gruplar ayrık → okunan satırlar önceki atamalardan etkilenmez
                k_aug = torch.randint(0, 8, (len(j),), generator=g_aug)
                aug_sayim_ep += np.bincount(k_aug.numpy(), minlength=8)
                k_dev = k_aug.to(cihaz)
                for t_ in range(1, 8):
                    m_ = k_dev == t_
                    if bool(m_.any()):
                        xb[m_] = D4.donustur(xb[m_], t_)
                xb = xb.contiguous()
            yb = ytr[j].to(cihaz, non_blocking=True)
            if w_t is None:
                loss = F.cross_entropy(model(xb), yb)  # H02 yolu (β = 0), değiştirilmedi
            else:
                loss = F.cross_entropy(model(xb), yb, weight=w_t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            lv = float(loss.detach())
            if ilk_adim_loss is None:
                ilk_adim_loss = lv
            top += lv * len(j)
            n_ornek += len(j)
            adim += 1
            if a.azami_adim and adim >= a.azami_adim:
                break
        sch.step()
        if cihaz.type == "cuda":
            torch.cuda.synchronize()
        egitim_ep = time.time() - t_ep
        tr_loss = top / max(n_ornek, 1)
        if ilk_epoch_loss is None:
            ilk_epoch_loss = tr_loss
        pv = val_olasilik()
        acc, (auc, _) = _acc(yva, pv), _auc(yva, pv)
        den = dengeli_acc(yva, pv.argmax(axis=1))  # yalnız ölçüm (H05 ikincil 10)
        if acc > en["acc"] or (acc == en["acc"] and auc > en["auc"]):
            en = {"acc": acc, "auc": auc, "epoch": ep, "p": pv,
                  "state": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})}
        if auc > en_auc["auc"]:  # rakibin kuralı: val AUC ilk kesin artış
            en_auc = {"auc": auc, "epoch": ep, "acc": acc, "p": pv}
        satir = {"epoch": ep, "train_loss": tr_loss, "val_acc": acc, "val_auc": auc, "val_dengeli_acc": den,
                 "egitim_sn": round(egitim_ep, 2),
                 "epoch_sn": round(time.time() - t_ep, 2), "lr": sch.get_last_lr()[0]}
        if aug_sayim_ep is not None:
            satir["augment_sayim"] = aug_sayim_ep.tolist()
            aug_sayim_top += aug_sayim_ep
        gecmis.append(satir)
        progress_yaz(run_dir, adim=ep, epoch_toplam=a.epoch, loss=tr_loss, val_acc=acc, val_auc=auc,
                     val_dengeli_acc=den, val_acc_eniyi=en["acc"], epoch_sn=satir["epoch_sn"], gecen_sn=round(time.time() - t0, 1))
        print(f"[ince_ayar] epoch {ep}/{a.epoch} loss={tr_loss:.4f} val_acc={acc:.4f} val_auc={auc:.4f} "
              f"({satir['epoch_sn']:.1f}s)", flush=True)
    sure = round(time.time() - t0, 2)
    p_son = pv  # son epoch'un val olasılığı (yeniden hesap yok)

    # ── kayıt ────────────────────────────────────────────────────────────────────────
    torch.save(en["state"], run_dir / "best_model.pth")
    tahmin_kaydet(run_dir, en["p"], yva)
    np.save(run_dir / "tahmin_val_son.npy", p_son)
    # H02 kartı: bootstrap'ın val_acc_aucsecim kolu için val-AUC seçimli epoch'un val olasılığı
    np.save(run_dir / "tahmin_val_aucsecim.npy", np.asarray(en_auc["p"], dtype=np.float32))
    atomik_json(run_dir / "gecmis.json", gecmis)
    m = degerlendir(yva, en["p"], on="val", val_idx=D["val"]["idx"])
    alt_grup = m.pop("alt_grup")
    # val-AUC seçimli epoch'un (rakip kuralı) ikincilleri — kart (b'') ve "val_acc_aucsecim vb."
    m_auc = degerlendir(yva, en_auc["p"], on="val", val_idx=D["val"]["idx"], bootstrap=False)
    aucsecim_ek = {f"{k}_aucsecim": m_auc[k] for k in ("val_auc", "val_dengeli_acc", "val_makro_f1", "val_ece",
                   "val_tahmin_nv_payi", "val_acc_molemax_disi", "val_sizinti_tabakali_SN", "val_auc_mel", "val_auc_bkl")
                   if k in m_auc}
    if abs(m_auc["val_acc"] - en_auc["acc"]) > 1e-12:
        raise RuntimeError("val_acc_aucsecim tutarsız (kaydedilen olasılık ≠ seçilen epoch)")
    metrics = {
        **m,
        "en_iyi_epoch": en["epoch"], "epoch_sayisi": a.epoch, "val_acc_aucsecim": en_auc["acc"],
        "en_iyi_epoch_aucsecim": en_auc["epoch"], **aucsecim_ek, "val_acc_son": gecmis[-1]["val_acc"],
        "val_auc_son": gecmis[-1]["val_auc"],
        "ilk_adim_loss": ilk_adim_loss, "ilk_epoch_loss": ilk_epoch_loss,
        "epoch_sure_ort_sn": float(np.mean([g_["epoch_sn"] for g_ in gecmis])),
        "egitim_sure_sn": sure, "sure_sn": sure, "param_sayisi": n_param, "giris_cozunurluk": a.size,
        "n_train": int(N), "seed": a.seed, "cihaz": cihaz_ad, "alt_grup": alt_grup,
        "sinif_agirlik_us": beta,
        "val_dengeli_acc_eniyi_epoch": float(max(g_["val_dengeli_acc"] for g_ in gecmis)),
        "val_dengeli_acc_son": gecmis[-1]["val_dengeli_acc"],
    }
    if w_np is not None:
        metrics.update({f"sinif_agirlik_{V.SINIF_KISA[c]}": float(w_np[c]) for c in range(V.SINIF_SAYISI)})
    metrics["train_loss_son"] = gecmis[-1]["train_loss"]  # yalnız ölçüm (H08/H09 manipülasyon denetimi)
    if a.augment_d4:
        fr = aug_sayim_top / max(int(aug_sayim_top.sum()), 1)
        metrics.update({f"augment_frekans_{k_}": float(fr[k_]) for k_ in range(8)})
        metrics["augment_frekans_maks_sapma"] = float(np.max(np.abs(fr - 0.125)))
        metrics["augment_d4"] = 1
        metrics["augment_toplam_ornek"] = int(aug_sayim_top.sum())
    if G_renk > 0:
        metrics["augment_renk"] = G_renk
        for k_, v_ in renk_ist.items():
            metrics[f"renk_carpan_{k_}_ort"] = v_["top"] / max(v_["n"], 1)
            metrics[f"renk_carpan_{k_}_min"] = v_["min"]
            metrics[f"renk_carpan_{k_}_max"] = v_["max"]
    if a.ozdeslik_ref:
        import json as _json
        ref = _json.loads(Path(a.ozdeslik_ref).read_text(encoding="utf-8"))
        for k_ in ("ilk_adim_loss", "ilk_epoch_loss", "val_acc", "val_dengeli_acc"):
            if k_ in ref:
                metrics[f"ozdeslik_{k_}_fark"] = abs(float(metrics[k_]) - float(ref[k_]))
        metrics["ozdeslik_ref"] = str(a.ozdeslik_ref)
    if cihaz.type == "cuda":
        metrics["tepe_gpu_bellek_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    if a.alt_kume or a.azami_adim:
        metrics["hata_ayiklama"] = 1
    config_ekle(run_dir, {
        "donanim": {**donanim_bilgisi(), "cihaz": cihaz_ad}, "veri": veri_bilgi,
        "model": {"ad": "torchvision.models.resnet18", "agirlik": "IMAGENET1K_V1", "url": w.url,
                  "agirlik_sha256": agirlik_sha, "fc": "Linear(512,7)", "param_sayisi": n_param,
                  "torchvision": torchvision.__version__},
        "protokol": {"optimizer": "AdamW", "lr": a.lr, "weight_decay": a.wd, "takvim": f"CosineAnnealingLR(T_max={a.epoch})",
                     "isinma": "yok", "bs": a.bs, "epoch": a.epoch,
                     "kayip": "CrossEntropyLoss" if w_np is None else "CrossEntropyLoss(weight=w)",
                     "sinif_agirlik": {"beta": beta, "pi_train": pi_train.tolist(),
                                       "w": None if w_np is None else w_np.tolist(),
                                       "formul": "w_c = π_c^-β / Σ_j π_j^(1-β)", "sinif_sirasi": V.SINIF_KISA},
                     "augment": ("D4 düzgün, örnek başına, GPU, g_aug seed+100003" if a.augment_d4 else
                                 (f"fotometrik B→C→S U[{1 - G_renk:g};{1 + G_renk:g}], ton yok, GPU, g_renk seed+200003"
                                  if G_renk > 0 else "yok")),
                     "normalizasyon": "ImageNet (GPU'da)", "hassasiyet": "fp32", "size": a.size,
                     "secim": "val ACC kesin artış (eşitlikte val AUC); ek: val AUC seçimi (rakip kuralı)",
                     "seed": a.seed, "cudnn_deterministic": True, "cudnn_benchmark": False,
                     "alt_kume": a.alt_kume, "azami_adim": a.azami_adim},
    })
    metrics_yaz(out, metrics)
    print(f"[ince_ayar] BİTTİ {sure}s · en iyi epoch {en['epoch']} · val_acc {m['val_acc']:.4f} "
          f"val_auc {m['val_auc']:.4f} → {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
