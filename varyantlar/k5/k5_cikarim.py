#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
k5_cikarim.py — K5 ÇIKARIMI: donmuş bir kaynak koşumun (best_model.pth / sonda_agirlik.npz / rakip tahminleri) val'de
YÜKLEME ÖZDEŞLİĞİ (K5-ONKAYIT §0 P6) ve — yalnız --test-kilidi-acik ile — test tahmini (§1; C1/C2'de C-test, §4–§5).
Eğitim YOK, seçim YOK: epoch/C*/τ*/T kaynak koşumda ve k5_dondurma.json'da donmuştur.

Kollar (--tur):
  r18        H02c/H08c (ince_ayar.py): torchvision resnet18 + fc Linear(512,7); val_bs 128; ImageNet norm. GPU'da.
  vit_kismi  H10 (ince_ayar_vit.py --mod kismi): donuk blocks[:L0] çıkışı fp16 token (bs 32, val ile AYNI kod yolu: donuk_ileri →
             .half()) → blocks[L0:] → norm → baş (val_bs 64). Test görüntüleri de aynı biçimde fp16'ya yuvarlanır (P6).
  vit_tam    H11 (ince_ayar_vit.py --mod tam): tam ağ görüntüden, val_bs 64; set_fused_attn(1) (eğitimle aynı).
  sonda      H01c/H12 (sonda.py): öznitelikler ozn_cikar.cikar ile (train+val önbelleği bs 32; test YALNIZ --test-kilidi-acik ile
             ayrı `_test.npz`); kayıtlı scaler + W, b → softmax(F.linear) GPU'da (eğitimle aynı).
  rakip      baseline_c: YENİDEN ÇIKARIM YOK (§1 sıra 1). Val: tahmin_val.npy'den ACC = metrics val_acc (±1e-4).
             Test: kaynak tahmin_test.npy / tahmin_test_son.npy / etiket_test.npy bu koşuma kopyalanır; ACC = metrics test_acc (±1e-4).
Özdeşlik (P6): argmax N/N aynı VE maks |Δp| ≤ --ozdeslik-tol (1e-5) → ozdeslik_gecti = 1. Test kipinde özdeşlik geçmezse
  TEST AÇILMAZ (rc = 4; DUR, teşhis). Kaynak dosyaların SHA256'sı varyant_bilgi.json'a (a1 ↔ a3 aynı olmalı; k5_test.py denetler).
Kaynak: --kaynak <run_id> (runs/ altında) YA DA --kaynak-varyant V (+ --seed; runs/ altında tek rc=0 aşama-3 koşum; C1/C2'de
  eğitim aynı kuyrukta koşar, run_id önceden bilinmez).
Veri: --veri resmi (dermamnist_224.npz; test = resmî test 2.005) | c1 | c2 (k5/c_bolme.py yaması; test = C-test 1.227).

Çıktılar ({RUN_DIR}): metrics.json (val_* + ozdeslik_*; test kipinde test_*), tahmin_val_yeniden.npy, [tahmin_test.npy,
  etiket_test.npy, (rakip) tahmin_test_son.npy], varyant_bilgi.json/config.json eki.
Kullanım (WS kökü; Colab'da `python colab/fp32_kos.py varyantlar/k5/k5_cikarim.py ...`):
  ... --tur r18 --kaynak 20261011-130855_h02_r18in_ft224_colab_s0 --seed 0 --out {RUN_DIR}/metrics.json [--test-kilidi-acik]
  Yerel sınama: --cihaz cpu --cpu-izin --alt-kume 32 (yalnız val; ölçüm değil)
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(WS / "data" / "hf_cache"))
os.environ.setdefault("TORCH_HOME", str(WS / "data" / "torch_cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

TURLER = ("r18", "vit_kismi", "vit_tam", "sonda", "rakip")
IMAGENET_ORT = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sha256(p: Path):
    if not Path(p).exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for parca in iter(lambda: f.read(1 << 22), b""):
            h.update(parca)
    return h.hexdigest()


def kaynak_bul(runs: Path, varyant: str, seed: int, kuyruk_onek: str | None) -> Path:
    bulunan = []
    for d in sorted(runs.glob("*/")):
        cf, dn = d / "config.json", d / ".done"
        if not (cf.exists() and dn.exists()):
            continue
        c = json.loads(cf.read_text(encoding="utf-8"))
        j = json.loads(dn.read_text(encoding="utf-8"))
        if c.get("varyant") == varyant and str(c.get("seed")) == str(seed) and j.get("rc") == 0 \
                and int(c.get("asama", 3)) == 3 and (not kuyruk_onek or str(c.get("kuyruk", "")).startswith(kuyruk_onek)):
            bulunan.append(d)
    if len(bulunan) != 1:
        raise SystemExit(f"HATA: {varyant} s{seed} için tek başarılı aşama-3 koşum bekleniyordu, bulunan {[d.name for d in bulunan]}")
    return bulunan[0]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tur", choices=TURLER, required=True)
    p.add_argument("--kaynak", default=None, help="kaynak run_id (runs/ altında) ya da mutlak yol")
    p.add_argument("--kaynak-varyant", default=None)
    p.add_argument("--kaynak-kuyruk", default=None, help="--kaynak-varyant araması için kuyruk adı öneki (ör. k5-c1)")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--veri", choices=["resmi", "c1", "c2"], default="resmi")
    p.add_argument("--test-kilidi-acik", action="store_true", help="YALNIZ K5 test kuyruğu / C1-C2 C-test satırları")
    p.add_argument("--ozdeslik-tol", type=float, default=1e-5)
    p.add_argument("--acc-tol", type=float, default=1e-4, help="rakip: dosyadan ACC ↔ metrics ACC")
    p.add_argument("--val-bs", type=int, default=None, help="varsayılan: r18 128, vit 64 (eğitimdeki değer)")
    p.add_argument("--onbellek-bs", type=int, default=32)
    p.add_argument("--ozn-bs", type=int, default=32)
    p.add_argument("--runs", default=None, help="runs kökü (varsayılan WS/runs)")
    p.add_argument("--cihaz", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--cpu-izin", action="store_true")
    p.add_argument("--alt-kume", type=int, default=None, help="yalnız yerel sınama: val'in ilk N görüntüsü (test yasak)")
    a = p.parse_args()
    if a.alt_kume and a.test_kilidi_acik:
        p.error("--alt-kume yalnız val sınamasında")
    if a.veri != "resmi":           # C yaması öznitelik modülleri içe aktarılmadan ÖNCE
        from k5 import c_bolme
        c_bolme.uygula(a.veri)
    from ortak import veri as V
    from ortak.degerlendir import _acc, _auc, dengeli_acc, ece
    from ortak.kayit import atomik_json, config_ekle, donanim_bilgisi, metrics_yaz, progress_yaz

    out = Path(a.out).resolve()
    run_dir = out.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    runs = Path(a.runs) if a.runs else WS / "runs"
    if a.kaynak:
        kd = Path(a.kaynak) if Path(a.kaynak).is_absolute() else runs / a.kaynak
    elif a.kaynak_varyant:
        kd = kaynak_bul(runs, a.kaynak_varyant, a.seed, a.kaynak_kuyruk)
    else:
        p.error("--kaynak ya da --kaynak-varyant gerekli")
    kc = json.loads((kd / "config.json").read_text(encoding="utf-8"))
    km = json.loads((kd / "metrics.json").read_text(encoding="utf-8"))
    if int(kc["seed"]) != a.seed:
        raise SystemExit(f"HATA: kaynak seed {kc['seed']} ≠ --seed {a.seed}")
    kdone = json.loads((kd / ".done").read_text(encoding="utf-8")) if (kd / ".done").exists() else {}
    if kdone.get("rc") != 0:
        raise SystemExit(f"HATA: kaynak koşum rc≠0: {kd.name}")
    p_val_kayitli = np.load(kd / "tahmin_val.npy").astype(np.float64)
    y_val_kayitli = np.load(kd / "etiket_val.npy").reshape(-1).astype(np.int64)
    dosyalar = {ad: sha256(kd / ad) for ad in ("best_model.pth", "sonda_agirlik.npz", "tahmin_val.npy", "etiket_val.npy",
                                               "metrics.json", "config.json")}
    t0 = time.time()
    progress_yaz(run_dir, adim=0, asama_ad="hazirlik", gecen_sn=0.0)
    print(f"[k5] kaynak {kd.name} (varyant {kc.get('varyant')}, s{a.seed}) · tür {a.tur} · veri {a.veri} · "
          f"test {'AÇIK' if a.test_kilidi_acik else 'kapalı'}", flush=True)

    p_test = y_test = None
    n_param, cihaz_ad, ek_bilgi = 0, "cpu (yeniden çıkarım yok)", {}
    if a.tur == "rakip":
        p_val = p_val_kayitli
        acc_dosya = float((p_val.argmax(1) == y_val_kayitli).mean())
        acc_fark = abs(acc_dosya - float(km["val_acc"]))
        ozd = {"ozdeslik_argmax_ayni": int(len(y_val_kayitli)), "ozdeslik_argmax_oran": 1.0, "ozdeslik_p_maks_fark": 0.0,
               "ozdeslik_val_acc_fark": acc_fark, "ozdeslik_gecti": int(acc_fark <= a.acc_tol)}
        y_val = y_val_kayitli
        n_param = int(km.get("param_sayisi", 0))
        if a.test_kilidi_acik and ozd["ozdeslik_gecti"]:
            for ad in ("tahmin_test.npy", "tahmin_test_son.npy", "etiket_test.npy"):
                if not (kd / ad).exists():
                    raise SystemExit(f"HATA: rakip kaynakta {ad} yok")
                shutil.copyfile(kd / ad, run_dir / ad)
                dosyalar[ad] = sha256(kd / ad)
            p_test = np.load(run_dir / "tahmin_test.npy").astype(np.float64)
            y_test = np.load(run_dir / "etiket_test.npy").reshape(-1).astype(np.int64)
            fark_t = abs(float((p_test.argmax(1) == y_test).mean()) - float(km["test_acc"]))
            p_son = np.load(run_dir / "tahmin_test_son.npy").astype(np.float64)
            fark_son = abs(float((p_son.argmax(1) == y_test).mean()) - float(km["test_acc_son"]))
            ozd.update({"ozdeslik_test_acc_fark": fark_t, "ozdeslik_test_acc_son_fark": fark_son})
            if fark_t > a.acc_tol or fark_son > a.acc_tol:
                ozd["ozdeslik_gecti"] = 0
            # test etiketleri resmî npz ile aynı mı (veri hizalaması)
            with np.load(V.npz_yolu(28)) as z:
                if not np.array_equal(z["test_labels"].reshape(-1).astype(np.int64), y_test):
                    ozd["ozdeslik_gecti"] = 0
                    print("[k5] HATA: rakip etiket_test.npy resmî test etiketleriyle aynı değil", flush=True)
    else:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from ortak.kayit import cihaz_sec, seed_bagla
        cihaz, cihaz_ad = cihaz_sec(a.cihaz, a.cpu_izin)
        seed_bagla(a.seed)
        ort = torch.tensor(IMAGENET_ORT, device=cihaz).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=cihaz).view(1, 3, 1, 1)

        def hazirla(xb_u8):
            x = xb_u8.to(cihaz, non_blocking=True).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
            return ((x - ort) / std).contiguous()

        def goruntuler(bolum):
            D = V.yukle(224, (bolum,), alt_kume=(a.alt_kume if bolum == "val" else None), test_izin=(bolum == "test"))
            return D[bolum]

        veri_bilgi = V.npz_dogrula(224)
        ek_bilgi["veri"] = veri_bilgi
        if a.tur == "r18":
            from torchvision.models import resnet18
            model = resnet18(weights=None)
            model.fc = nn.Linear(512, V.SINIF_SAYISI)
            model.load_state_dict(torch.load(kd / "best_model.pth", map_location="cpu", weights_only=True), strict=True)
            model.to(cihaz).eval()
            n_param = int(sum(q.numel() for q in model.parameters()))
            vbs = a.val_bs or 128

            def olasilik(bolum):
                B = goruntuler(bolum)
                x = torch.from_numpy(B["x"])
                ps = []
                with torch.no_grad():
                    for i in range(0, len(B["y"]), vbs):
                        ps.append(torch.softmax(model(hazirla(x[i:i + vbs])), dim=1).float().cpu().numpy())
                return np.concatenate(ps).astype(np.float32), B["y"]
        elif a.tur in ("vit_kismi", "vit_tam"):
            import timm
            from timm.layers import set_fused_attn
            from dinov2_ft import ince_ayar_vit as IV
            bm = json.loads((kd / "best_model_meta.json").read_text(encoding="utf-8"))
            set_fused_attn(bool(km.get("fused_attn", 1)))
            m = timm.create_model(bm["timm_ad"], pretrained=False, img_size=224, num_classes=0, drop_path_rate=0.0)
            m.head = nn.Linear(m.num_features, V.SINIF_SAYISI)
            m.num_classes = V.SINIF_SAYISI
            m.load_state_dict(torch.load(kd / "best_model.pth", map_location="cpu", weights_only=True), strict=True)
            m.to(cihaz).eval()
            n_param = int(sum(q.numel() for q in m.parameters()))
            L0 = int(bm["donuk_blok"])
            if (a.tur == "vit_kismi") != (bm["mod"] == "kismi"):
                raise SystemExit(f"HATA: --tur {a.tur} ↔ kaynak mod {bm['mod']}")
            vbs = a.val_bs or 64
            ek_bilgi["vit"] = {"timm_ad": bm["timm_ad"], "mod": bm["mod"], "donuk_blok": L0, "onbellek_bs": a.onbellek_bs,
                               "val_bs": vbs, "fused_attn": int(km.get("fused_attn", 1))}

            def olasilik(bolum):
                B = goruntuler(bolum)
                x = torch.from_numpy(B["x"])
                n = len(B["y"])
                with torch.no_grad():
                    if a.tur == "vit_kismi":
                        # eğitimdeki önbellek yolu: bs 32 ile donuk_ileri → fp16 (CPU numpy) → val_bs ile eğitilen kısım
                        tok = []
                        for i in range(0, n, a.onbellek_bs):
                            t = IV.donuk_ileri(m, hazirla(x[i:i + a.onbellek_bs]), L0)
                            tok.append(t.half().cpu().numpy())
                        tok = np.concatenate(tok)
                        ps = []
                        for i in range(0, n, vbs):
                            ti = torch.from_numpy(np.ascontiguousarray(tok[i:i + vbs])).to(cihaz).float()
                            ps.append(torch.softmax(IV.egitilen_ileri(m, ti, L0, False), dim=1).float().cpu().numpy())
                    else:
                        ps = [torch.softmax(m(hazirla(x[i:i + vbs])), dim=1).float().cpu().numpy() for i in range(0, n, vbs)]
                return np.concatenate(ps).astype(np.float32), B["y"]
        else:  # sonda
            from dinov2_ozn import ozn_cikar as OZ
            z = np.load(kd / "sonda_agirlik.npz", allow_pickle=False)
            model_ad = str(z["model"])
            kisa = next(k for k, v in OZ.MODELLER.items() if v[0] == model_ad)
            W = torch.from_numpy(np.asarray(z["W"], dtype=np.float32)).to(cihaz)
            b = torch.from_numpy(np.asarray(z["b"], dtype=np.float32)).to(cihaz)
            s_ort, s_std = np.asarray(z["scaler_ort"], dtype=np.float64), np.asarray(z["scaler_std"], dtype=np.float64)
            n_param = int(W.numel() + b.numel())
            ek_bilgi["sonda"] = {"model": model_ad, "C": float(z["C"]), "kaynak_px": int(z["kaynak_px"])}

            def olasilik(bolum):
                if bolum == "val":
                    yol = OZ.onbellek_yolu(kisa, 224, a.alt_kume)
                    if not yol.exists():
                        OZ.cikar(kisa, 224, ("train", "val"), a.ozn_bs, a.alt_kume, a.cihaz, a.cpu_izin, progress_dir=run_dir)
                    Dz, meta = OZ.onbellek_oku(yol)
                    X, y = Dz["val_oz"].astype(np.float32), Dz["val_etiket"].astype(np.int64)
                    # scaler denetimi: aynı train öznitelikleri → aynı ort/std (alt kümede atlanır)
                    if not a.alt_kume:
                        Xtr = Dz["train_oz"].astype(np.float32)
                        ek_bilgi["sonda"]["scaler_ort_maks_fark"] = float(np.max(np.abs(Xtr.mean(0, dtype=np.float64) - s_ort)))
                    ek_bilgi["sonda"]["ozn_meta_val"] = {k: meta.get(k) for k in ("determinizm_maks_fark", "img_s", "sure_sn")}
                else:
                    yol = OZ.cikar(kisa, 224, ("test",), a.ozn_bs, None, a.cihaz, a.cpu_izin, progress_dir=run_dir,
                                   test_kilidi_acik=True)
                    with np.load(yol, allow_pickle=False) as zt:   # K5: test önbelleği (onbellek_oku test'i reddeder, M4)
                        X, y = zt["test_oz"].astype(np.float32), zt["test_etiket"].astype(np.int64)
                Xs = torch.from_numpy(((X - s_ort) / s_std).astype(np.float32)).to(cihaz)
                with torch.no_grad():
                    pv = torch.softmax(F.linear(Xs, W, b), dim=1).cpu().numpy().astype(np.float32)
                return pv, y

        p_val, y_val = olasilik("val")
        p_val = p_val.astype(np.float64)
        ref = p_val_kayitli[: len(y_val)]
        if not np.array_equal(y_val, y_val_kayitli[: len(y_val)]):
            raise SystemExit("HATA: val etiketleri kaynak koşumunkiyle aynı değil (veri/bölme hizası)")
        ayni = int((p_val.argmax(1) == ref.argmax(1)).sum())
        pf = float(np.max(np.abs(p_val - ref)))
        ozd = {"ozdeslik_argmax_ayni": ayni, "ozdeslik_argmax_oran": ayni / len(y_val), "ozdeslik_p_maks_fark": pf,
               "ozdeslik_gecti": int(ayni == len(y_val) and pf <= a.ozdeslik_tol)}
        np.save(run_dir / "tahmin_val_yeniden.npy", p_val.astype(np.float32))
        progress_yaz(run_dir, adim=1, asama_ad="val_ozdeslik", ozdeslik_p_maks_fark=pf, gecen_sn=round(time.time() - t0, 1))
        print(f"[k5] val özdeşlik: argmax {ayni}/{len(y_val)} · maks|Δp| {pf:.3g} (tol {a.ozdeslik_tol:g}) → "
              f"{'GEÇTİ' if ozd['ozdeslik_gecti'] else 'İHLAL'}", flush=True)
        if a.test_kilidi_acik and ozd["ozdeslik_gecti"]:
            p_test, y_test = olasilik("test")
            p_test = p_test.astype(np.float64)
            np.save(run_dir / "tahmin_test.npy", p_test.astype(np.float32))
            np.save(run_dir / "etiket_test.npy", y_test.astype(np.int64))
        if a.cihaz == "cuda":
            ek_bilgi["tepe_gpu_bellek_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)

    sure = round(time.time() - t0, 2)
    tah = p_val.argmax(1)
    metrics = {"val_acc": _acc(y_val, p_val), "val_auc": _auc(y_val, p_val)[0], "val_dengeli_acc": dengeli_acc(y_val, tah),
               "val_n": int(len(y_val)), "kaynak_val_acc": float(km["val_acc"]), **ozd,
               "egitim_sure_sn": sure, "sure_sn": sure, "param_sayisi": n_param, "seed": a.seed, "cihaz": cihaz_ad,
               "k5_test_kilidi": int(a.test_kilidi_acik), "veri_c": {"resmi": 0, "c1": 1, "c2": 2}[a.veri],
               "alt_grup": {f"sinif_{c}": float((tah[y_val == c] == c).mean()) for c in range(V.SINIF_SAYISI) if (y_val == c).any()}}
    if a.alt_kume:
        metrics["hata_ayiklama"] = 1
    rc = 0
    if a.test_kilidi_acik:
        if not ozd["ozdeslik_gecti"]:
            print("[k5] ✖ ÖZDEŞLİK İHLALİ — TEST AÇILMADI (P6: DUR, teşhis)", flush=True)
            rc = 4
        else:
            tt = p_test.argmax(1)
            metrics.update({"test_acc": _acc(y_test, p_test), "test_auc": _auc(y_test, p_test)[0],
                            "test_dengeli_acc": dengeli_acc(y_test, tt), "test_ece": ece(y_test, p_test),
                            "test_nll": float(-np.mean(np.log(np.clip(p_test[np.arange(len(y_test)), y_test], 1e-12, 1)))),
                            "test_n": int(len(y_test)),
                            "test_sonlu": int(bool(np.isfinite(p_test).all() and np.allclose(p_test.sum(1), 1, atol=1e-3)))})
    config_ekle(run_dir, {"k5": {"kaynak_run": kd.name, "kaynak_varyant": kc.get("varyant"), "tur": a.tur, "veri": a.veri,
                                 "kaynak_sha256": dosyalar, "ozdeslik_tol": a.ozdeslik_tol, "acc_tol": a.acc_tol,
                                 "test_kilidi_acik": a.test_kilidi_acik, **ek_bilgi},
                          "donanim": {**donanim_bilgisi(), "cihaz": cihaz_ad}})
    metrics_yaz(out, metrics, test_izin=a.test_kilidi_acik)
    progress_yaz(run_dir, adim=2, asama_ad="bitti", gecen_sn=sure)
    print(f"[k5] BİTTİ {sure}s · val_acc {metrics['val_acc']:.4f} (kaynak {km['val_acc']:.4f}) · özdeşlik "
          f"{'GEÇTİ' if ozd['ozdeslik_gecti'] else 'İHLAL'}" + (" · test yazıldı" if "test_n" in metrics else "") + f" → {out}",
          flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
