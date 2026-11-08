#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
e2_cikarim.py — H15 (E2) ÇIKARIM: arşivdeki eğitilmiş bir kol-seed'i (best_model.pth / sonda_agirlik.npz) YENİDEN EĞİTMEDEN
yükler, iç val özdeşliğini sınar (K-c, Durum A/B/C) ve — yalnız KİLİT altında — BCN20000 için 7 sınıflı olasılık yazar.
BCN ETİKETİ OKUNMAZ (model girdisi etiketsizdir; `varyantlar/e2/bcn_veri.py`). Eşik/sıcaklık/önsel ayarı YOK (argmax analizde).

Kipler (--kip):
  duman   a1 (H10 s0): val özdeşliği + BCN merkez_kare'nin ilk --bcn-ilk (≤ 50) görüntüsü (tahmin YAZILMAZ; yalnız şekil,
          sonluluk, olasılık toplamı, tek sınıf payı) + --kal: 8 HAM kalibrasyon görüntüsü BCN yoluyla ↔ resmî
          dermamnist_224.npz TRAIN'deki aynı görüntüler (argmax uyumu; beklenen ≥ 7/8). KİLİT gerekmez (kart a1).
  onucus  a2: yalnız val özdeşliği (resmî kollar resmî val 1.003, C1 kolları C-val 573). BCN YOK.
  bcn     a3: --dis-kilit-acik ZORUNLU ve sonuclar/e2_bcn.KILIT bu kol-seed'i ve bu betiğin/beklenen dosyasının SHA256'sını
          içermeli (K-f). Val özdeşliği yeniden sınanır; Durum C ya da yanlış val kümesi → BCN AÇILMAZ (rc 4). Aynı kol-seed için
          önceki bir BCN tahmini varsa → DUR (rc 8; yalnız --ikinci-acilis "<neden>" ile, K-f beyanı). Her geometri için girdinin
          içerik SHA256'sı beklenenle aynı olmalı (rc 9). Çıktı: tahmin_bcn_<geometri>.npy (12.413 × 7 float32, girdi sırası).
Model yükleyiciler ve dönüşümler `varyantlar/k5/k5_cikarim.py` ile AYNI (o dosyada `main()` içindeki kapanışlar olduğundan
içe aktarılamaz; satır satır aktarıldı, k5_cikarim değiştirilmedi — eşdeğerlik: README "Yerel doğrulama"; Colab'da K-c).
  r18        torchvision resnet18 + fc(512,7); bs 128; ImageNet norm.
  vit_kismi  donuk blocks[:L0] (bs 32, fp16 token, CPU) → blocks[L0:] → norm → baş (bs 64). BCN'de 1.024'lük parçalar
             (32 ve 64'ün katı → toplu iş sınırları tüm-dizi yoluyla AYNI).
  vit_tam    tam ağ, bs 64, set_fused_attn(kaynak).
  sonda      DINOv2 CLS (ozn_cikar: model_olustur + _ileri, bs 32) → kayıtlı scaler + W, b → softmax. Val öznitelikleri
             ozn_cikar önbelleğinden (k5 ile aynı); BCN öznitelikleri geometri başına BİR KEZ data/oznitelik_bcn/ altına
             (H01c ve h01_c1 aynı DINOv2-B önbelleğini paylaşır; H12 ayrı).
Durum (K-c): A = argmax N/N ∧ maks|Δp| ≤ 1e-5 · B = argmax ≥ 0,995 ∧ maks|Δp| ≤ 1e-3 ∧ torch/CUDA/cuDNN/timm'den en az biri
  kaynak koşumun config.json → donanim değerinden farklı · C = diğer. ozdeslik_gecti = A ∨ B. C → rc 4.
Çıkış kodları: 0 · 4 özdeşlik C / yanlış val kümesi · 5 dejenere BCN çıktısı (K-e) · 7 KİLİT yok/uyuşmaz · 8 ikinci BCN
  tahmini · 9 BCN girdisi SHA256 uyuşmaz.
Kullanım (Colab; kuyruk satırı): python colab/fp32_kos.py varyantlar/e2/e2_cikarim.py --kol h10 --kaynak <run_id> --seed 0
  --kip bcn --dis-kilit-acik --out {RUN_DIR}/metrics.json
Yerel sınama (ölçüm değil): --cihaz cpu --cpu-izin --alt-kume 16 [--bcn-dizin <vekil> --beklenen <vekil> --kilit <vekil>]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(WS / "data" / "hf_cache"))
os.environ.setdefault("TORCH_HOME", str(WS / "data" / "torch_cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from e2 import bcn_veri as BV  # noqa: E402
from e2 import kollar as KL  # noqa: E402
from k5.k5_cikarim import IMAGENET_ORT, IMAGENET_STD, sha256  # noqa: E402  (aynı sabitler / SHA256 işlevi)

KILIT = WS / "sonuclar" / "e2_bcn.KILIT"
BCN_OZN = Path(os.environ.get("E2_BCN_OZN_DIZIN", str(WS / "data" / "oznitelik_bcn")))   # ortam değişkeni yalnız yerel sınama
TOL_A, TOL_B, ARGMAX_B = 1e-5, 1e-3, 0.995
DURUM_KOD = {"A": 1, "B": 2, "C": 3}
PARCA = 1024   # BCN ilerleme parçası: 128 (r18), 64 (vit), 32 (öznitelik) ve --bcn-parca'nın katı → sonuç tüm-dizi yoluyla bit-bit aynı


def kilit_denetle(kilit: Path, kol: str, seed: int, kaynak_ad: str, beklenen: Path) -> tuple:
    if not kilit.exists():
        return 0, "KİLİT dosyası yok"
    try:
        K = json.loads(kilit.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return 0, f"KİLİT okunamadı: {e}"
    if (K.get("kollar") or {}).get(kol, [None] * 5)[seed] != kaynak_ad:
        return 0, f"KİLİT kol listesinde {kol} s{seed} ≠ {kaynak_ad}"
    if (K.get("sha256") or {}).get("e2_cikarim") != sha256(Path(__file__)):
        return 0, "KİLİT'teki e2_cikarim.py SHA256'sı bu dosyayla aynı değil"
    if (K.get("sha256") or {}).get("beklenen") != sha256(beklenen):
        return 0, "KİLİT'teki e2_bcn_beklenen.json SHA256'sı aynı değil"
    return 1, "tamam"


def onceki_bcn(runs: Path, varyant: str, seed: int, run_dir: Path) -> list:
    out = []
    for d in Path(runs).glob("*/"):
        if d.resolve() == run_dir.resolve() or not (d / "config.json").exists():
            continue
        if not any((d / f"tahmin_bcn_{g}.npy").exists() for g in BV.GEOMETRILER):
            continue
        try:
            c = json.loads((d / "config.json").read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if c.get("varyant") == varyant and str(c.get("seed")) == str(seed):
            out.append(d.name)
    return sorted(out)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kol", choices=sorted(KL.KOLLAR), required=True)
    p.add_argument("--kaynak", required=True, help="kaynak run_id (runs/ altında); kollar.py ile aynı olmalı")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--kip", choices=["duman", "onucus", "bcn"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bcn-ilk", type=int, default=None, help="duman: BCN merkez_kare'nin ilk N görüntüsü (≤ 50; YAZILMAZ)")
    p.add_argument("--kal", action="store_true", help="duman: 8 HAM kalibrasyon görüntüsü ön-işleme hattı denetimi")
    p.add_argument("--geometri", nargs="+", default=list(BV.GEOMETRILER), choices=list(BV.GEOMETRILER))
    p.add_argument("--dis-kilit-acik", action="store_true", help="YALNIZ a3 (kip bcn) satırları")
    p.add_argument("--ikinci-acilis", default=None, help="K-f: aynı kol-seed için ikinci BCN tahmininin NEDENİ")
    p.add_argument("--val-bs", type=int, default=None)
    p.add_argument("--onbellek-bs", type=int, default=32)
    p.add_argument("--ozn-bs", type=int, default=32)
    p.add_argument("--bcn-parca", type=int, default=1024, help="vit_kismi BCN parça boyu (32 ve 64'ün katı)")
    p.add_argument("--runs", default=None)
    p.add_argument("--cihaz", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--cpu-izin", action="store_true")
    p.add_argument("--alt-kume", type=int, default=None, help="yalnız yerel sınama: val'in ilk N görüntüsü")
    p.add_argument("--bcn-dizin", default=None, help="yalnız yerel sınama: vekil girdi dizini")
    p.add_argument("--beklenen", default=None, help="yalnız yerel sınama")
    p.add_argument("--kilit", default=None, help="yalnız yerel sınama")
    a = p.parse_args()
    sinama = bool(a.alt_kume or a.bcn_dizin or a.beklenen or a.kilit or a.cihaz == "cpu")
    K = KL.KOLLAR[a.kol]
    if a.kip == "bcn" and not a.dis_kilit_acik:
        p.error("--kip bcn yalnız --dis-kilit-acik ile (K-f)")
    if a.kip != "bcn" and (a.dis_kilit_acik or a.ikinci_acilis):
        p.error("--dis-kilit-acik / --ikinci-acilis yalnız --kip bcn")
    if a.kip == "duman" and (a.bcn_ilk is None or not 1 <= a.bcn_ilk <= 50):
        p.error("--kip duman: --bcn-ilk 1..50 (kart a1: ≤ 50 görüntü)")
    if a.kip != "duman" and (a.bcn_ilk is not None or a.kal):
        p.error("--bcn-ilk / --kal yalnız duman")
    if a.kip == "bcn" and a.alt_kume and not a.bcn_dizin:
        p.error("--alt-kume ile gerçek BCN açılamaz (yalnız vekil girdiyle sınama)")
    if a.bcn_parca % 64 or PARCA % a.bcn_parca:
        p.error(f"--bcn-parca 64'ün katı ve {PARCA}'ün böleni olmalı (toplu iş sınırları tüm-dizi yoluyla aynı kalsın)")
    if not sinama and a.kaynak != KL.run_id(a.kol, a.seed):
        p.error(f"--kaynak {a.kaynak} ≠ kol kaydı {KL.run_id(a.kol, a.seed)} (varyantlar/e2/kollar.py)")
    veri = K["veri"]
    if veri == "c1":                 # C yaması öznitelik modülleri içe aktarılmadan ÖNCE
        from k5 import c_bolme
        c_bolme.uygula("c1")
    from ortak import veri as V
    from ortak.degerlendir import _acc, _auc, dengeli_acc
    from ortak.kayit import config_ekle, donanim_bilgisi, metrics_yaz, progress_yaz

    # iç test hiçbir yolda okunmaz: yükleyici sarılır (sayım + reddetme)
    OKUMA = {"test": 0}
    _yukle = V.yukle

    def yukle_izli(size, bolumler=("train", "val"), alt_kume=None, test_izin=False, **kw):
        if "test" in bolumler:
            OKUMA["test"] += 1
            raise V.TestKilidiHatasi("E2: iç test (resmî test / C-test) açılmaz (H15)")
        return _yukle(size, bolumler, alt_kume=alt_kume, test_izin=False)
    V.yukle = yukle_izli

    out = Path(a.out).resolve()
    run_dir = out.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    runs = Path(a.runs) if a.runs else WS / "runs"
    kd = Path(a.kaynak) if Path(a.kaynak).is_absolute() else runs / a.kaynak
    kc = json.loads((kd / "config.json").read_text(encoding="utf-8"))
    km = json.loads((kd / "metrics.json").read_text(encoding="utf-8"))
    if int(kc["seed"]) != a.seed:
        raise SystemExit(f"HATA: kaynak seed {kc['seed']} ≠ --seed {a.seed}")
    kdone = json.loads((kd / ".done").read_text(encoding="utf-8")) if (kd / ".done").exists() else {}
    if kdone.get("rc") != 0:
        raise SystemExit(f"HATA: kaynak koşum rc≠0: {kd.name}")
    if not sinama and kc.get("varyant") != K["varyant"]:
        raise SystemExit(f"HATA: kaynak varyant {kc.get('varyant')} ≠ {K['varyant']}")
    kaynak_veri = "c1" if (kd / "c_bolme.json").exists() else "resmi"
    p_val_kayitli = np.load(kd / "tahmin_val.npy").astype(np.float64)
    y_val_kayitli = np.load(kd / "etiket_val.npy").reshape(-1).astype(np.int64)
    dosyalar = {ad: sha256(kd / ad) for ad in ("best_model.pth", "sonda_agirlik.npz", "tahmin_val.npy", "etiket_val.npy",
                                               "metrics.json", "config.json")}
    bcn_dizin = Path(a.bcn_dizin) if a.bcn_dizin else BV.DIZIN
    beklenen = Path(a.beklenen) if a.beklenen else BV.BEKLENEN
    kilit = Path(a.kilit) if a.kilit else KILIT
    t0 = time.time()
    progress_yaz(run_dir, adim=0, asama_ad="hazirlik", gecen_sn=0.0)
    print(f"[e2] kol {a.kol} ({K['ad']}, {K['tur']}, veri {veri}) · kaynak {kd.name} s{a.seed} · kip {a.kip}"
          + (" · SINAMA" if sinama else ""), flush=True)
    M = {"bcn_etiket_okundu": 0}
    ek = {"kol": a.kol, "kip": a.kip, "kaynak_run": kd.name, "kaynak_varyant": kc.get("varyant"), "tur": K["tur"], "veri": veri,
          "kaynak_veri": kaynak_veri, "kaynak_sha256": dosyalar, "sinama": sinama}

    # ── K-f: KİLİT ve ikinci çıkarım denetimi (model yüklenmeden ÖNCE) ──
    B = None
    rc_erken = None
    if a.kip in ("duman", "bcn") and (a.kip == "bcn" or a.bcn_ilk or a.kal):
        B = BV.beklenen_oku(beklenen)
        ek["beklenen_sha256"] = sha256(beklenen)
    if a.kip == "bcn":
        kv, neden = kilit_denetle(kilit, a.kol, a.seed, kd.name, beklenen)
        M["dis_kilit_var"] = kv
        ek["kilit"] = {"yol": str(kilit), "sha256": sha256(kilit), "denetim": neden}
        onceki = onceki_bcn(runs, KL.e2_varyant(a.kol, "bcn"), a.seed, run_dir)
        M["bcn_ikinci_cikarim"] = len(onceki)
        ek["onceki_bcn"] = onceki
        if not kv:
            print(f"[e2] ✖ K-f: {neden} → BCN AÇILMAZ (rc 7)", flush=True)
            rc_erken = 7
        elif onceki and not a.ikinci_acilis:
            print(f"[e2] ✖ K-f: aynı kol-seed için önceki BCN tahmini var {onceki} → DUR (rc 8; yalnız --ikinci-acilis)", flush=True)
            rc_erken = 8
        if a.ikinci_acilis:
            ek["ikinci_acilis_nedeni"] = a.ikinci_acilis
    if rc_erken is not None:
        M.update({"seed": a.seed, "cihaz": "yok (model yüklenmedi)", "egitim_sure_sn": 0.0, "sure_sn": 0.0, "param_sayisi": 0,
                  "resmi_test_okundu": 0, "c_test_okundu": 0})
        config_ekle(run_dir, {"e2": ek})
        metrics_yaz(out, M)
        return rc_erken

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from ortak.kayit import cihaz_sec, seed_bagla
    cihaz, cihaz_ad = cihaz_sec(a.cihaz, a.cpu_izin)
    seed_bagla(a.seed)
    if cihaz.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    ort = torch.tensor(IMAGENET_ORT, device=cihaz).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=cihaz).view(1, 3, 1, 1)

    def hazirla(xb_u8):
        x = xb_u8.to(cihaz, non_blocking=True).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
        return ((x - ort) / std).contiguous()

    def goruntuler(bolum):
        D = V.yukle(224, (bolum,), alt_kume=(a.alt_kume if bolum == "val" else None))
        return D[bolum]

    ek["veri_npz"] = V.npz_dogrula(224)
    tur = K["tur"]
    import timm
    surum = {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "timm": timm.__version__}
    ref_surum = {k: (kc.get("donanim") or {}).get(k) for k in surum}
    surum_farkli = any(str(surum[k]) != str(ref_surum[k]) for k in surum)
    ek["surum"], ek["kaynak_surum"] = surum, ref_surum

    # ── model + olasılık işlevleri (k5_cikarim.py ile aynı yollar) ──
    ozn = {}
    if tur == "r18":
        from torchvision.models import resnet18
        model = resnet18(weights=None)
        model.fc = nn.Linear(512, V.SINIF_SAYISI)
        model.load_state_dict(torch.load(kd / "best_model.pth", map_location="cpu", weights_only=True), strict=True)
        model.to(cihaz).eval()
        n_param = int(sum(q.numel() for q in model.parameters()))
        vbs = a.val_bs or 128

        def olasilik_dizi(x):
            ps = []
            with torch.no_grad():
                for i in range(0, len(x), vbs):
                    xb = torch.from_numpy(np.array(x[i:i + vbs]))
                    ps.append(torch.softmax(model(hazirla(xb)), dim=1).float().cpu().numpy())
            return np.concatenate(ps).astype(np.float32)
    elif tur in ("vit_kismi", "vit_tam"):
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
        if (tur == "vit_kismi") != (bm["mod"] == "kismi"):
            raise SystemExit(f"HATA: tür {tur} ↔ kaynak mod {bm['mod']}")
        vbs = a.val_bs or 64
        ek["vit"] = {"timm_ad": bm["timm_ad"], "mod": bm["mod"], "donuk_blok": L0, "onbellek_bs": a.onbellek_bs, "val_bs": vbs,
                     "fused_attn": int(km.get("fused_attn", 1)), "bcn_parca": a.bcn_parca}

        def _kismi_parca(xp):
            n = len(xp)
            tok = []
            for i in range(0, n, a.onbellek_bs):
                xb = torch.from_numpy(np.array(xp[i:i + a.onbellek_bs]))
                tok.append(IV.donuk_ileri(m, hazirla(xb), L0).half().cpu().numpy())
            tok = np.concatenate(tok)
            ps = []
            for i in range(0, n, vbs):
                ti = torch.from_numpy(np.ascontiguousarray(tok[i:i + vbs])).to(cihaz).float()
                ps.append(torch.softmax(IV.egitilen_ileri(m, ti, L0, False), dim=1).float().cpu().numpy())
            return np.concatenate(ps)

        def olasilik_dizi(x):
            with torch.no_grad():
                if tur == "vit_kismi":
                    ps = [_kismi_parca(x[j:j + a.bcn_parca]) for j in range(0, len(x), a.bcn_parca)]
                else:
                    ps = [torch.softmax(m(hazirla(torch.from_numpy(np.array(x[i:i + vbs])))), dim=1).float().cpu().numpy()
                          for i in range(0, len(x), vbs)]
            return np.concatenate(ps).astype(np.float32)
    else:  # sonda
        from dinov2_ozn import ozn_cikar as OZ
        z = np.load(kd / "sonda_agirlik.npz", allow_pickle=False)
        model_ad = str(z["model"])
        kisa = next(k for k, v in OZ.MODELLER.items() if v[0] == model_ad)
        W = torch.from_numpy(np.asarray(z["W"], dtype=np.float32)).to(cihaz)
        b = torch.from_numpy(np.asarray(z["b"], dtype=np.float32)).to(cihaz)
        s_ort, s_std = np.asarray(z["scaler_ort"], dtype=np.float64), np.asarray(z["scaler_std"], dtype=np.float64)
        n_param = int(W.numel() + b.numel())
        ek["sonda"] = {"model": model_ad, "C": float(z["C"]), "kaynak_px": int(z["kaynak_px"])}

        def omurga():
            if "m" not in ozn:
                ozn["m"], _, ozn["ort"], ozn["std"] = OZ.model_olustur(kisa, cihaz)
                ozn["agirlik"] = OZ._agirlik_bilgisi(ozn["m"])
            return ozn["m"]

        def oznitelik_dizi(x):
            mm = omurga()
            with torch.no_grad():
                return np.concatenate([OZ._ileri(mm, np.array(x[i:i + a.ozn_bs]), ozn["ort"], ozn["std"], cihaz)
                                       for i in range(0, len(x), a.ozn_bs)]).astype(np.float32)

        def sonda_olasilik(X):
            Xs = torch.from_numpy(((X - s_ort) / s_std).astype(np.float32)).to(cihaz)
            with torch.no_grad():
                return torch.softmax(F.linear(Xs, W, b), dim=1).cpu().numpy().astype(np.float32)

        def olasilik_dizi(x):
            return sonda_olasilik(oznitelik_dizi(x))

        def val_oznitelik():
            yol = OZ.onbellek_yolu(kisa, 224, a.alt_kume)
            if not yol.exists():
                OZ.cikar(kisa, 224, ("train", "val"), a.ozn_bs, a.alt_kume, a.cihaz, a.cpu_izin, progress_dir=run_dir)
            Dz, meta = OZ.onbellek_oku(yol)
            if not a.alt_kume:
                Xtr = Dz["train_oz"].astype(np.float32)
                ek["sonda"]["scaler_ort_maks_fark"] = float(np.max(np.abs(Xtr.mean(0, dtype=np.float64) - s_ort)))
            ek["sonda"]["ozn_meta_val"] = {k: meta.get(k) for k in ("determinizm_maks_fark", "img_s", "sure_sn")}
            return Dz["val_oz"].astype(np.float32), Dz["val_etiket"].astype(np.int64)

        def bcn_oznitelik(X, geo, icerik):
            """Geometri başına BİR KEZ (aynı oturumda paylaşılan önbellek; kilitli + atomik)."""
            BCN_OZN.mkdir(parents=True, exist_ok=True)
            yol = BCN_OZN / f"{OZ.MODELLER[kisa][1]}_bcn_{geo}_{icerik[:16]}.npy"
            meta_yol = yol.with_suffix(".json")
            if yol.exists() and meta_yol.exists():
                me = json.loads(meta_yol.read_text(encoding="utf-8"))
                if me.get("girdi_icerik_sha256") == icerik and me.get("model") == OZ.MODELLER[kisa][0]:
                    Fz = np.load(yol, allow_pickle=False)
                    if BV.icerik_sha256(Fz) == me.get("oznitelik_icerik_sha256"):
                        ek["sonda"].setdefault("bcn_onbellek", {})[geo] = {"yol": yol.name, "yeniden_kullanildi": True}
                        return Fz
            kilit_ = yol.with_suffix(".kilit")
            if kilit_.exists():   # kuyruk tek GPU'da sıralı: kalan kilit yalnız çökmüş bir önceki satırdan olabilir
                print(f"[e2] UYARI: eski öznitelik kilidi {kilit_.name} siliniyor (önceki satır yarıda kalmış)", flush=True)
            kilit_.write_text(str(os.getpid()), encoding="utf-8")
            try:
                t1 = time.time()
                parcalar = []
                for i in range(0, len(X), PARCA):
                    parcalar.append(oznitelik_dizi(X[i:i + PARCA]))
                    progress_yaz(run_dir, adim=2, asama_ad=f"bcn_oznitelik_{geo}", islenen=i + len(parcalar[-1]), toplam=len(X),
                                 gecen_sn=round(time.time() - t0, 1))
                Fz = np.concatenate(parcalar)
                det = float(np.abs(oznitelik_dizi(X[:a.ozn_bs]) - Fz[:a.ozn_bs]).max())
                tmp = yol.with_name(yol.stem + ".tmp.npy")
                np.save(tmp, Fz, allow_pickle=False)
                os.replace(tmp, yol)
                me = {"model": OZ.MODELLER[kisa][0], "geometri": geo, "girdi_icerik_sha256": icerik, "n": int(len(Fz)),
                      "bs": a.ozn_bs, "oznitelik_icerik_sha256": BV.icerik_sha256(Fz), "determinizm_maks_fark": det,
                      "agirlik": ozn.get("agirlik"), "sure_sn": round(time.time() - t1, 1), "torch": torch.__version__,
                      "timm": timm.__version__}
                meta_yol.write_text(json.dumps(me, ensure_ascii=False, indent=1), encoding="utf-8")
                ek["sonda"].setdefault("bcn_onbellek", {})[geo] = {"yol": yol.name, "yeniden_kullanildi": False,
                                                                   "determinizm_maks_fark": det}
                return Fz
            finally:
                try:
                    kilit_.unlink()
                except OSError:
                    pass

    # ── val özdeşliği (K-c) ──
    if tur == "sonda":
        Xv, y_val = val_oznitelik()
        p_val = sonda_olasilik(Xv).astype(np.float64)
    else:
        Bv = goruntuler("val")
        y_val = Bv["y"]
        p_val = olasilik_dizi(Bv["x"]).astype(np.float64)
    ref = p_val_kayitli[: len(y_val)]
    if not np.array_equal(y_val, y_val_kayitli[: len(y_val)]):
        raise SystemExit("HATA: val etiketleri kaynak koşumunkiyle aynı değil (veri/bölme hizası)")
    ayni = int((p_val.argmax(1) == ref.argmax(1)).sum())
    oran = ayni / len(y_val)
    pf = float(np.max(np.abs(p_val - ref)))
    durum = "A" if (ayni == len(y_val) and pf <= TOL_A) else (
        "B" if (oran >= ARGMAX_B and pf <= TOL_B and surum_farkli) else "C")
    val_kumesi_dogru = int(veri == kaynak_veri and (bool(a.alt_kume) or len(y_val) == KL.VAL_N[veri]))
    M.update({"ozdeslik_argmax_ayni": ayni, "ozdeslik_argmax_oran": oran, "ozdeslik_p_maks_fark": pf,
              "ozdeslik_gecti": int(durum in ("A", "B")), "ozdeslik_durum_kod": DURUM_KOD[durum],
              "ozdeslik_surum_farkli": int(surum_farkli), "val_kumesi_dogru": val_kumesi_dogru})
    ek["ozdeslik"] = {"durum": durum, "tol_A": TOL_A, "tol_B": TOL_B, "argmax_B": ARGMAX_B}
    np.save(run_dir / "tahmin_val_yeniden.npy", p_val.astype(np.float32))
    progress_yaz(run_dir, adim=1, asama_ad="val_ozdeslik", ozdeslik_p_maks_fark=pf, gecen_sn=round(time.time() - t0, 1))
    print(f"[e2] val özdeşlik: argmax {ayni}/{len(y_val)} · maks|Δp| {pf:.3g} · sürüm farkı {surum_farkli} → Durum {durum}"
          f" · val kümesi {'doğru' if val_kumesi_dogru else 'YANLIŞ'}", flush=True)
    rc = 0
    if durum == "C" or not val_kumesi_dogru:
        rc = 4

    def bcn_denet(P, onek=""):
        sonlu = bool(np.isfinite(P).all())
        sap = float(np.max(np.abs(P.astype(np.float64).sum(1) - 1.0))) if sonlu else float("nan")
        pay = float(np.bincount(P.argmax(1), minlength=V.SINIF_SAYISI).max() / len(P)) if sonlu else 1.0
        return {f"{onek}tahmin_sonlu": int(sonlu), f"{onek}olasilik_toplam_maks_sapma": sap, f"{onek}tahmin_sinif_maks_pay": pay}

    # ── a1 duman: kalibrasyon + ilk N BCN görüntüsü (tahmin YAZILMAZ) ──
    if a.kip == "duman" and rc == 0:
        if a.kal:
            Kx, kidx, kal_ok = BV.kal_oku(bcn_dizin, B)
            with np.load(V.npz_yolu(224)) as zz:
                Rx = zz["train_images"][kidx]
            pk, pr = olasilik_dizi(Kx), olasilik_dizi(Rx)
            M.update({"kal_argmax_uyum": int((pk.argmax(1) == pr.argmax(1)).sum()), "kal_n": int(len(kidx)),
                      "kal_p_maks_fark": float(np.abs(pk - pr).max()), "kal_girdi_sha_eslesti": int(kal_ok)})
            print(f"[e2] kalibrasyon: argmax uyumu {M['kal_argmax_uyum']}/{len(kidx)} (beklenen ≥ 7) · girdi SHA "
                  f"{'tamam' if kal_ok else 'UYUŞMAZ'}", flush=True)
        if a.bcn_ilk:
            Xb, _, _ = BV.geometri_oku(bcn_dizin, "merkez_kare", B, ilk=a.bcn_ilk)
            Pb = olasilik_dizi(Xb)
            M.update({"bcn_n": int(len(Pb)), **{("bcn_" + k): v for k, v in bcn_denet(Pb).items()}})
            print(f"[e2] BCN duman: {len(Pb)} görüntü · sonlu {M['bcn_tahmin_sonlu']} · toplam sapma "
                  f"{M['bcn_olasilik_toplam_maks_sapma']:.2g} · tek sınıf payı {M['bcn_tahmin_sinif_maks_pay']:.2f} (YAZILMADI)", flush=True)

    # ── a3 BCN (KİLİT) ──
    if a.kip == "bcn":
        if rc != 0:
            print("[e2] ✖ Durum C ya da yanlış val kümesi → BCN AÇILMADI (rc 4; K-c/K-h)", flush=True)
        else:
            P, sha_ok, t_b = {}, True, time.time()
            girdi_bilgi = {}
            for g in a.geometri:
                Xg, adlar, ok = BV.geometri_oku(bcn_dizin, g, B)
                girdi_bilgi[g] = {"icerik_eslesti": ok, "n": int(len(Xg))}
                sha_ok &= bool(ok)
                if not ok:
                    break
                tg = time.time()
                if tur == "sonda":
                    P[g] = sonda_olasilik(bcn_oznitelik(Xg, g, B["icerik"][g]))
                else:
                    parcalar = []
                    for j in range(0, len(Xg), PARCA):
                        parcalar.append(olasilik_dizi(Xg[j:j + PARCA]))
                        progress_yaz(run_dir, adim=2, asama_ad=f"bcn_{g}", islenen=j + len(parcalar[-1]), toplam=len(Xg),
                                     gecen_sn=round(time.time() - t0, 1))
                    P[g] = np.concatenate(parcalar)
                girdi_bilgi[g]["sure_sn"] = round(time.time() - tg, 1)
                progress_yaz(run_dir, adim=2, asama_ad=f"bcn_{g}", gecen_sn=round(time.time() - t0, 1))
                print(f"[e2] BCN {g}: {len(Xg)} görüntü {girdi_bilgi[g]['sure_sn']}s", flush=True)
            M["bcn_npz_sha_eslesti"] = int(sha_ok)
            ek["bcn_girdi"] = girdi_bilgi
            ek["bcn_sira_sha256"] = B["sira_sha256"]
            if not sha_ok:
                print("[e2] ✖ BCN girdisinin içerik SHA256'sı beklenenle aynı değil → DUR (rc 9)", flush=True)
                rc = 9
            else:
                d_ = {g: bcn_denet(P[g]) for g in P}
                M.update({"bcn_n": int(min(len(v) for v in P.values())), "bcn_geometri_sayisi": len(P),
                          "bcn_tahmin_sonlu": int(all(v["tahmin_sonlu"] for v in d_.values())),
                          "bcn_olasilik_toplam_maks_sapma": float(max(v["olasilik_toplam_maks_sapma"] for v in d_.values())),
                          "bcn_tahmin_sinif_maks_pay": float(max(v["tahmin_sinif_maks_pay"] for v in d_.values())),
                          "bcn_cikarim_sure_sn": round(time.time() - t_b, 1),
                          "bcn_img_s": round(sum(len(v) for v in P.values()) / max(time.time() - t_b, 1e-9), 1)})
                for g, P_ in P.items():   # atomik yazım: önce geçici, sonra yeniden adlandırma
                    tmp = run_dir / f"tahmin_bcn_{g}.tmp.npy"
                    np.save(tmp, P_.astype(np.float32), allow_pickle=False)
                    os.replace(tmp, run_dir / f"tahmin_bcn_{g}.npy")
                    ek.setdefault("bcn_tahmin_sha256", {})[g] = sha256(run_dir / f"tahmin_bcn_{g}.npy")
                if not (M["bcn_tahmin_sonlu"] and M["bcn_tahmin_sinif_maks_pay"] <= 0.95):
                    print("[e2] ✖ K-e: dejenere BCN çıktısı (sonlu değil ya da tek sınıf payı > 0,95) → DUR (rc 5)", flush=True)
                    rc = 5
    if cihaz.type == "cuda":
        M["tepe_gpu_bellek_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    else:
        M["tepe_gpu_bellek_mb"] = 0.0
    sure = round(time.time() - t0, 2)
    tah = p_val.argmax(1)
    M.update({"val_acc": _acc(y_val, p_val), "val_auc": _auc(y_val, p_val)[0], "val_dengeli_acc": dengeli_acc(y_val, tah),
              "val_n": int(len(y_val)), "kaynak_val_acc": float(km["val_acc"]), "veri_c": {"resmi": 0, "c1": 1}[veri],
              "resmi_test_okundu": OKUMA["test"] if veri == "resmi" else 0, "c_test_okundu": OKUMA["test"] if veri == "c1" else 0,
              "egitim_sure_sn": sure, "sure_sn": sure, "param_sayisi": n_param, "seed": a.seed, "cihaz": cihaz_ad,
              "alt_grup": {f"sinif_{c}": float((tah[y_val == c] == c).mean()) for c in range(V.SINIF_SAYISI) if (y_val == c).any()}})
    if sinama:
        M["hata_ayiklama"] = 1
    config_ekle(run_dir, {"e2": ek, "donanim": {**donanim_bilgisi(), "cihaz": cihaz_ad}})
    metrics_yaz(out, M)
    progress_yaz(run_dir, adim=3, asama_ad="bitti", gecen_sn=sure)
    print(f"[e2] BİTTİ {sure}s · rc {rc} · val_acc {M['val_acc']:.4f} (kaynak {km['val_acc']:.4f}) · Durum {durum}"
          + (f" · BCN {M.get('bcn_geometri_sayisi', 0)} geometri yazıldı" if a.kip == "bcn" and rc in (0, 5) else ""), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
