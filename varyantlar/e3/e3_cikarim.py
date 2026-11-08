#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
e3_cikarim.py — H16 (E3) ÇIKARIM: arşivdeki eğitilmiş bir RESMÎ kol-seed'i YENİDEN EĞİTMEDEN yükler; AYNI OTURUMDA ve aynı
model yüklemesiyle (1) yerel resmî val girdisini (MedMNIST+ dermamnist_224.npz → val_images, 1.003) ve (2) BCN ön-işleme
hattıyla HAM ISIC özgün JPEG'lerinden üretilen val girdilerini (data/ham_val_isic/e3_val_<hat>_224.npz) çalıştırır.
HAT KOLLARI ETİKET-KÖRDÜR: hat tahminleri için doğruluk/dengeli doğruluk HESAPLANMAZ (yalnız özdeşlik, sonluluk, argmax uyumu,
|Δp|, sınıf payları). Etiket yalnız yerel kolun yeniden çıkarım doğruluğu (val_acc; beklenti kapısı çıpası) için okunur.
Eşik/sıcaklık/önsel ayarı YOK. Analiz ayrı ve bir kez: scripts/e3_onisleme_analiz.py.

Kipler (--kip) → kuyruk varyantı:
  duman   e3_<kol>_duman   yerel + birincil hat (1 hat geçişi)          (a1)
  onucus  e3_<kol>_onucus  yerel + hat, jpeg, z125, z080 (4 hat geçişi) (a2)
  tam     e3_<kol>         yerel + 4 hat geçişi; tahmin dosyaları        (a3)
İzinli kollar: {h02c, h08c, h10, h11} (resmî bölme). `veri != "resmi"` ya da başka kol → rc 6 (K-g; C1 kolu resmî val'e sokulmaz).
Model yükleyici ve `hazirla`: sonuclar/e2_incele_ek/v3_model.py `kol_yukle` dallarından (r18, vit_kismi, vit_tam) BİREBİR
taşındı (v3'te 30 kol-seed resmî val özdeşliği %100 argmax, maks|Δp| ≤ 1,1e-4): r18 bs 64; ViT bs 16, vit_kismi donuk
blocks[:L0] → fp16 → son bloklar; ağırlıklar Drive'dan geçici yerel kopyayla (`yerel()`: 5 deneme, 10 sn ara; OSError 22 dersi).
Kaynak SHA256 ve taşınan satır aralığı config.json'a (e3.tasima) yazılır.
Durum (K-c): A = argmax N/N ∧ maks|Δp| ≤ 1e-5 · B = argmax ≥ 0,995 ∧ maks|Δp| ≤ 1e-3 ∧ torch/CUDA/cuDNN/timm'den biri kaynak
  config.json → donanim'dan farklı · C = diğer → hat geçişi YAPILMAZ, rc 4.
Çıkış kodları: 0 · 4 Durum C · 5 dejenere hat çıktısı (K-e) · 6 izinsiz kol (K-g) · 9 hat girdisi SHA256 ≠ beklenen ·
  10 K-f bütçe: (kip tam) e3-onisleme kuyruğunda harcanan süre + kalan a3 satırlarının a2 süresinden tahmini > 2,0 GPU-sa
  (ya da a2 süresi yok) → model yüklenmeden DUR ("a2'den sonra a3 süresi tahmin edilir; tahmin tavanı aşıyorsa a3 başlamaz").
Kullanım (kuyruk satırı; WS kökü):
  env/.venv/Scripts/python.exe -I varyantlar/e3/e3_cikarim.py --kol h10 --seed 0 --kip tam --out {RUN_DIR}/metrics.json
Yerel sınama (ölçüm değil): --cihaz cpu --cpu-izin --alt-kume 16 [--hat-dizin <vekil> --beklenen <vekil>]
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(WS / "data" / "hf_cache"))
os.environ.setdefault("TORCH_HOME", str(WS / "data" / "torch_cache"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
sys.path.insert(0, str(WS / "varyantlar"))

import numpy as np  # noqa: E402

from e2 import bcn_veri as BV  # noqa: E402  (icerik_sha256, sira_sha256, dosya_sha256, ETIKET_ANAHTARLARI)
from e2 import kollar as KL  # noqa: E402  (KOLLAR, run_id — TEK KAYNAK)
from k5.k5_cikarim import IMAGENET_ORT, IMAGENET_STD  # noqa: E402

DR = Path("<DRIVE_KOK>/3-1/runs")
NPZ224 = WS / "data" / "medmnist" / "dermamnist_224.npz"
HAT_DIZIN = WS / "data" / "ham_val_isic"
BEKLENEN = WS / "kuyruk" / "e3_onisleme_beklenen.json"
V3 = WS / "sonuclar" / "e2_incele_ek" / "v3_model.py"
V3_SATIR = "45-80 (yerel, kol_yukle: r18 + vit_kismi/vit_tam dalları); 19-21 (hazirla)"
IZINLI = ("h02c", "h08c", "h10", "h11")
HATLAR_TUM = ("hat", "jpeg", "z125", "z080")
TOL_A, TOL_B, ARGMAX_B = 1e-5, 1e-3, 0.995
BUTCE_SA, KUYRUK = 2.0, "e3-onisleme"
DURUM_KOD = {"A": 1, "B": 2, "C": 3}
NK, NV = 7, 5


def sha(p) -> str:
    return BV.dosya_sha256(Path(p))


def butce_tahmini(runs: Path, run_dir: Path) -> dict:
    """K-f: harcanan (e3-onisleme kuyruğundaki bitmiş koşumların .done sure_sn toplamı) + kalan a3 kol-seed'lerinin
    a2 (onucus) süresiyle tahmini. Bu satır (henüz bitmemiş) kalanlar arasında sayılır."""
    harcanan, a2, a3_bitti = 0.0, {}, set()
    for d in Path(runs).glob("*_e3_*"):
        if d.resolve() == run_dir.resolve():
            continue
        try:
            c = json.loads((d / "config.json").read_text(encoding="utf-8"))
            j = json.loads((d / ".done").read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if not str(c.get("kuyruk", "")).startswith(KUYRUK):
            continue
        harcanan += float(j.get("sure_sn") or 0)
        v = str(c.get("varyant", ""))
        for k in IZINLI:
            if v == f"e3_{k}_onucus" and j.get("rc") == 0:
                a2[k] = max(a2.get(k, 0.0), float(j.get("sure_sn") or 0))
            if v == f"e3_{k}" and j.get("rc") == 0:
                a3_bitti.add((k, str(c.get("seed"))))
    eksik_a2 = [k for k in IZINLI if k not in a2]
    kalan = sum(a2.get(k, 0.0) for k in IZINLI for s in KL.SEEDLER if (k, str(s)) not in a3_bitti)
    tahmin = (harcanan + kalan) / 3600
    return {"harcanan_sa": harcanan / 3600, "kalan_tahmin_sa": kalan / 3600, "toplam_tahmin_sa": tahmin, "a2_sure_sn": a2,
            "eksik_a2": eksik_a2, "gecti": bool(not eksik_a2 and tahmin <= BUTCE_SA)}


def varyant_adi(kol: str, kip: str) -> str:
    return {"duman": f"e3_{kol}_duman", "onucus": f"e3_{kol}_onucus", "tam": f"e3_{kol}"}[kip]


class ValNpz:
    """dermamnist_224.npz'den YALNIZ val_images / val_labels (K-a). Başka anahtar → reddedilir ve sayılır."""
    IZINLI = ("val_images", "val_labels")

    def __init__(self, yol=NPZ224):
        self.z = np.load(yol, allow_pickle=False)
        self.okunan = {}

    def __getitem__(self, k):
        self.okunan[k] = self.okunan.get(k, 0) + 1
        if k not in self.IZINLI:
            raise SystemExit(f"HATA (K-a): {k!r} okunamaz")
        return self.z[k]

    def test_okundu(self) -> int:
        return int(sum(v for k, v in self.okunan.items() if k.startswith("test")))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kol", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--kip", choices=["duman", "onucus", "tam"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--drive", default=str(DR), help="eğitim koşumlarının runs/ dizini (Drive)")
    p.add_argument("--cihaz", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--cpu-izin", action="store_true")
    p.add_argument("--alt-kume", type=int, default=None, help="yalnız yerel sınama: ilk N val görüntüsü")
    p.add_argument("--hat-dizin", default=None, help="yalnız yerel sınama: vekil hat girdisi dizini")
    p.add_argument("--beklenen", default=None, help="yalnız yerel sınama: vekil beklenen dosyası")
    a = p.parse_args()
    sinama = bool(a.alt_kume or a.hat_dizin or a.beklenen or a.cihaz == "cpu")
    out = Path(a.out).resolve()
    run_dir = out.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    from ortak.kayit import config_ekle, donanim_bilgisi, metrics_yaz, progress_yaz
    t0 = time.time()
    progress_yaz(run_dir, adim=0, asama_ad="hazirlik", gecen_sn=0.0)
    M = {"resmi_test_okundu": 0, "bcn_okundu": 0, "hat_etiketle_degerlendirildi": 0, "seed": a.seed}
    ek = {"kol": a.kol, "kip": a.kip, "varyant": varyant_adi(a.kol, a.kip) if a.kol in KL.KOLLAR else None, "sinama": sinama}

    # ── K-g: izinli kol ──
    K = KL.KOLLAR.get(a.kol)
    if a.kol not in IZINLI or K is None or K["veri"] != "resmi" or K["tur"] not in ("r18", "vit_kismi", "vit_tam"):
        print(f"[e3] ✖ K-g: kol {a.kol!r} izinli değil (yalnız {IZINLI}; resmî bölme) → rc 6", flush=True)
        M.update({"c1_kol": 1, "cihaz": "yok", "egitim_sure_sn": 0.0, "param_sayisi": 0})
        config_ekle(run_dir, {"e3": ek})
        metrics_yaz(out, M)
        return 6
    M["c1_kol"] = 0
    tur = K["tur"]
    run_id = KL.run_id(a.kol, a.seed)
    kd = Path(a.drive) / run_id
    hatlar = ("hat",) if a.kip == "duman" else HATLAR_TUM
    if a.kip == "tam" and not sinama:
        bt = butce_tahmini(WS / "runs", run_dir)
        ek["butce"] = bt
        M.update({"butce_toplam_tahmin_sa": round(bt["toplam_tahmin_sa"], 4), "butce_gecti": int(bt["gecti"])})
        print(f"[e3] K-f bütçe: harcanan {bt['harcanan_sa']:.3f} + kalan tahmini {bt['kalan_tahmin_sa']:.3f} = "
              f"{bt['toplam_tahmin_sa']:.3f} GPU-sa (tavan {BUTCE_SA}; a2 eksik {bt['eksik_a2']})", flush=True)
        if not bt["gecti"]:
            print("[e3] ✖ K-f: a3 bütçe tahmini tavanı aşıyor ya da a2 süresi yok → model yüklenmedi (rc 10)", flush=True)
            M.update({"cihaz": "yok", "egitim_sure_sn": 0.0, "param_sayisi": 0})
            config_ekle(run_dir, {"e3": ek})
            metrics_yaz(out, M)
            return 10

    # ── geçici yerel kopya (v3 `yerel()`; taşındı) ──
    TMP = Path(tempfile.mkdtemp(prefix="e3_"))

    def yerel(p_):
        for i in range(5):
            try:
                q = TMP / p_.name
                shutil.copyfile(p_, q)
                return q
            except OSError as e:
                print("yeniden dene", p_, e, flush=True)
                time.sleep(10)
        raise

    try:
        kc = json.loads(yerel(kd / "config.json").read_text(encoding="utf-8"))
        km = json.loads(yerel(kd / "metrics.json").read_text(encoding="utf-8"))
        kdone = json.loads(yerel(kd / ".done").read_text(encoding="utf-8"))
        if int(kc["seed"]) != a.seed or kc.get("varyant") != K["varyant"] or kdone.get("rc") != 0:
            raise SystemExit(f"HATA: kaynak {kd.name} seed/varyant/rc kol kaydıyla uyuşmuyor")
        p_arsiv = np.load(yerel(kd / "tahmin_val.npy")).astype(np.float64)
        y_arsiv = np.load(yerel(kd / "etiket_val.npy")).reshape(-1).astype(np.int64)
        progress_yaz(run_dir, adim=0, asama_ad="kaynak_okundu", gecen_sn=round(time.time() - t0, 1))

        # ── veri: yerel val (yalnız val_images/val_labels) ──
        V = ValNpz()
        Xv = V["val_images"]
        yv = V["val_labels"].reshape(-1).astype(np.int64)
        if a.alt_kume:
            Xv, yv, p_arsiv, y_arsiv = Xv[:a.alt_kume], yv[:a.alt_kume], p_arsiv[:a.alt_kume], y_arsiv[:a.alt_kume]
        if not np.array_equal(yv, y_arsiv):
            raise SystemExit("HATA: val_labels kaynak koşumun etiket_val.npy'siyle aynı değil (hiza)")

        # ── hat girdileri (etiketsiz; SHA256 = beklenen) ──
        bek = Path(a.beklenen) if a.beklenen else BEKLENEN
        hdz = Path(a.hat_dizin) if a.hat_dizin else HAT_DIZIN
        if not bek.exists():
            raise SystemExit(f"HATA: {bek} yok — a0 (scripts/e3_ham_val_hazirla.py) olmadan çıkarım yapılmaz")
        B = json.loads(bek.read_text(encoding="utf-8"))
        HX, sha_ok, sira_ok, h_index, h_ids = {}, True, True, None, None
        for g in hatlar:
            ad = f"e3_val_{g}_224.npz"
            dosya_ok = sha(hdz / ad) == (B.get("dosyalar") or {}).get(ad, {}).get("sha256")
            with np.load(hdz / ad, allow_pickle=False) as z:
                if (BV.ETIKET_ANAHTARLARI | {"val_labels", "y"}) & set(z.files):
                    raise SystemExit(f"HATA: {ad} etiket anahtarı taşıyor {z.files}")
                X_, ix_, id_ = z["images"], z["index"].astype(np.int64), z["image_id"]
            icerik_ok = BV.icerik_sha256(X_) == (B.get("icerik") or {}).get(g)
            s_ok = BV.sira_sha256(id_.tolist()) == B.get("sira_sha256") and BV.icerik_sha256(ix_) == (B.get("icerik") or {}).get("index")
            if h_index is None:
                h_index, h_ids = ix_, id_
            s_ok &= bool(np.array_equal(ix_, h_index) and np.array_equal(id_, h_ids))
            sha_ok &= bool(dosya_ok and icerik_ok)
            sira_ok &= bool(s_ok)
            if a.alt_kume:
                m_ = h_index < a.alt_kume if g == hatlar[0] else m_
                X_ = X_[m_]
            HX[g] = X_
        if a.alt_kume:
            h_index = h_index[h_index < a.alt_kume]
        M.update({"hat_girdi_sha_eslesti": int(sha_ok), "hat_sira_sha_eslesti": int(sira_ok), "hat_n": int(len(h_index)),
                  "hat_gecis_sayisi": len(hatlar), "val_n": int(len(yv))})
        if not (sha_ok and sira_ok):
            print("[e3] ✖ hat girdisinin dosya/içerik/sıra SHA256'sı beklenenle aynı değil → rc 9", flush=True)
            config_ekle(run_dir, {"e3": ek})
            M.update({"cihaz": "yok", "egitim_sure_sn": 0.0, "param_sayisi": 0})
            metrics_yaz(out, M)
            return 9

        # ── model (v3_model.py kol_yukle dalları; birebir) ──
        import torch
        import torch.nn as nn
        import timm
        from timm.layers import set_fused_attn
        from dinov2_ft import ince_ayar_vit as IV
        from ortak.kayit import cihaz_sec, seed_bagla
        cih, cihaz_ad = cihaz_sec(a.cihaz, a.cpu_izin)
        seed_bagla(a.seed)                                  # cudnn deterministik, benchmark kapalı, TF32 kapalı
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if cih.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        ort = torch.tensor(IMAGENET_ORT, device=cih).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=cih).view(1, 3, 1, 1)

        def hazirla(xb):
            x = torch.from_numpy(np.ascontiguousarray(xb)).to(cih).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
            return (x - ort) / std

        def kol_yukle():
            if tur == "r18":
                from torchvision.models import resnet18
                m = resnet18(weights=None)
                m.fc = nn.Linear(512, 7)
                m.load_state_dict(torch.load(yerel(kd / "best_model.pth"), map_location="cpu", weights_only=True), strict=True)
                m.to(cih).eval()

                def f(X):
                    with torch.no_grad():
                        return np.concatenate([torch.softmax(m(hazirla(X[i:i + 64])), 1).cpu().numpy() for i in range(0, len(X), 64)])
                return f, m, {}
            bm = json.loads(yerel(kd / "best_model_meta.json").read_text(encoding="utf-8"))
            set_fused_attn(bool(km.get("fused_attn", 1)))
            m = timm.create_model(bm["timm_ad"], pretrained=False, img_size=224, num_classes=0, drop_path_rate=0.0)
            m.head = nn.Linear(m.num_features, 7)
            m.num_classes = 7
            m.load_state_dict(torch.load(yerel(kd / "best_model.pth"), map_location="cpu", weights_only=True), strict=True)
            m.to(cih).eval()
            L0 = int(bm["donuk_blok"])
            if (tur == "vit_kismi") != (bm["mod"] == "kismi"):
                raise SystemExit(f"HATA: tür {tur} ↔ kaynak mod {bm['mod']}")

            def f(X):
                out_ = []
                with torch.no_grad():
                    for i in range(0, len(X), 16):
                        if tur == "vit_kismi":
                            t = IV.donuk_ileri(m, hazirla(X[i:i + 16]), L0).half().float()
                            out_.append(torch.softmax(IV.egitilen_ileri(m, t, L0, False), 1).cpu().numpy())
                        else:
                            out_.append(torch.softmax(m(hazirla(X[i:i + 16])), 1).cpu().numpy())
                return np.concatenate(out_)
            return f, m, {"timm_ad": bm["timm_ad"], "mod": bm["mod"], "donuk_blok": L0, "fused_attn": int(km.get("fused_attn", 1))}

        f, model, vit_bilgi = kol_yukle()
        n_param = int(sum(q.numel() for q in model.parameters()))
        progress_yaz(run_dir, adim=1, asama_ad="model_yuklendi", gecen_sn=round(time.time() - t0, 1))
        surum = {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "timm": timm.__version__}
        ref_surum = {k: (kc.get("donanim") or {}).get(k) for k in surum}
        surum_farkli = any(str(surum[k]) != str(ref_surum[k]) for k in surum)

        # ── (1) yerel kol + özdeşlik (K-c) ──
        tc = time.time()
        P_yerel = f(Xv).astype(np.float64)
        sure = {"yerel": round(time.time() - tc, 1)}
        ayni = int((P_yerel.argmax(1) == p_arsiv.argmax(1)).sum())
        oran = ayni / len(yv)
        pf = float(np.max(np.abs(P_yerel - p_arsiv)))
        durum = "A" if (ayni == len(yv) and pf <= TOL_A) else ("B" if (oran >= ARGMAX_B and pf <= TOL_B and surum_farkli) else "C")
        t_y = P_yerel.argmax(1)
        from ortak.degerlendir import dengeli_acc
        M.update({"ozdeslik_argmax_ayni": ayni, "ozdeslik_argmax_oran": oran, "ozdeslik_p_maks_fark": pf,
                  "ozdeslik_durum_kod": DURUM_KOD[durum], "ozdeslik_gecti": int(durum in ("A", "B")),
                  "ozdeslik_surum_farkli": int(surum_farkli),
                  "val_acc": float((t_y == yv).mean()), "val_dengeli_acc": float(dengeli_acc(yv, t_y)),
                  "kaynak_val_acc": float(km["val_acc"]),
                  "alt_grup": {f"sinif_{c}": float((t_y[yv == c] == c).mean()) for c in range(NK) if (yv == c).any()}})
        np.save(run_dir / "tahmin_val_yerel.npy", P_yerel.astype(np.float32))
        progress_yaz(run_dir, adim=2, asama_ad="yerel", ozdeslik_p_maks_fark=pf, gecen_sn=round(time.time() - t0, 1))
        print(f"[e3] {a.kol} s{a.seed} ({tur}) yerel: argmax {ayni}/{len(yv)} · maks|Δp| {pf:.3g} · sürüm farkı {surum_farkli} "
              f"→ Durum {durum} · val_acc {M['val_acc']:.4f} (kaynak {km['val_acc']:.4f})", flush=True)
        rc = 0
        if durum == "C":
            print("[e3] ✖ K-c Durum C → hat geçişi YAPILMADI (rc 4)", flush=True)
            rc = 4
        else:
            # ── (2) hat geçişleri (ETİKET-KÖR) ──
            Pyh = P_yerel[h_index]
            ty = Pyh.argmax(1)
            np.save(run_dir / "hat_index.npy", h_index)
            denet = {}
            for j, g in enumerate(hatlar):
                tc = time.time()
                P = f(HX[g]).astype(np.float32)
                sure[g] = round(time.time() - tc, 1)
                sonlu = bool(np.isfinite(P).all())
                th = P.argmax(1)
                dp = np.abs(P.astype(np.float64) - Pyh)
                denet[g] = {"sonlu": sonlu,
                            "sapma": float(np.max(np.abs(P.astype(np.float64).sum(1) - 1.0))) if sonlu else float("nan"),
                            "pay": float(np.bincount(th, minlength=NK).max() / len(th)) if sonlu else 1.0}
                M.update({f"{g}_argmax_uyum": float((th == ty).mean()), f"{g}_p_maks_fark": float(dp.max()),
                          f"{g}_p_ort_fark": float(dp.mean()), f"{g}_sinif_maks_pay": denet[g]["pay"],
                          f"{g}_nv_pay": float((th == NV).mean()), f"{g}_sure_sn": sure[g]})
                tmp = run_dir / f"tahmin_val_{g}.tmp.npy"
                np.save(tmp, P, allow_pickle=False)
                os.replace(tmp, run_dir / f"tahmin_val_{g}.npy")
                ek.setdefault("tahmin_sha256", {})[g] = sha(run_dir / f"tahmin_val_{g}.npy")
                progress_yaz(run_dir, adim=3 + j, asama_ad=f"hat_{g}", gecen_sn=round(time.time() - t0, 1))
                print(f"[e3]   {g}: uyum {M[f'{g}_argmax_uyum']:.4f} · maks|Δp| {M[f'{g}_p_maks_fark']:.3g} · sonlu {sonlu} · "
                      f"tek sınıf payı {denet[g]['pay']:.3f} · {sure[g]}s (ETİKET-KÖR)", flush=True)
            M.update({"hat_tahmin_sonlu": int(all(d["sonlu"] for d in denet.values())),
                      "hat_olasilik_toplam_maks_sapma": float(max(d["sapma"] for d in denet.values())),
                      "hat_tahmin_sinif_maks_pay": float(max(d["pay"] for d in denet.values())),
                      "hat_argmax_uyum": M["hat_argmax_uyum"], "yerel_nv_pay": float((t_y == NV).mean())})
            if not (M["hat_tahmin_sonlu"] and M["hat_olasilik_toplam_maks_sapma"] <= 1e-4 and M["hat_tahmin_sinif_maks_pay"] <= 0.95):
                print("[e3] ✖ K-e: dejenere hat çıktısı → DUR (rc 5)", flush=True)
                rc = 5
        M.update({"cikarim_sure_sn": round(sum(sure.values()), 1), "tepe_gpu_bellek_mb":
                  round(torch.cuda.max_memory_allocated() / 2**20, 1) if cih.type == "cuda" else 0.0,
                  "resmi_test_okundu": V.test_okundu(), "param_sayisi": n_param, "cihaz": cihaz_ad})
        ek.update({"kaynak_run": run_id, "kaynak_dizin": str(kd), "tur": tur, "hatlar": list(hatlar), "durum": durum,
                   "surum": surum, "kaynak_surum": ref_surum, "vit": vit_bilgi, "sureler_sn": sure,
                   "beklenen_sha256": sha(bek), "hat_dizin": str(hdz), "beklenen_q_BCN": B.get("q_BCN"),
                   "tasima": {"kaynak": str(V3.relative_to(WS)).replace("\\", "/"), "kaynak_sha256": sha(V3), "satirlar": V3_SATIR},
                   "betik_sha256": sha(Path(__file__)), "npz_okunan_anahtarlar": V.okunan,
                   "tol": {"A": TOL_A, "B": TOL_B, "argmax_B": ARGMAX_B}})
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    toplam = round(time.time() - t0, 2)
    M.update({"egitim_sure_sn": toplam, "sure_sn": toplam})
    if sinama:
        M["hata_ayiklama"] = 1
    config_ekle(run_dir, {"e3": ek, "donanim": {**donanim_bilgisi(), "cihaz": M.get("cihaz")}})
    metrics_yaz(out, M)
    progress_yaz(run_dir, adim=9, asama_ad="bitti", gecen_sn=toplam)
    print(f"[e3] BİTTİ {toplam}s · rc {rc} · Durum {ek.get('durum')} · hat geçişi {len(hatlar) if rc in (0, 5) else 0}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
