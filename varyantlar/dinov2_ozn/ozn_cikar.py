#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ozn_cikar.py — Donuk DINOv2 (timm) CLS öznitelik çıkarımı → seed'den bağımsız önbellek (H01/H03).

Kart karşılığı (H01 "Uygulama notu", H03 "Değişen tek şey"):
- Model: timm==1.0.30 `vit_{small,base,large}_patch14_dinov2.lvd142m`, pretrained=True, img_size=224
  (varsayılan 518/37×37 konum gömmesi oluşturma anında bir kez 16×16'ya yeniden örneklenir), num_classes=0.
  CLS = forward_features(x)[:, 0] (son LayerNorm sonrası). eval(), torch.no_grad(), FP32 (AMP YOK).
- Girdi: dermamnist{,_64,_128,_224}.npz (MD5 resmî değerle doğrulanır), uint8 → [r≠224 ise
  PIL.Image.resize((224,224), BICUBIC) uint8 CPU'da, normalizasyondan ÖNCE] → /255 → ImageNet ort/std
  (0.485,0.456,0.406 / 0.229,0.224,0.225). Augmentasyon yok. r=224'te yeniden boyutlandırma yok.
- H03 DUR koşulu: r≠28 ise train/val etiketleri 28 px dosyasıyla bayt bayt aynı olmalı (değilse rc=2).
- Önbellek: WS/data/oznitelik/dinov2_vit{s,b,l}14_r<r>[_n<N>].npz  (+ aynı adlı .json okunur meta)
  anahtarlar: {train,val}_oz float32 (N,D) · {train,val}_etiket int64 · {train,val}_idx int64 (resmî indeks) · meta (JSON str)
  meta: model adı, timm/torch sürümü, ağırlık dosyası + SHA256 + HF revizyonu, girdi npz MD5/SHA256, r,
  çıkarım süresi/hızı, tepe GPU belleği, determinizm denetimi (aynı ilk batch iki kez → maks |fark|).
- TEST (M4): varsayılan bölümler train,val. `test` yalnız `--test-kilidi-acik` ile ve AYRI dosyaya
  (`..._test.npz`) yazılır — K5 işidir; sonda.py bu dosyayı hiç açmaz.
- HF_HOME = WS/data/hf_cache (ağırlık indirmesi WS içinde).

Kullanım (WS kökünden; normalde sonda.py önbellek yoksa bunu süreç içinde çağırır):
  env/.venv/Scripts/python.exe varyantlar/dinov2_ozn/ozn_cikar.py --model b --r 224 [--bs 32] [--alt-kume 256]
      [--cihaz cuda|cpu --cpu-izin] [--progress-dir {RUN_DIR}] [--zorla]
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(WS / "data" / "hf_cache"))
os.environ.setdefault("TORCH_HOME", str(WS / "data" / "torch_cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ortak import veri as V  # noqa: E402
from ortak.kayit import atomik_json, progress_yaz  # noqa: E402

MODELLER = {
    "s": ("vit_small_patch14_dinov2.lvd142m", "dinov2_vits14"),
    "b": ("vit_base_patch14_dinov2.lvd142m", "dinov2_vitb14"),
    "l": ("vit_large_patch14_dinov2.lvd142m", "dinov2_vitl14"),
}
IMAGENET_ORT = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
GIRIS_PX = 224
# DERMA_OZN_DIZIN yalnız yerel mini doğrulama için (gerçek önbellek: WS/data/oznitelik)
OZN_DIZIN = Path(os.environ.get("DERMA_OZN_DIZIN", str(WS / "data" / "oznitelik")))


def onbellek_yolu(model: str, r: int, alt_kume: int | None = None, test: bool = False) -> Path:
    kisa = MODELLER[model][1]
    return OZN_DIZIN / f"{kisa}_r{r}{f'_n{alt_kume}' if alt_kume else ''}{'_test' if test else ''}.npz"


def onbellek_oku(yol: Path) -> tuple[dict, dict]:
    """(diziler, meta). Test dosyası reddedilir (M4)."""
    if yol.name.endswith("_test.npz"):
        raise V.TestKilidiHatasi("test öznitelik önbelleği K4'te açılamaz")
    with np.load(yol, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files if k != "meta"}
        meta = json.loads(str(z["meta"]))
    return d, meta


def _agirlik_bilgisi(model) -> dict:
    """timm pretrained_cfg → HF önbelleğindeki safetensors dosyası, SHA256 ve snapshot (revizyon) adı."""
    hf_id = (getattr(model, "pretrained_cfg", None) or {}).get("hf_hub_id")
    bilgi = {"hf_hub_id": hf_id}
    if not hf_id:
        return bilgi
    yol = None
    try:
        from huggingface_hub import try_to_load_from_cache
        for ad in ("model.safetensors", "pytorch_model.bin"):
            p = try_to_load_from_cache(hf_id, ad)
            if isinstance(p, str) and os.path.exists(p):
                yol = Path(p)
                break
    except Exception as e:  # noqa: BLE001
        bilgi["hata"] = str(e)
    if yol is not None:
        _, sha = V.dosya_hash(yol)
        bilgi.update({"dosya": yol.name, "sha256": sha, "bayt": yol.stat().st_size,
                      "hf_revizyon": yol.parent.name})
    return bilgi


def _on_isle(x_u8: np.ndarray, r: int) -> np.ndarray:
    """uint8 (B,r,r,3) → uint8 (B,224,224,3). r=224 → özdeşlik. PIL bicubic, CPU, normalizasyondan önce."""
    if r == GIRIS_PX and x_u8.shape[1] == GIRIS_PX:
        return x_u8
    from PIL import Image
    return np.stack([np.asarray(Image.fromarray(im).resize((GIRIS_PX, GIRIS_PX), Image.BICUBIC)) for im in x_u8])


def model_olustur(model: str, cihaz):
    import timm
    import torch
    ad = MODELLER[model][0]
    m = timm.create_model(ad, pretrained=True, img_size=GIRIS_PX, num_classes=0)
    m.eval().to(cihaz)
    for p in m.parameters():
        p.requires_grad_(False)
    n_param = int(sum(p.numel() for p in m.parameters()))
    ort = torch.tensor(IMAGENET_ORT, device=cihaz).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=cihaz).view(1, 3, 1, 1)
    return m, n_param, ort, std


def _ileri(m, x_u8: np.ndarray, ort, std, cihaz):
    import torch
    x = torch.from_numpy(x_u8).to(cihaz).permute(0, 3, 1, 2).float().div_(255.0)
    x = (x - ort) / std
    f = m.forward_features(x)
    return f[:, 0].float().cpu().numpy()  # CLS (son norm sonrası)


def cikar(model: str = "b", r: int = 224, bolumler=("train", "val"), bs: int = 32, alt_kume: int | None = None,
          cihaz_str: str = "cuda", cpu_izin: bool = False, progress_dir: Path | None = None, zorla: bool = False,
          test_kilidi_acik: bool = False, ilerleme_aralik: int = 500, log=print) -> Path:
    """Önbelleği üretir (varsa ve geçerliyse dokunmaz). Dönüş: önbellek yolu."""
    import torch
    from ortak.kayit import cihaz_sec
    test = "test" in bolumler
    if test and (not test_kilidi_acik or set(bolumler) != {"test"}):
        raise V.TestKilidiHatasi("test çıkarımı yalnız --test-kilidi-acik ile ve tek başına (bölüm=test) — K5")
    yol = onbellek_yolu(model, r, alt_kume, test=test)
    if yol.exists() and not zorla:
        log(f"[ozn] önbellek mevcut: {yol}")
        return yol
    OZN_DIZIN.mkdir(parents=True, exist_ok=True)
    kilit = yol.with_suffix(".kilit")
    try:
        fd = os.open(str(kilit), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()} {datetime.now().isoformat()}".encode())
        os.close(fd)
    except FileExistsError:
        # başka bir süreç çıkarıyor → bitmesini bekle (progress yazarak; kosum.py takılma kuralı tetiklenmesin)
        t0 = time.time()
        while kilit.exists() and not yol.exists() and time.time() - t0 < 3600:
            if progress_dir:
                progress_yaz(progress_dir, adim=0, asama_ad="oznitelik_bekle", gecen_sn=round(time.time() - t0, 1))
            time.sleep(10)
        if yol.exists():
            return yol
        raise SystemExit(f"HATA: kilit {kilit} 1 sa'tir duruyor ve önbellek oluşmadı (yetim kilit? elle sil)")
    try:
        cihaz, cihaz_ad = cihaz_sec(cihaz_str, cpu_izin)
        veri_bilgi = V.npz_dogrula(r)
        etiket_ok = V.etiket_tutarliligi(r)
        if not etiket_ok:
            raise SystemExit(f"HATA (H03 DUR): {V.npz_adi(r)} train/val etiketleri 28 px dosyasıyla aynı değil")
        D = V.yukle(r, bolumler=bolumler, alt_kume=alt_kume, test_izin=test_kilidi_acik)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if cihaz.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t_model = time.time()
        m, n_param, ort, std = model_olustur(model, cihaz)
        agirlik = _agirlik_bilgisi(m)
        log(f"[ozn] {MODELLER[model][0]} yüklendi ({n_param:,} param, {time.time() - t_model:.1f}s) · "
            f"cihaz {cihaz_ad} · r={r} → {GIRIS_PX} · bs {bs} · ağırlık sha256 {str(agirlik.get('sha256'))[:12]}")
        diziler, sureler = {}, {}
        toplam = sum(len(D[b]["y"]) for b in bolumler)
        islenen, t_bas = 0, time.time()
        son_ilerleme = 0
        det_fark = None
        with torch.no_grad():
            for b in bolumler:
                x, y, idx = D[b]["x"], D[b]["y"], D[b]["idx"]
                cikti = []
                t_b = time.time()
                for i in range(0, len(y), bs):
                    xb = _on_isle(x[i:i + bs], r)
                    cikti.append(_ileri(m, xb, ort, std, cihaz))
                    islenen += len(xb)
                    if progress_dir and (islenen - son_ilerleme >= ilerleme_aralik or islenen == toplam):
                        son_ilerleme = islenen
                        gecen = time.time() - t_bas
                        progress_yaz(progress_dir, adim=0, asama_ad="oznitelik", bolum=b, islenen=islenen,
                                     toplam=toplam, img_s=round(islenen / max(gecen, 1e-9), 2),
                                     gecen_sn=round(gecen, 1))
                        log(f"[ozn] {islenen}/{toplam} ({islenen / max(gecen, 1e-9):.1f} img/s)")
                F = np.concatenate(cikti).astype(np.float32)
                if cihaz.type == "cuda":
                    torch.cuda.synchronize()
                sureler[b] = round(time.time() - t_b, 2)
                diziler[f"{b}_oz"], diziler[f"{b}_etiket"], diziler[f"{b}_idx"] = F, y, idx
                if det_fark is None:  # determinizm denetimi: ilk batch'i aynı boyutla yeniden çıkar
                    tekrar = _ileri(m, _on_isle(x[:bs], r), ort, std, cihaz)
                    det_fark = float(np.abs(tekrar - F[:len(tekrar)]).max())
        sure = round(time.time() - t_bas, 2)
        import timm
        meta = {
            "model": MODELLER[model][0], "model_kisa": MODELLER[model][1], "timm": timm.__version__,
            "torch": torch.__version__, "agirlik": agirlik, "param_omurga": n_param,
            "oznitelik": "CLS = forward_features(x)[:,0] (son LayerNorm sonrası)", "oznitelik_boyut": int(F.shape[1]),
            "giris_px": GIRIS_PX, "kaynak_px": r,
            "buyutme": "yok (özdeşlik)" if r == GIRIS_PX else "PIL.Image.resize((224,224), BICUBIC), uint8, CPU, normalizasyondan önce",
            "normalizasyon": {"ort": IMAGENET_ORT, "std": IMAGENET_STD, "olcek": "/255"},
            "hassasiyet": "fp32 (AMP yok; TF32 kapalı)", "bs": bs, "cihaz": cihaz_ad,
            "veri": veri_bilgi, "etiket_28_ile_ayni": bool(etiket_ok), "bolumler": list(bolumler),
            "n": {b: int(len(D[b]["y"])) for b in bolumler}, "alt_kume": alt_kume,
            "sure_sn": sure, "sure_bolum_sn": sureler, "img_s": round(toplam / max(sure, 1e-9), 2),
            "tepe_gpu_bellek_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if cihaz.type == "cuda" else 0.0,
            "determinizm_maks_fark": det_fark, "tarih": datetime.now().isoformat(timespec="seconds"),
        }
        tmp = yol.with_name(yol.stem + ".tmp.npz")
        np.savez(tmp, meta=np.array(json.dumps(meta, ensure_ascii=False)), **diziler)
        os.replace(tmp, yol)
        atomik_json(yol.with_suffix(".json"), meta)
        log(f"[ozn] yazıldı {yol} · {toplam} görüntü {sure}s ({meta['img_s']} img/s) · determinizm maks|fark| {det_fark:.3g}")
        return yol
    finally:
        try:
            kilit.unlink()
        except OSError:
            pass


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=sorted(MODELLER), default="b")
    p.add_argument("--r", type=int, choices=V.BOYUTLAR, default=224, help="kaynak çözünürlük (npz)")
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--alt-kume", type=int, default=None, help="her bölümün ilk N örneği (duman)")
    p.add_argument("--cihaz", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--cpu-izin", action="store_true", help="yalnız hata ayıklama; ölçüm koşumunda KULLANILMAZ")
    p.add_argument("--threads", type=int, default=None, help="CPU iş parçacığı (mini doğrulama)")
    p.add_argument("--progress-dir", default=None)
    p.add_argument("--zorla", action="store_true", help="önbellek varsa da yeniden çıkar")
    p.add_argument("--test-kilidi-acik", action="store_true", help="YALNIZ K5: test özniteliklerini ayrı dosyaya çıkarır")
    a = p.parse_args()
    if a.threads:
        import torch
        torch.set_num_threads(a.threads)
    bolumler = ("test",) if a.test_kilidi_acik else ("train", "val")
    yol = cikar(a.model, a.r, bolumler, a.bs, a.alt_kume, a.cihaz, a.cpu_izin,
                Path(a.progress_dir) if a.progress_dir else None, a.zorla, a.test_kilidi_acik)
    print(json.dumps(json.loads(yol.with_suffix(".json").read_text(encoding="utf-8")), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
