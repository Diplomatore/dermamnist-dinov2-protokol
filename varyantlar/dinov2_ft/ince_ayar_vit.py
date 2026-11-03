#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ince_ayar_vit.py — Tur-4 ön-ölçüm aracı: DINOv2 (timm) native 224 px DermaMNIST ince ayarı.
KART YOK (2026-10-11): bu betik SIRALAMA Tur-4 notu + tahlil-tur3 §5'teki "önce ölçüm" içindir. Kart yazıldığında
kartın "Uygulama notu" bu betiğin argümanlarına bağlanır; varsayılanlar kart değildir.

Modlar
- `--mod kismi --son-k K`: ilk L0 = 12 − K blok DONUK. Donuk kısmın çıkış token'ları (patch_embed → pos_embed →
  blocks[:L0], son LayerNorm ÖNCESİ; CLS + 256 patch = 257 token) train/val için BİR KEZ önbelleğe alınır
  (tek görünüm, augment yok): `data/oznitelik/dinov2_vit{s,b}14_r224_blok{L0}_tok[_n<N>]_{train,val}.npy`
  (float16, (N, 257, D); np.lib.format → memmap ile okunur, RAM'e tam yüklenmez) + `..._tok[_n<N>].json` meta.
  ViT-B: 8.010 × 257 × 768 × 2 B ≈ 3,16 GB (train 2,77 + val 0,40). fp32 olsaydı 6,3 GB → fp16 saklanır; fp16
  yuvarlama etkisi meta'da ölçülür (`fp16_cls_maks_fark`). Eğitilen: blocks[L0:], son norm, baş.
  Önbellek doğruluğu her koşumda denetlenir: ilk 8 val görüntüsünde tam ağ (fp32, görüntüden) ↔ önbellek yolu
  (fp16 token → blocks[L0:] → norm) normlanmış CLS maks |fark| → `onbellek_tutarlilik_maks_fark`.
- `--mod tam`: bütün ağ eğitilir; grad checkpointing (varsayılan açık), mikro-batch + gradyan biriktirme.

Tarif (varsayılanlar; literatür: Wu & Papież 2025 arXiv:2501.14685 — AdamW, sınıflandırıcı lr 1e-3, kodlayıcı lr
{1e-4, 1e-5, 1e-6} ızgarası; H02 ebeveyninden: AdamW wd 0,01, etkin bs 64, kosinüs, ısınma yok, FP32, augment yok)
- Baş: Linear(D, 7) — torch.manual_seed(seed) sonrası PyTorch varsayılan init (H02'nin fc.reset_parameters() ile aynı
  mantık). Girdi: son LayerNorm sonrası CLS (timm global_pool='token'; H01 sondasıyla aynı öznitelik).
- AdamW; parametre grupları: baş `--lr-bas` (1e-3), gövde `--lr-govde` (1e-5); `--katman-azalma d` < 1 ise blok i'nin
  lr'si lr_govde · d^(L−1−i) (LLRD; varsayılan 1 = kapalı; patch_embed/pos_embed/cls_token en alt kademe).
  Weight decay `--wd` (0,01) yalnız ≥ 2 boyutlu ağırlıklara; bias, LayerNorm, LayerScale, pos_embed, cls_token muaf
  (timm no_weight_decay + 1-B kuralı; ViT ince ayar standardı).
- Takvim: ADIM başına kosinüs (eta_min 0), isteğe bağlı doğrusal ısınma `--isinma-epoch` (varsayılan 0).
- Etkin bs `--bs` (64) = mikro-batch'ler (`--mikro-bs`) üzerinde gradyan birikimi; kayıp = Σ CE / |etkin batch|
  (ViT'de BatchNorm yok → büyük batch ile matematiksel olarak aynı).
- `--drop-path` (0) timm stochastic depth; `--fused-attn` (1) F.scaled_dot_product_attention.
- Her epoch val; seçim: en yüksek val ACC (kesin artış; eşitlikte val AUC) → tahmin_val.npy + best_model.pth
  (TAM state_dict; kısmi modda donuk bloklar önağırlıkla aynı). Ek: rakip kuralı (val AUC ilk kesin artış) →
  val_acc_aucsecim + tahmin_val_aucsecim.npy. metrics anahtarları H02 (`ortak/degerlendir.py`) ile aynı.
  TEST HİÇ OKUNMAZ (M4): veri yükleyici test kilitli, metrics_yaz test_ anahtarını reddeder.
- progress.json: epoch sonunda (adim = epoch) + epoch içinde ≈ 60 s'de bir (adim = tamamlanan epoch, adim_ici);
  önbellek yazımında adim = 0, asama_ad = "onbellek" (kosum.py takılma kuralı tetiklenmesin).
- `--cpu-izin` olmadan CUDA zorunlu (rc=97). Seed: random/numpy/torch + cudnn.deterministic, TF32 kapalı.

Ölçüm alanları (metrics): tepe_gpu_bellek_mb, tepe_ram_ws_mb (tepe çalışma kümesi; memmap sayfaları dahil),
tepe_ram_ozel_mb (tepe özel commit; memmap hariç), egitim_img_s_kararli (ilk 5 optimizasyon adımı hariç),
epoch_egitim_sn_ort / epoch_val_sn_ort, onbellek_yazma_sn (yalnız bu koşumda yazıldıysa), onbellek_bayt.

Kullanım (WS kökünden):
  env/.venv/Scripts/python.exe varyantlar/dinov2_ft/ince_ayar_vit.py --mod kismi --son-k 2 --seed 0 --epoch 1 --out {RUN_DIR}/metrics.json
  env/.venv/Scripts/python.exe varyantlar/dinov2_ft/ince_ayar_vit.py --mod kismi --son-k 4 --sadece-onbellek --out {RUN_DIR}/metrics.json
  env/.venv/Scripts/python.exe varyantlar/dinov2_ft/ince_ayar_vit.py --mod tam --model s --seed 0 --epoch 1 --out {RUN_DIR}/metrics.json
  Hata ayıklama: --alt-kume 16 --azami-adim 2 --cihaz cpu --cpu-izin (+ DERMA_OZN_DIZIN=<geçici dizin>; ölçüm değildir)
"""
import argparse
import copy
import json
import math
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

from dinov2_ozn.ozn_cikar import IMAGENET_ORT, IMAGENET_STD, MODELLER, OZN_DIZIN, _agirlik_bilgisi  # noqa: E402
from ortak import veri as V  # noqa: E402
from ortak.degerlendir import _acc, _auc, degerlendir, dengeli_acc  # noqa: E402
from ortak.kayit import (atomik_json, cihaz_sec, config_ekle, donanim_bilgisi, metrics_yaz,  # noqa: E402
                         progress_yaz, seed_bagla, tahmin_kaydet)

GIRIS_PX = 224
PROGRESS_ARALIK_SN = 60


# ── süreç belleği (psutil yok; Windows: GetProcessMemoryInfo) ─────────────────────────────────
def ram_tepe_mb() -> dict:
    try:
        if os.name == "nt":
            import ctypes
            import ctypes.wintypes as wt

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = wt.HANDLE
            ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
            ctypes.windll.psapi.GetProcessMemoryInfo.restype = wt.BOOL
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            ctypes.windll.psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
            return {"tepe_ram_ws_mb": round(pmc.PeakWorkingSetSize / 2**20, 1),
                    "tepe_ram_ozel_mb": round(pmc.PeakPagefileUsage / 2**20, 1)}
        import resource
        return {"tepe_ram_ws_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
    except Exception as e:  # noqa: BLE001
        print(f"[vit] UYARI: RAM ölçülemedi: {e}")
        return {}


# ── önbellek ───────────────────────────────────────────────────────────────────────────────
def onbellek_kok(model: str, L0: int, alt_kume) -> Path:
    return OZN_DIZIN / f"{MODELLER[model][1]}_r{GIRIS_PX}_blok{L0}_tok{f'_n{alt_kume}' if alt_kume else ''}"


def onbellek_yollari(kok: Path) -> dict:
    return {"train": kok.with_name(kok.name + "_train.npy"), "val": kok.with_name(kok.name + "_val.npy"),
            "meta": kok.with_name(kok.name + ".json"), "kilit": kok.with_name(kok.name + ".kilit")}


def donuk_ileri(m, x, L0: int):
    """patch_embed → _pos_embed → patch_drop → norm_pre → blocks[:L0] (timm forward_features ile aynı sıra)."""
    t = m.patch_embed(x)
    t = m._pos_embed(t)
    t = m.patch_drop(t)
    t = m.norm_pre(t)
    for blk in m.blocks[:L0]:
        t = blk(t)
    return t


def egitilen_ileri(m, t, L0: int, ckpt: bool = False):
    """blocks[L0:] → norm → forward_head (CLS → baş). ckpt: blok başına gradyan checkpointing (yalnız eğitimde)."""
    import torch
    from torch.utils.checkpoint import checkpoint
    for blk in m.blocks[L0:]:
        t = checkpoint(blk, t, use_reentrant=False) if (ckpt and torch.is_grad_enabled()) else blk(t)
    return m.forward_head(m.norm(t))


def onbellek_olustur(m, model: str, L0: int, D: dict, yol: dict, hazirla, cihaz, bs: int, alt_kume,
                     veri_bilgi: dict, agirlik: dict, run_dir: Path, log=print) -> dict:
    import torch
    OZN_DIZIN.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(yol["kilit"]), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()} {datetime.now().isoformat()}".encode())
        os.close(fd)
    except FileExistsError:
        raise SystemExit(f"HATA: {yol['kilit']} var (başka süreç yazıyor ya da yetim kilit; elle sil)")
    try:
        m.eval()
        if cihaz.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        toplam = sum(len(D[b]["y"]) for b in ("train", "val"))
        islenen, son_p, maks_abs, sureler = 0, 0, 0.0, {}
        T, Dm = None, None
        for b in ("train", "val"):
            x_u8 = D[b]["x"]
            n = len(D[b]["y"])
            tmp = yol[b].with_name(yol[b].stem + ".tmp.npy")
            mm = None
            t_b = time.time()
            with torch.no_grad():
                for i in range(0, n, bs):
                    t = donuk_ileri(m, hazirla(torch.from_numpy(x_u8[i:i + bs])), L0)
                    ma = float(t.abs().max())
                    if not math.isfinite(ma) or ma >= 65504.0:
                        raise SystemExit(f"HATA: token değeri fp16 aralığı dışında / sonlu değil (maks|x| {ma})")
                    maks_abs = max(maks_abs, ma)
                    if mm is None:
                        T, Dm = int(t.shape[1]), int(t.shape[2])
                        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(n, T, Dm))
                    mm[i:i + len(t)] = t.half().cpu().numpy()
                    islenen += len(t)
                    if islenen - son_p >= 500 or islenen == toplam:
                        son_p = islenen
                        g = time.time() - t0
                        progress_yaz(run_dir, adim=0, asama_ad="onbellek", bolum=b, islenen=islenen, toplam=toplam,
                                     img_s=round(islenen / max(g, 1e-9), 2), gecen_sn=round(g, 1))
                        log(f"[vit] önbellek {islenen}/{toplam} ({islenen / max(g, 1e-9):.1f} img/s)")
            mm.flush()
            del mm
            os.replace(tmp, yol[b])
            sureler[b] = round(time.time() - t_b, 2)
        sure = round(time.time() - t0, 2)
        import timm
        meta = {
            "model": MODELLER[model][0], "model_kisa": MODELLER[model][1], "timm": timm.__version__,
            "torch": torch.__version__, "agirlik": agirlik, "donuk_blok": L0,
            "icerik": f"patch_embed → pos_embed → blocks[:{L0}] çıkışı (son LayerNorm ÖNCESİ), CLS + patch token'ları",
            "sekil": {"train": [len(D["train"]["y"]), T, Dm], "val": [len(D["val"]["y"]), T, Dm]},
            "dtype": "float16", "hesap_hassasiyet": "fp32 (TF32 kapalı)", "giris_px": GIRIS_PX,
            "normalizasyon": {"ort": IMAGENET_ORT, "std": IMAGENET_STD, "olcek": "/255"}, "augment": "yok (tek görünüm)",
            "veri": veri_bilgi, "alt_kume": alt_kume, "n": {b: int(len(D[b]["y"])) for b in ("train", "val")},
            "idx": {b: [int(D[b]["idx"][0]), int(D[b]["idx"][-1])] for b in ("train", "val")},
            "token_maks_abs": maks_abs, "bs": bs, "sure_sn": sure, "sure_bolum_sn": sureler,
            "img_s": round(toplam / max(sure, 1e-9), 2),
            "bayt": int(sum(yol[b].stat().st_size for b in ("train", "val"))),
            "tarih": datetime.now().isoformat(timespec="seconds"),
        }
        atomik_json(yol["meta"], meta)
        log(f"[vit] önbellek yazıldı: {sure}s ({meta['img_s']} img/s), {meta['bayt'] / 2**30:.2f} GiB, maks|x| {maks_abs:.1f}")
        return meta
    finally:
        try:
            yol["kilit"].unlink()
        except OSError:
            pass


def onbellek_dogrula(meta: dict, model: str, L0: int, D: dict, veri_bilgi: dict):
    hatalar = []
    if meta.get("model") != MODELLER[model][0]:
        hatalar.append(f"model {meta.get('model')}")
    if meta.get("donuk_blok") != L0:
        hatalar.append(f"donuk_blok {meta.get('donuk_blok')}")
    if (meta.get("veri") or {}).get("md5") != veri_bilgi["md5"]:
        hatalar.append("girdi npz md5")
    for b in ("train", "val"):
        if meta["n"].get(b) != len(D[b]["y"]):
            hatalar.append(f"n_{b}")
    if hatalar:
        raise SystemExit(f"HATA: önbellek meta uyuşmazlığı ({', '.join(hatalar)}) — --zorla-onbellek ile yeniden yaz")


def etiket_yukle(alt_kume=None, n_val_goruntu: int = 8) -> dict:
    """V.yukle ile aynı sözleşme (resmî split, ilk N, etiket (N,) int64, idx), ama train görüntüsü okunmaz;
    val'den yalnız ilk n_val_goruntu görüntü (önbellek tutarlılık denetimi). Test anahtarına dokunulmaz (M4)."""
    out = {}
    with np.load(V.npz_yolu(GIRIS_PX)) as z:
        for b in ("train", "val"):
            y = z[f"{b}_labels"].reshape(-1).astype(np.int64)
            idx = np.arange(len(y), dtype=np.int64)
            if alt_kume:
                y, idx = y[:alt_kume], idx[:alt_kume]
            out[b] = {"y": y, "idx": idx, "x": None}
        out["val"]["x"] = np.ascontiguousarray(z["val_images"][:min(n_val_goruntu, len(out["val"]["y"]))])
    return out


# ── ana ────────────────────────────────────────────────────────────────────────────────────
def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--model", choices=["s", "b"], default="b")
    p.add_argument("--mod", choices=["kismi", "tam"], required=True)
    p.add_argument("--son-k", type=int, default=None, help="kısmi: eğitilen son blok sayısı (1..derinlik−1)")
    p.add_argument("--epoch", type=int, default=15)
    p.add_argument("--bs", type=int, default=64, help="etkin batch (gradyan birikimiyle)")
    p.add_argument("--mikro-bs", type=int, default=None, help="GPU'ya giren alt batch (varsayılan kısmi 32, tam 16)")
    p.add_argument("--lr-bas", type=float, default=1e-3)
    p.add_argument("--lr-govde", type=float, default=1e-5)
    p.add_argument("--katman-azalma", type=float, default=1.0, help="LLRD çarpanı d (1 = kapalı)")
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--isinma-epoch", type=float, default=0.0)
    p.add_argument("--drop-path", type=float, default=0.0)
    p.add_argument("--grad-ckpt", type=int, choices=[0, 1], default=None, help="varsayılan: tam 1, kısmi 0")
    p.add_argument("--fused-attn", type=int, choices=[0, 1], default=1)
    p.add_argument("--val-bs", type=int, default=64)
    p.add_argument("--onbellek-bs", type=int, default=32)
    p.add_argument("--onbellek-ram", action="store_true", help="kısmi: token önbelleğini memmap yerine RAM'e yükle")
    p.add_argument("--zorla-onbellek", action="store_true")
    p.add_argument("--sadece-onbellek", action="store_true", help="kısmi: yalnız önbelleği yaz/doğrula, eğitme")
    p.add_argument("--alt-kume", type=int, default=None, help="hata ayıklama: her bölümün ilk N örneği")
    p.add_argument("--azami-adim", type=int, default=None, help="ölçüm/hata ayıklama: epoch başına en çok optimizasyon adımı")
    p.add_argument("--cihaz", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--cpu-izin", action="store_true")
    p.add_argument("--threads", type=int, default=None)
    a = p.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import timm
    from timm.layers import set_fused_attn
    if a.threads:
        torch.set_num_threads(a.threads)
    out = Path(a.out).resolve()
    run_dir = out.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    cihaz, cihaz_ad = cihaz_sec(a.cihaz, a.cpu_izin)
    seed_bagla(a.seed)
    if cihaz.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    kismi = a.mod == "kismi"
    grad_ckpt = bool(a.grad_ckpt if a.grad_ckpt is not None else (0 if kismi else 1))
    mikro = a.mikro_bs or (32 if kismi else 16)
    mikro = min(mikro, a.bs)

    # ── model ──
    set_fused_attn(bool(a.fused_attn))
    ad, kisa = MODELLER[a.model][0], MODELLER[a.model][1]
    t_m = time.time()
    m = timm.create_model(ad, pretrained=True, img_size=GIRIS_PX, num_classes=0, drop_path_rate=a.drop_path)
    derinlik = len(m.blocks)
    Dm = m.num_features
    torch.manual_seed(a.seed)  # baş başlatması seed'e bağlı (H02 fc.reset_parameters() mantığı)
    m.head = nn.Linear(Dm, V.SINIF_SAYISI)
    m.num_classes = V.SINIF_SAYISI
    agirlik = _agirlik_bilgisi(m)
    if kismi:
        if a.son_k is None or not (1 <= a.son_k < derinlik):
            p.error(f"--mod kismi için 1 ≤ --son-k < {derinlik}")
        L0 = derinlik - a.son_k
    else:
        L0 = 0
    n_param = int(sum(q.numel() for q in m.parameters()))
    print(f"[vit] {ad} ({n_param:,} param, derinlik {derinlik}, D {Dm}, {time.time() - t_m:.1f}s) · mod {a.mod}"
          f"{f' son-k {a.son_k} (donuk {L0})' if kismi else ''} · cihaz {cihaz_ad} · sha256 {str(agirlik.get('sha256'))[:12]}")
    ort = torch.tensor(IMAGENET_ORT, device=cihaz).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=cihaz).view(1, 3, 1, 1)

    def hazirla(xb_u8):
        # .contiguous(): permute sonrası NHWC-adımlı tensör deterministik cuDNN'de yavaş yola düşüyor (ince_ayar.py dersi)
        x = xb_u8.to(cihaz, non_blocking=True).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
        return ((x - ort) / std).contiguous()

    # ── veri ──
    veri_bilgi = V.npz_dogrula(GIRIS_PX)
    t_yuk = time.time()
    on_meta, onbellek_yazma_sn = None, None
    if kismi:
        yol = onbellek_yollari(onbellek_kok(a.model, L0, a.alt_kume))
        var = all(yol[b].exists() for b in ("train", "val", "meta"))
        if a.zorla_onbellek or not var:
            D = V.yukle(GIRIS_PX, ("train", "val"), alt_kume=a.alt_kume)
            m.to(cihaz)
            on_meta = onbellek_olustur(m, a.model, L0, D, yol, hazirla, cihaz, a.onbellek_bs, a.alt_kume,
                                       veri_bilgi, agirlik, run_dir)
            onbellek_yazma_sn = on_meta["sure_sn"]
        else:
            on_meta = json.loads(yol["meta"].read_text(encoding="utf-8"))
            # önbellek varken train görüntüleri RAM'e alınmaz (≈ 1 GB): yalnız etiket/idx + tutarlılık için 8 val görüntüsü
            D = etiket_yukle(a.alt_kume)
            print(f"[vit] önbellek mevcut: {yol['meta'].name} ({on_meta['bayt'] / 2**30:.2f} GiB)")
        onbellek_dogrula(on_meta, a.model, L0, D, veri_bilgi)
        mmode = None if a.onbellek_ram else "r"
        tok = {b: np.load(yol[b], mmap_mode=mmode) for b in ("train", "val")}
        for b in ("train", "val"):
            if tok[b].shape[0] != len(D[b]["y"]):
                raise SystemExit(f"HATA: önbellek {b} satır sayısı {tok[b].shape[0]} ≠ {len(D[b]['y'])}")
        # tutarlılık: tam ağ (görüntüden, fp32) ↔ önbellek (fp16) → normlanmış CLS
        m.to(cihaz).eval()
        with torch.no_grad():
            nk = min(8, len(D["val"]["y"]))
            t32 = donuk_ileri(m, hazirla(torch.from_numpy(D["val"]["x"][:nk])), L0)
            def cls_(t):
                for blk in m.blocks[L0:]:
                    t = blk(t)
                return m.norm(t)[:, 0]
            c_tam = cls_(t32)
            c_on = cls_(torch.from_numpy(np.array(tok["val"][:nk])).to(cihaz).float())
            c_16 = cls_(t32.half().float())
            tutarlilik = float((c_tam - c_on).abs().max())
            fp16_fark = float((c_tam - c_16).abs().max())
            cls_olcek = float(c_tam.abs().max())
        print(f"[vit] önbellek tutarlılık maks|ΔCLS| {tutarlilik:.3g} (fp16 yuvarlama payı {fp16_fark:.3g}; |CLS| maks {cls_olcek:.2f})")
        if tutarlilik > max(10 * fp16_fark, 1e-3) + 1e-6:
            raise SystemExit("HATA: önbellek tam ağla tutarsız (yanlış blok derinliği / bozuk dosya?)")
        # donuk kısım GPU'dan iner (bellek); eğitilen kısım GPU'da kalır
        for mod_ in (m.patch_embed, *m.blocks[:L0]):
            mod_.to("cpu")
        m.cls_token.data = m.cls_token.data.cpu()
        m.pos_embed.data = m.pos_embed.data.cpu()
        if cihaz.type == "cuda":
            torch.cuda.empty_cache()
        D["train"]["x"] = D["val"]["x"] = None  # görüntüler artık gerekmez (RAM)
        if a.sadece_onbellek:
            mt = {"onbellek_yazma_sn": onbellek_yazma_sn, "onbellek_yazildi": int(onbellek_yazma_sn is not None),
                  "onbellek_img_s": on_meta["img_s"], "onbellek_bayt": on_meta["bayt"], "donuk_blok": L0,
                  "onbellek_tutarlilik_maks_fark": tutarlilik, "fp16_cls_maks_fark": fp16_fark,
                  "token_maks_abs": on_meta["token_maks_abs"], "cihaz": cihaz_ad, "seed": a.seed}
            if cihaz.type == "cuda":
                mt["tepe_gpu_bellek_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            mt.update(ram_tepe_mb())
            if a.alt_kume:
                mt["hata_ayiklama"] = 1
            metrics_yaz(out, mt)
            print(f"[vit] BİTTİ (yalnız önbellek) → {out}")
            return 0
    else:
        if a.sadece_onbellek:
            p.error("--sadece-onbellek yalnız --mod kismi ile")
        D = V.yukle(GIRIS_PX, ("train", "val"), alt_kume=a.alt_kume)
        m.to(cihaz)
        tutarlilik = fp16_fark = None
    ytr = torch.from_numpy(D["train"]["y"])
    yva = D["val"]["y"]
    N = len(ytr)
    print(f"[vit] veri {veri_bilgi['npz']} md5 OK · train {N} val {len(yva)} ({time.time() - t_yuk:.1f}s hazırlık)")

    # ── eğitilen parametreler ve gruplar ──
    if kismi:
        for q in m.parameters():
            q.requires_grad_(False)
        egitilen_moduller = [*m.blocks[L0:], m.norm, m.head]
        for mod_ in egitilen_moduller:
            for q in mod_.parameters():
                q.requires_grad_(True)
    if grad_ckpt:
        m.set_grad_checkpointing(True)
    nwd = set(m.no_weight_decay()) if hasattr(m, "no_weight_decay") else set()

    def kademe(isim: str) -> int:
        """LLRD kademesi: gömme 0, blok i → i+1, norm/baş → derinlik+1."""
        if isim.startswith(("patch_embed", "cls_token", "pos_embed", "reg_token", "norm_pre")):
            return 0
        if isim.startswith("blocks."):
            return int(isim.split(".")[1]) + 1
        return derinlik + 1

    gruplar = {}
    for isim, q in m.named_parameters():
        if not q.requires_grad:
            continue
        bas = isim.startswith("head.")
        k = kademe(isim)
        lr = a.lr_bas if bas else a.lr_govde * (a.katman_azalma ** max(derinlik - k, 0))
        wd = 0.0 if (q.ndim < 2 or isim in nwd) else a.wd
        anahtar = ("bas" if bas else f"k{k}", wd)
        g_ = gruplar.setdefault(anahtar, {"params": [], "lr": lr, "weight_decay": wd, "ad": f"{anahtar[0]}_wd{wd:g}"})
        g_["params"].append(q)
    param_gruplari = list(gruplar.values())
    n_egitilen = int(sum(q.numel() for g_ in param_gruplari for q in g_["params"]))
    opt = torch.optim.AdamW(param_gruplari, lr=a.lr_govde, weight_decay=a.wd)
    adim_epoch = math.ceil(N / a.bs)
    if a.azami_adim:
        adim_epoch = min(adim_epoch, a.azami_adim)
    T_top = a.epoch * adim_epoch
    T_isin = int(round(a.isinma_epoch * adim_epoch))

    def lr_carpan(t):
        if t < T_isin:
            return (t + 1) / T_isin
        return 0.5 * (1.0 + math.cos(math.pi * min(t - T_isin, T_top - T_isin) / max(T_top - T_isin, 1)))

    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_carpan)
    i_bas = next(i_ for i_, g_ in enumerate(param_gruplari) if g_["ad"].startswith("bas"))
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    print(f"[vit] eğitilen {n_egitilen:,} param ({len(param_gruplari)} grup) · etkin bs {a.bs} = mikro {mikro} × "
          f"≤{math.ceil(a.bs / mikro)} · grad_ckpt {int(grad_ckpt)} · adım/epoch {adim_epoch} · toplam {T_top}")

    def girdi(bolum: str, j: np.ndarray):
        if kismi:
            return torch.from_numpy(np.ascontiguousarray(tok[bolum][j])).to(cihaz, non_blocking=True).float()
        return hazirla(torch.from_numpy(D[bolum]["x"][j]))

    def ileri(x):
        if kismi:
            return egitilen_ileri(m, x, L0, grad_ckpt)
        return m(x)

    def egitim_modu(acik: bool):
        if kismi:
            for mod_ in egitilen_moduller:
                mod_.train(acik)
        else:
            m.train(acik)

    def val_olasilik():
        egitim_modu(False)
        ps = []
        nv = len(yva)
        with torch.no_grad():
            for i in range(0, nv, a.val_bs):
                j = np.arange(i, min(i + a.val_bs, nv))
                ps.append(torch.softmax(ileri(girdi("val", j)), dim=1).float().cpu().numpy())
        egitim_modu(True)
        return np.concatenate(ps).astype(np.float32)

    gecmis = []
    en = {"acc": -1.0, "auc": -1.0, "epoch": 0, "p": None, "state": None}
    en_auc = {"auc": 0.0, "epoch": 0, "acc": None, "p": None}
    ilk_adim_loss, ilk_epoch_loss = None, None
    kararli = {"t": None, "n": 0}
    t0 = time.time()
    t_prog = time.time()
    egitim_modu(True)
    for ep in range(1, a.epoch + 1):
        t_ep = time.time()
        perm = torch.randperm(N, generator=g).numpy()
        top, n_ornek, adim = 0.0, 0, 0
        for i in range(0, N, a.bs):
            j_all = np.sort(perm[i:i + a.bs])  # batch bir kümedir; sıralı okuma memmap yerelliği için
            opt.zero_grad(set_to_none=True)
            loss_top = 0.0
            for c in range(0, len(j_all), mikro):
                j = j_all[c:c + mikro]
                yb = ytr[torch.from_numpy(j)].to(cihaz, non_blocking=True)
                loss = F.cross_entropy(ileri(girdi("train", j)), yb, reduction="sum") / len(j_all)
                loss.backward()
                loss_top += float(loss.detach())
            opt.step()
            sch.step()
            if ilk_adim_loss is None:
                ilk_adim_loss = loss_top
            top += loss_top * len(j_all)
            n_ornek += len(j_all)
            adim += 1
            if ep == 1 and adim == 5:
                if cihaz.type == "cuda":
                    torch.cuda.synchronize()
                kararli["t"], kararli["n"] = time.time(), n_ornek
            if not math.isfinite(loss_top):
                progress_yaz(run_dir, adim=ep - 1, epoch_toplam=a.epoch, loss=loss_top, gecen_sn=round(time.time() - t0, 1))
                raise SystemExit("HATA: kayıp sonlu değil (ıraksama)")
            if time.time() - t_prog > PROGRESS_ARALIK_SN:
                t_prog = time.time()
                pr = {"adim": ep - 1, "epoch_toplam": a.epoch, "asama_ad": "egitim", "epoch": ep, "adim_ici": adim,
                      "adim_epoch": adim_epoch, "loss": top / max(n_ornek, 1), "gecen_sn": round(time.time() - t0, 1)}
                if en["acc"] >= 0:
                    pr["val_acc_eniyi"] = en["acc"]
                progress_yaz(run_dir, **pr)
                print(f"[vit]   epoch {ep} adım {adim}/{adim_epoch} loss {top / max(n_ornek, 1):.4f} "
                      f"({n_ornek / max(time.time() - t_ep, 1e-9):.1f} img/s)", flush=True)
            if a.azami_adim and adim >= a.azami_adim:
                break
        if cihaz.type == "cuda":
            torch.cuda.synchronize()
        egitim_sn = time.time() - t_ep
        if ep == 1:
            kararli["img_s"] = ((n_ornek - kararli["n"]) / max(time.time() - kararli["t"], 1e-9)
                                if kararli["t"] and n_ornek > kararli["n"] else n_ornek / max(egitim_sn, 1e-9))
            kararli["n_olculen"] = n_ornek - kararli["n"] if kararli["t"] else n_ornek
        tr_loss = top / max(n_ornek, 1)
        if ilk_epoch_loss is None:
            ilk_epoch_loss = tr_loss
        t_val = time.time()
        pv = val_olasilik()
        val_sn = time.time() - t_val
        acc, (auc, _) = _acc(yva, pv), _auc(yva, pv)
        den = dengeli_acc(yva, pv.argmax(axis=1))
        if acc > en["acc"] or (acc == en["acc"] and auc > en["auc"]):
            en = {"acc": acc, "auc": auc, "epoch": ep, "p": pv,
                  "state": copy.deepcopy({k: v.detach().cpu() for k, v in m.state_dict().items()})}
        if auc > en_auc["auc"]:
            en_auc = {"auc": auc, "epoch": ep, "acc": acc, "p": pv}
        satir = {"epoch": ep, "train_loss": tr_loss, "val_acc": acc, "val_auc": auc, "val_dengeli_acc": den,
                 "egitim_sn": round(egitim_sn, 2), "val_sn": round(val_sn, 2), "n_ornek": n_ornek, "adim": adim,
                 "epoch_sn": round(time.time() - t_ep, 2), "lr_bas": opt.param_groups[i_bas]["lr"]}
        gecmis.append(satir)
        progress_yaz(run_dir, adim=ep, epoch_toplam=a.epoch, loss=tr_loss, val_acc=acc, val_auc=auc,
                     val_dengeli_acc=den, val_acc_eniyi=en["acc"], epoch_sn=satir["epoch_sn"], gecen_sn=round(time.time() - t0, 1))
        t_prog = time.time()
        print(f"[vit] epoch {ep}/{a.epoch} loss={tr_loss:.4f} val_acc={acc:.4f} val_auc={auc:.4f} "
              f"(eğitim {egitim_sn:.1f}s + val {val_sn:.1f}s; {n_ornek / max(egitim_sn, 1e-9):.1f} img/s)", flush=True)
    sure = round(time.time() - t0, 2)
    p_son = pv

    # ── kayıt ──
    torch.save(en["state"], run_dir / "best_model.pth")
    atomik_json(run_dir / "best_model_meta.json", {
        "timm_ad": ad, "img_size": GIRIS_PX, "num_classes_omurga": 0, "bas": f"Linear({Dm},{V.SINIF_SAYISI})",
        "mod": a.mod, "donuk_blok": L0, "not": "state_dict TAM ağdır; kısmi modda blocks[:donuk_blok] önağırlıkla aynı",
        "yukleme": "m = timm.create_model(timm_ad, pretrained=False, img_size=224, num_classes=0); "
                   "m.head = nn.Linear(D, 7); m.load_state_dict(torch.load(best_model.pth))"})
    tahmin_kaydet(run_dir, en["p"], yva)
    np.save(run_dir / "tahmin_val_son.npy", p_son)
    np.save(run_dir / "tahmin_val_aucsecim.npy", np.asarray(en_auc["p"], dtype=np.float32))
    atomik_json(run_dir / "gecmis.json", gecmis)
    mt = degerlendir(yva, en["p"], on="val", val_idx=D["val"]["idx"])
    alt_grup = mt.pop("alt_grup")
    m_auc = degerlendir(yva, en_auc["p"], on="val", val_idx=D["val"]["idx"], bootstrap=False)
    aucsecim_ek = {f"{k}_aucsecim": m_auc[k] for k in ("val_auc", "val_dengeli_acc", "val_makro_f1", "val_ece",
                   "val_tahmin_nv_payi", "val_acc_molemax_disi", "val_sizinti_tabakali_SN", "val_auc_mel", "val_auc_bkl")
                   if k in m_auc}
    if abs(m_auc["val_acc"] - en_auc["acc"]) > 1e-12:
        raise RuntimeError("val_acc_aucsecim tutarsız (kaydedilen olasılık ≠ seçilen epoch)")
    metrics = {
        **mt,
        "en_iyi_epoch": en["epoch"], "epoch_sayisi": a.epoch, "val_acc_aucsecim": en_auc["acc"],
        "en_iyi_epoch_aucsecim": en_auc["epoch"], **aucsecim_ek, "val_acc_son": gecmis[-1]["val_acc"],
        "val_auc_son": gecmis[-1]["val_auc"],
        "ilk_adim_loss": ilk_adim_loss, "ilk_epoch_loss": ilk_epoch_loss, "train_loss_son": gecmis[-1]["train_loss"],
        "epoch_sure_ort_sn": float(np.mean([g_["epoch_sn"] for g_ in gecmis])),
        "epoch_egitim_sn_ort": float(np.mean([g_["egitim_sn"] for g_ in gecmis])),
        "epoch_val_sn_ort": float(np.mean([g_["val_sn"] for g_ in gecmis])),
        "egitim_img_s_kararli": float(kararli.get("img_s", float("nan"))),
        "egitim_img_s_olculen_ornek": int(kararli.get("n_olculen", 0)),
        "egitim_sure_sn": sure, "sure_sn": sure, "param_sayisi": n_param, "param_egitilen": n_egitilen,
        "giris_cozunurluk": GIRIS_PX, "n_train": int(N), "seed": a.seed, "cihaz": cihaz_ad, "alt_grup": alt_grup,
        "mod_kismi": int(kismi), "son_k": int(a.son_k or derinlik), "donuk_blok": L0, "model_boyut": a.model,
        "bs_etkin": a.bs, "mikro_bs": mikro, "grad_ckpt": int(grad_ckpt), "fused_attn": a.fused_attn,
        "lr_bas": a.lr_bas, "lr_govde": a.lr_govde, "katman_azalma": a.katman_azalma, "adim_epoch": adim_epoch,
        "val_dengeli_acc_eniyi_epoch": float(max(g_["val_dengeli_acc"] for g_ in gecmis)),
        "val_dengeli_acc_son": gecmis[-1]["val_dengeli_acc"],
    }
    if kismi:
        metrics.update({"onbellek_yazildi": int(onbellek_yazma_sn is not None), "onbellek_bayt": on_meta["bayt"],
                        "onbellek_img_s": on_meta["img_s"], "onbellek_tutarlilik_maks_fark": tutarlilik,
                        "fp16_cls_maks_fark": fp16_fark, "onbellek_ram": int(a.onbellek_ram)})
        if onbellek_yazma_sn is not None:
            metrics["onbellek_yazma_sn"] = onbellek_yazma_sn
    if cihaz.type == "cuda":
        metrics["tepe_gpu_bellek_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        metrics["tepe_gpu_ayrilan_mb"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    metrics.update(ram_tepe_mb())
    if a.alt_kume or a.azami_adim:
        metrics["hata_ayiklama"] = 1
    config_ekle(run_dir, {
        "donanim": {**donanim_bilgisi(), "cihaz": cihaz_ad}, "veri": veri_bilgi,
        "model": {"ad": ad, "kisa": kisa, "img_size": GIRIS_PX, "agirlik": agirlik, "param_sayisi": n_param,
                  "param_egitilen": n_egitilen, "bas": f"Linear({Dm},{V.SINIF_SAYISI}) CLS (son norm sonrası)",
                  "timm": timm.__version__},
        "protokol": {"mod": a.mod, "son_k": a.son_k, "donuk_blok": L0,
                     "onbellek": (str(onbellek_kok(a.model, L0, a.alt_kume)) + "_{train,val}.npy (fp16)") if kismi else None,
                     "optimizer": "AdamW", "lr_bas": a.lr_bas, "lr_govde": a.lr_govde, "katman_azalma": a.katman_azalma,
                     "weight_decay": a.wd, "wd_muaf": "1-B parametreler + no_weight_decay()",
                     "takvim": f"adım başına kosinüs (T={T_top}), ısınma {T_isin} adım", "bs_etkin": a.bs,
                     "mikro_bs": mikro, "grad_ckpt": grad_ckpt, "fused_attn": a.fused_attn, "drop_path": a.drop_path,
                     "epoch": a.epoch, "kayip": "CrossEntropy (Σ / etkin batch)", "augment": "yok",
                     "normalizasyon": "ImageNet (GPU'da)", "hassasiyet": "fp32 (TF32 kapalı; önbellek fp16 saklama)",
                     "secim": "val ACC kesin artış (eşitlikte val AUC); ek: val AUC seçimi (rakip kuralı)",
                     "seed": a.seed, "cudnn_deterministic": True, "alt_kume": a.alt_kume, "azami_adim": a.azami_adim,
                     "param_gruplari": [{"ad": g_["ad"], "lr": g_["lr"], "wd": g_["weight_decay"],
                                         "n": int(sum(q.numel() for q in g_["params"]))} for g_ in param_gruplari]},
    })
    metrics_yaz(out, metrics)
    print(f"[vit] BİTTİ {sure}s · en iyi epoch {en['epoch']} · val_acc {mt['val_acc']:.4f} val_auc {mt['val_auc']:.4f} "
          f"· tepe GPU {metrics.get('tepe_gpu_bellek_mb')} MB · RAM ws {metrics.get('tepe_ram_ws_mb')} / özel "
          f"{metrics.get('tepe_ram_ozel_mb')} MB → {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
