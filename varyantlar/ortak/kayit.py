# -*- coding: utf-8 -*-
"""
ortak/kayit.py — metrics.json / progress.json / tahmin_*.npy / config.json eki + seed bağlama + cihaz.

Sözleşme (deney-sozlesmesi.md "Standart koşum sözleşmesi", scripts/rakip_kos.py ile aynı biçim):
- metrics.json: DÜZ sözlük; iç içe yalnız "alt_grup". Sayısal değerler + "cihaz" (str).
  K4'te `test_` önekli anahtar YAZILAMAZ (M4) → metrics_yaz() hata verir (test_izin=True yalnız K5 '-test').
- progress.json: {"adim", "loss", "val_acc", "val_acc_eniyi", "gecen_sn", ...} — kosum.py --kural bunu okur.
  Test değeri yazılmaz.
- tahmin_val.npy: (N,7) float32 softmax; etiket_val.npy: (N,) int64.
- config.json: kosum.py'nin yazdığı dosyaya `donanim`/`veri`/`model`/`protokol` alanları eklenir (R1/R6).
"""
import json
import math
import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np


def atomik_json(yol: Path, veri):
    yol = Path(yol)
    tmp = yol.with_suffix(yol.suffix + ".tmp")
    tmp.write_text(json.dumps(_py(veri), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, yol)


def _py(v):
    """numpy türlerini JSON'a uygun Python türlerine çevirir."""
    if isinstance(v, dict):
        return {str(k): _py(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_py(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def seed_bagla(seed: int, deterministik: bool = True):
    """rakip_kos.py ile aynı ayar: random/numpy/torch/torch.cuda + cudnn.deterministic, benchmark=False."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministik
    torch.backends.cudnn.benchmark = False
    # FP32 zorunlu (GTX 1650; ayrıca Ampere+'da TF32'yi kapatarak donanımlar arası tutarlılık)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def cihaz_sec(istek: str, cpu_izin: bool):
    """istek: 'cuda' | 'cpu'. CUDA istenip yoksa → SystemExit(97) (sessiz CPU düşüşü yasak).
    CPU yalnız --cpu-izin ile (hata ayıklama / mini doğrulama; ölçüm koşumu değildir)."""
    import torch
    if istek == "cuda":
        if not torch.cuda.is_available():
            print("HATA: CUDA yok — sessiz CPU düşüşü yasak (rc=97)")
            raise SystemExit(97)
        return torch.device("cuda:0"), f"cuda:0 / {torch.cuda.get_device_name(0)}"
    if not cpu_izin:
        print("HATA: --cihaz cpu yalnız --cpu-izin ile (ölçüm koşumu GPU'da yapılır)")
        raise SystemExit(97)
    return torch.device("cpu"), "cpu"


def donanim_bilgisi() -> dict:
    import torch
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        smi = f"nvidia-smi yok: {e}"
    b = {"nvidia_smi": smi, "torch": torch.__version__, "cuda": torch.version.cuda,
         "cudnn": torch.backends.cudnn.version(), "numpy": np.__version__,
         "python": platform.python_version(), "platform": platform.platform(), "makine": platform.node(),
         "torch_threads": torch.get_num_threads()}
    for paket in ("torchvision", "timm", "sklearn", "medmnist"):
        try:
            b[paket] = __import__(paket).__version__
        except Exception:  # noqa: BLE001
            pass
    return b


def config_ekle(run_dir: Path, ek: dict):
    """kosum.py'nin config.json'ına alan ekler (yoksa ayrı varyant_bilgi.json yeter)."""
    atomik_json(Path(run_dir) / "varyant_bilgi.json", ek)
    cfg = Path(run_dir) / "config.json"
    if cfg.exists():
        try:
            c = json.loads(cfg.read_text(encoding="utf-8"))
            c.update(ek)
            atomik_json(cfg, c)
        except Exception as e:  # noqa: BLE001
            print(f"[kayit] UYARI: config.json'a ek yazılamadı: {e}")


def progress_yaz(run_dir: Path, **kv):
    atomik_json(Path(run_dir) / "progress.json", kv)


def metrics_yaz(out: Path, metrics: dict, test_izin: bool = False) -> list:
    """Doğrular ve yazar. Dönüş: sonlu olmayan anahtarların listesi (uyarı)."""
    m = _py(metrics)
    for k, v in m.items():
        if k.startswith("test_") and not test_izin:
            raise RuntimeError(f"M4 ihlali: K4 metrics.json'da test anahtarı ({k}) yazılamaz")
        if isinstance(v, dict) and k != "alt_grup":
            raise RuntimeError(f"metrics.json düz olmalı (iç içe yalnız alt_grup): {k}")
        if isinstance(v, (list, tuple)):
            raise RuntimeError(f"metrics.json'da liste yok: {k}")
    sonlu_degil = [k for k, v in m.items() if isinstance(v, float) and not math.isfinite(v)]
    sonlu_degil += [f"alt_grup.{k}" for k, v in (m.get("alt_grup") or {}).items()
                    if isinstance(v, float) and not math.isfinite(v)]
    for k in sonlu_degil:
        print(f"[kayit] UYARI: {k} sonlu değil")
    atomik_json(out, m)
    return sonlu_degil


def tahmin_kaydet(run_dir: Path, p_val: np.ndarray, y_val: np.ndarray, ek_ad: str = ""):
    np.save(Path(run_dir) / f"tahmin_val{ek_ad}.npy", np.asarray(p_val, dtype=np.float32))
    np.save(Path(run_dir) / "etiket_val.npy", np.asarray(y_val, dtype=np.int64))
