#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rakip_kos.py — K3 metrik köprüsü: MedMNIST resmî ResNet-18 (28 px, sıfırdan) eğitim betiğini
(`rakip/experiments/MedMNIST2D/train_and_eval_pytorch.py` @ 70b6b3a) DEĞİŞTİRMEDEN çağırır.

Rakip koduna dokunulmaz (R2). Köprü yalnız şunları yapar (import + monkeypatch, rakip dosyaları aynen):
  (a) seed bağlama: random / numpy / torch / torch.cuda + cudnn.deterministic=True, benchmark=False
      (resmî kodda seed yok — R5). DataLoader shuffle torch global RNG'sini kullanır (num_workers=0).
  (b) veri kökü: resmî kod `root` vermez (varsayılan ~/.medmnist). `medmnist.DermaMNIST` ve
      `medmnist.Evaluator` alt sınıflarla değiştirilir; tek fark `root=--veri-kok`. download=False.
  (c) kayıt: resmî `train()` ve `test()` fonksiyonları sarmalanır (girdi/çıktı aynen geçer); her
      epoch'un train loss'u ve val/test [loss, AUC, ACC] değerleri ile Evaluator'a giden olasılıklar
      (softmax) kaydedilir. AUC/ACC = medmnist.Evaluator (makro OvR AUC; argmax ACC) — resmî tanım.
  (d) {RUN_DIR}/progress.json her epoch: {"adim","loss","val_acc","val_auc","val_acc_eniyi",...}
  (e) {RUN_DIR}/metrics.json (düz sözlük, sayısal + cihaz + alt_grup).

Model seçimi RESMÎDİR: resmî kod her epoch val AUC'ye bakar, `cur_auc > best_auc` ise
`best_model = deepcopy(model)` (70b6b3a'da referans hatası düzeltilmiş hâli). Eğitim sonunda resmî
kod best_model'i train/val/test'te değerlendirir → `val_*` ve `test_*` buradan. Ayrıca son epoch'un
(100.) değerleri `*_son` anahtarlarıyla yazılır (2023-05 öncesi `best_model = model` hatası: makale
sayıları muhtemelen son epoch — notlar/literatur-cipa.md §1.4).

TEST ANAHTARLARI: yalnız bu rakip köprüsünde (şef kararı, orkestrasyon_durumu.md: replikasyon test'te
bir kez, baseline test sayıları dondurulur). K4 varyant köprüleri test yazmaz (M4).

Çıktılar ({RUN_DIR} = --out'un klasörü):
  metrics.json · progress.json · gecmis.json (epoch geçmişi) · rakip_bilgi.json (commit, sürümler,
  veri hash'i, donanım; ayrıca config.json'a `donanim`/`veri`/`rakip` alanları eklenir) ·
  tahmin_val.npy / tahmin_test.npy (seçilen epoch, softmax, float32) · tahmin_val_son.npy /
  tahmin_test_son.npy (son epoch) · etiket_val.npy / etiket_test.npy · resmi_cikti/ (resmî betiğin
  kendi çıktısı: best_model.pth, standart değerlendirme CSV'leri, Tensorboard, *_log.txt)

Kullanım (WS kökünden; kosum.py cwd'si WS köküdür):
  python scripts/rakip_kos.py --seed 0 --epoch 100 --out {RUN_DIR}/metrics.json
  [--veri-kok data/medmnist] [--size 28] [--batch-size 128] [--cpu-izin]
"""
import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("TQDM_MININTERVAL", "15")  # stdout.log'u boğmasın (yalnız gösterim)

WS = Path(__file__).resolve().parents[1]
RAKIP_DIR = WS / "rakip" / "experiments" / "MedMNIST2D"
RAKIP_COMMIT = "70b6b3a7ad7afddff1df2a3b735235830fbdb142"
# git blob SHA1 (LF normalize) — commit 70b6b3a ağacındaki nesnelerle aynı; .git olmadan da doğrular
RAKIP_BLOB = {
    "train_and_eval_pytorch.py": "09157bd8d6032be12cbf1eea0cfa8ccbd39914cb",
    "models.py": "24579d6612441aa1e1af63f3d246d4e61db822fd",
}
DATA_FLAG = "dermamnist"


def atomik_json(yol: Path, veri):
    tmp = yol.with_suffix(yol.suffix + ".tmp")
    tmp.write_text(json.dumps(veri, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, yol)


def git_blob_sha1(yol: Path, crlf_yap: bool = False) -> str:
    """Yukarı akış blob'u CRLF içerir; ham bayt = blob. crlf_yap: LF'e çevrilmiş kopyalar için yedek."""
    b = yol.read_bytes()
    if crlf_yap:
        b = b.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def dosya_hash(yol: Path):
    h5, h256 = hashlib.md5(), hashlib.sha256()
    with open(yol, "rb") as f:
        for parca in iter(lambda: f.read(1 << 20), b""):
            h5.update(parca); h256.update(parca)
    return h5.hexdigest(), h256.hexdigest()


def rakip_dogrula():
    sonuc = {}
    for ad, beklenen in RAKIP_BLOB.items():
        olcul = git_blob_sha1(RAKIP_DIR / ad)
        if olcul != beklenen:
            olcul = git_blob_sha1(RAKIP_DIR / ad, crlf_yap=True)
        if olcul != beklenen:
            raise SystemExit(f"HATA (R2): rakip dosyası değişmiş: {ad} blob {olcul} ≠ {beklenen} (@{RAKIP_COMMIT[:7]})")
        sonuc[ad] = olcul
    return sonuc


def seed_bagla(seed: int):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True, help="{RUN_DIR}/metrics.json")
    p.add_argument("--epoch", type=int, default=100, help="resmî varsayılan 100 (milestone'lar resmî kodda 0.5E/0.75E)")
    p.add_argument("--size", type=int, default=28, help="resmî rakip protokolü 28 (resize yok)")
    p.add_argument("--batch-size", type=int, default=128, help="resmî varsayılan 128")
    p.add_argument("--veri-kok", default="data/medmnist", help="dermamnist*.npz klasörü; göreli ise WS köküne göre")
    p.add_argument("--cpu-izin", action="store_true", help="yalnız hata ayıklama; ölçüm koşumunda KULLANILMAZ")
    a = p.parse_args()

    out = Path(a.out).resolve()
    run_dir = out.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    veri_kok = Path(a.veri_kok)
    if not veri_kok.is_absolute():
        veri_kok = (WS / veri_kok).resolve()
    if a.size != 28:
        print(f"[rakip_kos] UYARI: size={a.size} — rakip protokolü 28 px; bu koşum replikasyon sayılmaz")

    blob = rakip_dogrula()
    sys.path.insert(0, str(RAKIP_DIR))

    import numpy as np
    import torch
    import medmnist
    from medmnist import INFO

    if not torch.cuda.is_available() and not a.cpu_izin:
        print("[rakip_kos] HATA: CUDA yok — sessiz CPU düşüşü yasak (hesaplama-hedefi.md). Çıkılıyor.")
        return 97
    cihaz = (f"cuda:0 / {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "cpu")
    seed_bagla(a.seed)

    npz_ad = f"{DATA_FLAG}{'' if a.size == 28 else '_' + str(a.size)}.npz"
    npz = veri_kok / npz_ad
    if not npz.exists():
        print(f"[rakip_kos] HATA: veri yok: {npz}")
        return 2
    md5, sha256 = dosya_hash(npz)
    md5_resmi = INFO[DATA_FLAG]["MD5" + ("" if a.size == 28 else f"_{a.size}")]
    if md5 != md5_resmi:
        print(f"[rakip_kos] HATA: MD5 uyuşmuyor {md5} ≠ resmî {md5_resmi}")
        return 2

    # ── (b) veri kökü: yalnız `root` enjekte eden alt sınıflar ──────────────────────────
    _Derma, _Eval = medmnist.DermaMNIST, medmnist.Evaluator
    KOK = str(veri_kok)

    class KokluDermaMNIST(_Derma):
        def __init__(self, *args, **kw):
            kw.setdefault("root", KOK)
            super().__init__(*args, **kw)

    class KayitliEvaluator(_Eval):
        """medmnist.Evaluator ile aynı hesap; yalnız son y_score'u saklar."""
        def __init__(self, flag, split, size=None, root=None):
            super().__init__(flag, split, size=size, root=root or KOK)
            self.son_skor = None

        def evaluate(self, y_score, save_folder=None, run=None):
            self.son_skor = np.array(y_score, dtype=np.float32, copy=True)
            return super().evaluate(y_score, save_folder, run)

    KokluDermaMNIST.__name__ = "DermaMNIST"
    medmnist.DermaMNIST = KokluDermaMNIST
    medmnist.Evaluator = KayitliEvaluator

    import train_and_eval_pytorch as resmi  # rakip modülü (aynen)

    # ── (c) kayıt sarmalayıcıları ─────────────────────────────────────────────────────
    durum = {"epoch_kayit": {}, "gecmis": [], "son": {}, "final": {}, "param": None,
             "train_loss": None, "eniyi_val_acc": -1.0, "t0": None}
    _train, _test = resmi.train, resmi.test

    def train_sarici(model, *args, **kw):
        if durum["param"] is None:
            durum["param"] = int(sum(p_.numel() for p_ in model.parameters()))
        loss = _train(model, *args, **kw)
        durum["train_loss"] = float(loss)
        return loss

    def test_sarici(*args, **kw):
        res = _test(*args, **kw)
        evaluator = args[1] if len(args) > 1 else kw["evaluator"]
        save_folder = args[7] if len(args) > 7 else kw.get("save_folder")
        kayit = {"loss": float(res[0]), "auc": float(res[1]), "acc": float(res[2]), "skor": evaluator.son_skor}
        if save_folder is not None:          # eğitim sonu: resmî best_model değerlendirmesi
            durum["final"][evaluator.split] = kayit
            return res
        durum["epoch_kayit"][evaluator.split] = kayit
        if evaluator.split == "test":        # resmî döngüde sıra: train → val → test → epoch bitti
            ep = len(durum["gecmis"]) + 1
            v, t, tr = durum["epoch_kayit"]["val"], kayit, durum["epoch_kayit"].get("train", {})
            satir = {"epoch": ep, "train_loss": durum["train_loss"], "train_acc": tr.get("acc"),
                     "train_auc": tr.get("auc"), "val_loss": v["loss"], "val_acc": v["acc"], "val_auc": v["auc"],
                     "test_acc": t["acc"], "test_auc": t["auc"], "sure_sn": round(time.time() - durum["t0"], 1)}
            durum["gecmis"].append(satir)
            durum["son"] = {"val": v, "test": t}
            durum["eniyi_val_acc"] = max(durum["eniyi_val_acc"], v["acc"])
            # (d) progress.json — kosum.py --kural canlı erken durdurma bunu okur (test değeri YOK)
            atomik_json(run_dir / "progress.json", {
                "adim": ep, "epoch_toplam": a.epoch, "loss": durum["train_loss"], "val_loss": v["loss"],
                "val_acc": v["acc"], "val_auc": v["auc"], "val_acc_eniyi": durum["eniyi_val_acc"],
                "gecen_sn": satir["sure_sn"]})
            print(f"[rakip_kos] epoch {ep}/{a.epoch} loss={durum['train_loss']:.4f} "
                  f"val_acc={v['acc']:.4f} val_auc={v['auc']:.4f}", flush=True)
        return res

    resmi.train = train_sarici
    resmi.test = test_sarici

    # ── rakip_bilgi + config.json eki (R1/R6) ─────────────────────────────────────────
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        smi = f"nvidia-smi yok: {e}"
    import sklearn
    import torchvision
    bilgi = {
        "rakip": {"depo": "https://github.com/MedMNIST/experiments", "commit": RAKIP_COMMIT,
                  "betik": "MedMNIST2D/train_and_eval_pytorch.py", "blob_sha1": blob,
                  "medmnist_paket": medmnist.__version__,
                  "medmnist_depo_commit": "805b74237fb5bd2db376fd94e50646b6a8905d9d"},
        "protokol": {"data_flag": DATA_FLAG, "size": a.size, "num_epochs": a.epoch, "batch_size": a.batch_size,
                     "lr": 0.001, "optimizer": "Adam", "milestones": [0.5 * a.epoch, 0.75 * a.epoch], "gamma": 0.1,
                     "kayip": "CrossEntropyLoss", "donusum": "ToTensor + Normalize(0.5, 0.5)", "augment": "yok",
                     "resize": False, "model": "models.ResNet18 (3x3 stem, maxpool yok)",
                     "secim": "val AUC (cur_auc > best_auc, deepcopy)", "seed": a.seed,
                     "cudnn_deterministic": True, "cudnn_benchmark": False},
        "veri": {"npz": npz_ad, "kok": KOK, "md5": md5, "md5_resmi": md5_resmi, "sha256": sha256,
                 "bayt": npz.stat().st_size},
        "donanim": {"cihaz": cihaz, "nvidia_smi": smi, "torch": torch.__version__, "cuda": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version(), "torchvision": torchvision.__version__,
                    "sklearn": sklearn.__version__, "numpy": np.__version__, "python": platform.python_version(),
                    "platform": platform.platform(), "makine": platform.node()},
    }
    atomik_json(run_dir / "rakip_bilgi.json", bilgi)
    cfg = run_dir / "config.json"
    if cfg.exists():
        try:
            c = json.loads(cfg.read_text(encoding="utf-8"))
            c.update({"donanim": bilgi["donanim"], "veri": bilgi["veri"], "rakip": bilgi["rakip"],
                      "protokol": bilgi["protokol"]})
            atomik_json(cfg, c)
        except Exception as e:  # noqa: BLE001
            print(f"[rakip_kos] UYARI: config.json'a ek yazılamadı: {e}")

    print(f"[rakip_kos] rakip @{RAKIP_COMMIT[:7]} doğrulandı · cihaz {cihaz} · seed {a.seed} · "
          f"epoch {a.epoch} · veri {npz} (md5 OK)", flush=True)

    # ── resmî main() — argümanlar resmî CLI varsayılanlarıyla birebir ────────────────
    resmi_cikti = run_dir / "resmi_cikti"
    durum["t0"] = time.time()
    resmi.main(DATA_FLAG, str(resmi_cikti), a.epoch, "0" if torch.cuda.is_available() else "-1",
               a.batch_size, a.size, False, "resnet18", False, False, None, f"s{a.seed}")
    sure = round(time.time() - durum["t0"], 1)

    # ── metrikler ─────────────────────────────────────────────────────────────────────
    G, F, S = durum["gecmis"], durum["final"], durum["son"]
    if len(G) != a.epoch or not all(k in F for k in ("train", "val", "test")):
        print(f"[rakip_kos] HATA: kayıt eksik (epoch {len(G)}/{a.epoch}, final {sorted(F)})")
        return 3
    en_iyi, eniyi_auc = 0, 0.0  # resmî kuralın aynısı: ilk kesin artış
    for i, g in enumerate(G):
        if g["val_auc"] > eniyi_auc:
            en_iyi, eniyi_auc = i, g["val_auc"]
    secim_farki = abs(F["val"]["auc"] - G[en_iyi]["val_auc"])
    if secim_farki > 1e-6:
        print(f"[rakip_kos] UYARI: seçilen epoch val AUC ({G[en_iyi]['val_auc']:.6f}) ile best_model val AUC "
              f"({F['val']['auc']:.6f}) farklı (fark {secim_farki:.2e})")

    y_val = np.load(veri_kok / npz_ad)["val_labels"].reshape(-1).astype(np.int64)
    y_test = np.load(veri_kok / npz_ad)["test_labels"].reshape(-1).astype(np.int64)
    p_val, p_test = F["val"]["skor"], F["test"]["skor"]
    tah_val, tah_test = p_val.argmax(1), p_test.argmax(1)
    n_sinif = len(INFO[DATA_FLAG]["label"])
    alt = {}
    for c in range(n_sinif):
        m = y_val == c
        alt[f"sinif_{c}"] = float((tah_val[m] == c).mean()) if m.any() else float("nan")
    val_dengeli = float(np.mean([alt[f"sinif_{c}"] for c in range(n_sinif)]))
    test_dengeli = float(np.mean([(tah_test[y_test == c] == c).mean() for c in range(n_sinif) if (y_test == c).any()]))
    # Evaluator ACC = argmax doğruluğu — bağımsız çapraz kontrol
    if abs(float((tah_val == y_val).mean()) - F["val"]["acc"]) > 1e-9:
        print("[rakip_kos] UYARI: val ACC Evaluator ile argmax doğruluğu uyuşmuyor")

    np.save(run_dir / "tahmin_val.npy", p_val)
    np.save(run_dir / "tahmin_test.npy", p_test)
    np.save(run_dir / "tahmin_val_son.npy", S["val"]["skor"])
    np.save(run_dir / "tahmin_test_son.npy", S["test"]["skor"])
    np.save(run_dir / "etiket_val.npy", y_val)
    np.save(run_dir / "etiket_test.npy", y_test)
    atomik_json(run_dir / "gecmis.json", G)

    metrics = {
        "val_acc": F["val"]["acc"], "val_auc": F["val"]["auc"], "val_loss": F["val"]["loss"],
        "val_dengeli_acc": val_dengeli,
        "test_acc": F["test"]["acc"], "test_auc": F["test"]["auc"], "test_dengeli_acc": test_dengeli,
        "val_acc_son": S["val"]["acc"], "val_auc_son": S["val"]["auc"],
        "test_acc_son": S["test"]["acc"], "test_auc_son": S["test"]["auc"],
        "train_acc": F["train"]["acc"], "train_auc": F["train"]["auc"],
        "en_iyi_epoch": en_iyi + 1, "epoch_sayisi": a.epoch, "secim_val_auc_farki": secim_farki,
        "egitim_sure_sn": sure, "sure_sn": sure, "param_sayisi": durum["param"], "seed": a.seed,
        "cihaz": cihaz, "alt_grup": alt,
    }
    for k, v in metrics.items():
        if isinstance(v, float) and not math.isfinite(v):
            print(f"[rakip_kos] UYARI: {k} sonlu değil: {v}")
    atomik_json(out, metrics)
    print(f"[rakip_kos] BİTTİ {sure}s · en iyi epoch {en_iyi + 1} · val_acc {metrics['val_acc']:.4f} "
          f"val_auc {metrics['val_auc']:.4f} → {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
