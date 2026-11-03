#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
kuyruk_yaz.py — Colab hattı kuyruklarını `kuyruk.py ekle` ile YAZAR (başlatmaz; kosum-izleme.md K1).

Kuyruklar (WS/kuyruk/; sonra drive_paketle.py Drive'a kopyalar):
  kc1-tekrar     — COLAB TEKRAR KAPISI (SIRALAMA.md ön-kaydı): baseline (rakip 28 px), H01, H02, H08; yerel tariflerin aynısı,
                   varyant adları `_colab` son ekli; a1 duman (s0) → a2 ön-uçuş (s0,s1) → a3 (s0–4).
  kc2-tur4-h12 / -h10 / -h11 — Tur-4 kartları (H12.md, H10.md, H11.md uygulama notları; komutlar kartla AYNI).
Komut biçimi (Colab): python colab/fp32_kos.py <betik> <argümanlar> --out {RUN_DIR}/metrics.json
  (H11 a3: python colab/erteleme_kapisi.py --esik 23.0 --baslangic-sa 15.3 -- python colab/fp32_kos.py …)
Ebeveyn: tekrar kolları → tekrarlanan YEREL koşumun run_id'si; Tur-4 → '<colab ebeveyn varyantı>:s<seed>' sembolik
  (Colab run_id'leri koşumda doğar; eşleme kc1 bitince kart Sonuç'una yazılır — KOMUTLAR §2 emsali).

Kullanım (WS kökünden, uv Python): PYTHONIOENCODING=utf-8 <uv-python> colab/hat/kuyruk_yaz.py [--zorla]
"""
import json
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
WS = Path(__file__).resolve().parents[2]
SK = WS.parents[1] / ".claude" / "skills" / "konudan-makale" / "scripts"
KUYRUK_PY = SK / "kuyruk.py"

KURAL_LP = '{"nan_kes":true,"ilerleme_zorunlu_sn":300,"sabit_sn":300,"min_adim":8,"alt_sinir":{"val_acc":0.69}}'
KURAL_FT = '{"nan_kes":true,"ilerleme_zorunlu_sn":300,"sabit_sn":300,"min_adim":3,"alt_sinir":{"val_acc_eniyi":0.69}}'
KURAL_RAKIP_A1 = '{"nan_kes":true,"ilerleme_zorunlu_sn":300,"sabit_sn":300}'
KURAL_RAKIP_A2 = '{"nan_kes":true,"ilerleme_zorunlu_sn":300,"sabit_sn":300,"min_adim":10,"alt_sinir":{"val_acc_eniyi":0.60}}'
KURAL_RAKIP_A3 = '{"nan_kes":true,"ilerleme_zorunlu_sn":300,"sabit_sn":300,"min_adim":30,"alt_sinir":{"val_acc_eniyi":0.60}}'

FP32 = ["python", "colab/fp32_kos.py"]
OUT = ["--out", "{RUN_DIR}/metrics.json"]
SEEDLER = [0, 1, 2, 3, 4]

# tekrarlanan yerel a3 koşumları (ebeveyn = tekrar edilen koşum)
YEREL = {
    "baseline": ["20261010-171037_baseline_s0", "20261010-174311_baseline_s1", "20261010-181653_baseline_s2",
                 "20261010-184922_baseline_s3", "20261010-192128_baseline_s4"],
    "h01": ["20261010-221110_h01_dinov2b_lp_r224_s0", "20261010-221455_h01_dinov2b_lp_r224_s1",
            "20261010-221838_h01_dinov2b_lp_r224_s2", "20261010-222218_h01_dinov2b_lp_r224_s3",
            "20261010-222554_h01_dinov2b_lp_r224_s4"],
    "h02": ["20261010-222930_h02_r18in_ft224_s0", "20261010-225428_h02_r18in_ft224_s1", "20261010-231909_h02_r18in_ft224_s2",
            "20261010-234351_h02_r18in_ft224_s3", "20261011-000829_h02_r18in_ft224_s4"],
    "h08": ["20261011-045930_h08_r18in_ft224_d4_s0", "20261011-052423_h08_r18in_ft224_d4_s1",
            "20261011-054917_h08_r18in_ft224_d4_s2", "20261011-061415_h08_r18in_ft224_d4_s3",
            "20261011-063910_h08_r18in_ft224_d4_s4"],
}
YEREL_DUMAN = {"baseline": "20261010-092815_baseline_duman_s0", "h01": "20261010-203908_h01_dinov2b_lp_r224_duman_s0",
               "h02": "20261010-204010_h02_r18in_ft224_duman_s0", "h08": "20261011-043926_h08_r18in_ft224_d4_duman_s0"}
YEREL_ONUCUS = {"baseline": ["20261010-092925_baseline_onucus_s0", "20261010-093554_baseline_onucus_s1"],
                "h01": ["20261010-204443_h01_dinov2b_lp_r224_onucus_s0", "20261010-204629_h01_dinov2b_lp_r224_onucus_s1"],
                "h02": ["20261010-204816_h02_r18in_ft224_onucus_s0", "20261010-205252_h02_r18in_ft224_onucus_s1"],
                "h08": ["20261011-044143_h08_r18in_ft224_d4_onucus_s0", "20261011-044603_h08_r18in_ft224_d4_onucus_s1"]}
REF_H02 = "colab/yerel_ref/20261010-204010_h02_r18in_ft224_duman_s0.metrics.json"
REF_H08 = "colab/yerel_ref/20261011-043926_h08_r18in_ft224_d4_duman_s0.metrics.json"


def ekle(ad, asama, varyant, seed, hipotez, ebeveyn, etiket, zs, kural, komut, gpu=True):
    cmd = [sys.executable, str(KUYRUK_PY), "--kok", str(WS), "ekle", ad, "--asama", str(asama), "--varyant", varyant,
           "--seed", str(seed), "--hipotez", hipotez, "--etiket", etiket, "--zaman-siniri-dk", str(zs)]
    if ebeveyn:
        cmd += ["--ebeveyn", ebeveyn]
    if gpu:
        cmd += ["--gpu-zorunlu"]
    if kural:
        cmd += ["--kural", kural]
    cmd += ["--"] + komut
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise SystemExit(f"ekle başarısız: {r.stdout}\n{r.stderr}")


def rakip(seed, epoch):
    return FP32 + ["scripts/rakip_kos.py", "--seed", str(seed), "--epoch", str(epoch), "--veri-kok", "data/medmnist"] + OUT


def sonda(seed, model, ek):
    return FP32 + ["varyantlar/dinov2_sonda/sonda.py", "--seed", str(seed)] + (["--model", model] if model != "b" else []) \
        + ["--r", "224"] + ek + OUT


def r18(seed, epoch, ek=()):
    return FP32 + ["varyantlar/r18_imagenet/ince_ayar.py", "--seed", str(seed), "--epoch", str(epoch)] + list(ek) + OUT


VIT_ORTAK = ["--lr-bas", "1e-3", "--lr-govde", "1e-5", "--wd", "0.01", "--katman-azalma", "1", "--isinma-epoch", "0",
             "--drop-path", "0"]


def vit_h10(seed, epoch):
    return FP32 + ["varyantlar/dinov2_ft/ince_ayar_vit.py", "--mod", "kismi", "--model", "b", "--son-k", "4", "--seed", str(seed),
                   "--epoch", str(epoch)] + VIT_ORTAK + ["--mikro-bs", "32"] + OUT


def vit_h11(seed, epoch):
    return FP32 + ["varyantlar/dinov2_ft/ince_ayar_vit.py", "--mod", "tam", "--model", "s", "--grad-ckpt", "0", "--mikro-bs", "16",
                   "--seed", str(seed), "--epoch", str(epoch)] + VIT_ORTAK + OUT


def kc1():
    ad = "kc1-tekrar"
    # a1 duman (s0)
    ekle(ad, 1, "baseline_colab_duman", 0, "K3", YEREL_DUMAN["baseline"], "Colab tekrar duman: rakip 28 px, 3 epoch", 15,
         KURAL_RAKIP_A1, rakip(0, 3))
    ekle(ad, 1, "h01_dinov2b_lp_r224_colab_duman", 0, "H01", YEREL_DUMAN["h01"],
         "Colab tekrar duman: DINOv2-B LP r224, 256+256, C=1", 15, KURAL_LP,
         sonda(0, "b", ["--alt-kume", "256", "--C", "1.0", "--secim-seedleri", "0,1"]))
    ekle(ad, 1, "h02_r18in_ft224_colab_duman", 0, "H02", YEREL_DUMAN["h02"],
         "Colab tekrar duman: R18-IN 224, 1 epoch (özdeşlik ref = yerel H02 dumanı)", 10, KURAL_FT,
         r18(0, 1, ["--ozdeslik-ref", REF_H02]))
    ekle(ad, 1, "h08_r18in_ft224_d4_colab_duman", 0, "H08", YEREL_DUMAN["h08"],
         "Colab tekrar duman: R18-IN 224 + D4, 1 epoch (özdeşlik ref = yerel H08 dumanı)", 10, KURAL_FT,
         r18(0, 1, ["--augment-d4", "--ozdeslik-ref", REF_H08]))
    # a2 ön-uçuş (s0, s1)
    for i, s in enumerate((0, 1)):
        ekle(ad, 2, "baseline_colab_onucus", s, "K3", YEREL_ONUCUS["baseline"][i], "Colab tekrar ön-uçuş: rakip 20 epoch", 30,
             KURAL_RAKIP_A2, rakip(s, 20))
    for i, s in enumerate((0, 1)):
        ekle(ad, 2, "h01_dinov2b_lp_r224_colab_onucus", s, "H01", YEREL_ONUCUS["h01"][i],
             "Colab tekrar ön-uçuş: DINOv2-B LP r224, C ızgarası, C* = ort(s0,s1)", 15, KURAL_LP,
             sonda(s, "b", ["--C", "izgara", "--secim-seedleri", "0,1"]))
    for i, s in enumerate((0, 1)):
        ekle(ad, 2, "h02_r18in_ft224_colab_onucus", s, "H02", YEREL_ONUCUS["h02"][i], "Colab tekrar ön-uçuş: H02 5 epoch", 15,
             KURAL_FT, r18(s, 5))
    for i, s in enumerate((0, 1)):
        ekle(ad, 2, "h08_r18in_ft224_d4_colab_onucus", s, "H08", YEREL_ONUCUS["h08"][i], "Colab tekrar ön-uçuş: H08 5 epoch",
             15, KURAL_FT, r18(s, 5, ["--augment-d4"]))
    # a3 tam (s0–4): H01 → H02 → H08 → rakip (en uzun en sonda)
    for s in SEEDLER:
        ek = ["--C", "izgara", "--secim-seedleri", "0,1,2,3,4"] + ([] if s == 0 else ["--lbfgs-yok"])
        ekle(ad, 3, "h01_dinov2b_lp_r224_colab", s, "H01", YEREL["h01"][s],
             "Colab tekrar H01: donuk DINOv2 ViT-B/14 CLS (native 224) + L2 softmax-LR, C* val'de", 20, KURAL_LP, sonda(s, "b", ek))
    for s in SEEDLER:
        ekle(ad, 3, "h02_r18in_ft224_colab", s, "H02", YEREL["h02"][s],
             "Colab tekrar H02: ResNet-18 IMAGENET1K_V1 native 224 ince ayar (AdamW 1e-4, kosinüs, 30 ep, bs 64)", 40, KURAL_FT,
             r18(s, 30))
    for s in SEEDLER:
        ekle(ad, 3, "h08_r18in_ft224_d4_colab", s, "H08", YEREL["h08"][s],
             "Colab tekrar H08: H02 + eğitimde D4 (30 ep)", 40, KURAL_FT, r18(s, 30, ["--augment-d4"]))
    for s in SEEDLER:
        ekle(ad, 3, "baseline_colab", s, "K3", YEREL["baseline"][s],
             "Colab tekrar rakip: MedMNIST resmî ResNet-18 28 px, 100 epoch (@70b6b3a)", 150, KURAL_RAKIP_A3, rakip(s, 100))


def kc2():
    # H12 (analiz) — kart H12.md uygulama notu
    ad = "kc2-tur4-h12"
    ekle(ad, 1, "h12_dinov2s_lp_r224_duman", 0, "H12", "h01_dinov2b_lp_r224_colab_duman:s0",
         "H12 duman: DINOv2-S LP r224, 256+256, C=1", 10, KURAL_LP,
         sonda(0, "s", ["--alt-kume", "256", "--C", "1.0", "--secim-seedleri", "0"]))
    for s in (0, 1):
        ekle(ad, 2, "h12_dinov2s_lp_r224_onucus", s, "H12", f"h01_dinov2b_lp_r224_colab_onucus:s{s}",
             "H12 ön-uçuş: DINOv2-S LP r224, C ızgarası, C* = ort(s0,s1)", 15, KURAL_LP,
             sonda(s, "s", ["--C", "izgara", "--secim-seedleri", "0,1"]))
    for s in SEEDLER:
        ek = ["--C", "izgara", "--secim-seedleri", "0,1,2,3,4"] + ([] if s == 0 else ["--lbfgs-yok"])
        ekle(ad, 3, "h12_dinov2s_lp_r224", s, "H12", f"h01_dinov2b_lp_r224_colab:s{s}",
             "H12: donuk DINOv2 ViT-S/14 CLS (native 224) + L2 softmax-LR (H01 tarifi)", 15, KURAL_LP, sonda(s, "s", ek))
    # H10 — kart H10.md uygulama notu
    ad = "kc2-tur4-h10"
    ekle(ad, 1, "h10_dinov2b_k4_ft224_duman", 0, "H10", "h02_r18in_ft224_colab_duman:s0",
         "H10 duman: DINOv2-B son 4 blok, 1 epoch (önbellek L4'te yazılır)", 10, KURAL_FT, vit_h10(0, 1))
    for s in (0, 1):
        ekle(ad, 2, "h10_dinov2b_k4_ft224_onucus", s, "H10", f"h02_r18in_ft224_colab_onucus:s{s}",
             "H10 ön-uçuş: DINOv2-B son 4 blok, 5 epoch", 35, KURAL_FT, vit_h10(s, 5))
    for s in SEEDLER:
        ekle(ad, 3, "h10_dinov2b_k4_ft224", s, "H10", f"h02_r18in_ft224_colab:s{s}",
             "H10: DINOv2-B/14 son 4 blok kısmi ince ayar (baş 1e-3 / gövde 1e-5, adım kosinüs, 15 ep, mikro 32x2)", 90, KURAL_FT,
             vit_h10(s, 15))
    # H11 — kart H11.md uygulama notu (+ erteleme kuralı a3)
    ad = "kc2-tur4-h11"
    ekle(ad, 1, "h11_dinov2s_ft224_duman", 0, "H11", "h02_r18in_ft224_colab_duman:s0",
         "H11 duman: DINOv2-S tam ince ayar, 1 epoch", 10, KURAL_FT, vit_h11(0, 1))
    for s in (0, 1):
        ekle(ad, 2, "h11_dinov2s_ft224_onucus", s, "H11", f"h02_r18in_ft224_colab_onucus:s{s}",
             "H11 ön-uçuş: DINOv2-S tam ince ayar, 5 epoch", 30, KURAL_FT, vit_h11(s, 5))
    for s in SEEDLER:
        ekle(ad, 3, "h11_dinov2s_ft224", s, "H11", f"h02_r18in_ft224_colab:s{s}",
             "H11: DINOv2-S/14 tam ince ayar (grad-ckpt 0, mikro 16x4, 15 ep) — erteleme kapılı", 80, KURAL_FT,
             ["python", "colab/erteleme_kapisi.py", "--esik", "23.0", "--baslangic-sa", "15.3", "--"] + vit_h11(s, 15))


def main():
    zorla = "--zorla" in sys.argv
    adlar = ["kc1-tekrar", "kc2-tur4-h12", "kc2-tur4-h10", "kc2-tur4-h11"]
    for ad in adlar:
        j = WS / "kuyruk" / f"{ad}.jsonl"
        b = WS / "kuyruk" / f"{ad}.beklenti.json"
        if not b.exists():
            raise SystemExit(f"HATA: önce beklenti yazılır (ön-kayıt): {b}")
        if j.exists():
            if not zorla:
                raise SystemExit(f"HATA: {j} zaten var (--zorla ile silinip yeniden yazılır; koşum başladıysa YAPMA)")
            j.unlink()
    kc1()
    kc2()
    for ad in adlar:
        n = sum(1 for l in (WS / "kuyruk" / f"{ad}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())
        print(f"{ad}: {n} satır")
    return 0


if __name__ == "__main__":
    sys.exit(main())
