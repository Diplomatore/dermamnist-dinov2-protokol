#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
e2_kuyruk_yaz.py — E2 (H15, BCN20000) Colab kuyruğu `e2-degerlendirme`'yi YAZAR (başlatmaz; kosum-izleme.md K1).
Önce beklenti dosyası (ön-kayıt; kartın TASLAK'ından kesinleştirilir), SONRA satırlar (`kuyruk.py ekle`).
Kaynak: hipotezler/H15.md (KULLANICI ONAYI H15, 2026-11-07; SHA256 eeef8714…). Kol listesi: varyantlar/e2/kollar.py.

  a1  e2_veri_kapisi (CPU; bcn_veri.py kapi: K-a nn_tam = 0, K-b sayılar, Colab kopyasının dosya + içerik SHA256'sı)
      c_bolme_veri_c1 (CPU; C1 kolları için C-val verisi; E1 ile aynı satır)
      e2_h10_duman (H10 s0; val özdeşliği + BCN ilk 48 görüntü ETİKETSİZ, YAZILMAZ + 8 HAM kalibrasyon görüntüsü)
  a2  e2_<kol>_onucus × 11 kol × s0–4 (yalnız iç val yeniden çıkarımı + arşiv özdeşliği; BCN YOK)
  a3  e2_kilit_ac (CPU; scripts/e2_kilit.py: K-d kuru koşu + girdi SHA + a2 → sonuclar/e2_bcn.KILIT)
      e2_<kol>-bcn × 11 kol × s0–4 (--dis-kilit-acik; val özdeşliği yeniden + BCN 12.413 × 2 geometri, ETİKETSİZ)
a2/a3 GPU satırları E2 bütçe kapısıyla sarılır (H15 K-g: e2-degerlendirme koşum toplamı > 3,0 L4-sa → rc 94, satır başlamaz).
Sıra: birincil kollar önce (kollar.SIRA). Etiketli analiz kuyrukta DEĞİL: şef, Colab bitince yerelde bir kez
`scripts/e2_bcn_analiz.py` koşar (kart a3 son adımı; KİLİT + K-d yeniden).
Kullanım (WS kökü, uv Python): PYTHONIOENCODING=utf-8 <uv-python> colab/hat/e2_kuyruk_yaz.py [--zorla]
"""
import copy
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HAT = Path(__file__).resolve().parent
sys.path.insert(0, str(HAT))
import kuyruk_yaz as KY  # noqa: E402

WS = KY.WS
sys.path.insert(0, str(WS / "varyantlar"))
from e2 import kollar as KL  # noqa: E402

AD = "e2-degerlendirme"
BEKLENTI = WS / "kuyruk" / f"{AD}.beklenti.json"
TASLAK_SHA = "2bd91eb178d21ac56007ec7ab1abed1365a2cd2ce71e07a83eef30042f6356c9"   # kart taslağı (2026-11-07 19:43:53)
BUTCE = ["python", "colab/erteleme_kapisi.py", "--esik", "3.0", "--baslangic-sa", "0", "--kuyruk-onek", AD,
         "--kural", "E2 bütçe (H15 K-g: e2-degerlendirme koşum toplamı > 3,0 L4-sa → DUR)", "--"]
KURAL_CIKARIM = '{"nan_kes":true,"ilerleme_zorunlu_sn":600,"sabit_sn":600}'   # yalnız takılma/NaN (çıkarım; öğrenme yok)


def cikarim(k, s, kip, ek=()):
    return KY.FP32 + ["varyantlar/e2/e2_cikarim.py", "--kol", k, "--kaynak", KL.run_id(k, s), "--seed", str(s), "--kip", kip] \
        + list(ek) + KY.OUT


def beklenti_kesinlestir(B: dict) -> dict:
    """Kart taslağı (kuyruk/e2-degerlendirme.beklenti.json, TASLAK) → kesin. Aralıklar ve kurallar DEĞİŞMEZ; yalnız durum
    satırı kalkar, notlar eklenir ve a1'e daha sıkı bir kural (kal_girdi_sha_eslesti = 1) eklenir (gevşetme yok)."""
    B = copy.deepcopy(B)
    B.pop("_durum", None)
    B["_aciklama"] = B["_aciklama"] + (" KESİNLEŞTİRME: deney-yürütücü, 2026-11-07, KULLANICI ONAYI H15 sonrası (kart SHA256 "
                                       "eeef871456c78ee3…, mtime 19:46:24), kuyruk satırlarından ÖNCE; taslak SHA256 "
                                       f"{TASLAK_SHA[:16]}…. Ayrıntı hipotezler/H15-UYGULAMA-NOTU.md.")
    B["_notlar"] = B["_notlar"] + [
        "a1'de e2_veri_kapisi (bcn_veri.py kapi; metrik anahtarları kapi_*) ve c_bolme_veri_c1 satırları val_acc yazmaz → kapının "
        "'başarılı koşum' sayımına ve ikincil_aralik kurallarına girmez; rc ≠ 0 (DUR) ise mekanik İHLAL olur. a3'teki e2_kilit_ac "
        "satırı da aynıdır (KİLİT yazılamazsa rc ≠ 0; o zaman 55 çıkarım satırı KİLİT bulamaz ve rc 7 ile biter).",
        "e2_cikarim.py çıkış kodları: 4 özdeşlik Durum C / yanlış val kümesi (BCN AÇILMAZ) · 5 dejenere BCN çıktısı (K-e) · 7 KİLİT "
        "yok/uyuşmaz · 8 aynı kol-seed için ikinci BCN tahmini · 9 BCN girdisi içerik SHA256'sı uyuşmaz. Hepsi rc ≠ 0 → İHLAL.",
        "a1 kal_girdi_sha_eslesti [1, 1] kartta yazılı değildir; kalibrasyon girdisinin a0'da donan içerik SHA256'sıyla aynı olmasını "
        "ister (daha sıkı kural; gevşetme değil).",
        "ozdeslik_durum_kod (1 A · 2 B · 3 C) ve ozdeslik_surum_farkli her satırda yazılır; Durum A/B ayrımı analizde ve KİLİT'te.",
    ]
    for r in B["asama"]["1"]["kurallar"]:
        assert r.get("metrik") != "kal_girdi_sha_eslesti"
    B["asama"]["1"]["kurallar"].append({"tur": "ikincil_aralik", "metrik": "kal_girdi_sha_eslesti", "alt": 1, "ust": 1,
                                        "kaynak": "a0 kuyruk/e2_bcn_beklenen.json kal_images içerik SHA256 (daha sıkı kural)"})
    return B


def beklenti_denetle(B: dict):
    assert B["metrik"] == "val_acc" and set(B["asama"]) == {"1", "2", "3"} and "_durum" not in B
    a2 = {r["varyant"] for r in B["asama"]["2"]["kurallar"] if r["tur"] == "aralik"}
    assert a2 == {KL.e2_varyant(k, "onucus") for k in KL.SIRA}, a2
    a3 = {r["varyant"] for r in B["asama"]["3"]["kurallar"] if r["tur"] == "aralik"}
    assert a3 == {KL.e2_varyant(k, "bcn") for k in KL.BIRINCIL}, a3
    mk = {r["n"] for k in "123" for r in B["asama"][k]["kurallar"] if r["tur"] == "min_kosum"}
    assert mk == {1, 55}, mk


def kuyruk():
    KY.ekle(AD, 1, "e2_veri_kapisi", 0, "H15", None, "E2 a1 VERİ KAPISI: K-a nn_tam=0, K-b sayılar, BCN girdisi dosya+içerik SHA256 (CPU)",
            15, None, ["python", "varyantlar/e2/bcn_veri.py", "kapi", "--out", "{RUN_DIR}/metrics.json"], gpu=False)
    KY.ekle(AD, 1, "c_bolme_veri_c1", 0, "H15", None, "E2 C1 veri kurulumu/denetimi (c_bolme_npz.py --kip c1; C1 kollarının C-val'i)",
            70, None, ["python", "scripts/c_bolme_npz.py", "--kip", "c1", "--out", "{RUN_DIR}/metrics.json"], gpu=False)
    KY.ekle(AD, 1, KL.e2_varyant("h10", "duman"), 0, "H15", KL.run_id("h10", 0),
            "E2 duman: H10 s0 val özdeşliği + BCN ilk 48 görüntü (ETİKETSİZ, yazılmaz) + 8 HAM kalibrasyon görüntüsü", 25,
            KURAL_CIKARIM, cikarim("h10", 0, "duman", ["--bcn-ilk", "48", "--kal"]))
    for k in KL.SIRA:
        for s in KL.SEEDLER:
            KY.ekle(AD, 2, KL.e2_varyant(k, "onucus"), s, "H15", KL.run_id(k, s),
                    f"E2 ön-uçuş: {KL.KOLLAR[k]['ad']} s{s} val yeniden çıkarımı ↔ arşiv (K-c; BCN YOK)", 15, KURAL_CIKARIM,
                    BUTCE + cikarim(k, s, "onucus"))
    KY.ekle(AD, 3, "e2_kilit_ac", 0, "H15", None, "E2 a3 DIŞ TEST KİLİDİ: K-d kuru koşu + girdi SHA256 + a2 → sonuclar/e2_bcn.KILIT (CPU)",
            30, None, ["python", "scripts/e2_kilit.py", "--out", "{RUN_DIR}/metrics.json"], gpu=False)
    for k in KL.SIRA:
        for s in KL.SEEDLER:
            KY.ekle(AD, 3, KL.e2_varyant(k, "bcn"), s, "H15", KL.run_id(k, s),
                    f"E2 BCN (KİLİT): {KL.KOLLAR[k]['ad']} s{s} — val özdeşliği + 12.413 × 2 geometri, etiketsiz, bir kez", 40,
                    KURAL_CIKARIM, BUTCE + cikarim(k, s, "bcn", ["--dis-kilit-acik"]))


def denetle() -> dict:
    say = {1: 0, 2: 0, 3: 0}
    for satir in (WS / "kuyruk" / f"{AD}.jsonl").read_text(encoding="utf-8").splitlines():
        if not satir.strip():
            continue
        j = json.loads(satir)
        say[int(j["asama"])] += 1
        k = j["komut"]
        bcn = "--dis-kilit-acik" in k
        if bcn != (j["varyant"].endswith("-bcn")):
            raise SystemExit(f"HATA: --dis-kilit-acik yalnız a3 -bcn satırlarında ({j['varyant']})")
        if bcn and int(j["asama"]) != 3:
            raise SystemExit(f"HATA: BCN satırı a3 dışında ({j['varyant']})")
        if any(x in k for x in ("--test-kilidi-acik", "--ikinci-acilis", "--alt-kume")):
            raise SystemExit(f"HATA: yasak bayrak ({j['varyant']})")
        if any(str(x).endswith("e2_cikarim.py") for x in k):
            kol = k[k.index("--kol") + 1]
            s = int(k[k.index("--seed") + 1])
            if k[k.index("--kaynak") + 1] != KL.run_id(kol, s) or str(s) != str(j["seed"]):
                raise SystemExit(f"HATA: {j['varyant']} kaynak/seed kol kaydıyla aynı değil")
    if say != {1: 3, 2: 55, 3: 56}:
        raise SystemExit(f"HATA: satır sayıları {say} ≠ a1 3 / a2 55 / a3 56")
    return say


def main():
    zorla = "--zorla" in sys.argv
    j = WS / "kuyruk" / f"{AD}.jsonl"
    for kok in (WS / "runs" / "kuyruklar", Path("<DRIVE_KOK>/3-1/runs/kuyruklar")):
        if kok.exists() and list(kok.glob(f"{AD}*.baslat.json")):
            raise SystemExit(f"HATA: {AD} başlamış ({kok}) — ön-kayıt ezilemez")
    if j.exists() and not zorla:
        raise SystemExit(f"HATA: {j.name} zaten var (--zorla yalnız hiçbir e2 koşumu başlamadıysa)")
    B = json.loads(BEKLENTI.read_text(encoding="utf-8"))
    if "_durum" in B:                     # 1) beklenti (ön-kayıt) — satırlardan ÖNCE
        import hashlib
        if hashlib.sha256(BEKLENTI.read_bytes()).hexdigest() != TASLAK_SHA:
            raise SystemExit("HATA: taslak beklenti dosyası kartla birlikte yazılan sürüm değil (SHA256)")
        B = beklenti_kesinlestir(B)
        BEKLENTI.write_text(json.dumps(B, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"beklenti kesinleştirildi: {BEKLENTI}")
    beklenti_denetle(B)
    j.unlink(missing_ok=True)
    kuyruk()                              # 2) satırlar
    print(f"{AD}: {denetle()} satır (a1 / a2 / a3) · denetim: --dis-kilit-acik yalnız a3 -bcn ✓, kaynak = kollar.py ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
