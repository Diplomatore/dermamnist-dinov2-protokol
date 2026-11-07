#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
e1_kuyruk_yaz.py — E1 (H14) Colab kuyruklarını YAZAR (başlatmaz; kosum-izleme.md K1). Önce beklenti dosyaları (ön-kayıt),
sonra satırlar (`kuyruk.py ekle`). Kaynak: hipotezler/H14.md (KULLANICI ONAYI 2026-11-03, yalnız E1 kapsamı).

  e1-h14   a1 c_bolme_veri_c1 · h14_c1_d4kapali_duman (s0, 1 ep, 8 görünüm önbelleğini YAZAR, D4 KAPALI, --ozdeslik-ref
               20261011-215538_h10_c1_duman_s0 → kural f Durum A/B/C) · h14_c1_duman (s0, 1 ep, D4)
           a2 h14_c1_onucus (s0, s1 × 5 ep, D4)
           a3 h14_c1 (s0–4 × 15 ep, D4) + h14_c1-test (s0–4; k5_cikarim --veri c1 --test-kilidi-acik; C-test bir kez)
  e1-h10r  (KOŞULLU — yalnız Durum B; defter yalnız kontrol kolu Durum B ise koşar) H10 C1 tarifi aynen, yeni ortamda:
           a1 c_bolme_veri_c1 · h10r_c1_duman (+ --ozdeslik-ref @e1-h14/h14_c1_d4kapali_duman: eski tek görünüm yolu ↔
               kontrol kolu bit-bit; yalnız okuma, eğitimi değiştirmez — H14-UYGULAMA-NOTU §2)
           a2 h10r_c1_onucus (s0) · a3 h10r_c1 (s0–4) + h10r_c1-test (s0–4)
Komutlar H10 C1 (k5-c1-B) satırının aynısı + yalnız H14 bayrakları. a2/a3 ve C-test satırları E1 bütçe kapısıyla sarılır
(H14 kural h: E1 birleşik L4-sa > 5,0 → rc=94, satır başlamaz): colab/erteleme_kapisi.py --kuyruk-onek e1-.
Koruma: her k5_cikarim satırında `--veri c1` yoksa yazıcı HATA verir (H14 kural i).
Beklenti: e1-h14 = H14.md "Beklenti dosyası taslağı" JSON bloğu AYNEN (karttan okunur, değiştirilmez);
          e1-h10r = k5-c1-B'nin h10 kuralları (k5_kuyruk_yaz.beklenti_c), varyant adları h10r_*, + kontrol_* ve veri_c.
Kullanım (WS kökü, uv Python): PYTHONIOENCODING=utf-8 <uv-python> colab/hat/e1_kuyruk_yaz.py [--zorla]
"""
import copy
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HAT = Path(__file__).resolve().parent
sys.path.insert(0, str(HAT))
import kuyruk_yaz as KY  # noqa: E402
import k5_kuyruk_yaz as K5  # noqa: E402

WS = KY.WS
KART = WS / "hipotezler" / "H14.md"
SEED = [0, 1, 2, 3, 4]
H10_C1 = ["20261011-225511_h10_c1_s0", "20261011-230845_h10_c1_s1", "20261011-232219_h10_c1_s2",
          "20261011-233553_h10_c1_s3", "20261011-234926_h10_c1_s4"]
H10_C1_ONUCUS = ["20261011-220331_h10_c1_onucus_s0", "20261011-220816_h10_c1_onucus_s1"]
H10_C1_DUMAN = "20261011-215538_h10_c1_duman_s0"
BUTCE = ["python", "colab/erteleme_kapisi.py", "--esik", "5.0", "--baslangic-sa", "0", "--kuyruk-onek", "e1-",
         "--kural", "E1 bütçe (H14 kural h: E1 birleşik L4-sa > 5,0 → DUR)", "--"]
G8 = ["--onbellek-gorunum", "8"]


def butceli(komut):
    return BUTCE + komut


def c_bolme_satiri(ad):
    KY.ekle(ad, 1, "c_bolme_veri_c1", 0, "H14", None, "E1 C1 veri kurulumu/denetimi (c_bolme_npz.py --kip c1)", 70, None,
            ["python", "scripts/c_bolme_npz.py", "--kip", "c1", "--out", "{RUN_DIR}/metrics.json"], gpu=False)


def cikarim(kaynak_varyant, kuyruk, s):
    return K5.cikarim("vit_kismi", ["--kaynak-varyant", kaynak_varyant, "--kaynak-kuyruk", kuyruk], s, veri="c1", test=True)


# ───────────────────────── beklentiler (ön-kayıt; satırlardan ÖNCE) ─────────────────────────
def beklenti_h14() -> dict:
    """Kartın 'Beklenti dosyası taslağı' JSON bloğu — AYNEN (kart salt okunur)."""
    metin = KART.read_text(encoding="utf-8")
    i = metin.index("## Beklenti dosyası taslağı")
    m = re.search(r"```json\s*\n(.*?)\n```", metin[i:], flags=re.S)
    if not m:
        raise SystemExit("HATA: H14.md'de beklenti JSON bloğu bulunamadı")
    B = json.loads(m.group(1))
    assert B["metrik"] == "val_acc" and set(B["asama"]) == {"1", "2", "3"}, B.keys()
    return B


def beklenti_h10r() -> dict:
    B = copy.deepcopy(K5.beklenti_c("e1-h10r", ["h10"], "c1"))
    ad = {"h10_c1_duman": "h10r_c1_duman", "h10_c1_onucus": "h10r_c1_onucus", "h10_c1": "h10r_c1", "h10_c1-test": "h10r_c1-test"}
    for k, a in B["asama"].items():
        for r in a["kurallar"]:
            if r.get("varyant") in ad:
                r["varyant"] = ad[r["varyant"]]
            if r["tur"] == "min_kosum":
                r["n"] = {"1": 1, "2": 1, "3": 10}[k]
    B["asama"]["1"]["kurallar"] += [
        {"tur": "ikincil_aralik", "metrik": "kontrol_ilk_adim_loss_fark", "alt": 0.0, "ust": 1e-06,
         "kaynak": "H14 kural f Durum B ek koşulu: h10r_c1_duman (eski tek görünüm yolu, yeni ortam) ↔ h14_c1_d4kapali_duman bit-bit"},
        {"tur": "ikincil_aralik", "metrik": "kontrol_p_maks_fark", "alt": 0.0, "ust": 1e-05,
         "kaynak": "H14 kural f Durum B ek koşulu (maks |Δp| ≤ 1e-5)"},
    ]
    B["asama"]["3"]["kurallar"].append(
        {"tur": "ikincil_aralik", "metrik": "veri_c", "alt": 1, "ust": 1, "kaynak": "H14 kural i: yalnız C1 C-test; resmî test YOK"})
    B["asama"]["1"]["aciklama"] = "DUMAN — C1 veri kurulumu + h10r_c1_duman (H10 C1 tarifi aynen, yeni ortam; kontrol kolu ↔ bit-bit)"
    B["asama"]["2"]["aciklama"] = "ÖN-UÇUŞ — h10r_c1_onucus s0 × 5 ep"
    B["asama"]["3"]["aciklama"] = "TAM — h10r_c1 s0–4 × 15 ep + C-test çıkarımı (bir kez)"
    B["_aciklama"] = ("ÖN-KAYIT (kuyruk e1-h10r; E1 KOŞULLU H10r — yalnız H14 kural f Durum B; KULLANICI ONAYI 2026-11-03; "
                      "hiçbir e1 koşumundan ÖNCE yazıldı; koşum sonrası DEĞİŞTİRİLMEZ, devam kuyruklarına BAYT BAYT kopyalanır). "
                      "Kaynak: hipotezler/H14.md 'Koşullu e1-h10r.beklenti.json' + k5-c1-B.beklenti.json h10 kuralları (K5-ONKAYIT §4.2 M8).")
    B["_notlar"] = [
        "Aralıklar ARIZA ALARMIDIR. Satırlar k5-c1-B h10 satırlarının aynısı (varyant adı h10r_*); a2 yalnız s0 (kart) → min_kosum 1.",
        "h10r_c1_duman --ozdeslik-ref @e1-h14/h14_c1_d4kapali_duman yalnız eğitim SONRASI okuma yapar (kontrol_* anahtarları); "
        "eğitim yolu H10 ile aynıdır (--onbellek-gorunum 1 varsayılan). Bkz. hipotezler/H14-UYGULAMA-NOTU.md §2.",
        "kontrol_p_maks_fark kuralı kartın kural (f) Durum B metninden ('∧ maks |Δp| ≤ 1e-5'); kartın h10r beklenti cümlesi yalnız "
        "ilk_adim_loss'u anar — eklenen kural daha sıkıdır, gevşetme değildir.",
        "a2/a3/C-test satırları E1 bütçe kapısıyla sarılı (kural h).",
    ]
    return B


# ───────────────────────── satırlar ─────────────────────────
def kuyruk_h14():
    ad = "e1-h14"
    c_bolme_satiri(ad)
    KY.ekle(ad, 1, "h14_c1_d4kapali_duman", 0, "H14", H10_C1_DUMAN,
            "E1 H14 KONTROL KOLU: 8 görünüm önbellek yolu, D4 KAPALI, 1 ep ↔ h10_c1_duman_s0 (kural f Durum A/B/C); önbelleği yazar",
            20, KY.KURAL_FT, K5.sar("c1", K5.vit_h10(0, 1) + G8 + ["--ozdeslik-ref", H10_C1_DUMAN]))
    KY.ekle(ad, 1, "h14_c1_duman", 0, "H14", H10_C1_DUMAN, "E1 H14 duman: H10 C1 + eğitim-zamanı D4 (önbellek görünümü), 1 ep",
            10, KY.KURAL_FT, K5.sar("c1", K5.vit_h10(0, 1) + G8 + ["--augment-d4"]))
    for s in (0, 1):
        KY.ekle(ad, 2, "h14_c1_onucus", s, "H14", H10_C1_ONUCUS[s], "E1 H14 ön-uçuş: H10 C1 + D4, 5 ep", 20, KY.KURAL_FT,
                butceli(K5.sar("c1", K5.vit_h10(s, 5) + G8 + ["--augment-d4"])))
    for s in SEED:
        KY.ekle(ad, 3, "h14_c1", s, "H14", H10_C1[s],
                "E1 H14: DINOv2-B son 4 blok + eğitim-zamanı D4 (H10 C1 tarifi + D4; 8 görünüm fp16 önbellek), 15 ep", 45,
                KY.KURAL_FT, butceli(K5.sar("c1", K5.vit_h10(s, 15) + G8 + ["--augment-d4"])))
    for s in SEED:
        KY.ekle(ad, 3, "h14_c1-test", s, "H14", f"h14_c1:s{s}", f"E1 C-test çıkarımı (bir kez; test sonrası keşifsel): h14_c1 s{s}",
                15, None, butceli(cikarim("h14_c1", ad, s)))


def kuyruk_h10r():
    ad = "e1-h10r"
    c_bolme_satiri(ad)
    KY.ekle(ad, 1, "h10r_c1_duman", 0, "H14", H10_C1_DUMAN,
            "E1 H10r duman (yalnız Durum B): H10 C1 tarifi aynen, yeni ortam, 1 ep ↔ h14_c1_d4kapali_duman bit-bit",
            15, KY.KURAL_FT, K5.sar("c1", K5.vit_h10(0, 1) + ["--ozdeslik-ref", "@e1-h14/h14_c1_d4kapali_duman"]))
    KY.ekle(ad, 2, "h10r_c1_onucus", 0, "H14", H10_C1_ONUCUS[0], "E1 H10r ön-uçuş: H10 C1 tarifi aynen, 5 ep", 35, KY.KURAL_FT,
            butceli(K5.sar("c1", K5.vit_h10(0, 5))))
    for s in SEED:
        KY.ekle(ad, 3, "h10r_c1", s, "H14", H10_C1[s], "E1 H10r: H10 C1 tarifi aynen (yeni ortam; Durum B karşılaştırıcısı), 15 ep",
                90, KY.KURAL_FT, butceli(K5.sar("c1", K5.vit_h10(s, 15))))
    for s in SEED:
        KY.ekle(ad, 3, "h10r_c1-test", s, "H14", f"h10r_c1:s{s}", f"E1 C-test çıkarımı (bir kez): h10r_c1 s{s}", 15, None,
                butceli(cikarim("h10r_c1", ad, s)))


def denetle(ad) -> int:
    n = 0
    for satir in (WS / "kuyruk" / f"{ad}.jsonl").read_text(encoding="utf-8").splitlines():
        if not satir.strip():
            continue
        j = json.loads(satir)
        n += 1
        k = j["komut"]
        if any(str(x).endswith("k5_cikarim.py") for x in k):
            i = k.index("--veri") if "--veri" in k else -1
            if i < 0 or k[i + 1] != "c1":
                raise SystemExit(f"HATA (H14 kural i): {ad} {j['varyant']} s{j['seed']}: k5_cikarim satırında --veri c1 YOK")
            if not j["varyant"].endswith("_c1-test"):
                raise SystemExit(f"HATA: {j['varyant']} k5_cikarim satırı ama -test varyantı değil")
        if "--test-kilidi-acik" in k and not j["varyant"].endswith("_c1-test"):
            raise SystemExit(f"HATA: test kilidi yalnız C-test satırlarında ({j['varyant']})")
        if any(str(x).endswith("ince_ayar_vit.py") for x in k) and "--augment-d4" in k and "--onbellek-gorunum" not in k:
            raise SystemExit(f"HATA: {j['varyant']} --augment-d4 var ama --onbellek-gorunum 8 yok")
    return n


def main():
    zorla = "--zorla" in sys.argv
    adlar = ["e1-h14", "e1-h10r"]
    kuyruklar_dizin = WS / "runs" / "kuyruklar"
    for ad in adlar:
        j, b = WS / "kuyruk" / f"{ad}.jsonl", WS / "kuyruk" / f"{ad}.beklenti.json"
        if (j.exists() or b.exists()) and not zorla:
            raise SystemExit(f"HATA: {j.name}/{b.name} zaten var (--zorla yalnız hiçbir e1 koşumu başlamadıysa)")
        for kok in (kuyruklar_dizin, Path("<DRIVE_KOK>/3-1/runs/kuyruklar")):
            if kok.exists() and list(kok.glob(f"{ad}*.baslat.json")):
                raise SystemExit(f"HATA: {ad} başlamış ({kok}) — ön-kayıt ezilemez")
        j.unlink(missing_ok=True)
    # 1) beklentiler (ön-kayıt) — satırlardan ÖNCE
    for ad, B in (("e1-h14", beklenti_h14()), ("e1-h10r", beklenti_h10r())):
        (WS / "kuyruk" / f"{ad}.beklenti.json").write_text(json.dumps(B, ensure_ascii=False, indent=1), encoding="utf-8")
    # 2) satırlar
    kuyruk_h14()
    kuyruk_h10r()
    for ad in adlar:
        print(f"{ad}: {denetle(ad)} satır (denetim: k5_cikarim --veri c1 ✓, test kilidi yalnız C-test ✓)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
