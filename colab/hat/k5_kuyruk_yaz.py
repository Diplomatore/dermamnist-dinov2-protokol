#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
k5_kuyruk_yaz.py — K5 Colab kuyruklarını YAZAR (başlatmaz; kosum-izleme.md K1). Önce beklenti dosyaları (ön-kayıt), sonra satırlar
(`kuyruk.py ekle`). Kaynak: hipotezler/K5-ONKAYIT.md (KULLANICI KARARI 10 ile onaylı) §0 P6, §1, §4, §5.

Kuyruklar (WS/kuyruk/; drive_paketle.py Drive'a kopyalar). Sekme sınırı (kullanıcı 2026-10-11): aynı anda EN ÇOK 2 Colab oturumu.
  k5-test    (tek sekme; izin kuyruk/k5-test.izin.json — ŞEF yazar)
             a1 = P6 YÜKLEME ÖZDEŞLİĞİ (yalnız val; test AÇILMAZ): 7 kol × 5 seed (`<kaynak>-ozdeslik`)
             a3 = RESMÎ TEST ÇIKARIMI (--test-kilidi-acik): aynı 35 satır (`<kaynak>-test`); a1 kapısı geçmezse BAŞLAMAZ.
  k5-c1-A / k5-c1-B  (C1; iki paralel sekme; izin kuyruk/k5-c1.izin.json — şef, resmî test sonucu SHA256 ile donduktan SONRA)
             A: baseline + H02 + H01 · B: H10 + H08 (süre dengesi; KOMUTLAR §K5). Her kuyruğun ilk satırı C1 verisini kurar
             (scripts/c_bolme_npz.py --kip c1; içerik SHA256 = kuyruk/k5_c_bolme_beklenen.json). a1 duman → a2 ön-uçuş →
             a3 tam + C-test çıkarımı (`<kol>_c1-test`; rakip C-test tahminini kendisi yazar).
  k5-c2      (C2; tek sekme, C1 bitince iki sekmeden birinde; izin kuyruk/k5-c2.izin.json — şef, §5 bütçe kapısı ≤ 28,8)
             baseline + H10 + H08, C npz (Zenodo), aynı yapı.
Tarifler kc1-tekrar / kc2-tur4 satırlarıyla AYNI komut + argüman; tek fark sarmalayıcı (k5/c_bolme_sar.py / c_rakip_kos.py --veri)
ve varyant adı. Ebeveyn: resmî kolun aynı seed Colab run_id'si (C1) / aynı seed C1 kolu (C2, sembolik).
Kullanım (WS kökü, uv Python): PYTHONIOENCODING=utf-8 <uv-python> colab/hat/k5_kuyruk_yaz.py [--zorla]
"""
import json
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HAT = Path(__file__).resolve().parent
sys.path.insert(0, str(HAT))
import kuyruk_yaz as KY  # noqa: E402  (KURAL_*, FP32, OUT, VIT_ORTAK, ekle)

WS = KY.WS
SEED = [0, 1, 2, 3, 4]
RESMI = {  # K5-ONKAYIT §1 (Colab kolları; seed 0..4)
    "baseline": ("baseline_colab", "rakip", ["20261011-142338", "20261011-143613", "20261011-144846", "20261011-150122", "20261011-151359"]),
    "h01": ("h01_dinov2b_lp_r224_colab", "sonda", ["20261011-125349", "20261011-125744", "20261011-130031", "20261011-130320", "20261011-130608"]),
    "h02": ("h02_r18in_ft224_colab", "r18", ["20261011-130855", "20261011-131620", "20261011-132346", "20261011-133112", "20261011-133837"]),
    "h08": ("h08_r18in_ft224_d4_colab", "r18", ["20261011-134604", "20261011-135335", "20261011-140107", "20261011-140837", "20261011-141608"]),
    "h10": ("h10_dinov2b_k4_ft224", "vit_kismi", ["20261011-160211", "20261011-161411", "20261011-162611", "20261011-163811", "20261011-165010"]),
    "h11": ("h11_dinov2s_ft224", "vit_tam", ["20261011-160117", "20261011-161247", "20261011-162419", "20261011-163552", "20261011-164724"]),
    "h12": ("h12_dinov2s_lp_r224", "sonda", ["20261011-174515", "20261011-174846", "20261011-175135", "20261011-175422", "20261011-175710"]),
}
HIP = {"baseline": "K3", "h01": "H01", "h02": "H02", "h08": "H08", "h10": "H10", "h11": "H11", "h12": "H12"}
TEST_SIRA = ["baseline", "h02", "h08", "h10", "h11", "h01", "h12"]


def run_id(k, s):
    v, _, ids = RESMI[k]
    return f"{ids[s]}_{v}_s{s}"


def cikarim(tur, kaynak_args, s, veri="resmi", test=False):
    return KY.FP32 + ["varyantlar/k5/k5_cikarim.py", "--tur", tur] + kaynak_args + ["--seed", str(s)] + \
        (["--veri", veri] if veri != "resmi" else []) + (["--test-kilidi-acik"] if test else []) + KY.OUT


def sar(veri, betik_args):
    return KY.FP32 + ["varyantlar/k5/c_bolme_sar.py", "--veri", veri, "--"] + betik_args + KY.OUT


def rakip(veri, s, epoch):
    return KY.FP32 + ["varyantlar/k5/c_rakip_kos.py", "--veri", veri, "--", "--seed", str(s), "--epoch", str(epoch)] + KY.OUT


def r18(s, epoch, d4=False):
    return ["varyantlar/r18_imagenet/ince_ayar.py", "--seed", str(s), "--epoch", str(epoch)] + (["--augment-d4"] if d4 else [])


def sonda(s, ek):
    return ["varyantlar/dinov2_sonda/sonda.py", "--seed", str(s), "--r", "224"] + ek


def vit_h10(s, epoch):
    return ["varyantlar/dinov2_ft/ince_ayar_vit.py", "--mod", "kismi", "--model", "b", "--son-k", "4", "--seed", str(s),
            "--epoch", str(epoch)] + KY.VIT_ORTAK + ["--mikro-bs", "32"]


# ───────────────────────── beklentiler (ön-kayıt; satırlardan ÖNCE yazılır) ─────────────────────────
ORTAK_NOT_TEST = [
    "K5-ONKAYIT §0 P6 (özdeşlik): argmax 1.003/1.003 aynı ve maks |Δp| ≤ 1e-5; baseline_c için yeniden çıkarım YOK → dosyadan ACC = metrics val_acc/test_acc (±1e-4). Herhangi bir ihlal → DUR, teşhis (a1 kapısı a3'ü başlatmaz; a3'te ihlal olursa k5_cikarim.py test açmadan rc=4).",
    "Yerel kanıt (2026-10-11, GTX 1650; scratchpad sınaması, ölçüm değil): dört çıkarım yolu (r18, vit_kismi fp16 token yolu, vit_tam, sonda) yerel kaynak koşumlarının tahmin_val.npy'sini BİT BİT yeniden üretti (maks |Δp| = 0). Colab'da aynı L4 sınıfı + aynı torch sürümü + deterministik ayarlar → aynı beklenir.",
    "Kapı metriği ozdeslik_p_maks_fark (yon min): BİLİNÇLİ OLARAK test metriği DEĞİL — aşama raporları ve defter çıktısı test sayısı göstermez; test sayıları yalnız scripts/k5_test.py --kip test çıktısında (bir kez).",
    "farkli_seedler_farkli KULLANILMADI: kapı metriği özdeşlik farkıdır (beklenen 0, seed'ler arasında aynı olabilir).",
]


def beklenti_test():
    kur = lambda n, test: [  # noqa: E731
        {"tur": "sonlu"},
        {"tur": "min_kosum", "n": n},
        {"tur": "ikincil_aralik", "metrik": "ozdeslik_gecti", "alt": 1, "ust": 1, "kaynak": "K5-ONKAYIT §0 P6"},
        {"tur": "ikincil_aralik", "metrik": "ozdeslik_argmax_oran", "alt": 1.0, "ust": 1.0, "kaynak": "P6: argmax 1.003/1.003"},
        {"tur": "ikincil_aralik", "metrik": "ozdeslik_p_maks_fark", "alt": 0.0, "ust": 1e-05, "kaynak": "P6: olasılık maks |Δ| ≤ 1e-5"},
        {"tur": "ikincil_aralik", "metrik": "ozdeslik_val_acc_fark", "alt": 0.0, "ust": 0.0001, "kaynak": "P6/§1 sıra 1: baseline dosyadan ACC = metrics (±1e-4)"},
        {"tur": "ikincil_aralik", "metrik": "val_n", "alt": 1003, "ust": 1003, "kaynak": "resmî val"},
    ] + ([
        {"tur": "ikincil_aralik", "metrik": "ozdeslik_test_acc_fark", "alt": 0.0, "ust": 0.0001, "kaynak": "§1 sıra 1: baseline dondurulmuş test_acc"},
        {"tur": "ikincil_aralik", "metrik": "ozdeslik_test_acc_son_fark", "alt": 0.0, "ust": 0.0001, "kaynak": "§1 sıra 1: tahmin_test_son ikincil satır"},
        {"tur": "ikincil_aralik", "metrik": "test_n", "alt": 2005, "ust": 2005, "kaynak": "resmî test (data_manifest)"},
        {"tur": "ikincil_aralik", "metrik": "test_sonlu", "alt": 1, "ust": 1, "kaynak": "olasılıklar sonlu ve toplamı 1"},
        {"tur": "ikincil_aralik", "metrik": "k5_test_kilidi", "alt": 1, "ust": 1, "kaynak": "a3 satırları --test-kilidi-acik"},
    ] if test else [
        {"tur": "ikincil_aralik", "metrik": "k5_test_kilidi", "alt": 0, "ust": 0, "kaynak": "a1 test AÇMAZ"},
    ])
    return {"_aciklama": "ÖN-KAYIT (kuyruk k5-test; K5 resmî test geçişi; deney-yürütücü 2026-10-11, hiçbir k5 koşumundan ÖNCE; "
                         "koşum sonrası DEĞİŞTİRİLMEZ, devam kuyruklarına BAYT BAYT kopyalanır). Kaynak: hipotezler/K5-ONKAYIT.md §0 P6, §1.",
            "_notlar": ORTAK_NOT_TEST, "metrik": "ozdeslik_p_maks_fark", "yon": "min",
            "asama": {"1": {"aciklama": "P6 YÜKLEME ÖZDEŞLİĞİ — 7 kol × 5 seed, yalnız val (test kapalı)", "kurallar": kur(35, False)},
                      "3": {"aciklama": "RESMÎ TEST ÇIKARIMI — 7 kol × 5 seed (--test-kilidi-acik); bu kapı yalnız bilgi (son aşama)",
                            "kurallar": kur(35, True)}}}


def beklenti_c(ad, kollar, veri):
    """C1/C2 kuyruk beklentisi. Aralıklar §4.2 M8: arıza alarmı baseline C-val ACC [0,74; 0,90], diğerleri [0,78; 0,96]."""
    son = "c1" if veri == "c1" else "c2"
    kaynak_a3 = "K5-ONKAYIT §4.2 M8 (C-val nv payı %78; ezber azaldığından alt uç geniş)" + ("" if veri == "c1" else "; §5 tarif ve seçim C1 ile aynı")
    a1 = [{"tur": "sonlu"}, {"tur": "min_kosum", "n": len(kollar)},
          {"tur": "kollar_farkli", "kaynak": "plasebo yakalayıcı (aynı seed'de kollar farklı olmalı)"}]
    a2 = [{"tur": "sonlu"}, {"tur": "min_kosum", "n": 2 * len(kollar)},
          {"tur": "kollar_farkli", "kaynak": "plasebo yakalayıcı"}]
    a3 = [{"tur": "sonlu"}, {"tur": "min_kosum", "n": 5 * len(kollar)}]
    for k in kollar:
        if k == "baseline":
            a1.append({"tur": "aralik", "varyant": f"baseline_{son}_duman", "alt": 0.68, "ust": 0.90,
                       "kaynak": "C-val çoğunluk tabanı 0,782 (448/573) − 0,10 duman payı (3 epoch); üst = §4.2 baseline üst 0,90"})
            a2.append({"tur": "aralik", "varyant": f"baseline_{son}_onucus", "alt": 0.70, "ust": 0.90,
                       "kaynak": "20 epoch: §4.2 alt 0,74 − 0,04 ön-uçuş payı; üst 0,90"})
            a3.append({"tur": "aralik", "varyant": f"baseline_{son}", "alt": 0.74, "ust": 0.90, "kaynak": kaynak_a3})
        else:
            a1.append({"tur": "aralik", "varyant": f"{k}_{son}_duman", "alt": 0.68, "ust": 0.96,
                       "kaynak": "C-val çoğunluk 0,782 − 0,10 (1 epoch / 256 örnek sonda); üst §4.2 0,96"})
            a2.append({"tur": "aralik", "varyant": f"{k}_{son}_onucus", "alt": 0.74, "ust": 0.96,
                       "kaynak": "5 epoch / iki-seed C*: §4.2 alt 0,78 − 0,04; üst 0,96"})
            a3.append({"tur": "aralik", "varyant": f"{k}_{son}", "alt": 0.78, "ust": 0.96, "kaynak": kaynak_a3})
            a3.append({"tur": "aralik", "varyant": f"{k}_{son}-test", "alt": 0.78, "ust": 0.96,
                       "kaynak": kaynak_a3 + " (C-test satırının val_acc'si = aynı modelin C-val yeniden çıkarımı)"})
    ortak = [
        {"tur": "ikincil_aralik", "metrik": "n_train", "alt": 8215, "ust": 8215, "kaynak": "C-train (Abhishek vd. 2025; data_manifest)"},
        {"tur": "ikincil_aralik", "metrik": "val_n", "alt": 573, "ust": 573, "kaynak": "C-val"},
        {"tur": "ikincil_aralik", "metrik": "tepe_gpu_bellek_mb", "alt": 1, "ust": 22500, "kaynak": "L4 23.034 MiB"},
        {"tur": "ikincil_aralik", "metrik": "secim_val_auc_farki", "alt": 0.0, "ust": 1e-06, "kaynak": "rakip köprüsü: resmî best_model val AUC = seçilen epoch"},
        {"tur": "ikincil_aralik", "metrik": "onbellek_tutarlilik_maks_fark", "alt": 0.0, "ust": 0.005, "kaynak": "H10 öldürme (f)"},
        {"tur": "ikincil_aralik", "metrik": "param_egitilen", "alt": 11180103, "ust": 28364551, "kaynak": "R18 11.180.103 / H10 28.364.551 (Colab K4)"},
    ]
    a1 += ortak + [
        {"tur": "ikincil_aralik", "metrik": "ilk_adim_loss", "alt": 1.5, "ust": 2.6, "kaynak": "≈ ln 7 (yeni 7 sınıflı kafa); kc1/kc2 a1"},
        {"tur": "ikincil_aralik", "metrik": "augment_d4", "alt": 1, "ust": 1, "kaynak": "H08: D4 devrede"},
        {"tur": "ikincil_aralik", "metrik": "augment_toplam_ornek", "alt": 8215, "ust": 8215, "kaynak": "H08: 1 epoch × C-train"},
        {"tur": "ikincil_aralik", "metrik": "param_omurga", "alt": 85724928, "ust": 85724928, "kaynak": "H01: ViT-B/14 r224"},
        {"tur": "ikincil_aralik", "metrik": "oznitelik_determinizm_maks_fark", "alt": 0.0, "ust": 1e-05, "kaynak": "H01 kartı"},
        {"tur": "ikincil_aralik", "metrik": "ayni_seed_tahmin_maks_fark", "alt": 0.0, "ust": 0.0, "kaynak": "H01 R5"},
    ]
    a2 += ortak + [
        {"tur": "ikincil_aralik", "metrik": "val_auc", "alt": 0.80, "ust": 0.995, "kaynak": "kc1/kc2 a2 bantları (0,85/0,88) − C-val nadir sınıf payı (df 5, vasc 7)"},
        {"tur": "ikincil_aralik", "metrik": "yakinsama_fark", "alt": 0.0, "ust": 0.005, "kaynak": "H01 öldürme (e)"},
        {"tur": "ikincil_aralik", "metrik": "seed_tahmin_en_kucuk_fark", "alt": 1e-06, "ust": 1.0, "kaynak": "H01 R5"},
    ]
    a3 += ortak + [
        {"tur": "ikincil_aralik", "metrik": "ozdeslik_gecti", "alt": 1, "ust": 1, "kaynak": "P6 eşi: C-test satırı modeli doğru yükledi (C-val bit-bit)"},
        {"tur": "ikincil_aralik", "metrik": "ozdeslik_p_maks_fark", "alt": 0.0, "ust": 1e-05, "kaynak": "P6 eşi"},
        {"tur": "ikincil_aralik", "metrik": "test_n", "alt": 1227, "ust": 1227, "kaynak": "C-test"},
        {"tur": "ikincil_aralik", "metrik": "test_sonlu", "alt": 1, "ust": 1, "kaynak": "olasılıklar sonlu"},
    ]
    return {"_aciklama": f"ÖN-KAYIT (kuyruk {ad}; K5 {veri.upper()}; deney-yürütücü 2026-10-11, hiçbir {veri} koşumundan ÖNCE; "
                         "koşum sonrası DEĞİŞTİRİLMEZ, devam kuyruklarına BAYT BAYT kopyalanır). Kaynak: hipotezler/K5-ONKAYIT.md "
                         + ("§4.1–4.2 (M8 aralıkları)" if veri == "c1" else "§5 (tarif/seçim C1 ile aynı → aynı M8 aralıkları)"),
            "_notlar": [
                "Aralıklar ARIZA ALARMIDIR, karar değil (§4.2 M8). Karar ölçütleri scripts/k5_test.py --kip " + son + " (C-test, bir kez).",
                "İlk satır (c_bolme_veri_*) veri kurar/doğrular; val_acc yazmaz → kapı sayımına girmez, rc≠0 ise mekanik İHLAL.",
                "C-test satırları (`<kol>_" + son + "-test`) a3'tedir; a3 son aşama olduğundan kapı yalnız bilgidir (defter hücre 10). "
                "k5_cikarim.py C-val özdeşliği geçmezse C-test açmaz (rc=4).",
                "rakip (baseline) C-test tahminini resmî protokolle kendisi yazar (her epoch test değerlendirir, seçime katmaz; §4.2 'rakip kodu dokunulmaz').",
                "ikincil_aralik varyant filtresi taşımaz → yalnız kola özgü anahtarlar ya da tüm kollar için aynı değerli anahtarlar (n_train, val_n) kullanıldı.",
                "Gruplama: kullanıcı kısıtı (aynı anda en çok 2 Colab oturumu) → C1 iki sekmede (A: baseline+H02+H01, B: H10+H08); C2 C1 bitince tek sekmede.",
            ],
            "metrik": "val_acc", "yon": "max",
            "asama": {"1": {"aciklama": "DUMAN — veri kurulumu + kol başına s0 kısa koşum", "kurallar": a1},
                      "2": {"aciklama": "ÖN-UÇUŞ — kol başına s0, s1 kısaltılmış", "kurallar": a2},
                      "3": {"aciklama": "TAM — s0–4 + C-test çıkarımı (bilgi)", "kurallar": a3}}}


# ───────────────────────── satırlar ─────────────────────────
def kuyruk_test():
    ad = "k5-test"
    for asama, test in ((1, False), (3, True)):
        for k in TEST_SIRA:
            v, tur, _ = RESMI[k]
            for s in SEED:
                ek = "-test" if test else "-ozdeslik"
                et = (f"K5 {'TEST çıkarımı' if test else 'P6 özdeşlik (val)'}: {v} s{s}"
                      + (" — yeniden çıkarım yok, dondurulmuş tahminler" if tur == "rakip" else ""))
                KY.ekle(ad, asama, v + ek, s, HIP[k], run_id(k, s), et, 15, None,
                        cikarim(tur, ["--kaynak", run_id(k, s)], s, test=test), gpu=(tur != "rakip"))


def c_kol(ad, veri, k, asama, s):
    """Bir C1/C2 eğitim satırı (tarif kc1/kc2 ile aynı)."""
    son = veri
    ebeveyn = run_id(k, s) if veri == "c1" else f"{k}_c1:s{s}"
    hip = HIP[k]
    if k == "baseline":
        ep, zs, kural, v = {1: (3, 15, KY.KURAL_RAKIP_A1, "_duman"), 2: (20, 30, KY.KURAL_RAKIP_A2, "_onucus"),
                            3: (100, 150, KY.KURAL_RAKIP_A3, "")}[asama]
        KY.ekle(ad, asama, f"baseline_{son}{v}", s, hip, ebeveyn, f"{veri.upper()} rakip 28 px {ep} ep (rakip kodu aynen; veri {veri})",
                zs, kural, rakip(veri, s, ep))
    elif k in ("h02", "h08"):
        ep, zs, v = {1: (1, 10, "_duman"), 2: (5, 15, "_onucus"), 3: (30, 40, "")}[asama]
        KY.ekle(ad, asama, f"{k}_{son}{v}", s, hip, ebeveyn, f"{veri.upper()} {hip}: R18-IN 224 {ep} ep{' + D4' if k == 'h08' else ''}",
                zs, KY.KURAL_FT, sar(veri, r18(s, ep, d4=(k == "h08"))))
    elif k == "h10":
        ep, zs, v = {1: (1, 15, "_duman"), 2: (5, 35, "_onucus"), 3: (15, 90, "")}[asama]
        KY.ekle(ad, asama, f"h10_{son}{v}", s, hip, ebeveyn, f"{veri.upper()} H10: DINOv2-B son 4 blok {ep} ep (blok-8 önbelleği {veri})",
                zs, KY.KURAL_FT, sar(veri, vit_h10(s, ep)))
    elif k == "h01":
        if asama == 1:
            ek, zs, v = ["--alt-kume", "256", "--C", "1.0", "--secim-seedleri", "0,1"], 15, "_duman"
        elif asama == 2:
            ek, zs, v = ["--C", "izgara", "--secim-seedleri", "0,1"], 15, "_onucus"
        else:
            ek, zs, v = ["--C", "izgara", "--secim-seedleri", "0,1,2,3,4"] + ([] if s == 0 else ["--lbfgs-yok"]), 20, ""
        KY.ekle(ad, asama, f"h01_{son}{v}", s, hip, ebeveyn, f"{veri.upper()} H01: DINOv2-B CLS sonda, C* C-val'de",
                zs, KY.KURAL_LP, sar(veri, sonda(s, ek)))


def c_test_satiri(ad, veri, k, s):
    tur = {"h01": "sonda", "h02": "r18", "h08": "r18", "h10": "vit_kismi"}[k]
    KY.ekle(ad, 3, f"{k}_{veri}-test", s, HIP[k], f"{k}_{veri}:s{s}", f"{veri.upper()} C-test çıkarımı (bir kez): {k}_{veri} s{s}", 15, None,
            cikarim(tur, ["--kaynak-varyant", f"{k}_{veri}", "--kaynak-kuyruk", f"k5-{veri}"], s, veri=veri, test=True))


def kuyruk_c(ad, veri, kollar_a3_sira):
    KY.ekle(ad, 1, f"c_bolme_veri_{veri}", 0, "K5", None, f"{veri.upper()} veri kurulumu/denetimi (c_bolme_npz.py --kip {veri})",
            70, None, ["python", "scripts/c_bolme_npz.py", "--kip", veri, "--out", "{RUN_DIR}/metrics.json"], gpu=False)
    for k in kollar_a3_sira:
        c_kol(ad, veri, k, 1, 0)
    for k in kollar_a3_sira:
        for s in (0, 1):
            c_kol(ad, veri, k, 2, s)
    for k in kollar_a3_sira:
        for s in SEED:
            c_kol(ad, veri, k, 3, s)
    for k in kollar_a3_sira:
        if k != "baseline":
            for s in SEED:
                c_test_satiri(ad, veri, k, s)


PLAN = {"k5-test": None,
        "k5-c1-A": ("c1", ["h01", "h02", "baseline"]),
        "k5-c1-B": ("c1", ["h08", "h10"]),
        "k5-c2": ("c2", ["h08", "h10", "baseline"])}


def main():
    zorla = "--zorla" in sys.argv
    for ad in PLAN:
        j, b = WS / "kuyruk" / f"{ad}.jsonl", WS / "kuyruk" / f"{ad}.beklenti.json"
        if (j.exists() or b.exists()) and not zorla:
            raise SystemExit(f"HATA: {j.name}/{b.name} zaten var (--zorla yalnız hiçbir k5 koşumu başlamadıysa)")
        baslat = list((WS / "runs" / "kuyruklar").glob(f"{ad}*.baslat.json")) if (WS / "runs" / "kuyruklar").exists() else []
        if baslat:
            raise SystemExit(f"HATA: {ad} başlamış — ön-kayıt ezilemez")
        j.unlink(missing_ok=True)
    # 1) beklentiler (ön-kayıt) — satırlardan ÖNCE
    (WS / "kuyruk" / "k5-test.beklenti.json").write_text(json.dumps(beklenti_test(), ensure_ascii=False, indent=1), encoding="utf-8")
    for ad, (veri, kollar) in ((k, v) for k, v in PLAN.items() if v):
        (WS / "kuyruk" / f"{ad}.beklenti.json").write_text(json.dumps(beklenti_c(ad, kollar, veri), ensure_ascii=False, indent=1),
                                                         encoding="utf-8")
    # 2) satırlar
    kuyruk_test()
    for ad, (veri, kollar) in ((k, v) for k, v in PLAN.items() if v):
        kuyruk_c(ad, veri, kollar)
    for ad in PLAN:
        n = sum(1 for l in (WS / "kuyruk" / f"{ad}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())
        print(f"{ad}: {n} satır")
    return 0


if __name__ == "__main__":
    sys.exit(main())
