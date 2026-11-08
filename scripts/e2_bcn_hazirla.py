# -*- coding: utf-8 -*-
"""
e2_bcn_hazirla.py — H15 (E2) a0 VERİ KAPISI ve ETİKETSİZ MODEL GİRDİSİ (CPU; GPU ve model YOK; hipotezler/H15.md §Aşama planı a0).

`kuyruk/e2-veri` (indir → onisleme → nn_tam) BİTTİKTEN SONRA, yerelde, bir kez koşulur. Model çıktısına hiç bakmaz.
Aşamalar (--asama hepsi = sırayla):
  kapi     K-a: `bcn20000_nn_tam.json` RMSE ≤ 5 çifti 0 değilse DUR (rc 2). K-b: figshare train CSV MD5'i; zip CRC32+boyut ile
           ikili-özdeş kopya grupları → 47 fazla + 2 çelişkili-etiket temsilcisi = 49 çıkarma; E1 kopyasız sayıları
           (11.933 görüntü / 3.451 lezyon; sınıf başına) kart ve `bcn20000_analiz.json` ile aynı olmalı; aynı etiketli farklı
           lezyon kimlikli kopyalardan lezyon birleştirmeleri = kartın iki çifti; eşleme birim sınaması (E1/E2 → DermaMNIST
           indeksleri). Çıktı: data/bcn20000/kopyasiz_liste.json, data/bcn20000/lezyon_kumeleri.json (yalnız dosya adı /
           lezyon kimliği; tanı YOK).
  cerceve  S9: yazarların siyah çerçeve kırpıcısı (varyantlar/e2/cerceve.py) tam zip'ten, 4 süreç. Yalnız kırpılan satırlar
           yazılır (kırpılmayanlar ANA merkez_kare ile aynı; her 50. görüntüde ana yolla bayt bayt sınanır).
  girdi    ETİKETSİZ model girdileri: e2-veri'nin 224 px npz'sinden yalnız `images` + `bcn_filename` (tanı anahtarına
           dokunulmaz); 8 HAM kalibrasyon görüntüsü BCN yoluyla (merkez_kare + PIL BILINEAR 224).
  beklenen kuyruk/e2_bcn_beklenen.json: dosya + içerik SHA256'ları, n = 12.413, sıra SHA256, kopyasız liste / lezyon kümesi
           SHA256, eşleme tabloları, sayılar, kapı sonuçları, çerçeve oranı, cv2 sürümü, kırpıcı SHA256. Rapor data/bcn20000/e2_a0_rapor.md.
Çıkış: 0 GEÇTİ · 2 DUR (K-a / K-b).
Kullanım (WS kökü):
  PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe scripts/e2_bcn_hazirla.py --asama hepsi
  (sınama: --zip / --mk-npz / --nn-json / --cikti-dizin / --beklenen-cikti / --kok-cikti ile geçici yollara)
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "varyantlar"))
from e2 import bcn_veri as BV  # noqa: E402

KOK = WS / "data" / "bcn20000"
META = KOK / "meta"
CSV_MD5 = "5fd3522a69512dcf5b3c225dc1664ec6"     # figshare computed_md5 (bcn_20k_train.csv)
DM = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]   # DermaMNIST indeks sırası (ortak/veri.py SINIF_KISA)
BIREBIR = {"NV": "nv", "MEL": "mel", "BCC": "bcc", "BKL": "bkl", "DF": "df", "VASC": "vasc"}
ESLEME = {"E1": dict(BIREBIR, AK="akiec"), "E2": dict(BIREBIR, AK="akiec", SCC="akiec")}
KART = {"E1_kopyasiz_goruntu": 11933, "E1_kopyasiz_lezyon": 3451, "cikarma": 49, "kopya_fazla": 47, "celiskili_temsilci": 2,
        "birlesik": [["BCN_0002049", "BCN_0002919"], ["BCN_0001604", "BCN_0004816"]], "n": 12413}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sha(p: Path) -> str:
    return BV.dosya_sha256(p)


def csv_oku(p):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def dur(msg, rapor=None):
    log(f"DUR: {msg}")
    if rapor is not None:
        rapor.setdefault("dur", []).append(msg)
    raise SystemExit(2)


# ───────────────────────── kapı ─────────────────────────
def esleme_birim_sinama() -> dict:
    sys.path.insert(0, str(WS / "varyantlar"))
    from ortak import veri as V
    assert list(V.SINIF_KISA) == DM, V.SINIF_KISA
    E1 = {k: DM.index(v) for k, v in ESLEME["E1"].items()}
    E2 = {k: DM.index(v) for k, v in ESLEME["E2"].items()}
    assert sorted(E1.values()) == list(range(7)), "E1 7 sınıfı birebir kapsamalı"
    assert E1 == {"NV": 5, "MEL": 4, "BCC": 1, "BKL": 2, "DF": 3, "VASC": 6, "AK": 0}, E1
    assert E2 == dict(E1, SCC=0) and "SCC" not in E1
    return {"E1": E1, "E2": E2, "dm_siniflar": DM, "gecti": True}


def kapi(a, R: dict) -> dict:
    nn_yol = Path(a.nn_json) if a.nn_json else KOK / "bcn20000_nn_tam.json"
    if not nn_yol.exists():
        dur(f"{nn_yol} yok — e2-veri nn_tam bitmeden a0 koşulmaz", R)
    nn = json.loads(nn_yol.read_text(encoding="utf-8"))
    R["nn_tam"] = {k: v for k, v in nn.items() if k != "en_yakin_30"}
    R["nn_tam"]["sha256"] = sha(nn_yol)
    if nn.get("n_bcn") != KART["n"]:
        dur(f"nn_tam n_bcn {nn.get('n_bcn')} ≠ {KART['n']}", R)
    if int(nn["esik5_alti"]) != 0:
        dur(f"K-a SIZINTI: RMSE ≤ 5 BCN↔DermaMNIST çifti {nn['esik5_alti']} > 0 — E2 başlamaz, kullanıcıya rapor (H15b)", R)
    R["K-a"] = {"esik5_alti": 0, "esik10_alti_bilgi": int(nn.get("esik10_alti", -1)), "gecti": True}
    csv_yol = META / "bcn_20k_train.csv"
    md5 = hashlib.md5(csv_yol.read_bytes()).hexdigest()
    if md5 != CSV_MD5:
        dur(f"K-b: bcn_20k_train.csv MD5 {md5} ≠ figshare {CSV_MD5}", R)
    tr = csv_oku(csv_yol)
    lab = {r["bcn_filename"]: r for r in tr}
    if len(lab) != KART["n"]:
        dur(f"K-b: CSV {len(lab)} satır ≠ {KART['n']}", R)
    grp = defaultdict(list)
    for r in csv_oku(META / "zip_icerik_train.csv"):
        grp[(r["crc32"], r["boyut"])].append(Path(r["ad"]).name)
    dup = sorted(sorted(v) for v in grp.values() if len(v) > 1)
    kopya_fazla = sorted(x for v in dup for x in v[1:])
    celiski_grup = [v for v in dup if len({lab[x]["diagnosis"] for x in v}) > 1]
    celiski = sorted(x for v in celiski_grup for x in v)
    dislanan = sorted(set(kopya_fazla) | set(celiski))
    # aynı etiketli, farklı lezyon kimlikli kopyalar → lezyon birleştirmesi (union-find)
    ebeveyn = {}

    def bul(u):
        while ebeveyn.get(u, u) != u:
            u = ebeveyn[u]
        return u
    for v in dup:
        lz = sorted({lab[x]["lesion_id"] for x in v})
        if len(lz) > 1 and len({lab[x]["diagnosis"] for x in v}) == 1:
            for b_ in lz[1:]:
                ebeveyn[bul(b_)] = bul(lz[0])
    kume = defaultdict(set)
    for u in set(ebeveyn) | set(ebeveyn.values()):
        kume[bul(u)].add(u)
    birlesik = sorted(sorted(s) for s in kume.values() if len(s) > 1)
    if birlesik != sorted(sorted(p) for p in KART["birlesik"]):
        dur(f"K-b: lezyon birleştirmeleri {birlesik} ≠ kart {KART['birlesik']}", R)
    if (len(kopya_fazla), len(dislanan), len(celiski_grup)) != (KART["kopya_fazla"], KART["cikarma"], KART["celiskili_temsilci"]):
        dur(f"K-b: kopya fazla {len(kopya_fazla)} / çıkarma {len(dislanan)} / çelişkili grup {len(celiski_grup)} ≠ kart", R)
    es = esleme_birim_sinama()
    R["esleme_sinama"] = es
    analiz = json.loads((KOK / "bcn20000_analiz.json").read_text(encoding="utf-8"))["esleme_secenekleri_figshare_train"]
    sayilar = {}
    for ad in ("E1", "E2"):
        for tem in ("ham", "kopyasiz"):
            rows = [r for r in tr if r["diagnosis"] in ESLEME[ad] and (tem == "ham" or r["bcn_filename"] not in set(dislanan))]
            g = Counter(ESLEME[ad][r["diagnosis"]] for r in rows)
            lz = Counter(ESLEME[ad][d] for d in {r["lesion_id"]: r["diagnosis"] for r in rows}.values())
            sayilar[f"{ad}|{tem}"] = {"goruntu": len(rows), "lezyon": len({r["lesion_id"] for r in rows}),
                                      "sinif_goruntu": {k: g.get(k, 0) for k in DM}, "sinif_lezyon": {k: lz.get(k, 0) for k in DM}}
            ref = analiz[f"{'E1_AK>akiec_SCC_disla' if ad == 'E1' else 'E2_AK+SCC>akiec'}|{tem}"]
            for q in ("goruntu", "lezyon", "sinif_goruntu", "sinif_lezyon"):
                if sayilar[f"{ad}|{tem}"][q] != ref[q]:
                    dur(f"K-b: {ad}|{tem} {q} {sayilar[f'{ad}|{tem}'][q]} ≠ manifest {ref[q]}", R)
    s = sayilar["E1|kopyasiz"]
    if (s["goruntu"], s["lezyon"]) != (KART["E1_kopyasiz_goruntu"], KART["E1_kopyasiz_lezyon"]):
        dur(f"K-b: E1 kopyasız {s['goruntu']}/{s['lezyon']} ≠ kart 11.933/3.451", R)
    R["K-b"] = {"gecti": True, "csv_md5": md5, "sayilar": sayilar}
    out_dir = Path(a.kok_cikti) if a.kok_cikti else KOK
    kl = {"kural": "zip CRC32+boyut ile ikili-özdeş kopya gruplarından her birinin sıralı ilk dosyası tutulur (fazlalar çıkar); "
                   "etiketi çelişen grupların bütün üyeleri çıkar (tutulan temsilciler dahil). Model çıktısına bakılmaz (H15 §2).",
          "kopya_grubu": len(dup), "kopya_fazla": kopya_fazla, "celiskili_grup_uyeleri": celiski, "dislanan": dislanan,
          "n_dislanan": len(dislanan), "kaynak": "data/bcn20000/meta/zip_icerik_train.csv + bcn_20k_train.csv (MD5 doğrulandı)"}
    lk = {"kural": "aynı görüntüyü paylaşan AYNI etiketli farklı lezyon kimlikleri tek bootstrap kümesi (H15 §2)", "birlesik": birlesik}
    for ad, d in (("kopyasiz_liste.json", kl), ("lezyon_kumeleri.json", lk)):
        (out_dir / ad).write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        R.setdefault("dosyalar_a0", {})[ad] = sha(out_dir / ad)
    log(f"kapı GEÇTİ: K-a esik5 0 (esik10 {R['K-a']['esik10_alti_bilgi']}), K-b E1 kopyasız {s['goruntu']}/{s['lezyon']}, "
        f"çıkarma {len(dislanan)}, birleşik {birlesik}")
    return R


# ───────────────────────── çerçeve ─────────────────────────
def zip_sira(zip_yol: Path):
    import zipfile
    z = zipfile.ZipFile(zip_yol)
    adlar = {Path(i.filename).name: i.filename for i in z.infolist() if not i.is_dir()}
    tr = {r["bcn_filename"] for r in csv_oku(META / "bcn_20k_train.csv")}
    return sorted(fn for fn in tr if fn in adlar), adlar


def cerceve(a, R: dict):
    from concurrent.futures import ProcessPoolExecutor
    import cv2
    from e2 import cerceve as CV
    zip_yol = Path(a.zip) if a.zip else KOK / "zip" / "BCN_20k_train.zip"
    lst, adlar = zip_sira(zip_yol)
    mk_yol = Path(a.mk_npz) if a.mk_npz else KOK / "bcn20000_train_merkez_kare_224.npz"
    with np.load(mk_yol, allow_pickle=False) as z:     # yalnız images + bcn_filename (tanı anahtarına dokunulmaz)
        mk_ad = z["bcn_filename"].tolist()
        if mk_ad != lst:
            dur("çerçeve: zip sırası e2-veri merkez_kare sırasıyla aynı değil", R)
        MK = z["images"]
    es = CV.es_degerlik_sina()
    if es["esit"] != es["n"]:
        dur(f"çerçeve: oran_vek ↔ özgün intensity_ratio eşdeğerliği bozuk {es}", R)
    t0 = time.time()
    idx, X, bilgi, kontrol_fark, kontrol_n = [], [], [None] * len(lst), 0, 0
    gorev = [(i, adlar[fn], i % 50 == 0) for i, fn in enumerate(lst)]
    with ProcessPoolExecutor(a.isci, initializer=CV.isci_baslat, initargs=(str(zip_yol),)) as ex:
        for j, (i, x, b, mk) in enumerate(ex.map(CV.isci, gorev, chunksize=16)):
            bilgi[i] = {"ad": lst[i], **b}
            if x is not None:
                idx.append(i); X.append(x)
            if mk is not None:
                kontrol_n += 1
                kontrol_fark += int(not np.array_equal(mk, MK[i]))
            if j % 1000 == 0:
                log(f"  çerçeve {j}/{len(lst)} (kırpılan {len(idx)}) {time.time() - t0:.0f}s")
    if kontrol_fark:
        dur(f"çerçeve: ANA yol yeniden üretimi {kontrol_fark}/{kontrol_n} görüntüde e2-veri merkez_kare ile aynı değil", R)
    neden = Counter(b["neden"] for b in bilgi)
    boy = Counter(b["boyut"] for b in bilgi)
    out = Path(a.cikti_dizin) if a.cikti_dizin else BV.DIZIN
    out.mkdir(parents=True, exist_ok=True)
    Xa = np.stack(X) if X else np.zeros((0, 224, 224, 3), np.uint8)
    np.savez_compressed(out / BV.NPZ["cerceve_fark"], idx=np.array(idx, np.int64), images=Xa)
    (out / "cerceve_kayit.json").write_text(json.dumps(bilgi, ensure_ascii=False), encoding="utf-8")
    R["cerceve"] = {"n": len(lst), "kirpilan": len(idx), "oran": round(len(idx) / len(lst), 4), "neden": dict(neden),
                    "boyutlar": dict(boy), "yon_uyusmaz": sum(b["yon_uyusmaz"] for b in bilgi),
                    "ana_yol_kontrol": {"n": kontrol_n, "fark": kontrol_fark}, "es_degerlik": es, "cv2": cv2.__version__,
                    "image_cropper_sha256": sha(WS / "varyantlar" / "e2" / "ucuncu" / "image_cropper.py"),
                    "cerceve_py_sha256": sha(WS / "varyantlar" / "e2" / "cerceve.py"), "sure_sn": round(time.time() - t0, 1),
                    "kayit_sha256": sha(out / "cerceve_kayit.json")}
    log(f"çerçeve: {len(idx)}/{len(lst)} kırpıldı ({R['cerceve']['oran']}), boyutlar {dict(boy)}, {R['cerceve']['sure_sn']}s")
    return R


# ───────────────────────── girdi ─────────────────────────
def girdi(a, R: dict):
    from PIL import Image
    from e2 import cerceve as CV
    mk_yol = Path(a.mk_npz) if a.mk_npz else KOK / "bcn20000_train_merkez_kare_224.npz"
    js = mk_yol.with_suffix(".json")
    if js.exists() and json.loads(js.read_text(encoding="utf-8")).get("sha256") not in (None, sha(mk_yol)):
        dur(f"girdi: {mk_yol.name} SHA256 e2-veri kaydıyla ({js.name}) uyuşmuyor", R)
    with np.load(mk_yol, allow_pickle=False) as z:
        X, adlar = z["images"], z["bcn_filename"]
    if not a.mk_npz and (len(X) != KART["n"] or X.shape[1:] != (224, 224, 3)):
        dur(f"girdi: merkez_kare {X.shape} ≠ ({KART['n']}, 224, 224, 3)", R)
    out = Path(a.cikti_dizin) if a.cikti_dizin else BV.DIZIN
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / BV.NPZ["merkez_kare"], images=X, bcn_filename=adlar)
    kl = json.loads((KOK / "kalibrasyon_ham" / "kalibrasyon_listesi.json").read_text(encoding="utf-8"))["goruntuler"]
    K_ = np.stack([CV.merkez_kare_bayt((KOK / "kalibrasyon_ham" / f"{g['image_id']}.jpg").read_bytes()) for g in kl])
    kidx = np.array([int(g["index"]) for g in kl], np.int64)
    assert all(g["split"] == "train" for g in kl), "kalibrasyon yalnız DermaMNIST train"
    with np.load(WS / "data" / "medmnist" / "dermamnist_224.npz") as z:
        ref = z["train_images"][kidx]
    rmse = [float(np.sqrt(np.mean((K_[i].astype(np.float64) - ref[i].astype(np.float64)) ** 2))) for i in range(len(kidx))]
    np.savez_compressed(out / BV.NPZ["kal"], images=K_, idx=kidx)
    R["girdi"] = {"kaynak_npz": str(mk_yol.relative_to(WS)) if mk_yol.is_relative_to(WS) else str(mk_yol),
                  "kaynak_npz_sha256": sha(mk_yol), "n": int(len(X)),
                  "kal": {"n": int(len(kidx)), "idx": kidx.tolist(), "rmse_ort": round(float(np.mean(rmse)), 3),
                          "rmse_maks": round(float(np.max(rmse)), 3), "beklenen_rmse_ort": 1.76}}
    log(f"girdi: merkez_kare {X.shape}, kalibrasyon RMSE ort {np.mean(rmse):.3f} (kalibrasyonda 1,76)")
    return R


# ───────────────────────── beklenen ─────────────────────────
def beklenen(a, R: dict):
    out = Path(a.cikti_dizin) if a.cikti_dizin else BV.DIZIN
    with np.load(out / BV.NPZ["merkez_kare"], allow_pickle=False) as z:
        X, adlar = z["images"], z["bcn_filename"]
    with np.load(out / BV.NPZ["cerceve_fark"], allow_pickle=False) as z:
        idx, Xf = z["idx"], z["images"]
    with np.load(out / BV.NPZ["kal"], allow_pickle=False) as z:
        K_, kidx = z["images"], z["idx"]
    C = X.copy()
    C[idx] = Xf
    kok_out = Path(a.kok_cikti) if a.kok_cikti else KOK
    B = {"_aciklama": "H15 (E2) a0 donmuş veri beklentisi (scripts/e2_bcn_hazirla.py). E2 çıkarımı, KİLİT ve analiz bu dosyayla "
                      "doğrular; uyuşmazlık → DUR (K-b). Model çıktısına bakılmadan yazıldı.",
         "zaman": datetime.now().isoformat(timespec="seconds"), "n": int(len(X)),
         "dosyalar": {ad: {"sha256": sha(out / ad), "bayt": (out / ad).stat().st_size} for ad in BV.NPZ.values()},
         "icerik": {"merkez_kare": BV.icerik_sha256(X), "cerceve": BV.icerik_sha256(C), "sira": BV.icerik_sha256(adlar),
                    "kal_images": BV.icerik_sha256(K_)},
         "kal_idx": kidx.tolist(), "sira_sha256": BV.sira_sha256(adlar.tolist()),
         "kopyasiz_liste": {"yol": "data/bcn20000/kopyasiz_liste.json", "sha256": sha(kok_out / "kopyasiz_liste.json")},
         "lezyon_kumeleri": {"yol": "data/bcn20000/lezyon_kumeleri.json", "sha256": sha(kok_out / "lezyon_kumeleri.json")},
         "esleme": {k: {kk: DM.index(vv) for kk, vv in v.items()} for k, v in ESLEME.items()}, "dm_siniflar": DM,
         "etiket_kaynagi": {"yol": "data/bcn20000/meta/bcn_20k_train.csv", "md5": CSV_MD5,
                            "sha256": "0f219633f49c07b0a1c844c4faa0fa1fc26690743f2c44dc7337a7a4135bc40b"},
         "sayilar": R.get("K-b", {}).get("sayilar"), "kapilar": {"K-a": R.get("K-a"), "K-b": {"gecti": R.get("K-b", {}).get("gecti")}},
         "nn_tam": R.get("nn_tam"), "cerceve": R.get("cerceve"), "girdi": R.get("girdi"),
         "esleme_sinama": R.get("esleme_sinama")}
    if not (B["kapilar"]["K-a"] and B["kapilar"]["K-a"].get("gecti") and B["kapilar"]["K-b"]["gecti"]):
        dur("beklenen: K-a/K-b kapıları bu çağrıda geçmedi (önce --asama kapi)", R)
    hedef = Path(a.beklenen_cikti) if a.beklenen_cikti else BV.BEKLENEN
    hedef.write_text(json.dumps(B, ensure_ascii=False, indent=1), encoding="utf-8")
    md = ["# H15 / E2 — a0 veri kapısı raporu", f"*{B['zaman']}* · `scripts/e2_bcn_hazirla.py` · beklenen `{hedef.name}` "
          f"SHA256 `{sha(hedef)[:16]}…`", "",
          f"- **K-a (sızıntı):** RMSE ≤ 5 çifti **0** · ≤ 10: {B['kapilar']['K-a']['esik10_alti_bilgi']} (bilgi) · "
          f"koyu köşe oranı (28 px, <40) BCN {B['nn_tam'].get('koyu_kose_orani_bcn_28px(<40)')} / DermaMNIST "
          f"{B['nn_tam'].get('koyu_kose_orani_dermamnist_train_28px(<40)')}",
          f"- **K-b:** CSV MD5 eşleşti · E1 kopyasız **{B['sayilar']['E1|kopyasiz']['goruntu']} görüntü / "
          f"{B['sayilar']['E1|kopyasiz']['lezyon']} lezyon** · çıkarma 49 · lezyon birleştirme 2 çift · eşleme birim sınaması geçti",
          f"- **Çerçeve (S9):** yazar kırpıcısı {B['cerceve']['kirpilan']}/{B['n']} görüntüyü kırptı (oran {B['cerceve']['oran']}); "
          f"nedenler {B['cerceve']['neden']}; boyutlar {B['cerceve']['boyutlar']}; yön uyuşmazlığı {B['cerceve']['yon_uyusmaz']}; "
          f"cv2 {B['cerceve']['cv2']}",
          f"- **Kalibrasyon girdisi:** 8 HAM görüntü RMSE ort {B['girdi']['kal']['rmse_ort']} (kalibrasyonda 1,76)",
          "", "| dosya | bayt | SHA256 |", "|---|---|---|"]
    md += [f"| `{k}` | {v['bayt']:,} | `{v['sha256']}` |" for k, v in B["dosyalar"].items()]
    md += ["", "İçerik SHA256: " + ", ".join(f"{k} `{v[:16]}…`" for k, v in B["icerik"].items()),
           "", "Sonraki adım (şef): `colab/hat/e2_veri_kopya.py` ile üç npz'yi Drive `yz/3-1/veri_dis/bcn20000/e2_girdi/`'ye "
           "kopyala (geri okuyup SHA256 doğrular), sonra `colab/hat/drive_paketle.py`. Manifestin (data/dis_veri_manifest.md) "
           "tam veri satırları bu rapordan güncellenir."]
    (kok_out / "e2_a0_rapor.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (kok_out / "e2_a0_rapor.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    log(f"beklenen yazıldı: {hedef} · rapor {kok_out / 'e2_a0_rapor.md'}")
    return R


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--asama", required=True, choices=["kapi", "cerceve", "girdi", "beklenen", "hepsi"])
    p.add_argument("--isci", type=int, default=4)
    p.add_argument("--zip", default=None)
    p.add_argument("--mk-npz", default=None, help="sınama: merkez_kare kaynak npz (varsayılan e2-veri çıktısı)")
    p.add_argument("--nn-json", default=None)
    p.add_argument("--cikti-dizin", default=None, help="sınama: model girdisi dizini (varsayılan data/bcn20000/e2_girdi)")
    p.add_argument("--kok-cikti", default=None, help="sınama: kopyasız liste / rapor dizini (varsayılan data/bcn20000)")
    p.add_argument("--beklenen-cikti", default=None, help="sınama: beklenen JSON yolu (varsayılan kuyruk/e2_bcn_beklenen.json)")
    p.add_argument("--metrics", default=None)
    a = p.parse_args()
    t0 = time.time()
    R = {"zaman": datetime.now().isoformat(timespec="seconds"), "betik_sha256": sha(Path(__file__))}
    adimlar = ["kapi", "cerceve", "girdi", "beklenen"] if a.asama == "hepsi" else [a.asama]
    ara = (Path(a.kok_cikti) if a.kok_cikti else KOK) / "e2_a0_ara.json"   # adımlar ayrı çağrılırsa ara sonuçlar
    if a.asama != "hepsi" and ara.exists():
        R = {**json.loads(ara.read_text(encoding="utf-8")), **R}
    if a.asama == "beklenen" and "K-a" not in R:
        adimlar = ["kapi", "beklenen"]   # kapı sonuçları beklenen dosyasına girer (hızlı; CPU)
    for adim in adimlar:
        {"kapi": kapi, "cerceve": cerceve, "girdi": girdi, "beklenen": beklenen}[adim](a, R)
        ara.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if a.metrics:
        Path(a.metrics).write_text(json.dumps({"asama": a.asama, "sure_sn": round(time.time() - t0, 1), "gecti": 1},
                                              ensure_ascii=False), encoding="utf-8")
    log(f"a0 {a.asama} TAMAM ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
