# -*- coding: utf-8 -*-
"""
nitel_panel_val.py — Makale nitel paneli için VAL tabanlı görüntü seçimi + karo üretimi.
(makale için sonradan eklendi — deney-gelistirici, 2026-10-12; mevcut dosyalara dokunmaz)

Kural (M4 / brief madde 13): test tahminleri ve test görüntüleri AÇILMAZ. Yalnız
  - data/medmnist/dermamnist_224.npz → val_images / val_labels (varyantlar/ortak/veri.yukle, test kilidi açık değil)
  - runs_colab/<run_id>/tahmin_val.npy + etiket_val.npy (B, H02c, H08c, H10; 5 seed)
  - data/splits/dermamnist_lezyon_esleme.csv (veri.sizinti_maskesi("val") → S/N; image_id/lesion_id)
  - sonuclar/k5_dondurma.json (run id + tahmin_val SHA256), runs_colab/ARSIV_MANIFEST.json (SHA256)
  - sonuclar/hata_kirilim_ozet_*.json (hep-yanlış indeks denetimi, P5)

Seçim kuralı (ön-kayıt: makale WS notes/eksik-kod-teklifi.md, koşumdan önce yazıldı):
  c_a(i) = 5 seed'de doğru argmax sayısı; çoğunluk doğru ⇔ c ≥ 3; p̄_a(i) = gerçek sınıf ort. olasılığı.
  Yönlü kategoriler (N-a, N-b, S-a, S-b, DV-a, DV-b): kazanan c≥3 ∧ kaybeden c≤2;
    g = p̄_kazanan − p̄_kaybeden; g'si kategori medyanına en yakın k görüntü (eşitlikte küçük indeks).
  Z: dört kolun hepsinde c = 0; dört kol p̄ ortalaması çekirdek medyanına en yakın k görüntü.
  Çakışma: sabit sıra N-a, N-b, S-a, S-b, DV-a, DV-b, Z; görüntü ilk kategoride kalır.

Kullanım (DW kökünden):
  PYTHONIOENCODING=utf-8 env/.venv/Scripts/python.exe -I scripts/nitel_panel_val.py \
      --cikti-kok <makale_WS> [--k 2]
Çıktı: <makale_WS>/figures/data/nitel_panel_val.json, <makale_WS>/figures/tiles/nitel/val_<indeks>.png
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

DW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DW / "varyantlar"))
from ortak import veri  # noqa: E402

SINIF = veri.SINIF_KISA  # ["akiec","bcc","bkl","df","mel","nv","vasc"]
KOLLAR = {  # panel adı → k5_dondurma.json anahtarı
    "B": "baseline",
    "H02c": "h02",
    "H08c": "h08",
    "H10": "h10",
}
OZET_JSON = {
    "B": "hata_kirilim_ozet_baseline_colab.json",
    "H02c": "hata_kirilim_ozet_h02_r18in_ft224_colab.json",
    "H08c": "hata_kirilim_ozet_h08_r18in_ft224_d4_colab.json",
    "H10": "hata_kirilim_ozet_h10_dinov2b_k4_ft224.json",
}
KATEGORILER = [  # (kod, açıklama, alt küme, kazanan, kaybeden)
    ("N-a", "N: H08c çoğunluk doğru, H10 çoğunluk yanlış", "N", "H08c", "H10"),
    ("N-b", "N: H10 çoğunluk doğru, H08c çoğunluk yanlış", "N", "H10", "H08c"),
    ("S-a", "S: H08c çoğunluk doğru, H10 çoğunluk yanlış", "S", "H08c", "H10"),
    ("S-b", "S: H10 çoğunluk doğru, H08c çoğunluk yanlış", "S", "H10", "H08c"),
    ("DV-a", "df/vasc: H08c çoğunluk doğru, H02c çoğunluk yanlış", "DV", "H08c", "H02c"),
    ("DV-b", "df/vasc: H02c çoğunluk doğru, H08c çoğunluk yanlış", "DV", "H02c", "H08c"),
    ("Z", "ortak zor çekirdek: dört kol 5 seed'de de yanlış", "Z", None, None),
]
BASLIK = ("Görüntüler val kümesinden önceden yazılmış kuralla seçildi: her kategoride (S/N × hangi kolun "
          "5 seed'lik çoğunluk oyuyla doğru olduğu; df/vasc'ta H08c↔H02c; dört kolun beş seed'de de yanlış "
          "olduğu çekirdek) kazanan ve kaybeden kolun gerçek-sınıf ortalama olasılık farkı kategori medyanına "
          "en yakın iki görüntü (eşitlikte küçük indeks).")


def sha256(yol: Path) -> str:
    h = hashlib.sha256()
    with open(yol, "rb") as f:
        for p in iter(lambda: f.read(1 << 20), b""):
            h.update(p)
    return h.hexdigest()


def medyana_yakin(adaylar: np.ndarray, deger: np.ndarray, k: int, dislanan: set) -> tuple[list, float | None]:
    """adaylar: val indeksleri; deger: aynı sırada skor. Medyan TÜM adaylardan hesaplanır (dışlanmadan önce)."""
    if len(adaylar) == 0:
        return [], None
    med = float(np.median(deger))
    sira = sorted(range(len(adaylar)), key=lambda j: (abs(float(deger[j]) - med), int(adaylar[j])))
    secilen = []
    for j in sira:
        i = int(adaylar[j])
        if i in dislanan:
            continue
        secilen.append(i)
        if len(secilen) == k:
            break
    return secilen, med


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # -I altında PYTHONIOENCODING yok sayılır
    ap = argparse.ArgumentParser()
    ap.add_argument("--cikti-kok", required=True, help="makale workspace kökü")
    ap.add_argument("--k", type=int, default=2)
    a = ap.parse_args()
    cikti = Path(a.cikti_kok)
    karo_dir = cikti / "figures" / "tiles" / "nitel"
    veri_dir = cikti / "figures" / "data"
    karo_dir.mkdir(parents=True, exist_ok=True)
    veri_dir.mkdir(parents=True, exist_ok=True)
    denetim = {}

    # ---- veri (yalnız val) ----
    npz_bilgi = veri.npz_dogrula(224)
    v = veri.yukle(224, bolumler=("val",))["val"]
    X, y = v["x"], v["y"]
    N = len(y)
    S = veri.sizinti_maskesi("val")
    assert S.shape == (N,)
    meta = {}
    with open(veri.LEZYON_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "val":
                meta[int(r["index"])] = (r["image_id"], r["lesion_id"], r["dx"])
    assert all(SINIF[y[i]] == meta[i][2] for i in range(N)), "lezyon CSV dx ↔ npz etiket uyuşmuyor"
    denetim["P4_S_sayisi"] = int(S.sum())
    assert S.sum() == 348, f"P4: S sayısı {S.sum()} ≠ 348"

    # ---- tahminler + köken denetimi ----
    dond = json.load(open(DW / "sonuclar/k5_dondurma.json", encoding="utf-8"))["kaynak_tahmin_val_sha256"]
    arsiv = {d["yol"]: d["sha256"] for d in
             json.load(open(DW / "runs_colab/ARSIV_MANIFEST.json", encoding="utf-8"))["dosyalar"]}
    P, kosumlar = {}, {}
    for ad, anahtar in KOLLAR.items():
        runlar = sorted(dond[anahtar].keys(), key=lambda r: int(r.rsplit("_s", 1)[1]))
        assert len(runlar) == 5
        kosumlar[ad] = runlar
        ps = []
        for r in runlar:
            d = DW / "runs_colab" / r
            h = sha256(d / "tahmin_val.npy")
            assert h == dond[anahtar][r], f"P1: {r} tahmin_val SHA ≠ dondurma"
            assert h == arsiv[f"{r}/tahmin_val.npy"], f"P1: {r} tahmin_val SHA ≠ ARSIV_MANIFEST"
            yv = np.load(d / "etiket_val.npy").ravel().astype(int)
            assert np.array_equal(yv, y), f"P2: {r} etiket_val hizasız"
            p = np.load(d / "tahmin_val.npy").astype(np.float64)
            assert p.shape == (N, 7), f"P3: {r} boyut {p.shape}"
            assert np.allclose(p.sum(1), 1.0, atol=1e-4), f"P3: {r} olasılık toplamı"
            ps.append(p)
        P[ad] = np.stack(ps)  # (5, N, 7)
    denetim["P1_P2_P3"] = "20/20 tahmin_val SHA256 dondurma + ARSIV_MANIFEST ile aynı; etiket hizası ve olasılık toplamı tamam"

    T = {ad: P[ad].argmax(2) for ad in KOLLAR}           # (5, N)
    C = {ad: (T[ad] == y[None]).sum(0) for ad in KOLLAR}  # (N,)
    PG = {ad: P[ad][:, np.arange(N), y].mean(0) for ad in KOLLAR}  # p̄ gerçek sınıf

    # P5: hep-yanlış kümeleri dondurulmuş hata kırılımıyla aynı mı
    p5 = {}
    for ad, fn in OZET_JSON.items():
        o = json.load(open(DW / "sonuclar" / fn, encoding="utf-8"))
        assert o["kosumlar"] == kosumlar[ad], f"P5: {ad} koşum listesi farklı"
        ref = set(int(i) for i in o["zor"]["hep_yanlis_indeks"])
        bu = set(np.where(C[ad] == 0)[0].tolist())
        assert ref == bu, f"P5: {ad} hep-yanlış kümesi farklı ({len(ref)} ↔ {len(bu)})"
        p5[ad] = len(bu)
    denetim["P5_hep_yanlis"] = p5

    # P6: plasebo kol yok
    p6 = {}
    adlar = list(KOLLAR)
    for i1 in range(len(adlar)):
        for i2 in range(i1 + 1, len(adlar)):
            f = float((T[adlar[i1]] != T[adlar[i2]]).mean())
            p6[f"{adlar[i1]}~{adlar[i2]}"] = round(f, 4)
            assert f > 0, f"P6: {adlar[i1]} ve {adlar[i2]} argmax özdeş"
    denetim["P6_argmax_farkli_oran"] = p6

    def cogunluk(ad, i):
        oy = np.bincount(T[ad][:, i], minlength=7)
        ort = P[ad][:, i, :].mean(0)
        en = oy.max()
        aday = [c for c in range(7) if oy[c] == en]
        return max(aday, key=lambda c: (ort[c], -c))

    # ---- aday kümeleri ----
    alt = {"N": ~S, "S": S, "DV": np.isin(y, [3, 6]), "Z": np.ones(N, bool)}
    aday_say, secim, dislanan, medyan = {}, {}, set(), {}
    for kod, _, ak, kaz, kay in KATEGORILER:
        if kod == "Z":
            m = np.all([C[ad] == 0 for ad in KOLLAR], axis=0)
            idx = np.where(m)[0]
            skor = np.mean([PG[ad][idx] for ad in KOLLAR], axis=0) if len(idx) else np.array([])
        else:
            m = alt[ak] & (C[kaz] >= 3) & (C[kay] <= 2)
            idx = np.where(m)[0]
            skor = PG[kaz][idx] - PG[kay][idx]
        aday_say[kod] = int(len(idx))
        sec, med = medyana_yakin(idx, skor, a.k, dislanan)
        secim[kod] = sec
        medyan[kod] = med
        dislanan.update(sec)

    # bilgi: şefin örneğindeki B kıyası ve ek sayımlar
    dv = alt["DV"]
    bilgi = {
        "DV_H08c_dogru_B_yanlis_aday": int((dv & (C["H08c"] >= 3) & (C["B"] <= 2)).sum()),
        "DV_B_dogru_H08c_yanlis_aday": int((dv & (C["B"] >= 3) & (C["H08c"] <= 2)).sum()),
        "DV_destek": {"df": int((y == 3).sum()), "vasc": int((y == 6).sum())},
        "N_say": int((~S).sum()), "S_say": int(S.sum()),
        "ucunun_ortak_hep_yanlis_H02c_H08c_H10": int(np.all([C[ad] == 0 for ad in ("H02c", "H08c", "H10")], axis=0).sum()),
        "kategori_medyan_skor": medyan,
        "not": "Sayılar val (n=1003) üzerindedir; test'e bakılmadı. Panel betimseldir, karar taşımaz.",
    }

    # ---- kayıtlar + karolar ----
    goruntuler = []
    p7_ihlal = 0
    for kod, aciklama, ak, kaz, kay in KATEGORILER:
        for i in secim[kod]:
            karo = karo_dir / f"val_{i:04d}.png"
            img = np.ascontiguousarray(X[i])
            assert img.shape == (224, 224, 3) and img.dtype == np.uint8
            Image.fromarray(img).save(karo, optimize=False)
            geri = np.asarray(Image.open(karo).convert("RGB"))
            assert np.array_equal(geri, img), f"P8: karo {karo.name} npz ile özdeş değil"
            kol = {}
            for ad in KOLLAR:
                cg = cogunluk(ad, i)
                kol[ad] = {
                    "cogunluk_tahmin": SINIF[cg],
                    "dogru_seed_sayisi": int(C[ad][i]),
                    "ort_olasilik_gercek": round(float(PG[ad][i]), 4),
                    "ort_olasilik_cogunluk": round(float(P[ad][:, i, cg].mean()), 4),
                    "seed_tahminleri": [SINIF[t] for t in T[ad][:, i]],
                }
            if kaz is not None and not (C[kaz][i] >= 3 and C[kay][i] <= 2):
                p7_ihlal += 1
            goruntuler.append({
                "kategori": kod, "kategori_aciklama": aciklama, "val_indeks": int(i),
                "image_id": meta[i][0], "lesion_id": meta[i][1], "gercek_sinif": SINIF[y[i]],
                "S_N": "S" if S[i] else "N",
                "karo": str(karo.relative_to(cikti)).replace("\\", "/"),
                "g_kazanan_eksi_kaybeden": None if kaz is None else round(float(PG[kaz][i] - PG[kay][i]), 4),
                "kollar": kol,
            })
    assert p7_ihlal == 0
    denetim["P7_secim_kosulu"] = "tüm seçilenlerde kazanan c≥3, kaybeden c≤2"
    denetim["P8_karo"] = f"{len(goruntuler)} karo 224×224×3 uint8, npz pikselleriyle bayt-özdeş"

    # ---- E (ön-kayıtlı beklentiler) ----
    e = {
        "E1_S-b_gt_S-a": {"S-a": aday_say["S-a"], "S-b": aday_say["S-b"], "tuttu": aday_say["S-b"] > aday_say["S-a"]},
        "E2_N-a_ge_N-b_ya_da_yakin_(<=5)": {"N-a": aday_say["N-a"], "N-b": aday_say["N-b"],
                                            "tuttu": aday_say["N-a"] >= aday_say["N-b"] or abs(aday_say["N-a"] - aday_say["N-b"]) <= 5},
        "E3_DV-a_gt_DV-b": {"DV-a": aday_say["DV-a"], "DV-b": aday_say["DV-b"], "tuttu": aday_say["DV-a"] > aday_say["DV-b"]},
        "E4_1_le_Z_le_14": {"Z": aday_say["Z"], "tuttu": 1 <= aday_say["Z"] <= 14},
    }

    betik = Path(__file__).resolve()
    out = {
        "aciklama": "Nitel panel verisi (VAL). Makale için sonradan eklendi — deney-gelistirici. Test açılmadı.",
        "kaynak": {
            "betik": str(betik.relative_to(DW)).replace("\\", "/"),
            "betik_sha256": sha256(betik),
            "zaman": dt.datetime.now().isoformat(timespec="seconds"),
            "python": sys.version.split()[0], "numpy": np.__version__,
            "npz": npz_bilgi["npz"], "npz_md5": npz_bilgi["md5"], "npz_md5_resmi_eslesti": npz_bilgi["md5"] == npz_bilgi["md5_resmi"],
            "kollar": kosumlar,
            "tahmin_dosyasi": "tahmin_val.npy (seçilen epoch softmax; B için resmî val-AUC seçimi)",
            "tahmin_val_sha256_dogrulandi": True,
        },
        "lisans": "Görüntüler DermaMNIST (MedMNIST v2/v3, HAM10000 türevi), CC BY-NC 4.0 — atıf zorunlu, ticari kullanım yok.",
        "tanimlar": {
            "dogru_seed_sayisi": "5 seed içinde argmax'ı gerçek sınıfa eşit olan seed sayısı",
            "cogunluk_dogru": "dogru_seed_sayisi >= 3",
            "cogunluk_tahmin": "seed argmax modu; eşitlikte ort. olasılığı yüksek sınıf, sonra küçük sınıf indeksi",
            "ort_olasilik_gercek": "gerçek sınıf softmax olasılığının 5-seed ortalaması",
            "S_N": "S = görüntünün lezyonu resmî train'de var; N = yok (veri.sizinti_maskesi)",
        },
        "secim_olcutu": {
            "k": a.k,
            "kategoriler": [{"kod": k, "aciklama": ac, "kazanan": kz, "kaybeden": ky} for k, ac, _, kz, ky in KATEGORILER],
            "kural": "Yönlü kategoriler: g = p̄_kazanan − p̄_kaybeden; g'si kategori medyanına en yakın k görüntü "
                     "(eşitlikte küçük val indeksi). Z: dört kol p̄ ortalaması çekirdek medyanına en yakın k görüntü. "
                     "Çakışmada sabit sıra N-a, N-b, S-a, S-b, DV-a, DV-b, Z (görüntü ilk kategoride kalır). Seed'siz, deterministik.",
            "on_kayit": "makale WS notes/eksik-kod-teklifi.md (koşumdan önce yazıldı)",
        },
        "baslik_cumlesi": BASLIK,
        "kategori_aday_sayisi": aday_say,
        "bilgi": bilgi,
        "denetim_P": denetim,
        "beklenti_E": e,
        "goruntuler": goruntuler,
    }
    yol = veri_dir / "nitel_panel_val.json"
    with open(yol, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"yazıldı: {yol}")
    print("aday sayıları:", aday_say)
    print("E:", {k: v["tuttu"] for k, v in e.items()})
    print("seçim:", secim)


if __name__ == "__main__":
    main()
