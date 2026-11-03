#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
c_bolme_npz.py — K5-ONKAYIT §4.1 C1 VERİ YAPIMI (yapım kuralı ön-kayıtta dondu) + §5 C2 verisinin hazırlık denetimi.

!!! SIRA KURALI (§4.1): C-train resmî testin 778 ve resmî val'in 430 görüntüsünü İÇERİR. Bu betik resmî TEST görüntü ve
etiketlerini okur → YALNIZ resmî test geçişi bitip çıktıları SHA256 ile dondurulduktan SONRA koşulur (k5-c1 kuyruklarının ilk
satırı; kuyruk şef izni `kuyruk/k5-c1.izin.json` olmadan başlamaz). Test geçişinden önce `--kip c1` ÇAĞRILMAZ.

--kip c1  Yapım: data/medmnist/dermamnist{,_224}.npz (resmî MD5) satırları data/splits/dermamnist_c_esleme.csv
          (split_C, split_resmi, index_resmi) ile CSV SIRASIYLA (= C npz sırası; data_manifest §DermaMNIST-C) yeniden dizilir →
          data/dermamnist_c_resmipiksel_{28,224}.npz: {train,val,test}_{images,labels} (+ {bolum}_index_resmi int64,
          {bolum}_split_resmi uint8 0=train 1=val 2=test). Yanına .json meta (dosya + içerik SHA256, denetimler).
          Denetimler (DUR = rc 2): sayılar 8.215/573/1.227 · etiketler = C etiketi (dermamnist_corrected_28.npz labels VE CSV dx) ·
          lezyon ayrıklığı train∩val = train∩test = val∩test = 0 · C-test görüntüleri resmî testin aynı indeksinin bayt-özdeş
          kopyası (28 ve 224) · 28 ↔ 224 etiketleri aynı · içerik SHA256 = kuyruk/k5_c_bolme_beklenen.json (yoksa DUR;
          --ilk-yapim ile şef yerelde test geçişinden sonra bir kez üretip yazar).
--kip c2  C'nin yayımlanmış npz'leri (arka plan indirmesi; defter hücre 6) gelene kadar bekler, Zenodo MD5'ini doğrular.
Her iki kip DermaMNIST-C indirmesini en çok --bekle-dk dakika bekler (progress.json yazarak; kosum.py takılma kuralı).
Çıktı: {RUN_DIR}/metrics.json (denetim_gecti, n_*, süre) — kosum.py damgası için. Kullanım:
  python scripts/c_bolme_npz.py --kip c1 --out {RUN_DIR}/metrics.json
  python scripts/c_bolme_npz.py --kip c1 --ilk-yapim --out <dizin>/metrics.json   (YALNIZ şef, test geçişinden SONRA, yerelde)
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "varyantlar"))
from k5.c_bolme import C2_MD5, C_ESLEME, BEKLENEN, dosya_hash, icerik_sha256  # noqa: E402

DX = {"akiec": 0, "bcc": 1, "bkl": 2, "df": 3, "mel": 4, "nv": 5, "vasc": 6}
RESMI_MD5 = {28: "0744692d530f8e62ec473284d019b0c7", 224: "8974907d8e169bef5f5b96bc506ae45d"}
SPLIT_KOD = {"train": 0, "val": 1, "test": 2}
BEKLENEN_N = {"train": 8215, "val": 573, "test": 1227}


def atomik_json(yol: Path, veri):
    tmp = yol.with_suffix(yol.suffix + ".tmp")
    tmp.write_text(json.dumps(veri, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, yol)


def bekle(dosyalar: dict, dk: float, run_dir: Path):
    t0 = time.time()
    while True:
        eksik = [str(p) for p, b in dosyalar.items() if not (p.exists() and p.stat().st_size == b)]
        if not eksik:
            return round(time.time() - t0, 1)
        if time.time() - t0 > dk * 60:
            raise SystemExit(f"HATA: {dk} dk içinde gelmedi: {eksik} (durum/veri_c_<oturum>.json'a bak)")
        atomik_json(run_dir / "progress.json", {"adim": 0, "asama_ad": "veri_bekle", "eksik": eksik,
                                                 "gecen_sn": round(time.time() - t0, 1)})
        time.sleep(15)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kip", choices=["c1", "c2"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bekle-dk", type=float, default=60)
    ap.add_argument("--ilk-yapim", action="store_true", help="şef: donmuş içerik SHA256'sını kuyruk/k5_c_bolme_beklenen.json'a yaz")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    run_dir = out.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cdir = WS / "data" / "dermamnist_c"
    c_dosya = {cdir / "dermamnist_corrected_28.npz": 20698236, cdir / "dermamnist_corrected_224.npz": 1131258316}
    bekleme = bekle(c_dosya, a.bekle_dk, run_dir)
    for s in (28, 224):
        md5, _ = dosya_hash(cdir / f"dermamnist_corrected_{s}.npz")
        if md5 != C2_MD5[s]:
            raise SystemExit(f"HATA: dermamnist_corrected_{s}.npz MD5 {md5} ≠ Zenodo {C2_MD5[s]}")
    m = {"denetim_gecti": 0, "c_veri_bekleme_sn": bekleme, "c2_md5_dogrulandi": 1, "cihaz": "cpu"}
    if a.kip == "c2":
        m["denetim_gecti"] = 1
        m["egitim_sure_sn"] = m["sure_sn"] = round(time.time() - t0, 1)
        atomik_json(out, m)
        print(f"[c_bolme_npz] C2 verisi hazır (MD5 Zenodo ile aynı) → {out}")
        return 0

    satir = list(csv.DictReader(open(C_ESLEME, encoding="utf-8")))
    hatalar, rapor = [], {"tarih": datetime.now().isoformat(timespec="seconds"),
                          "c_esleme_sha256": hashlib.sha256(C_ESLEME.read_bytes()).hexdigest()}
    # lezyon ayrıklığı (CSV)
    lez = {s: {r["lesion_id"] for r in satir if r["split_C"] == s} for s in BEKLENEN_N}
    rapor["lezyon_kesisim"] = {"train_val": len(lez["train"] & lez["val"]), "train_test": len(lez["train"] & lez["test"]),
                               "val_test": len(lez["val"] & lez["test"])}
    if any(rapor["lezyon_kesisim"].values()):
        hatalar.append(f"lezyon ayrıklığı bozuk {rapor['lezyon_kesisim']}")
    with np.load(cdir / "dermamnist_corrected_28.npz") as zc:
        c_etiket = {s: zc[f"{s}_labels"].reshape(-1).astype(np.int64) for s in BEKLENEN_N}
    icerik, dosya_sha, etiketler = {}, {}, {}
    for size in (28, 224):
        kaynak = WS / "data" / "medmnist" / (f"dermamnist{'' if size == 28 else '_224'}.npz")
        md5, _ = dosya_hash(kaynak)
        if md5 != RESMI_MD5[size]:
            raise SystemExit(f"HATA: {kaynak.name} MD5 {md5} ≠ resmî {RESMI_MD5[size]}")
        with np.load(kaynak) as z:
            R = {s: (z[f"{s}_images"], z[f"{s}_labels"].reshape(-1)) for s in ("train", "val", "test")}
        cikti = {}
        for s in BEKLENEN_N:
            rs = [r for r in satir if r["split_C"] == s]
            if len(rs) != BEKLENEN_N[s]:
                hatalar.append(f"{s}: {len(rs)} ≠ {BEKLENEN_N[s]}")
            sp = np.array([SPLIT_KOD[r["split_resmi"]] for r in rs], dtype=np.uint8)
            ix = np.array([int(r["index_resmi"]) for r in rs], dtype=np.int64)
            img = np.empty((len(rs),) + R["train"][0].shape[1:], dtype=np.uint8)
            lab = np.empty((len(rs), 1), dtype=np.uint8)
            for kod, ad in ((0, "train"), (1, "val"), (2, "test")):
                mk = sp == kod
                img[mk] = R[ad][0][ix[mk]]
                lab[mk, 0] = R[ad][1][ix[mk]]
            # denetimler
            if not np.array_equal(lab[:, 0].astype(np.int64), c_etiket[s]):
                hatalar.append(f"{size}px {s}: etiket ≠ C npz etiketi")
            if not np.array_equal(lab[:, 0].astype(np.int64), np.array([DX[r["dx"]] for r in rs])):
                hatalar.append(f"{size}px {s}: etiket ≠ CSV dx")
            if s == "test":
                if not (sp == 2).all() or not np.array_equal(img, R["test"][0][ix]):
                    hatalar.append(f"{size}px C-test resmî testin aynı indeksinin bayt-özdeş kopyası DEĞİL")
            cikti[f"{s}_images"], cikti[f"{s}_labels"] = img, lab
            cikti[f"{s}_index_resmi"], cikti[f"{s}_split_resmi"] = ix, sp
        etiketler[size] = {s: cikti[f"{s}_labels"] for s in BEKLENEN_N}
        hedef = WS / "data" / f"dermamnist_c_resmipiksel_{size}.npz"
        tmp = hedef.with_name(hedef.stem + ".tmp.npz")
        np.savez(tmp, **cikti)
        os.replace(tmp, hedef)
        with np.load(hedef) as z:
            icerik[str(size)] = icerik_sha256(z)
        dosya_sha[str(size)] = dosya_hash(hedef)[1]
        del R, cikti
    for s in BEKLENEN_N:
        if not np.array_equal(etiketler[28][s], etiketler[224][s]):
            hatalar.append(f"28 ↔ 224 etiket farkı ({s})")
    rapor.update({"n": BEKLENEN_N, "icerik_sha256": icerik, "dosya_sha256": dosya_sha, "hatalar": hatalar})
    if a.ilk_yapim and not hatalar:
        if BEKLENEN.exists():
            onceki = json.loads(BEKLENEN.read_text(encoding="utf-8"))
            if onceki["icerik_sha256"] != icerik:
                hatalar.append(f"ilk yapım: {BEKLENEN.name} zaten var ve FARKLI")
        else:
            atomik_json(BEKLENEN, {"icerik_sha256": icerik, "yazim": rapor["tarih"],
                                   "aciklama": "C1 verisi (resmî piksel + C bölmesi) donmuş içerik SHA256'sı; "
                                               "scripts/c_bolme_npz.py --kip c1 --ilk-yapim (şef, resmî test geçişinden SONRA)"})
    if not hatalar:
        if not BEKLENEN.exists():
            hatalar.append(f"{BEKLENEN} yok — şef test geçişinden sonra --ilk-yapim ile yazar (DUR)")
        else:
            bek = json.loads(BEKLENEN.read_text(encoding="utf-8"))["icerik_sha256"]
            if bek != icerik:
                hatalar.append(f"içerik SHA256 donmuş değerle aynı değil: {icerik} ≠ {bek}")
    gecti = not hatalar
    for size in (28, 224):
        atomik_json(WS / "data" / f"dermamnist_c_resmipiksel_{size}.json",
                    {**rapor, "boyut": size, "dosya_sha256": dosya_sha[str(size)], "icerik_sha256": icerik[str(size)],
                     "denetim_gecti": gecti})
    atomik_json(run_dir / "c_bolme_rapor.json", rapor)
    m.update({"denetim_gecti": int(gecti), "n_train": BEKLENEN_N["train"], "n_val": BEKLENEN_N["val"],
              "n_test": BEKLENEN_N["test"], "lezyon_kesisim_toplam": int(sum(rapor["lezyon_kesisim"].values())),
              "egitim_sure_sn": round(time.time() - t0, 1), "sure_sn": round(time.time() - t0, 1)})
    atomik_json(out, m)
    print(json.dumps(rapor, ensure_ascii=False, indent=1))
    if not gecti:
        print(f"[c_bolme_npz] ✖ DENETİM BAŞARISIZ: {hatalar}")
        return 2
    print(f"[c_bolme_npz] C1 verisi hazır ve donmuş değerle aynı → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
