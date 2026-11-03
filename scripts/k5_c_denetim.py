# -*- coding: utf-8 -*-
"""
k5_c_denetim.py — K5-ONKAYIT §0 P7: C-test lezyon ayrıklığı + C-test ⊂ resmî test + eşleme tutarlılığı.
YALNIZ METADATA (CSV): görüntü/etiket/tahmin dizisi OKUNMAZ (test kilidi kapalı kalır).

Kaynaklar:
  data/splits/dermamnist_c_esleme.csv     (split_C, image_id, lesion_id, dx, split_resmi, index_resmi) — 10.015 satır
  data/splits/dermamnist_lezyon_esleme.csv (split, index, image_id, lesion_id, dx) — resmî bölme, 10.015 satır
  data/dermamnist_c/DermaMNIST-C.csv       (Abhishek vd. 2025; lesion_id, image_id, …, split) — C'nin yayımlanmış bölmesi
Denetimler (≠ beklenen → rc=2, DUR):
  (1) |C-test| = 1.227; lezyon(C-test) ∩ lezyon(resmî train ∪ resmî val) = ∅ (beklenen 0/1.227 görüntü).
  (2) C-test ⊂ resmî test: her C-test satırının split_resmi = test (1.227/1.227).
  (3) eşleme tutarlılığı: C-test satırının (image_id, lesion_id) çifti resmî eşlemede (split=test, index=index_resmi)
      aynı → 1.227/1.227; DermaMNIST-C.csv'deki C bölmesiyle split_C aynı (10.015/10.015); index_resmi benzersiz.
  (4) bilgi: C-val/C-train için aynı ayrıklık (C bölmesinin kendisi lezyon-ayrık mı), C-train'deki resmî test/val sayıları.
Çıktı: sonuclar/k5_c_denetim.json
Kullanım (WS kökü): PYTHONIOENCODING=utf-8 <python> -I scripts/k5_c_denetim.py
"""
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
C_ESL = WS / "data/splits/dermamnist_c_esleme.csv"
R_ESL = WS / "data/splits/dermamnist_lezyon_esleme.csv"
C_CSV = WS / "data/dermamnist_c/DermaMNIST-C.csv"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    C = list(csv.DictReader(open(C_ESL, encoding="utf-8")))
    R = list(csv.DictReader(open(R_ESL, encoding="utf-8")))
    CY = list(csv.DictReader(open(C_CSV, encoding="utf-8")))
    rkey = {(r["split"], int(r["index"])): r for r in R}
    lez_resmi = {s: {r["lesion_id"] for r in R if r["split"] == s} for s in ("train", "val", "test")}
    Cs = {s: [r for r in C if r["split_C"] == s] for s in ("train", "val", "test")}
    lezC = {s: {r["lesion_id"] for r in Cs[s]} for s in Cs}
    out = {"tarih": datetime.now().isoformat(timespec="seconds"),
           "kaynak_sha256": {p.name: sha(p) for p in (C_ESL, R_ESL, C_CSV)},
           "n": {s: len(v) for s, v in Cs.items()}}
    hatalar = []
    # (1) lezyon ayrıklığı
    yasak = lez_resmi["train"] | lez_resmi["val"]
    cakisan_goruntu = [r["image_id"] for r in Cs["test"] if r["lesion_id"] in yasak]
    out["P7_ctest_lezyon_kesisim_resmi_train_val"] = {"goruntu": len(cakisan_goruntu), "n_ctest": len(Cs["test"]),
                                                       "lezyon": len(lezC["test"] & yasak)}
    out["P7_ctest_lezyon_kesisim_resmi_train"] = len({r["image_id"] for r in Cs["test"] if r["lesion_id"] in lez_resmi["train"]})
    out["P7_ctest_lezyon_kesisim_resmi_val"] = len({r["image_id"] for r in Cs["test"] if r["lesion_id"] in lez_resmi["val"]})
    if len(Cs["test"]) != 1227:
        hatalar.append(f"|C-test| {len(Cs['test'])} ≠ 1227")
    if cakisan_goruntu:
        hatalar.append(f"C-test ∩ lezyon(resmî train∪val) = {len(cakisan_goruntu)} görüntü ≠ 0")
    # (2) C-test ⊂ resmî test
    alt = sum(1 for r in Cs["test"] if r["split_resmi"] == "test")
    out["P7_ctest_resmi_test_alt_kume"] = {"eslesen": alt, "n": len(Cs["test"])}
    if alt != len(Cs["test"]):
        hatalar.append(f"C-test ⊄ resmî test ({alt}/{len(Cs['test'])})")
    # (3) eşleme tutarlılığı
    ok = sum(1 for r in Cs["test"] if (k := rkey.get((r["split_resmi"], int(r["index_resmi"])))) is not None
             and k["image_id"] == r["image_id"] and k["lesion_id"] == r["lesion_id"])
    out["P7_ctest_esleme_tutarli"] = {"eslesen": ok, "n": len(Cs["test"])}
    if ok != len(Cs["test"]):
        hatalar.append(f"C-test eşleme {ok}/{len(Cs['test'])}")
    ok_tum = sum(1 for r in C if (k := rkey.get((r["split_resmi"], int(r["index_resmi"])))) is not None
                 and k["image_id"] == r["image_id"] and k["lesion_id"] == r["lesion_id"])
    out["esleme_tutarli_tum"] = {"eslesen": ok_tum, "n": len(C)}
    cy = {r["image_id"]: r["split"] for r in CY}
    ayni_bolme = sum(1 for r in C if cy.get(r["image_id"]) == r["split_C"])
    out["c_csv_bolme_ayni"] = {"eslesen": ayni_bolme, "n": len(C)}
    # CSV satır sırası = C npz sırası (data_manifest §DermaMNIST-C): split içi image_id dizisi iki CSV'de aynı mı
    sira_ayni = all([r["image_id"] for r in Cs[s]] == [r["image_id"] for r in CY if r["split"] == s] for s in Cs)
    out["c_csv_sira_ayni"] = bool(sira_ayni)
    ir = Counter((r["split_resmi"], r["index_resmi"]) for r in C)
    out["index_resmi_benzersiz"] = all(v == 1 for v in ir.values()) and len(ir) == 10015
    for k_, v_ in (("tum eşleme", ok_tum == len(C)), ("C CSV bölme", ayni_bolme == len(C)), ("sıra", sira_ayni),
                   ("index_resmi benzersiz", out["index_resmi_benzersiz"])):
        if not v_:
            hatalar.append(f"{k_} tutarsız")
    # (4) bilgi
    out["bilgi"] = {
        "cval_lezyon_kesisim_ctrain": len(lezC["val"] & lezC["train"]),
        "ctest_lezyon_kesisim_ctrain_cval": len(lezC["test"] & (lezC["train"] | lezC["val"])),
        "ctrain_resmi_kaynak": dict(Counter(r["split_resmi"] for r in Cs["train"])),
        "cval_resmi_kaynak": dict(Counter(r["split_resmi"] for r in Cs["val"])),
        "ctest_resmi_test_payi": len(Cs["test"]) / 2005,
    }
    out["hatalar"] = hatalar
    out["karar"] = "GEÇTİ" if not hatalar else "DUR"
    (WS / "sonuclar" / "k5_c_denetim.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0 if not hatalar else 2


if __name__ == "__main__":
    sys.exit(main())
