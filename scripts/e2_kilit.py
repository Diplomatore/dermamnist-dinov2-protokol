# -*- coding: utf-8 -*-
"""
e2_kilit.py — H15 (E2) a3'ün İLK satırı: dış test KİLİDİNİ açar (`sonuclar/e2_bcn.KILIT`; kart §Aşama planı a3, K-f).

Kilit yazılmadan önce (hepsi geçmeli; aksi rc 2 → a3 çıkarım satırları KİLİT bulamaz, rc 7):
  1) hipotezler/H15.md SHA256 = onaylı değer (KULLANICI ONAYI H15, 2026-11-07).
  2) K-d kuru koşu: `scripts/e2_bcn_analiz.py` işlevleriyle arşiv iç tahminleri (k5-test / k5-c1 / e1-h14) iç değerleri
     ±0,005 puan içinde yeniden üretir (Colab'da donmuş k5_c1_sonuc.json yoksa yalnız kart sayılarıyla — kartın ölçütü).
  3) BCN girdisi: taşınan npz dosya SHA256'ları ve açılmış .npy içerik SHA256'ları = kuyruk/e2_bcn_beklenen.json.
  4) a2: 55 kol-seed'in ön-uçuş koşumu rc 0 ve ozdeslik_gecti = 1 (birincil kolda eksik → DUR; diğerleri sayılır, yazılır).
KİLİT içeriği: açılış zamanı; kartın, analiz betiğinin, beklenti dosyasının, beklenen dosyasının, e2_cikarim.py'nin SHA256'sı;
kopyasız liste ve lezyon kümesi SHA256'ları (beklenen dosyasından); iki geometrinin girdi içerik SHA256'ları; eşleme tablosu;
kol → run_id listesi; ikinci_acilis: null. KİLİT zaten varsa ve aynı içerikse (oturum devamı) yeniden yazılmaz; farklıysa DUR.
Kullanım (kuyruk satırı, CPU): python scripts/e2_kilit.py --out {RUN_DIR}/metrics.json
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "scripts"))
sys.path.insert(0, str(WS / "varyantlar"))
import e2_bcn_analiz as A  # noqa: E402
from e2 import bcn_veri as BV  # noqa: E402
from e2 import kollar as KL  # noqa: E402


def a2_durumu(runs: Path) -> dict:
    gecen, eksik = {}, []
    for k in KL.SIRA:
        v = KL.e2_varyant(k, "onucus")
        ok = set()
        for d in Path(runs).glob(f"*_{v}_s*"):
            try:
                c = json.loads((d / "config.json").read_text(encoding="utf-8"))
                j = json.loads((d / ".done").read_text(encoding="utf-8"))
                m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            if c.get("varyant") == v and str(c.get("kuyruk", "")).startswith(A.KUYRUK) and j.get("rc") == 0 \
                    and m.get("ozdeslik_gecti") == 1 and m.get("val_kumesi_dogru") == 1:
                ok.add(int(c["seed"]))
        gecen[k] = sorted(ok)
        eksik += [f"{k}/s{s}" for s in KL.SEEDLER if s not in ok]
    return {"gecen": gecen, "eksik": eksik, "birincil_eksik": [x for x in eksik if x.split("/")[0] in KL.BIRINCIL]}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--runs", default=str(WS / "runs"))
    p.add_argument("--kilit", default=str(A.KILIT))
    p.add_argument("--bcn-dizin", default=str(BV.DIZIN))
    p.add_argument("--beklenen", default=str(BV.BEKLENEN))
    p.add_argument("--a2-atla", action="store_true", help="yalnız yerel kod sınaması")
    a = p.parse_args()
    M = {"kilit_yazildi": 0}
    hata = []
    kart = A.sha(A.KART)
    if kart != A.ONAYLI_KART_SHA:
        hata.append(f"H15.md SHA256 {kart[:12]} ≠ onaylı {A.ONAYLI_KART_SHA[:12]}")
    k5s = A.k5_sha_denetle()
    ic = A.Ic(Path(a.runs))
    oz = A.kuru_ozdeslik(ic)
    M["kuru_gecti"] = int(oz["gecti"])
    if not oz["gecti"]:
        hata.append("K-d kuru koşu iç değerleri ±0,005 puan içinde yeniden üretmedi")
    beklenen = Path(a.beklenen)
    B = BV.beklenen_oku(beklenen)
    dz = Path(a.bcn_dizin)
    try:
        BV.dosyalari_dogrula(dz, B)
        ok = all(BV.geometri_oku(dz, g, B)[2] for g in BV.GEOMETRILER)
    except SystemExit as e:
        print(e)
        ok = False
    M["bcn_girdi_dogrulandi"] = int(bool(ok))
    if not ok:
        hata.append("BCN girdisi (dosya/içerik SHA256) beklenenle aynı değil")
    a2 = {"atlandi": True} if a.a2_atla else a2_durumu(Path(a.runs))
    M["a2_eksik_kol_seed"] = len(a2.get("eksik", []))
    if a2.get("birincil_eksik"):
        hata.append(f"a2 ön-uçuşu geçmemiş birincil kol-seed: {a2['birincil_eksik']}")
    yeni = {"kart": "H15 (E2, BCN20000)", "sha256": {
        "kart": kart, "analiz": A.sha(WS / "scripts" / "e2_bcn_analiz.py"), "beklenti": A.sha(A.BEKLENTI), "beklenen": A.sha(beklenen),
        "e2_cikarim": A.sha(WS / "varyantlar" / "e2" / "e2_cikarim.py"), "kopyasiz_liste": B["kopyasiz_liste"]["sha256"],
        "lezyon_kumeleri": B["lezyon_kumeleri"]["sha256"], "k5_test": k5s},
        "girdi_dosyalar": B["dosyalar"], "girdi_icerik": B["icerik"], "sira_sha256": B["sira_sha256"], "esleme": B["esleme"],
        "kollar": {k: [KL.run_id(k, s) for s in KL.SEEDLER] for k in KL.SIRA}, "ikinci_acilis": None,
        "kuru": {"gecti": oz["gecti"], "donmus_json": oz["donmus_json"],
                 "satirlar": {k: {"bu_puan": v["bu_puan"], "fark_kart": v["fark_kart"]} for k, v in oz["satirlar"].items()}},
        "a2": a2}
    kp = Path(a.kilit)
    if hata:
        print("DUR (KİLİT YAZILMADI): " + " · ".join(hata), flush=True)
        Path(a.out).write_text(json.dumps(M, ensure_ascii=False, indent=1), encoding="utf-8")
        return 2
    if kp.exists():
        eski = json.loads(kp.read_text(encoding="utf-8"))
        ayni = all(eski.get(q) == yeni[q] for q in ("sha256", "girdi_icerik", "kollar", "esleme"))
        if not ayni or eski.get("analiz_zamani"):
            print("DUR: KİLİT zaten var ve içeriği farklı / analiz yapılmış — ikinci açılış yalnız kullanıcı kararıyla (K-f)", flush=True)
            Path(a.out).write_text(json.dumps(M, ensure_ascii=False, indent=1), encoding="utf-8")
            return 3
        M.update({"kilit_yazildi": 1, "kilit_onceden_vardi": 1})
        print(f"KİLİT zaten açık ve aynı içerik ({eski.get('acilis_zamani')}) — oturum devamı", flush=True)
    else:
        yeni["acilis_zamani"] = datetime.now().isoformat(timespec="seconds")
        tmp = kp.with_name(kp.name + ".tmp")
        tmp.write_text(json.dumps(yeni, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(kp)
        M.update({"kilit_yazildi": 1, "kilit_onceden_vardi": 0})
        print(f"KİLİT AÇILDI: {kp} ({yeni['acilis_zamani']}) · K-d GEÇTİ · girdi doğrulandı · a2 eksik {M['a2_eksik_kol_seed']}", flush=True)
    Path(a.out).write_text(json.dumps(M, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
