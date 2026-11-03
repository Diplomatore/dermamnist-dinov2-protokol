#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dogrula.py — ön-kayıt zinciri ve dondurulmuş sonuç dosyalarının SHA256 doğrulaması (yalnız stdlib).

Kullanım (depo kökünden):  python tools/dogrula.py [--ayrinti]
Çıkış kodu: 0 = ZORUNLU denetimlerin hepsi geçti; 1 = en az bir uyuşmazlık.

Denetim katmanları
  Z1  sonuclar/*.sha256 yan dosyaları (k5_test_sonuc.json, k5_c1_sonuc.json, k5_dondurma.json)      ZORUNLU
  Z2  sonuclar/k5_test.KILIT → scripts/k5_test.py (test açılışındaki betik)                          ZORUNLU
  Z3  sonuclar/k5_dondurma.json (test öncesi dondurma, 2026-10-11 18:54:44): ön-kayıt, betikler,
      kuyruk/beklenti dosyaları; sonuclar/k5_dondurma_ek_c1.json                                     ZORUNLU
      - Test SONRASI belgelenmiş C1 uygulama düzeltmeleri (hipotezler/K5-C1-UYGULAMA-NOTU.md) ve
        gruplar.json'a eklenen v2/v3 kuyrukları (dosya içi _aciklama_k5_v2/_v3) "BELGELİ SAPMA"
        olarak raporlanır; düzeltme sonrası özet de nota karşı denetlenir.
  B1  makale/evidence/lkb, makale/evidence/c1mel: donmuş kaynak özetleri                              BİLGİ*
  B2  makale/evidence/evidence.json → deney_meta.kaynak_sha256                                       BİLGİ*
  B3  runs_meta/runs_colab/ARSIV_MANIFEST*.json → depoda bulunan JSON damgaları                       BİLGİ*
  (*) BİLGİ katmanlarında da uyuşmazlık (FAIL) çıkış kodunu 1 yapar; "depoda yok" (tahmin .npy, veri,
      makale notları) ve "hijyen eşlemesi" (HIJYEN_DEGISIKLIKLERI.json'daki yol temizliği) FAIL sayılmaz.
"""
import hashlib
import json
import sys
from pathlib import Path

K = Path(__file__).resolve().parents[1]
AYRINTI = "--ayrinti" in sys.argv
try:
    HIJ = json.loads((K / "HIJYEN_DEGISIKLIKLERI.json").read_text(encoding="utf-8"))["dosyalar"]
except FileNotFoundError:
    HIJ = {}

# K5-C1-UYGULAMA-NOTU.md "Düzeltme sonrası dosya SHA256" tablosu (önek…sonek)
C1_DUZELTME = {
    "varyantlar/k5/c_bolme.py": ("6759e808", "865e"),
    "varyantlar/k5/c_rakip_kos.py": ("a319fc30", "bb1b"),
    "varyantlar/k5/c_bolme_sar.py": ("6f2bb61b", "78f2"),
}
GRUPLAR_NOT = "kuyruk/gruplar.json"

sayac = {"GECTI": 0, "HIJYEN": 0, "BELGELI": 0, "YOK": 0, "FAIL": 0}
satirlar = []


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def denetle(katman, rel, beklenen, zorunlu=False):
    p = K / rel
    if not p.is_file():
        durum = "FAIL" if zorunlu else "YOK"
        gercek = "-"
    else:
        gercek = sha(p)
        if gercek == beklenen:
            durum = "GECTI"
        elif HIJ.get(rel, {}).get("onceki_sha256") == beklenen and HIJ[rel].get("yeni_sha256") == gercek:
            durum = "HIJYEN"
        elif rel in C1_DUZELTME and katman == "Z3":
            on, son = C1_DUZELTME[rel]
            durum = "BELGELI" if gercek.startswith(on) and gercek.endswith(son) else "FAIL"
        elif rel == GRUPLAR_NOT and katman == "Z3":
            t = p.read_text(encoding="utf-8")
            durum = "BELGELI" if "_aciklama_k5_v2" in t and "_aciklama_k5_v3" in t else "FAIL"
        else:
            durum = "FAIL"
    sayac[durum] += 1
    if durum != "GECTI" or AYRINTI:
        satirlar.append(f"[{katman}] {durum:7s} {rel}  beklenen={beklenen[:12]}… gercek={gercek[:12]}…")
    return durum


def jyukle(rel):
    return json.loads((K / rel).read_text(encoding="utf-8"))


# ---- Z1: yan .sha256 dosyaları
for yan in sorted((K / "sonuclar").glob("*.sha256")):
    parca = yan.read_text(encoding="utf-8").split()
    beklenen, ad = parca[0], parca[1].lstrip("*")
    rel = ad if (K / ad).is_file() else (yan.parent / ad).relative_to(K).as_posix()
    denetle("Z1", rel, beklenen, zorunlu=True)

# ---- Z2: test kilidi
kilit = jyukle("sonuclar/k5_test.KILIT")
denetle("Z2", "scripts/k5_test.py", kilit["betik_sha256"], zorunlu=True)

# ---- Z3: dondurma
dn = jyukle("sonuclar/k5_dondurma.json")
denetle("Z3", "hipotezler/K5-ONKAYIT.md", dn["onkayit_sha256"], zorunlu=True)
denetle("Z3", "scripts/k5_test.py", dn["k5_test_sha256"], zorunlu=True)
denetle("Z3", "varyantlar/logit_ayari/katlar.json", dn["katlar_json_sha256"], zorunlu=True)
for rel, h in dn["dosya_sha256"].items():
    denetle("Z3", rel, h, zorunlu=True)
for kol, d in dn.get("kaynak_tahmin_val_sha256", {}).items():
    for run, h in d.items():
        denetle("Z3", f"runs_meta/runs_colab/{run}/tahmin_val.npy", h)  # depoda yok → YOK
ek = jyukle("sonuclar/k5_dondurma_ek_c1.json")
denetle("Z3", "kuyruk/k5_c_bolme_beklenen.json", ek["k5_c_bolme_beklenen_sha256"], zorunlu=True)
denetle("Z3", "sonuclar/k5_test_sonuc.json", ek["k5_test_sonuc_sha256"], zorunlu=True)

# ---- B1: makale duyarlılık analizleri
for rel in ["makale/evidence/lkb/lkb_bootstrap_sonuc.json", "makale/evidence/c1mel/c1_mel_sonuc.json"]:
    if not (K / rel).is_file():
        continue
    d = jyukle(rel)
    if "k5_test_sha256" in d:
        denetle("B1", "scripts/k5_test.py", d["k5_test_sha256"])
    for r, h in d.get("donmus_kaynak_sha256", {}).items():
        denetle("B1", r, h)

# ---- B2: kanıt defteri kaynakları
ev = jyukle("makale/evidence/evidence.json")
ONEK = "../../makale_workspace/dermamnist-dinov2-protokol/"
for r, h in ev.get("deney_meta", {}).get("kaynak_sha256", {}).items():
    denetle("B2", "makale/" + r[len(ONEK):] if r.startswith(ONEK) else r, h)

# ---- B3: Colab arşiv manifesti (yalnız depodaki JSON damgaları)
for man in sorted((K / "runs_meta/runs_colab").glob("ARSIV_MANIFEST.json")):
    m = json.loads(man.read_text(encoding="utf-8"))
    dosyalar = m.get("dosyalar") or m.get("files") or {}
    ogeler = dosyalar.items() if isinstance(dosyalar, dict) else ((x.get("yol") or x.get("dosya"), x.get("sha256")) for x in dosyalar)
    for r, h in ogeler:
        if isinstance(h, dict):
            h = h.get("sha256")
        if not r or not h or not str(r).endswith(".json"):
            continue
        r = str(r).replace("\\", "/")
        denetle("B3", "runs_meta/runs_colab/" + r.split("runs_colab/")[-1], h)

print("\n".join(satirlar))
print(f"\nÖZET: GEÇTİ={sayac['GECTI']}  HİJYEN-EŞLEMESİ={sayac['HIJYEN']}  BELGELİ-SAPMA={sayac['BELGELI']}  "
      f"DEPODA-YOK={sayac['YOK']}  UYUŞMAZ={sayac['FAIL']}")
print("SONUÇ:", "GEÇTİ" if sayac["FAIL"] == 0 else "BAŞARISIZ")
sys.exit(0 if sayac["FAIL"] == 0 else 1)
