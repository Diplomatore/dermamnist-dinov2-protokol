# -*- coding: utf-8 -*-
"""F6 — E2 / H15: BCN20000 dış merkez değerlendirmesi (revizyon 2026-11-08).
Kaynak: DW/sonuclar/e2_bcn_sonuc.json (SHA256 .sha256 ile denetlenir) + DW/hipotezler/H15.md (§Beklentiler düşüş aralığı).
  (a) S6.sacilim[*].{ic, dis} (iç lezyon-ayrık ↔ BCN dengeli, 5 seed ort.) + ana.kol.<k>.dengeli_ga (lezyon-kümeli GA)
  (b) ana.S.h10-h02.{ic_R_dengeli, ic_C0_dengeli, ic_C1_dengeli, dis_R_dengeli, dis_C1_dengeli,
                     ic_R_acc, ic_C0_acc, ic_C1_acc, dis_R_accpiR, dis_C1_accpiC} (nokta + iki düzeyli bootstrap GA)
Sayı gömülmez; kart aralığı metinden ayrıştırılır. BCN tahmin dosyası açılmaz."""
import hashlib
import re

from _ortak import DW, dat_yaz, eng, jload

REL = "sonuclar/e2_bcn_sonuc.json"
beyan = (DW / (REL + ".sha256")).read_text(encoding="utf-8").split()[0]
if beyan != hashlib.sha256((DW / REL).read_bytes()).hexdigest():
    raise SystemExit("fig_e2: e2_bcn_sonuc.json SHA256 .sha256 ile uyuşmuyor")
d = jload(REL)

# --- (a) saçılım: aile → renk/işaretçi grubu (fig_e2.tex'te 'grup' sütunuyla süzülür)
AILE = {"h10": 1, "h10_c1": 1, "h14_c1": 1,     # DINOv2-B kısmi ince ayar
        "h11": 2,                               # DINOv2-S tam ince ayar
        "h01c": 3, "h12": 3, "h01_c1": 3,       # donuk sondalar
        "h02c": 4, "h02_c1": 4,                 # ResNet-18
        "h08c": 5, "h08_c1": 5}                 # ResNet-18 + D4
sat = []
for s in d["S6"]["sacilim"]:
    k = s["kol"]
    v = d["ana"]["kol"][k]
    if abs(v["dengeli"] - s["dis"]) > 1e-12:
        raise SystemExit(f"fig_e2: {k} saçılım dış değeri ana.kol ile uyuşmuyor")
    lo, hi = v["dengeli_ga"]
    sat.append([AILE[k], 1 if s["veri"] == "resmi" else 2, f"{s['ic']:.5f}", f"{s['dis']:.5f}",
                f"{s['dis'] - lo:.5f}", f"{hi - s['dis']:.5f}", k])
dat_yaz("fig_e2_sacilim.dat", ["grup", "protokol", "ic", "dis", "em", "ep", "kol"], sat,
        f"{REL}: S6.sacilim[*].ic/dis, ana.kol.<kol>.dengeli_ga (protokol 1 = resmî eğitim, 2 = C1)")

# kartın ön-kayıtlı düşüş aralığı (puan) → bant: dis = ic + alt/100 … ic + üst/100
satirlar = (DW / "hipotezler/H15.md").read_text(encoding="utf-8").splitlines()
i, s = next((i, s) for i, s in enumerate(satirlar, 1) if s.startswith("| Düşüş (BCN dengeli − lezyon-ayrık iç dengeli)"))
m = re.search(r"\[([−-]\d+); ([−-]\d+)\]", s)
alt, ust = (float(x.replace("−", "-")) for x in m.groups())
xs = [0.55, 0.85]
dat_yaz("fig_e2_bant.dat", ["x", "y"],
        [[x, f"{x + alt / 100:.4f}"] for x in xs] + [[x, f"{x + ust / 100:.4f}"] for x in reversed(xs)],
        f"hipotezler/H15.md satır {i}: kol başına beklenen düşüş [{alt:g}; {ust:g}] puan (BCN görülmeden); "
        "kapalı çokgen: alt kenar soldan sağa, üst kenar sağdan sola")

# --- (b) H10 − H02c: iç ↔ dış (y: üstten alta)
S = d["ana"]["S"]["h10-h02"]
SIRA = [  # (y, anahtar, ingilizce etiket, tür: 0 iç / 1 dış)
    (10.6, "ic_R_dengeli", "Internal, official split", 0),
    (9.6, "ic_C0_dengeli", "Internal, C0", 0),
    (8.6, "ic_C1_dengeli", "Internal, C1 (retrained)", 0),
    (7.6, "dis_R_dengeli", "BCN, official-split arms (B1)", 1),
    (6.6, "dis_C1_dengeli", "BCN, C1 arms (S1)", 1),
    (4.4, "ic_R_acc", "Internal ACC, official split", 0),
    (3.4, "ic_C0_acc", "Internal ACC, C0", 0),
    (2.4, "ic_C1_acc", "Internal ACC, C1", 0),
    (1.4, "dis_R_accpiR", "BCN ACC$_{\\pi_R}$, official arms", 1),
    (0.4, "dis_C1_accpiC", "BCN ACC$_{\\pi_C}$, C1 arms", 1),
]
fs, yl = [], []
for y, a, lab, tur in SIRA:
    x = S[a]
    n, (lo, hi) = x["nokta"], x["ga"]
    fs.append([f"{y:.1f}", tur, f"{n:.4f}", f"{n - lo:.4f}", f"{hi - n:.4f}",
               "{" + f"{eng(n)} [{eng(lo)}, {eng(hi)}]" + "}"])
    yl.append([f"{y:.1f}", "{" + lab + "}"])
dat_yaz("fig_e2_fark.dat", ["y", "tur", "d", "em", "ep", "lab"], fs,
        f"{REL}: ana.S.h10-h02.<anahtar>.nokta/ga (puan; iki düzeyli eşleştirilmiş bootstrap, B = 10.000)")
dat_yaz("fig_e2_fark_yl.dat", ["y", "lab"], sorted(yl, key=lambda r: float(r[0])), "fig_e2.py SIRA (etiketler)")
