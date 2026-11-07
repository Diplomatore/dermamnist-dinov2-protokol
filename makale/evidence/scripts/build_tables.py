#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_tables.py — evidence/tables/*.tex üreticisi (figure-style.md §A4 + kalite-kurallari.md S8).

- Hücreler YALNIZ evidence/evidence.json kayıtlarından gelir (id ile); elle sayı yok.
- Tek sütunlu Springer (sn-jnl) → `table` + `tabularx{\\textwidth}` / `tabular*{\\textwidth}`; \\small;
  caption üstte; booktabs (\\botrule); dikey çizgi yok.
- \\iyi{}/\\kotu{} kodla atanır: sütun tanımı yönü taşır (yon: "yuksek"|"dusuk"); kıyas tablolarında
  yalnız ön-kayıtlı kuralı geçen / GA'sı sıfırı dışlayan farklar renklenir; referans satırı (B) nötr.
- Başlık bütçesi: tablo caption ≤ 25 kelime (figure-style §E2) — aşarsa üretici durur.
Kullanım: python evidence/scripts/build_tables.py
"""
import json
import math
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
WS = Path(__file__).resolve().parents[2]
EV = json.loads((WS / "evidence" / "evidence.json").read_text(encoding="utf-8"))
K = {r["id"]: r for r in EV["kayitlar"] if r.get("durum", "gecerli") != "terk"}
OUT = WS / "evidence" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
KULLANILAN = {}


def R(i, tablo):
    KULLANILAN.setdefault(tablo, []).append(i)
    return K[i]


# ------------------------------------------------------------------ biçim yardımcıları
def sayi(x, n=4):
    """0.89397 → '0,8940' (metin kipi)"""
    return f"{x:.{n}f}".replace(".", ",")


def _metin(x, n):
    """Metin kipinde işaretli sayı: +2,26 / −0,36 (\\iyi/\\kotu kalın/eğik uygulanabilsin)."""
    s = f"{x:+.{n}f}".replace(".", ",")
    return s.replace("-", "\\textminus{}")


def isaretli(x, n=2):
    return _metin(x, n)


def ga(g, n=2):
    return f"[{_metin(g[0], n)}; {_metin(g[1], n)}]"


def pdeg(p):
    if p is None:
        return "--"
    if round(p, 3) < 1e-3:
        e = math.floor(math.log10(p))
        m = round(p / 10 ** e, 1)
        if m >= 10:
            m, e = m / 10, e + 1
        return f"${m:.1f}".replace(".", "{,}") + f"{{\\times}}10^{{{e}}}$"
    return f"{p:.3f}".replace(".", ",")


def pm(x, s, n=4):
    return f"{sayi(x, n)}\\,$\\pm$\\,{sayi(s, n)}"


def iki(ust, alt):
    # \\relax: ikinci satır '[' ile başlarsa \\\\'in isteğe bağlı argümanı sanılmasın
    return f"\\begin{{tabular}}[c]{{@{{}}c@{{}}}}{ust}\\\\\\relax {alt}\\end{{tabular}}"


def renk_ga(metin, g, aktif=True):
    """GA tamamen sıfırın üstündeyse \\iyi, altındaysa \\kotu; aksi nötr."""
    if not aktif or g is None:
        return metin
    if g[0] > 0:
        return f"\\iyi{{{metin}}}"
    if g[1] < 0:
        return f"\\kotu{{{metin}}}"
    return metin


def ga_ust_sifir_ayni_karar(r):
    """Holm ailesi kaydında 'GA sıfırın üstünde' (renk) ile ön-kayıtlı karar aynı olmalı (review-tur2 K5)."""
    karar = r.get("anlamli") if r.get("anlamli") is not None else r.get("kural_gecti")
    if bool(r["ga"][0] > 0) != bool(karar):
        raise SystemExit(f"{r['id']}: GA>0 ({r['ga'][0] > 0}) ile karar ({karar}) ayrışıyor — renk tek anlamlı olmaz")


SAYI_AD = {1: "bir", 2: "iki", 3: "üç", 4: "dört", 5: "beş"}


def kelime(s):
    s = re.sub(r"\\(iyi|kotu)\{([^}]*)\}", r"\2", s)
    s = re.sub(r"\$[^$]*\$", "X", s)
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    return len([w for w in re.split(r"\s+", s.replace("~", " ")) if re.search(r"\w", w)])


def tablo(ad, baslik, etiket, govde, notlar=None):
    n = kelime(baslik)
    if n > 25:
        raise SystemExit(f"{ad}: başlık {n} kelime > 25")
    satir = ["% OTOMATİK — evidence/scripts/build_tables.py; elle düzenlemeyin (kaynak: evidence/evidence.json)",
             f"% kullanılan kanıt id'leri: {', '.join(dict.fromkeys(KULLANILAN.get(ad, [])))}",
             "\\begin{table}[!htbp]", "\\centering\\small", "\\setlength{\\tabcolsep}{4pt}",
             f"\\caption{{{baslik}}}\\label{{{etiket}}}"] + govde
    if notlar:
        satir.append("\\par\\smallskip\\raggedright " + notlar)
    satir.append("\\end{table}")
    (OUT / f"{ad}.tex").write_text("\n".join(satir) + "\n", encoding="utf-8")
    print(f"{ad}.tex: başlık {n} kelime")


KOLLAR = [("baseline", "B (rakip)", "R18, 28 px, sıfırdan"), ("h01", "H01c", "DINOv2-B donuk + sonda"),
          ("h02", "H02c", "R18-ImageNet, tam ince ayar"), ("h08", "H08c", "H02c + eğitimde D4"),
          ("h10", "H10 (W)", "DINOv2-B, son 4 blok"), ("h11", "H11", "DINOv2-S tam ince ayar"),
          ("h12", "H12", "DINOv2-S donuk + sonda")]


def sutun_uc(degerler, yon):
    """degerler: {kol: x}; yön 'yuksek'|'dusuk' → (en_iyi_kol, en_kotu_kol)"""
    sir = sorted(degerler, key=lambda k: degerler[k], reverse=(yon == "yuksek"))
    return sir[0], sir[-1]


def boya(metin, kol, uc):
    # Şef kararı (2026-10-12): düzey-sırası renklendirmesi kapalı — üst kollar arası farklar
    # ayırt edilemiyor; renk "H08 kazanandır" (karar.md §7.2 yasak iddia) diye okunabiliyordu.
    # Renk yalnız GA tabanlı kıyas tablolarında (renk_ga) kullanılır.
    return metin
    if kol == uc[0]:
        return f"\\iyi{{{metin}}}"
    if kol == uc[1]:
        return f"\\kotu{{{metin}}}"
    return metin


# ------------------------------------------------------------------ (a) ana tablo
def tab_ana():
    ad = "tab_ana"
    SUT = [("acc", "ACC", "yuksek"), ("auc", "AUC", "yuksek"), ("dengeli_acc", "Dengeli ACC", "yuksek"),
           ("ece15", "ECE-15", "dusuk")]
    renkli = [k for k, _, _ in KOLLAR if k != "baseline"]  # B = referans satırı, nötr
    uclar = {m: sutun_uc({k: R(f"T-{k}-{m}", ad)["deger"] for k in renkli}, yon) for m, _, yon in SUT}
    g = ["\\begin{tabularx}{\\textwidth}{>{\\raggedright\\arraybackslash}X cccc}", "\\toprule",
         "Kol & " + " & ".join(b for _, b, _ in SUT) + " \\\\", "\\midrule"]
    for k, isim, tanim in KOLLAR:
        h = []
        for m, _, _ in SUT:
            r = R(f"T-{k}-{m}", ad)
            h.append(pm(r["deger"], r["std"]) if k == "baseline" else boya(pm(r["deger"], r["std"]), k, uclar[m]))
        g.append(f"{isim} & " + " & ".join(h) + " \\\\")
    g.append("\\midrule")
    h = [pm(R(f"T-baseline_son-{m}", ad)["deger"], R(f"T-baseline_son-{m}", ad)["std"]) for m, _, _ in SUT]
    g.append("B son-epoch (ikincil) & " + " & ".join(h) + " \\\\")
    g += ["\\botrule", "\\end{tabularx}"]
    tablo(ad, "Resmî test sonuçları (n = 2.005; 5 seed, ort.\\,$\\pm$\\,std). Kollar arası "
              "üstünlük için Tablo~\\ref{tab:kiyas} ve Tablo~\\ref{tab:protokol}'deki GA'lara bakınız.", "tab:ana", g)


# ------------------------------------------------------------------ (b) Aile 1 + Aile 2 (+ C1)
def tab_kiyas():
    ad = "tab_kiyas"
    g = ["\\begin{tabularx}{\\textwidth}{>{\\raggedright\\arraybackslash}X cccc}", "\\toprule",
         " & \\multicolumn{2}{c}{Resmî test (n = 2.005)} & \\multicolumn{2}{c}{C1: lezyon-ayrık (n = 1.227)} \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
         "Kıyas & $\\Delta$ACC [puan] & Holm $p$ & $\\Delta$ACC [puan] & Holm $p$ \\\\",
         " & [\\%95 GA] & ($t$; boot) & [\\%95 GA] & ($t$; boot) \\\\", "\\midrule"]

    def hucre(r, renk):
        d = r["deger"]
        return iki(renk_ga(isaretli(d), r["ga"], renk), ga(r["ga"]))

    def holm(r):
        if r.get("holm_t") is None:
            return iki("--", "--")
        return iki(pdeg(r["holm_t"]), pdeg(r["holm_boot"]))

    bloklar = [
        ("Aile 1: rakibe karşı (marj $\\geq +2{,}0$)",
         [("H01c $-$ B", "T-AILE1-h01-baseline", "C1-A1-h01-baseline"),
          ("H02c $-$ B", "T-AILE1-h02-baseline", "C1-A1-h02-baseline"),
          ("H08c $-$ B", "T-AILE1-h08-baseline", "C1-A1-h08-baseline"),
          ("H10 $-$ B", "T-AILE1-h10-baseline", "C1-A1-h10-baseline")]),
        # tur-1b (review 8.3(a), şef kararı): Aile-2 ve H10 − H08c hücrelerinin tek yuvası tab_protokol
        ("Betimsel (Holm dışı)",
         [("B son-epoch $-$ B", "T-K-baseline_son-baseline-acc", None)]),
    ]
    for bi, (bas, satirlar) in enumerate(bloklar):
        if bi:
            g.append("\\midrule")
        g.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{bas}}}}} \\\\")
        for isim, tid, cid in satirlar:
            rt = R(tid, ad)
            # tur-1 (review 7.2): betimsel (Holm dışı) satırlar her tabloda renksiz
            # tur-2 (review-tur2 K5): mavi = GA sıfırın üstünde (tab_protokol ile tek anlam); Holm
            # satırlarında bu, ön-kayıtlı kararla (anlamli / kural_gecti) birebir örtüşmek ZORUNDA.
            renk_t = "AILE" in tid
            if renk_t:
                ga_ust_sifir_ayni_karar(rt)
            h = [hucre(rt, renk_t), holm(rt)]
            if cid:
                rc = R(cid, ad)
                renk_c = cid.startswith("C1-A")
                if renk_c:
                    ga_ust_sifir_ayni_karar(rc)
                h += [hucre(rc, renk_c), holm(rc)]
            else:
                h += ["--", "--"]
            g.append(f"{isim} & " + " & ".join(h) + " \\\\")
    g += ["\\botrule", "\\end{tabularx}"]
    tablo(ad, "Rakibe karşı eşleştirilmiş kıyaslar: $\\Delta$ACC, görüntü-bootstrap GA, Holm düzeltmeli tek yönlü $p$. "
              "Mavi kalın: GA sıfırın üstünde; betimsel satır renksiz. Aile~2: Tablo~\\ref{tab:protokol}.",
          "tab:kiyas", g)


# ------------------------------------------------------------------ (c) ablasyon zinciri
def tab_ablasyon():
    ad = "tab_ablasyon"
    # (kol, ad, ABL kaydı, kısa tanım — tam metin ABL kaydında; sayı içermez)
    # tur-1 (review 7.7): ebeveyn ABL kaydının 'taban' alanından; tek zincir izlenimi verilmez
    satirlar = [("baseline", "B", "ABL-1", "rakip, 28 px, sıfırdan"),
                ("h01", "H01c", "ABL-2", "224 px, DINOv2-B donuk sonda"),
                ("h02", "H02c", "ABL-3", "R18-ImageNet, 224 px, tam ince ayar"),
                ("h10", "H10 (W)", "ABL-4", "DINOv2-B, son 4 blok"),
                ("h08", "H08c", "ABL-5", "eğitimde D4"),
                ("h12", "H12", "ABL-7", "DINOv2-S donuk sonda"),
                ("h11", "H11", "ABL-8", "DINOv2-S tam ince ayar")]
    ADK = {"baseline": "B", "h01": "H01c", "h02": "H02c", "h08": "H08c", "h10": "H10"}
    taban_test = R("DM-test-nv", ad)["deger"] / R("T-meta-n", ad)["deger"]  # = D09.0 (notes/derivations.md)
    SUT = [("val", "V-{k}-acc", 4, False), ("test", "T-{k}-acc", 4, False), ("testm", "T-{k}-acc", 2, "test"),
           ("c0", "T-C0-{k}-acc", 4, False),
           ("c0m", "T-C0-{k}-accmaj", 2, True), ("c1", "C1-{k}-acc", 4, False), ("c1m", "C1-{k}-accmaj", 2, True)]
    uclar = {}
    for s, sab, _, _ in SUT:
        deg = {k: K[sab.format(k=k)]["deger"] for k, _, _, _ in satirlar if k != "baseline" and sab.format(k=k) in K}
        uclar[s] = sutun_uc(deg, "yuksek")
    g = ["\\setlength{\\tabcolsep}{2.6pt}",
         "\\begin{tabularx}{\\textwidth}{>{\\raggedright\\arraybackslash}X ccccccc}", "\\toprule",
         " & Doğr. & \\multicolumn{2}{c}{" + iki("Resmî test", "(n = 2.005)") + "} & \\multicolumn{2}{c}{"
         + iki("C0", "(n = 1.227)") + "} & \\multicolumn{2}{c}{" + iki("C1", "(n = 1.227)") + "} \\\\",
         "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
         "Kol (ebeveyn): fark & ACC & ACC & $-$\\,taban & ACC & $-$\\,taban & ACC & $-$\\,taban \\\\",
         "\\midrule"]
    for i, (k, isim, abl, kisa) in enumerate(satirlar):
        if k == "h12":
            g += ["\\midrule", "\\multicolumn{8}{@{}l}{\\textit{Omurga boyutu satırları (betimsel)}} \\\\"]
        ebeveyn = R(abl, ad).get("taban")
        h = []
        for s, sab, n, puanla in SUT:
            kid = sab.format(k=k)
            if kid not in K:
                h.append("--")
                continue
            x = R(kid, ad)["deger"]
            if puanla == "test":
                x = x - taban_test
            m = isaretli(100 * x, n) if puanla else sayi(x, n)
            h.append(m if k == "baseline" else boya(m, k, uclar[s]))
        ad_ = "H10 = W" if k == "h10" else isim
        kok = f" ({ADK[ebeveyn]}'den)" if ebeveyn else ""
        g.append(f"{ad_}{kok}: {kisa} & " + " & ".join(h) + " \\\\")
    g += ["\\botrule", "\\end{tabularx}"]
    tb_c = R("T-C0-cogunluk", ad)["deger"]
    # tur-2 (review-tur2 K7/K10): Doğr. sütunu L4 tanımıdır (V-*-acc); seçim döngüsünün yerel değerleri LB-*
    # (eşleme: VAL-tanim-esleme)
    R("VAL-tanim-esleme", ad)
    assert all("L4 tanımı" in K[f"V-{k}-acc"]["tanim"] for k, _, _, _ in satirlar)
    tablo(ad, "Ablasyon: doğrulamada (Doğr.; Colab L4) ve üç değerlendirmede ACC (5 seed ort.); "
              f"$-$\\,taban: çoğunluk tabanına (test {sayi(taban_test, 3)}; C {sayi(tb_c, 4)}) göre fark [puan].",
          "tab:ablasyon", g)


# ------------------------------------------------------------------ (d) S/N + C0/C1 + KÇ
def tab_protokol():
    ad = "tab_protokol"
    SUT = [("H10 $-$ H02c", "h10-h02"), ("H08c $-$ H02c", "h08-h02"), ("H10 $-$ H08c$^{\\dagger}$", "h10-h08")]
    BETIMSEL = {"h10-h08"}  # tur-1 (review 7.2): betimsel sütun renksiz
    satirlar = [
        ("Resmî test $\\Delta$ACC (n = 2.005)", {"h10-h02": "T-AILE2-h10-h02", "h08-h02": "T-AILE2-h08-h02",
                                                "h10-h08": "T-K-h10-h08-acc"}, True),
        ("$\\Delta_S$: eğitimde görülmüş lezyon (S, n~=~693)", {c: f"T-K-{c}-dS" for _, c in SUT}, True),
        ("$\\Delta_N$: görülmemiş lezyon (N, n~=~1.312)", {c: f"T-K-{c}-dN" for _, c in SUT}, True),
        ("$\\Delta_S - \\Delta_N$ (ham; kural nesnesi)", {c: f"T-K-{c}-dSdN" for _, c in SUT}, False),
        ("$\\Delta_S - \\Delta_N$ (tabaka-std.; ikincil)", {c: f"T-K-{c}-tab_std_dS_eksi_dN" for _, c in SUT}, False),
        ("C0: C-test alt kümesi, yeniden eğitimsiz", {c: f"T-C0-K-{c}" for _, c in SUT}, True),
        ("C1: C bölmesi, yeniden eğitimli", {"h10-h02": "C1-A2-h10-h02", "h08-h02": "C1-A2-h08-h02",
                                             "h10-h08": "C1-K-h10-h08-acc"}, True),
        # revizyon 2026-11-03 (dış eleştiri M3): C1 dengeli ACC (ön-kayıtlı yön, K5-ONKAYIT §4.3; karar vermez)
        # ve C1 mel duyarlılığı (test-sonrası analiz; renksiz, §)
        ("\\quad C1 $\\Delta$ dengeli ACC", {c: f"C1-K-{c}-dengeli_acc" for _, c in SUT}, True),
        ("\\quad C1 $\\Delta$ mel duyarlılığı (mel n~=~70)$^{\\S}$",
         {c: f"C1-REV-K-{c}-duyarlilik_mel" for _, c in SUT}, False),
        ("$K_0$ = resmî $-$ C0 (KÇ$_0$ son sütunda)", {"h10-h02": "T-C0-K0-H10", "h08-h02": "T-C0-K0-H08",
                                                       "h10-h08": "T-C0-KC0"}, False),
        ("$K$ = resmî $-$ C1 (KÇ son sütunda)", {"h10-h02": "C1-K-H10", "h08-h02": "C1-K-H08",
                                                     "h10-h08": "C1-KC"}, False),
    ]
    HOLM_SONRA = {satirlar[0][0], satirlar[6][0]}  # resmî test ve C1 ΔACC satırlarının altına
    # tur-2 (review-tur2 K4): lezyon-kümeli GA'sı sıfıra göre konum değiştiren hücreler ‡ ile işaretlenir
    oz = R("SENS-LKB-ozet", ad)
    LKB_DEG = {R(i, ad)["ref_id"] for i in oz["degisen"]}
    # revizyon 2026-11-03: C1 dengeli/mel hücrelerinin lezyon-kümeli konumu C1-REV-* kayıtlarından
    for _, c in SUT:
        lr = R(f"C1-REV-K-{c}-dengeli_acc-lezyon", ad)
        if lr["isaret_degisti"]:
            LKB_DEG.add(lr["ref_id"])
        mr = R(f"C1-REV-K-{c}-duyarlilik_mel", ad)
        if mr["isaret_degisti_lezyon"]:
            LKB_DEG.add(mr["id"])
    n_mel = R("C1-REV-n-mel", ad)["deger"]
    assert n_mel == 70, "satır etiketindeki mel n kanıtla uyuşmuyor"
    HOLM_SATIR = {0: {"h10-h02", "h08-h02"}, 6: {"h10-h02", "h08-h02"}}  # Aile 2 hücreleri
    isaretlenen, kotu_var = [], False
    AILE2_SUT = {"h10-h02", "h08-h02"}
    g = ["\\begin{tabularx}{\\textwidth}{>{\\raggedright\\arraybackslash}X ccc}", "\\toprule",
         "Ölçü [puan; \\%95 GA] & " + " & ".join(s for s, _ in SUT) + " \\\\", "\\midrule"]
    for si, (isim, idler, renkli) in enumerate(satirlar):
        h = []
        for _, c in SUT:
            r = R(idler[c], ad)
            if c in HOLM_SATIR.get(si, ()):
                ga_ust_sifir_ayni_karar(r)
            if "tab_std" in idler[c]:  # B1: T-SN-* tab_puan/tab_ga ile çapraz denetim
                sn = R(f"T-SN-{c}", ad)
                assert abs(sn["tab_puan"] - r["deger"]) < 1e-4
                assert all(abs(a - b) < 1e-4 for a, b in zip(sn["tab_ga"], r["ga"]))
            hucre = renk_ga(isaretli(r["deger"]), r["ga"], renkli and c not in BETIMSEL)
            kotu_var |= hucre.startswith("\\kotu")
            if r["id"] in LKB_DEG:
                hucre += "$^{\\ddagger}$"
                isaretlenen.append(r["id"])
            h.append(iki(hucre, ga(r["ga"])))
        g.append(f"{isim} & " + " & ".join(h) + " \\\\")
        # tur-1b (review 8.3(a)): Aile-2 Holm p'leri tab_kiyas'tan buraya taşındı (tek yuva)
        if isim in HOLM_SONRA:
            h2 = []
            for _, c in SUT:
                kid = idler[c]
                r = R(kid, ad) if c in AILE2_SUT else None
                h2.append(iki(pdeg(r["holm_t"]), pdeg(r["holm_boot"])) if r else "--")
            g.append(f"\\quad Holm $p$, tek yönlü ($t$; boot) & " + " & ".join(h2) + " \\\\")
        if isim.startswith("$\\Delta_S - \\Delta_N$ (tabaka"):
            h2 = []
            for _, c in SUT:
                r = R(f"T-K-{c}-Spay", ad)
                h2.append(f"\\%{100 * r['deger']:.0f}" if 0 <= r["deger"] <= 1 else "--")
            g.append("Net ACC kazancının S payı & " + " & ".join(h2) + " \\\\")
    g += ["\\botrule", "\\end{tabularx}"]
    if sorted(isaretlenen) != sorted(LKB_DEG):
        raise SystemExit(f"tab_protokol: ‡ işareti eksik/fazla: {sorted(isaretlenen)} ↔ {sorted(LKB_DEG)}")
    kirmizi = " Altında kırmızı eğik." if kotu_var else ""
    tablo(ad, "S/N ayrışımı, lezyon-ayrık değerlendirmeler. Mavi kalın: $\\Delta$ACC, $\\Delta_S$, $\\Delta_N$, "
              f"C0, C1 satırlarında GA sıfırın üstünde.{kirmizi} $^{{\\dagger}}$Betimsel, renksiz. "
              "$^{\\ddagger}$Lezyon-kümeli GA sıfırı kapsar. $^{\\S}$Test-sonrası analiz, renksiz.", "tab:protokol", g,
          notlar=None)


# ------------------------------------------------------------------ (e) replikasyon
def tab_replikasyon():
    ad = "tab_replikasyon"
    g = ["\\begin{tabularx}{\\textwidth}{>{\\raggedright\\arraybackslash}X ccc}", "\\toprule",
         "Kaynak / ortam & Test ACC & Test AUC & Test ACC \\\\",
         " & (val-AUC seçimi) & & (son epoch) \\\\", "\\midrule",
         "\\multicolumn{4}{@{}l}{\\textit{Literatür (bağlam; aynı koşulda üretilmedi)}} \\\\"]
    y1, y2 = R("LIT-yang2023-acc", ad), R("LIT-yang2023-auc", ad)
    g.append(f"Yang vd.\\ 2023, Tablo~3~\\cite{{{y1['atif']}}} & {sayi(y1['deger'], 3)} & {sayi(y2['deger'], 3)} & -- \\\\")
    d1, d2 = R("LIT-doerrich2025-acc", ad), R("LIT-doerrich2025-auc", ad)
    g.append(f"Doerrich vd.\\ 2025, Ek~S4 (ImageNet, sıfır dolgu; 3 seed)~\\cite{{{d1['atif']}}} & "
             f"{sayi(d1['deger'] / 100)}\\,$\\pm$\\,{sayi(d1['std'] / 100)} & "
             f"{sayi(d2['deger'] / 100)}\\,$\\pm$\\,{sayi(d2['std'] / 100)} & -- \\\\")
    m = R("LIT-medeiros2026-acc", ad)
    g.append(f"Medeiros 2026, Tablo~2~\\cite{{{m['atif']}}} & {sayi(m['deger'], 3)} & -- & -- \\\\")
    ka = R("REP-kabul-araligi", ad)["deger"]
    g.append(f"Kabul aralığı (ön-kayıtlı) & [{sayi(ka[0])}; {sayi(ka[1], 3)}] & -- & -- \\\\")
    g += ["\\midrule", "\\multicolumn{4}{@{}l}{\\textit{Bu çalışma (5 seed)}} \\\\"]
    a, b, c = R("REP-yerel-test_acc", ad), R("REP-yerel-test_auc", ad), R("REP-yerel-test_acc_son", ad)
    g.append(f"Yerel GTX 1650 (replikasyon kapısı) & {pm(a['deger'], a['std'])} & {pm(b['deger'], b['std'])} & "
             f"{pm(c['deger'], c['std'])} \\\\")
    a, b, c = R("T-baseline-acc", ad), R("T-baseline-auc", ad), R("T-baseline_son-acc", ad)
    R("REP-L4-test_acc", ad)
    g.append(f"Colab L4, FP32 (ana test satırı) & {pm(a['deger'], a['std'])} & {pm(b['deger'], b['std'])} & "
             f"{pm(c['deger'], c['std'])} \\\\")
    g += ["\\botrule", "\\end{tabularx}"]
    tablo(ad, "Rakip ResNet-18 (28 px) replikasyonu: literatür çıpaları ve iki donanım. Renk kodu yok: bütün "
              "satırlar aynı rakibin referans değerleridir.", "tab:replikasyon", g)


# ------------------------------------------------------------------ (f) Ek: T4 betimsel
def tab_t4():
    ad = "tab_t4"
    g = ["\\begin{tabularx}{\\textwidth}{>{\\raggedright\\arraybackslash}X cc}", "\\toprule",
         "Ölçü & Değer & Değerlendirme \\\\", "\\midrule"]
    for m, isim in [("acc", "ACC"), ("auc", "AUC"), ("dengeli_acc", "Dengeli ACC"), ("ece15", "ECE-15")]:
        r = R(f"T-t4-{m}", ad)
        g.append(f"T4 {isim} (ort.\\,$\\pm$\\,std) & {pm(r['deger'], r['std'])} & resmî test \\\\")
    r = R("T-C0-t4-acc", ad)
    tb = R("T-C0-cogunluk", ad)["deger"]  # D12 (notes/derivations.md)
    g.append(f"T4 ACC (çoğunluk tabanı {isaretli(100 * (r['deger'] - tb))} puan) & {pm(r['deger'], r['std'])} & C0 \\\\")
    r = R("V-t4-acc", ad)
    g.append(f"T4 ACC (test öncesi görüldü) & {pm(r['deger'], r['std'])} & doğrulama \\\\")
    g.append("\\midrule")
    for c, isim in [("t4-h10", "T4 $-$ H10"), ("t4-h08", "T4 $-$ H08c"), ("t4-h02", "T4 $-$ H02c")]:
        r = R(f"T-K-{c}-acc", ad)
        g.append(f"{isim}: $\\Delta$ACC [puan; \\%95 GA] & {isaretli(r['deger'])} {ga(r['ga'])} & resmî test \\\\")
    g += ["\\botrule", "\\end{tabularx}"]
    tablo(ad, "Ek: T4 = H08c $\\oplus$ H10 seed-eşli topluluğu, betimsel; çoklu karşılaştırma düzeltmesi yok, "
              "bu yüzden renk kodu uygulanmaz.", "tab:t4", g)


# ------------------------------------------------------------------ (g) başarısız / tutmayan ön-kayıtlar
def tab_basarisiz():
    ad = "tab_basarisiz"
    g = ["\\begin{tabularx}{\\textwidth}{>{\\raggedright\\arraybackslash}p{2.6cm} >{\\raggedright\\arraybackslash}X cc}",
         "\\toprule", "Madde & Ön-kayıtlı kural & Ölçülen [puan; \\%95 GA] & Sonuç \\\\", "\\midrule",
         "\\multicolumn{4}{@{}l}{\\textit{A. Öldürülen kartlar (yalnız doğrulama, n = 1.003)}} \\\\"]
    h05 = re.search(r"ACC < (\d,\d+)", K["BO-H05"]["olurme_kurali"]).group(1)  # review 4.13: kural tam
    A = [("BO-H05", "H05 sınıf ağırlıklı CE (dengeli ACC)",
          "dengeli $\\Delta < +0{,}5$ ya da $p_t \\geq 0{,}05$ ya da ACC $< " + h05.replace(",", "{,}") + "$"),
         ("BO-H07", "H07 TTA, 8 dihedral görünüm", "$\\Delta < +0{,}5$ ya da GA alt $\\leq 0$"),
         ("BO-H09", "H09 renk oynaması", "$\\Delta < +0{,}5$ ya da $p_t \\geq 0{,}05$"),
         ("BO-H13", "H13 topluluk H02c $\\oplus$ H10", "$\\Delta < +0{,}5$ ya da $p_t \\geq 0{,}05$")]
    for kid, isim, kural in A:
        r = R(kid, ad)
        assert "0,5" in r["olurme_kurali"], kid
        pt = f"; $p_t$ {pdeg(r['p_t'])}" if r.get("p_t") is not None else ""
        g.append(f"{isim} & {kural} & {isaretli(r['deger'])} {ga(r['ga'])}{pt} & \\kotu{{ÖLDÜR}} \\\\")
    g += ["\\midrule", "\\multicolumn{4}{@{}l}{\\textit{B. Tutmayan test ön-kayıtları (resmî test / C0 / C1)}} \\\\"]
    r = R("C1-A2-h10-h02", ad)
    g.append(f"L: H10 $-$ H02c, C1 & GA alt $> 0$ & {isaretli(r['deger'])} {ga(r['ga'])} & \\kotu{{ayırt edilemedi}} \\\\")
    r = R("T-C0-K-h10-h02", ad)
    g.append(f"C0-1: H10 $-$ H02c, C0 & GA alt $> 0$ & {isaretli(r['deger'])} {ga(r['ga'])} & \\kotu{{ayırt edilemedi}} \\\\")
    r = R("T-SN-h08-h02", ad)
    g.append(f"T-SN2: H08c $-$ H02c, $\\Delta_S \\approx \\Delta_N$ & ham $\\Delta_S-\\Delta_N$ GA'sı 0'ı kapsar & "
             f"{isaretli(r['deger'])} {ga(r['ga'])} & \\kotu{{tutmadı ($S > N$)}} \\\\")
    r = R("T-C0-KC0", ad)
    g.append(f"KÇ$_0$: $K_{{0,\\mathrm{{H10}}}} > K_{{0,\\mathrm{{H08}}}}$ & GA alt $> 0$ & "
             f"{isaretli(r['deger'])} {ga(r['ga'])} & \\kotu{{ayırt edilemedi}} \\\\")
    r = R("C1-KC", ad)
    g.append(f"KÇ: $K_{{\\mathrm{{H10}}}} > K_{{\\mathrm{{H08}}}}$ & GA alt $> 0$ & "
             f"{isaretli(r['deger'])} {ga(r['ga'])} & \\kotu{{ayırt edilemedi}} \\\\")
    r = R("C1-KC-oran", ad)
    R("C1-KC-ifade", ad)
    g.append(f"KÇ ifade eşiği & oran $\\geq 0{{,}}5$ ve KÇ desteklenir & oran "
             f"{sayi(r['deger'], 3)}; KÇ desteklenmedi & \\kotu{{FALSE}} \\\\")
    g.append("C2: yayımlanmış C pikselleri & ikincil analiz & koşulmadı (bütçe kararı) & -- \\\\")
    g += ["\\botrule", "\\end{tabularx}"]
    tablo(ad, "Öldürülen kartlar ve tutmayan ön-kayıtlar. Kırmızı eğik: ön-kayıtlı beklenti tutmadı.", "tab:basarisiz", g)


# ------------------------------------------------------------------ (h) Ek: lezyon-kümeli bootstrap (review 9.1(a))
def tab_lkb():
    """Test sonrası duyarlılık analizi; renk YOK (kullanıcı/şef: yalnız GA yan yana + konum değişimi)."""
    ad = "tab_lkb"
    AD = {"baseline": "B", "h01": "H01c", "h02": "H02c", "h08": "H08c", "h10": "H10", "h11": "H11", "h12": "H12",
          "t4": "T4", "baseline_son": "B son-epoch"}
    CIFT = {"h10-h02": "H10 $-$ H02c", "h08-h02": "H08c $-$ H02c", "h10-h08": "H10 $-$ H08c"}
    EK = {"dS": "$\\Delta_S$", "dN": "$\\Delta_N$", "dSdN": "$\\Delta_S{-}\\Delta_N$ ham",
          "tabstd": "$\\Delta_S{-}\\Delta_N$ tab.-std."}

    def cad(c):
        b, a = c.split("-")
        return f"{AD[b]} $-$ {AD[a]}"

    def etiket(anahtar):
        p = anahtar.split("-")
        if anahtar in ("C0-KC0",):
            return "KÇ$_0$ ($K_0$ farkı)"
        if anahtar == "C1-KC":
            return "KÇ ($K$ farkı)"
        if anahtar.startswith("C0-K0-"):
            return "$K_0$: " + ("H10 $-$ H02c" if p[-1] == "H10" else "H08c $-$ H02c")
        if anahtar.startswith("C1-K-"):
            return "$K$: " + ("H10 $-$ H02c" if p[-1] == "H10" else "H08c $-$ H02c")
        if len(p) == 4:  # T-h10-h02-dS
            return f"{CIFT[p[1] + '-' + p[2]]}: {EK[p[3]]}"
        return cad(f"{p[1]}-{p[2]}")

    # Sayfa sığması için bloklar birleştirildi; tablolarda yer almayan H11/H12 betimsel satırları
    # (SENS-LKB-T-h11-*, -h12-*) yalnız defterde ve JSON'dadır (hiçbirinde konum değişmedi).
    BLOK = [(("resmî test, aile1", "resmî test, aile2", "resmî test, betimsel", "resmî test, S/N"),
             "Resmî test (n = 2.005): Aile 1, Aile 2, betimsel; S/N ayrışımı"),
            (("C0", "KÇ_0"), "C0: C-test alt kümesi (n = 1.227); $K_0$ ve KÇ$_0$ ortak bootstrap"),
            (("C1, aile_C1_1", "C1, aile_C1_2", "C1, betimsel", "KÇ"),
             "C1 (n = 1.227): Aile 1, Aile 2, betimsel; $K$ ve KÇ ortak bootstrap")]
    kap = R("SENS-LKB-tablo-kapsam", ad)  # tur-2 (review-tur2 K1): 44 aralık ↔ tablo satırı, tek kaynak
    TABLO_DISI = set(kap["tablo_disi"])
    kay = [r for r in EV["kayitlar"] if r["id"].startswith("SENS-LKB-") and "blok" in r]
    assert not any(K[i]["isaret_degisti"] for i in TABLO_DISI)
    kay = [r for r in kay if r["id"] not in TABLO_DISI]
    R("SENS-LKB-dogrulama", ad); R("SENS-LKB-kume-test", ad); R("SENS-LKB-kume-Ctest", ad); R("SENS-LKB-ozet", ad)
    g = ["\\setlength{\\tabcolsep}{3pt}",
         "\\begin{tabularx}{\\textwidth}{>{\\raggedright\\arraybackslash}X cccc}", "\\toprule",
         "Kıyas & " + " & ".join([iki("$\\Delta$", "[puan]"), iki("Görüntü-düzeyi", "\\%95 GA"),
                                  iki("Lezyon-kümeli", "\\%95 GA"), iki("Konum", "değişti mi")]) + " \\\\",
         "\\midrule"]
    for bi, (blok, baslik) in enumerate(BLOK):
        sat = [r for r in kay if r["blok"] in blok]
        assert sat, blok
        g.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{baslik}}}}} \\\\")
        for r in sat:
            R(r["id"], ad)
            deg = "evet" if r["isaret_degisti"] else "hayır"
            g.append(f"SAT:{etiket(r['id'][len('SENS-LKB-'):])} & {isaretli(r['deger'])} & {ga(r['ga_goruntu'])} & "
                     f"{ga(r['ga'])} & {deg} \\\\")
    g += ["\\botrule", "\\end{tabularx}"]
    n_sat = sum(1 for x in g if x.startswith("SAT:"))
    g = [x[4:] if x.startswith("SAT:") else x for x in g]
    if n_sat != kap["deger"] or n_sat + kap["tablo_disi_sayisi"] != kap["toplam_aralik"]:
        raise SystemExit(f"tab_lkb: {n_sat} satır ↔ kanıt {kap['deger']} / {kap['toplam_aralik']}")
    assert kap["tablo_disi_konum_degisen"] == 0
    tablo(ad, "Test sonrası duyarlılık: görüntü-düzeyi ve lezyon-kümeli bootstrap \\%95 GA'ları (B = 2.000). "
              f"Son sütun: sıfıra göre konum. Konumu değişmeyen {SAYI_AD[kap['tablo_disi_sayisi']]} betimsel H11/H12 "
              "satırı gösterilmedi.", "tab:lkb", g)


# ------------------------------------------------------------------ (h) ön-kayıt sapma tablosu (revizyon 2026-11-03; M4)
def tab_sapma():
    """Ön-kayıtlı plan ↔ gerçekleşen ↔ sapma ↔ sonuca etkisi (revizyon 2026-11-03; eleştiri M4).
    Sayı ve zamanlar yalnız kanıt kayıtlarından (R); niteleyici ifadeler (aşıldı, GA > 0, …) assert ile denetlenir.
    Tek sayfaya sığmadığı için kronolojik iki floata bölünür: tab:sapma (test öncesi + test geçişi) ve
    tab:sapma-devam (test sonrası)."""
    ad = "tab_sapma"

    def z(kid, alan="deger"):
        return str(R(kid, ad)[alan]).replace("T", " ")

    def p(kid, n=2):
        return isaretli(R(kid, ad)["deger"], n)

    def pg(kid, n=2):
        r = R(kid, ad)
        return f"{isaretli(r['deger'], n)} {ga(r['ga'], n)}"

    # ---- tutarlılık denetimleri (metindeki niteleyici ifadeler kanıtla uyuşmalı)
    esik = R("SAP-H08-esik", ad)["deger"]
    assert R("SAP-H08-val-dengeli", ad)["deger"] > esik
    assert R("T-K-h08-h02-dengeli_acc", ad)["deger"] > esik and R("T-G2-H08-bayrak", ad)["deger"] is True
    assert R("T-DOG-H08", ad)["deger"] is True
    sn1_ham, sn1_tab = R("T-K-h10-h02-dSdN", ad), R("T-K-h10-h02-tab_std_dS_eksi_dN", ad)
    sn2_ham, sn2_tab = R("T-K-h08-h02-dSdN", ad), R("T-K-h08-h02-tab_std_dS_eksi_dN", ad)
    assert sn1_ham["ga"][0] > 0 and sn1_tab["ga"][0] > 0                       # T-SN1: iki ölçüde aynı etiket
    assert sn2_ham["ga"][0] > 0 and sn2_tab["ga"][0] < 0 < sn2_tab["ga"][1]    # T-SN2: ölçüye bağlı
    ekb, dond, acil = R("ON-tarih-ekb", ad), R("ON-tarih-dondurma", ad), R("T-meta-tarih", ad)
    assert ekb["dondurmada_sha_var"] is True and dond["deger"] < acil["deger"]
    for k in ("SAP-C1-A-a1", "SAP-C1-Av2-a1"):
        assert R(k, ad)["deger"] == "İHLAL"
    for k in ("SAP-C1-Av3-a1", "SAP-C1-Av3-a3", "SAP-C1-B-a1", "SAP-C1-B-a3"):
        assert R(k, ad)["deger"] == "GEÇTİ"
    assert R("SAP-C1-kuyruk-kol", ad)["deger"] == "k5-c1-B"
    but = R("BUT-toplam", ad)
    assert R("T-t4-auc", ad)["deger"] > 0.99 and R("SAP-T4-auc-denetim", ad)["deger"] is False
    lkb, lkbd = R("SENS-LKB-ozet", ad), R("SENS-LKB-dogrulama", ad)
    assert lkbd["maks_mutlak_fark"] == 0 and lkbd["uyusmaz"] == 0
    d10, d08 = R("C1-K-h10-h02-dengeli_acc", ad), R("C1-K-h08-h02-dengeli_acc", ad)
    assert d10["ga"][0] > 0 and d08["ga"][0] > 0
    mel = {c: R(f"C1-REV-K-{c}-duyarlilik_mel", ad) for c in ("h10-h02", "h08-h02", "h10-h08")}
    assert [c for c, r in mel.items() if r["ga"][0] > 0] == ["h08-h02"]
    nmel = R("C1-REV-n-mel", ad)["deger"]
    assert R("SAP-E1", ad)["deger"] == "beklemede"
    # E1 gerçekleşen (revizyon 2026-11-07): SAP-E1 kaydı 'beklemede' kalır; devamı SAP-E1-gerceklesen + E1-*
    e1g = R("SAP-E1-gerceklesen", ad)
    assert e1g["deger"] == "tamamlandı"
    e1k = R("E1-kontrol-durum", ad)
    assert e1k["deger"] == "A" and e1k["ilk_adim_loss_fark"] == 0 and e1k["argmax_oran"] == 1.0
    e1c = R("E1-ortam-cudnn", ad)
    assert R("E1-ctest-acilis", ad)["deger"] == 5 and R("E1-ortam-torch", ad)["ekimle_ayni"] is True
    e1b = R("E1-butce", ad)
    e1k2 = R("E1-K2-h14-h08-acc", ad)
    assert e1k2["konum"] == "ayırt edilemedi" and e1k2["ga"][0] < 0 < e1k2["ga"][1]
    od_e1 = ", ".join(str(k["satir"]) for k in e1g["kaynaklar"] if "satir" in k)
    for k in ("SAP-C2-kapi", "SAP-T4-plan", "SAP-ON-6-3", "SAP-H10D4-plan", "SAP-C1-dengeli-onkayit", "SAP-C1-A-neden"):
        R(k, ad)  # kaynak satırı defterde olmalı (kullanılan id listesine girer)
    bin_ = format(lkbd["deger"], ",").replace(",", ".")

    def kay(*k):
        return "\\newline{\\itshape " + "; ".join(k) + "}"

    A = [
        ("H08 bayrağı, teste taşınma" + kay("H08:30", f"SIR:626 ({z('SAP-H08-yolA')})", f"KR §2.1, §12 ({z('SAP-karar12')})"),
         f"Dengeli ACC $\\Delta >$ {sayi(esik, 1)} puan ise İNCELE, TUT yok; test yolu öngörülmemiş.",
         f"Doğrulamada $\\Delta$ {pg('SAP-H08-val-dengeli')}; arıza bulunmadı, TUT verilmedi. Yazar kararıyla test "
         f"doğrulama adayı oldu (yol~A); kuralı test öncesi yazıldı. Testte bayrak yine aşıldı: {pg('T-K-h08-h02-dengeli_acc')}.",
         f"Kazanan (H10) değişmedi. H08c kuralı geçti ($\\Delta$ACC {p('T-AILE2-h08-h02')}), ResNet-18 satırı oldu; eşik "
         "gevşetilmedi."),
        ("S/N kural nesnesi (ek~b)" + kay(f"ONb; GN:33 ({z('ON-tarih-ekb')})", f"dondurma {z('ON-tarih-dondurma')[:16]}"),
         "§2.6 kuralının nesnesi (ham ya da tabaka-std. $\\Delta_S-\\Delta_N$) belirsiz; düzeltme yalnız ek~b ile, test öncesi.",
         f"Ek~b ham farkı seçti; iki ölçünün doğrulama değerleri ({p('ON-ekb-val-ham')}; {p('ON-ekb-val-tab')}) "
         f"görülmüştü. Ek dondurmaya girdi, test açılışından ({z('T-meta-tarih')[11:16]}) önce.",
         f"T-SN1 iki ölçüde aynı. T-SN2 ham ölçüde tutmadı; tabaka-std. ölçüde ({pg('T-K-h08-h02-tab_std_dS_eksi_dN')}) "
         "tutardı."),
        ("C1 kuyruk tekrarları (A, A-v2, A-v3)" + kay("BK", "UN", "GN:71--72", "koşum config"),
         "§4.2: C1 tarifleri, kolları, seed'leri donmuş; kapı İHLAL'inde düzeltme ve yeni kuyruk adı.",
         f"Kuyruk A (B, H01c, H02c): a1 {z('SAP-C1-A-a1', 'zaman')[11:]} İHLAL (iki duman rc = 1) $\\to$ üç yardımcı betik "
         f"düzeltildi; A-v2 a1 {z('SAP-C1-Av2-a1', 'zaman')[11:]} İHLAL (kapı kuralı dumana da uygulandı) $\\to$ yalnız "
         f"kapsam düzeltildi; A-v3 GEÇTİ ({z('SAP-C1-Av3-a3', 'zaman')[11:]}). H08c ve H10 kuyruk B'de bir kez koştu "
         f"({z('SAP-C1-B-a3', 'zaman')[11:]}).",
         "Geçersiz ölçüm koşumu yok (yalnız duman). Tarif, kural ve analiz betiği aynı; üç betiğin SHA256'sı dondurmayla "
         "uyuşmaz."),
        ("C2 (yayımlanmış C pikselleri)" + kay("ON:214", f"GN:35 ({z('SAP-C2-karar')})", "GN:56"),
         f"İkincil; C1 sonunda defter $\\leq$ {sayi(R('SAP-C2-kapi', ad)['deger'], 1)}~sa ise koşulur, değilse sınırlılık.",
         f"Yazar kararıyla C1 sürerken atlandı; kapı işletilmedi. C1 sonu defter {sayi(but['deger'], 1)} oturum-sa "
         f"(koşum toplamı {sayi(but['kosum_suresi_saat'], 1)}~sa).",
         "C2 $-$ C1 görüntü hazırlama etkisi ölçülmedi; C1 kararları etkilenmez."),
    ]
    B = [
        ("T4 (H08c $\\oplus$ H10)" + kay("ON:68", f"KR §2.2, §12 ({z('SAP-karar12')})"),
         "Ek tabloda; iddia yalnız H08 doğrulanırsa; kazanan olamaz.",
         f"H08 doğrulandı; yine de yalnız betimsel ek satır. Test AUC'si {sayi(R('T-t4-auc', ad)['deger'], 4)} "
         "($> 0{,}99$) için teste yeniden bakılmadı.",
         "Plandan dar iddia; karar ya da Holm ailesi yok; AUC bayrağı açık."),
        ("H10 + D4 birleşim kartı" + kay("SIR:86--87", f"GN:32 ({z('SAP-karar10')})", "ON §6.3", "KR §12"),
         "H08 K5'te doğrulanır ve DINOv2 kolu TUT alırsa H10/H11 + D4 kartı açılır.",
         "Koşullar ancak test açılınca sağlandı; seçim döngüsü Tur~5'te kapanmıştı; test sonrası yeni kol eklenmez.",
         "H10 $\\leftrightarrow$ H08c kıyası omurga ile augmentasyonu karıştırır (bkz. E1)."),
        ("Lezyon-kümeli bootstrap" + kay(f"MOD:40 ({z('SAP-LKB-karar')})", "LKB"),
         "Test sonrası analiz ``keşifsel'' etiketiyle, ayrı bölümde (§6.3).",
         f"Yazar kararıyla (hakem önerisi) dondurulmuş tahminlerde yapıldı; önce {bin_} alan birebir yeniden üretildi.",
         f"Karar ve Holm sonuçları aynı; {lkb['toplam_satir']} GA'dan {lkb['deger']}'ünün konumu değişti, hepsi betimsel "
         "ya da ikincil (Ek~\\ref{sec:ek-lkb})."),
        ("C1 dengeli ACC ve mel duyarlılığı" + kay("ON:201", "C1M (2026-11-03)"),
         "§4.3: C1 dengeli ACC GA'larıyla; yön H10 $-$ H02c ve H08c $-$ H02c $> 0$. Mel ön-kayıtlı değil.",
         "Dengeli GA'lar ilk sürümde verilmemişti, Tablo~\\ref{tab:protokol}'e eklendi. Mel test sonrası hesaplandı, "
         "aynı tabloda § işaretiyle verildi (§6.3 ``ayrı bölüm'' koşulundan sapma).",
         f"Dengeli yön iki çiftte desteklendi ({p('C1-K-h10-h02-dengeli_acc')}; {p('C1-K-h08-h02-dengeli_acc')}). Mel: "
         f"yalnız H08c $-$ H02c GA'sı sıfırın üstünde; n~=~{nmel}, GA geniş."),
        ("E1 / H14: H10 + D4 (C1)" + kay(f"EP §2b ({z('SAP-E1', 'zaman')})", "OD:15",
                                          f"OD:{od_e1} ({z('SAP-E1-gerceklesen', 'zaman')})"),
         f"Test sonrası yeni kol yok (§6.3); H10 + D4 sınırlılık ({z('SAP-karar12')}).",
         f"Yazar kararıyla (dış eleştiri) C1'de ``test sonrası keşifsel'' kol; H14 kartı koşumdan önce yazıldı "
         f"({z('SAP-E1-gerceklesen', 'kart_zaman')}). Beş seed, C-test seed başına bir kez; kontrol kolu Ekim dumanını "
         f"tolerans içinde üretti (Durum~A: ilk adım kaybı aynı, olasılık farkı $\le 10^{{-5}}$; cuDNN {e1c['karsi_ekim']} $\\to$ {e1c['deger']}); {sayi(e1b['deger'], 1)}~L4-sa.",
         f"Karar ve Holm sonuçları aynı (düzeltmesiz, Holm dışı). H14 $-$ H08c $\\Delta$ACC {pg('E1-K2-h14-h08-acc')}: "
         "ayırt edilemedi (Tablo~\\ref{tab:e1})."),
    ]
    BAS = ("\\begin{tabularx}{\\textwidth}{>{\\raggedright\\arraybackslash\\hsize=0.80\\hsize}X "
           ">{\\raggedright\\arraybackslash\\hsize=0.95\\hsize}X >{\\raggedright\\arraybackslash\\hsize=1.35\\hsize}X "
           ">{\\raggedright\\arraybackslash\\hsize=0.90\\hsize}X}")

    def govde(satirlar):
        g = [BAS, "\\toprule", "Madde (kaynak) & Ön-kayıtlı plan & Gerçekleşen ve sapma & Sonuca etkisi \\\\", "\\midrule"]
        for i, (m, a, b, c) in enumerate(satirlar):
            g.append(f"{m} & {a} & {b} & {c} \\\\")
            if i < len(satirlar) - 1:
                g.append("\\addlinespace")
        return g + ["\\botrule", "\\end{tabularx}"]

    notlar = ("\\textit{Kaynaklar} (dosya:satır; zamanlar yerel saat): H08 = \\texttt{hipotezler/H08.md}; "
              "SIR = \\texttt{hipotezler/SIRALAMA.md}; ON = \\texttt{hipotezler/K5-ONKAYIT.md}; "
              "ONb = \\texttt{hipotezler/K5-ONKAYIT-b.md}; UN = \\texttt{hipotezler/K5-C1-UYGULAMA-NOTU.md}; "
              "BK = \\texttt{sonuclar/beklenti-k5-c1-*.md}; KR = \\texttt{sonuclar/karar.md}; "
              "GN = \\texttt{gunluk/deney-gunlugu.md}; OD = \\texttt{orkestrasyon\\_durumu.md}; "
              "MOD, EP = makale çalışma notları; LKB, C1M = \\texttt{evidence/lkb}, \\texttt{evidence/c1mel} çıktıları.")
    tablo(ad, "Ek: ön-kayıtlı plan, gerçekleşen akış, sapma ve sonuca etkisi (1/2: test öncesi ve test geçişi).",
          "tab:sapma", govde(A), notlar=notlar)
    KULLANILAN["tab_sapma_b"] = list(KULLANILAN[ad])
    tablo("tab_sapma_b", "Ek: ön-kayıt sapmaları (2/2: test sonrası). Kaynak kısaltmaları Tablo~\\ref{tab:sapma} "
                         "altındadır.", "tab:sapma-devam", govde(B))


# ------------------------------------------------------------------ (i) E1 / H14 (revizyon 2026-11-07; test sonrası keşifsel)
def tab_e1():
    """H14 = H10 + eğitim-zamanı D4, C1. Test sonrası keşifsel ve düzeltmesiz → renk/kazanan vurgusu YOK
    (kullanıcı/şef kararı; tab_lkb ile aynı ilke). Hücreler yalnız E1-* kayıtlarından."""
    ad = "tab_e1"
    KIY = [("E1-K1-h14-h10", "H10", "K1"), ("E1-K2-h14-h08", "H08c", "K2"), ("E1-BET-h14-h02", "H02c", "betimsel")]
    MET = [("acc", "ACC"), ("dengeli_acc", "Dengeli ACC"), ("dengeli_acc_5sinif", "Dengeli-5 ACC"),
           ("duyarlilik_mel", "Mel duyarlılığı")]
    assert R("E1-dogrulama", ad)["maks_fark_puan"] == 0 and R("E1-ctest-acilis", ad)["deger"] == 5
    assert R("E1-kontrol-durum", ad)["deger"] == "A"
    def uc(a, b, c):
        return f"\\begin{{tabular}}[c]{{@{{}}c@{{}}}}{a}\\\\\\relax {b}\\\\\\relax {c}\\end{{tabular}}"

    g = ["\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccc}", "\\toprule",
         "Metrik & " + " & ".join([uc("H14", "ort.\\,$\\pm$\\,std", "[oran]")] +
                                  [uc(f"H14 $-$ {kol}", f"({rol})", "$\\Delta$ [puan], GA") for _, kol, rol in KIY])
         + " \\\\", "\\midrule"]
    nmel = None
    for m, etiket in MET:
        h = R(f"E1-H14-{m}", ad)
        hucre = [pm(h["deger"], h["std"], 4)]
        for onek, _, _ in KIY:
            r = R(f"{onek}-{m}", ad)
            assert r["konum"] in ("ayırt edilemedi", "H14 üstün", "H14 geride")
            if m == "duyarlilik_mel":
                nmel = r["n_mel"]
            hucre.append(iki(isaretli(r["deger"]), ga(r["ga"])))
        g.append(f"{etiket} & " + " & ".join(hucre) + " \\\\")
        g.append("\\addlinespace")
    g[-1] = "\\midrule"
    for t, etiket in [("t_acc", "$t$ testi $p$: ACC"), ("t_dengeli", "$t$ testi $p$: dengeli")]:
        g.append(f"{etiket} & -- & " + " & ".join(pdeg(R(f"{onek}-{t}", ad)["p_iki_yonlu"]) for onek, _, _ in KIY)
                 + " \\\\")
    g += ["\\botrule", "\\end{tabular*}"]
    assert not any("\\iyi" in x or "\\kotu" in x for x in g)  # keşifsel: vurgu yok
    notlar = ("$\\Delta$ = H14 $-$ kol; GA: görüntü-bootstrap \\%95 yüzdelik aralığı (B = 2.000, rng 0, iki kola aynı "
              f"ağırlıklar); mel n = {nmel}. $t$: seed-eşli, iki yönlü, n = 5, yalnız bilgi. Test sonrası keşifsel, "
              "düzeltmesiz; K5 Holm ailelerine girmez; vurgu yok.")
    tablo(ad, "E1: H14 (H10 + eğitimde D4), lezyon-ayrık C1 C-test, 5 seed. H14 değerleri ve arşiv kollarına "
              "eşleştirilmiş farklar.", "tab:e1", g, notlar=notlar)


if __name__ == "__main__":
    tab_ana()
    tab_kiyas()
    tab_ablasyon()
    tab_protokol()
    tab_replikasyon()
    tab_t4()
    tab_basarisiz()
    tab_lkb()
    tab_sapma()
    tab_e1()
    (OUT / "KULLANILAN_IDLER.json").write_text(json.dumps(KULLANILAN, ensure_ascii=False, indent=1), encoding="utf-8")
