# -*- coding: utf-8 -*-
"""Faz 6 birleştirici: sections/*.tex + evidence/tables/*.tex -> paper/main.tex (Springer, gömülü)."""
import re
import sys
from pathlib import Path

WS = Path(sys.argv[1])
SEC = WS / "sections"
TAB = WS / "evidence" / "tables"


def yorum_sil(metin: str) -> str:
    out = []
    for satir in metin.splitlines():
        m = re.search(r"(?<!\\)%", satir)
        if m:
            kod = satir[:m.start()].rstrip()
            if not kod.strip():
                continue  # tamamen yorum satırı
            out.append(kod)
        else:
            out.append(satir.rstrip())
    # art arda boş satırları teke indir
    s = "\n".join(out)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip("\n") + "\n"


def tablo(ad: str) -> str:
    return yorum_sil((TAB / f"{ad}.tex").read_text(encoding="utf-8"))


def bolum(ad: str) -> str:
    s = (SEC / f"{ad}.tex").read_text(encoding="utf-8")
    s = yorum_sil(s)
    s = re.sub(r"\\input\{\.\./evidence/tables/([\w_]+)\}", lambda m: tablo(m.group(1)).rstrip("\n"), s)
    s = re.sub(r"\\input\{\.\./figures/([\w_]+)\}", r"\\input{figures/\1}", s)
    return s


ozet = bolum("ozet")
assert ozet.startswith("\\abstract{")
ozet = ozet.replace("\\abstract{", "\\abstract{\\unboldmath ", 1)
govde = [bolum(a) for a in ["giris", "ilgili-calismalar", "yontem", "deney-kurulumu", "replikasyon",
                             "deney-kaydi", "bulgular", "basarisiz-denemeler", "tartisma", "sonuc"]]

ek = bolum("ek")
i = ek.index("\\subsection{Veri ve kod erişilebilirliği}")
ek_govde, erisim = ek[:i].rstrip() + "\n", ek[i:]
veri = re.search(r"\\paragraph\{Veri\.\}\s*(.*?)\n\s*\n", erisim + "\n\n", flags=re.S).group(1).strip()
kod = re.search(r"\\paragraph\{Kod\.\}\s*(.*?)\s*$", erisim, flags=re.S).group(1).strip()

BEKLE = "\\textbf{[Yazar beyanı bekleniyor]}"
FLOAT_BARIYER = []

main = r"""%% Springer Nature (sn-jnl) — makale-uret Faz 6 birleştirmesi.
%% Derleme: tectonic main.tex  ya da  latexmk -pdf main.tex
%% Gerekli dosyalar bu klasörde: sn-jnl.cls, sn-mathphys-num.bst (bst/), references.bib,
%% figures/ (pgfplots kaynakları, .dat verileri, nitel panel karoları).
\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}

\usepackage{iftex}
\ifXeTeX
  \usepackage{fontspec}
\else
  \usepackage[utf8]{inputenc}
  \usepackage[T1]{fontenc}
\fi
\usepackage[turkish,shorthands=off]{babel}
\usepackage{graphicx}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{amsthm}
\usepackage{dsfont}
\usepackage{xcolor}
\usepackage{booktabs,array,tabularx}
\usepackage[title]{appendix}
\usepackage{placeins}
\usepackage{xurl}
%% Kaynakçadaki \burl (sn-mathphys-num) alt çizgili URL'lerde \url gibi davransın
\let\burl\url
\hypersetup{bookmarksdepth=2}
\usepackage{pgfplots}
\usepackage{pgfplotstable}
\pgfplotsset{compat=1.18}
\usepgfplotslibrary{groupplots}
\usetikzlibrary{arrows.meta,positioning,fit,backgrounds,calc,babel}

%% Tablo renk makroları (figure-style kural 7)
\definecolor{iyiRenk}{RGB}{0,84,159}
\definecolor{kotuRenk}{RGB}{178,24,43}
\newcommand{\iyi}[1]{\textcolor{iyiRenk}{\textbf{#1}}}
\newcommand{\kotu}[1]{\textcolor{kotuRenk}{\textit{#1}}}
\newcolumntype{Y}{>{\centering\arraybackslash}X}

%% Figür kaynak yolları (figures/*.tex bu makroları kullanır)
\newcommand{\figveri}{figures/data}
\newcommand{\figtiles}{figures/tiles}

%% sn-jnl anahtar kelime başlığı (Türkçe)
\def\keywordname{Anahtar kelimeler}

\raggedbottom
%% raggedbottom altında başlık öncesi negatif ceza sayfaları erken kesiyor; nötrle
\makeatletter\@secpenalty=0\makeatother

\begin{document}

\title[Bölmeye Duyarlı Üstünlük: DermaMNIST'te DINOv2, ResNet-18 ve Lezyon Tekrarı]{DermaMNIST'te Temel Model Üstünlüğü Bölme Protokolüne Bağlı mı? Lezyon-Ayrık Değerlendirmeyle DINOv2 ve ResNet-18'in Ön-Kayıtlı Karşılaştırması}

\author*[1]{\fnm{<YAZAR_1_AD>} \sur{<YAZAR_1_SOYAD>}}\email{<e-posta>}
\author[1]{\fnm{<YAZAR_2_AD>} \sur{<YAZAR_2_SOYAD>}}\email{<e-posta>}
\author[1]{\fnm{<YAZAR_3_AD>} \sur{<YAZAR_3_SOYAD>}}\email{<e-posta>}
\affil[1]{\orgname{Ostim Technical University},
  \orgaddress{\street{OSTİM, 100. Yıl Bulvarı, 55/F}, \postcode{06374},
  \city{Yenimahalle}, \state{Ankara}, \country{Türkiye}}}

""" + ozet + r"""
\maketitle

""" + "\n".join(govde) + r"""
\backmatter

\section*{Beyanlar}

\bmhead{Finansman}
Bu çalışma herhangi bir kurum ya da kuruluştan fon desteği almamıştır.

\bmhead{Çıkar çatışması}
Yazarlar, bu makalenin içeriğiyle ilgili herhangi bir çıkar çatışması bulunmadığını beyan eder.

\bmhead{Etik onay ve katılım onayı}
Uygulanamaz. Çalışmada yalnız kamuya açık ve anonimleştirilmiş DermaMNIST (HAM10000) görüntüleri kullanılmış; insan ya da hayvan katılımcıdan yeni veri toplanmamıştır.

\bmhead{Yayın onayı}
Uygulanamaz.

\bmhead{Veri erişilebilirliği}
""" + veri + r"""

\bmhead{Kod erişilebilirliği}
""" + kod + r"""

\bmhead{Yazar katkıları}
K.K.: kavramsallaştırma, yöntem, yazılım, deneylerin yürütülmesi, veri düzenleme, görselleştirme, yazım (ilk taslak). İ.Y. ve U.A.: danışmanlık, yöntemin gözden geçirilmesi, yazım (gözden geçirme ve düzenleme). Tüm yazarlar son metni okumuş ve onaylamıştır.

\begin{appendices}
\renewcommand{\theHtable}{ek.\arabic{table}}

""" + ek_govde + r"""
\end{appendices}

\bibliography{references}

\end{document}
"""
main = main.replace("BEKLE", BEKLE)
for bas in FLOAT_BARIYER:
    assert main.count(bas) == 1, bas
    main = main.replace(bas, "\\FloatBarrier\n" + bas)
(WS / "paper" / "main.tex").write_text(main, encoding="utf-8", newline="\n")
print("yazıldı", len(main.splitlines()), "satır")
