# -*- coding: utf-8 -*-
"""F5 — nitel panel (VAL; betimsel, karar taşımaz). figure-style §A2.
Kaynak: figures/data/nitel_panel_val.json (Faz 1.5, deney-gelistirici; DW/scripts/nitel_panel_val.py,
SHA256 json içinde) + karolar figures/tiles/nitel/val_XXXX.png (npz pikselleriyle bayt-özdeş, eksensiz).
Bu betik figures/fig_nitel.tex'i ÜRETİR: bütün etiketler (sınıf, S/N, tahmin, doğru seed sayısı) LaTeX
kaynağında kalır; raster yalnız karo içindedir. Test görüntüsü/tahmini kullanılmaz."""
import json
import re
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
J = json.loads((WS / "figures/data/nitel_panel_val.json").read_text(encoding="utf-8"))
assert "Test açılmadı" in J["aciklama"]

KAT_EN = {
    "N-a": "Unseen lesion (N):\\\\H08c right, H10 wrong",
    "N-b": "Unseen lesion (N):\\\\H10 right, H08c wrong",
    "S-a": "Seen lesion (S):\\\\H08c right, H10 wrong",
    "S-b": "Seen lesion (S):\\\\H10 right, H08c wrong",
    "DV-a": "df/vasc:\\\\H08c right, H02c wrong",
    "DV-b": "df/vasc:\\\\H02c right, H08c wrong",
    "Z": "Shared hard core:\\\\all four arms wrong\\\\in 5/5 seeds",
}
KOLLAR = [("B", "B"), ("H02c", "H02c"), ("H08c", "H08c"), ("H10", "H10")]
kategoriler = [k["kod"] for k in J["secim_olcutu"]["kategoriler"]]
gor = {}
for g in J["goruntuler"]:
    gor.setdefault(g["kategori"], []).append(g)
n_aday = J["kategori_aday_sayisi"]


def bilgi(g):
    s = [f"\\#{g['val_indeks']} $\\cdot$ {g['gercek_sinif']} $\\cdot$ {g['S_N']}"]
    for k, ad in KOLLAR:
        v = g["kollar"][k]
        dogru = v["dogru_seed_sayisi"] >= 3          # tanim: cogunluk_dogru = dogru_seed_sayisi >= 3
        # çoğunluk doğruysa en sık tahmin gerçek sınıftır; tersi gerekmez (ör. 2/5 doğru + bölünmüş oylar)
        assert (not dogru) or v["cogunluk_tahmin"] == g["gercek_sinif"], (g["val_indeks"], k)
        metin = f"{ad}: {v['cogunluk_tahmin']} {v['dogru_seed_sayisi']}/5"
        s.append(f"\\textcolor{{iyiRenk}}{{\\textbf{{$\\checkmark$\\,{metin}}}}}" if dogru
                 else f"\\textcolor{{kotuRenk}}{{\\textit{{$\\times$\\,{metin}}}}}")
    return "\\begin{tabular}[c]{@{}l@{}}" + "\\\\".join(s) + "\\end{tabular}"


def karo(g):
    yol = re.sub(r"^figures/tiles/", "", g["karo"])
    assert (WS / g["karo"]).is_file(), g["karo"]
    return f"\\raisebox{{-0.5\\height}}{{\\includegraphics[width=\\nitelKaro]{{\\figtiles/{yol}}}}}"


satirlar = []
for kod in kategoriler:
    bas = f"\\begin{{tabular}}[c]{{@{{}}l@{{}}}}{KAT_EN[kod]}\\\\(candidates: {n_aday[kod]})\\end{{tabular}}"
    gl = gor.get(kod, [])
    if not gl:
        assert n_aday[kod] == 0
        satirlar.append(f"{bas} & \\multicolumn{{4}}{{c}}{{\\textit{{no candidate in the validation set}}}} \\\\")
        continue
    hucre = []
    for g in gl[:2]:
        hucre += [karo(g), bilgi(g)]
    satirlar.append(f"{bas} & " + " & ".join(hucre) + " \\\\[5pt]")

# tur-1 (review 7.1): başlık TAMAMI ≤ 40 kelime. Seçim ölçütü cümlesi (J["baslik_cumlesi"]) gövdeye taşındı
# (bulgular, sec:bulgular-h10h08); "DV-b'de aday yok" bilgisi panelin kendisinde ("no candidate") duruyor.
kalan = ("Doğrulama kümesi nitel paneli (betimsel, karar taşımaz): dört kolun en sık tahmini ve doğru seed sayısı; "
         "mavi kalın $\\checkmark$ en az 3/5 doğru, kırmızı eğik $\\times$ değil. Görüntüler önceden yazılmış bir "
         "kuralla seçildi (Bölüm~\\ref{sec:bulgular-h10h08}). DermaMNIST (HAM10000), CC BY-NC 4.0.")
kelime = len([w for w in re.sub(r"\\ref\{[^}]*\}", "X", re.sub(r"\$[^$]*\$", "X", kalan)).replace("~", " ").split()
              if re.search(r"\w", w)])
assert kelime <= 40, kelime   # §E2: figür başlığı ≤ 40 kelime (tamamı)

tex = [
    "% F5 — NİTEL PANEL (val). OTOMATİK: figures/scripts/fig_nitel.py üretir; elle düzenlemeyin.",
    "% Kaynak: figures/data/nitel_panel_val.json (betik_sha256 " + J["kaynak"]["betik_sha256"][:12] + "…);"
    " karolar figures/tiles/nitel/*.png (§A2 raster istisnası: yalnız piksel; etiketler LaTeX'te).",
    "\\providecommand{\\figtiles}{figures/tiles}",
    "\\providecolor{iyiRenk}{RGB}{0,84,159}",
    "\\providecolor{kotuRenk}{RGB}{178,24,43}",
    "\\ifdefined\\nitelKaro\\else\\newlength{\\nitelKaro}\\fi",
    "\\setlength{\\nitelKaro}{1.62cm}",
    "\\begin{figure}[!htbp]",
    "\\centering\\footnotesize",
    "\\setlength{\\tabcolsep}{2.5pt}\\renewcommand{\\arraystretch}{1.0}",
    "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l c l c l@{}}",
    "\\toprule",
    "Category & \\multicolumn{2}{c}{Example 1} & \\multicolumn{2}{c}{Example 2} \\\\",
    "\\midrule",
] + satirlar + [
    "\\bottomrule",
    "\\end{tabular*}",
    f"\\caption{{{kalan}}}",
    "\\label{fig:nitel}",
    "\\end{figure}",
]
(WS / "figures/fig_nitel.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
print(f"fig_nitel.tex: {sum(len(v) for v in gor.values())} görüntü, {len(kategoriler)} kategori; başlık (tamamı) {kelime} kelime")
