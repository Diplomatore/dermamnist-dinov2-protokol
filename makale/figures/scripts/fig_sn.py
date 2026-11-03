# -*- coding: utf-8 -*-
"""F2 — S/N ayrışımı (nokta + %95 görüntü-bootstrap GA), resmî test.
Kaynak [T] sonuclar/k5_test_sonuc.json:
  kiyas.<c>.acc (genel Δ), kiyas.<c>.acc_S_ham, kiyas.<c>.acc_N_ham, kiyas.<c>.ham_dS_eksi_dN ; n, n_S, n_N
Seed düzeyi S/N JSON'da yok → çizilmez (brief madde 13)."""
from _ortak import dat_yaz, eng, jload

T = jload("sonuclar/k5_test_sonuc.json")
satirlar = []
# y: 3 = genel (üst), 2 = S, 1 = N, 0 = S − N (alt)
for p, c in enumerate(["h10-h02", "h08-h02"]):
    k = T["kiyas"][c]
    for y, (d, g) in [(3, (k["acc"]["delta"], k["acc"]["ga"])),
                      (2, (k["acc_S_ham"]["delta"], k["acc_S_ham"]["ga"])),
                      (1, (k["acc_N_ham"]["delta"], k["acc_N_ham"]["ga"])),
                      (0, (k["ham_dS_eksi_dN"]["nokta"], k["ham_dS_eksi_dN"]["ga"]))]:
        d, lo, hi = 100 * d, 100 * g[0], 100 * g[1]
        lab = f"{eng(d)} [{eng(lo)}, {eng(hi)}]"
        satirlar.append([p, y, f"{d:.4f}", f"{d - lo:.4f}", f"{hi - d:.4f}", "{" + lab + "}"])
dat_yaz("fig_sn.dat", ["panel", "y", "d", "em", "ep", "lab"], satirlar,
        "[T] kiyas.{h10,h08}-h02.{acc, acc_S_ham, acc_N_ham, ham_dS_eksi_dN} ×100")
yl = [[3, "{Overall (n = %d)}" % T["n"]], [2, "{Seen lesion, S (n = %d)}" % T["n_S"]],
      [1, "{Unseen lesion, N (n = %d)}" % T["n_N"]], [0, "{$\\Delta_S - \\Delta_N$}"]]
dat_yaz("fig_sn_yl.dat", ["y", "lab"], sorted(yl), "[T] n, n_S, n_N")
