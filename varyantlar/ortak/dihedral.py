# -*- coding: utf-8 -*-
"""
dihedral.py — D4 (224×224 karenin 8 dihedral dönüşümü), NCHW tensörde, uzamsal eksenler (2, 3).
H07 (TTA) ve H08 (eğitim augmentasyonu) AYNI tanımı kullanır (tek kaynak).

İndeks ↔ dönüşüm (H07/H08 kart sırası: kimlik, 90°/180°/270° döndürme, yatay çevirme, dikey çevirme, transpoz,
ters-transpoz):
  0 kimlik · 1 rot90(k=1) · 2 rot90(k=2) (180°) · 3 rot90(k=3) · 4 flip(3) (yatay çevirme) · 5 flip(2) (dikey çevirme)
  6 transpose(2,3) (transpoz) · 7 ters-transpoz = flip(3)(rot90(x, 1)), yani B[i,j] = A[n−1−j, n−1−i].
Not (kart yazımı "rot90(1)∘flip(3)"): boru hattı sırasıyla okunur (önce rot90, sonra flip(3)). Matematiksel bileşke
okuması (önce flip(3), sonra rot90) transpozu verir ve D4'ü 7 ayrık elemana düşürürdü; kart "8 dihedral" dediği için
boru hattı okuması tek tutarlı yorumdur. `oz_sinama()` 8 dönüşümün ayrık ve grup kapalı olduğunu doğrular.
D2 alt kümesi (sıkıştırma eksenini koruyan): {0, 2, 4, 5}. 90° alt kümesi (eksenleri değiştiren): {1, 3, 6, 7}.
Her dönüşümden sonra .contiguous() (NHWC/adımlı tensör deterministik cuDNN'i yavaş yola sokuyordu; KOMUTLAR §T2).
"""
D2 = (0, 2, 4, 5)
R90 = (1, 3, 6, 7)
AD = ["kimlik", "rot90", "rot180", "rot270", "yatay_cevirme", "dikey_cevirme", "transpoz", "ters_transpoz"]


def donustur(x, k: int):
    import torch
    k = int(k)
    if k == 0:
        return x
    if k in (1, 2, 3):
        y = torch.rot90(x, k, dims=(2, 3))
    elif k == 4:
        y = torch.flip(x, dims=(3,))
    elif k == 5:
        y = torch.flip(x, dims=(2,))
    elif k == 6:
        y = x.transpose(2, 3)
    elif k == 7:
        y = torch.flip(torch.rot90(x, 1, dims=(2, 3)), dims=(3,))
    else:
        raise ValueError(f"D4 indeksi 0..7 olmalı: {k}")
    return y.contiguous()


def oz_sinama(n: int = 5) -> bool:
    """8 dönüşüm ayrık mı, ters-transpoz tanıma uyuyor mu, küme bileşkeye kapalı mı (rastgele n×n tensörde)."""
    import torch
    g = torch.Generator().manual_seed(0)
    x = torch.rand((1, 1, n, n), generator=g)
    V = [donustur(x, k) for k in range(8)]
    for a in range(8):
        for b in range(a + 1, 8):
            if torch.equal(V[a], V[b]):
                return False
    A = x[0, 0]
    tt = torch.empty_like(A)
    for i in range(n):
        for j in range(n):
            tt[i, j] = A[n - 1 - j, n - 1 - i]
    if not torch.equal(V[7][0, 0], tt):
        return False
    for a in range(8):  # kapalılık: T_a(T_b(x)) kümede
        for b in range(8):
            y = donustur(donustur(x, b), a)
            if not any(torch.equal(y, v) for v in V):
                return False
    return True
