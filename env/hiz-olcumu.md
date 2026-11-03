# Yerel GPU Hız Ölçümü — 2026-10-10 (sentetik veri, GTX 1650 4 GB, torch 2.5.1+cu121)

| İş | Hız | Bellek |
|---|---|---|
| ResNet-18 28 px eğitim FP32 bs128 | 4261 img/s | 3.06 GB |
| ResNet-18 224 px eğitim FP32 bs64 | 214 img/s | 2.81 GB |
| ResNet-18 224 px eğitim AMP-FP16 bs64 | 62 img/s (FP16 patolojik yavaş) | 2.25 GB |
| DINOv2 ViT-S/14 çıkarım FP32 (FP16) | 109 (34) img/s | — |
| DINOv2 ViT-B/14 çıkarım FP32 (FP16) | 26 (9) img/s | — |
| DINOv2 ViT-L/14 çıkarım FP32 (FP16) | 10 (3) img/s | — |
| DINOv2 ViT-B/14 tam ince ayar FP32 + grad ckpt bs16 | 7 img/s | 1.64 GB |

Betikler: scratchpad `hiz.py`, `hiz2.py` (timm `*_dinov2`, rastgele ağırlık, 224 px). Sonuç: DINOv2-B tam ince ayar
5 seed ≈ 28 sa (bütçe dışı) → kullanıcı Colab Pro'ya geçti. Bu sayılar makalede donanım karşılaştırması olarak
KULLANILMAZ (sentetik, tek koşum); yalnız hedef kararının gerekçesidir.
