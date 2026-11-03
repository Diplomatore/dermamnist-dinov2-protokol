#!/usr/bin/env bash
# Resmî rakip kodunu (MedMNIST/experiments) pinli commit'te rakip/experiments/ altına getirir.
# Resmî kod bu depoya kopyalanmaz; köprüler: scripts/rakip_kos.py, varyantlar/k5/c_rakip_kos.py
set -euo pipefail
cd "$(dirname "$0")"
PIN=70b6b3a7ad7afddff1df2a3b735235830fbdb142
[ -d experiments/.git ] || git clone https://github.com/MedMNIST/experiments.git experiments
git -C experiments fetch --quiet origin
git -C experiments checkout --quiet "$PIN"
git -C experiments log -1 --format='%H %ci %s'
# Köprünün denetlediği blob SHA1'leri (bkz. KURULUM.md §1):
git -C experiments rev-parse "$PIN:MedMNIST2D/train_and_eval_pytorch.py"   # 09157bd8d6032be12cbf1eea0cfa8ccbd39914cb
git -C experiments rev-parse "$PIN:MedMNIST2D/models.py"                   # 24579d6612441aa1e1af63f3d246d4e61db822fd
