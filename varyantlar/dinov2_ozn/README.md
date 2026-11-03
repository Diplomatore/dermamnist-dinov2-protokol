# dinov2_ozn — donuk DINOv2 CLS öznitelik önbelleği (H01, H03)
Ebeveynden fark: yok (yardımcı adım); sondanın girdisi. Seed'den bağımsız, girdi başına bir kez.
Önbellek `data/oznitelik/dinov2_vit{s,b,l}14_r<r>[_n<N>].npz` + `.json` meta (model, timm sürümü, ağırlık SHA256 +
HF revizyonu, girdi npz MD5/SHA256, süre, img/s, determinizm farkı). Test öznitelikleri yalnız K5'te, ayrı `_test.npz`.
Normalde `sonda.py` önbellek yoksa bunu kendisi çağırır. Elle: bkz. `../KOMUTLAR.md` §0.
