# Ortam Kurulumu — 3.1 (hedef A: bu makine, GPU)

Donanım: NVIDIA GeForce GTX 1650 4 GB, sürücü 555.97 · RAM 15.9 GB · Windows 11 (`env/ortam-yerel.md`, rc=0).
Not: sistem Python 3.13 bozuk (Lib\ yok) → uv-yönetimli Python 3.11.16 kullanılır.

```bash
UV=uv
$UV venv env/.venv -p 3.11
$UV pip install --python env/.venv/Scripts/python.exe torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
$UV pip install --python env/.venv/Scripts/python.exe medmnist timm scikit-learn scipy pandas "numpy<2.3" pillow tqdm huggingface_hub hf_xet
# tam tekrar:
# $UV pip install --python env/.venv/Scripts/python.exe -r env/requirements.lock --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple
```

Doğrulama (2026-10-10):
```
$ env/.venv/Scripts/python.exe -c "import torch,torchvision,timm,medmnist,sklearn; print(...)"
2.5.1+cu121 True NVIDIA GeForce GTX 1650 0.20.1+cu121 1.0.30 3.0.2 1.9.1
```
Pinler: torch 2.5.1+cu121 · torchvision 0.20.1+cu121 · timm 1.0.30 · medmnist 3.0.2 · scikit-learn 1.9.1
(tamamı `env/requirements.lock`, 44 paket).

Koşum kuralları:
- Kuyruk ve deney komutlarında `python` DEĞİL `env/.venv/Scripts/python.exe` (sistem python bozuk).
- Orkestra betikleri (kuyruk dışı): `<PYTHON311>`, `PYTHONIOENCODING=utf-8`.
- `kuyruk.py kos` deney venv'inin Python'u ile çağrılır (`--gpu-zorunlu` için torch gerekir).
- Önbellekler WS içinde: `HF_HOME=WS/data/hf_cache`, `TORCH_HOME=WS/data/torch_cache`, MedMNIST kökü `WS/data/medmnist`.
- GPU tek: aynı anda tek eğitim (`--paralel 1`).

## GÜNCELLEME 2026-10-10 — Hedef D: Colab Pro (L4)
Tüm ölçüm koşumları Colab Pro'da, sabit **L4 24 GB** sınıfında. Yerel `env/.venv` yalnız duman testleri ve analiz için.
Colab ortamı: defterin ilk hücresi aynı pinleri kurar (`torch==2.5.1` cu121, `torchvision==0.20.1`, `timm==1.0.30`,
`medmnist==3.0.2`, `scikit-learn==1.9.1`) ve `nvidia-smi` + `torch.cuda.get_device_name()` çıktısını
`runs/<id>/config.json`'a (`donanim` alanı) yazar; L4 değilse defter durur.

## GÜNCELLEME 2026-10-10 — K3 rakip bağımlılığı
Resmî rakip betiği (`MedMNIST/experiments@70b6b3a`, `train_and_eval_pytorch.py`) `tensorboardX` içe aktarır →
`uv pip install --python env/.venv/Scripts/python.exe tensorboardX` (tensorboardx 2.6.5 + protobuf 7.36.2);
`env/requirements.lock` yeniden donduruldu (46 paket). Colab defteri `tensorboardX==2.6.5` pinini kurar. Ayrıntı: `rakip/KURULUM.md`.
