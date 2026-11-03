# Colab ölçüm ortamı (K4/K5 koşumları)

Kaynak: `runs_meta/runs_colab/<id>/config.json` → `donanim`. Her koşum kendi ortamını yazar. Aşağıdaki tablo bu
alanların özetidir: 191 koşumda alanlar dolu (173'ünde `timm`/`medmnist` alanları da var), 5 koşumda `donanim` alanı
yok. Alanı dolu bütün koşumlar tek bir ortamda koşmuştur.

| bileşen | sürüm |
|---|---|
| GPU | NVIDIA L4 24 GB (sürücü 580.82.07) |
| Python | 3.13.15 (Linux 6.6, glibc 2.39) |
| torch / torchvision | 2.11.0+cu130 / 0.26.0+cu130 (CUDA 13.0, cuDNN 9.27) |
| timm | 1.0.30 |
| medmnist | 3.0.2 |
| scikit-learn | 1.6.1 |
| numpy | 2.1.3 |

Sayısal ayarlar (`fp32_durum.json`): TF32 kapalı (`allow_tf32=False`, `fp32_precision="ieee"`,
`float32_matmul_precision="highest"`, `NVIDIA_TF32_OVERRIDE=0`), cuDNN deterministik, `benchmark=False`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. AMP kullanılmadı.

Yerel ortam (duman testleri, K3 yerel replikasyonu, analiz betikleri) için pin dosyası `env/requirements.lock`'tur
(Python 3.11.16, torch 2.5.1+cu121). Bu iki ortam farklıdır. Raporlanan test ve C1 sayılarının tamamı Colab ortamından gelir.
