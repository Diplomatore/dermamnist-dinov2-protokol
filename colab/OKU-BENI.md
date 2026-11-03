# Colab'da K3 baseline koşumu: adım adım

Bu iş, MedMNIST'in resmî ResNet-18 modelini (rakip) DermaMNIST üzerinde 8 kez eğitir: 1 duman, 2 ön-uçuş ve 5 tam koşum.
Yaklaşık **1,5–2 saat** sürer. Sen yalnız başlatıp sonucu indireceksin. Kapıları ve kararları defter ve şef yürütür.

1. **Dosyaları bul.** Bilgisayarında şu iki dosya hazır:
   `deney_workspace\3-1-dermamnist-dinov2\colab\3-1-paket.zip` ve aynı klasördeki `k3_baseline.ipynb`.
2. **Drive'a yükle.** drive.google.com → *Bilgisayarım* → `yz` adında klasör aç (yoksa) → iki dosyayı da bu klasöre
   sürükle. **Zip'i açma.** Defter onu kendisi `yz/3-1-dermamnist-dinov2/` altına açar.
3. **Defteri aç.** Drive'da `k3_baseline.ipynb`'ye sağ tıkla → *Birlikte aç* → *Google Colaboratory*.
4. **GPU'yu L4 yap.** Menüden *Çalışma zamanı* → *Çalışma zamanı türünü değiştir* → *Donanım hızlandırıcı*: **L4 GPU** → *Kaydet*.
   Başka bir GPU (T4, A100) seçme. Defter L4 görmezse durur. Karşılaştırmadaki tüm koşumlar aynı GPU sınıfında olmalı.
5. **Hepsini çalıştır.** *Çalışma zamanı* → *Tümünü çalıştır*. İlk hücre Drive erişim izni ister. Aynı Google hesabıyla izin ver.
   Kurulum ve veri birkaç dakika sürer, ardından kuyruk başlar.
6. **Bekle, sekmeyi kapatma.** Colab Pro arka planda çalışmaz. Sekme açık kalsın, bilgisayar uyku moduna geçmesin.
   Hücre 5'te `[kuyruk] ✔ AŞAMA 1 … geçti` ve `AŞAMA 2 … geçti` satırlarını, sonra 5 tam koşumu görürsün.
   `✖ … GEÇEMEDİ` görürsen panik yok. Defter tam koşumu bilerek başlatmamıştır, 7. adıma geç.
   **Oturum koparsa:** defteri yeniden aç, L4'ü kontrol et, yine *Tümünü çalıştır*. Hücre 5b bitmiş koşumları atlar,
   yalnız kalanları koşar.
7. **Sonucu indir.** Son hücre `yz/k3_baseline_sonuc.zip` dosyasını Drive'a yazar ve tarayıcıdan indirmeyi dener.
   İnmezse Drive → *Bilgisayarım* → `yz` → `k3_baseline_sonuc.zip` → *İndir*.
8. **Yerine koy.** Zip'i **açmadan** bilgisayarında
   `deney_workspace\3-1-dermamnist-dinov2\colab\gelen\` klasörüne koy (klasör yoksa oluştur). Sonra şefe "geldi" de.

Sorun olursa son hücrenin çıktısının ekran görüntüsü yeterli. `runs/` klasörü Drive'da durur, hiçbir şey kaybolmaz.
