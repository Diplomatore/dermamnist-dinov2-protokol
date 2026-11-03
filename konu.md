# Konu Kartı — 3.1 · Düşük Çözünürlükte Transfer Öğrenme: DermaMNIST’te ResNet-18’in DINOv2 ile Aşılması

*Kaynak: konu-katalogu.json (yz5), oluşturulma: 2026-10-10*

| Alan | Değer |
|---|---|
| Kategori | 03 — Tıbbi Görüntüleme |
| Konu kodu / id | 3.1 / med-1 |
| İngilizce başlık | Beating the MedMNIST Baseline with Self-Supervised Features |
| Zorluk | 1 · orkestra uyumu: Çok yüksek |
| Süre tahmini | 2–3 hafta |
| Yayın hedefi | SCI dergi (medikal) / Workshop |

## Veri seti
- **Ad:** DermaMNIST
- **Detay:** 7 sınıf cilt lezyonu · pip install medmnist ile iner · CC lisans
- **URL:** https://medmnist.com/

## Rakip (otoriter yöntem / baseline)
- **Ad:** ResNet-18 (28px), sıfırdan
- **Kod / protokol:** MedMNIST resmi benchmark kodu (MedMNIST/MedMNIST)
- **Literatürdeki sonuç (replikasyon çıpası):** ACC 0,735 / AUC 0,917 (MedMNIST v2, Tablo 3)

## Hedef ve metrik
- **Hedef:** ACC ≥ 0,80 (DINOv2 özellikleriyle %92,02 raporlanmış)
- **Ana metrik:** ACC + AUC

## Katalogdaki iyileştirme stratejisi (manipülasyon adayları — ön-kayıt için ham liste)
1. Ajan veri setini indirir, resmi ResNet-18 baseline’ını koşar
2. 224px ölçekleme + ImageNet önağırlıkları
3. DINOv2 öznitelik çıkarımı + doğrusal sonda varyantı
4. Sınıf dengesizliği için focal loss
5. Augmentation araması + TTA

## Donanım / sistem
- **GPU:** Colab T4 yeterli (~30–60 dk/koşu)
- **Sistem:** Ubuntu 22.04 · CUDA 12.1 · PyTorch 2.3+ · medmnist/timm

## Kullanıcı notları / kısıtlar (elle doldur)
- Hesaplama hedefi (yerel / ssh / colab): **D — Colab Pro, tüm K3/K4/K5 koşumları** (kullanıcı kararı 2026-10-10).
  Sabit donanım sınıfı: **NVIDIA L4 (24 GB)** — karşılaştırılan TÜM koşumlar (rakip dahil) bu sınıfta; L4 yoksa koşum
  ertelenir, başka GPU'da koşulan sonuç leaderboard'a karıştırılmaz (ayrı `donanim` etiketi). Yerel GTX 1650 yalnız
  duman testleri (kod doğruluğu) için. Gerekçe: yerel ölçüm (`env/hiz-olcumu.md`) — 224 px R18 214 img/s, DINOv2-B
  tam ince ayar 7 img/s (5 seed ≈ 28 sa, bütçe dışı); FP16 bu kartta 3–4× yavaş.
- Zaman bütçesi (saat) ve tur sayısı üst sınırı: **K4 ≤ 8 tur** (kolay konu), tur başına ≤3 kart;
  toplam GPU bütçesi **≤ 24 L4-saat** (K3 dahil; Colab işlem birimi tüketimi günlükte izlenir); seed: **5** (MedMNIST küçük veri → ≥5; seed kümesi {0,1,2,3,4}).
  **Durma kuralı:** ardışık 2 turda en iyi val-ACC artışı < 0,5 puan → K5; ya da bütçe bitişi.
  Ana seçim metriği: **val ACC** (eşitlikte val AUC); test yalnız K5'te bir kez.
- Kapsam dışı bırakılanlar: klinik/tanısal iddia (araştırma benchmark'ı; HAM10000 türevi); harici veri
  (ISIC vb.) ile ön-eğitim/ekleme; DermaMNIST dışındaki MedMNIST alt kümeleri (yalnız tartışmada anılabilir);
  DINOv2'nin kendisini yeniden ön-eğitmek. MedMNIST+ (64/128/224 px resmî sürümler) çözünürlük varyantı
  olarak KAPSAM İÇİ (aynı split).

## GÜNCELLEME 2026-10-10 ~09:25 (kullanıcı kararı)
Hesaplama hedefi **A — bu makine (GTX 1650, FP32)**; Colab yalnız tıkanınca (eşleştirilmiş, rakiple aynı defterde).
Bütçe: **40 sa duvar saati, ≤ 32 GPU-sa, ≤ 8 tur**, 5 seed; durma kuralı aynen. Plan: `notlar/plan-40-saat.md`.
Yukarıdaki "D — Colab Pro / L4" satırı geçersizdir (tarihçe için korunur).

## GÜNCELLEME 2026-10-10 20:15 (kullanıcı kararı) — durma kuralı
**Durma kuralı (yeni):** ardışık 2 turda NE en iyi val ACC NE en iyi val dengeli ACC ≥ 0,5 puan iyileşmezse → K5
(ya da bütçe bitişi). Gerekçe: Tur-2 (dengesizlik) ön-kayıtlı olarak ACC-nötr; tek ölçütlü kural erken kapanış yapardı.
