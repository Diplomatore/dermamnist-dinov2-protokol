# H15 / E2 — a0 veri kapısı raporu
*2026-11-07T21:50:44* · `scripts/e2_bcn_hazirla.py` · beklenen `e2_bcn_beklenen.json` SHA256 `d81fccdc6f679803…`

- **K-a (sızıntı):** RMSE ≤ 5 çifti **0** · ≤ 10: 20 (bilgi) · koyu köşe oranı (28 px, <40) BCN 0.4198 / DermaMNIST 0.0004
- **K-b:** CSV MD5 eşleşti · E1 kopyasız **11933 görüntü / 3451 lezyon** · çıkarma 49 · lezyon birleştirme 2 çift · eşleme birim sınaması geçti
- **Çerçeve (S9):** yazar kırpıcısı 2210/12413 görüntüyü kırptı (oran 0.178); nedenler {'oran_dusuk': 10203, 'kirp': 2210}; boyutlar {'1024x1024': 12413}; yön uyuşmazlığı 0; cv2 4.12.0
- **Kalibrasyon girdisi:** 8 HAM görüntü RMSE ort 1.759 (kalibrasyonda 1,76)

| dosya | bayt | SHA256 |
|---|---|---|
| `bcn_e2_merkez_kare_224.npz` | 1,196,527,773 | `4382880cc0715a0134d2a232111880a71bc3d7c22ad9533a7153197e3538af38` |
| `bcn_e2_cerceve_fark_224.npz` | 229,980,209 | `2fbcb1ed5d5f5de3a6ce4e8c92f6fa61387af9338fd7847e50f1fa8dd93af3ca` |
| `bcn_e2_kal_224.npz` | 809,495 | `05ba97dbd836caf227a7ed9771d8b22e802c484269f5fc852b5bd47f21af7e9a` |

İçerik SHA256: merkez_kare `7e7a93d00af1f306…`, cerceve `c285946ed85bc9d4…`, sira `8dd121bc9a6ceb5f…`, kal_images `61d8828b9e5f6fba…`

Sonraki adım (şef): `colab/hat/e2_veri_kopya.py` ile üç npz'yi Drive `yz/3-1/veri_dis/bcn20000/e2_girdi/`'ye kopyala (geri okuyup SHA256 doğrular), sonra `colab/hat/drive_paketle.py`. Manifestin (data/dis_veri_manifest.md) tam veri satırları bu rapordan güncellenir.
