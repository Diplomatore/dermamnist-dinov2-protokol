# Beklenti Kapısı — kuyruk `tur5-h13rt` aşama 3 → **GEÇTİ**
*2026-10-11 18:06* · metrik `val_ece` (min) · beklenti: `tur5-h13rt.beklenti.json` · 5 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| rt_h13_sicaklik | 5 | 0.02138 | 0.01738, 0.0229, 0.03009, 0.02112, 0.01544 | 5.1 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 5 başarılı koşum (en az 5) |
| 🟢 OK | aralik | rt_h13_sicaklik: ort 0.02138 ∈ [0.01, 0.07] (R-T görülmemiş < 0,010; üst = H10 ham ECE 0,060 + 0,010) |
| 🟢 OK | ikincil_aralik | rt_h13_sicaklik/s0 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h13_sicaklik/s1 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h13_sicaklik/s2 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h13_sicaklik/s3 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h13_sicaklik/s4 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h13_sicaklik/s0 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h13_sicaklik/s1 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h13_sicaklik/s2 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h13_sicaklik/s3 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h13_sicaklik/s4 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h13_sicaklik/s0 T_tamval=1.57 ∈ [0.51,2.99] |
| 🟢 OK | ikincil_aralik | rt_h13_sicaklik/s1 T_tamval=1.65 ∈ [0.51,2.99] |
| 🟢 OK | ikincil_aralik | rt_h13_sicaklik/s2 T_tamval=1.67 ∈ [0.51,2.99] |
| 🟢 OK | ikincil_aralik | rt_h13_sicaklik/s3 T_tamval=1.35 ∈ [0.51,2.99] |
| 🟢 OK | ikincil_aralik | rt_h13_sicaklik/s4 T_tamval=1.67 ∈ [0.51,2.99] |
