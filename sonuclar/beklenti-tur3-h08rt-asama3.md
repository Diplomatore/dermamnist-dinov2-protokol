# Beklenti Kapısı — kuyruk `tur3-h08rt` aşama 3 → **GEÇTİ**
*2026-10-11 09:14* · metrik `val_ece` (min) · beklenti: `tur3-h08rt.beklenti.json` · 5 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| rt_h08_sicaklik | 5 | 0.02124 | 0.02372, 0.02089, 0.02815, 0.01629, 0.01716 | 5.0 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 5 başarılı koşum (en az 5) |
| 🟢 OK | aralik | rt_h08_sicaklik: ort 0.02124 ∈ [0.01, 0.0845] (R-T görülmemiş < 0,010; üst = H02 ham ECE 0,0745 + 0,010 (rt_h04 emsali)) |
| 🟢 OK | ikincil_aralik | rt_h08_sicaklik/s0 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h08_sicaklik/s1 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h08_sicaklik/s2 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h08_sicaklik/s3 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h08_sicaklik/s4 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h08_sicaklik/s0 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h08_sicaklik/s1 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h08_sicaklik/s2 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h08_sicaklik/s3 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h08_sicaklik/s4 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h08_sicaklik/s0 T_tamval=1.69 ∈ [0.51,2.99] |
| 🟢 OK | ikincil_aralik | rt_h08_sicaklik/s1 T_tamval=1.57 ∈ [0.51,2.99] |
| 🟢 OK | ikincil_aralik | rt_h08_sicaklik/s2 T_tamval=1.67 ∈ [0.51,2.99] |
| 🟢 OK | ikincil_aralik | rt_h08_sicaklik/s3 T_tamval=1.7 ∈ [0.51,2.99] |
| 🟢 OK | ikincil_aralik | rt_h08_sicaklik/s4 T_tamval=1.71 ∈ [0.51,2.99] |
