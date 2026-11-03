# Beklenti Kapısı — kuyruk `tur3-h07rt` aşama 3 → **GEÇTİ**
*2026-10-11 04:29* · metrik `val_ece` (min) · beklenti: `tur3-h07rt.beklenti.json` · 5 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| rt_h07_sicaklik | 5 | 0.02435 | 0.0234, 0.02556, 0.02494, 0.02124, 0.02663 | 4.9 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 5 başarılı koşum (en az 5) |
| 🟢 OK | aralik | rt_h07_sicaklik: ort 0.02435 ∈ [0.01, 0.0845] (R-T görülmemiş < 0,010; üst = H02 ham ECE 0,0745 + 0,010 (rt_h04 emsali)) |
| 🟢 OK | ikincil_aralik | rt_h07_sicaklik/s0 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h07_sicaklik/s1 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h07_sicaklik/s2 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h07_sicaklik/s3 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h07_sicaklik/s4 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h07_sicaklik/s0 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h07_sicaklik/s1 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h07_sicaklik/s2 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h07_sicaklik/s3 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h07_sicaklik/s4 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h07_sicaklik/s0 T_tamval=1.36 ∈ [0.51,2.99] |
| 🟢 OK | ikincil_aralik | rt_h07_sicaklik/s1 T_tamval=1.21 ∈ [0.51,2.99] |
| 🟢 OK | ikincil_aralik | rt_h07_sicaklik/s2 T_tamval=1.4 ∈ [0.51,2.99] |
| 🟢 OK | ikincil_aralik | rt_h07_sicaklik/s3 T_tamval=1.16 ∈ [0.51,2.99] |
| 🟢 OK | ikincil_aralik | rt_h07_sicaklik/s4 T_tamval=1.33 ∈ [0.51,2.99] |
