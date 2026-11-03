# Beklenti Kapısı — kuyruk `tur3-h09rt` aşama 3 → **GEÇTİ**
*2026-10-11 09:14* · metrik `val_ece` (min) · beklenti: `tur3-h09rt.beklenti.json` · 5 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| rt_h09_sicaklik | 5 | 0.02588 | 0.03216, 0.01549, 0.02718, 0.02541, 0.02915 | 5.0 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 5 başarılı koşum (en az 5) |
| 🟢 OK | aralik | rt_h09_sicaklik: ort 0.02588 ∈ [0.01, 0.0845] (R-T görülmemiş < 0,010; üst = H02 ham ECE 0,0745 + 0,010 (rt_h04 emsali)) |
| 🟢 OK | ikincil_aralik | rt_h09_sicaklik/s0 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h09_sicaklik/s1 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h09_sicaklik/s2 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h09_sicaklik/s3 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h09_sicaklik/s4 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h09_sicaklik/s0 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h09_sicaklik/s1 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h09_sicaklik/s2 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h09_sicaklik/s3 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h09_sicaklik/s4 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h09_sicaklik/s0 T_tamval=1.72 ∈ [0.51,2.99] |
| 🟢 OK | ikincil_aralik | rt_h09_sicaklik/s1 T_tamval=1.66 ∈ [0.51,2.99] |
| 🟢 OK | ikincil_aralik | rt_h09_sicaklik/s2 T_tamval=1.82 ∈ [0.51,2.99] |
| 🟢 OK | ikincil_aralik | rt_h09_sicaklik/s3 T_tamval=1.68 ∈ [0.51,2.99] |
| 🟢 OK | ikincil_aralik | rt_h09_sicaklik/s4 T_tamval=1.78 ∈ [0.51,2.99] |
