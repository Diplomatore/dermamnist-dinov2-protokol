# Beklenti Kapısı — kuyruk `tur2-rt` aşama 3 → **GEÇTİ**
*2026-10-11 01:20* · metrik `val_ece` (min) · beklenti: `tur2-rt.beklenti.json` · 20 koşum

| varyant | n | ort | seed değerleri | ort süre (s) |
|---|---|---|---|---|
| rt_h02_sicaklik | 5 | 0.03096 | 0.03244, 0.02673, 0.04399, 0.03236, 0.01928 | 5.2 |
| rt_h01_sicaklik | 5 | 0.02014 | 0.01865, 0.02009, 0.01896, 0.02188, 0.02114 | 4.9 |
| rt_baseline_sicaklik | 5 | 0.03251 | 0.03697, 0.03186, 0.03013, 0.03647, 0.02713 | 4.9 |
| rt_h04_sicaklik | 5 | 0.02854 | 0.02303, 0.03117, 0.02869, 0.03148, 0.02833 | 5.0 |

## Kurallar
| seviye | kural | bulgu |
|---|---|---|
| 🟢 OK | sonlu | tüm değerler sonlu |
| 🟢 OK | min_kosum | 20 başarılı koşum (en az 20) |
| 🟢 OK | aralik | rt_h02_sicaklik: ort 0.03096 ∈ [0.01, 0.0745] (SIRALAMA R-T: görülmemiş < 0,010; üst = H02 ham ECE 0,0745 (tahlil §1b). Ön-kayıtlı beklenti [0,025; 0,045] (Guo 2017 göreli düşüş %66–92)) |
| 🟢 OK | aralik | rt_h01_sicaklik: ort 0.02014 ∈ [0.01, 0.0409] (H01 ham ECE 0,0309 (tahlil §1b) + 0,010; alt R-T görülmemiş) |
| 🟢 OK | aralik | rt_baseline_sicaklik: ort 0.03251 ∈ [0.01, 0.0465] (B ham ECE 0,0365 + 0,010; alt R-T görülmemiş) |
| 🟢 OK | aralik | rt_h04_sicaklik: ort 0.02854 ∈ [0.01, 0.0845] (H04 = H02 + önsel kaydırma; üst H02 ham 0,0745 + 0,010; alt R-T görülmemiş) |
| 🟢 OK | kollar_farkli | kollar birbirinden farklı |
| 🟢 OK | ikincil_aralik | rt_h02_sicaklik/s0 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h02_sicaklik/s1 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h02_sicaklik/s2 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h02_sicaklik/s3 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h02_sicaklik/s4 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h01_sicaklik/s0 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h01_sicaklik/s1 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h01_sicaklik/s2 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h01_sicaklik/s3 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h01_sicaklik/s4 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_baseline_sicaklik/s0 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_baseline_sicaklik/s1 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_baseline_sicaklik/s2 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_baseline_sicaklik/s3 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_baseline_sicaklik/s4 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h04_sicaklik/s0 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h04_sicaklik/s1 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h04_sicaklik/s2 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h04_sicaklik/s3 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h04_sicaklik/s4 acc_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h02_sicaklik/s0 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h02_sicaklik/s1 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h02_sicaklik/s2 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h02_sicaklik/s3 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h02_sicaklik/s4 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h01_sicaklik/s0 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h01_sicaklik/s1 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h01_sicaklik/s2 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h01_sicaklik/s3 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h01_sicaklik/s4 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_baseline_sicaklik/s0 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_baseline_sicaklik/s1 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_baseline_sicaklik/s2 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_baseline_sicaklik/s3 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_baseline_sicaklik/s4 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h04_sicaklik/s0 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h04_sicaklik/s1 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h04_sicaklik/s2 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h04_sicaklik/s3 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h04_sicaklik/s4 dengeli_degisim=0 ∈ [0.0,0.0] |
| 🟢 OK | ikincil_aralik | rt_h02_sicaklik/s0 T_tamval=1.82 ∈ [0.51,2.99] |
| 🟢 OK | ikincil_aralik | rt_h02_sicaklik/s1 T_tamval=1.5 ∈ [0.51,2.99] |
| 🟢 OK | ikincil_aralik | rt_h02_sicaklik/s2 T_tamval=1.86 ∈ [0.51,2.99] |
| 🟢 OK | ikincil_aralik | rt_h02_sicaklik/s3 T_tamval=1.48 ∈ [0.51,2.99] |
| 🟢 OK | ikincil_aralik | rt_h02_sicaklik/s4 T_tamval=1.73 ∈ [0.51,2.99] |
| 🟢 OK | ikincil_aralik | rt_h01_sicaklik/s0 T_tamval=1.19 ∈ [0.51,2.99] |
| 🟢 OK | ikincil_aralik | rt_h01_sicaklik/s1 T_tamval=1.19 ∈ [0.51,2.99] |
| 🟢 OK | ikincil_aralik | rt_h01_sicaklik/s2 T_tamval=1.19 ∈ [0.51,2.99] |
| 🟢 OK | ikincil_aralik | rt_h01_sicaklik/s3 T_tamval=1.19 ∈ [0.51,2.99] |
| 🟢 OK | ikincil_aralik | rt_h01_sicaklik/s4 T_tamval=1.19 ∈ [0.51,2.99] |
| 🟢 OK | ikincil_aralik | rt_baseline_sicaklik/s0 T_tamval=1.19 ∈ [0.51,2.99] |
| 🟢 OK | ikincil_aralik | rt_baseline_sicaklik/s1 T_tamval=1.06 ∈ [0.51,2.99] |
| 🟢 OK | ikincil_aralik | rt_baseline_sicaklik/s2 T_tamval=1.06 ∈ [0.51,2.99] |
| 🟢 OK | ikincil_aralik | rt_baseline_sicaklik/s3 T_tamval=1.16 ∈ [0.51,2.99] |
| 🟢 OK | ikincil_aralik | rt_baseline_sicaklik/s4 T_tamval=1.08 ∈ [0.51,2.99] |
| 🟢 OK | ikincil_aralik | rt_h04_sicaklik/s0 T_tamval=1.57 ∈ [0.51,2.99] |
| 🟢 OK | ikincil_aralik | rt_h04_sicaklik/s1 T_tamval=1.26 ∈ [0.51,2.99] |
| 🟢 OK | ikincil_aralik | rt_h04_sicaklik/s2 T_tamval=1.58 ∈ [0.51,2.99] |
| 🟢 OK | ikincil_aralik | rt_h04_sicaklik/s3 T_tamval=1.3 ∈ [0.51,2.99] |
| 🟢 OK | ikincil_aralik | rt_h04_sicaklik/s4 T_tamval=1.47 ∈ [0.51,2.99] |
