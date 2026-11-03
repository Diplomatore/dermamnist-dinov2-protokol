#!/usr/bin/env bash
# Kullanıcı kararı (2026-10-11 ~23:10): C2 atlanır; C1 (k5-c1-A-v3 + k5-c1-B) bitince C1 analizi bir kez koşulur,
# ardından bilgisayar 300 sn sonra kapatılır (iptal: komut isteminde `shutdown /a`).
WS="${WS:-$(cd "$(dirname "$0")/.." && pwd)}"
UVPY="${PYTHON311:-python}"
cd "$WS" || exit 1
export PYTHONIOENCODING=utf-8
echo "[$(date +%H:%M:%S)] C1 bekleniyor"
"$UVPY" scripts/colab_izle.py --kuyruklar k5-c1-A-v3,k5-c1-B --aralik 60 --asili-dk 20 > sonuclar/c1_izle_son.log 2>&1
rc=$?; echo "[$(date +%H:%M:%S)] izleyici bitti rc=$rc"; tail -3 sonuclar/c1_izle_son.log
env/.venv/Scripts/python.exe -I scripts/k5_test.py --kip c1 --runs "${DRIVE_KOK:?DRIVE_KOK tanimlayin}/3-1/runs" --test-kilidi-acik > sonuclar/k5_c1_analiz.log 2>&1
echo "[$(date +%H:%M:%S)] C1 analizi rc=$?"; tail -5 sonuclar/k5_c1_analiz.log
[ -f sonuclar/k5_c1_sonuc.json ] && sha256sum sonuclar/k5_c1_sonuc.json > sonuclar/k5_c1_sonuc.json.sha256
echo "[$(date +%H:%M:%S)] kapatma komutu (300 sn)"
MSYS_NO_PATHCONV=1 shutdown /s /t 300 /c "3.1 deneyleri: C1 bitti, 5 dk sonra kapanıyor (iptal: shutdown /a)"
echo "[$(date +%H:%M:%S)] kapatma zamanlandı rc=$?"
