#!/usr/bin/env bash
# tur5_kos.sh — H13 (Tur-5) CPU koşumları, foreground kosum.py (her satır < 5 s). Kart: hipotezler/H13.md.
# Kullanım (WS kökünden): bash varyantlar/topluluk/tur5_kos.sh a1|a3|katman|ikincil
# Üyeler: yerel arşiv kopyası runs_colab/ (kaynak <DRIVE_KOK>/3-1/runs; SHA256 runs_colab/ARSIV_MANIFEST.json).
set -euo pipefail
WS="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$WS"
PY="$WS/env/.venv/Scripts/python.exe"
UVPY="${PYTHON311:-python}"
SK="$WS/tools"
export PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=
RC=runs_colab
H10=(20261011-160211_h10_dinov2b_k4_ft224_s0 20261011-161411_h10_dinov2b_k4_ft224_s1 20261011-162611_h10_dinov2b_k4_ft224_s2 20261011-163811_h10_dinov2b_k4_ft224_s3 20261011-165010_h10_dinov2b_k4_ft224_s4)
H02=(20261011-130855_h02_r18in_ft224_colab_s0 20261011-131620_h02_r18in_ft224_colab_s1 20261011-132346_h02_r18in_ft224_colab_s2 20261011-133112_h02_r18in_ft224_colab_s3 20261011-133837_h02_r18in_ft224_colab_s4)
H01=(20261011-125349_h01_dinov2b_lp_r224_colab_s0 20261011-125744_h01_dinov2b_lp_r224_colab_s1 20261011-130031_h01_dinov2b_lp_r224_colab_s2 20261011-130320_h01_dinov2b_lp_r224_colab_s3 20261011-130608_h01_dinov2b_lp_r224_colab_s4)
H08=(20261011-134604_h08_r18in_ft224_d4_colab_s0 20261011-135335_h08_r18in_ft224_d4_colab_s1 20261011-140107_h08_r18in_ft224_d4_colab_s2 20261011-140837_h08_r18in_ft224_d4_colab_s3 20261011-141608_h08_r18in_ft224_d4_colab_s4)
BAS=(20261011-142338_baseline_colab_s0 20261011-143613_baseline_colab_s1 20261011-144846_baseline_colab_s2 20261011-150122_baseline_colab_s3 20261011-151359_baseline_colab_s4)
TOP="$PY varyantlar/topluluk/topluluk.py"

kos() {  # kos <kuyruk> <asama> <varyant> <seed> <hipotez> <ebeveyn> <etiket> -- komut...
  local q=$1 a=$2 v=$3 s=$4 h=$5 e=$6 et=$7; shift 8
  "$UVPY" "$SK/kosum.py" --kok . --varyant "$v" --seed "$s" --hipotez "$h" --ebeveyn "$e" --kuyruk "$q" --asama "$a" \
    --zaman-siniri-dk 5 --etiket "$et" -- "$@"
}

case "${1:-}" in
a1)
  kos tur5-h13 1 h13_h02c_h10_lpo_duman 0 H13 "${H10[0]}" "H13 duman: H10_s0 ⊕ H02c_s0 log-olasılık ½;½ + tek üye özdeşliği" -- \
    $TOP --uyeler "$RC/${H10[0]}" "$RC/${H02[0]}" --agirlik 0.5,0.5 --birlesim logolasilik --kirp 1e-12 --ozdeslik \
    --seed-denet hepsi --seed 0 --out "{RUN_DIR}/metrics.json"
  ;;
a3)
  for s in 0 1 2 3 4; do
    kos tur5-h13 3 h13_h02c_h10_lpo $s H13 "${H10[$s]}" "H13: H10_s ⊕ H02c_s log-olasılık ½;½ (seed-eşleşik)" -- \
      $TOP --uyeler "$RC/${H10[$s]}" "$RC/${H02[$s]}" --agirlik 0.5,0.5 --birlesim logolasilik --kirp 1e-12 --ozdeslik \
      --seed-denet hepsi --seed $s --out "{RUN_DIR}/metrics.json"
  done
  ;;
katman)
  for s in 0 1 2 3 4; do
    kos tur5-h13k 3 h13k_h13_logitayar $s H13 "${H10[$s]}" "H13 ikinci satır: H04 logit ayarı (katlar.json, τ 0:1:0,1, kısıt 0,010)" -- \
      $PY varyantlar/logit_ayari/logit_ayari.py --ebeveyn-varyant h13_h02c_h10_lpo --ebeveyn-dosya tahmin_val.npy \
      --tau-izgara 0:1:0.1 --kat 5 --kat-rng 20261011 --kisit 0.010 --seed $s --out "{RUN_DIR}/metrics.json"
  done
  for s in 0 1 2 3 4; do
    kos tur5-h13rt 3 rt_h13_sicaklik $s H13 "${H10[$s]}" "H13 üçüncü satır: R-T sıcaklık (OOF ECE)" -- \
      $PY varyantlar/logit_ayari/sicaklik.py --kaynak-varyant h13_h02c_h10_lpo --kaynak-dosya tahmin_val.npy \
      --seed $s --out "{RUN_DIR}/metrics.json"
  done
  ;;
ikincil)
  q=tur5-h13x
  for s in 0 1 2 3 4; do
    kos $q 3 h13x_h02c_h10_op $s H13 "${H10[$s]}" "H13x (7): H10_s ⊕ H02c_s olasılık ortalaması" -- \
      $TOP --uyeler "$RC/${H10[$s]}" "$RC/${H02[$s]}" --birlesim olasilik --seed-denet hepsi --seed $s --out "{RUN_DIR}/metrics.json"
    kos $q 3 h13x_t2_h01c $s H13 "${H10[$s]}" "H13x T2: H10_s ⊕ H02c_s ⊕ H01c_s log-olasılık ⅓" -- \
      $TOP --uyeler "$RC/${H10[$s]}" "$RC/${H02[$s]}" "$RC/${H01[$s]}" --birlesim logolasilik --seed-denet hepsi --seed $s --out "{RUN_DIR}/metrics.json"
    kos $q 3 h13x_t4_h08c_h10_lpo $s H13 "${H10[$s]}" "H13x T4 (koşullu): H10_s ⊕ H08c_s log-olasılık ½;½" -- \
      $TOP --uyeler "$RC/${H10[$s]}" "$RC/${H08[$s]}" --birlesim logolasilik --seed-denet hepsi --seed $s --out "{RUN_DIR}/metrics.json"
    kos $q 3 h13x_m3_baseline_h02c_lpo $s H13 "${H02[$s]}" "H13x M3: baseline_c_s ⊕ H02c_s log-olasılık ½;½ (rakibe topluluk hakkı)" -- \
      $TOP --uyeler "$RC/${H02[$s]}" "$RC/${BAS[$s]}" --birlesim logolasilik --seed-denet hepsi --seed $s --out "{RUN_DIR}/metrics.json"
    s1=$(( (s + 1) % 5 ))
    kos $q 3 h13x_abl_h10_hom2 $s H13 "${H10[$s]}" "H13x ablasyon (i): H10_s ⊕ H10_(s+1 mod 5) log-olasılık ½;½ (çiftler örtüşür)" -- \
      $TOP --uyeler "$RC/${H10[$s]}" "$RC/${H10[$s1]}" --birlesim logolasilik --seed-denet ilk --seed $s --out "{RUN_DIR}/metrics.json"
  done
  kos $q 3 h13x_t3_h10_hom5 0-4 H13 "h10_dinov2b_k4_ft224:s0-4" "H13x T3: homojen 5-seed H10 log-olasılık (n = 1)" -- \
    $TOP --uyeler "$RC/${H10[0]}" "$RC/${H10[1]}" "$RC/${H10[2]}" "$RC/${H10[3]}" "$RC/${H10[4]}" --birlesim logolasilik --seed-denet yok --seed 0-4 --out "{RUN_DIR}/metrics.json"
  kos $q 3 h13x_t3_h02c_hom5 0-4 H13 "h02_r18in_ft224_colab:s0-4" "H13x T3: homojen 5-seed H02c log-olasılık (n = 1)" -- \
    $TOP --uyeler "$RC/${H02[0]}" "$RC/${H02[1]}" "$RC/${H02[2]}" "$RC/${H02[3]}" "$RC/${H02[4]}" --birlesim logolasilik --seed-denet yok --seed 0-4 --out "{RUN_DIR}/metrics.json"
  kos $q 3 h13x_m3_baseline_hom5 0-4 H13 "baseline_colab:s0-4" "H13x M3: homojen 5-seed baseline_c log-olasılık (n = 1)" -- \
    $TOP --uyeler "$RC/${BAS[0]}" "$RC/${BAS[1]}" "$RC/${BAS[2]}" "$RC/${BAS[3]}" "$RC/${BAS[4]}" --birlesim logolasilik --seed-denet yok --seed 0-4 --out "{RUN_DIR}/metrics.json"
  ;;
*) echo "kullanım: $0 a1|a3|katman|ikincil"; exit 2 ;;
esac
