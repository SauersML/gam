#!/usr/bin/env bash
# Every measurement behind the example programs, on one rented GPU (#2951; rp-run, RP_PYENV=oracle):
# Qwen3-0.6B patch tables, vpd4l neuron tables, transcoder-feature tables for IOI and greater-than,
# then every example rebuilt from its English and paraphrased by Qwen3-8B (vLLM). Outputs in OUT.
#   RP_OWNER=g-mech RP_PYENV=oracle RP_HF_MODELS="Qwen/Qwen3-0.6B Qwen/Qwen3-8B" RP_MAX_PRICE=0.70 \
#   rp-run oracle-train-g-mech-measure auto 1.5 -- bash /Users/user/gam/bench/oracle/graph/examples/measure/pod_measure.sh \
#       /Users/user/mpd-data/runpod/oracle-train-g-mech-measure /Users/user/mpd-data/graph_oracle/behaviors \
#       QWEN_LIST VPD_LIST /Users/user/mpd-data/vpd/t-9d2b8f02
# QWEN_LIST: lines "qwen3-0.6b/ID.json"; VPD_LIST: lines "vpd4l/ID.json:LAYER" (relative to the behaviors).
set -Eeuo pipefail
here=$(cd "$(dirname "$0")" && pwd)
OUT=$1 BEH=$2 QWEN=$3 VPD=$4 WEIGHTS=$5
mkdir -p "$OUT"
DATA=$(cd "$WEIGHTS/../.." && pwd)  # the run's mpd-data: vpd_model reads ~/mpd-data/vpd/...
python3 "$here/qwen_patch.py" "$OUT/patch_qwen3" $(sed "s|^|$BEH/|" "$QWEN") 2>&1 | tee "$OUT/qwen_patch.log"
HOME=$(dirname "$DATA") python3 "$here/vpd_neurons.py" "$OUT/neurons_vpd4l" $(sed "s|^|$BEH/|" "$VPD") 2>&1 | tee "$OUT/vpd_neurons.log"
python3 "$here/qwen_tc_patch.py" "$BEH/qwen3-0.6b/ioi.argument.json" 16,18,19,22 "$OUT/tc_ioi.argument.json" 2>&1 | tee "$OUT/tc.log"
python3 "$here/qwen_tc_patch.py" "$BEH/qwen3-0.6b/greater_than.war.json" 14,16,18,20,22 "$OUT/tc_greater_than.war.json" 2>&1 | tee -a "$OUT/tc.log"
bash "$here/../../mats_text.sh" Qwen/Qwen3-8B "$OUT/text" 2>&1 | tee "$OUT/text.log"
