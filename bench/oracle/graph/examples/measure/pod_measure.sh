#!/usr/bin/env bash
# Every measurement behind the example programs, on one rented GPU (#2951; rp-run, RP_PYENV=oracle):
# vpd4l neuron tables, transcoder-feature tables for IOI and greater-than, every example rebuilt from its
# English and paraphrased by Qwen3-8B (vLLM), then Qwen3-0.6B patch tables (the longest step, last).
# Outputs in OUT.
#   RP_OWNER=g-mech RP_PYENV=oracle RP_HF_MODELS="Qwen/Qwen3-0.6B Qwen/Qwen3-8B" RP_MAX_PRICE=0.70 \
#   rp-run oracle-train-g-mech-measure auto 1.5 -- bash /Users/user/gam/bench/oracle/graph/examples/measure/pod_measure.sh \
#       /Users/user/mpd-data/runpod/oracle-train-g-mech-measure /Users/user/mpd-data/graph_oracle/behaviors \
#       QWEN_LIST VPD_LIST /Users/user/mpd-data/vpd/t-9d2b8f02 /Users/user/gam/bench/oracle /Users/user/gam/bench/vpd_2951
# QWEN_LIST: lines "qwen3-0.6b/ID.json"; VPD_LIST: lines "vpd4l/ID.json:LAYER" (relative to the behaviors).
# The two source directories are named only so that rp-run ships them to the pod.
set -uo pipefail  # no -e: a failed step leaves the others running
here=$(cd "$(dirname "$0")" && pwd)
OUT=$1 BEH=$2 QWEN=$3 VPD=$4 WEIGHTS=$5
mkdir -p "$OUT"
DATA=$(cd "$WEIGHTS/../.." && pwd)  # the run's mpd-data: vpd_model reads ~/mpd-data/vpd/...
HOME=$(dirname "$DATA") python3 "$here/vpd_neurons.py" "$OUT/neurons_vpd4l" $(sed "s|^|$BEH/|" "$VPD") 2>&1 | tee "$OUT/vpd_neurons.log"
python3 "$here/qwen_tc_patch.py" "$BEH/qwen3-0.6b/ioi.argument.json" 16,18,19,22 "$OUT/tc_ioi.argument.json" 2>&1 | tee "$OUT/tc.log"
python3 "$here/qwen_tc_patch.py" "$BEH/qwen3-0.6b/greater_than.war.json" 16,18,20,21,22,26 "$OUT/tc_greater_than.war.json" 2>&1 | tee -a "$OUT/tc.log"
G=$(cd "$here/../.." && pwd)
for target in vpd4l qwen3-0.6b; do
    progs=$(python3 -c "import json; idx = json.load(open('$G/examples/index.json')); print(' '.join('$G/examples/' + n + '.py' for n, e in idx.items() if e['model'] == '$target'))")
    python3 "$G/rebuild.py" $progs --model $target --coder Qwen/Qwen3-8B --backend vllm --out "$OUT/rebuild_$target.jsonl" 2>&1 | tee -a "$OUT/text.log"
    python3 "$G/paraphrase.py" $progs --model $target --writer Qwen/Qwen3-8B --backend vllm --out "$OUT/paraphrased_$target" 2>&1 | tee -a "$OUT/text.log"
done
python3 "$here/qwen_patch.py" "$OUT/patch_qwen3" $(sed "s|^|$BEH/|" "$QWEN") 2>&1 | tee "$OUT/qwen_patch.log"
