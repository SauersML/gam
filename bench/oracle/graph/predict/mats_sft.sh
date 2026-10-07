#!/usr/bin/env bash
# The oracle's prediction SFT and its held-out evaluation on one MATS L40 (#2951):
#   MATS_GPUS=1 mats-run predict-sft-q06 8 40 6 -- bash /Users/user/gam/bench/oracle/graph/predict/mats_sft.sh \
#       Qwen/Qwen3-8B /Users/user/mpd-data/cluster/predict-sft-q06 'TRAIN_GLOB[,GLOB...]' STEPS HOURS \
#       prompts=GLOB pieces=GLOB behaviors=GLOB
# sft.py trains (chat format, PEFT export in OUT/peft) for at most HOURS and scores answer-token bits per
# held-out set with a learning curve; eval_kl.py then scores the answers as distributions (KL against
# M's measured one, beside the no-change answer) for the base and the trained oracle on every set.
# ORACLE may be a Hugging Face id in the cluster's cache or a snapshot directory (a RunPod pod).
set -Eeuo pipefail
here=$(cd "$(dirname "$0")" && pwd)
ORACLE=$1 OUT=$2 TRAIN=$3 STEPS=$4 HOURS=$5
shift 5
PY=$HOME/oracle-venv/bin/python
[ -x "$PY" ] || PY=python3
if [ ! -d "$ORACLE" ]; then
    export HF_HUB_OFFLINE=1
    ORACLE=$($PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('$ORACLE'))")
fi
sets=()
for spec in "$@"; do sets+=(--heldout "$spec"); done
mkdir -p "$OUT"
$PY "$here/sft.py" --model "$ORACLE" --train "$TRAIN" "${sets[@]}" --out "$OUT" --steps "$STEPS" --hours "$HOURS"
$PY "$here/eval_kl.py" --model "$ORACLE" "${sets[@]}" --out "$OUT/eval_kl_base.json"
$PY "$here/eval_kl.py" --model "$ORACLE" --adapters "$OUT/adapters.safetensors" "${sets[@]}" --out "$OUT/eval_kl_trained.json"
