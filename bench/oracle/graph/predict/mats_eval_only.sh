#!/usr/bin/env bash
# Held-out bits of existing adapters against the base oracle on one MATS L40 (#2951; sft.py --eval-only):
#   MATS_GPUS=1 mats-run NAME 6 32 2 -- bash /Users/user/gam/bench/oracle/graph/predict/mats_eval_only.sh ORACLE ADAPTERS OUT FORMAT NAME=GLOB...
set -Eeuo pipefail
here=$(cd "$(dirname "$0")" && pwd)
ORACLE=$1 ADAPTERS=$2 OUT=$3 FORMAT=$4
shift 4
PY=${PY:-$HOME/oracle-venv/bin/python}
[ -x "$PY" ] || PY=python3
snap=$(ls -d "$HOME/.cache/huggingface/hub/models--${ORACLE//\//--}/snapshots/"*/ 2> /dev/null | head -1)
[ -n "$snap" ] || snap=$($PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('$ORACLE'))")
sets=()
for spec in "$@"; do sets+=(--heldout "$spec"); done
first=${1#*=}
$PY "$here/sft.py" --model "${snap%/}" --format "$FORMAT" --train "$first" "${sets[@]}" --out "$OUT" --eval-only "$ADAPTERS" --eval-per-type 64 --batch 8
