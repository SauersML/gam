#!/usr/bin/env bash
# One prediction-data task on a MATS L40 (#2951). As a job array, task k reads texts k*TEXTS .. (k+1)*TEXTS-1
# of the file's fixed text order, so tasks never repeat a text:
#   MATS_GPUS=1 MATS_ARRAY=0-15%2 mats-run predict-gen-q06 8 24 2 -- bash /Users/user/gam/bench/oracle/graph/predict/mats_gen.sh \
#       Qwen/Qwen3-0.6B /Users/user/mpd-data/cluster/predict-gen-q06/train_{task}.jsonl \
#       /Users/user/mpd-data/qwen3_fineweb/train/windows_T128.u32 train train 8192 {task} [generate.py options...]
# MODEL is a Hugging Face id in the cluster's cache (or "vpd4l"); SPLIT names the texts' split, PIECE_SPLIT
# the pieces' (train / heldout, generate.py --piece-split); WINDOWS may be "-" with --behaviors DIR.
set -Eeuo pipefail
here=$(cd "$(dirname "$0")" && pwd)
MODEL=$1 OUT=$2 WIN=$3 SPLIT=$4 PSPLIT=$5 TEXTS=$6 TASK=$7
shift 7
PY=${PY:-$HOME/oracle-venv/bin/python}
[ -x "$PY" ] || PY=python3
mkdir -p "$(dirname "$OUT")"
if [ "$MODEL" = vpd4l ]; then
    target=(--target vpd4l)
else
    # The cached snapshot directory itself (offline snapshot_download refuses snapshots without README/LICENSE).
    snap=$(ls -d "$HOME/.cache/huggingface/hub/models--${MODEL//\//--}/snapshots/"*/ 2> /dev/null | head -1)
    [ -n "$snap" ] || snap=$($PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL'))")
    target=(--model "${snap%/}")
fi
windows=()
[ "$WIN" = - ] || windows=(--windows "$WIN" --texts "$TEXTS" --offset $((TASK * TEXTS)))
$PY "$here/generate.py" "${target[@]}" "${windows[@]}" --out "$OUT" --seed "$TASK" --split "$SPLIT" --piece-split "$PSPLIT" --batch 64 "$@"
