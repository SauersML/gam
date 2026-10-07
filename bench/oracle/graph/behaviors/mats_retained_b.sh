#!/usr/bin/env bash
# Wording B of the retained-state families on Qwen3-0.6B (#2951): retained_state.py makes the turn-1 runs the way the
# study's hidden_choice.py does (its sampling settings and openings; the study's script itself needs a newer transformers
# than the cluster venv for its arms), then writes the behavior files. One GPU job:
#   MATS_QOS=debug MATS_GPUS=1 mats-run behaviors-retB 8 24 1 -- bash /Users/user/gam/bench/oracle/graph/behaviors/mats_retained_b.sh \
#       /Users/user/mpd-data/cluster/behaviors-retB /Users/user/mpd-data/graph_oracle/retained_src
set -Eeuo pipefail
here=$(cd "$(dirname "$0")" && pwd)
OUT=$1 STUDY=$2 MODEL=${3:-qwen3-0.6b} HFID=${4:-Qwen/Qwen3-0.6B}
PY=$HOME/oracle-venv/bin/python
mkdir -p "$OUT"
RETAINED_STUDY=$STUDY $PY "$here/retained_state.py" --model "$MODEL" --device cuda --out "$OUT" --wording B --generate 640 \
    --study-file "$OUT/${MODEL//-/_}_B.json"
