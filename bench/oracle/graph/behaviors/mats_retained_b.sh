#!/usr/bin/env bash
# Wording B of the retained-state families on Qwen3-0.6B (#2951): the study's hidden_choice.py (a read-only copy) makes
# the turn-1 runs, then retained_state.py writes the behavior files. One GPU job:
#   MATS_QOS=debug MATS_GPUS=1 mats-run behaviors-retB 8 24 1 -- bash /Users/user/gam/bench/oracle/graph/behaviors/mats_retained_b.sh \
#       /Users/user/mpd-data/cluster/behaviors-retB /Users/user/mpd-data/graph_oracle/retained_src
set -Eeuo pipefail
here=$(cd "$(dirname "$0")" && pwd)
OUT=$1 STUDY=$2 MODEL=${3:-qwen3-0.6b} HFID=${4:-Qwen/Qwen3-0.6B}
PY=$HOME/oracle-venv/bin/python
mkdir -p "$OUT"
# hidden_choice.py loads with device_map, which needs accelerate; the shared venv lacks it, so it goes in a private dir
~/.local/bin/uv pip install -q --python "$PY" --target "$OUT/pylib" --no-deps accelerate
export PYTHONPATH=$OUT/pylib${PYTHONPATH:+:$PYTHONPATH}
(cd "$STUDY" && $PY hidden_choice.py --model "$HFID" --n 640 --wording B --arms visible --device cuda --seed 1 --out "$OUT/${MODEL//-/_}_B.json")
RETAINED_STUDY=$STUDY $PY "$here/retained_state.py" --model "$MODEL" --device cuda --out "$OUT" --study-file "$OUT/${MODEL//-/_}_B.json"
