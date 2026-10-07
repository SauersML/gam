#!/usr/bin/env bash
# Print the winning examples (print_winners.py) on a MATS CPU job (#2951):
#   mats-run print-gm 8 24 3 -- bash /Users/user/gam/bench/oracle/graph/examples/measure/mats_print.sh \
#       /Users/user/mpd-data/cluster/print-gm/printed /Users/user/mpd-data/graph_oracle/behaviors/vpd4l RESCORE.jsonl [...] \
#       /Users/user/mpd-data/vpd/t-9d2b8f02
set -Eeuo pipefail
here=$(cd "$(dirname "$0")" && pwd)
OUT=$1 BEH=$2
shift 2
tables=()
for a in "$@"; do case $a in *.jsonl) tables+=("$a") ;; esac; done
mkdir -p "$OUT"
PY=$HOME/oracle-venv/bin/python
[ -x "$PY" ] || PY=python3
export PRINTED_DIR="$OUT" BEHAVIORS_DIR="$BEH" CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1
$PY "$here/print_winners.py" "${tables[@]}"
