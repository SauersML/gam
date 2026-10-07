#!/usr/bin/env bash
# R2's arms (empty, native, VPD view, library view, mixed) on a MATS CPU job with the job's own checker build:
#   MATS_BUILD=1 mats-run r2-arms 16 48 6 -- bash /Users/user/gam/bench/oracle/graph/examples/measure/mats_arms.sh \
#       /Users/user/mpd-data/cluster/r2-arms/arms.jsonl BEHAVIOR.json [...] \
#       /Users/user/mpd-data/engine/vpd4l /Users/user/mpd-data/engine/vpd4l_decomposition /Users/user/mpd-data/decomp/start.components.json
# (the last three are named so mats-run uploads them; score_arms.py reads them from ~/mpd-data)
set -Eeuo pipefail
here=$(cd "$(dirname "$0")" && pwd)
OUT=$1
shift
behaviors=()
for a in "$@"; do case $a in *behaviors*.json) behaviors+=("$a") ;; esac; done
mkdir -p "$(dirname "$OUT")"
export GRAPH_CHECKER="$MPD_BIN/mpd_graph_2951" MPD_MEM_GIB=1 MEM_LEASE_GIB=1 CUDA_VISIBLE_DEVICES= RAYON_NUM_THREADS=16
python3 "$here/score_arms.py" "$OUT" "${behaviors[@]}"
