#!/usr/bin/env bash
# score_programs.py on a MATS CPU job with the job's checker build and the VPD view (#2951):
#   MATS_BUILD=1 mats-run vocab-gm-0 8 32 6 -- bash /Users/user/gam/bench/oracle/graph/examples/measure/mats_score_programs.sh \
#       /Users/user/mpd-data/cluster/vocab-gm-0/out.jsonl /Users/user/mpd-data/graph_oracle/behaviors/vpd4l 0 4 PROGRAM.py [...] \
#       /Users/user/mpd-data/engine/vpd4l /Users/user/mpd-data/engine/vpd4l_decomposition
set -Eeuo pipefail
here=$(cd "$(dirname "$0")" && pwd)
OUT=$1 BEH=$2 PART=$3 PARTS=$4
shift 4
programs=()
for a in "$@"; do case $a in *.py) programs+=("$a") ;; esac; done
mkdir -p "$(dirname "$OUT")"
export GRAPH_CHECKER="$MPD_BIN/mpd_graph_2951" MPD_MEM_GIB=1 MEM_LEASE_GIB=1 CUDA_VISIBLE_DEVICES= RAYON_NUM_THREADS=8
python3 "$here/score_programs.py" "$OUT" "$BEH" "$PART" "$PARTS" "${programs[@]}"
