#!/usr/bin/env bash
# Rescore the native vpd4l examples on a MATS CPU job, part PART of PARTS (#2951):
#   MATS_BUILD=1 mats-run rescore-gm-0 8 24 4 -- bash /Users/user/gam/bench/oracle/graph/examples/measure/mats_rescore.sh \
#       /Users/user/mpd-data/cluster/rescore-gm-0/out.jsonl 0 4 /Users/user/mpd-data/graph_oracle/behaviors/vpd4l /Users/user/mpd-data/engine/vpd4l
set -Eeuo pipefail
here=$(cd "$(dirname "$0")" && pwd)
OUT=$1 PART=$2 PARTS=$3 BEH=$4
mkdir -p "$(dirname "$OUT")"
export GRAPH_CHECKER="$MPD_BIN/mpd_graph_2951" MPD_MEM_GIB=1 MEM_LEASE_GIB=1 CUDA_VISIBLE_DEVICES= RAYON_NUM_THREADS=8
python3 "$here/rescore_examples.py" "$OUT" "$PART" "$PARTS" "$BEH"
