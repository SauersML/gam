#!/usr/bin/env bash
# vpd_sub_patch.py tables (VPD MLP subcomponents by patch recovery) for many vpd4l behaviors on a MATS CPU job:
#   mats-run vpdsub-gm-0 8 24 6 -- bash /Users/user/gam/bench/oracle/graph/examples/measure/mats_vpdsub.sh \
#       /Users/user/mpd-data/cluster/vpdsub-gm-0 BEHAVIOR.json [...] /Users/user/mpd-data/vpd/t-9d2b8f02 /Users/user/mpd-data/oracle/vpd/uv.safetensors
set -Eeuo pipefail
here=$(cd "$(dirname "$0")" && pwd)
OUT=$1
shift
behaviors=()
for a in "$@"; do case $a in *behaviors*.json) behaviors+=("$a") ;; esac; done
mkdir -p "$OUT"
PY=$HOME/oracle-venv/bin/python
[ -x "$PY" ] || PY=python3
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8
$PY "$here/vpd_sub_patch.py" "$OUT" "${behaviors[@]}" --blocks mlp
