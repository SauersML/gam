#!/usr/bin/env bash
# Neuron tables (vpd_neurons.py) for many vpd4l behavior:layer jobs on one MATS L40 (#2951):
#   MATS_GPUS=1 mats-run neurons-vpd4l 8 32 2 -- bash /Users/user/gam/bench/oracle/graph/examples/measure/mats_neurons.sh \
#       /Users/user/mpd-data/cluster/neurons-vpd4l BEHAVIOR.json:LAYER [...] [vpd4l weight files, named so mats-run uploads them]
set -Eeuo pipefail
here=$(cd "$(dirname "$0")" && pwd)
OUT=$1
shift
PY=$HOME/oracle-venv/bin/python
[ -x "$PY" ] || PY=python3
export HF_HUB_OFFLINE=1
mkdir -p "$OUT"
$PY "$here/vpd_neurons.py" "$OUT" "$@"
