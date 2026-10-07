#!/usr/bin/env bash
# Patch tables (qwen_patch.py) for many Qwen3-0.6B behaviors on one MATS L40 (#2951):
#   MATS_GPUS=1 mats-run patch-q06 8 32 2 -- bash /Users/user/gam/bench/oracle/graph/examples/measure/mats_patch.sh \
#       /Users/user/mpd-data/cluster/patch-q06 BEHAVIOR.json [...]
set -Eeuo pipefail
here=$(cd "$(dirname "$0")" && pwd)
OUT=$1
shift
PY=$HOME/oracle-venv/bin/python
[ -x "$PY" ] || PY=python3
export HF_HUB_OFFLINE=1
mkdir -p "$OUT"
$PY "$here/qwen_patch.py" "$OUT" "$@"
