#!/usr/bin/env bash
# The behavior suite and the retained-state families on larger Qwen3 models on MATS (#2951), one GPU job:
#   MATS_QOS=debug MATS_GPUS=1 mats-run behaviors-scale 8 72 2 -- bash /Users/user/gam/bench/oracle/graph/behaviors/mats_scale.sh \
#       /Users/user/mpd-data/cluster/behaviors-scale /Users/user/mpd-data/circuits/sva /Users/user/mpd-data/graph_oracle/retained_src \
#       qwen3-1.7b,qwen3-4b,qwen3-8b
# Float32 on an L40 (48 GB holds Qwen3-8B in float32), the same precision as the Mac's Qwen3-0.6B files. Outputs land in
# OUT/<model>/ and OUT/summary.tsv; `mats-pull behaviors-scale` brings them back. MATS_ENV="RETAINED_ONLY=1" runs only the
# retained-state families.
set -Eeuo pipefail
here=$(cd "$(dirname "$0")" && pwd)
OUT=$1 SVA=$2 STUDY=$3 MODELS=${4:-qwen3-1.7b,qwen3-4b,qwen3-8b} FAMS=${5:-}  # FAMS: a family subset, no retained-state run
PY=${PY:-$HOME/oracle-venv/bin/python}  # a RunPod run passes PY=python3 (RP_PYENV=oracle)
mkdir -p "$OUT"
for m in ${MODELS//,/ }; do
  [ -n "${RETAINED_ONLY:-}" ] || $PY "$here/build.py" --model "$m" --device cuda --out "$OUT" --sva "$SVA" ${FAMS:+--families "$FAMS"}
  [ -n "$FAMS" ] || RETAINED_STUDY=$STUDY $PY "$here/retained_state.py" --model "$m" --device cuda --out "$OUT"
done
