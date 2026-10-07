#!/usr/bin/env bash
# The English checks on one MATS L40 (#2951): every example program rebuilt from its English alone
# and paraphrased, by a frozen local model (vLLM):
#   MATS_GPUS=1 mats-run graph-text 8 48 2 -- bash /Users/user/gam/bench/oracle/graph/mats_text.sh \
#       Qwen/Qwen3-8B /Users/user/mpd-data/cluster/graph-text
set -Eeuo pipefail
here=$(cd "$(dirname "$0")" && pwd)
MODEL=$1 OUT=$2
PY=$HOME/oracle-venv/bin/python
[ -x "$PY" ] || PY=python3
export HF_HUB_OFFLINE=1
mkdir -p "$OUT"
for target in vpd4l qwen3-0.6b; do
    progs=$($PY -c "import json; idx = json.load(open('$here/examples/index.json')); print(' '.join('$here/examples/' + n + '.py' for n, e in idx.items() if e['model'] == '$target'))")
    $PY "$here/rebuild.py" $progs --model $target --coder "$MODEL" --backend vllm --out "$OUT/rebuild_$target.jsonl"
    $PY "$here/paraphrase.py" $progs --model $target --writer "$MODEL" --backend vllm --out "$OUT/paraphrased_$target"
done
