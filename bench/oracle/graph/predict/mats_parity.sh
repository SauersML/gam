#!/usr/bin/env bash
# Parity of generate.py against oracle.rs on MATS (#2951), one CPU job with the commit's binaries:
#   MATS_QOS=debug MATS_BUILD=1 mats-run predict-parity 16 24 1 -- bash /Users/user/gam/bench/oracle/graph/predict/mats_parity.sh \
#       /Users/user/mpd-data/cluster/predict-parity /Users/user/mpd-data/qwen3_fineweb/heldout/windows_T128.u32
# A small held-out shard is generated on the CPU (float32), then each sampled question is measured again by
# mpd_oracle_2951 (float64 host execution); the report is OUT/parity.json.
set -Eeuo pipefail
here=$(cd "$(dirname "$0")" && pwd)
OUT=$1 WIN=$2
PY=$HOME/oracle-venv/bin/python
export HF_HUB_OFFLINE=1
SNAP=$($PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-0.6B'))")
mkdir -p "$OUT"
$PY "$here/generate.py" --model "$SNAP" --windows "$WIN" --out "$OUT/parity_shard.jsonl" --texts 32 --batch 16 --lengths 24,32 --split heldout --seed 7
PORT=$((40000 + ${SLURM_JOB_ID:-1} % 20000))
"$MPD_BIN/mpd_oracle_2951" "127.0.0.1:$PORT" 8 "qwen=$SNAP" > "$OUT/server.log" 2>&1 &
server=$!
$PY -c "
import socket, sys, time
for _ in range(900):
    try:
        socket.create_connection(('127.0.0.1', $PORT)).close(); sys.exit(0)
    except OSError:
        time.sleep(2)
sys.exit('server did not start')"
$PY "$here/parity.py" --shard "$OUT/parity_shard.jsonl" --windows "$WIN" --server "127.0.0.1:$PORT" --model qwen --per-type 6 > "$OUT/parity.json"
kill $server
