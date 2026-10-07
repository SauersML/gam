#!/usr/bin/env bash
# Prediction data and the oracle's first SFT on one MATS L40 (#2951):
#   MATS_QOS=debug MATS_GPUS=1 mats-run predict-sft1 8 64 2 -- bash /Users/user/gam/bench/oracle/graph/predict/mats_sft.sh \
#       /Users/user/mpd-data/cluster/predict-sft1 /Users/user/mpd-data/qwen3_fineweb/train/windows_T128.u32 \
#       /Users/user/mpd-data/qwen3_fineweb/heldout/windows_T128.u32 TRAIN_TEXTS STEPS
# generate.py writes a held-out shard (FineWeb held-out texts) and a training shard (training texts) on the
# GPU, then sft.py trains Qwen3-8B + LoRA on them in the job's remaining time and evaluates base vs trained.
set -Eeuo pipefail
here=$(cd "$(dirname "$0")" && pwd)
OUT=$1 TRAIN=$2 HELD=$3 TEXTS=$4 STEPS=$5
PY=$HOME/oracle-venv/bin/python
export HF_HUB_OFFLINE=1
SNAP=$($PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-0.6B'))")
mkdir -p "$OUT/data"
[ -s "$OUT/data/heldout_000.jsonl" ] || $PY "$here/generate.py" --model "$SNAP" --windows "$HELD" --out "$OUT/data/heldout_000.jsonl" --texts 256 --batch 32 --split heldout --seed 1000
[ -s "$OUT/data/train_000.jsonl" ] || $PY "$here/generate.py" --model "$SNAP" --windows "$TRAIN" --out "$OUT/data/train_000.jsonl" --texts "$TEXTS" --batch 64 --split train --seed 0
HOURS=$($PY -c "print(round(1.95 - $SECONDS / 3600, 3))")
$PY "$here/sft.py" --model Qwen/Qwen3-8B --train "$OUT/data/train_*.jsonl" --heldout "$OUT/data/heldout_*.jsonl" --out "$OUT/sft" --steps "$STEPS" --hours "$HOURS"
