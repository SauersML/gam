#!/usr/bin/env bash
# Learnability test of R5 on one GPU (#2951): can the oracle learn what a FIXED set of Qwen3-0.6B parts does?
#   r5_parts_pod.sh ORACLE_DIR TARGET_DIR TRAIN_WINDOWS HELDOUT_WINDOWS PIECES.json OUT STEPS HOURS
# Every question asks about one of the parts in PIECES.json (generate.py --pieces): edits (scale by 0, 0.5,
# 2), swaps from another text, cuts from the part into a later read or the logits with x' another text.
# Training: 16,384 training texts; half of each type's draws from questions whose measured change exceeds 0.1
# bit. Held out: texts never trained on, the SAME parts: every question (parts_all) and those above 0.1 bit
# (parts_moved). sft.py scores both every 100 steps and saves adapters every 300 steps; eval_kl.py scores the
# base oracle and every saved adapter.
set -Eeuo pipefail
here=$(cd "$(dirname "$0")" && pwd)
ORACLE=$1 TARGET=$2 TRAINW=$3 HELDW=$4 PIECES=$5 OUT=$6 STEPS=$7 HOURS=$8
PY=${PY:-python3}
mkdir -p "$OUT/data"
# DATA_DIR: reuse another arm's questions (train, parts_all, parts_moved), so two arms see the same data.
if [ -n "${DATA_DIR:-}" ]; then
    for f in train parts_all parts_moved; do [ -s "$OUT/data/$f.jsonl" ] || cp "$DATA_DIR/$f.jsonl" "$OUT/data/$f.jsonl"; done
fi
# VECTORS=1: the vector channel; every question about a part carries the part's vectors (vectors.py) as soft tokens.
vec=()
if [ -n "${VECTORS:-}" ]; then
    [ -s "$OUT/vectors.safetensors" ] || $PY "$here/vectors.py" --model "$TARGET" --pieces "$PIECES" --out "$OUT/vectors.safetensors"
    vec=(--vectors "$OUT/vectors.safetensors")
fi
fixed=(--types edit,swap,cut --pieces "$PIECES" --batch 64 --answer "${ANSWER:-distribution}")  # ANSWER=delta: the change only
gen() { [ -s "$OUT/data/$1.jsonl" ] || $PY "$here/generate.py" --model "$TARGET" --out "$OUT/data/$1.jsonl" "${fixed[@]}" "${@:2}"; }
gen train --windows "$TRAINW" --texts 16384 --offset 340000 --split train --seed 300
gen parts_all --windows "$HELDW" --texts 2048 --offset 50000 --split heldout --seed 301
gen parts_moved --windows "$HELDW" --texts 4096 --offset 52048 --split heldout --min-kl 0.1 --seed 302
sets=(--heldout "parts_all=$OUT/data/parts_all.jsonl" --heldout "parts_moved=$OUT/data/parts_moved.jsonl")
[ -s "$OUT/sft/adapters.safetensors" ] || $PY "$here/sft.py" --model "$ORACLE" --train "$OUT/data/train.jsonl" "${sets[@]}" --out "$OUT/sft" \
    --steps "$STEPS" --hours "$HOURS" --changed-min 0.1 --changed-share 0.5 --eval-every 100 --curve-per-type 32 --eval-per-type 64 --save-every 300 "${vec[@]}"
kl() { [ -s "$OUT/sft/eval_kl_$1.json" ] || $PY "$here/eval_kl.py" --model "$ORACLE" "${sets[@]}" --per-type 64 --stratify --batch 16 --out "$OUT/sft/eval_kl_$1.json" "${@:2}"; }
kl base
ch() { [ ${#vec[@]} -gt 0 ] && echo "--channel $OUT/sft/$1.safetensors"; }  # the channel trained beside those adapters
kl trained --adapters "$OUT/sft/adapters.safetensors" "${vec[@]}" $(ch channel)
for f in "$OUT"/sft/adapters_step*.safetensors; do
    s=${f##*_step}; s=${s%.safetensors}
    kl "step$s" --adapters "$f" "${vec[@]}" $(ch "channel_step$s")
done
