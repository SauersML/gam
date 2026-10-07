#!/usr/bin/env bash
# R5's real test on one GPU (#2951): prediction SFT of the oracle on Qwen3-0.6B questions (counterfactual cuts,
# half of each type's draws from questions whose answer differs from no change), then eval_kl of the base
# oracle and of the adapters at SAVE_AT and at the end on held-out prompts, held-out pieces and held-out
# behaviors, each type's questions spread over the sizes of the measured change (report.py splits them):
#   r5_sft_pod.sh ORACLE_DIR TARGET_DIR 'TRAIN_GLOB[,...]' HELDOUT_WINDOWS BEHAVIORS_DIR OUT STEPS SAVE_AT HOURS
set -Eeuo pipefail
here=$(cd "$(dirname "$0")" && pwd)
ORACLE=$1 TARGET=$2 TRAIN=$3 HELDW=$4 BEH=$5 OUT=$6 STEPS=$7 SAVE_AT=$8 HOURS=$9
PY=${PY:-python3}
mkdir -p "$OUT/data"
gen() { [ -s "$OUT/data/$1.jsonl" ] || $PY "$here/generate.py" --model "$TARGET" --out "$OUT/data/$1.jsonl" --batch 64 "${@:2}"; }
gen heldout_prompts --windows "$HELDW" --texts 512 --offset 0 --split heldout --piece-split train --seed 100
gen heldout_pieces --windows "$HELDW" --texts 768 --offset 512 --split heldout --piece-split heldout --seed 101
gen behaviors_heldout --behaviors "$BEH" --behavior-split heldout --per-behavior 48 --split heldout --seed 102
sets=(--heldout "prompts=$OUT/data/heldout_prompts.jsonl" --heldout "pieces=$OUT/data/heldout_pieces.jsonl" --heldout "behaviors=$OUT/data/behaviors_heldout.jsonl")
[ -s "$OUT/sft/adapters.safetensors" ] || $PY "$here/sft.py" --model "$ORACLE" --train "$TRAIN" "${sets[@]}" --out "$OUT/sft" --steps "$STEPS" \
    --save-at "$SAVE_AT" --eval-every 100 --curve-per-type 8 --eval-per-type 32 --hours "$HOURS"
kl() { [ -s "$OUT/sft/eval_kl_$1.json" ] || $PY "$here/eval_kl.py" --model "$ORACLE" "${sets[@]}" --per-type 64 --stratify --batch 16 --out "$OUT/sft/eval_kl_$1.json" "${@:2}"; }
kl base
for s in ${SAVE_AT//,/ }; do
    if [ -s "$OUT/sft/adapters_step$s.safetensors" ]; then kl "step$s" --adapters "$OUT/sft/adapters_step$s.safetensors"; fi  # absent when time cut training short
done
kl trained --adapters "$OUT/sft/adapters.safetensors"
