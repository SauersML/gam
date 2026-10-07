#!/usr/bin/env bash
# R5 on internal interventions with large effects, one GPU (#2951): does the oracle learn what Qwen3-0.6B's
# parts do once the training questions stop being dominated by edits that change nothing?
#   r5_big_pod.sh ORACLE_DIR TARGET_DIR TRAIN_WINDOWS HELDOUT_WINDOWS OUT STEPS HOURS
# Training: edit, swap and cut questions only (generate.py --types edit,swap,cut --aim large), kept when the
# measured change KL(M || M_e) is at least 0.1 bits; a quarter of the draws from those above 1 bit.
# Held out (texts never trained on): questions above 1 bit about training pieces (big_prompts) and about
# held-out pieces (big_pieces), and 0.1-1 bit questions about training pieces (mid_prompts). sft.py scores
# them every 100 steps (answer bits and measured-minus-no-change bits per type) and saves adapters every
# 300 steps; eval_kl.py then scores the base oracle and every saved adapter on the three sets.
set -Eeuo pipefail
here=$(cd "$(dirname "$0")" && pwd)
ORACLE=$1 TARGET=$2 TRAINW=$3 HELDW=$4 OUT=$5 STEPS=$6 HOURS=$7
PY=${PY:-python3}
mkdir -p "$OUT/data"
big=(--types edit,swap,cut --aim large --batch 64)
gen() { [ -s "$OUT/data/$1.jsonl" ] || $PY "$here/generate.py" --model "$TARGET" --out "$OUT/data/$1.jsonl" "${big[@]}" "${@:2}"; }
gen train --windows "$TRAINW" --texts 32768 --offset 300000 --split train --piece-split train --min-kl 0.1 --seed 200
gen big_prompts --windows "$HELDW" --texts 8192 --offset 20000 --split heldout --piece-split train --min-kl 1.0 --seed 201
gen big_pieces --windows "$HELDW" --texts 8192 --offset 30000 --split heldout --piece-split heldout --min-kl 1.0 --seed 202
gen mid_prompts --windows "$HELDW" --texts 1024 --offset 40000 --split heldout --piece-split train --min-kl 0.1 --seed 203
$PY - "$OUT/data/mid_prompts.jsonl" "$OUT/data/mid_prompts_0.1_1.jsonl" <<'PY'
import json, sys
with open(sys.argv[2], "w") as f:
    for line in open(sys.argv[1]):
        if json.loads(line)["numbers"]["kl_bits"] < 1.0:
            f.write(line)
PY
sets=(--heldout "big_prompts=$OUT/data/big_prompts.jsonl" --heldout "big_pieces=$OUT/data/big_pieces.jsonl" --heldout "mid_prompts=$OUT/data/mid_prompts_0.1_1.jsonl")
[ -s "$OUT/sft/adapters.safetensors" ] || $PY "$here/sft.py" --model "$ORACLE" --train "$OUT/data/train.jsonl" "${sets[@]}" --out "$OUT/sft" \
    --steps "$STEPS" --hours "$HOURS" --changed-min 1.0 --changed-share 0.25 --eval-every 100 --curve-per-type 24 --eval-per-type 48 --save-every 300
kl() { [ -s "$OUT/sft/eval_kl_$1.json" ] || $PY "$here/eval_kl.py" --model "$ORACLE" "${sets[@]}" --per-type 48 --batch 16 --out "$OUT/sft/eval_kl_$1.json" "${@:2}"; }
kl base
kl trained --adapters "$OUT/sft/adapters.safetensors"
for f in "$OUT"/sft/adapters_step*.safetensors; do
    s=${f##*_step}; s=${s%.safetensors}
    kl "step$s" --adapters "$f"
done
