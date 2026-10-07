#!/usr/bin/env bash
# R5 on one GPU (#2951): does the prediction-trained oracle predict interventions better than "no change"?
#   r5_pod.sh ORACLE_DIR TARGET_DIR HELDOUT_WINDOWS SEEN.json ADAPTERS PROMPTS.jsonl BEHAVIORS.jsonl OUT [FORMAT]
# 1. questions about held-out pieces on held-out texts (generate.py --piece-split heldout, 4,096 texts),
#    kept only where every named head and neuron is absent from the oracle's training questions (unseen.py);
# 2. eval_kl.py for the base oracle and the trained one on held-out prompts, held-out behaviors and those
#    unseen pieces: KL(M_e || answer) in bits beside the no-change answer's, 48 questions per type and set.
set -Eeuo pipefail
here=$(cd "$(dirname "$0")" && pwd)
ORACLE=$1 TARGET=$2 HELDW=$3 SEEN=$4 ADAPTERS=$5 PROMPTS=$6 BEHAVIORS=$7 OUT=$8 FORMAT=${9:-raw}
PY=${PY:-python3}
mkdir -p "$OUT"
[ -s "$OUT/pieces.jsonl" ] || $PY "$here/generate.py" --model "$TARGET" --windows "$HELDW" --texts 4096 --offset 8192 --split heldout \
    --piece-split heldout --seed 77 --batch 64 --out "$OUT/pieces.jsonl"
$PY "$here/unseen.py" filter --seen "$SEEN" --shard "$OUT/pieces.jsonl" --out "$OUT/unseen_pieces.jsonl"
sets=(--heldout "prompts=$PROMPTS" --heldout "behaviors=$BEHAVIORS" --heldout "unseen_pieces=$OUT/unseen_pieces.jsonl")
$PY "$here/eval_kl.py" --model "$ORACLE" --format "$FORMAT" "${sets[@]}" --per-type 48 --batch 16 --out "$OUT/eval_kl_base.json"
$PY "$here/eval_kl.py" --model "$ORACLE" --format "$FORMAT" --adapters "$ADAPTERS" "${sets[@]}" --per-type 48 --batch 16 --out "$OUT/eval_kl_trained.json"
