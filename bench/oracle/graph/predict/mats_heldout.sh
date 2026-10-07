#!/usr/bin/env bash
# The held-out prediction sets of one target on one MATS L40 (#2951), each scored separately by sft.py:
#   mats_heldout.sh MODEL OUT_DIR HELDOUT_WINDOWS TEXTS BEHAVIORS_DIR [generate.py options...]
#   OUT_DIR/heldout_prompts.jsonl    held-out texts, training pieces (seen pieces, unseen prompts)
#   OUT_DIR/heldout_pieces.jsonl     held-out texts, held-out pieces (pieces never in training)
#   OUT_DIR/behaviors_train.jsonl    the training behaviors' prompts (training data)
#   OUT_DIR/behaviors_heldout.jsonl  the held-out behaviors' prompts (behaviors never in training)
set -Eeuo pipefail
here=$(cd "$(dirname "$0")" && pwd)
MODEL=$1 D=$2 WIN=$3 TEXTS=$4 BEH=$5
shift 5
bash "$here/mats_gen.sh" "$MODEL" "$D/heldout_prompts.jsonl" "$WIN" heldout train "$TEXTS" 0 "$@"
bash "$here/mats_gen.sh" "$MODEL" "$D/heldout_pieces.jsonl" "$WIN" heldout heldout "$TEXTS" 1 "$@"
bash "$here/mats_gen.sh" "$MODEL" "$D/behaviors_train.jsonl" - train train 0 2 --behaviors "$BEH" --behavior-split train "$@"
bash "$here/mats_gen.sh" "$MODEL" "$D/behaviors_heldout.jsonl" - heldout train 0 3 --behaviors "$BEH" --behavior-split heldout "$@"
