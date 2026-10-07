#!/usr/bin/env bash
# The self-explanation arm (#2951): Qwen3-8B + LoRA explaining itself (target qwen3-8b) against the same
# oracle explaining Qwen3-0.6B (target qwen3-0.6b). Both arms get identical settings: the same oracle base,
# start (--init, g-predict's SFT for that target when it exists), behavior families (g-behaviors' build.py
# for each size, same templates and split), steps, group size, score (score.py's checker + reader) and
# evaluation. Bits are not comparable across targets, so each arm reports per held-out behavior the share
# of the gap from the empty program to the search baseline that the oracle's mean program closes:
# (S_empty - S_oracle) / (S_empty - S_search), from eval.jsonl (train.py's evaluation scores all three on
# the same experiments).
#
#   self_explain.sh [STEPS] [extra train.py arguments]      submits both arms to MATS (2 L40s each)
#
# Refuses to submit until every prerequisite exists for both targets: behavior files, mech's shape
# registry entry (mech.py shapes --write) and score.py's export of the model for the checker.
set -Eeuo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
graph=$(dirname "$here")
steps=${1:-300}
shift || true
missing=0
for target in qwen3-0.6b qwen3-8b; do
    ls "$HOME/mpd-data/graph_oracle/behaviors/$target/"*.json > /dev/null 2>&1 || { echo "missing: behaviors for $target (g-behaviors build.py --model $target)"; missing=1; }
    python3 -c "import json,sys; sys.exit('$target' not in json.load(open('$graph/shapes.json')))" || { echo "missing: $target in mech's shapes.json (g-mech)"; missing=1; }
    grep -q "\"$target\"" "$graph/score.py" || { echo "missing: $target in score.py's EXPORTS (g-exec)"; missing=1; }
done
[ $missing = 0 ] || exit 3
for target in qwen3-0.6b qwen3-8b; do
    name=oracle-rl-self-${target//./}
    init=()
    [ -d "$HOME/mpd-data/graph_oracle/predict/$target/sft" ] && init=(--init "$HOME/mpd-data/graph_oracle/predict/$target/sft")
    MATS_GPUS=2 "$graph/../../mats/mats-run" "$name" 24 128 23 -- bash "$here/mats_rl.sh" --mode grpo --base Qwen/Qwen3-8B "${init[@]}" --model "$target" \
        --behaviors "$HOME/mpd-data/graph_oracle/behaviors" --scorer checker --score-workers 8 --out "$HOME/mpd-data/cluster/$name" \
        --steps "$steps" --hours 22 --behaviors-per-step 8 --samples 8 --eval-every 25 "$@"
done
