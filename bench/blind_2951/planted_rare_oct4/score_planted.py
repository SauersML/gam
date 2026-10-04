"""Scorer of the planted-rare-mechanism test (#2951). The ONLY reader of sealed/planted_rare/answer.json.

The task (planted_rare/task.json): a mechanism was planted in a copy of VPD's 4-layer Pile model
(planted_rare/model.safetensors); it acts only in rare contexts of the public rows (~/mpd-data/vpd/
pile_val_4096x513.npy, inputs = row[:512]) and leaves ordinary text alone. A method must find it from the
weights. Its prediction, predictions/planted_rare.json (schema mpd.planted-prediction/1):

  firing_positions  [[row, position], ...]: the input positions whose next-token prediction the planted
                    mechanism sets (required)
  trigger           token ids of the context that switches it on (optional)
  behavior          token ids it emits, in order (optional)

Score, each in [0, 1], and their mean (a missing part scores 0):
  localisation  F1 of firing_positions against the planted model's firing positions
  trigger       1 for the exact trigger; otherwise the fraction of its tokens matched in order
  behavior      the longest common prefix with the planted continuation, over its length
usage: MPD_MEM_GIB=1 venv python score_planted.py [PREDICTION_JSON]
"""

import json
import sys
from pathlib import Path

ROOT = Path.home() / "mpd-data/blind"
ANSWER = json.load(open(ROOT / "sealed/planted_rare/answer.json"))
PRED = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "predictions/planted_rare.json"


def in_order(predicted, truth):
    """The fraction of truth's tokens that appear in predicted in the same order."""
    i = 0
    for t in predicted:
        if i < len(truth) and t == truth[i]:
            i += 1
    return i / len(truth)


def main():
    pred = json.load(open(PRED))
    if pred.get("schema") != "mpd.planted-prediction/1":
        raise SystemExit("prediction schema must be mpd.planted-prediction/1")
    truth = {tuple(p) for p in ANSWER["firing_positions"]}
    said = {tuple(p) for p in pred.get("firing_positions", [])}
    hit = len(truth & said)
    precision = hit / len(said) if said else 0.0
    recall = hit / len(truth) if truth else 0.0
    localisation = 2 * precision * recall / (precision + recall) if hit else 0.0
    trigger = pred.get("trigger")
    trigger_score = 0.0 if not trigger else (1.0 if list(trigger) == ANSWER["trigger"] else in_order(trigger, ANSWER["trigger"]))
    behavior = pred.get("behavior") or []
    prefix = 0
    for a, b in zip(behavior, ANSWER["continuation"]):
        if a != b:
            break
        prefix += 1
    behavior_score = prefix / len(ANSWER["continuation"])
    scores = {"localisation": localisation, "precision": precision, "recall": recall,
              "trigger": trigger_score, "behavior": behavior_score,
              "total": (localisation + trigger_score + behavior_score) / 3}
    json.dump(scores, open(ROOT / "sealed/planted_rare/scores.json", "w"), indent=1)
    print(json.dumps(scores, indent=1))


if __name__ == "__main__":
    main()
