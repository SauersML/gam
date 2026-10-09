"""The family algorithms (bench/oracle/graph/algorithms/, index.json: behavior family -> algorithm) on their vpd4l
behaviors: each one's answer (or one of its set of answers) is the prompt's next token on nearly every target, and no
held-out family has one.

  ~/mpd-data/venv/bin/python -m pytest bench/oracle/graph/test_algorithms.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import family  # noqa: E402
import mech  # noqa: E402

BEHAVIORS = Path.home() / "mpd-data/graph_oracle/behaviors/vpd4l"


def test_every_algorithm_is_indexed():
    assert sorted(set(family.INDEX.values())) == sorted(p.stem for p in (HERE / "algorithms").glob("*.py"))


def test_no_heldout_family_has_an_algorithm():
    sys.path.insert(0, str(HERE / "behaviors"))
    import build

    assert [f for f in family.INDEX if build.split_of(f) != "train"] == []
    assert family.source("induction_random") and "def answer" in family.source("induction_random")


def test_algorithms_predict_their_behaviors():
    if not BEHAVIORS.exists():
        return
    low = {}
    for path in sorted(BEHAVIORS.glob("*.json")):
        behavior = json.loads(path.read_text())
        if behavior["family"] not in family.INDEX:
            continue
        payload = mech.behavior_tokens(behavior, "vpd4l")
        alg = family.Algorithm(behavior["family"])
        hits = []
        for tokens, targets in zip(payload["prompts"], payload["targets"]):
            for t in targets:
                answer = alg.at(tokens, t)["answer"]
                nxt = tokens[t + 1] if t + 1 < len(tokens) else None
                hits.append(nxt is not None and (answer == nxt or not isinstance(answer, str) and nxt in (answer or ())
                                                 or isinstance(answer, str) and len(answer) > len(nxt) and answer.startswith(nxt)))
        if sum(hits) < 0.95 * len(hits):
            low[behavior["id"]] = sum(hits) / len(hits)
    assert not low, low
