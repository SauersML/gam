"""vary.py: per-variable counterfactual items keep the query context and vary the variable they name."""

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import vary  # noqa: E402

BEHAVIOR = Path.home() / "mpd-data/graph_oracle/behaviors/vpd4l/induction_random.words8.json"


def test_query_context():
    # "A B C D . A B C": the repeat "A B C" occurred before, so edits stop at the second copy
    assert vary.context([0, 1, 2, 3, 4, 9, 1, 2, 3], 1, 8) == 6
    assert vary.context([0, 5, 6, 7], 1, 3) == 3  # nothing repeats: only the target token is kept


def test_induction_items_reorder_the_first_copy():
    if not BEHAVIOR.exists():
        return
    behavior = json.loads(BEHAVIOR.read_text())
    behavior["prompts"] = [p for p in behavior["prompts"] if "of" not in p][:12]
    new, counts = vary.vary(behavior, random.Random(0))
    assert counts == {"tokens": 12} and new and all(p["varies"] == ["match"] and p["edit"][0] == "swap" for p in new)
    alg = vary.family.Algorithm(behavior["family"])
    for p in new:
        o = behavior["prompts"][p["of"]]
        t = p["target_positions"][0]
        start = vary.context(o["token_ids"], 1, t)
        assert p["token_ids"][start: t + 1] == o["token_ids"][start: t + 1]  # the repeat is untouched
        a, b = (alg.at(alg_text, t) for alg_text in (
            [str(i) for i in p["token_ids"]], [str(i) for i in o["token_ids"]]))
        assert a["match"] != b["match"] and a["prev"] == b["prev"] and a["answer"] != b["answer"]
        assert p["counterfactual"]["token_ids"] == o["token_ids"]
