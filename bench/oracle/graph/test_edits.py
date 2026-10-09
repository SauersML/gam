"""edits: dropping names from a gate program, credit and refinement under a stand-in score.

  ~/mpd-data/venv/bin/python -m pytest bench/oracle/graph/test_edits.py
"""

import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import edits  # noqa: E402

JOINED = 'def on(tokens, targets):\n    return {0: "<p:0.fc.1><p:0.down.2>", 3: "<p:2.v.80><p:2.o.63><p:3.down.885>"}\n'
LISTED = 'def on(tokens, targets):\n    return {3: ["<p:2.v.80>", "<p:2.o.63>"], 0: ["<p:0.fc.1>"]}\n'
NEEDED = {"<p:2.v.80>", "<p:2.o.63>"}


def named(source):
    return re.findall(r"<p:[^>]+>", source)


def stand_in(sources):
    """Excess 10 per needed name missing, pairs 1 per name: the key is (excess, pairs)."""
    return [{"excess": 10 * len(NEEDED - set(named(s))), "pairs": len(named(s))} for s in sources]


def key(s):
    return (s["excess"], s["pairs"])


def test_names_and_drops():
    spans = edits.names(JOINED)
    assert [JOINED[a:b] for a, b in spans] == ["<p:0.fc.1>", "<p:0.down.2>", "<p:2.v.80>", "<p:2.o.63>", "<p:3.down.885>"]
    assert edits.drop(JOINED, spans[:2]) == 'def on(tokens, targets):\n    return {0: "", 3: "<p:2.v.80><p:2.o.63><p:3.down.885>"}\n'
    listed = edits.names(LISTED)
    assert edits.drop(LISTED, listed[:1]) == 'def on(tokens, targets):\n    return {3: ["<p:2.o.63>"], 0: ["<p:0.fc.1>"]}\n'
    assert edits.drop(LISTED, listed[1:2]) == 'def on(tokens, targets):\n    return {3: ["<p:2.v.80>"], 0: ["<p:0.fc.1>"]}\n'
    assert edits.drop(LISTED, listed[2:]) == 'def on(tokens, targets):\n    return {3: ["<p:2.v.80>", "<p:2.o.63>"], 0: []}\n'


def test_credit_signs_and_refinement():
    base, signs = edits.credit(JOINED, stand_in, key)
    assert base == (0, 5)
    by_name = {JOINED[a:b]: v for (a, b), v in signs.items()}
    assert by_name == {"<p:0.fc.1>": -1, "<p:0.down.2>": -1, "<p:2.v.80>": 1, "<p:2.o.63>": 1, "<p:3.down.885>": -1}
    best, k, dropped = edits.refine(JOINED, stand_in, key, rounds=2)
    assert set(named(best)) == NEEDED and k == (0, 2) and dropped == 3
    _, sampled = edits.credit(JOINED, stand_in, key, k=2, rng=random.Random(0))
    assert len(sampled) == 2
