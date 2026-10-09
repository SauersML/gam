"""edits: dropping parents (edges) from a graph answer, credit and refinement under a stand-in score.

  ~/mpd-data/venv/bin/python -m pytest bench/oracle/graph/test_edits.py
"""

import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import edits  # noqa: E402

GRAPH = '''def graph(tokens, targets):
    return {
        (31, "<p:3.o.281>"): {3: "<p:3.v.676><p:3.v.5>"},
        (3, "<p:3.v.676>"): "<p:0.down.3473><p:1.down.9>",
        (31, "<p:2.down.773>"): ["<p:2.fc.40>", "<p:2.fc.41>"],
        "out": "<p:3.o.281><p:2.down.773>",
    }
'''
NEEDED = {"<p:3.v.676>", "<p:0.down.3473>", "<p:3.o.281>", "<p:2.fc.40>"}


def parents(source):
    keys = {m.group(1) for m in edits.KEY.finditer(source)}
    return [source[a:b] for a, b in edits.names(source)], keys


def stand_in(sources):
    """Excess 10 per needed parent missing, size 1 per parent: the key is (excess, size)."""
    out = []
    for s in sources:
        ps, _ = parents(s)
        out.append({"excess": 10 * len(NEEDED - set(ps)), "size": len(ps)})
    return out


def key(s):
    return (s["excess"], s["size"])


def test_parents_and_drops():
    ps, keys = parents(GRAPH)
    assert ps == ["<p:3.v.676>", "<p:3.v.5>", "<p:0.down.3473>", "<p:1.down.9>", "<p:2.fc.40>", "<p:2.fc.41>", "<p:3.o.281>", "<p:2.down.773>"]
    assert keys == {"<p:3.o.281>", "<p:3.v.676>", "<p:2.down.773>"}  # readers are keys, not edges
    spans = edits.names(GRAPH)
    one = edits.drop(GRAPH, [spans[1]])
    assert '{3: "<p:3.v.676>"}' in one
    lst = edits.drop(GRAPH, [spans[5]])
    assert '["<p:2.fc.40>"]' in lst
    both = edits.drop(GRAPH, spans[4:6])
    assert "[]" in both


def test_credit_signs_and_refinement():
    base, signs = edits.credit(GRAPH, stand_in, key)
    assert base == (0, 8)
    by = {GRAPH[a:b]: v for (a, b), v in signs.items()}
    assert {p: v for p, v in by.items() if p in NEEDED} == {p: 1 for p in NEEDED}
    assert all(v == -1 for p, v in by.items() if p not in NEEDED)
    best, k, dropped = edits.refine(GRAPH, stand_in, key, rounds=2)
    assert set(parents(best)[0]) == NEEDED and k == (0, 4) and dropped == 4
    assert re.search(r"def graph", best)
    _, sampled = edits.credit(GRAPH, stand_in, key, k=2, rng=random.Random(0))
    assert len(sampled) == 2
