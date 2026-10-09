"""edits: parsing a format v4 explanation into nodes, edges and labels, one-edit neighbours, credit and refinement under
a stand-in score.

  ~/mpd-data/venv/bin/python -m pytest bench/oracle/graph/test_edits.py
"""

import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import edits  # noqa: E402
from edits import Answer, Edit  # noqa: E402

SOURCE = '''def marks(tokens):
    return ['"' in t for t in tokens]


nodes = {
    "mark": {"subcomponents": ["<p:0.fc.225>", "<p:0.down.663>"], "at": marks},
    "close": {"subcomponents": ["<p:2.v.80>", "<p:2.o.63>", "<p:3.down.885>"], "at": "targets"},
}
edges = [
    ("input", "mark"),
    ("mark", "close", "value"),
    ("close", "output"),
]
labels = {"mark": "inside"}
'''


def test_parse_and_source_round_trip():
    a = Answer.parse(SOURCE)
    assert [(s.line, s.variable, s.at) for s in a.statements] == [(5, "mark", "marks"), (6, "close", "targets")]
    assert a.statements[1].parts == ("<p:2.v.80>", "<p:2.o.63>", "<p:3.down.885>")
    assert a.edges == (("input", "mark"), ("mark", "close", "value"), ("close", "output")) and a.edge_lines == (9, 10, 11)
    assert a.labels == (("mark", "inside"),)
    assert a.source() == SOURCE
    assert Answer.parse("nodes = {1: 2}\nedges = []\n").statements == () and Answer.parse("def f(:\n").statements == ()


def test_edits():
    a = Answer.parse(SOURCE)
    dropped = edits.apply(a, Edit("drop", "close", part="<p:2.v.80>"))
    assert dropped.statements[1].parts == ("<p:2.o.63>", "<p:3.down.885>")
    gone = edits.apply(a, Edit("unalign", "mark"))
    assert [s.variable for s in gone.statements] == ["close"] and gone.edges == (("close", "output"),) and gone.labels == ()
    emptied = edits.apply(edits.apply(a, Edit("drop", "mark", part="<p:0.fc.225>")), Edit("drop", "mark", part="<p:0.down.663>"))
    assert emptied.source() == gone.source()
    cut = edits.apply(a, Edit("cut", "mark>close>value"))
    assert cut.edges == (("input", "mark"), ("close", "output"))
    added = edits.apply(a, Edit("add", "close", part="<p:3.fc.7>"))
    assert added.statements[1].parts[-1] == "<p:3.fc.7>"
    assert edits.adds(a, {"close": ["<p:2.v.80>", "<p:3.fc.7>", "<p:3.fc.8>"], "nope": ["<p:1.v.1>"]}, 1) == [Edit("add", "close", part="<p:3.fc.7>")]


def stand_in(sources):
    """A score: 10 bits per subcomponent and 5 per edge, 100 more without <p:2.o.63>, invalid without an output edge."""
    out = []
    for src in sources:
        a = Answer.parse(src)
        parts = [p for s in a.statements for p in s.parts]
        valid = any(r == "output" for _, r, *_ in a.edges)
        out.append({"total_bits": 10 * len(parts) + 5 * len(a.edges) + (0 if "<p:2.o.63>" in parts else 100), "valid": valid})
    return out


def test_credit_and_refine():
    a = Answer.parse(SOURCE)
    total, dS = edits.credit(a, stand_in, k=16, rng=random.Random(0))
    assert total == 65
    assert dS[Edit("drop", "close", part="<p:2.v.80>")] == -10 and dS[Edit("drop", "close", part="<p:2.o.63>")] == 90
    assert dS[Edit("cut", "close>output")] == float("inf") and dS[Edit("cut", "mark>close>value")] == -5
    best, bits, accepted = edits.refine(a, stand_in, rounds=6)
    assert [p for s in best.statements for p in s.parts] == ["<p:2.o.63>"] and best.edges == (("close", "output"),) and bits == 15
