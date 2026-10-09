"""edits: parsing a format v3 explanation into groups, one-edit neighbours, credit and refinement under a stand-in score.

  ~/mpd-data/venv/bin/python -m pytest bench/oracle/graph/test_edits.py
"""

import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import edits  # noqa: E402
from edits import Answer, Edit  # noqa: E402

SOURCE = '''groups = {
    "quote": {"subcomponents": ["<p:0.fc.225>", "<p:0.down.663>"], "reads": ["input"], "label": "inside"},
    "answer": {"subcomponents": ["<p:2.v.80>", "<p:2.o.63>", "<p:3.down.885>"], "reads": ["input", "quote"], "writes": "output"},
}
'''


def test_parse_and_source_round_trip():
    a = Answer.parse(SOURCE)
    assert [(s.line, s.variable, s.label, s.writes) for s in a.statements] == [(1, "quote", "inside", None), (2, "answer", None, "output")]
    assert a.statements[1].parts == ("<p:2.v.80>", "<p:2.o.63>", "<p:3.down.885>")
    assert a.source() == SOURCE
    assert Answer.parse("groups = {1: 2}\n").statements == () and Answer.parse("def f(:\n").statements == ()


def test_edits():
    a = Answer.parse(SOURCE)
    dropped = edits.apply(a, Edit("drop", "answer", part="<p:2.v.80>"))
    assert dropped.statements[1].parts == ("<p:2.o.63>", "<p:3.down.885>")
    gone = edits.apply(a, Edit("unalign", "quote"))
    assert [s.variable for s in gone.statements] == ["answer"] and gone.statements[0].reads == ("input",)
    emptied = edits.apply(edits.apply(a, Edit("drop", "quote", part="<p:0.fc.225>")), Edit("drop", "quote", part="<p:0.down.663>"))
    assert [s.variable for s in emptied.statements] == ["answer"] and emptied.statements[0].reads == ("input",)
    added = edits.apply(a, Edit("add", "answer", part="<p:3.fc.7>"))
    assert added.statements[1].parts[-1] == "<p:3.fc.7>"
    assert edits.adds(a, {"answer": ["<p:2.v.80>", "<p:3.fc.7>", "<p:3.fc.8>"], "nope": ["<p:1.v.1>"]}, 1) == [Edit("add", "answer", part="<p:3.fc.7>")]


def stand_in(sources):
    """A score: 10 bits per subcomponent, 100 more without <p:2.o.63>, invalid without an output group."""
    out = []
    for src in sources:
        a = Answer.parse(src)
        parts = [p for s in a.statements for p in s.parts]
        valid = any(s.writes == "output" for s in a.statements)
        out.append({"total_bits": 10 * len(parts) + (0 if "<p:2.o.63>" in parts else 100), "valid": valid})
    return out


def test_credit_and_refine():
    a = Answer.parse(SOURCE)
    total, dS = edits.credit(a, stand_in, k=16, rng=random.Random(0))
    assert total == 50
    assert dS[Edit("drop", "answer", part="<p:2.v.80>")] == -10 and dS[Edit("drop", "answer", part="<p:2.o.63>")] == 90
    assert dS[Edit("unalign", "answer")] == float("inf")  # no output group: invalid
    best, bits, accepted = edits.refine(a, stand_in, rounds=4)
    assert bits == 10 and [p for s in best.statements for p in s.parts] == ["<p:2.o.63>"]
