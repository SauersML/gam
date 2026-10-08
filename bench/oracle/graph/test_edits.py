"""edits.py on a stand-in score: parsing keeps the source, credit signs, refine reaches the optimum."""

import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from edits import Answer, Edit, adds, apply, credit, refine  # noqa: E402

SOURCE = '''"""doc"""
from mech import align, claim


def prev(tokens):
    # the token before each position
    return [tokens[t - 1] if t else None for t in range(len(tokens))]


def answer(tokens, prev):
    return prev


claim(prev, <p:1.q.316>, <p:1.k.329>)
align(prev, <p:1.v.228>, <p:1.o.311>)
align(answer, <p:2.v.559>, <p:2.o.735>, <p:3.o.806>)
'''

NEEDED = {"<p:1.v.228>", "<p:1.o.311>", "<p:2.v.559>", "<p:2.o.735>", "<p:2.v.9>"}


def stand_in(sources):
    """Total bits: 10 per needed part missing, 1 per part named (a part's name), invalid without the answer."""
    out = []
    for src in sources:
        a = Answer.parse(src)
        named = {p for s in a.statements for p in s.parts}
        valid = any(s.variable == "answer" for s in a.statements)
        out.append({"total_bits": 10.0 * len(NEEDED - named) + len(named), "valid": valid})
    return out


def test_parse_keeps_the_source():
    a = Answer.parse(SOURCE)
    assert a.source() == SOURCE
    assert [s.variable for s in a.statements] == ["prev", "prev", "answer"]


def test_edits():
    a = Answer.parse(SOURCE)
    dropped = apply(a, Edit("drop", "answer", "align", "<p:3.o.806>"))
    assert "<p:3.o.806>" not in dropped.source() and "<p:2.o.735>" in dropped.source()
    gone = apply(a, Edit("unalign", "prev", "claim"))
    assert "claim(" not in gone.source() and "align(prev" in gone.source()
    added = apply(a, Edit("add", "fresh", "align", "<p:0.v.1>"))
    assert added.source().rstrip().endswith("align(fresh, <p:0.v.1>)")


def test_credit_signs():
    a = Answer.parse(SOURCE)
    s, dS = credit(a, stand_in, k=100, rng=random.Random(0))
    assert dS[Edit("drop", "answer", "align", "<p:3.o.806>")] == -1  # not needed: dropping saves its name
    assert dS[Edit("drop", "answer", "align", "<p:2.v.559>")] == 9  # needed
    assert dS[Edit("unalign", "answer", "align")] == float("inf")  # the answer must stay aligned


def test_refine_reaches_the_optimum():
    a = Answer.parse(SOURCE)
    best, s, accepted = refine(a, stand_in, {"answer": ["<p:0.m.0>", "<p:2.v.9>"]}, rounds=8, per_variable=2)
    named = {p for st in best.statements for p in st.parts}
    assert named == NEEDED, named
    assert s == len(NEEDED)


def test_adds_go_to_the_align_statement():
    a = Answer.parse("align(answer, <p:2.v.559>)\nclaim(answer, <p:1.q.3>)\n")
    edit = adds(a, {"answer": ["<p:2.v.9>"]}, 1)[0]
    assert edit.kind == "align", edit
    assert apply(a, edit).statements[0].parts == ("<p:2.v.559>", "<p:2.v.9>")
