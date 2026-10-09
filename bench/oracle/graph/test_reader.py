"""The English reader and the English spans of a program: python -m pytest bench/oracle/graph/test_reader.py"""

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prompt  # noqa: E402
import reader  # noqa: E402

SOURCE = '''def graph(tokens, targets):
    """The last position reads "princess"."""
    return [
        # what reads "princess"
        {"out": "<p:3.o.281>"},
    ]
'''


def test_english_spans():
    spans = prompt.english_spans(SOURCE)
    assert [SOURCE[a:b] for a, b in spans] == ['"""The last position reads "princess"."""', '# what reads "princess"']


def test_reader_bits():
    """English that says position 3 matters saves bits on a flip at 3 and none elsewhere; no English saves 0."""
    def generate(prompts):
        out = []
        for q in prompts:
            told = "position 3 decides" in q and "position 3," in q
            p = 0.9 if told else 0.5
            out.append({"Yes": math.log(p), "No": math.log(1 - p)})
        return out

    r = reader.Reader(generate, lambda ids: [f"t{i}" for i in ids])
    task = {"prompts": [{"token_ids": [10, 11, 12, 13], "model_top": [[[" her", 0.5]]]}]}
    events = [{"position": 3, "old": 13, "new": 99, "flipped": True}, {"position": 1, "old": 11, "new": 98, "flipped": False}]
    good, empty = r.bits(task, ["position 3 decides the prediction", ""], events)
    assert abs(good - (1 - math.log2(1 / 0.9)) / 2) < 1e-9 and empty == 0.0
    assert r.bits(task, ["x"], []) == [None]
    assert reader.english({"explanation": "A.", "notes": ["b", ""]}) == "A.\nb"
