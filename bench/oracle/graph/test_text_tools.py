"""Tests of the English tools with a stand-in generator (no model): rebuild.py and paraphrase.py.

  ~/mpd-data/venv/bin/python -m pytest bench/oracle/graph/test_text_tools.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mech  # noqa: E402
import paraphrase  # noqa: E402
import rebuild  # noqa: E402

NATIVE = (HERE / "examples/vpd4l_induction_native.py").read_text()


def upper(asks):
    return [a.split("<<<\n", 1)[1].rsplit("\n>>>", 1)[0].upper() for a in asks]


def test_paraphrase_keeps_code():
    for name, model in (("vpd4l_induction_native", "vpd4l"), ("vpd4l_induction_vpd", "vpd4l"),
                        ("qwen3_induction_heads", "qwen3-0.6b")):
        source = (HERE / "examples" / f"{name}.py").read_text()
        [new] = paraphrase.paraphrase([source], upper)
        before, after = mech.trace_inline(source, model), mech.trace_inline(new, model)
        assert after["valid"], after["error"]
        assert (before["nodes"], before["edges"], before["python_tokens"]) == (after["nodes"], after["edges"],
                                                                               after["python_tokens"])
        assert mech.english(new) == mech.english(source).upper()


def test_paraphrase_quotes():
    source = 'def f():\n    """a "quoted" \\\\ word\n    second line"""\nx = 1  # note\n'
    [new] = paraphrase.paraphrase([source], lambda asks: ['said """x""" \\ y\nnext' if "quoted" in a else "remark"
                                                         for a in asks])
    assert mech.english(new) == 'said """x""" \\ y\nnext\nremark'
    assert mech.code_length(new) == mech.code_length(source)


def test_rebuild():
    program = "from mech import node, edges, L, logits\nh = node(L[2].head[4])  # copies\nedges(h >> logits)\n"
    answers = []

    def generate(asks):
        answers.extend(asks)
        return ["Here it is:\n```python\n" + program + "```\n"]

    [r] = rebuild.rebuild([NATIVE], "vpd4l", generate)
    assert r["english"] in answers[0] and "L[1].head[1]" not in answers[0]
    assert r["source"] == program and r["ir"]["valid"] and r["ir"]["nodes"][0]["id"] == "h"
    assert 0 < r["overlap"]["pieces"] < 1 and 0 < r["overlap"]["edges"] < 1  # L2.H4 >> logits is shared
    [same] = rebuild.rebuild([NATIVE], "vpd4l", lambda asks: ["```python\n" + NATIVE + "```"])
    assert same["overlap"] == {"pieces": 1.0, "edges": 1.0}
