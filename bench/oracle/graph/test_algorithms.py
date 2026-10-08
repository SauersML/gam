"""The family algorithms (bench/oracle/graph/algorithms/, index.json: behavior family -> algorithm) on their
vpd4l and Qwen3-0.6B behaviors: each traces with its answer aligned, and its answer (or one of its set of
answers) is the prompt's next token on nearly every target.

  ~/mpd-data/venv/bin/python -m pytest bench/oracle/graph/test_algorithms.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mech  # noqa: E402

BEHAVIORS = Path.home() / "mpd-data/graph_oracle/behaviors"
ANSWER = {"vpd4l": "<p:3.v.0>, <p:3.o.0>", "qwen3-0.6b": "<p:27.h.0>"}  # any parts that write the residual
INDEX = json.loads((HERE / "algorithms/index.json").read_text())


def test_every_algorithm_is_indexed():
    assert sorted(set(INDEX.values())) == sorted(p.stem for p in (HERE / "algorithms").glob("*.py"))


def test_no_heldout_family_has_an_algorithm():
    # held-out families (behaviors/build.py's split) never become training answers
    sys.path.insert(0, str(HERE / "behaviors"))
    import build
    import teacher

    assert [f for f in INDEX if build.split_of(f) != "train"] == []
    for split in ("heldout", None):
        try:
            teacher.algorithm_of({"id": "x", "family": "induction_random", "split": split})
        except ValueError as e:
            assert "held-out" in str(e)
        else:
            raise AssertionError(f"algorithm_of gave an algorithm to a behavior of split {split}")
    assert "def answer" in teacher.algorithm_of({"id": "x", "family": "induction_random", "split": "train"})


def test_algorithms_predict_their_behaviors():
    low = {}
    for model, answer in ANSWER.items():
        for path in sorted((BEHAVIORS / model).glob("*.json")):
            behavior = json.loads(path.read_text())
            if behavior["family"] in INDEX:
                check(behavior, model, answer, low)
    assert not low, low


def check(behavior, model, answer, low):
    source = (HERE / "algorithms" / f"{INDEX[behavior['family']]}.py").read_text() + f"\nalign(answer, {answer})\n"
    ir = mech.trace(source, model, behavior=behavior)
    assert ir["valid"], (model, behavior["id"], ir["error"])
    assert ir["bindings"][0]["pairs"], (model, behavior["id"])
    if ir["algorithm_accuracy"] < 0.95:
        low[(model, behavior["id"])] = ir["algorithm_accuracy"]


def test_teacher_assignments():
    import teacher

    path = BEHAVIORS / "vpd4l/induction_random.words8.json"
    if not path.exists():
        return
    behavior = json.loads(path.read_text())
    search = {"source": "from mech import node, edges, PD, embed, logits\n"
                        "va1 = node(<p:1.q.316>, <p:1.k.329>, <p:1.v.228>, <p:1.o.311>)\n"
                        "va2 = node(<p:2.q.335>, <p:2.k.206>, <p:2.v.559>, <p:2.o.735>)\n"
                        "va3 = node(<p:3.v.677>, <p:3.o.806>)\nedges(embed >> va1, va1 >> va2, va2 >> va3, va3 >> logits)\n"}
    ir = teacher.search_ir(search, "vpd4l")
    assert teacher.patterns(teacher.algorithm_of(behavior), behavior) == {"back", "match"}
    found = teacher.assignments(ir, teacher.algorithm_of(behavior), behavior)
    tails = [s.split("\n\n\n")[-1] for s in found]
    # answer alone; prev = layer 1 and answer = layers 2-3; prev = layers 1-2 and answer = layer 3
    assert len(found) == 3 and all(t.count("align(answer") == 1 for t in tails)
    assert "align(prev, <p:1.q.316>, <p:1.k.329>, <p:1.v.228>, <p:1.o.311>)\nclaim(back, <p:1.q.316>, <p:1.k.329>)" in tails[1]
    traced = mech.trace_inline(found[1], "vpd4l", behavior=behavior)
    assert traced["valid"] and [b["variable"] for b in traced["bindings"]] == ["prev", "answer"]
