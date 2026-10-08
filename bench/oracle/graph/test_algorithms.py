"""The family algorithms (bench/oracle/graph/algorithms/, index.json: behavior family -> algorithm) on their
vpd4l and Qwen3-0.6B behaviors: each traces with its answer bound, and its answer is the prompt's next
token on nearly every target (greater-than's answer is one valid year of many, so it has no accuracy bar).

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


def test_algorithms_predict_their_behaviors():
    low = {}
    for model, answer in ANSWER.items():
        for path in sorted((BEHAVIORS / model).glob("*.json")):
            behavior = json.loads(path.read_text())
            if behavior["family"] in INDEX:
                check(behavior, model, answer, low)
    assert not low, low


def check(behavior, model, answer, low):
    source = (HERE / "algorithms" / f"{INDEX[behavior['family']]}.py").read_text() + f"\nbind(answer, {answer})\n"
    ir = mech.trace(source, model, behavior=behavior)
    assert ir["valid"], (model, behavior["id"], ir["error"])
    assert ir["bindings"][0]["pairs"], (model, behavior["id"])
    if behavior["family"] != "greater_than" and ir["algorithm_accuracy"] < 0.95:
        low[(model, behavior["id"])] = ir["algorithm_accuracy"]
