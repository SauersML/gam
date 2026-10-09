"""mech, the loader of format v4 explanations: nodes (with positions), edges and labels -> the checker's IR, behavior
variables -> label tests, the sandbox and the code length.

  ~/mpd-data/venv/bin/python -m pytest bench/oracle/graph/test_mech.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mech  # noqa: E402

QUOTE = '''def marks(tokens):
    return ['"' in t for t in tokens]


nodes = {
    "mark": {"subcomponents": ["<p:0.fc.225>", "<p:0.down.663>"], "at": marks},
    "carry": {"subcomponents": ["<p:2.v.80>", "<p:2.o.63>"], "at": "targets"},
    "close": {"subcomponents": ["<p:3.fc.1013>", "<p:3.down.885>"]},
}
edges = [
    ("input", "mark"),
    ("mark", "carry", "value"),
    ("input", "carry"),
    ("carry", "close"),
    ("close", "output"),
]
labels = {"carry": "inside"}
'''

# Two prompts and their changed prompts: the first prompt's change changes "inside", the second's does not.
STRINGS = [["a", ' "', "b", "c"], ["d", "e", "f", "g"]]
CHANGED = [["a", " (", "b", "c"], ["d", "x", "f", "g"]]
BEHAVIOR = {"prompts": STRINGS, "counterfactuals": CHANGED, "targets": [[3], [3]], "varies": [["inside"], []],
            "variables": ["inside"],
            "sequences": [([1, 2, 3, 4], STRINGS[0], [3]), ([5, 6, 7, 8], STRINGS[1], [3]), ([1, 9, 3, 4], CHANGED[0], [3]), ([5, 10, 7, 8], CHANGED[1], [3])]}


def edges(ir):
    return {(e["from"], e["to"], e["route"]) for e in ir["edges"]}


def test_nodes_edges_and_positions():
    ir = mech._trace(QUOTE, "vpd4l", BEHAVIOR)
    assert ir["valid"], ir["error"]
    assert ir["standin"] == "counterfactual" and [n["id"] for n in ir["nodes"]] == ["mark", "carry", "close"]
    mark, carry, close = ir["nodes"]
    assert [s["positions"] for s in mark["at"]] == [[1], [], [], []], "marks: the quote token, in each sequence"
    assert [s["positions"] for s in carry["at"]] == [[3], [3], [3], [3]] and close["at"] == [], "targets; everywhere"
    assert mark["at"][0]["tokens"] == [1, 2, 3, 4]
    assert edges(ir) == {("embed", "mark", "input"), ("mark", "carry", "value"), ("embed", "carry", "input"), ("carry", "close", "input"),
                         ("close", "logits", "input"), ("embed", "logits", "input")}
    assert ir["labels"] == {"carry": "inside"} and ir["python_tokens"] > 0  # the position function is code


def test_a_variable_is_tested_on_the_prompts_that_change_it():
    ir = mech._trace(QUOTE, "vpd4l", BEHAVIOR)
    assert ir["alignments"] == [{"variable": "inside", "nodes": ["carry"], "pairs": [{"base": 0, "changed": True}]}]
    unlabeled = mech._trace(QUOTE.replace('labels = {"carry": "inside"}\n', ""), "vpd4l", BEHAVIOR)
    assert unlabeled["valid"] and unlabeled["alignments"] == [{"variable": "inside", "nodes": [], "pairs": [{"base": 0, "changed": True}]}]


def error(source, behavior=BEHAVIOR):
    ir = mech._trace(source, "vpd4l", behavior)
    assert not ir["valid"]
    return ir["error"]


def test_invalid_explanations_say_why():
    assert "not a variable of this behavior" in error(QUOTE.replace('{"carry": "inside"}', '{"carry": "outside"}'))
    assert "no node writes the output" in error(QUOTE.replace('    ("close", "output"),\n', ""))
    assert "connects nothing" in error(QUOTE.replace('("carry", "close")', '("close", "carry")'))
    assert "belongs to one node" in error(QUOTE.replace('"<p:2.v.80>"', '"<p:0.fc.225>"'))
    assert "subcomponents 0.." in error(QUOTE.replace("<p:0.fc.225>", "<p:0.fc.99999>"))
    assert "is not a subcomponent" in error(QUOTE.replace("<p:0.fc.225>", "<p:0.h.2>"))
    assert "a function the file defines" in error(QUOTE.replace('"at": marks', '"at": "middle"'))
    assert "returned" in error(QUOTE.replace("return ['\"' in t for t in tokens]", "return 3"))
    no_writer = QUOTE.replace('"<p:2.v.80>", "<p:2.o.63>"', '"<p:2.v.80>"').replace('    ("carry", "close"),\n', '    ("input", "close"),\n')
    assert "write nothing a swap can carry" in error(no_writer)
    assert "imports nothing" in error("import os\n" + QUOTE)
    assert "defines no `nodes`" in error("x = 1\n")


def test_the_sandbox_limits_and_the_server():
    assert "attribute" in error("x = ().__class__\n" + QUOTE)
    ir = mech.trace("while True:\n    pass\n" + QUOTE, "vpd4l", timeout=1)
    assert not ir["valid"] and "time limit" in ir["error"]
    ir = mech.trace(QUOTE, "vpd4l")
    assert ir["valid"] and ir["alignments"] == [{"variable": "inside", "nodes": ["carry"], "pairs": []}]


def test_the_structure_is_not_code():
    tokens, _ = mech.code_length(QUOTE)
    only, _ = mech.code_length("def marks(tokens):\n    return ['\"' in t for t in tokens]\n")
    assert tokens == only, "nodes, edges and labels are structure the checker prices"
