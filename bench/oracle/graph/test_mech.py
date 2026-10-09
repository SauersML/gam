"""mech, the loader of format v3 explanations: groups -> nodes and edges, behavior variables -> label tests, the sandbox
and the code length.

  ~/mpd-data/venv/bin/python -m pytest bench/oracle/graph/test_mech.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mech  # noqa: E402

QUOTE = '''groups = {
    "quote": {"subcomponents": ["<p:0.fc.225>", "<p:0.down.663>"], "reads": ["input"], "label": "inside"},
    "answer": {"subcomponents": ["<p:2.v.80>", "<p:2.o.63>", "<p:3.fc.1013>", "<p:3.down.885>"], "reads": ["quote", "input"], "writes": "output"},
}
'''

# Three prompts of one length: the first two have a changed prompt that changes "inside", the third one that does not.
BEHAVIOR = {"prompts": [["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]], "counterfactuals": [["a", "x", "c"], ["d", "y", "f"], ["g", "z", "i"]],
            "targets": [[2], [2], [2]], "varies": [["inside"], ["inside"], []], "variables": ["inside"]}


def edges(ir):
    return {(e["from"], e["to"], e["route"]) for e in ir["edges"]}


def test_groups_become_nodes_and_edges():
    ir = mech._trace(QUOTE, "vpd4l", BEHAVIOR)
    assert ir["valid"], ir["error"]
    assert [n["id"] for n in ir["nodes"]] == ["quote", "answer.2.attn", "answer.3.mlp"]
    assert ir["nodes"][0]["pieces"] == [{"view": "vpd", "layer": 0, "kind": "c_fc", "index": 225}, {"view": "vpd", "layer": 0, "kind": "down_proj", "index": 663}]
    assert edges(ir) == {("embed", "quote", "input"), ("quote", "answer.2.attn", "input"), ("quote", "answer.3.mlp", "input"),
                         ("embed", "answer.2.attn", "input"), ("embed", "answer.3.mlp", "input"), ("answer.2.attn", "answer.3.mlp", "input"),
                         ("answer.2.attn", "logits", "input"), ("answer.3.mlp", "logits", "input"), ("embed", "logits", "input")}
    assert ir["group_nodes"] == {"quote": ["quote"], "answer": ["answer.2.attn", "answer.3.mlp"]} and ir["labels"] == {"quote": "inside"}
    assert ir["python_tokens"] == 0  # the groups statement is structure, priced by the checker


def test_a_variable_is_tested_on_the_prompts_that_change_it():
    ir = mech._trace(QUOTE, "vpd4l", BEHAVIOR)
    assert ir["alignments"] == [{"variable": "inside", "nodes": ["quote"], "pairs": [{"base": 0, "changed": True}, {"base": 1, "changed": True}]}]
    alone = 'groups = {"answer": {"subcomponents": ["<p:2.v.80>", "<p:2.o.63>"], "reads": ["input"], "writes": "output"}}\n'
    unplaced = mech._trace(alone, "vpd4l", BEHAVIOR)
    assert unplaced["valid"] and unplaced["alignments"] == [{"variable": "inside", "nodes": [], "pairs": [{"base": 0, "changed": True}, {"base": 1, "changed": True}]}]


def error(source, behavior=BEHAVIOR):
    ir = mech._trace(source, "vpd4l", behavior)
    assert not ir["valid"]
    return ir["error"]


def test_invalid_explanations_say_why():
    assert "not a variable of this behavior" in error(QUOTE.replace('"label": "inside"', '"label": "outside"'))
    assert "carries a behavior variable" in error(QUOTE.replace(', "label": "inside"', ''))
    assert "carries no variable" in error('groups = {"answer": {"subcomponents": ["<p:2.v.80>", "<p:2.o.63>"], "reads": ["input"], '
                                          '"writes": "output", "label": "inside"}}\n')
    assert "no group writes the output" in error(QUOTE.replace(', "writes": "output"', ', "label": "other"'), {**BEHAVIOR, "variables": ["inside", "other"]})
    assert "belongs to one group" in error(QUOTE.replace('"<p:2.v.80>"', '"<p:0.fc.225>"'))
    assert "subcomponents 0.." in error(QUOTE.replace("<p:0.fc.225>", "<p:0.fc.99999>"))
    assert "is not a subcomponent" in error(QUOTE.replace("<p:0.fc.225>", "<p:0.h.2>"))
    assert "writers must come first" in error(QUOTE.replace('"reads": ["input"], "label"', '"reads": ["answer"], "label"'))
    assert "no o subcomponent" in error(QUOTE.replace('"<p:0.fc.225>", "<p:0.down.663>"', '"<p:1.v.4>"'))
    assert "imports nothing" in error("import os\n" + QUOTE)
    assert "defines no `groups`" in error("x = 1\n")


def test_the_sandbox_limits_and_the_server():
    assert "attribute" in error("x = ().__class__\n" + QUOTE)
    ir = mech.trace("while True:\n    pass\n" + QUOTE, "vpd4l", timeout=1)
    assert not ir["valid"] and "time limit" in ir["error"]
    ir = mech.trace(QUOTE, "vpd4l")
    assert ir["valid"] and ir["alignments"] == [{"variable": "inside", "nodes": ["quote"], "pairs": []}]


def test_code_beside_the_groups_is_charged():
    tokens, types = mech.code_length("# a note\nk = [1, 2]\n" + QUOTE)
    assert tokens == 1 + 1 + 1 + 1 + 1 + 1 + 1  # k = [ 1 , 2 ]: names and operators one token each, numbers per character
    assert types > len(mech.KEYWORDS) + len(mech.OPERATORS)
