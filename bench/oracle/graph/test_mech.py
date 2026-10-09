"""mech, the loader of graph answers: graph(tokens, targets) -> nodes, parents and the prediction's reads; the
connection rule; the errors it reports; the sandbox.

  ~/mpd-data/venv/bin/python -m pytest bench/oracle/graph/test_mech.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mech  # noqa: E402

STRINGS = ["The", " princess", " lost", " her"]
BEHAVIOR = {"sequences": [([1, 2, 3, 4], STRINGS, [3])]}

GRAPH = '''def graph(tokens, targets):
    t = targets[0]
    p = tokens.index(" princess")
    return {
        (t, "<p:3.o.281>"): {p: "<p:3.v.676>", t: "<p:3.v.5>"},
        (p, "<p:3.v.676>"): "<p:0.down.3473>",
        (t, "<p:2.fc.40>"): "",
        (t, "<p:2.down.773>"): ["<p:2.fc.40>"],
        "out": "<p:3.o.281><p:2.down.773>",
    }
'''


def error(source, behavior=BEHAVIOR):
    ir = mech._trace(source, "vpd4l", behavior)
    assert not ir["valid"]
    return ir["error"]


def test_a_graph_answer():
    ir = mech._trace(GRAPH, "vpd4l", BEHAVIOR)
    assert ir["valid"], ir["error"]
    g = ir["graph"]
    nodes = [tuple(n) for n in g["nodes"]]
    assert set(nodes) == {(3, "o_proj", 3, 281), (3, "v_proj", 1, 676), (3, "v_proj", 3, 5), (0, "down_proj", 1, 3473),
                          (2, "c_fc", 3, 40), (2, "down_proj", 3, 773)}
    edges = {(nodes[r], nodes[w]) for r, w in g["parents"]}
    assert edges == {((3, "o_proj", 3, 281), (3, "v_proj", 1, 676)), ((3, "o_proj", 3, 281), (3, "v_proj", 3, 5)),
                     ((3, "v_proj", 1, 676), (0, "down_proj", 1, 3473)), ((2, "down_proj", 3, 773), (2, "c_fc", 3, 40))}
    assert {nodes[w] for w in g["out"]} == {(3, "o_proj", 3, 281), (2, "down_proj", 3, 773)}


def test_the_connection_rule():
    assert mech.connects(0, "down_proj", 1, 3, "v_proj", 1, [3])  # an MLP output into a later value at its position
    assert not mech.connects(3, "o_proj", 1, 3, "v_proj", 1, [3])  # not into its own layer's value (it is written after)
    assert mech.connects(1, "o_proj", 2, 1, "c_fc", 2, [3])  # an attention output into its own layer's MLP input
    assert not mech.connects(0, "down_proj", 1, 3, "v_proj", 2, [3])  # the residual stream is one position's
    assert mech.connects(3, "v_proj", 1, 3, "o_proj", 3, [3]) and not mech.connects(3, "v_proj", 3, 3, "o_proj", 1, [3])
    assert not mech.connects(2, "v_proj", 1, 3, "o_proj", 3, [3])  # values reach their own layer's attention output only
    assert mech.connects(2, "c_fc", 3, 2, "down_proj", 3, [3]) and not mech.connects(2, "c_fc", 3, 3, "down_proj", 3, [3])
    assert mech.connects(2, "down_proj", 3, None, None, None, [3]) and not mech.connects(2, "down_proj", 2, None, None, None, [3])


def test_a_graph_answer_says_what_is_wrong():
    assert "no such connection" in error(GRAPH.replace('(p, "<p:3.v.676>"): "<p:0.down.3473>"', '(p, "<p:3.v.676>"): "<p:3.down.3473>"'))
    assert "the prediction reads" in error(GRAPH.replace('"out": "<p:3.o.281><p:2.down.773>"', '"out": "<p:2.fc.40>"'))
    assert "holds only" in error(GRAPH.replace("<p:3.v.5>", "<p:3.x.5>"))
    assert "is not a subcomponent" in error(GRAPH.replace('["<p:2.fc.40>"]', '["<p:2.h.40>"]'))
    assert "subcomponents 0.." in error(GRAPH.replace("<p:3.v.5>", "<p:3.v.4096>"))
    assert "positions are 0..3" in error(GRAPH.replace("{p: ", "{9: "))
    assert "a remainder is not a graph node" in error(GRAPH.replace("<p:3.v.5>", "<p:3.v.rest>"))
    assert "needs the task" in error(GRAPH, behavior=None)
    assert "defines no function graph" in error("x = 1\n")
    assert "imports nothing" in error("import os\n" + GRAPH)


def test_the_sandbox_limits_and_the_server():
    assert "attribute" in error("x = ().__class__\n" + GRAPH)
    task = {"prompts": [{"token_ids": [1, 2, 3, 4], "target_positions": [3]}]}
    ir = mech.trace("while True:\n    pass\n" + GRAPH.replace('tokens.index(" princess")', "1"), "vpd4l", timeout=1, behavior=task)
    assert not ir["valid"] and "time limit" in ir["error"]
    ir = mech.trace(GRAPH.replace('tokens.index(" princess")', "1"), "vpd4l", behavior=task)
    assert ir["valid"] and len(ir["graph"]["nodes"]) == 6
