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


STEPS = '''def graph(tokens, targets):
    """The attention output at the last position reads the value written at "princess"."""
    t = targets[0]
    p = tokens.index(" princess")
    return [
        # the prediction reads an attention output that reads "princess"
        {(t, "<p:3.o.281>"): {p: "<p:3.v.676>"}, "out": "<p:3.o.281>"},
        # what that value reads, and an MLP path at the last position
        {(p, "<p:3.v.676>"): "<p:0.down.3473>", (t, "<p:2.down.773>"): "<p:2.fc.40>", "out": "<p:2.down.773><p:3.o.281>"},
    ]
'''


def test_ordered_steps():
    ir = mech._trace(STEPS, "vpd4l", BEHAVIOR)
    assert ir["valid"], ir["error"]
    g = ir["graph"]
    nodes = [tuple(n) for n in g["nodes"]]
    assert g["steps"] == 2 and g["explanation"].startswith("The attention output")
    assert g["notes"] == ["the prediction reads an attention output that reads \"princess\"", "what that value reads, and an MLP path at the last position"]
    first = {(nodes[r], nodes[w]) for (r, w), s in zip(g["parents"], g["parent_step"]) if s == 0}
    assert first == {((3, "o_proj", 3, 281), (3, "v_proj", 1, 676))}
    assert [nodes[w] for w, s in zip(g["out"], g["out_step"])] == [(3, "o_proj", 3, 281), (2, "down_proj", 3, 773)]  # a repeat is not a new edge
    assert g["out_step"] == [0, 1] and sorted(g["node_step"]) == [0, 0, 1, 1, 1]
    assert mech._trace(GRAPH, "vpd4l", BEHAVIOR)["graph"]["steps"] == 1  # a single dict is one step
    assert "a list of steps" in error(STEPS.replace("return [", "return 3 or ["))


def test_uses():
    ir = mech._trace(GRAPH.replace('"out": "<p:3.o.281><p:2.down.773>",', '"out": "<p:3.o.281><p:2.down.773>",\n        "uses": ["m0", "m3"],'), "vpd4l", BEHAVIOR)
    assert ir["valid"], ir["error"]
    assert ir["graph"]["uses"] == ["m0", "m3"]
    assert "uses" in error(GRAPH.replace('"out": "<p:3.o.281><p:2.down.773>",', '"out": "<p:3.o.281><p:2.down.773>",\n        "uses": [3],'))


def test_the_connection_rule():
    assert mech.connects(0, "down_proj", 1, 3, "v_proj", 1, [3])  # an MLP output into a later value at its position
    assert not mech.connects(3, "o_proj", 1, 3, "v_proj", 1, [3])  # not into its own layer's value (it is written after)
    assert mech.connects(1, "o_proj", 2, 1, "c_fc", 2, [3])  # an attention output into its own layer's MLP input
    assert not mech.connects(0, "down_proj", 1, 3, "v_proj", 2, [3])  # the residual stream is one position's
    assert mech.connects(3, "v_proj", 1, 3, "o_proj", 3, [3]) and not mech.connects(3, "v_proj", 3, 3, "o_proj", 1, [3])
    assert not mech.connects(2, "v_proj", 1, 3, "o_proj", 3, [3])  # values reach their own layer's attention output only
    assert mech.connects(2, "c_fc", 3, 2, "down_proj", 3, [3]) and not mech.connects(2, "c_fc", 3, 3, "down_proj", 3, [3])
    assert mech.connects(2, "down_proj", 3, None, None, None, [3]) and not mech.connects(2, "down_proj", 2, None, None, None, [3])


def dropped(source, behavior=BEHAVIOR):
    ir = mech._trace(source, "vpd4l", behavior)
    assert ir["valid"], ir["error"]
    return " | ".join(why for _, why in ir["graph"]["dropped"]), ir["graph"]


def test_a_graph_answer_says_what_is_wrong():
    """What is not a node or a connection the model has is left out of the graph and listed; an answer that is not in
    the format is invalid."""
    why, g = dropped(GRAPH.replace('(p, "<p:3.v.676>"): "<p:0.down.3473>"', '(p, "<p:3.v.676>"): "<p:3.down.3473>"'))
    assert "no such connection" in why and len(g["parents"]) == 3 and [3, "down_proj", 1, 3473] not in g["nodes"]  # the writer is not added
    why, g = dropped(GRAPH.replace('"out": "<p:3.o.281><p:2.down.773>"', '"out": "<p:2.fc.40><p:3.o.281>"'))
    assert "the prediction reads" in why and len(g["out"]) == 1
    assert "is not a subcomponent" in dropped(GRAPH.replace('["<p:2.fc.40>"]', '["<p:2.h.40>"]'))[0]
    assert "subcomponents 0.." in dropped(GRAPH.replace("<p:3.v.5>", "<p:3.v.4096>"))[0]
    assert "positions are 0..3" in dropped(GRAPH.replace("{p: ", "{9: "))[0]
    assert "a remainder is not a graph node" in dropped(GRAPH.replace("<p:3.v.5>", "<p:3.v.rest>"))[0]
    assert dropped(GRAPH)[0] == ""
    assert "in a string of subcomponents is not a subcomponent" in dropped(GRAPH.replace("<p:3.v.5>", "<p:3.x.5>"))[0]  # stray text is dropped, not fatal
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


ESCAPES = [  # known ways out of a restricted Python namespace; an answer is code the verifier runs, and RL searches for these
    "x = ().__class__.__bases__[0].__subclasses__()",  # object's subclasses (dunder attributes)
    "x = getattr((), 'x')",  # getattr is not a builtin here
    "x = type(())",  # nor type
    "x = __import__('os')",  # dunder name
    "import os",
    "from os import system",
    "x = vars()",
    "x = globals()",
    "x = locals()",
    "x = eval('1')",
    "x = exec('x = 1')",
    "x = open('/etc/passwd')",
    "x = compile('1', 'f', 'eval')",
    "x = breakpoint()",
    "x = '{0.__class__}'.format(1)",  # format is banned, and the string holds '__'
    "x = '{.real}'.format_map({})",
    "x = f'{(1).__class__}'",  # an f-string's expression is checked like any other
    "g = (i for i in [1])\nx = g.gi_frame",  # frames reach globals and builtins
    "def f():\n    yield 1\nx = f().gi_code",
    "x = (lambda: 0).__code__",
    "class A:\n    pass",  # class bodies are not allowed
    "@len\ndef f(): pass",  # decorators
    "x = [c for c in ().__class__.__mro__]",
    "x = 'a' + '_' * 2 + 'class' + '_' * 2",  # a dunder name built at run time stays a string: nothing reads attributes by name
    "with open('x') as f:\n    pass",
    "try:\n    pass\nexcept Exception:\n    pass",
    "async def f():\n    pass",
    "x = super()",
    "x = memoryview(b'')",
    "x = bytearray(1)",
]


def test_the_sandbox_rejects_known_escapes():
    """Each of these is invalid or harmless: it cannot import, read a frame, reach builtins beyond the safe set, or
    touch files. (The last string-building case is valid but inert.)"""
    for src in ESCAPES:
        ir = mech._trace(src + "\n" + GRAPH, "vpd4l", BEHAVIOR)
        harmless = src.startswith("x = 'a' + '_' * 2")  # valid, inert
        assert harmless or not ir["valid"], f"accepted: {src!r}"
