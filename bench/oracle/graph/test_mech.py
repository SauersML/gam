"""mech, the loader of gate programs: on(tokens, targets) -> the checker's IR (one node per block and set of positions),
the errors it reports, and the sandbox.

  ~/mpd-data/venv/bin/python -m pytest bench/oracle/graph/test_mech.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mech  # noqa: E402

STRINGS = [["a", ' "', "b", "c"], ["d", "e", "f", "g"]]
BEHAVIOR = {"sequences": [([1, 2, 3, 4], STRINGS[0], [3]), ([5, 6, 7, 8], STRINGS[1], [3])]}

GATES = '''QUOTES = {'"', ' "'}


def on(tokens, targets):
    gates = {i: ["<p:0.fc.225>", "<p:0.down.663>"] for i, t in enumerate(tokens) if t in QUOTES}
    for i in targets:
        gates[i] = gates.get(i, []) + ["<p:2.v.80>", "<p:2.o.63>"]
    return gates
'''


def error(source, behavior=BEHAVIOR):
    ir = mech._trace(source, "vpd4l", behavior)
    assert not ir["valid"]
    return ir["error"]


def test_a_gate_program_names_what_acts_where():
    ir = mech._trace(GATES, "vpd4l", BEHAVIOR)
    assert ir["valid"], ir["error"]
    assert ir["wiring"] == "model" and ir["edges"] == [] and ir["standin"] is None
    mlp, attn = ir["nodes"]
    assert [p["kind"] for p in mlp["pieces"]] == ["c_fc", "down_proj"] and [p["kind"] for p in attn["pieces"]] == ["v_proj", "o_proj"]
    assert [s["positions"] for s in mlp["at"]] == [[1], []], "the quote, in the one sequence holding it"
    assert [s["positions"] for s in attn["at"]] == [[3], [3]] and mlp["at"][0]["tokens"] == [1, 2, 3, 4]
    nothing = mech._trace("def on(tokens, targets):\n    return {}\n", "vpd4l", BEHAVIOR)
    assert nothing["valid"] and nothing["nodes"] == [] and nothing["edges"] == [{"from": "embed", "to": "logits", "route": "input"}]


def test_a_string_of_names_is_a_list_of_them():
    joined = mech._trace('def on(tokens, targets):\n    return {3: "<p:2.v.80><p:2.o.63>", 0: ""}\n', "vpd4l", BEHAVIOR)
    listed = mech._trace('def on(tokens, targets):\n    return {3: ["<p:2.v.80>", "<p:2.o.63>"]}\n', "vpd4l", BEHAVIOR)
    assert joined["valid"] and joined["nodes"] == listed["nodes"]
    pairs = mech._trace('def on(tokens, targets):\n    return [(3, "<p:2.v.80>"), (3, "<p:2.o.63>")]\n', "vpd4l", BEHAVIOR)
    assert pairs["nodes"] == listed["nodes"]


def test_a_gate_program_says_what_is_wrong():
    assert "position 9" in error(GATES.replace("for i in targets:", "for i in [9]:"))
    assert "is not a subcomponent" in error(GATES.replace("<p:2.v.80>", "<p:2.x.80>"))
    assert "subcomponents 0.." in error(GATES.replace("<p:2.v.80>", "<p:2.v.4096>"))
    assert "holds only names" in error('def on(tokens, targets):\n    return {3: "<p:2.v.80> and more"}\n')
    assert "needs the behavior" in error(GATES, behavior=None)
    assert "defines no function on" in error("x = 1\n")
    assert "imports nothing" in error("import os\n" + GATES)


def test_the_sandbox_limits_and_the_server():
    assert "attribute" in error("x = ().__class__\n" + GATES)
    task = {"prompts": [{"token_ids": [1, 2, 3, 4], "target_positions": [3]}]}
    ir = mech.trace("while True:\n    pass\n" + GATES, "vpd4l", timeout=1, behavior=task)
    assert not ir["valid"] and "time limit" in ir["error"]
    ir = mech.trace(GATES, "vpd4l", behavior=task)
    assert ir["valid"] and [s["positions"] for s in ir["nodes"][-1]["at"]] == [[3]]
