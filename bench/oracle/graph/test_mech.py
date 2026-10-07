"""Tests of the mech tracer: valid and invalid programs, Python token counts, sandbox escapes.

  ~/mpd-data/venv/bin/python -m pytest bench/oracle/graph/test_mech.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mech  # noqa: E402

HEAD = "from mech import node, edges, L, PD, embed, logits\n"


def invalid(source: str, model: str = "vpd4l", sandboxed: bool = False) -> str:
    ir = (mech.trace(source, model, timeout=3.0) if sandboxed else mech.trace_inline(source, model))
    assert not ir["valid"] and ir["nodes"] == [] and ir["edges"] == [], ir
    return ir["error"]


def test_examples_trace():
    import json

    index = json.loads((HERE / "examples/index.json").read_text())
    assert sorted(index) == sorted(p.stem for p in (HERE / "examples").glob("*.py"))
    for path in sorted((HERE / "examples").glob("*.py")):
        model = index[path.stem]["model"]
        source = path.read_text()
        ir = mech.trace_inline(source, model)
        assert ir["valid"], (path.name, ir["error"])
        assert ir["nodes"] and ir["edges"] and ir["python_tokens"] > 0
        assert mech.trace(source, model) == ir
        assert mech.english(source)


def test_ir_fields():
    ir = mech.trace_inline(HEAD + "a = node(L[1].head[1], L[1].head[3])\nb = node(PD.vpd[3].c_fc[2], PD.vpd[3].down_proj[5:8])\n"
                           "edges(a >> b, embed >> a.query, b >> logits)\nnode(L[0].mlp[9]) >> a.key\n", "vpd4l")
    assert ir["valid"], ir["error"]
    assert set(ir) == {"model", "nodes", "edges", "python_tokens", "token_types", "source", "valid", "error"}
    assert ir["nodes"] == [
        {"id": "a", "pieces": [{"view": "native", "layer": 1, "kind": "head", "index": [1, 3]}], "rule": None},
        {"id": "b", "pieces": [{"view": "vpd", "layer": 3, "kind": "c_fc", "index": 2},
                               {"view": "vpd", "layer": 3, "kind": "down_proj", "index": [5, 6, 7]}], "rule": None},
        {"id": "node0", "pieces": [{"view": "native", "layer": 0, "kind": "mlp", "index": 9}], "rule": None},
    ]
    assert ir["edges"] == [{"from": "a", "to": "b", "route": "input"}, {"from": "embed", "to": "a", "route": "query"},
                           {"from": "b", "to": "logits", "route": "input"}, {"from": "node0", "to": "a", "route": "key"}]


def test_qwen_views():
    ir = mech.trace_inline(HEAD + "f = node(PD.tc[14][163839, 7])\nh = node(L[20].head[15])\n"
                           "edges(f >> h.value, h >> logits)\n", "qwen3-0.6b")
    assert ir["valid"], ir["error"]
    assert ir["nodes"][0]["pieces"] == [{"view": "transcoder", "layer": 14, "kind": "feature", "index": [7, 163839]}]
    assert "out of range" in invalid(HEAD + "node(PD.tc[14][163840])", "qwen3-0.6b")
    assert "not available" in invalid(HEAD + "node(PD.vpd[1].c_fc[0])", "qwen3-0.6b")
    assert "not available" in invalid(HEAD + "node(PD.tc[1][0])", "vpd4l")
    assert "not available" in invalid(HEAD + "node(PD.lib[1].c_fc[0])", "vpd4l")
    assert "layers are 0..27" in invalid(HEAD + "node(L[28].head[0])", "qwen3-0.6b")


def test_invalid_programs():
    assert "layers are 0..3" in invalid(HEAD + "node(L[4].head[0])")
    assert "out of range" in invalid(HEAD + "node(L[1].head[6])")
    assert "out of range" in invalid(HEAD + "node(L[1].mlp[-1])")
    assert "out of range" in invalid(HEAD + "node(PD.vpd[0].q_proj[512])")
    assert "sites are" in invalid(HEAD + "node(PD.vpd[0].gate_proj[1])")
    assert "connects nothing" in invalid(HEAD + "a = node(L[3].head[0])\nb = node(L[1].head[0])\nedges(a >> b.key)")
    assert "connects nothing" in invalid(HEAD + "a = node(L[1].mlp[0])\nb = node(L[1].head[0])\nedges(a >> b)")
    assert "connects nothing" in invalid(HEAD + "a = node(PD.vpd[1].c_fc[0])\nb = node(L[2].mlp[0])\nedges(a >> b)")
    assert "no query read" in invalid(HEAD + "a = node(L[1].mlp[0])\nedges(embed >> a.query)")
    assert "cannot read itself" in invalid(HEAD + "a = node(L[1].head[0])\nedges(a >> a)")
    assert "two views" in invalid(HEAD + "a = node(L[1].head[0])\nb = node(PD.vpd[1].q_proj[3])")
    assert "nodes a and b" in invalid(HEAD + "a = node(L[1].mlp[0, 1])\nb = node(L[1].mlp[1])")
    assert "cannot write" in invalid(HEAD + "a = node(L[1].mlp[0])\nedges(logits >> a)")
    assert "is a read" in invalid(HEAD + "a = node(L[1].head[0])\nb = node(L[2].mlp[0])\nedges(a.key >> b)")
    assert "not an edge" in invalid(HEAD + "a = node(L[1].mlp[0])\nedges(a)")
    assert "line 2" in invalid(HEAD + "node(L[9].head[0])")
    assert "one layer's attention" in invalid(HEAD + "node(L[1].head[0], L[2].head[0])")
    assert "one layer's attention" in invalid(HEAD + "node(L[1].head[0], L[1].mlp[0])")
    assert "syntax error" in invalid(HEAD + "a = node(")
    assert "unknown model" in invalid(HEAD, model="gpt2")
    # a same-layer edge into an internal stream is legal
    assert mech.trace_inline(HEAD + "a = node(PD.vpd[1].c_fc[0])\nb = node(PD.vpd[1].down_proj[3])\nedges(a >> b)\n",
                             "vpd4l")["valid"]


def test_sandbox():
    assert "only `from mech import" in invalid("import os\n")
    assert "only `from mech import" in invalid("from os import system\n")
    assert "mech has no" in invalid("from mech import trace\n")
    assert "not allowed" in invalid("x = ().__class__\n")
    assert "not allowed" in invalid("x = __import__('os')\n")
    assert "not allowed" in invalid("g = (i for i in [1])\nf = g.gi_frame\n")
    assert "not allowed" in invalid("s = '{0.__class__}'.format(1)\n")
    assert "not allowed" in invalid("s = '{0}'.format(1)\n")
    assert "not allowed" in invalid("class A:\n    pass\n")
    assert "not allowed" in invalid("try:\n    x = 1\nexcept Exception:\n    pass\n")
    assert "not allowed" in invalid("with x:\n    pass\n")
    assert "not allowed" in invalid("global x\n")
    assert "not allowed" in invalid("del x\n")
    assert "not allowed" in invalid("x = [s for s in ['a__b']]\n")
    assert "not allowed" in invalid("x = f'{L.__class__}'\n")
    assert "NameError" in invalid("open('/etc/passwd')\n")
    assert "NameError" in invalid("getattr(1, 'real')\n")
    assert "NameError" in invalid("eval('1')\n")
    assert "NameError" in invalid("node(L[1].head[0])\n")  # used without importing
    assert "not allowed" in invalid("while True:\n    pass\n")
    assert "time limit" in invalid("for i in range(10 ** 12):\n    pass\n", sandboxed=True)
    assert "time limit" in invalid("x = 10\ny = x ** x ** x ** x\n", sandboxed=True)
    assert "memory limit" in invalid("n = 10 ** 10\nx = [0] * n\n", sandboxed=True)
    assert "memory limit" in invalid("n = 10 ** 9\nx = list(range(n))\n", sandboxed=True)


def test_code_length():
    source = 'from mech import node, L\na = node(L[1].head[12])  # free\n"""free too"""\ndef f():\n    """free"""\n'
    tokens, types = mech.code_length(source)
    # from mech import node , L | a = node ( L [ 1 ] . head [ 12 ] ) | def f ( ) :
    assert tokens == 6 + (13 + 2) + 5
    fixed = len(mech.KEYWORDS) + len(mech.OPERATORS) + len(mech.FIXED_NAMES) + len(mech.LITERAL_CHARACTERS)
    assert types == fixed + 2  # the program's own names: a, f
    assert mech.code_length('x = "abc"\n')[0] == 2 + 5
    assert mech.code_length("x = 'é'\n") == (2 + 3, fixed + 1 + 1)
    assert mech.code_length("a_very_long_identifier_name = 1\n")[0] == 3
    tokens_doc, _ = mech.code_length('x = 1\n"""a long docstring that costs nothing"""\n')
    assert tokens_doc == 3


def test_english():
    source = '"""Module doc."""\n# first comment\nx = 1  # trailing\ndef f():\n    """Function\n    doc."""\n'
    assert mech.english(source) == "Module doc.\nfirst comment\ntrailing\nFunction\ndoc."


def test_shapes_match_files():
    data = Path.home() / "mpd-data"
    if not (data / "engine/vpd4l/export.json").exists():
        return
    import json

    assert mech.build_shapes(data) == json.loads(mech.SHAPES_FILE.read_text())


def test_prompt():
    import prompt

    if not prompt.TOKENIZERS["qwen3-0.6b"].exists():
        return
    ids = prompt.tokenizer("qwen3-0.6b").encode(" red green blue . red green blue").ids
    behavior = {"id": "toy", "model": "qwen3-0.6b", "description": "Induction.", "prompts": [
        {"token_ids": ids, "target_positions": [len(ids) - 2], "model_top": [[[" blue", 0.9], [" red", 0.05]]],
         "counterfactual": None}]}
    text = prompt.render(behavior, prompts=1, shots=1)
    assert "' red green blue . red green' -> ' blue' 0.90, ' red' 0.05" in text
    assert "PD.tc[l][i, ...]" in text and "PD.vpd" not in text.split("Example program")[0]
    induction = dict(behavior, family="induction_random")
    assert "qwen3_induction" not in str(prompt.examples(induction, 3)) and len(prompt.examples(induction, 3)) == 2
    assert prompt.examples(behavior, 1)[0][0] == "qwen3_induction_heads"
    assert prompt.program_of("x\n```python\nfrom mech import node\n```\n") == "from mech import node\n"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
