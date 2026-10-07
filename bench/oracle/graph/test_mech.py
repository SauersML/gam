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
    assert set(ir) == {"model", "standin", "nodes", "edges", "python_tokens", "token_types", "source", "valid",
                       "error"}
    assert ir["standin"] == "counterfactual"
    assert ir["nodes"] == [
        {"id": "a", "pieces": [{"view": "native", "layer": 1, "kind": "head", "index": [1, 3]}], "rule": None},
        {"id": "b", "pieces": [{"view": "vpd", "layer": 3, "kind": "c_fc", "index": 2},
                               {"view": "vpd", "layer": 3, "kind": "down_proj", "index": [5, 6, 7]}], "rule": None},
        {"id": "node0", "pieces": [{"view": "native", "layer": 0, "kind": "mlp", "index": 9}], "rule": None},
    ]
    assert ir["edges"] == [{"from": "a", "to": "b", "route": "input"}, {"from": "embed", "to": "a", "route": "query"},
                           {"from": "b", "to": "logits", "route": "input"}, {"from": "node0", "to": "a", "route": "key"}]


def test_whole_site():
    ir = mech.trace_inline(HEAD + "m = node(L[3].mlp)\nh = node(L[2].head[0:6])\nv = node(PD.vpd[1].c_fc[0:3072])\n"
                           "edges(h >> m, m >> logits)\n", "vpd4l")
    assert ir["valid"], ir["error"]
    assert [p["index"] for n in ir["nodes"] for p in n["pieces"]] == [None, None, None]


def test_standin_and_attn():
    ir = mech.trace_inline(HEAD.replace("logits", "logits, standin") + "standin('position')\nh = node(L[2].attn)\n",
                           "vpd4l")
    assert ir["valid"] and ir["standin"] == "position", ir["error"]
    assert ir["nodes"][0]["pieces"] == [{"view": "native", "layer": 2, "kind": "head", "index": None}]
    assert "choose one of" in invalid("from mech import standin\nstandin('mean')\n")
    assert "once per program" in invalid("from mech import standin\nstandin('global')\nstandin('position')\n")


def test_tracer_speed():
    import time

    mech.trace("from mech import L\n", "vpd4l")
    start = time.time()
    for _ in range(20):
        assert mech.trace("from mech import node, L\na = node(L[1].head[0])\n", "vpd4l")["valid"]
    assert (time.time() - start) / 20 < 1.0  # a fork each; generous for a loaded machine


def test_library_view():
    ir = mech.trace_inline(HEAD + "a = node(PD.lib[1].attn[3, 470])\nb = node(PD.lib[2].attn[3])\n"
                           "c = node(PD.lib[1].attn[3])\nm = node(PD.lib[2].mlp[562])\n"
                           "edges(a >> b.key, a >> m, m >> logits)\n", "vpd4l")
    assert ir["valid"], ir["error"]  # parts may overlap between nodes
    assert ir["nodes"][0]["pieces"] == [{"view": "library", "layer": 1, "kind": "attn", "index": [3, 470]}]
    assert "out of range" in invalid(HEAD + "node(PD.lib[1].attn[471])")
    assert "two views" in invalid(HEAD + "a = node(PD.lib[1].mlp[0])\nb = node(L[1].mlp[0])")


def test_rules():
    head = "from mech import node, edges, L, embed, logits, attend, tokens, shift\n"
    ir = mech.trace_inline(head + "prev = node(L[1].head[1], rule=attend(offset=1))\n"
                           "ind = node(L[2].head[4], rule=attend(query=tokens, key=shift(tokens, 1)))\n"
                           "sink = node(L[0].head[2], rule=attend(first=True))\n"
                           "edges(embed >> prev.value, prev >> ind.value, ind >> logits, embed >> sink)\n", "vpd4l")
    assert ir["valid"], ir["error"]
    assert [n["rule"] for n in ir["nodes"]] == [
        {"op": "attend", "offset": 1},
        {"op": "attend", "query": {"op": "tokens"}, "key": {"op": "shift", "arg": {"op": "tokens"}, "by": 1}},
        {"op": "attend", "first": True}]
    assert "replaces its key" in invalid(head + "a = node(L[1].head[0])\nb = node(L[2].head[1], rule=attend(offset=2))\n"
                                         "edges(a >> b.key)")
    assert "native heads" in invalid(head + "node(L[1].mlp[0], rule=attend(offset=1))")
    assert "exactly one" in invalid(head + "attend(offset=1, first=True)")
    assert "positive int" in invalid(head + "shift(tokens, 0)")
    assert "attention rule such as" in invalid(head + "node(L[1].head[0], rule=1)")


def test_forgiving_forms():
    ir = mech.trace_inline("import mech as m\nfrom mech import node as n, L\na = n([L[1].head[0], L[1].head[2]])\n"
                           "b = m.node(m.L[2].mlp[4])\nm.edges([a >> b, b >> m.logits])\n", "vpd4l")
    assert ir["valid"], ir["error"]
    assert ir["nodes"][0]["pieces"][0]["index"] == [0, 2] and len(ir["edges"]) == 2
    assert "a layer has .head" in invalid(HEAD + "node(L[1].heads[0])")
    assert "or `import mech`" in invalid("import os, mech\n")


def test_tracer_after_fork():
    import os

    assert mech.trace("from mech import L\n", "vpd4l")["valid"]
    pid = os.fork()
    if pid == 0:  # the child must not share the parent's tracer server
        ok = mech.trace("from mech import node, L\na = node(L[1].head[0])\n", "vpd4l")["valid"]
        os._exit(0 if ok else 1)
    assert mech.trace("from mech import node, L\nb = node(L[2].head[1])\n", "vpd4l")["nodes"][0]["id"] == "b"
    assert os.waitpid(pid, 0)[1] == 0


def test_program_from_patch(tmp_path):
    import json
    import subprocess

    patch = {"behavior": "toy.a", "base_bits": 4.0,
             "heads": [{"layer": l, "head": h, "recovery_bits": 2.0 if (l, h) == (2, 4) else 0.1 if (l, h) == (1, 1) else 0.0}
                       for l in range(4) for h in range(6)],
             "mlps": [{"layer": l, "recovery_bits": [3.9, 0.5, 0.0, 1.0][l]} for l in range(4)]}
    neurons = {"behavior": "toy.a", "layer": 0, "base_bits": 4.0, "order": list(range(3071, -1, -1)),
               "curve": [{"k": 4, "kl_bits": 1.0, "total": 1.1}], "best": {"k": 4}}
    behavior = {"id": "toy.a", "model": "vpd4l", "model_accuracy": 0.9, "description": "A toy."}
    for name, obj in (("p", patch), ("n", neurons), ("b", behavior)):
        (tmp_path / f"{name}.json").write_text(json.dumps(obj))
    run = lambda *extra: subprocess.run(  # noqa: E731
        [sys.executable, str(HERE / "examples/measure/program_from_patch.py"), str(tmp_path / "p.json"),
         str(tmp_path / "b.json"), *extra], capture_output=True, text=True, check=True).stdout
    plain = mech.trace_inline(run(), "vpd4l")
    assert plain["valid"] and [n["id"] for n in plain["nodes"]] == ["mlp0", "mlp1", "h2_4", "mlp3"]
    priced = mech.trace_inline(run("--neurons", str(tmp_path / "n.json"), "--head-price", "0.28", "--mlp-price", "3.375"),
                               "vpd4l")
    assert priced["valid"], priced["error"]
    assert [n["id"] for n in priced["nodes"]] == ["mlp0", "h2_4"]  # whole MLPs 1 and 3 do not pay
    assert priced["nodes"][0]["pieces"][0]["index"] == [3068, 3069, 3070, 3071]
    (tmp_path / "t.json").write_text(json.dumps({"behavior": "toy.a", "base_bits": 4.0, "layers": {
        "3": [{"feature": 9, "recovery_bits": 0.0001}, {"feature": 5, "recovery_bits": 0.5}, {"feature": 7, "recovery_bits": 0.2}]}}))
    behavior["model"] = "qwen3-0.6b"
    (tmp_path / "b.json").write_text(json.dumps(behavior))
    tc = mech.trace_inline(run("--tc", str(tmp_path / "t.json")), "qwen3-0.6b")
    assert tc["valid"], tc["error"]
    assert tc["nodes"][-1] == {"id": "tc3", "pieces": [{"view": "transcoder", "layer": 3, "kind": "feature", "index": [5, 7]}],
                               "rule": None}


def test_qwen_views():
    ir = mech.trace_inline(HEAD + "f = node(PD.tc[14][163839, 7])\nh = node(L[20].head[15])\n"
                           "edges(f >> h.value, h >> logits)\n", "qwen3-0.6b")
    assert ir["valid"], ir["error"]
    assert ir["nodes"][0]["pieces"] == [{"view": "transcoder", "layer": 14, "kind": "feature", "index": [7, 163839]}]
    assert "out of range" in invalid(HEAD + "node(PD.tc[14][163840])", "qwen3-0.6b")
    assert "not available" in invalid(HEAD + "node(PD.vpd[1].c_fc[0])", "qwen3-0.6b")
    assert "not available" in invalid(HEAD + "node(PD.tc[1][0])", "vpd4l")
    assert "not available" in invalid(HEAD + "node(PD.lib[1].mlp[0])", "qwen3-0.6b")
    assert "library parts are" in invalid(HEAD + "node(PD.lib[1].c_fc[0])", "vpd4l")
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
    assert "make it a node" in invalid(HEAD + "a = node(L[2].mlp[0])\nedges(L[1].head[0] >> a)")
    assert "not an edge" in invalid(HEAD + "a = node(L[1].mlp[0])\nedges(a)")
    assert "line 2" in invalid(HEAD + "node(L[9].head[0])")
    assert "one layer's attention" in invalid(HEAD + "node(L[1].head[0], L[2].head[0])")
    assert "nodes a and b" in invalid(HEAD + "a = node(L[1].mlp)\nb = node(L[1].mlp[7])")
    assert "one layer's attention" in invalid(HEAD + "node(L[1].head[0], L[1].mlp[0])")
    assert "syntax error" in invalid(HEAD + "a = node(")
    assert "unknown model" in invalid(HEAD, model="gpt2")
    # a same-layer edge into an internal stream is legal
    assert mech.trace_inline(HEAD + "a = node(PD.vpd[1].c_fc[0])\nb = node(PD.vpd[1].down_proj[3])\nedges(a >> b)\n",
                             "vpd4l")["valid"]


def test_sandbox():
    assert "only `from mech import" in invalid("import os\n")
    assert "only `from mech import" in invalid("def f():\n    import os\n")
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
    for big in ("n = 10 ** 10\nx = [0] * n\n", "n = 10 ** 10\nx = 'ab' * n\n"):
        error = invalid(big, sandboxed=True)  # killed at the footprint limit, or malloc refuses first
        assert "memory limit" in error or "MemoryError" in error, error


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
    shots = prompt.examples(induction, 9)
    assert shots and all(not e["family"].startswith("induction") and e["split"] == "train" for _, e, _ in shots)
    assert prompt.examples(behavior, 1)[0][0] == "qwen3_induction_heads"
    assert prompt.program_of("x\n```python\nfrom mech import node\n```\n") == "from mech import node\n"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
