"""Tests of the mech tracer: the algorithm and its bindings and claims, part tokens, the generic PD
vocabulary, invalid programs, Python token counts, sandbox escapes.

  ~/mpd-data/venv/bin/python -m pytest bench/oracle/graph/test_mech.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mech  # noqa: E402

HEAD = "from mech import node, edges, L, PD, embed, logits\n"
BEHAVIOR = Path.home() / "mpd-data/graph_oracle/behaviors/vpd4l/induction_random.words8.json"
INDUCTION = '''from mech import bind, claim

def back(tokens):
    return [[t - 1] if t else [0] for t in range(len(tokens))]

def prev(tokens, back):
    return [tokens[js[0]] if t else None for t, js in enumerate(back)]

def match(tokens, prev):
    return [[j for j in range(t) if prev[j] == tokens[t]] for t in range(len(tokens))]

def answer(tokens, match):
    return [tokens[js[-1]] if js else None for js in match]

claim(back, <p:1.q.316>, <p:1.k.329>)
bind(prev, <p:1.v.228>, <p:1.v.346>, <p:1.o.311>, <p:1.o.340>)
claim(match, <p:2.q.335>, <p:2.k.206>)
bind(answer, <p:2.v.559>, <p:2.o.735>, <p:3.v.677>, <p:3.o.806>)
'''
TOY = {"id": "toy", "model": "vpd4l", "prompts": [  # token ids decode to themselves through a fake table
    {"token_ids": [1, 2, 3, 1, 2], "target_positions": [3], "counterfactual": {"token_ids": [1, 4, 3, 1, 4]}},
    {"token_ids": [5, 6, 7, 5, 6], "target_positions": [3], "counterfactual": {"token_ids": [5, 8, 7, 5, 8]}},
    {"token_ids": [1, 6, 3, 1, 6], "target_positions": [3], "counterfactual": {"token_ids": [1, 2, 3, 1, 2]}},
]}


def invalid(source: str, model: str = "vpd4l", sandboxed: bool = False, **kw) -> str:
    ir = (mech.trace(source, model, timeout=3.0, **kw) if sandboxed else mech.trace_inline(source, model, **kw))
    assert not ir["valid"] and ir["nodes"] == [] and ir["edges"] == [], ir
    return ir["error"]


def toy_payload():
    payload = {"prompts": [[f"t{i}" for i in p["token_ids"]] for p in TOY["prompts"]],
               "counterfactuals": [[f"t{i}" for i in p["counterfactual"]["token_ids"]] for p in TOY["prompts"]],
               "targets": [p["target_positions"] for p in TOY["prompts"]]}
    return payload, {f"t{i}": i for i in range(10)}


def toy_trace(source: str) -> dict:
    payload, known = toy_payload()
    return mech._answer_ids(mech._trace(source, "vpd4l", payload), "vpd4l", known)


def test_examples_trace():
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


def test_algorithm_ir():
    ir = mech.trace_inline(INDUCTION, "vpd4l")
    assert ir["valid"], ir["error"]
    assert "standin" not in ir and ir["decomposition"] == "vpd" and ir["answer"] == "answer"
    nodes = {n["id"]: n for n in ir["nodes"]}
    assert sorted(nodes) == ["answer.2.attn", "answer.3.attn", "back", "match", "prev"]
    assert nodes["prev"]["pieces"] == [{"view": "vpd", "layer": 1, "kind": "v_proj", "index": [228, 346]},
                                       {"view": "vpd", "layer": 1, "kind": "o_proj", "index": [311, 340]}]
    assert nodes["back"]["claim"] is None and ir["claims"] == {"back": "back", "match": "match"}  # no behavior
    edges = {(e["from"], e["to"]) for e in ir["edges"]}
    assert edges == {("embed", "back"), ("embed", "prev"), ("back", "prev"), ("embed", "match"), ("prev", "match"),
                     ("embed", "answer.2.attn"), ("embed", "answer.3.attn"), ("match", "answer.2.attn"),
                     ("answer.2.attn", "answer.3.attn"), ("answer.2.attn", "logits"), ("answer.3.attn", "logits"),
                     ("embed", "logits")}
    assert all(e["route"] == "input" for e in ir["edges"])
    assert ir["bindings"] == [{"variable": "prev", "nodes": ["prev"], "pairs": []},
                              {"variable": "answer", "nodes": ["answer.2.attn", "answer.3.attn"], "pairs": []}]
    assert [(v["name"], v["role"], v["reads"]) for v in ir["variables"]] == [
        ("back", "claimed", ["tokens"]), ("prev", "bound", ["tokens", "back"]),
        ("match", "claimed", ["tokens", "prev"]), ("answer", "bound", ["tokens", "match"])]
    assert ir["python_tokens"] == mech.code_length(mech.quote_parts(INDUCTION))[0]


def test_algorithm_on_a_behavior():
    ir = toy_trace(INDUCTION)
    assert ir["valid"], ir["error"]
    assert ir["algorithm_accuracy"] == 1.0
    pairs = {b["variable"]: b["pairs"] for b in ir["bindings"]}
    # the source is the next prompt of the base's length under which the answer changes (prompt 1's t6 is
    # prompt 2's answer too, so prompt 1 takes prompt 0's)
    assert pairs["answer"] == [{"base": 0, "source": 1, "answer_text": ["t6"], "answer": [6]},
                               {"base": 1, "source": 0, "answer_text": ["t2"], "answer": [2]},
                               {"base": 2, "source": 0, "answer_text": ["t2"], "answer": [2]}]
    # prev into prompt 0: from prompt 1 nothing matches t1 (no answer), from prompt 2 t1 precedes position 1
    assert pairs["prev"][0] == {"base": 0, "source": 2, "answer_text": ["t2"], "answer": [2]}
    claimed = {n["id"]: n["claim"] for n in ir["nodes"] if n["claim"]}
    assert claimed["back"]["op"] == "pattern" and len(claimed["back"]["prompts"]) == 3
    assert claimed["back"]["prompts"][0] == [[1.0], [1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0],
                                             [0.0, 0.0, 0.0, 1.0, 0.0]]
    assert claimed["match"]["prompts"][0][3] == [0.0, 1.0, 0.0, 0.0]  # t1 at 3: position 1 follows t1
    assert claimed["match"]["prompts"][0][2] == [1.0, 0.0, 0.0]  # no match: position 0
    assert len(claimed["match"]["counterfactuals"]) == 3


def test_algorithm_on_a_real_behavior():
    if not BEHAVIOR.exists():
        return
    ir = mech.trace(INDUCTION, "vpd4l", behavior=BEHAVIOR)
    assert ir["valid"], ir["error"]
    assert ir["algorithm_accuracy"] == 1.0
    assert ir == mech.trace_inline(INDUCTION, "vpd4l", behavior=json.loads(BEHAVIOR.read_text()))
    record = json.loads(BEHAVIOR.read_text())
    for b in ir["bindings"]:
        assert len(b["pairs"]) > 48
        for pair in b["pairs"]:
            assert len(record["prompts"][pair["base"]]["token_ids"]) == len(record["prompts"][pair["source"]]["token_ids"])
            assert len(pair["answer"]) == len(record["prompts"][pair["base"]]["target_positions"])


def test_answers_start_with_a_token():
    bad = INDUCTION.replace("return [tokens[js[-1]] if js else None for js in match]",
                            "return ['' if js else None for js in match]")
    assert "starts with no token" in toy_trace(bad)["error"]
    longer = INDUCTION.replace("return [tokens[js[-1]] if js else None for js in match]",
                               "return [tokens[js[-1]] + ' and more' if js else None for js in match]")
    if BEHAVIOR.exists():  # a longer string's first token is the answer
        ir = mech.trace_inline(longer, "vpd4l", behavior=BEHAVIOR)
        assert ir["valid"] and ir["algorithm_accuracy"] == 1.0, ir["error"]


def test_answer_sets():
    sets = INDUCTION.replace("return [tokens[js[-1]] if js else None for js in match]",
                             "return [[tokens[js[-1]], 't9'] if js else None for js in match]")
    ir = toy_trace(sets)
    assert ir["valid"] and ir["algorithm_accuracy"] == 1.0, ir["error"]
    pair = next(b for b in ir["bindings"] if b["variable"] == "answer")["pairs"][0]
    assert pair["answers"] == [[6, 9]] and pair["answer"] == [6] and pair["answer_text"] == [["t6", "t9"]]
    plain = toy_trace(INDUCTION)
    assert all("answers" not in p for b in plain["bindings"] for p in b["pairs"])
    assert "collection of them" in toy_trace(INDUCTION.replace("return [tokens[js[-1]] if js else None",
                                                                "return [[1, 2] if js else None"))["error"]


def test_causal_answers():
    peek = INDUCTION.replace("return [tokens[js[-1]] if js else None for js in match]",
                             "return [tokens[t + 1] if t + 1 < len(tokens) else None for t in range(len(tokens))]")
    ir = toy_trace(peek)
    assert ir["valid"] and ir["algorithm_accuracy"] == 0.0  # the prompt is cut after the target


def test_binding_errors():
    def bad(text):
        return toy_trace(text)["error"]

    base = "from mech import bind, claim\n"
    assert "write no residual" in bad(base + "def a(tokens):\n    return tokens\nbind(a, <p:2.q.3>)\n")
    assert "names no variable" in bad(base + "def a(x):\n    return x\nbind(a, <p:2.o.3>)\n")
    assert "plain parameter" in bad(base + "def a(tokens=1):\n    return tokens\nbind(a, <p:2.o.3>)\n")
    assert "read each other" in bad(base + "def a(b):\n    return b\ndef b(a):\n    return a\nbind(a, <p:2.o.3>)\n")
    assert "defines with def" in bad(base + "bind(lambda tokens: tokens, <p:2.o.3>)\n")
    assert "are all unread" in bad(base + "def a(tokens):\n    return tokens\ndef b(tokens):\n    return tokens\n"
                                   "bind(a, <p:2.o.3>)\nbind(b, <p:3.o.3>)\n")
    assert "bind the answer" in bad(base + "def a(tokens):\n    return [[0]] * len(tokens)\nclaim(a, <p:2.q.3>)\n")
    assert "not both" in bad(base + "def a(tokens):\n    return tokens\nclaim(a, <p:2.q.3>)\nbind(a, <p:2.o.3>)\n")
    assert "q_proj and k_proj parts" in bad(base + "def p(tokens):\n    return [[0]] * len(tokens)\n"
                                      "def a(tokens, p):\n    return tokens\nclaim(p, <p:2.v.3>)\nbind(a, <p:2.o.3>)\n")
    assert "already carries the claim" in bad(base + "def p(tokens):\n    return [[0]] * len(tokens)\n"
                                              "def r(tokens):\n    return [[0]] * len(tokens)\n"
                                              "def a(tokens, p, r):\n    return tokens\nbind(a, <p:2.q.1>, <p:2.v.3>, <p:2.o.3>)\n"
                                              "claim(p, <p:2.q.1>)\nclaim(r, <p:2.q.1>)\n")
    assert "attention parts" in bad(base + "def p(tokens):\n    return [[0]] * len(tokens)\n"
                                    "def a(tokens, p):\n    return tokens\nclaim(p, <p:2.fc.3>)\nbind(a, <p:2.o.3>)\n")
    assert "no part of b writes" in bad(base + "def b(tokens):\n    return tokens\ndef a(tokens, b):\n    return b\n"
                                        "bind(b, <p:3.v.1>, <p:3.o.1>)\nbind(a, <p:2.v.3>, <p:2.o.3>)\n")
    assert "3 values for 4 positions" in bad(base + "def a(tokens):\n    return tokens[:-1]\nbind(a, <p:2.v.3>, <p:2.o.3>)\n")
    assert "ZeroDivisionError" in bad(base + "def a(tokens):\n    return [1 / 0] * len(tokens)\nbind(a, <p:2.v.3>, <p:2.o.3>)\n")
    assert "not a position" in bad(base + "def p(tokens):\n    return [[t + 1] for t in range(len(tokens))]\n"
                                   "def a(tokens, p):\n    return tokens\nclaim(p, <p:2.q.3>)\nbind(a, <p:2.v.3>, <p:2.o.3>)\n")
    assert "a part belongs to one node" in bad("from mech import bind, node\ndef a(tokens):\n    return tokens\n"
                                               "bind(a, <p:2.v.3>, <p:2.o.3>)\nn = node(<p:2.o.3>)\n")


def test_claims_on_bound_parts_and_steps():
    # a claim on parts already bound to the answer sits on the answer's node; `copy` is an unbound step
    src = ("from mech import bind, claim\n"
           "def match(tokens):\n    return [[j for j in range(t) if tokens[j] == tokens[t]] or [0] for t in range(len(tokens))]\n"
           "def copy(tokens, match):\n    return [tokens[js[-1] + 1] if js[-1] + 1 <= t else None for t, js in enumerate(match)]\n"
           "def answer(copy):\n    return copy\n"
           "bind(answer, <p:2.q.1>, <p:2.k.2>, <p:2.v.3>, <p:2.o.4>)\nclaim(match, <p:2.q.1>, <p:2.k.2>)\n")
    ir = toy_trace(src)
    assert ir["valid"], ir["error"]
    assert [n["id"] for n in ir["nodes"]] == ["answer"] and ir["nodes"][0]["claim"]["op"] == "pattern"
    assert {(e["from"], e["to"]) for e in ir["edges"]} == {("embed", "answer"), ("answer", "logits"), ("embed", "logits")}
    assert [v["role"] for v in ir["variables"]] == ["claimed", "step", "bound"]
    weights = ("from mech import bind, claim\n"
               "def p(tokens):\n    return [{0: 1, t: 3} for t in range(len(tokens))]\n"
               "def a(tokens, p):\n    return tokens\nclaim(p, <p:2.q.3>)\nbind(a, <p:2.v.3>, <p:2.o.3>)\n")
    ir = toy_trace(weights)
    assert ir["valid"], ir["error"]
    assert ir["nodes"][1]["claim"]["prompts"][0][2] == [1.0, 0.0, 3.0]


def test_part_tokens():
    ir = mech.trace_inline(HEAD + 'a = node(<p:2.q.3>, "<p:2.q.5>", PD[2].q_proj[7], <p:2.q.rest>)\n'
                           "edges(embed >> a)  # <p:0.q.0> in a comment stays text\n", "vpd4l")
    assert ir["valid"], ir["error"]
    assert ir["nodes"][0]["pieces"] == [{"view": "vpd", "layer": 2, "kind": "q_proj", "index": [3, 5, 7]},
                                        {"view": "vpd", "layer": 2, "kind": "q_proj", "index": "rest"}]
    assert mech.quote_parts('x = "<p:1.q.2>"  # <p:1.q.3>\ny = <p:1.fc.4>\n') == 'x = "<p:1.q.2>"  # <p:1.q.3>\ny = "<p:1.fc.4>"\n'
    assert "not a part token" in invalid(HEAD + 'node("<p:2.z.3>")')
    assert "out of range" in invalid(HEAD + "node(<p:2.q.512>)")
    assert "layers are 0..3" in invalid(HEAD + "node(<p:4.q.1>)")
    tokens = {"<p:2.v.559>": "PD[2].v_proj[559]", "<p:1.down.7>": "PD[1].down_proj[7]", "<p:0.q.rest>": "PD[0].q_proj.rest"}
    for token, name in tokens.items():  # a part token and its text spelling name the same part
        a, b = (mech.trace_inline(HEAD + f"n = node({x})\n", "vpd4l")["nodes"] for x in (token, name))
        assert a == b and a
    qwen = mech.trace_inline(HEAD + "f = node(<p:14.mlp.163839>, PD[14].mlp[7])\nh = node(<p:20.h.15>)\n"
                             "a = node(<p:3.a>)\nedges(f >> h.value, h >> logits)\n", "qwen3-0.6b")
    assert qwen["valid"], qwen["error"]
    assert qwen["nodes"][0]["pieces"] == [{"view": "transcoder", "layer": 14, "kind": "feature", "index": [7, 163839]}]
    assert qwen["nodes"][2]["pieces"] == [{"view": "native", "layer": 3, "kind": "head", "index": None}]
    native = mech.trace_inline(HEAD + "m = node(<p:2.m>)\n", "qwen3-1.7b")
    assert native["valid"] and native["decomposition"] is None
    assert native["nodes"][0]["pieces"] == [{"view": "native", "layer": 2, "kind": "mlp", "index": None}]


def test_piece_tokens_and_names():
    program = mech._Program("vpd4l", "vpd", {})
    mech._PROGRAM = program
    try:
        assert mech._part("<p:2.v.559>").tokens() == ["<p:2.v.559>"] and mech._part("<p:2.v.559>").name() == "PD[2].v_proj[559]"
        assert mech.PD[1].c_fc[3, 4].tokens() == ["<p:1.fc.3>", "<p:1.fc.4>"]
        assert mech.PD[0].k_proj.rest.tokens() == ["<p:0.k.rest>"] and mech.PD[0].k_proj.rest.name() == "PD[0].k_proj.rest"
    finally:
        mech._PROGRAM = None
    mech._PROGRAM = mech._Program("qwen3-0.6b", "transcoder", {})
    try:
        assert mech._part("<p:3.h.2>").tokens() == ["<p:3.h.2>"] and mech.L[3].attn.whole().tokens() == ["<p:3.a>"]
        assert mech.PD[5].mlp[9].tokens() == ["<p:5.mlp.9>"] and mech.PD[5].mlp[9].name() == "PD[5].mlp[9]"
    finally:
        mech._PROGRAM = None


def test_part_tokens_match_part_tokens_module():
    try:
        import part_tokens
    except ImportError:
        return
    for token in ("<p:2.v.559>", "<p:0.fc.3>", "<p:3.h.4>", "<p:1.a>", "<p:2.m>", "<p:1.q.rest>"):
        try:
            address = part_tokens.address_of(token)
        except ValueError:
            continue  # a form part_tokens does not have yet
        assert part_tokens.token_of(address) == token


def test_coverage():
    assert "decomposed by vpd" in invalid(HEAD + "node(L[1].head[0])")
    assert "decomposed by vpd" in invalid(HEAD + "node(<p:1.h.0>)")
    assert "decomposed by vpd" in invalid(HEAD + "node(L[1].mlp[3])")
    assert "decomposed by transcoder" in invalid(HEAD + "node(L[1].mlp)", "qwen3-0.6b")
    assert "no decomposition is attached" in invalid(HEAD + "node(PD[1].c_fc[0])", "qwen3-1.7b")
    assert "no decomposition is attached" in invalid(HEAD + "node(PD[1].c_fc[0])", decomposition="native")
    native = mech.trace_inline(HEAD + "h = node(L[1].head[0, 2])\nm = node(L[1].mlp[4])\nedges(h >> m, m >> logits)\n",
                               "vpd4l", decomposition="native")
    assert native["valid"] and native["decomposition"] is None, native["error"]


def test_library_view():
    ir = mech.trace_inline(HEAD + "a = node(PD[1].attn[3, 470])\nb = node(PD[2].attn[3])\n"
                           "c = node(<p:1.attn.3>)\nm = node(PD[2].mlp[562])\n"
                           "edges(a >> b.key, a >> m, m >> logits)\n", "vpd4l", decomposition="library")
    assert ir["valid"], ir["error"]  # parts may overlap between nodes
    assert ir["nodes"][0]["pieces"] == [{"view": "library", "layer": 1, "kind": "attn", "index": [3, 470]}]
    assert "out of range" in invalid(HEAD + "node(PD[1].attn[471])", decomposition="library")
    assert "sites are attn, mlp" in invalid(HEAD + "node(PD[1].c_fc[0])", decomposition="library")
    assert "sites are mlp" in invalid(HEAD + "node(PD[1].attn[0])", "qwen3-0.6b")
    assert "no library decomposition exists" in invalid(HEAD + "node(PD[1].mlp[0])", "qwen3-0.6b", decomposition="library")


def test_whole_site():
    ir = mech.trace_inline(HEAD + "m = node(L[3].mlp)\nh = node(L[2].head[0:6])\nedges(h >> m, m >> logits)\n",
                           "vpd4l", decomposition="native")
    assert ir["valid"], ir["error"]
    assert [p["index"] for n in ir["nodes"] for p in n["pieces"]] == [None, None]
    v = mech.trace_inline(HEAD + "v = node(PD[1].c_fc[0:3072])\n", "vpd4l")
    assert v["nodes"][0]["pieces"][0]["index"] is None


def test_low_level_ir():
    ir = mech.trace_inline(HEAD + "a = node(PD[1].q_proj[1], PD[1].o_proj[3])\nb = node(PD[3].c_fc[2], PD[3].down_proj[5:8])\n"
                           "edges(a >> b, embed >> a.query, b >> logits)\nnode(PD[0].down_proj[9], PD[0].c_fc[1]) >> a.input\n",
                           "vpd4l")
    assert ir["valid"], ir["error"]
    assert ir["nodes"] == [
        {"id": "a", "pieces": [{"view": "vpd", "layer": 1, "kind": "q_proj", "index": 1},
                               {"view": "vpd", "layer": 1, "kind": "o_proj", "index": 3}], "claim": None},
        {"id": "b", "pieces": [{"view": "vpd", "layer": 3, "kind": "c_fc", "index": 2},
                               {"view": "vpd", "layer": 3, "kind": "down_proj", "index": [5, 6, 7]}], "claim": None},
        {"id": "node0", "pieces": [{"view": "vpd", "layer": 0, "kind": "down_proj", "index": 9},
                                   {"view": "vpd", "layer": 0, "kind": "c_fc", "index": 1}], "claim": None},
    ]
    assert ir["edges"] == [{"from": "a", "to": "b", "route": "input"}, {"from": "embed", "to": "a", "route": "query"},
                           {"from": "b", "to": "logits", "route": "input"}, {"from": "node0", "to": "a", "route": "input"}]
    assert ir["bindings"] == [] and ir["answer"] is None


def test_tracer_speed():
    import time

    mech.trace("from mech import L\n", "vpd4l")
    start = time.time()
    for _ in range(20):
        assert mech.trace("from mech import node\na = node(<p:1.q.0>)\n", "vpd4l")["valid"]
    assert (time.time() - start) / 20 < 1.0  # a fork each; generous for a loaded machine


def test_forgiving_forms():
    ir = mech.trace_inline("import mech as m\nfrom mech import node as n, PD\na = n([PD[1].o_proj[0], PD[1].o_proj[2]])\n"
                           "b = m.node(m.PD[2].c_fc[4])\nm.edges([a >> b])\n", "vpd4l")
    assert ir["valid"], ir["error"]
    assert ir["nodes"][0]["pieces"][0]["index"] == [0, 2] and len(ir["edges"]) == 1
    assert "a layer has .head" in invalid(HEAD + "node(L[1].heads[0])")
    assert "or `import mech`" in invalid("import os, mech\n")


def test_tracer_after_fork():
    import os

    assert mech.trace("from mech import L\n", "vpd4l")["valid"]
    pid = os.fork()
    if pid == 0:  # the child must not share the parent's tracer server
        ok = mech.trace("from mech import node\na = node(<p:1.q.0>)\n", "vpd4l")["valid"]
        os._exit(0 if ok else 1)
    assert mech.trace("from mech import node\nb = node(<p:2.q.1>)\n", "vpd4l")["nodes"][0]["id"] == "b"
    assert os.waitpid(pid, 0)[1] == 0


def test_invalid_programs():
    assert "layers are 0..3" in invalid(HEAD + "node(PD[4].q_proj[0])")
    assert "out of range" in invalid(HEAD + "node(PD[1].q_proj[-1])")
    assert "out of range" in invalid(HEAD + "node(PD[0].q_proj[512])")
    assert "sites are" in invalid(HEAD + "node(PD[0].gate_proj[1])")
    assert "connects nothing" in invalid(HEAD + "a = node(PD[3].o_proj[0])\nb = node(PD[1].k_proj[0])\nedges(a >> b.key)")
    assert "connects nothing" in invalid(HEAD + "a = node(PD[1].down_proj[0])\nb = node(PD[1].q_proj[0])\nedges(a >> b)")
    assert "no key read" in invalid(HEAD + "a = node(PD[1].c_fc[0])\nedges(embed >> a.key)")
    assert "cannot read itself" in invalid(HEAD + "a = node(PD[1].c_fc[0], PD[1].down_proj[0])\nedges(a >> a)")
    assert "nodes a and b" in invalid(HEAD + "a = node(PD[1].c_fc[0, 1])\nb = node(PD[1].c_fc[1])")
    assert "nodes a and b" in invalid(HEAD + "a = node(PD[2].q_proj.rest)\nb = node(PD[2].q_proj.rest)")
    assert "only a VPD site" in invalid(HEAD + "node(PD[1].mlp.rest)", decomposition="library")
    assert "cannot write" in invalid(HEAD + "a = node(PD[1].c_fc[0])\nedges(logits >> a)")
    assert "is a read" in invalid(HEAD + "a = node(PD[1].q_proj[0])\nb = node(PD[2].c_fc[0])\nedges(a.query >> b)")
    assert "make it a node" in invalid(HEAD + "a = node(PD[2].c_fc[0])\nedges(PD[1].o_proj[0] >> a)")
    assert "not an edge" in invalid(HEAD + "a = node(PD[1].c_fc[0])\nedges(a)")
    assert "line 2" in invalid(HEAD + "node(PD[9].q_proj[0])")
    assert "one layer's attention" in invalid(HEAD + "node(PD[1].q_proj[0], PD[2].q_proj[0])")
    assert "one layer's attention" in invalid(HEAD + "node(PD[1].q_proj[0], PD[1].c_fc[0])")
    assert "syntax error" in invalid(HEAD + "a = node(")
    assert "unknown model" in invalid(HEAD, model="gpt2")
    assert "unknown decomposition" in invalid(HEAD, decomposition="sae")
    # a same-layer edge into an internal stream is legal
    assert mech.trace_inline(HEAD + "a = node(PD[1].c_fc[0])\nb = node(PD[1].down_proj[3])\nedges(a >> b)\n",
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
    assert "not allowed" in invalid("from mech import bind\ndef f(tokens):\n    return tokens.__class__\n")
    assert "NameError" in invalid("open('/etc/passwd')\n")
    assert "NameError" in invalid("getattr(1, 'real')\n")
    assert "NameError" in invalid("eval('1')\n")
    assert "NameError" in invalid("node(PD[1].q_proj[0])\n")  # used without importing
    assert "time limit" in invalid("while True:\n    pass\n", sandboxed=True)
    assert "time limit" in invalid("for i in range(10 ** 12):\n    pass\n", sandboxed=True)
    assert "time limit" in invalid("x = 10\ny = x ** x ** x ** x\n", sandboxed=True)
    for big in ("n = 10 ** 10\nx = [0] * n\n", "n = 10 ** 10\nx = 'ab' * n\n"):
        error = invalid(big, sandboxed=True)  # killed at the footprint limit, or malloc refuses first
        assert "memory limit" in error or "MemoryError" in error, error
    slow = INDUCTION.replace("def back(tokens):\n", "def back(tokens):\n    for i in range(10 ** 12):\n        pass\n")
    assert "time limit" in invalid(slow, sandboxed=True, behavior=BEHAVIOR) if BEHAVIOR.exists() else True


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
    # a part token is one token and one type: n = node ( <p> , <p> , <p> )
    parts = mech.quote_parts("n = node(<p:1.q.2>, <p:1.q.2>, <p:1.k.3>)\n")
    assert mech.code_length(parts) == (10, fixed + 1 + 2)


def test_english():
    source = '"""Module doc."""\n# first comment\nx = 1  # trailing\ndef f():\n    """Function\n    doc."""\n'
    assert mech.english(source) == "Module doc.\nfirst comment\ntrailing\nFunction\ndoc."
    assert mech.english("# uses <p:1.q.2>\nn = node(<p:1.q.2>)\n") == "uses <p:1.q.2>"


def test_prompt():
    import prompt

    ids = mech.tokenizer("qwen3-0.6b").encode(" red green blue . red green blue").ids
    behavior = {"id": "toy", "model": "qwen3-0.6b", "description": "Induction.", "prompts": [
        {"token_ids": ids, "target_positions": [len(ids) - 2], "model_top": [[[" blue", 0.9], [" red", 0.05]]],
         "counterfactual": None}]}
    text = prompt.render(behavior, prompts=1, shots=1, decomposition="transcoder")
    assert "' red green blue . red green' -> ' blue' 0.90, ' red' 0.05" in text
    reference = text.split("Example answer")[0]
    assert "<p:L.mlp.I>" in reference and "<p:L.h.I>" in reference and "PD" not in reference
    assert "from mech import bind, claim" in reference and "nobody else reads them" in reference
    assert "<p:L.S.rest>" in prompt.render(dict(behavior, model="vpd4l"), prompts=0, shots=0)
    induction = dict(behavior, family="induction_random")
    assert all(not e["family"].startswith("induction") and e["split"] == "train" for _, e, _, _ in prompt.examples(induction, 9))
    name, entry, source, explanation = prompt.examples(behavior, 1)[0]
    assert name == "qwen3_induction_bind" and "```" not in explanation and explanation in text
    answer = "notes\n```python\nfrom mech import bind\nbind(a, <p:2.h.4>)  # copies\n```\n\nExplanation: L2.H4 copies it.\n"
    assert prompt.split_answer(answer) == ("from mech import bind\nbind(a, <p:2.h.4>)  # copies\n", "L2.H4 copies it.")
    assert prompt.program_of("x\n```python\nfrom mech import node\n```\n") == "from mech import node\n"
    assert prompt.explanation_of("no code") == "" and prompt.program_of("no code") == "no code"
    broken = "```python\nx = (\n```\ntext"  # no block parses: the whole answer, no explanation
    assert prompt.split_answer(broken) == (broken, "")


def test_shapes_match_files():
    data = Path.home() / "mpd-data"
    if not (data / "engine/vpd4l/export.json").exists():
        return
    assert mech.build_shapes(data) == json.loads(mech.SHAPES_FILE.read_text())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
