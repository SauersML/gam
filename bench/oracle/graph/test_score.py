"""score.py against the checker binary on vpd4l (skipped without the binary or the export):

    ~/mpd-data/venv/bin/python -m pytest bench/oracle/graph/test_score.py

A behavior of eight induction prompts with their counterfactuals; the empty program, g-mech's induction
example and an invalid program are scored together and one at a time.
"""
import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score  # noqa: E402

SOURCE = Path.home() / "mpd-data/graph_oracle/behaviors/vpd4l/induction_random.words8.json"
TERMS = ["total_bits", "exec_error_bits", "complexity_bits", "structure_bits", "code_bits", "base_bits", "opaque_numbers", "python_tokens", "N", "experiments"]

pytestmark = pytest.mark.skipif(not (score.BINARY.exists() and score.EXPORTS["vpd4l"].exists() and SOURCE.exists()),
                                reason="no checker binary, vpd4l export or behavior file")


@pytest.fixture(scope="module")
def checker(tmp_path_factory):
    record = json.loads(SOURCE.read_text())
    record["prompts"] = record["prompts"][:8]
    path = tmp_path_factory.mktemp("behavior") / "induction8.json"
    path.write_text(json.dumps(record))
    with score.Checker("vpd4l") as c:
        answer = c.behavior(path)
        assert answer["site_experiments"] > 0, "the vpd4l manifest is the default"
        yield c


def test_batch_scores_every_term_and_equals_single(checker):
    example = ("from mech import node, edges, L, embed, logits\nprev = node(L[1].head[1])\nind = node(L[2].head[4])\n"
               "edges(embed >> prev, prev >> ind.key, embed >> ind, ind >> logits, embed >> logits)\n")  # no views: native
    programs = ["", example, "import os\nnode(L[9].head[0])\n"]
    together = checker.score_batch(programs, experiments=12, seed=5, reader=False)
    assert [s["valid"] for s in together] == [True, True, False]
    for s in together:
        assert all(isinstance(s[t], (int, float)) and math.isfinite(s[t]) for t in TERMS), s
        assert s["per_family"] and all(math.isfinite(f["mean_kl_bits"]) for f in s["per_family"].values())
    assert together[1]["opaque_numbers"] > together[0]["opaque_numbers"]
    # The behavior's half of the experiments is shared: the same families' draws for every program.
    alone = checker.score(example, experiments=12, seed=5, reader=False)
    assert abs(alone["exec_error_bits"] - together[1]["exec_error_bits"]) <= 1e-6 * max(1.0, alone["exec_error_bits"])


def test_reader_items(checker):
    answer = checker.score("", experiments=4, seed=1, reader=True, reader_top=5)
    items = answer.get("items")
    assert items, "the reader's items come back when no reader server is set"
    for it in items:
        assert len(it["candidates"]) == 5 and all(0.0 <= c["p"] <= 1.0 for c in it["candidates"])
        assert it["text"] and abs(sum(c["p"] for c in it["candidates"]) + it["other"] - 1.0) < 1e-6


def test_ir_carries_explanation():
    import types

    fake = types.SimpleNamespace(model="vpd4l", behavior_record=None, decomposition="vpd")
    ir = score.Checker.ir(fake, {"source": "from mech import L\n", "explanation": "L2.H4 copies the token."})
    assert ir["valid"] and ir["explanation"] == "L2.H4 copies the token."
    assert ir["explanation_tokens"] > 3 and ir["explanation_token_types"] > 150_000
    assert score.Checker.ir(fake, "from mech import L\n")["explanation"] == ""
    assert score.Checker.ir(fake, ir) is ir


def test_answer_is_not_interchange_tested():
    ir = {"answer": "answer", "alignments": [{"variable": "prev", "nodes": ["prev"], "pairs": [{"base": 0, "source": 1}]},
                                             {"variable": "answer", "nodes": ["a"], "pairs": [{"base": 0, "source": 1}]}]}
    got = score.output_aligned(ir)
    assert [a["pairs"] for a in got["alignments"]] == [[{"base": 0, "source": 1}], []]
    assert ir["alignments"][1]["pairs"]  # the caller's IR is left as it was
    plain = {"answer": None, "alignments": []}
    assert score.output_aligned(plain) is plain


def test_shared_base_joins_every_program(tmp_path):
    """A base of two layer-3 VPD subcomponents (c_fc and down_proj): the empty program is scored as the base alone
    (its parts priced in base_bits, outside total_bits); a program naming one of them takes it over."""
    vpd = Path.home() / "mpd-data/engine/vpd4l_decomposition"
    if not vpd.exists():
        pytest.skip("no vpd4l decomposition")
    record = json.loads(SOURCE.read_text())
    record["prompts"] = record["prompts"][:4]
    behavior = tmp_path / "induction4.json"
    behavior.write_text(json.dumps(record))
    base = {"model": "vpd4l", "nodes": [{"id": "base_m3", "pieces": [{"view": "vpd", "layer": 3, "kind": "c_fc", "index": [0]},
                                                                     {"view": "vpd", "layer": 3, "kind": "down_proj", "index": [0]}]}],
            "edges": [], "base": ["base_m3"], "valid": True}
    path = tmp_path / "base.json"
    path.write_text(json.dumps(base))
    with score.Checker("vpd4l", views={"vpd": vpd}, base=path) as c:
        assert c.behavior(behavior)["base"]["parts"] == 2
        empty = {"model": "vpd4l", "nodes": [], "edges": [], "python_tokens": 0, "token_types": 0, "valid": True}
        taking = {**empty, "nodes": [{"id": "m", "pieces": [{"view": "vpd", "layer": 3, "kind": "c_fc", "index": [0]}]}],
                  "edges": [{"from": "embed", "to": "m", "route": "input"}]}
        a, b = c.score_batch([empty, taking], experiments=2, reader=False, stand_in="delete")  # the base is deletion's
        assert a["valid"] and b["valid"] and a["base_bits"] > 0 and math.isclose(b["base_bits"], a["base_bits"] / 2, rel_tol=1e-9)
        assert math.isclose(a["total_bits"], a["exec_error_bits"] + a["complexity_bits"], rel_tol=1e-9)
