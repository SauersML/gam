"""score.py against the published checker binary on vpd4l (skipped without the binary, the export or the behavior):

    ~/mpd-data/venv/bin/python -m pytest bench/oracle/graph/test_score.py

Twelve closing-quote prompts with their changed prompts (each changes the variable inside), scored on the behavior's
answers with everything left out deleted: a graph whose middle node carries inside, the same graph without the label,
the same nodes at the target positions only, nothing named, and an invalid explanation.
"""
import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "e2e"))
import explain  # noqa: E402
import score  # noqa: E402

SOURCE = Path.home() / "mpd-data/graph_oracle/behaviors_v3/vpd4l/quote_close.said.json"
TERMS = ["total_bits", "exec_error_bits", "necessity_error_bits", "alignment_error_bits", "complexity_bits", "structure_bits", "N"]
GRAPH = '''nodes = {
    "read": {"subcomponents": ["<p:0.fc.225>", "<p:0.fc.2863>", "<p:0.down.663>", "<p:0.down.1224>"], "at": "all"},
    "carry": {"subcomponents": ["<p:2.v.80>", "<p:2.v.289>", "<p:2.o.63>", "<p:2.o.255>"], "at": "all"},
}
edges = [
    ("input", "read"),
    ("read", "carry"),
    ("input", "carry"),
    ("carry", "output"),
    ("read", "output"),
]
labels = {"carry": "inside"}
'''

pytestmark = pytest.mark.skipif(not (score.BINARY.exists() and score.EXPORTS["vpd4l"].exists() and SOURCE.exists()),
                                reason="no checker binary, vpd4l export or behavior file")


@pytest.fixture(scope="module")
def checker(tmp_path_factory):
    record = json.loads(SOURCE.read_text())
    record["prompts"] = record["prompts"][:12]
    path = tmp_path_factory.mktemp("behavior") / "quote12.json"
    path.write_text(json.dumps(record))
    with score.Checker("vpd4l", device="gpu") as c:
        c.behavior(path)
        yield c


def test_batch_scores_every_term_and_equals_single(checker):
    unlabeled = GRAPH.replace('labels = {"carry": "inside"}\n', "")
    targets = GRAPH.replace('"at": "all"', '"at": "targets"')
    programs = [GRAPH, unlabeled, targets, explain.ir([]), "import os\n"]
    together = checker.score_batch(programs, experiments=8, seed=5)
    assert [s["valid"] for s in together] == [True, True, True, True, False]
    for s in together[:4]:
        assert all(isinstance(s[t], (int, float)) and math.isfinite(s[t]) for t in TERMS), s
    labeled, plain, at_targets, empty = together[:4]
    assert math.isclose(labeled["exec_error_bits"], plain["exec_error_bits"], rel_tol=1e-6), "a label changes no execution"
    assert plain["alignment_error_bits"] > labeled["alignment_error_bits"] > 0, "an unplaced variable pays its whole signal"
    assert not math.isclose(at_targets["exec_error_bits"], labeled["exec_error_bits"], rel_tol=1e-6), "positions change what the graph computes"
    assert empty["exec_error_bits"] > labeled["exec_error_bits"]
    alone = checker.score(GRAPH, experiments=8, seed=5)
    assert math.isclose(alone["total_bits"], labeled["total_bits"], rel_tol=1e-6)


def test_ir_carries_the_explanation_length():
    import types

    fake = types.SimpleNamespace(model="vpd4l", behavior_record=None)
    ir = score.Checker.ir(fake, {"source": GRAPH, "explanation": "carry holds whether a quotation is open."})
    assert ir["valid"] and ir["explanation_tokens"] > 3 and ir["explanation_token_types"] > 150_000
    assert score.Checker.ir(fake, GRAPH)["explanation_tokens"] == 0
    assert score.Checker.ir(fake, ir) is ir
