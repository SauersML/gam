"""score.py against the published checker binary on vpd4l (skipped without the binary, the export or the behavior):

    ~/mpd-data/venv/bin/python -m pytest bench/oracle/graph/test_score.py

Twelve closing-quote prompts with their changed prompts (each changes the variable inside): an explanation placing
inside, the same subcomponents with inside unplaced, nothing named, and an invalid explanation, scored together and
one at a time.
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

SOURCE = Path.home() / "mpd-data/graph_oracle/behaviors_vary/vpd4l/quote_close.said.json"
TERMS = ["total_bits", "exec_error_bits", "necessity_error_bits", "alignment_error_bits", "complexity_bits", "structure_bits", "N"]
PLACED = '''groups = {
    "quote": {"subcomponents": ["<p:0.fc.225>", "<p:0.fc.2863>", "<p:0.down.663>", "<p:0.down.1224>"], "reads": ["input"], "label": "inside"},
    "answer": {"subcomponents": ["<p:2.v.80>", "<p:2.o.63>", "<p:3.fc.1013>", "<p:3.down.885>"], "reads": ["input", "quote"], "writes": "output"},
}
'''
UNPLACED = '''groups = {
    "answer": {"subcomponents": ["<p:0.fc.225>", "<p:0.fc.2863>", "<p:0.down.663>", "<p:0.down.1224>", "<p:2.v.80>", "<p:2.o.63>", "<p:3.fc.1013>", "<p:3.down.885>"], "reads": ["input"], "writes": "output"},
}
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
    programs = [PLACED, UNPLACED, explain.ir([]), "import os\n"]
    together = checker.score_batch(programs, experiments=8, seed=5)
    assert [s["valid"] for s in together] == [True, True, True, False]
    for s in together[:3]:
        assert all(isinstance(s[t], (int, float)) and math.isfinite(s[t]) for t in TERMS), s
    placed, unplaced, empty = together[:3]
    # the same subcomponents in one group or two: one execution; an unplaced variable pays its whole signal
    assert math.isclose(placed["exec_error_bits"], unplaced["exec_error_bits"], rel_tol=1e-6)  # float32 device: summation order
    assert unplaced["alignment_error_bits"] > placed["alignment_error_bits"] > 0
    assert empty["exec_error_bits"] > placed["exec_error_bits"]
    alone = checker.score(PLACED, experiments=8, seed=5)
    assert math.isclose(alone["total_bits"], placed["total_bits"], rel_tol=1e-6)


def test_ir_carries_the_explanation_length():
    import types

    fake = types.SimpleNamespace(model="vpd4l", behavior_record=None)
    ir = score.Checker.ir(fake, {"source": PLACED, "explanation": "Group quote carries whether a quotation is open."})
    assert ir["valid"] and ir["explanation_tokens"] > 3 and ir["explanation_token_types"] > 150_000
    assert score.Checker.ir(fake, PLACED)["explanation_tokens"] == 0
    assert score.Checker.ir(fake, ir) is ir
