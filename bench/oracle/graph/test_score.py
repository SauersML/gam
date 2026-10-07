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
TERMS = ["total_bits", "exec_error_bits", "code_bits", "opaque_bits", "opaque_numbers", "python_tokens", "N", "experiments"]

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
    example = (HERE / "examples/vpd4l_induction_native.py").read_text()
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
