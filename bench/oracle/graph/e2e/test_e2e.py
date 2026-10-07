"""End-to-end checks of the graph oracle's score (#2951) on vpd4l induction: the reference programs trace,
the oracle prompt renders, and the checker orders them the way every valid score must:
  hand-written induction program < empty program < random irrelevant heads   (total bits)
  full program (every piece declared, every edge listed): execution error ~0, the largest opaque cost.

  GRAPH_CHECKER=<mpd_graph_2951 binary> ~/mpd-data/venv/bin/python -m pytest bench/oracle/graph/e2e/test_e2e.py
The checker tests are skipped when the binary or the behavior file is missing.
"""

import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
os.environ.setdefault("MPD_MEM_GIB", "1")

import mech  # noqa: E402
import programs  # noqa: E402
import prompt  # noqa: E402
import score  # noqa: E402

BEHAVIOR = Path.home() / "mpd-data/graph_oracle/behaviors/vpd4l/induction_random.words8.json"
needs_checker = pytest.mark.skipif(not (score.BINARY.exists() and BEHAVIOR.exists()),
                                   reason="no checker binary (GRAPH_CHECKER) or no behavior file")


def test_references_trace():
    for name, source in programs.references("vpd4l").items():
        ir = mech.trace_inline(source, "vpd4l")
        assert ir["valid"], (name, ir["error"])
        assert (ir["nodes"] == []) == (name == "empty"), name
    full = mech.trace_inline(programs.full("vpd4l"), "vpd4l")
    s = mech.shapes("vpd4l")
    assert len(full["nodes"]) == 2 * s["layers"]
    # every writer (embed, then each block in causal order) into every later block and the logits
    blocks = 2 * s["layers"]
    assert len(full["edges"]) == blocks * (blocks + 1) // 2 + blocks + 1


def test_random_heads_avoid_hand():
    used = programs.hand_heads("vpd4l")
    ir = mech.trace_inline(programs.random_heads("vpd4l", 3, 0), "vpd4l")
    drawn = {(n["pieces"][0]["layer"], n["pieces"][0]["index"]) for n in ir["nodes"]}
    assert len(drawn) == 3 and not drawn & used


@pytest.mark.skipif(not BEHAVIOR.exists(), reason="no behavior file")
def test_prompt_renders():
    text = prompt.render(json.loads(BEHAVIOR.read_text()))
    assert "Write the program." in text and "vpd4l" in text


@pytest.fixture(scope="module")
def scores():
    with score.Checker("vpd4l") as c:
        c.behavior(BEHAVIOR)
        return {name: c.score(source, experiments=16, seed=0, reader=False)
                for name, source in programs.references("vpd4l").items()}


@needs_checker
def test_all_valid(scores):
    for name, r in scores.items():
        assert r["valid"], (name, r["error"])


@needs_checker
def test_full_program_is_the_model(scores):
    full = scores["full"]
    assert full["exec_error_bits"] / full["N"] < 1e-6, full["per_family"]
    assert full["opaque_numbers"] > max(r["opaque_numbers"] for n, r in scores.items() if n != "full")
    assert full["total_bits"] > scores["empty"]["total_bits"]


@needs_checker
def test_empty_beats_random_heads(scores):
    assert scores["empty"]["total_bits"] < scores["random"]["total_bits"]


@needs_checker
def test_hand_beats_empty(scores):
    assert scores["hand"]["total_bits"] < scores["empty"]["total_bits"], {
        n: (r["total_bits"], r["exec_error_bits"], r["opaque_bits"]) for n, r in scores.items()}
