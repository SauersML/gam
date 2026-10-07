"""The graph oracle's whole path on a tiny random model (#2951), for CI: mech traces the reference and
search programs, the Rust checker (GRAPH_CHECKER, built from crates/gam-mpd/examples/mpd_graph_2951.rs)
scores them on e2e/tiny.py's export and behavior, and the terms obey what every valid score must:
  - the full program (every piece, every edge) costs at most its numbers' exact price (precision pricing
    may quantize, exact weights being one option);
  - with counterfactual stand-ins the empty program's execution error is the behavior's own signal
    KL(M(x) || M(x')) at the targets: positive, and it costs no opaque numbers;
  - a program's opaque count grows with the pieces it declares; code bits are its Python tokens times
    log2 of the token types;
  - one score_batch request gives the same terms as separate requests.

  GRAPH_CHECKER=target/release/examples/mpd_graph_2951 python -m pytest bench/oracle/graph/e2e/test_tiny.py
Skipped without a checker binary.
"""

import math
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
os.environ.setdefault("MPD_MEM_GIB", "1")
os.environ.setdefault("MEM_LEASE_GIB", "1")  # no mem-lease wrapper around the checker (absent in CI)

import mech  # noqa: E402
import score  # noqa: E402
import search  # noqa: E402
import tiny  # noqa: E402

LAYERS = 2
pytestmark = pytest.mark.skipif(not score.BINARY.exists(), reason="no checker binary (GRAPH_CHECKER)")


@pytest.fixture(scope="module")
def checker(tmp_path_factory):
    tiny.register(LAYERS)
    root = tmp_path_factory.mktemp("tiny")
    export = tiny.export(root / "export", LAYERS)
    path = tiny.behavior(root / "tiny.copy.json")
    c = score.Checker("tiny", export)
    c.request({"op": "behavior", "path": str(path), "manifest": None})
    yield c
    c.close()


def ir(units, stand_in=None):
    r = mech.trace_inline(search.source(units), "tiny")
    assert r["valid"], r["error"]
    if stand_in:
        r["standin"] = stand_in
    return r


def scored(c, irs, seed=0):
    return c.request({"op": "score_batch", "programs": irs, "experiments": 8, "seed": seed, "routing": "edges",
                      "N": None, "reader_top": 0})["scores"]


def test_full_program_is_the_model(checker):
    """Every piece declared: with precision pricing each block may run quantized, but exact weights are one
    of its options, so the total never exceeds the exact price of its numbers, and its execution error is
    all quantization error (none from stand-ins: there are none)."""
    units = search.all_units("tiny")
    (s,) = scored(checker, [ir(units)])
    assert s["valid"], s["error"]
    exact = 0.5 * math.log2(s["N"]) * s["opaque_numbers"]
    assert s["opaque_bits"] <= exact + 1e-6 and s["total_bits"] <= exact + s["code_bits"] + 1e-6, (s["total_bits"], exact)


def test_empty_program_carries_the_signal_for_free(checker):
    (s,) = scored(checker, [ir([], "counterfactual")])
    assert s["valid"] and s["exec_error_bits"] > 0
    assert s["opaque_numbers"] == 0, s["opaque_numbers"]
    assert s["per_family"]["clean"]["mean_kl_bits"] > 0


def test_terms_follow_the_declared_pieces(checker):
    units = search.all_units("tiny")
    head, mlp = units[0], units[-1]
    small, large, everything = scored(checker, [ir([head]), ir([head, mlp]), ir(units)])
    assert 0 < small["opaque_numbers"] < large["opaque_numbers"] < everything["opaque_numbers"]
    for s, us in ((small, [head]), (large, [head, mlp])):
        tokens, types = mech.code_length(search.source(us))
        assert math.isclose(s["code_bits"], tokens * math.log2(types), rel_tol=1e-9)
        assert math.isclose(s["opaque_bits"], 0.5 * math.log2(s["N"]) * s["opaque_numbers"], rel_tol=1e-9)


def test_batch_equals_single_requests(checker):
    units = search.all_units("tiny")
    irs = [ir([]), ir(units[:1]), ir(units)]
    batch = scored(checker, irs, seed=3)
    for program, b in zip(irs, batch):
        single = checker.request({"op": "score", "program": program, "experiments": 8, "seed": 3, "routing": "edges",
                                  "N": None, "reader_top": 0})
        assert math.isclose(single["total_bits"], b["total_bits"], rel_tol=1e-9, abs_tol=1e-9)
