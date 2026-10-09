"""e2e/prune.py's refinement under a stand-in error: two redundant paths each carry the whole signal, so removing either
alone costs nothing and removing both costs everything.

  ~/mpd-data/venv/bin/python -m pytest bench/oracle/graph/test_prune.py
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "e2e"))

import prune  # noqa: E402

UNITS = [(0, "c_fc", i) for i in range(64)]
PATH_A = {(0, "c_fc", 3), (0, "c_fc", 17)}
PATH_B = {(0, "c_fc", 40), (0, "c_fc", 41), (0, "c_fc", 42)}
SIGNAL = 100.0


def error(named: set) -> float:
    """The signal unless a path is named whole; 1 bit per path left half named (a partial carrier)."""
    if PATH_A <= named or PATH_B <= named:
        return 0.0
    return SIGNAL - sum(1.0 for p in (PATH_A, PATH_B) if p & named)


def test_refine_keeps_one_whole_path(monkeypatch):
    monkeypatch.setattr(prune, "errors", lambda checker, sets, a: [error(set(prune.units(s))) for s in sets])
    a = argparse.Namespace(ks=[1, 2, 4, 8], tol=0.01)
    sets, rounds = prune.refine(None, [tuple(UNITS)], 0.0, SIGNAL, a, lambda m: None)
    assert set(sets["knee"]) in (PATH_A, PATH_B), "the smallest set within the tolerance is one path, whole"
    assert set(sets["knee"]) <= set(sets[4]) <= set(sets[8]) and len(sets[2]) == 2 and len(sets[1]) == 1
    assert rounds[0] == {"parts": 64, "chunks": 1, "error_bits": 0.0}
    assert all(r["error_bits"] == 0.0 for r in rounds if r["parts"] >= len(sets["knee"])), "no round before the knee lost the path"
