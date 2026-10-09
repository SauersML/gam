"""Scorers for the oracle's training (#2951): a batch of items (an answer's source and its task) -> score dicts
({"valid", "error", "kl_bits", "size", ...}; score.order ranks them against the task's eps).

  native  score.Scorer: traced and run on the model (one Scorer per process, on its device); a task's items of one
          seed are scored in one batch.
  mock    no model: KL 0 for a valid answer and its size from the trace. Plumbing only.
"""

from __future__ import annotations

import sys
from pathlib import Path

GRAPH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GRAPH))

DEVICE = None  # the native scorer's device (its default when None)
_SCORER = []


def native(items: list[dict]) -> list[dict]:
    import score

    if not _SCORER:
        _SCORER.append(score.Scorer(DEVICE))
    groups = {}
    for k, it in enumerate(items):
        groups.setdefault((it["behavior"]["path"], it.get("seed", 0)), []).append(k)
    out = [None] * len(items)
    for (_, seed), ks in groups.items():
        for k, r in zip(ks, _SCORER[0].score(items[ks[0]]["behavior"], [items[k]["source"] for k in ks], seed)):
            out[k] = r
    return out


def mock(items: list[dict]) -> list[dict]:
    import mech

    out = []
    for it in items:
        ir = mech.trace(it["source"], "vpd4l", behavior=it["behavior"])
        g = ir["graph"]
        out.append({"valid": ir["valid"], "error": ir.get("error"), "kl_bits": 0.0, "nodes": len(g["nodes"]),
                    "edges": len(g["parents"]) + len(g["out"]), "size": len(g["nodes"]) + len(g["parents"]) + len(g["out"]), "scorer": "mock"})
    return out


SCORERS = {"native": native, "mock": mock}
