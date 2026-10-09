"""Scorers for the oracle's training (#2951): a batch of items (an answer's source and its task) -> score dicts
({"valid", "error", "curve", "lo", "kl_bits", "bits", ...}; score.keys ranks the answers to one question).

  native  score.Scorer: traced and run on the model (one Scorer per process, on its device); a task's items of one
          seed are scored in one batch.
  mock    no model: a valid answer's curve falls from 1 bit to 0 at its whole graph's size from the trace. Plumbing only.
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
        groups.setdefault((it["behavior"]["path"], it.get("seed", 0), bool((it.get("options") or {}).get("necessity"))), []).append(k)
    out = [None] * len(items)
    for (_, seed, _), ks in groups.items():
        for k, r in zip(ks, _SCORER[0].score(items[ks[0]]["behavior"], [items[k]["source"] for k in ks], seed,
                                              necessity=bool((items[ks[0]].get("options") or {}).get("necessity")))):
            out[k] = r
    return out


def adversarial(tasks: list[dict], sources: list[str], seed: int = 0) -> list:
    """score.Scorer.adversarial on the native scorer's Scorer (evaluation only)."""
    import score

    if not _SCORER:
        _SCORER.append(score.Scorer(DEVICE))
    return _SCORER[0].adversarial(tasks, sources, seed)


def mock(items: list[dict]) -> list[dict]:
    import mech

    out = []
    for it in items:
        ir = mech.trace(it["source"], "vpd4l", behavior=it["behavior"])
        g = ir["graph"]
        size = len(g["nodes"]) + len(g["parents"]) + len(g["out"])
        out.append({"valid": ir["valid"], "error": ir.get("error"), "curve": [[0.0, 1.0]] + ([[float(size), 0.0]] if size else []), "lo": 1.0, "hi": 1e6,
                    "kl_bits": 0.0 if size else 1.0, "bits": float(size), "steps": g.get("steps", 0), "nodes": len(g["nodes"]),
                    "edges": len(g["parents"]) + len(g["out"]), "scorer": "mock"})
    return out


SCORERS = {"native": native, "mock": mock}
