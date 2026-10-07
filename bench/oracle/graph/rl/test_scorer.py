"""Checks of the RL scorer's checker path with a stand-in score.Checker (no binary): python test_scorer.py

Programs keep their order across behaviors, workers and seeds; each behavior is loaded on the server
that scores it, and each (behavior, seed) goes in one score_batch request."""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scorer  # noqa: E402

calls = []


class Checker:
    def __init__(self, model):
        self.model, self.path = model, None

    def behavior(self, path):
        self.path = path

    def score_batch(self, sources, seed=0):
        calls.append((self.path, seed, len(sources)))
        return [{"total_bits": len(x) + seed, "valid": True, "behavior": self.path} for x in sources]


def main():
    sys.modules["score"] = types.SimpleNamespace(Checker=Checker)
    scorer.WORKERS = 3
    items = [{"source": "x" * i, "behavior": {"model": "vpd4l", "path": f"b{i % 4}"}, "seed": 1 + i % 2} for i in range(16)]
    out = scorer.checker(items)
    assert [r["total_bits"] for r in out] == [i + 1 + i % 2 for i in range(16)]
    assert [r["behavior"] for r in out] == [f"b{i % 4}" for i in range(16)]
    assert sorted(calls) == sorted({(f"b{i % 4}", 1 + i % 2, 4) for i in range(16)}), calls
    print("ok: checker scorer order, behaviors and one batch per (behavior, seed)")


if __name__ == "__main__":
    main()
