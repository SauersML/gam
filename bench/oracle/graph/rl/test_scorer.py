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

    def request(self, message):
        self.path = message["path"]

    def score_batch(self, sources, experiments=32, seed=0, uniform_seeds=None):
        calls.append((self.path, seed, len(sources)))
        return [{"total_bits": len(x) + seed, "valid": True, "behavior": self.path} for x in sources]


def main():
    sys.modules["score"] = types.SimpleNamespace(Checker=Checker)
    sys.modules["run"] = types.SimpleNamespace(load_behavior=lambda c, path: c.request({"op": "behavior", "path": path, "manifest": None}))
    scorer.WORKERS = 3
    items = [{"source": "x" * i, "behavior": {"model": "vpd4l", "path": f"b{i % 4}"}, "seed": 1 + i % 2} for i in range(16)]
    out = scorer.checker(items)
    assert [r["total_bits"] for r in out] == [i + 1 + i % 2 for i in range(16)]
    assert [r["behavior"] for r in out] == [f"b{i % 4}" for i in range(16)]
    assert sorted(calls) == sorted({(f"b{i % 4}", 1 + i % 2, 4) for i in range(16)}), calls
    check_repair()
    print("ok: checker scorer order, behaviors and one batch per (behavior, seed); repair keeps a revision only when it lowers S")


def check_repair():
    """train.repair replaces a behavior's best program by a revision only when the revision is valid where
    the best was not, or lowers S; the revision prompt carries the program and the checker's error."""
    import train

    shown = []
    train.render = lambda b: f"input {b['id']}"
    tok = types.SimpleNamespace(decode=lambda c, skip_special_tokens=True: f"```python\nX = {c[0]}\n```")
    pol = types.SimpleNamespace(tok=tok, prompt_ids=lambda text: shown.append(text) or [0])
    sampler = lambda prompts, n, adapter, version: [[[v] for v in (70, 30)] for _ in prompts]  # noqa: E731
    score = lambda items: [{"valid": True, "total_bits": float(it["source"].split("=")[1])} for it in items]  # noqa: E731
    best = [{"completion": [0], "text": "```python\nbad(\n```", "score": {"valid": False, "total_bits": 1e4, "error": "line 1: syntax error"}},
            {"completion": [0], "text": "```python\nX = 10\n```", "score": {"valid": True, "total_bits": 10.0}}]
    args = types.SimpleNamespace(repair=1, samples=2, uniform_seeds=0, experiments=16)
    replaced = train.repair([{"id": "a"}, {"id": "b"}], best, pol, sampler, score, args, Path("."), 0)
    assert replaced == {0} and best[0]["score"]["total_bits"] == 30.0 and best[0]["completion"] == [30] and best[1]["score"]["total_bits"] == 10.0, (replaced, best)
    assert "line 1: syntax error" in shown[0] and "bad(" in shown[0] and shown[0].startswith("input a")


if __name__ == "__main__":
    main()
