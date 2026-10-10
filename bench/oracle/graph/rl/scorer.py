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


WORKERS = 1  # processes scoring at once (train.py --score-workers): the verifier's forward passes leave a GPU mostly idle one process at a time
_WORKER = []


def _start(device):
    import score

    _WORKER.append(score.Scorer(device))


def _job(job):
    behavior, sources, seed, necessity = job
    return _WORKER[0].score(behavior, sources, seed, necessity=necessity)


def native(items: list[dict]) -> list[dict]:
    """score.Scorer.score over items, one call per (task, seed, necessity) group of answers; with WORKERS > 1 the
    groups, cut into as many chunks of answers as fill the workers, go to that many processes (spawned per call and
    closed after, so a sampler sharing the GPU gets its memory back)."""
    import math

    import score

    groups = {}
    for k, it in enumerate(items):
        groups.setdefault((it["behavior"]["path"], it.get("seed", 0), bool((it.get("options") or {}).get("necessity"))), []).append(k)
    size = max(1, math.ceil(len(items) / WORKERS)) if WORKERS > 1 else len(items)
    chunks = [(seed, nec, ks[i:i + size]) for (_, seed, nec), ks in groups.items() for i in range(0, len(ks), size)]
    jobs = [(items[ks[0]]["behavior"], [items[k]["source"] for k in ks], seed, nec) for seed, nec, ks in chunks]
    results = None
    if WORKERS > 1 and len(jobs) > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        from concurrent.futures.process import BrokenProcessPool

        try:  # a worker that dies (out of memory) breaks the pool instead of hanging it; the calls then run here
            with ProcessPoolExecutor(min(WORKERS, len(jobs)), mp_context=multiprocessing.get_context("spawn"), initializer=_start, initargs=(DEVICE,)) as pool:
                results = list(pool.map(_job, jobs))
        except BrokenProcessPool as e:
            print(f"scorer: a worker died ({e}); scoring in this process", flush=True)
    if results is None:
        if not _SCORER:
            _SCORER.append(score.Scorer(DEVICE))
        results = [_SCORER[0].score(b, srcs, seed, necessity=nec) for b, srcs, seed, nec in jobs]
    out = [None] * len(items)
    for (_, _, ks), rs in zip(chunks, results):
        for k, r in zip(ks, rs):
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
