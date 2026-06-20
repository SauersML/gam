"""Direction-recovery metrics shared by the synthetic SAE benchmarks.

NumPy-only (no ``gamfit``/torch import) so it can be unit-tested in isolation
by ``bench/test_synth_sae_metrics.py``. The definitions follow SynthSAEBench
(arXiv:2602.14687), which separates three distinct ground-truth measurements
that the benchmark harnesses had previously conflated:

* **feature uniqueness** -- a *collision* metric. For each learned latent ``j``
  its best ground-truth match is ``i*(j) = argmax_i |w_j . d_i|`` and
  uniqueness is the fraction of *distinct* best targets,
  ``#unique(i*) / n_learned``. It is 1.0 iff every latent claims a different
  truth feature; duplicate / collapsed directions that all point at the same
  truth feature drag it toward ``1 / n_learned``. This is independent
  per-latent ``argmax`` matching -- NOT one-to-one assignment.

* **MCC** -- a *quality* metric: the optimal one-to-one |cosine| matching
  (Hungarian), reporting the mean |cosine| of the matched pairs.

* **direction recovery (precision / recall / F1)** -- a *coverage* metric that
  is aware of match quality. See :func:`recovery_scores`.

Why the split matters (#1413): a raw one-to-one *assignment count* is always
``min(n_learned, n_truth)`` because both the Hungarian algorithm and the greedy
fallback fill every row/column they can regardless of similarity. So
``n_matched / max(...)`` collapses to a pure width ratio ``min/max`` and never
inspects whether the matched pairs are any good -- equal-width perfect,
all-duplicate, and random dictionaries all score 1.0. Uniqueness (collisions)
and recovery F1 (quality-weighted coverage) each look at the actual directions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def match_directions(
    learned: np.ndarray, truth: np.ndarray
) -> tuple[np.ndarray, np.ndarray, str]:
    """Optimal one-to-one ``|cosine|`` matching, Hungarian with greedy fallback.

    ``learned`` (``L x D``) and ``truth`` (``T x D``) rows are assumed already
    unit-normalized (the callers normalize before matching). Returns
    ``(rows, cols, method)`` with ``len(rows) == min(L, T)``; ``method`` is one
    of ``"hungarian"``, ``"greedy"``, or ``"none"`` (empty input).
    """
    learned = np.asarray(learned, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if learned.size == 0 or truth.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int), "none"
    sim = np.abs(learned @ truth.T)
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        rows, cols = linear_sum_assignment(-sim)
        return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int), "hungarian"
    except Exception:
        pairs: list[tuple[int, int]] = []
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        for flat in np.argsort(sim, axis=None)[::-1]:
            row, col = np.unravel_index(int(flat), sim.shape)
            if int(row) in used_rows or int(col) in used_cols:
                continue
            pairs.append((int(row), int(col)))
            used_rows.add(int(row))
            used_cols.add(int(col))
            if len(pairs) == min(sim.shape):
                break
        rows = np.array([p[0] for p in pairs], dtype=int)
        cols = np.array([p[1] for p in pairs], dtype=int)
        return rows, cols, "greedy"


def feature_uniqueness(learned: np.ndarray, truth: np.ndarray) -> float:
    """SynthSAEBench feature uniqueness: fraction of distinct ``argmax`` targets.

    ``i*(j) = argmax_i |w_j . d_i|`` for each learned latent ``j``; the score is
    ``#distinct(i*) / n_learned``. 1.0 means every latent's best match is a
    different truth feature; duplicate / collapsed directions that share a best
    match pull it toward ``1 / n_learned``. Returns 0.0 for empty input.
    """
    learned = np.asarray(learned, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if learned.shape[0] == 0 or truth.shape[0] == 0:
        return 0.0
    sim = np.abs(learned @ truth.T)
    best = np.argmax(sim, axis=1)
    return float(np.unique(best).size / learned.shape[0])


@dataclass(frozen=True)
class RecoveryScores:
    """Quality-aware one-to-one recovery plus the matched-pair MCC."""

    mcc: float
    precision: float
    recall: float
    f1: float
    rows: np.ndarray
    cols: np.ndarray
    matching: str


def recovery_scores(learned: np.ndarray, truth: np.ndarray) -> RecoveryScores:
    """Quality-aware recovery (threshold-free) + matched-pair MCC.

    Take the optimal matched ``|cosine|`` values ``q`` (Hungarian). Their
    *mass* ``sum(q)`` is a soft count of recovered features, which yields::

        precision = mass / n_learned   # junk / excess learned dirs lower it
        recall    = mass / n_truth     # unrecovered truth features lower it
        f1        = 2 * mass / (n_learned + n_truth)
        mcc       = mean(q)            # SynthSAEBench matched-pair quality

    Unlike the raw assignment count (always ``min(L, T)``), every term reflects
    *how well* the assigned pairs align: equal-width duplicate or random
    dictionaries score well below 1.0, perfect tight recovery scores 1.0,
    excess/junk learned dirs lower precision, and unrecovered truth features
    lower recall. ``rows``/``cols`` are returned so callers can reuse the same
    one-to-one matching (e.g. for probing metrics) without rematching.
    """
    learned = np.asarray(learned, dtype=float)
    truth = np.asarray(truth, dtype=float)
    n_learned = int(learned.shape[0])
    n_truth = int(truth.shape[0])
    rows, cols, method = match_directions(learned, truth)
    if rows.size == 0:
        return RecoveryScores(0.0, 0.0, 0.0, 0.0, rows, cols, method)
    sim = np.abs(learned @ truth.T)
    q = sim[rows, cols]
    mass = float(q.sum())
    mcc = float(q.mean())
    precision = mass / max(n_learned, 1)
    recall = mass / max(n_truth, 1)
    f1 = 2.0 * mass / max(n_learned + n_truth, 1)
    return RecoveryScores(mcc, precision, recall, f1, rows, cols, method)
