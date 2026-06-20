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
  per-latent ``argmax`` matching -- NOT one-to-one assignment, and (per the
  paper) NOT a recovery score: a dictionary of distinct-but-poor directions can
  still score 1.0.

* **MCC** -- a *quality* metric: the optimal one-to-one |cosine| matching
  (Hungarian), reporting the mean |cosine| of the matched pairs.

* **direction recovery (precision / recall / F1 / Jaccard)** -- a *coverage*
  metric that is aware of match quality. See :func:`recovery_scores`.

Why the split matters (#1413): a raw one-to-one *assignment count* is always
``min(n_learned, n_truth)`` because the Hungarian algorithm saturates the
smaller side regardless of similarity. So ``n_matched / max(...)`` collapses to
the pure width ratio ``min/max`` and never inspects whether the matched pairs
are any good -- equal-width perfect, all-duplicate, and orthogonal dictionaries
all score 1.0. Algebraically ``min/max`` equals the Jaccard recovery one *would*
get if every forced assignment were perfect (``MCC = 1``), i.e. a recovery
*ceiling*, not a measurement. Uniqueness (collisions) and the quality-weighted
recovery scores below each look at the actual matched cosines.

Matching is exact: SciPy's ``linear_sum_assignment`` when available, otherwise
the dependency-free Hungarian in :func:`_hungarian_max_numpy`. The greedy
nearest-pair heuristic the harnesses used before is NOT exact (e.g. on
``[[0.9, 0.8], [0.8, 0.0]]`` it returns mass 0.9 vs the optimal 1.6), so a
quality metric must not depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _hungarian_max_numpy(sim: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact maximum-weight one-to-one matching, dependency-free.

    Jonker-Volgenant / Kuhn-Munkres on costs ``-sim`` (n = smaller side).
    Returns ``(rows, cols)`` saturating the smaller side. Used only when SciPy
    is unavailable; validated against brute force in the test suite.
    """
    sim = np.asarray(sim, dtype=float)
    transposed = sim.shape[0] > sim.shape[1]
    cost = (-sim.T if transposed else -sim).astype(float)
    n, m = cost.shape  # n <= m
    inf = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)  # p[j] = 1-indexed row matched to column j (0 = unset)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = -1
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0 != 0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    rows: list[int] = []
    cols: list[int] = []
    for j in range(1, m + 1):
        if p[j] != 0:
            rows.append(p[j] - 1)
            cols.append(j - 1)
    rows_arr = np.asarray(rows, dtype=int)
    cols_arr = np.asarray(cols, dtype=int)
    if transposed:
        rows_arr, cols_arr = cols_arr, rows_arr
    order = np.argsort(rows_arr)
    return rows_arr[order], cols_arr[order]


def match_directions(
    learned: np.ndarray, truth: np.ndarray
) -> tuple[np.ndarray, np.ndarray, str]:
    """Optimal one-to-one ``|cosine|`` matching (exact).

    ``learned`` (``L x D``) and ``truth`` (``T x D``) rows are assumed already
    unit-normalized (the callers normalize before matching). Returns
    ``(rows, cols, method)`` with ``len(rows) == min(L, T)``; ``method`` is
    ``"hungarian-scipy"``, ``"hungarian-numpy"``, or ``"none"`` (empty input).
    """
    learned = np.asarray(learned, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if learned.size == 0 or truth.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int), "none"
    sim = np.abs(learned @ truth.T)
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        rows, cols = linear_sum_assignment(-sim)
        return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int), "hungarian-scipy"
    except ImportError:
        rows, cols = _hungarian_max_numpy(sim)
        return rows, cols, "hungarian-numpy"


def feature_uniqueness(learned: np.ndarray, truth: np.ndarray) -> float:
    """SynthSAEBench feature uniqueness: fraction of distinct ``argmax`` targets.

    ``i*(j) = argmax_i |w_j . d_i|`` for each learned latent ``j``; the score is
    ``#distinct(i*) / n_learned``. 1.0 means every latent's best match is a
    different truth feature; duplicate / collapsed directions that share a best
    match pull it toward ``1 / n_learned``. This is a collision/diversity
    diagnostic, NOT recovery quality. Returns 0.0 for empty input.
    """
    learned = np.asarray(learned, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if learned.shape[0] == 0 or truth.shape[0] == 0:
        return 0.0
    sim = np.abs(learned @ truth.T)
    best = np.argmax(sim, axis=1)
    return float(np.unique(best).size / learned.shape[0])


def firing_latent_mask(activations: np.ndarray, threshold: float = 1e-8) -> np.ndarray:
    """Functional dead-latent mask (SynthSAEBench): a latent is *live* if it
    fires (``|activation| > threshold``) on at least one sample of the evaluation
    set.

    Unlike geometric deadness (zero decoder norm), a latent with a nonzero decoder
    direction that never activates is *functionally* dead -- wasted capacity that a
    decoder-norm check misses (#1435). Returns a 1-D boolean mask whose length is
    the latent count (``activations`` columns). Empty input returns an empty mask;
    a 1-D array is treated as a single latent.
    """
    a = np.abs(np.asarray(activations, dtype=float))
    if a.size == 0:
        return np.empty(0, dtype=bool)
    if a.ndim == 1:
        a = a[:, None]
    return np.any(a > threshold, axis=0)


def n_firing_latents(activations: np.ndarray, threshold: float = 1e-8) -> int:
    """Count of latents that fire on at least one evaluation sample.

    Functional (activation-based) live count, complementing the geometric
    (decoder-norm) live count. ``n_slots - n_firing_latents`` is the functional
    dead count (#1435).
    """
    return int(firing_latent_mask(activations, threshold).sum())


@dataclass(frozen=True)
class RecoveryScores:
    """Quality-aware one-to-one recovery plus the matched-pair MCC.

    ``precision``/``recall``/``f1``/``jaccard`` are soft (cosine-weighted)
    detection scores; ``mcc`` is the mean matched |cosine|;
    ``cardinality_ceiling`` is the honestly-named ``min/max`` width ratio (the
    recovery these widths would reach at perfect quality).
    """

    mcc: float
    precision: float
    recall: float
    f1: float
    jaccard: float
    cardinality_ceiling: float
    rows: np.ndarray
    cols: np.ndarray
    matching: str


def recovery_scores(
    learned: np.ndarray,
    truth: np.ndarray,
    n_learned_total: int | None = None,
) -> RecoveryScores:
    """Quality-aware recovery (threshold-free) + matched-pair MCC.

    Take the optimal matched ``|cosine|`` values ``q`` (exact Hungarian). Their
    *mass* ``A = sum(q)`` is a soft count of recovered features. With ``L`` the
    learned-slot count and ``T`` the truth count::

        precision = A / L            # junk / excess / dead learned dirs lower it
        recall    = A / T            # unrecovered truth features lower it
        f1        = 2A / (L + T)
        jaccard   = A / (L + T - A)
        mcc       = mean(q)          # SynthSAEBench matched-pair quality

    Each soft score equals the corresponding hard detection metric under the
    interpretation that a matched pair of quality ``q`` contributes ``q`` to
    true positives and ``1 - q`` to both false-positive and false-negative
    mass. F1 (and Jaccard) reach 1.0 only for a perfect bijection over the full
    truth dictionary; duplicate / random dictionaries score well below 1.0.

    ``n_learned_total`` overrides ``L`` so callers can include *dead* slots
    (zero-norm decoder directions never enter an optimal positive-mass matching,
    so ``A`` is unchanged, but counting them in ``L`` correctly penalizes wasted
    capacity). Defaults to ``learned.shape[0]``.
    """
    learned = np.asarray(learned, dtype=float)
    truth = np.asarray(truth, dtype=float)
    n_learned = int(learned.shape[0] if n_learned_total is None else n_learned_total)
    n_truth = int(truth.shape[0])
    ceiling = (
        min(n_learned, n_truth) / max(n_learned, n_truth)
        if max(n_learned, n_truth) > 0
        else 0.0
    )
    rows, cols, method = match_directions(learned, truth)
    if rows.size == 0:
        return RecoveryScores(0.0, 0.0, 0.0, 0.0, 0.0, ceiling, rows, cols, method)
    sim = np.abs(learned @ truth.T)
    q = sim[rows, cols]
    mass = float(q.sum())
    mcc = float(q.mean())
    precision = mass / max(n_learned, 1)
    recall = mass / max(n_truth, 1)
    f1 = 2.0 * mass / max(n_learned + n_truth, 1)
    denom = n_learned + n_truth - mass
    jaccard = mass / denom if denom > 1e-12 else 0.0
    return RecoveryScores(
        mcc, precision, recall, f1, jaccard, ceiling, rows, cols, method
    )
