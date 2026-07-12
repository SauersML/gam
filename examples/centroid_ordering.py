"""Centroid circular-ordering diagnostic: are candidate centers ordered on a ring?

Companion to ``gamfit.adjudicate_atom_shape``. The production race now owns
discrete cyclic structure through its ``ring_clusters`` class, and its
``circle_wins`` flag compares the combined stacking mass of the smooth-circle
and ring-of-clusters candidates against the combined non-circular mass. This
module is an independent ordering diagnostic: it can corroborate the selected
ring-cluster order, or test whether the centroids of a winning free
``mixture`` class nevertheless exhibit circular order against a matched
Gaussian null.

Procedure (validated in the two-tier census of real Qwen3-8B SAE features;
the same construction historically exposed calendar circles before the
ring-of-clusters density landed, and correctly failed rings masked past ~1x
their radius):

  1. Seeded k-means on the 2-D coordinates, with k = the adjudicator's
     ``ring_clusters_reporting_k`` when diagnosing its circular class, or
     ``mixture_reporting_k`` when diagnosing the free-mixture class (k >= 3
     is required for a meaningful ring test). The corresponding
     ``*_fold_selected_k`` values describe honest outer-fold prediction; the
     reporting order is the all-data fit used for this post-race diagnostic.
  2. Ring statistic on the centroids: coefficient of variation (CV) of
     centroid radii about the centroid mean (low CV = centroids sit on a
     circle), plus angular coverage (max angular gap between sorted
     centroid angles; a big gap = an arc or a clump, not a ring).
  3. Monte-Carlo null: centroid sets drawn from a 2-D Gaussian matched to
     the observed centroid mean/covariance; p = P(null CV <= observed CV).

Caveat carried over from the census: sparse non-negative codes pushed
through per-group 2-D PCA produce ring-like centroid arrangements on
STRUCTURELESS controls at a double-digit rate per run. A positive here is
"consistent with a ring", not proof — always run the byte-identical
pipeline on the full-pipeline per-dimension-shuffle and covariance-exact
Hadamard controls before interpreting rates.

Pure NumPy; analysis-side code in the examples/ (non-production) sense of
SPEC.md. Import it next to this file the same way
``harvest_residual_activations.py`` imports ``residual_shard_io``.
"""
from __future__ import annotations

import numpy as np

__all__ = ["kmeans_centroids", "ring_stats", "ring_mc_pvalue",
           "centroid_circular_ordering"]


def kmeans_centroids(coords: np.ndarray, k: int, seed: int = 0,
                     iters: int = 100) -> np.ndarray:
    """Seeded Lloyd k-means on ``(n, 2)`` coords; returns ``(k, 2)`` centers."""
    X = np.asarray(coords, dtype=np.float64)
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(X.shape[0], size=k, replace=False)].copy()
    labels = np.zeros(X.shape[0], dtype=int)
    for _ in range(iters):
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new = d2.argmin(axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            m = labels == c
            if m.any():
                centers[c] = X[m].mean(axis=0)
    return centers


def ring_stats(centers: np.ndarray) -> tuple[float, float]:
    """Return ``(radius_cv, max_gap_deg)`` for a centroid set.

    ``radius_cv`` is the coefficient of variation of centroid radii about
    the centroid mean (0 = perfect circle). ``max_gap_deg`` is the largest
    angular gap between sorted centroid angles (360 = degenerate).
    """
    c = np.asarray(centers, dtype=np.float64)
    c = c - c.mean(axis=0, keepdims=True)
    r = np.linalg.norm(c, axis=1)
    if r.mean() < 1e-12:
        return float("inf"), 360.0
    cv = float(r.std() / r.mean())
    ang = np.sort(np.arctan2(c[:, 1], c[:, 0]))
    gaps = np.diff(np.concatenate([ang, [ang[0] + 2 * np.pi]]))
    return cv, float(np.degrees(gaps.max()))


def ring_mc_pvalue(centers: np.ndarray, observed_cv: float, n_null: int = 2000,
                   seed: int = 0) -> float:
    """P(null radius-CV <= observed) with null centroid sets drawn from a
    2-D Gaussian matched to the observed centroid mean/covariance."""
    rng = np.random.default_rng(seed)
    c = np.asarray(centers, dtype=np.float64)
    mu = c.mean(axis=0)
    cov = np.cov((c - mu).T) + 1e-12 * np.eye(2)
    L = np.linalg.cholesky(cov)
    k = c.shape[0]
    hits = 0
    for _ in range(n_null):
        z = rng.standard_normal((k, 2)) @ L.T + mu
        cv, _ = ring_stats(z)
        if cv <= observed_cv:
            hits += 1
    return (hits + 1) / (n_null + 1)


def centroid_circular_ordering(coords: np.ndarray, k: int, *, seed: int = 0,
                               n_null: int = 2000, p_thresh: float = 0.05,
                               gap_thresh_deg: float = 150.0) -> dict:
    """Full second-tier test on a ``(n, 2)`` coordinate cloud.

    ``k`` should be the all-data reporting order of the class being diagnosed:
    ``ring_clusters_reporting_k`` for the constrained circular class or
    ``mixture_reporting_k`` for the free-mixture class. Returns a dict with
    ``radius_cv``, ``max_gap_deg``, ``mc_p``, the centers, and the combined
    verdict ``ordered_on_circle`` (p below threshold AND angular coverage
    without a large gap).
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be (n, 2); got {coords.shape}")
    if k < 3:
        raise ValueError(f"need k >= 3 centroids to test a ring; got k={k}")
    if coords.shape[0] < k:
        raise ValueError(f"need at least k={k} rows; got {coords.shape[0]}")
    centers = kmeans_centroids(coords, k, seed=seed)
    cv, gap = ring_stats(centers)
    p = ring_mc_pvalue(centers, cv, n_null=n_null, seed=seed + 202)
    return {
        "k": int(k),
        "centers": centers,
        "radius_cv": float(cv),
        "max_gap_deg": float(gap),
        "mc_p": float(p),
        "ordered_on_circle": bool(p < p_thresh and gap < gap_thresh_deg),
        "params": {"seed": seed, "n_null": n_null, "p_thresh": p_thresh,
                   "gap_thresh_deg": gap_thresh_deg},
    }
